from copy import copy
from collections import namedtuple

from .angles import relative_azimuth
from ..core.util import project_coords
from ..errors import logger

import numpy as np
import numexpr as ne
import xarray as xr
import dask
import dask.array as da


def dtor(x):
    """Converts degrees to radians"""
    return x * np.pi / 180.0


def get_phase_angle_dask(cos_vza, cos_sza, sin_vza, sin_sza, cos_raa):

    """
    Calculates the Phase angle component
    """

    cos_phase_angle = cos_vza * cos_sza + sin_vza * sin_sza * cos_raa

    cos_phase_angle = da.clip(cos_phase_angle, -1, 1)

    phase_angle = da.arccos(cos_phase_angle)
    sin_phase_angle = da.sin(phase_angle)

    return cos_phase_angle, phase_angle, sin_phase_angle


def __ross_kernel_part_delayed(angle_info, global_args):

    """
    Calculates the main part of Ross kernel
    """

    RossKernelOutputs = namedtuple('RossKernelOutputs', 'cos1 cos2 sin1 sin2 cos3 rosselement cosphaang phaang sinphaang ross')

    __cos1 = da.cos(da.from_delayed(angle_info.vza, shape=(global_args.size,), dtype=global_args.dtype).rechunk(global_args.flat_chunks))
    __cos2 = da.cos(da.from_delayed(angle_info.sza, shape=(global_args.size,), dtype=global_args.dtype).rechunk(global_args.flat_chunks))

    __sin1 = da.sin(da.from_delayed(angle_info.vza, shape=(global_args.size,), dtype=global_args.dtype).rechunk(global_args.flat_chunks))
    __sin2 = da.sin(da.from_delayed(angle_info.sza, shape=(global_args.size,), dtype=global_args.dtype).rechunk(global_args.flat_chunks))
    __cos3 = da.cos(da.from_delayed(angle_info.raa, shape=(global_args.size,), dtype=global_args.dtype).rechunk(global_args.flat_chunks))

    __cosphaang, __phaang, __sinphaang = get_phase_angle_dask(__cos1, __cos2, __sin1, __sin2, __cos3)

    rosselement = (global_args.m_pi2 - __phaang) * __cosphaang + __sinphaang

    return RossKernelOutputs(cos1=__cos1,
                             cos2=__cos2,
                             sin1=__sin1,
                             sin2=__sin2,
                             cos3=__cos3,
                             rosselement=rosselement,
                             cosphaang=__cosphaang,
                             phaang=__phaang,
                             sinphaang=__sinphaang,
                             ross=None)


def __ross_kernel_part_delayed_2d(angle_info, global_args):

    """
    Calculates the main part of Ross kernel
    """

    RossKernelOutputs = namedtuple('RossKernelOutputs', 'cos_vza cos_sza sin_vza sin_sza cos_raa ross_element cos_phase_angle phase_angle sin_phase_angle ross')

    cos_vza = da.cos(angle_info.vza_rad)
    cos_sza = da.cos(angle_info.sza_rad)

    sin_vza = da.sin(angle_info.vza_rad)
    sin_sza = da.sin(angle_info.sza_rad)
    cos_raa = da.cos(angle_info.raa_rad)

    cos_phase_angle, phase_angle, sin_phase_angle = get_phase_angle_dask(cos_vza, cos_sza, sin_vza, sin_sza, cos_raa)

    ross_element = (global_args.m_pi2 - phase_angle) * cos_phase_angle + sin_phase_angle

    return RossKernelOutputs(cos_vza=cos_vza,
                             cos_sza=cos_sza,
                             sin_vza=sin_vza,
                             sin_sza=sin_sza,
                             cos_raa=cos_raa,
                             ross_element=ross_element,
                             cos_phase_angle=cos_phase_angle,
                             phase_angle=phase_angle,
                             sin_phase_angle=sin_phase_angle,
                             ross=None)


def ross_thin_delayed(angle_info, global_args):

    """
    Public method - call to calculate Ross Thin kernel
    """

    RossThinOutputs = namedtuple('RossThinOutputs', 'ross phase_angle')

    ross_kernel_outputs = __ross_kernel_part_delayed_2d(angle_info, global_args)

    ross_ = ross_kernel_outputs.ross_element / (ross_kernel_outputs.cos_vza * ross_kernel_outputs.cos_sza)

    return RossThinOutputs(ross=ross_, phase_angle=ross_kernel_outputs.phase_angle)


def ross_thick_delayed(angle_info, global_args):

    """
    Public method - call to calculate RossThick kernel
    """

    RossThickOutputs = namedtuple('RossThickOutputs', 'ross phase_angle')

    # RossKernelOutputs = namedtuple('RossKernelOutputs', 'cos1 cos2 sin1 sin2 cos3 rosselement cosphaang phaang sinphaang ross')
    ross_kernel_outputs = __ross_kernel_part_delayed_2d(angle_info, global_args)

    ross_ = ross_kernel_outputs.ross_element / (ross_kernel_outputs.cos_vza + ross_kernel_outputs.cos_sza)

    return RossThickOutputs(ross=ross_, phase_angle=ross_kernel_outputs.phase_angle)


def ross_kernel_dask(angle_info, global_args):

    """
    Public method - call to calculate Ross Kernel
    """

    # RossKernelOutputs = namedtuple('RossKernelOutputs', 'cos1 cos2 sin1 sin2 cos3 rosselement cosphaang phaang sinphaang ross')
    if global_args.ross_type.lower() == 'thin':
        ross_kernel_outputs = ross_thin_delayed(angle_info, global_args)
    else:
        ross_kernel_outputs = ross_thick_delayed(angle_info, global_args)

    if global_args.ross_hs:
        ross = ross_kernel_outputs.ross * (1.0 + 1.0 / (1.0 + ross_kernel_outputs.phase_angle / 0.25))
    else:
        ross = ross_kernel_outputs.ross

    return ross


def get_pangles_dask(tan1, global_args):

    """
    Applies B/R transformation for ellipse shape
    """

    t = global_args.br * tan1

    t = da.where(t < 0, 0, t)

    # w = np.where(t < 0.)[0]
    # t[w] = 0.0

    angp = da.arctan(t)
    s = da.sin(angp)
    c = da.cos(angp)

    # have to make sure c is not 0
    c = da.where(c == 0, global_args.nearly_zero, c)

    # w = np.where(c == 0)[0]
    # c[w] = global_args.nearly_zero

    return c, s, t


def get_distance_dask(__tan1, __tan2, __cos3):

    """
    Gets distance component of Li kernels
    """

    temp = __tan1 * __tan1 + __tan2 * __tan2 - 2.0 * __tan1 * __tan2 * __cos3

    temp = da.where(temp < 0, 0, temp)

    # w = np.where(temp < 0)[0]
    # temp[w] = 0.0

    # TODO
    # self.__temp = temp  # used by other functions ??

    return da.sqrt(temp)


def get_overlap_dask(__cos1, __cos2, __tan1, __tan2, __sin3, __distance, global_args):

    """
    Applies HB ratio transformation
    """

    OverlapInfo = namedtuple('OverlapInfo', 'tvar sint overlap temp')

    __temp = (1.0 / __cos1) + (1.0 / __cos2)

    __cost = global_args.hb * da.sqrt(__distance * __distance + __tan1 * __tan1 * __tan2 * __tan2 * __sin3 * __sin3) / __temp

    __cost = da.clip(__cost, -1, 1)

    __tvar = da.arccos(__cost)
    __sint = da.sin(__tvar)
    __overlap = global_args.m_1_pi * (__tvar - __sint * __cost) * __temp

    __overlap = da.where(__overlap < 0, 0, __overlap)

    return OverlapInfo(tvar=__tvar, sint=__sint, overlap=__overlap, temp=__temp)


def li_kernel_delayed(angle_info, global_args):

    """Private method - call to calculate Li Kernel"""

    # LiKernelOutputs = namedtuple('LiKernelOutputs', 'li phi cos1 cos2 cos3 sin1 sin2 sin3 tan1 tan2 tanti tantv cosphaang phaang sinphaang distance')

    # at some point add in LiGround kernel & LiTransit
    # TODO
    # if self.LiType == 'Roujean':
    #     return self.RoujeanKernel()

    # first make sure its in range 0 to 2 pi
    __phi = da.fabs((da.from_delayed(angle_info.raa, shape=(global_args.size,), dtype=global_args.dtype).rechunk(global_args.flat_chunks) % (2.0 * global_args.m_pi)))
    __cos3 = da.cos(__phi)
    __sin3 = da.sin(__phi)
    __tanti = da.tan(da.from_delayed(angle_info.sza, shape=(global_args.size,), dtype=global_args.dtype).rechunk(global_args.flat_chunks))
    __tantv = da.tan(da.from_delayed(angle_info.vza, shape=(global_args.size,), dtype=global_args.dtype).rechunk(global_args.flat_chunks))
    __cos1, __sin1, __tan1 = get_pangles_dask(__tantv, global_args)
    __cos2, __sin2, __tan2 = get_pangles_dask(__tanti, global_args)

    # sets cos & sin phase angle terms
    __cosphaang, __phaang, __sinphaang = get_phase_angle_dask(__cos1, __cos2, __sin1, __sin2, __cos3)

    __distance = get_distance_dask(__tan1, __tan2, __cos3)

    # OverlapInfo = namedtuple('OverlapInfo', 'tvar sint overlap temp')
    overlap_info = get_overlap_dask(__cos1, __cos2, __tan1, __tan2, __sin3, __distance, global_args)

    if global_args.li_type.lower() == 'sparse':

        if global_args.recip_flag == True:
            li = overlap_info.overlap - overlap_info.temp + 0.5 * (1.0 + __cosphaang) / __cos1 / __cos2
        else:
            li = overlap_info.overlap - overlap_info.temp + 0.5 * (1.0 + __cosphaang) / __cos1

    # TODO
    # else:
    #     if self.LiType == 'Dense':
    #         if self.RecipFlag:
    #             self.Li = (1.0 + self.__cosphaang) / (
    #             self.__cos1 * self.__cos2 * (self.__temp - self.__overlap)) - 2.0;
    #         else:
    #             self.Li = (1.0 + self.__cosphaang) / (self.__cos1 * (self.__temp - self.__overlap)) - 2.0;
    #     else:
    #         B = self.__temp - self.__overlap
    #         w = np.where(B <= 2.0)
    #         self.Li = B * 0.0
    #         if self.RecipFlag == True:
    #             Li = self.__overlap - self.__temp + 0.5 * (1. + self.__cosphaang) / self.__cos1 / self.__cos2;
    #         else:
    #             Li = self.__overlap - self.__temp + 0.5 * (1. + self.__cosphaang) / self.__cos1;
    #         self.Li[w] = Li[w]
    #
    #         w = np.where(B > 2.0)
    #         if self.RecipFlag:
    #             Li = (1.0 + self.__cosphaang) / (self.__cos1 * self.__cos2 * (self.__temp - self.__overlap)) - 2.0;
    #         else:
    #             Li = (1.0 + self.__cosphaang) / (self.__cos1 * (self.__temp - self.__overlap)) - 2.0;
    #         self.Li[w] = Li[w]

    # return LiKernelOutputs(li=li, phi=__phi, cos1=__cos1, cos2=__cos2, cos3=__cos3, sin1=__sin1, sin2=__sin2, sin3=__sin3,
    #                        tan1=__tan1, tan2=__tan2, tanti=__tanti,
    #                        tantv=__tantv, cosphaang=__cosphaang, phaang=__phaang, sinphaang=__sinphaang,
    #                        distance=__distance)

    return li


# def LiKernel(self):
#
#     """Private method - call to calculate Li Kernel"""
#
#     # at some point add in LiGround kernel & LiTransit
#     if self.LiType == 'Roujean':
#         return self.RoujeanKernel()
#     # first make sure its in range 0 to 2 pi
#     self.__phi = np.abs((self.raa % (2. * self.__M_PI)))
#     self.__cos3 = np.cos(self.__phi)
#     self.__sin3 = np.sin(self.__phi)
#     self.__tanti = np.tan(self.sza)
#     self.__tantv = np.tan(self.vza)

#     self.__cos1, self.__sin1, self.__tan1 = self.GetpAngles(self.__tantv);
#     self.__cos2, self.__sin2, self.__tan2 = self.GetpAngles(self.__tanti);
#     self.__GetPhaang();  # sets cos & sin phase angle terms
#     self.__distance = self.GetDistance();  # sets self.temp
#     self.GetOverlap();  # also sets self.temp
#     if self.LiType == 'Sparse':
#         if self.RecipFlag == True:
#             self.Li = self.__overlap - self.__temp + 0.5 * (1. + self.__cosphaang) / self.__cos1 / self.__cos2;
#         else:
#             self.Li = self.__overlap - self.__temp + 0.5 * (1. + self.__cosphaang) / self.__cos1;
#     else:
#         if self.LiType == 'Dense':
#             if self.RecipFlag:
#                 self.Li = (1.0 + self.__cosphaang) / (
#                 self.__cos1 * self.__cos2 * (self.__temp - self.__overlap)) - 2.0;
#             else:
#                 self.Li = (1.0 + self.__cosphaang) / (self.__cos1 * (self.__temp - self.__overlap)) - 2.0;
#         else:
#             B = self.__temp - self.__overlap
#             w = np.where(B <= 2.0)
#             self.Li = B * 0.0
#             if self.RecipFlag == True:
#                 Li = self.__overlap - self.__temp + 0.5 * (1. + self.__cosphaang) / self.__cos1 / self.__cos2;
#             else:
#                 Li = self.__overlap - self.__temp + 0.5 * (1. + self.__cosphaang) / self.__cos1;
#             self.Li[w] = Li[w]
#
#             w = np.where(B > 2.0)
#             if self.RecipFlag:
#                 Li = (1.0 + self.__cosphaang) / (self.__cos1 * self.__cos2 * (self.__temp - self.__overlap)) - 2.0;
#             else:
#                 Li = (1.0 + self.__cosphaang) / (self.__cos1 * (self.__temp - self.__overlap)) - 2.0;
#             self.Li[w] = Li[w]
#     return


def li_kernel_dask(angle_info, global_args):

    """
    Private method - call to calculate Li Kernel
    """

    # first make sure its in range 0 to 2 pi
    __phi = da.fabs((angle_info.raa_rad % (2.0 * global_args.m_pi)))
    __cos3 = da.cos(__phi)
    __sin3 = da.sin(__phi)

    __tanti = da.tan(angle_info.sza_rad)
    __tantv = da.tan(angle_info.vza_rad)

    __cos1, __sin1, __tan1 = get_pangles_dask(__tantv, global_args)
    __cos2, __sin2, __tan2 = get_pangles_dask(__tanti, global_args)

    # sets cos & sin phase angle terms
    __cosphaang, __phaang, __sinphaang = get_phase_angle_dask(__cos1, __cos2, __sin1, __sin2, __cos3)

    __distance = get_distance_dask(__tan1, __tan2, __cos3)

    # OverlapInfo = namedtuple('OverlapInfo', 'tvar sint overlap temp')
    overlap_info = get_overlap_dask(__cos1, __cos2, __tan1, __tan2, __sin3, __distance, global_args)

    if global_args.li_type.lower() == 'sparse':

        if global_args.recip_flag:
            li = overlap_info.overlap - overlap_info.temp + 0.5 * (1.0 + __cosphaang) / __cos1 / __cos2
        else:
            li = overlap_info.overlap - overlap_info.temp + 0.5 * (1.0 + __cosphaang) / __cos1

    else:

        if global_args.li_type.lower() == 'dense':

            if global_args.recip_flag:
                li = (1.0 + __cosphaang) / (__cos1 * __cos2 * (overlap_info.temp - overlap_info.overlap)) - 2.0
            else:
                li = (1.0 + __cosphaang) / (__cos1 * (overlap_info.temp - overlap_info.overlap)) - 2.0

        else:

            b = overlap_info.temp - overlap_info.overlap
            li = b * 0.0

            if global_args.recip_flag:
                li_ = global_args.overlap - global_args.temp + 0.5 * (1.0 + __cosphaang) / __cos1 / __cos2
            else:
                li_ = overlap_info.overlap - overlap_info.temp + 0.5 * (1.0 + __cosphaang) / __cos1

            li = da.where(b <= 2, li_, li)

            if global_args.recip_flag:
                li_ = (1.0 + __cosphaang) / (__cos1 * __cos2 * (overlap_info.temp - overlap_info.overlap)) - 2.0
            else:
                li_ = (1.0 + __cosphaang) / (__cos1 * (overlap_info.temp - overlap_info.overlap)) - 2.0

            li = da.where(b <= 2, li_, li)

    return li


@dask.delayed
def set_angle_info_delayed(vza, sza, raa, global_args):

    """
    Store and organizes the input angle data
    """

    AngleInfo = namedtuple('AngleInfo', 'n vza_degrees sza_degrees raa_degrees vza sza raa')

    vza_degrees = np.array(vza).flatten()
    sza_degrees = np.array(sza).flatten()
    raa_degrees = np.array(raa).flatten()

    n = vza_degrees.shape[0]

    if n != len(sza_degrees) or n != len(raa_degrees):

        logger.error('kernels: inconsistent number of samples in vza, sza and raa data: ' + str(len(vza_degrees)) + ', ' + str(len(sza_degrees)) + ', ' + str(len(raa_degrees)))
        print(vza_degrees)
        print(sza_degrees)
        print(raa_degrees)
        return [-1]

    if global_args.normalize >= 1:

        # calculate nadir term by extending array
        vza_degrees = np.array(vza_degrees.tolist() + [0.0]).flatten()
        sza_degrees = np.array(sza_degrees.tolist() + [global_args.nbar]).flatten()
        raa_degrees = np.array(raa_degrees.tolist() + [0.0]).flatten()

        n = len(vza_degrees)

    vza = dtor(vza_degrees)
    sza = dtor(sza_degrees)  # -1 to make HS direction for raa = 0
    raa = dtor(raa_degrees)

    w = np.where(vza < 0)[0]
    vza[w] = -vza[w]
    raa[w] = raa[w] + global_args.m_pi
    w = np.where(sza < 0)[0]
    sza[w] = -sza[w]
    raa[w] = raa[w] + global_args.m_pi

    return AngleInfo(n=n,
                     vza_degrees=vza_degrees,
                     sza_degrees=sza_degrees,
                     raa_degrees=raa_degrees,
                     vza=vza,
                     sza=sza,
                     raa=raa)


def set_angle_info_delayed_2d(vza, sza, raa, global_args):

    """
    Store and organizes the input angle data
    """

    AngleInfo = namedtuple('AngleInfo', 'vza sza raa vza_rad sza_rad raa_rad')

    # if global_args.normalize >= 1:
    #
    #     # calculate nadir term by extending array
    #     vza = np.array(vza.tolist() + [0.0]).flatten()
    #     sza = np.array(sza.tolist() + [global_args.nbar]).flatten()
    #     raa = np.array(raa.tolist() + [0.0]).flatten()
    #
    #     n = len(vza_degrees)

    vza_rad = da.deg2rad(vza)
    sza_rad = da.deg2rad(sza)
    raa_rad = da.deg2rad(raa)

    vza_abs = da.fabs(vza_rad)
    sza_abs = da.fabs(sza_rad)

    raa_abs = da.where((vza_rad < 0) | (sza_rad < 0), global_args.m_pi, raa_rad)

    return AngleInfo(vza=vza,
                     sza=sza,
                     raa=raa,
                     vza_rad=vza_abs,
                     sza_rad=sza_abs,
                     raa_rad=raa_abs)


def post_process_ross_delayed(ross, angle_info, global_args):

    # li_norm = 0.0
    # ross_norm = 0.0
    # isotropic_norm = 0.0

    # if we are normalising the last element of self.Isotropic, self.Ross and self.Li contain the nadir-nadir kernel

    if global_args.normalize >= 1:

        # normalise nbar-nadir (so kernel is 0 at nbar-nadir)
        ross_norm = ross[-1]
        ross = ross - ross_norm

        # depreciate length of arrays
        # ross = ross[0:-1]

        # TODO
        # if hasattr(self, 'Isotropic'):
        #     self.Isotropic = self.Isotropic[0:-1]

        # angle_info = angle_info._update(vza_degrees=angle_info.vza_degrees[0:-1])
        # angle_info = angle_info._update(sza_degrees=angle_info.sza_degrees[0:-1])
        # angle_info = angle_info._update(raa_degrees=angle_info.raa_degrees[0:-1])

        # angle_info = angle_info._update(n=len(angle_info.vza_degrees))
        # angle_info = angle_info._update(vza=angle_info.vza[0:-1])
        # angle_info = angle_info._update(sza=angle_info.sza[0:-1])
        # angle_info = angle_info._update(raa=angle_info.raa[0:-1])

        # return angle_info, ross_kernel_outputs, li_kernel_outputs

    return ross


def post_process_li_delayed(li, angle_info, global_args):

    """
    Handles with normalisation
    """

    # li_norm = 0.0
    # ross_norm = 0.0
    # isotropic_norm = 0.0

    # if we are normalising the last element of self.Isotropic, self.Ross and self.Li contain the nadir-nadir kernel

    if global_args.normalize >= 1:

        # normalise nbar-nadir (so kernel is 0 at nbar-nadir)
        li_norm = li[-1]
        li = li - li_norm

        # depreciate length of arrays
        # li = li[0:-1]

        # TODO
        # if hasattr(self, 'Isotropic'):
        #     self.Isotropic = self.Isotropic[0:-1]

        # angle_info = angle_info._update(vza_degrees=angle_info.vza_degrees[0:-1])
        # angle_info = angle_info._update(sza_degrees=angle_info.sza_degrees[0:-1])
        # angle_info = angle_info._update(raa_degrees=angle_info.raa_degrees[0:-1])

        # angle_info = angle_info._update(n=len(angle_info.vza_degrees))
        # angle_info = angle_info._update(vza=angle_info.vza[0:-1])
        # angle_info = angle_info._update(sza=angle_info.sza[0:-1])
        # angle_info = angle_info._update(raa=angle_info.raa[0:-1])

        # return angle_info, ross_kernel_outputs, li_kernel_outputs

    return li


@dask.delayed
def post_process_delayed(ross_kernel_outputs, li_kernel_outputs, angle_info, global_args):

    """
    Handles with normalisation
    """

    li_norm = 0.0
    ross_norm = 0.0
    isotropic_norm = 0.0

    # if we are normalising the last element of self.Isotropic, self.Ross and self.Li contain the nadir-nadir kernel

    if global_args.normalize >= 1:

        # normalise nbar-nadir (so kernel is 0 at nbar-nadir)
        ross_norm = ross_kernel_outputs.ross[-1]
        li_norm = li_kernel_outputs.li[-1]
        ross_kernel_outputs = ross_kernel_outputs._update(ross=ross_kernel_outputs.ross - ross_norm)
        li_kernel_outputs = li_kernel_outputs._update(li=li_kernel_outputs.li - li_norm)

        # depreciate length of arrays
        ross_kernel_outputs = ross_kernel_outputs._update(ross=ross_kernel_outputs.ross[0:-1])
        li_kernel_outputs = li_kernel_outputs._update(li=li_kernel_outputs.li[0:-1])

        # TODO
        # if hasattr(self, 'Isotropic'):
        #     self.Isotropic = self.Isotropic[0:-1]

        angle_info = angle_info._update(vza_degrees=angle_info.vza_degrees[0:-1])
        angle_info = angle_info._update(sza_degrees=angle_info.sza_degrees[0:-1])
        angle_info = angle_info._update(raa_degrees=angle_info.raa_degrees[0:-1])

        angle_info = angle_info._update(n=len(angle_info.vza_degrees))
        angle_info = angle_info._update(vza=angle_info.vza[0:-1])
        angle_info = angle_info._update(sza=angle_info.sza[0:-1])
        angle_info = angle_info._update(raa=angle_info.raa[0:-1])

        return angle_info, ross_kernel_outputs, li_kernel_outputs


class Kernels(object):

    """
    Linear kernel models

    Source:
        Code taken from:
            https://github.com/jgomezdans/eoldas/blob/master/eoldas/kernels.py
    """

    def __init__(self,
                 vza,
                 sza,
                 raa,
                 delayed=False,
                 critical=1,
                 RossHS=True,
                 RecipFlag=True,
                 HB=2.0,
                 BR=1.0,
                 MODISSPARSE=True,
                 MODISDENSE=False,
                 RossType='Thick',
                 normalise=1,
                 normalize=0,
                 LiType='sparse',
                 doIntegrals=True,
                 BSAangles=None,
                 nbar=0.0):

        """
        The class creator sets up the kernels for some angle set. Default Li is MODISSPARSE parameter set

        The kernels are accessible from:
            self.Isotropic
            self.Ross
            self.Li
        The angles are accesible from:
            self.vza (or self.vzaDegrees)
            self.sza (or self.szaDegrees)
            self.raa (or self.raaDegrees)
            N.B. Hot spot direction is vza == sza and raa = 0.0
        Kernels integrals are acessible from:
            self.BSAangles (angles in degrees)
            self.BSA_Isotropic (directional-hemispherical integral of self.Isotropic)
            self.BSA_Ross (directional-hemispherical integral of self.Ross)
            self.BSA_Li (directional-hemispherical integral of self.Li)
            self.WSA_Isotropic (bi-hemispherical integral of self.Isotropic)
            self.WSA_Ross (bi-hemispherical integral of self.Ross)
            self.WSA_Li (bi-hemispherical integral of self.Li)
            N.B. You need to set the doIntegrals flag to True on creating an instance of the kernels class if you
            want access to integrals. The processing takes a bit of time.
        Printing methods are available:
            self.printIntegrals(header=True,reflectance=False)
            self.printKernels(header=True,reflectance=False)

        Required parameters:

            @param vza: an array containing view zenith angles in degrees
            @param sza: an array containing solar zenith angles in degrees
            @param raa: an array containing relative azimuth angles in degrees

        Options:
            @option critical=1: set to 1 to exit on error, 0 not to
            @option RecipFlag=True: Li reciprocal flag
            @option HB: Li kernel parameter HB
            @option BR: Li kernel parameter
            @option MODISSPARSE: set to True for default MODIS Li Sparse parameters (overrides BR and HB to 2.0 and 1.0)
            @option MODISDENSE: set to True for default MODIS Li Dense parameters (override BR and HB to 2.0 and 2.5)
            @option RossType: set to 'Thin' for Ross Thin (default) else 'Thick'
            @option LiType: set to 'Sparse' for LiSparse (default). Other options: 'Roujean', 'Dense'
            @option normalise: set to 1 to make kernels 0 at nadir view illumination (default), set to 0 for no normalisation (can also use US spelling, i.e. normalize)
            @option doIntegrals: set to True to calculate integrals of kernels numerically. Set to False not to calculate them. At some point will have Approx flag here as well.
            @option BSAangles: solar zenith angles at which to calculate directional-hemispherical integral of kernels (default 0-89 in steps of 1 degree). Units: degrees.
            @option nbar: the sza at which the isotropic term is set to if normalise=1 is turned on (default 0)

        Notes:
        Requires numpy. If you do integrals, this also requires scipy (or rather scipy.integrate)
        If you want to mimic the results in Wanner et al. 1995, I've set a special function called self.mimic at the end here.
        """

        GlobalArgs = namedtuple('GlobalArgs', 'nrows ncols size dtype flat_chunks nbar nearly_zero critical file_ outputfile li_type ross_type ross_hs do_integrals hb br normalize recip_flag m_pi m_pi2 m_pi4 m_1_pi')

        self.__setup(critical=critical,
                     RecipFlag=RecipFlag,
                     RossHS=RossHS,
                     HB=HB,
                     BR=BR,
                     MODISSPARSE=MODISSPARSE,
                     MODISDENSE=MODISDENSE,
                     RossType=RossType,
                     normalise=normalise,
                     normalize=normalize,
                     LiType=LiType,
                     doIntegrals=doIntegrals,
                     BSAangles=BSAangles,
                     nbar=nbar)

        if delayed:

            # TODO: make `flat_chunks` a user argument
            global_args = GlobalArgs(nrows=vza.shape[-2], ncols=vza.shape[-1], size=vza.size, dtype=vza.dtype.name,
                                     flat_chunks=np.array(list(vza.chunksize)).sum(),
                                     nbar=nbar, nearly_zero=self.__NEARLYZERO, critical=self.critical,
                                     file_=self.FILE, outputfile=self.outputFile,
                                     li_type=self.LiType, ross_type=self.RossType, ross_hs=self.RossHS,
                                     do_integrals=self.doIntegrals, hb=self.HB, br=self.BR, normalize=self.normalise,
                                     recip_flag=self.RecipFlag,
                                     m_pi=self.__M_PI, m_pi2=self.__M_PI_2, m_pi4=self.__M_PI_4, m_1_pi=self.__M_1_PI)

            # AngleInfo = namedtuple('AngleInfo', 'n vza sza raa vza_rad sza_rad raa_rad')
            # angle_info = set_angle_info_delayed(vza, sza, raa, global_args)
            angle_info = set_angle_info_delayed_2d(vza, sza, raa, global_args)

            # RossKernelOutputs = namedtuple('RossKernelOutputs', 'cos1 cos2 sin1 sin2 cos3 rosselement cosphaang phaang sinphaang ross')
            ross = ross_kernel_dask(angle_info, global_args)

            # LiKernelOutputs = namedtuple('LiKernelOutputs','li phi cos1 cos2 cos3 sin1 sin2 sin3 tan1 tan2 tanti tantv cosphaang phaang sinphaang distance')
            li = li_kernel_dask(angle_info, global_args)

            # TODO: add normalization for 2d dask arrays
            # ross = post_process_ross_delayed(ross, angle_info, global_args)
            # li = post_process_li_delayed(li, angle_info, global_args)

            # angle_info, ross_kernel_outputs, li_kernel_outputs = post_process_delayed(ross_kernel_outputs,
            #                                                                           li_kernel_outputs,
            #                                                                           angle_info,
            #                                                                           global_args)

            self.Ross = ross
            self.Li = li

        else:

            self.set_angle_info(vza, sza, raa)
            self.__doKernels()
            self.__postProcess()

    def __setup(self,
                critical=1,
                RecipFlag=True,
                RossHS=True,
                HB=2.0,
                BR=1.0,
                MODISSPARSE=True,
                MODISDENSE=False,
                RossType='Thick',
                normalise=1,
                normalize=0,
                LiType='Sparse',
                doIntegrals=True,
                BSAangles=None,
                nbar=0.0):

        self.nbar = nbar
        self.__NEARLYZERO = 1e-20
        self.critical = critical
        self.FILE = -1
        self.outputFile = ''
        # kernel options etc.
        self.LiType = LiType
        self.RossHS = RossHS
        self.doIntegrals = doIntegrals

        if MODISDENSE:

            LiType = 'Dense'
            self.HB = 2.
            self.BR = 2.5

        else:

            if MODISSPARSE:

                LiType = 'Sparse'
                self.HB = 2.0
                self.BR = 1.0

            else:

                self.HB = HB
                self.BR = BR

        # self.LiType = LiType
        self.RossType = RossType
        self.normalise = normalise
        self.RecipFlag = RecipFlag

        # some useful numbers
        self.__M_PI = np.pi
        self.__M_PI_2 = self.__M_PI * 0.5
        self.__M_PI_4 = self.__M_PI * 0.25
        self.__M_1_PI = 1.0 / self.__M_PI

        self.normalize = self.normalise

        # TODO
        # if self.doIntegrals:
        #     self.__integrateKernels(BSAangles=BSAangles)

        if (normalise >= 1) or (normalize >= 1):
            self.normalise = max(normalise, normalize)

    def __postProcess(self):

        """Private method for dealing with normalisation"""

        self.LiNorm = 0.
        self.RossNorm = 0.
        self.IsotropicNorm = 0.
        # if we are normalising the last element of self.Isotropic, self.Ross and self.Li  contain the nadir-nadir kernel

        if self.normalise >= 1:

            # normalise nbar-nadir (so kernel is 0 at nbar-nadir)
            self.RossNorm = self.Ross[-1]
            self.LiNorm = self.Li[-1]
            self.Ross = self.Ross - self.RossNorm
            self.Li = self.Li - self.LiNorm
            # depreciate length of arrays (well, teh ones we'll use again in any case)
            self.Ross = self.Ross[0:-1]
            self.Li = self.Li[0:-1]

            if hasattr(self, 'Isotropic'):
                self.Isotropic = self.Isotropic[0:-1]

            self.vzaDegrees = self.vzaDegrees[0:-1]
            self.szaDegrees = self.szaDegrees[0:-1]
            self.raaDegrees = self.raaDegrees[0:-1]
            self.N = len(self.vzaDegrees)
            self.vza = self.vza[0:-1]
            self.sza = self.sza[0:-1]
            self.raa = self.raa[0:-1]

    def __doKernels(self):

        """Private method to run the various kernel methods"""

        # the kernels
        # self.IsotropicKernel()
        self.RossKernel()
        self.LiKernel()

    def set_angle_info(self, vza, sza, raa):

        """Private method to store and organise the input angle data"""

        self.vzaDegrees = np.array([vza]).flatten()
        self.szaDegrees = np.array([sza]).flatten()
        self.raaDegrees = np.array([raa]).flatten()
        self.N = len(self.vzaDegrees)

        if (self.N != len(self.szaDegrees) or self.N != len(self.raaDegrees)):
            self.error('kernels: inconsistent number of samples in vza, sza and raa data: ' + str(
                len(self.vzaDegrees)) + ', ' + str(len(self.szaDegrees)) + ', ' + str(len(self.raaDegrees)),
                       critical=self.critical)
            print(self.vzaDegrees)
            print(self.szaDegrees)
            print(self.raaDegrees)
            return [-1]

        if (self.normalise >= 1):
            # calculate nadir term by extending array
            self.vzaDegrees = np.array(list(self.vzaDegrees) + [0.0]).flatten()
            self.szaDegrees = np.array(list(self.szaDegrees) + [self.nbar]).flatten()
            self.raaDegrees = np.array(list(self.raaDegrees) + [0.0]).flatten()
            # not N is one too many now
            self.N = len(self.vzaDegrees)

        self.vza = self.dtor(self.vzaDegrees)
        self.sza = self.dtor(self.szaDegrees)  # -1 to make HS direction for raa = 0
        self.raa = self.dtor(self.raaDegrees)
        w = np.where(self.vza < 0)[0]
        self.vza[w] = -self.vza[w]
        self.raa[w] = self.raa[w] + self.__M_PI
        w = np.where(self.sza < 0)[0]
        self.sza[w] = -self.sza[w]
        self.raa[w] = self.raa[w] + self.__M_PI

    def __integrateKernels(self, BSAangles=[]):

        """
        Private method to call integration functions for the kernels


             NB - this overwrites all kernel info ... so be careful how/where you call it
            @option: BSAangles=[] allows the user to set the sza angles at which directional-hemispherical intergal is
                calculated, else steps of 1 degree from 0 to 89 (though I wouldnt trust it down to 90)
            This function can be rather slow, so using fewer samples or an approximate function may be a god idea
        """

        import scipy.integrate
        if BSAangles == []:
            BSAangles = np.array(range(90)) * 1.0

        self.BSAangles = np.array(BSAangles).flatten()

        # isotropic integral
        self.BSA_Isotropic = np.zeros(len(self.BSAangles)) + 1.0
        self.BSA_Ross = np.zeros(len(self.BSAangles))
        self.BSA_Li = np.zeros(len(self.BSAangles))
        self.BSA_Isotropic_error = np.zeros(len(self.BSAangles))
        self.BSA_Ross_error = np.zeros(len(self.BSAangles))
        self.BSA_Li_error = np.zeros(len(self.BSAangles))

        i = 0
        mu = np.cos(self.BSAangles * self.__M_PI / 180.)
        for sza in self.BSAangles:
            # ross integral
            self.BSA_Ross[i], self.BSA_Ross_error[i] = scipy.integrate.dblquad(RossFunctionForIntegral, 0.0, 1.0,
                                                                               __gfun, __hfun, args=(sza, self))
            self.BSA_Li[i], self.BSA_Li_error[i] = scipy.integrate.dblquad(LiFunctionForIntegral, 0.0, 1.0, __gfun,
                                                                           __hfun, args=(sza, self))
            i = i + 1
        self.WSA_Ross = -2.0 * scipy.integrate.simps(self.BSA_Ross * mu, mu)
        self.WSA_Li = -2.0 * scipy.integrate.simps(self.BSA_Li * mu, mu)
        return

    def __GetPhaang(self):

        """Private method to calculate Phase angle component of kernel"""

        self.__cosphaang = self.__cos1 * self.__cos2 + self.__sin1 * self.__sin2 * self.__cos3
        # better check the bounds before arccos ... just to be safe
        w = np.where(self.__cosphaang < -1)[0]
        self.__cosphaang[w] = -1.0
        w = np.where(self.__cosphaang > 1)[0]
        self.__cosphaang[w] = 1.0
        self.__phaang = np.arccos(self.__cosphaang)
        self.__sinphaang = np.sin(self.__phaang)

    def __RossKernelPart(self):

        """Private method to calculate main part of Ross kernel"""

        self.__cos1 = np.cos(self.vza)
        self.__cos2 = np.cos(self.sza)

        self.__sin1 = np.sin(self.vza)
        self.__sin2 = np.sin(self.sza)
        self.__cos3 = np.cos(self.raa)
        self.__GetPhaang()

        self.rosselement = (self.__M_PI_2 - self.__phaang) * self.__cosphaang + self.__sinphaang

    def GetDistance(self):

        """Private method to get distance component of Li kernels"""

        temp = self.__tan1 * self.__tan1 + self.__tan2 * self.__tan2 - 2. * self.__tan1 * self.__tan2 * self.__cos3;
        w = np.where(temp < 0)[0]
        temp[w] = 0.0
        self.__temp = temp  # used by other functions ??
        distance = np.sqrt(temp)

        return distance

    def GetpAngles(self, tan1):

        """Private method to do B/R transformation for ellipse shape"""

        t = self.BR * tan1
        w = np.where(t < 0.)[0]
        t[w] = 0.0
        angp = np.arctan(t)
        s = np.sin(angp)
        c = np.cos(angp)
        # have to make sure c isnt 0
        w = np.where(c == 0)[0]
        c[w] = self.__NEARLYZERO

        return c, s, t

    def GetOverlap(self):

        """Private method to do HB ratio transformation"""

        self.__temp = 1. / self.__cos1 + 1. / self.__cos2

        self.__cost = self.HB * np.sqrt(
            self.__distance * self.__distance + self.__tan1 * self.__tan1 * self.__tan2 * self.__tan2 * self.__sin3 * self.__sin3) / self.__temp;
        w = np.where(self.__cost < -1)[0]
        self.__cost[w] = -1.0
        w = np.where(self.__cost > 1.0)[0]
        self.__cost[w] = 1.0
        self.__tvar = np.arccos(self.__cost)
        self.__sint = np.sin(self.__tvar)
        self.__overlap = self.__M_1_PI * (self.__tvar - self.__sint * self.__cost) * self.__temp
        w = np.where(self.__overlap < 0)[0]
        self.__overlap[w] = 0.0
        return

    def RoujeanKernel(self):

        """Private method - call to calculate Roujean shadowing kernel"""

        # first make sure its in range 0 to 2 pi
        self.__phi = np.abs((self.raa % (2. * self.__M_PI)))
        self.__cos3 = np.cos(self.__phi)
        self.__sin3 = np.sin(self.__phi)
        self.__tan1 = np.tan(self.sza)
        self.__tan2 = np.tan(self.vza)

        self.__distance = self.GetDistance()
        self.Li = 0.5 * self.__M_1_PI * (
        (self.__M_PI - self.__phi) * self.__cos3 + self.__sin3) * self.__tan1 * self.__tan2 - self.__M_1_PI * (
        self.__tan1 + self.__tan2 + self.__distance);
        return

    def LiKernel(self):

        """Private method - call to calculate Li Kernel"""

        # at some point add in LiGround kernel & LiTransit
        if self.LiType == 'Roujean':
            return self.RoujeanKernel()
        # first make sure its in range 0 to 2 pi
        self.__phi = np.abs((self.raa % (2. * self.__M_PI)))
        self.__cos3 = np.cos(self.__phi)
        self.__sin3 = np.sin(self.__phi)
        self.__tanti = np.tan(self.sza)
        self.__tantv = np.tan(self.vza)
        self.__cos1, self.__sin1, self.__tan1 = self.GetpAngles(self.__tantv);
        self.__cos2, self.__sin2, self.__tan2 = self.GetpAngles(self.__tanti);
        self.__GetPhaang();  # sets cos & sin phase angle terms
        self.__distance = self.GetDistance();  # sets self.temp
        self.GetOverlap();  # also sets self.temp
        if self.LiType == 'Sparse':
            if self.RecipFlag == True:
                self.Li = self.__overlap - self.__temp + 0.5 * (1. + self.__cosphaang) / self.__cos1 / self.__cos2;
            else:
                self.Li = self.__overlap - self.__temp + 0.5 * (1. + self.__cosphaang) / self.__cos1;
        else:
            if self.LiType == 'Dense':
                if self.RecipFlag:
                    self.Li = (1.0 + self.__cosphaang) / (
                    self.__cos1 * self.__cos2 * (self.__temp - self.__overlap)) - 2.0;
                else:
                    self.Li = (1.0 + self.__cosphaang) / (self.__cos1 * (self.__temp - self.__overlap)) - 2.0;
            else:
                B = self.__temp - self.__overlap
                w = np.where(B <= 2.0)
                self.Li = B * 0.0
                if self.RecipFlag == True:
                    Li = self.__overlap - self.__temp + 0.5 * (1. + self.__cosphaang) / self.__cos1 / self.__cos2;
                else:
                    Li = self.__overlap - self.__temp + 0.5 * (1. + self.__cosphaang) / self.__cos1;
                self.Li[w] = Li[w]

                w = np.where(B > 2.0)
                if self.RecipFlag:
                    Li = (1.0 + self.__cosphaang) / (self.__cos1 * self.__cos2 * (self.__temp - self.__overlap)) - 2.0;
                else:
                    Li = (1.0 + self.__cosphaang) / (self.__cos1 * (self.__temp - self.__overlap)) - 2.0;
                self.Li[w] = Li[w]
        return

    def IsotropicKernel(self):

        """Public method - call to calculate Isotropic kernel"""

        # default behaviour
        self.Isotropic = np.zeros(self.N) + 1.0

        return

    def RossThin(self):

        """Public method - call to calculate RossThin kernel"""

        self.__RossKernelPart()
        self.rosselement = self.rosselement / (self.__cos1 * self.__cos2)

        return

    def RossThick(self):

        """Public method - call to calculate RossThick kernel"""

        self.__RossKernelPart()
        self.rosselement = self.rosselement / (self.__cos1 + self.__cos2)

        return

    def RossKernel(self):

        """Public method - call to calculate Ross Kernel"""

        if self.RossType == 'Thin':
            self.RossThin()
        else:
            self.RossThick()

        self.Ross = self.rosselement

        if self.RossHS:
            self.Ross = self.Ross * (1. + 1. / (1. + self.__phaang / .25))

    def dtor(self, x):

        """Public method to convert degrees to radians"""

        return x * self.__M_PI / 180.

    def rtod(self, x):

        """Public method to convert radians to degrees"""

        return x * 180. / self.__M_PI

    def error(self, msg, critical=0, newline=1, code=-1):

        """
        Public method to do Class error reporting
        @param msg: error message
        @param critical: set to 1 if require exit (default critical=0)
        @param newline: set to 0 if newline not required (default newline=0)
        @param code: error code reported on exit if critical error (default code=-1)
        """

        if newline == 1:
            nl = '\n'
        else:
            nl = ''
        print(msg + nl)
        if critical == 1:
            logger.exception([code])

    def printIntegrals(self, header=True, reflectance=False):

        """
        Public method to print kernel integrals (to stdout only at present)
        """

        if (header == True):
            self.printer(
                '# ' + str(self.N) + ' samples Ross: ' + self.RossType + ' Li: ' + self.LiType + ' Reciprocal: ' + str(
                    self.RecipFlag) + ' normalisation: ' + str(self.normalise) + ' HB ' + str(self.HB) + ' BR ' + str(
                    self.BR) + '\n');
            self.printer('# WSA: Isotropic 1.0 Ross ' + str(self.WSA_Ross) + ' Li ' + str(self.WSA_Li))
            self.printer('# 1: SZA (degrees) 2: BSA Isotropic 3: BSA Ross 4: BSA Li')
            if (reflectance == True):
                self.printer(' ');
            self.printer('\n');

        for i in range(len(self.BSAangles)):
            self.printer(
                str(self.BSAangles[i]) + ' ' + str(self.BSA_Isotropic[i]) + ' ' + str(self.BSA_Ross[i]) + ' ' + str(
                    self.BSA_Li[i]))
            # print refl data if wanted
            if (reflectance == True):
                self.printer(' ');
            self.printer('\n');
        return

    def printKernels(self, header=True, reflectance=False, file=False):

        """
        Public method to print kernel values (to stdout only at present)
        """

        if (file != False):
            if (file != self.outputFile and self.FILE != -1):
                self.FILE.close()
            self.outputFile = file
            self.FILE = open(self.outputFile, 'w')

        if (header == True):
            self.printer(
                '# ' + str(self.N) + ' samples Ross: ' + self.RossType + ' Li: ' + self.LiType + ' Reciprocal: ' + str(
                    self.RecipFlag) + ' normalisation: ' + str(self.normalise) + ' HB ' + str(self.HB) + ' BR ' + str(
                    self.BR) + '\n');
            self.printer('# 1: VZA (degrees) 2: SZA (degrees) 3: RAA (degrees) 4: Isotropic 5: Ross 6: Li')
            if (reflectance == True):
                self.printer(' ');
            self.printer('\n');

        for i in range(self.N):
            self.printer(
                str(self.vzaDegrees[i]) + ' ' + str(self.szaDegrees[i]) + ' ' + str(self.raaDegrees[i]) + ' ' + str(
                    self.Isotropic[i]) + ' ' + str(self.Ross[i]) + ' ' + str(self.Li[i]))
            # print refl data if wanted
            if (reflectance == True):
                self.printer(' ');
            self.printer('\n');
        return

    def printer(self, msg):

        """
        Public print method ... make more flexible eg for printing to files at some point
        """

        if (self.FILE == -1):
            print(msg)
        else:
            self.FILE.write(msg)


# some things required for the numerical integration

def _Kernels__gfun(x):
    return 0.0


def _Kernels__hfun(x):
    return 2.0 * np.pi


def RossFunctionForIntegral(phi, mu, sza, self):
    # print phi
    # print mu
    # print sza
    # print '========'
    vza = np.arccos(mu)
    raa = self.rtod(phi)
    self.set_angle_info(vza, sza, raa)
    self.RossKernel()
    return mu * self.Ross[0] / np.pi


def LiFunctionForIntegral(phi, mu, sza, self):
    # print phi
    # print mu
    # print sza
    # print '========'
    vza = np.arccos(mu)
    raa = self.rtod(phi)
    self.set_angle_info(vza, sza, raa)
    self.LiKernel()
    return mu * self.Li[0] / np.pi


def readASCII(inputFile, dobands=False):
    FILE = open(inputFile, 'r')
    header = FILE.readline()
    nBands = int(header.split()[2])
    bands = header.split()[3:3 + nBands]
    Bands = np.zeros(nBands)
    for i in range(nBands):
        Bands[i] = float(bands[i])
    strdata = FILE.readlines()
    FILE.close()
    N = len(strdata)
    DOY = np.zeros(N)
    FLAG = np.zeros(N)
    VZA = np.zeros(N)
    SZA = np.zeros(N)
    RAA = np.zeros(N)
    REFL = np.zeros([nBands, N])
    for i in range(N):
        s = strdata[i].split()
        DOY[i] = float(s[0])
        FLAG[i] = int(s[1])
        VZA[i] = float(s[2])
        SZA[i] = float(s[4])
        RAA[i] = float(s[3]) - float(s[5])
        for j in range(nBands):
            REFL[j, i] = float(s[j + 6])
    w = np.where(FLAG == 1)
    doy = DOY[w]
    vza = VZA[w]
    sza = SZA[w]
    raa = RAA[w]
    refl = REFL[:, w]
    if dobands == True:
        return vza, sza, raa, refl, doy, Bands
    else:
        return vza, sza, raa, refl, doy


def readPOLDER(inputFile, type=1):
    FILE = open(inputFile, 'r')
    strdata = FILE.readlines()
    FILE.close()
    N = len(strdata)
    VZA = np.zeros(N)
    SZA = np.zeros(N)
    RAA = np.zeros(N)
    REFL = np.zeros([5, N])
    for i in range(N):
        s = strdata[i].split()
        if (type == 1):
            VZA[i] = float(s[4])
            SZA[i] = float(s[2])
            RAA[i] = float(s[5])
            for j in range(5):
                REFL[j, i] = float(s[j + 6])
        else:
            if (type == 2):
                VZA[i] = float(s[2])
                SZA[i] = float(s[4])
                RAA[i] = float(s[5]) - float(s[3])
                for j in range(5):
                    REFL[j, i] = float(s[j + 6])
    return VZA, SZA, RAA, REFL


def legend(*args, **kwargs):
    """
    Overwrites the pylab legend function.

    It adds another location identfier 'outer right'
    which locates the legend on the right side of the plot

    The args and kwargs are forwarded to the pylab legend function

    from http://www.mail-archive.com/matplotlib-users@lists.sourceforge.net/msg04256.html
    """
    import pylab
    if kwargs.has_key('loc'):
        loc = kwargs['loc']
        loc = loc.split()

        if loc[0] == 'outer':
            # make a legend with out the location
            # remove the location setting from the kwargs
            kwargs.pop('loc')
            leg = pylab.legend(loc=(0, 0), *args, **kwargs)
            frame = leg.get_frame()
            currentAxes = pylab.gca()
            currentAxesPos = np.array(currentAxes.get_position()).flatten()

            # scale plot by the part which is taken by the legend
            plotScaling = frame.get_width() / currentAxesPos[2]

            if loc[1] == 'right':
                # scale the plot
                currentAxes.set_position((currentAxesPos[0], currentAxesPos[1] - 0.05,
                                          currentAxesPos[2] * (1 - plotScaling),
                                          currentAxesPos[3] - 0.05))
                # set x and y coordinates of legend
                # leg._loc = (1 + leg.axespad, 1 - frame.get_height())
                leg._loc = (1 + leg.axespad, 0)
            # doesn't work
            # if loc[1] == 'left':
            #    # scale the plot
            #    currentAxes.set_position((currentAxesPos[0] +frame.get_width(),
            #                              currentAxesPos[1],
            #                              currentAxesPos[2] *(1-plotScaling),
            #                              currentAxesPos[3]))
            #    # set x and y coordinates of legend
            #    leg._loc = (1 -.05 -  leg.axespad - frame.get_width(), 1 -frame.get_height())

            pylab.draw_if_interactive()
            return leg

    return pylab.legend(*args, **kwargs)


def lutInvertRossHS(VZA, SZA, RAA, REFL, N=1000, fixXi=False, RossType='Thick', LiType='Dense', normalise=1,
                    RecipFlag=True, MODISSPARSE=True):
    if (fixXi != False):
        N = 1
        rhs = np.array([fixXi])
    else:
        rhs = np.array(range(N)) * 10 * (np.pi / 180.) / N
    rmse = np.zeros(N)
    for i in range(N):
        rmse[i], P, FWD, phaseAngle = invertData(VZA, SZA, RAA, REFL, RossType=RossType, LiType=LiType, RossHS=rhs[i],
                                                 normalise=normalise, RecipFlag=RecipFlag, MODISSPARSE=MODISSPARSE)
    i = np.argmin(rmse)
    RMSE, P, FWD, phaseAngle = invertData(VZA, SZA, RAA, REFL, RossType=RossType, LiType=LiType, RossHS=rhs[i],
                                          normalise=normalise, RecipFlag=RecipFlag, MODISSPARSE=MODISSPARSE)
    return RMSE, rhs[i], P, np.array(FWD), rhs, rmse, phaseAngle


def testLisa(inputFile, buff=30, LiType='Sparse', RossType='Thick', plot=False, verbose=False, fsza=0.0,
             forcedoy=False):
    import pdb
    bu = [0.004, 0.015, 0.003, 0.004, 0.013, 0.010, 0.006]
    vza, sza, raa, refl, doy, bands = readASCII(inputFile, dobands=True)

    if type(fsza) == type(True) and fsza == True:
        msza = np.median(sza)
    else:
        msza = fsza
    if verbose == True:
        print('nbar at', msza)

    nbands = len(bands)
    if nbands == 4:
        bux = [bu[1], bu[4], bu[0], bu[6]]
    else:
        bux = bu
    mind = min(doy)
    maxd = max(doy)
    w1 = np.where(doy >= (mind + buff))
    w2 = np.where(doy[w1] <= (maxd - buff))
    sampledays = doy[w1][w2]
    if forcedoy != False:
        sampledays = np.array([forcedoy])
    iso = np.zeros(len(bux))
    isoPost = np.zeros(len(bux))
    sig = np.zeros([len(sampledays), len(bux)])
    rel = np.zeros([len(sampledays), len(bux)])
    RMSE = 1e20
    count = 0
    mindoy = False
    minrmse = False
    minP = False
    minFWD = False
    minrefl = False

    # stuff for spectral mixture model
    loff = 400.0
    lmax = 2000.0
    ll = bands - loff
    llmax = lmax - loff
    lk = ll - ll * ll / (2 * llmax)
    lk = lk / max(lk)
    K = np.matrix(np.ones([3, nbands]))
    K[1][:] = lk
    mincount = -1
    for dos in sampledays:
        rmse, P, FWD, refl, idoy, unc = lisaInvert(vza, sza, raa, refl, doy, dos, LiType=LiType, RossType=RossType,
                                                   nbar=msza)
        # calculate significance of step change in 1st 2 bands
        # P[i,6] is the magnitude of step change
        dos2 = dos + 1
        for i in range(len(bux)):
            iso[i] = P[i, 0] + dos * P[i, 3] + dos * dos * P[i, 4] + dos * dos * dos * P[i, 5]
            sig[count][i] = P[i, 6] / (unc * bux[i])
            rel[count][i] = P[i, 6] / iso[i]
            isoPost[i] = P[i, 0] + dos2 * P[i, 3] + dos2 * dos2 * P[i, 4] + dos2 * dos2 * dos2 * P[i, 5] + P[i, 6]
        # do spectral mixture modelling on iso
        if nbands == 7:
            # loff = 400
            # l' = l - loff
            # lmax = 2000 - loff
            # rhoBurn = a0 + a1(l' - l'^2/(2 lmax)) = a0 + a1 * lk
            # lmax =
            # post = pre * (1-fcc) + fcc * rhoBurn
            # post = pre * (1-fcc) + fcc * a0 + fcc * a1 * lk
            # post - pre = A + B * lk - fcc * pre
            # where
            # A = fcc * a0
            # B = fcc * a1
            y = np.matrix(isoPost - iso)
            K[2] = iso
            M = K * K.transpose()
            MI = M.I
            V = K * y.transpose()
            # spectral parsamsters
            sP = np.array((MI * V).transpose())[0]
            fcc = -sP[2]
            a0 = sP[0] / fcc
            a1 = sP[1] / fcc
            sBurn = a0 + lk * a1
            sFWD = iso * (1 - fcc) + fcc * sBurn
            sPre = iso
            sPost = isoPost
        else:
            fcc = 0
            a0 = 0
            a1 = 0
            sBurn = 0
            sFWD = 0
            sPre = 0
            sPost = 0
        if nbands == 4:
            Test = sig[count][0] < 0 and sig[count][1] < 0 and (
            (sig[count][2] > sig[count][0] and sig[count][2] > sig[count][1]) or (
            sig[count][3] > sig[count][0] and sig[count][3] > sig[count][1]))
        else:
            Test = a0 >= 0 and a1 >= 0 and fcc >= 0 and fcc <= 1 and a0 + a1 <= 1.0 and P[1, 6] < 0 and P[4, 6] < 0

            # put in conditions etc...
        if Test:
            # valid sample
            rmse1 = np.matrix(rmse)
            rmse1 = np.array(np.sqrt(rmse1 * rmse1.transpose() / len(bux)))[0][0]
            thissig = min([sig[count][0], sig[count][1]])
            # print dos,thissig
            # if mindoy == False or thissig < minsig:
            if nbands == 4:
                Test2 = mindoy == False or rmse1 < minrmsei1
            else:
                Test2 = mindoy == False or fcc > maxfcc
            if verbose:
                print(dos, fcc, a0, a1, thissig, rmse1)
            if Test2:
                maxpre = sPre
                maxpost = sPost
                maxfcc = fcc
                maxa0 = a0
                maxa1 = a1
                maxsBurn = sBurn
                maxsFWD = sFWD
                minsig = thissig
                mindoy = dos
                minrmse1 = rmse1
                minrmse = rmse
                minP = P
                minFWD = FWD
                minrefl = refl
                mincount = count
        count += 1
    if mincount != -1:
        if nbands == 4:
            return doy, minrmse, minP, minFWD, minrefl, mindoy, sig[mincount], rel[mincount]
        else:
            if plot:
                import pylab
                x = [mindoy, mindoy]
                y = [0.0, max(np.array([minFWD.flatten(), minrefl.flatten()]).flatten()) + 0.1]
                pylab.plot(x, y)
                colours = ['k', 'b', 'g', 'r', 'c', 'm', 'y']
                for i in range(nbands):
                    norm = minP[i, 0] + doy * minP[i, 3] + doy * doy * minP[i, 4] + doy * doy * doy * minP[i, 5]
                    w = np.where(doy > mindoy)
                    norm[w] += minP[i, 6]
                    pylab.plot(doy, minFWD[i].flatten(), colours[i] + 's', label='model ' + str(bands[i]))
                    pylab.plot(doy, minrefl[i].flatten(), colours[i] + '^', label='obs ' + str(bands[i]))
                    pylab.plot(doy, norm.flatten(), colours[i] + '-', label='norm ' + str(bands[i]))
                legend(loc='outer right')
                pylab.show()
                preday = mindoy
                postday = mindoy + 1
                # print minP[:,0].shape,bands.shape
                prenorm = minP[:, 0] + preday * minP[:, 3] + preday * preday * minP[:,
                                                                               4] + preday * preday * preday * minP[:,
                                                                                                               5]
                postnorm = minP[:, 0] + postday * minP[:, 3] + postday * postday * minP[:,
                                                                                   4] + postday * postday * postday * minP[
                                                                                                                      :,
                                                                                                                      5] + minP[
                                                                                                                           :,
                                                                                                                           6]
                prenorm = np.squeeze(np.array(prenorm))
                postnorm = np.squeeze(np.array(postnorm))

                pylab.plot(bands, prenorm, 'bo', label='pre-burn')
                pylab.plot(bands, postnorm, 'go', label='post-burn')
                pylab.plot(bands, maxsFWD, 'g^', label='fwd model')
                pylab.plot(bands, maxfcc * maxsBurn, 'rD', label='fcc * burn signal')
                pylab.legend(loc=0)
                pylab.show()
            return doy, minrmse, minP, minFWD, minrefl, mindoy, sig[mincount], rel[mincount], maxfcc, maxa0, maxa1
    else:
        if nbands == 4:
            return False, False, False, False, False, False, False, False
        else:
            return False, False, False, False, False, False, False, False, False, False, False


def lisaInvert(vza, sza, raa, refl, doy, dos, LiType='Sparse', RossType='Thick', xi=False, nbar=0.0):
    doy2 = doy * doy
    doy3 = doy2 * doy
    kk = Kernels(vza, sza, raa, doIntegrals=False, RossHS=xi, RossType=RossType, LiType=LiType, normalise=1,
                 RecipFlag=True, MODISSPARSE=True, nbar=nbar)
    K = np.ones([7, len(vza)])
    K[1, :] = kk.Ross[:]
    K[2, :] = kk.Li[:]
    K[3, :] = doy
    K[4, :] = doy2
    K[5, :] = doy3
    w = np.where(doy <= dos)
    K[6, w] = 0.0

    # form matrix
    K = np.matrix(K)
    M = K * K.transpose()
    MI = M.I
    nBands = len(refl[:, 0])
    P = np.matrix(np.zeros([nBands, 7]))
    for i in range(nBands):
        R = np.matrix(refl[i, :])
        V = K * R.transpose()
        P[i, :] = (MI * V).transpose()
    # rmse
    mse = np.zeros(nBands)
    FWD = refl.copy() * 0.0
    for i in range(nBands):
        FWD[i, :] = P[i, :] * K
        d = np.array((FWD[i, :] - refl[i, :])[0])
        mse[i] = (d * d).mean()
    rmse = np.sqrt(mse)
    return rmse, P, FWD, refl, doy, np.sqrt(MI[6, 6])


# pylab.plot(doy,refl[0,:].flatten())
# pylab.plot(doy,FWD[0,:].flatten())
# pylab.show()


def testMe(fixXi=.02617993877991494365, LiType='Sparse', RossType='Thick',
           file='polder.modis.tiles.cover.04.dat.count.top.1.all.h08v05.256.509.dat', ofile=False, type=1, N=1000):
    VZA, SZA, RAA, REFL = readPOLDER(file, type=type)

    rmse, xi, P, FWD, x, y, phi = lutInvertRossHS(VZA, SZA, RAA, REFL, LiType=LiType, RossType=RossType, N=N,
                                                  fixXi=fixXi)
    if (ofile == True):
        aofile = file + '.kernelModelled'
        FILE = open(aofile, 'w')
        FILE.write('# xi = ' + str(xi) + ' rmse = ' + str(
            rmse) + '1:vza 2:sza 3:relphi 4:obs(443) 5:mod(443) 6:obs(565) 7:mod(565) 8:obs(670) 9:mod(670) 10:obs(765) 11:mod(765) 12:obs(865) 13:mod(865)\n')
        for i in range(len(VZA)):
            ostr = str(VZA[i]) + ' ' + str(SZA[i]) + ' ' + str(-RAA[i]) + ' '
            for j in range(5):
                ostr = ostr + str(REFL[j, i]) + ' ' + str(FWD[j, i]) + ' '
            ostr = ostr + '\n'
            FILE.write(ostr)
        FILE.close()
    vza = np.array(range(141)) * 1.0 - 70
    raa = vza * 0.0
    sza = raa - int(SZA.mean())
    sza = raa - 40.0
    kk = Kernels(vza, sza, raa, doIntegrals=False, RossHS=xi, RossType=RossType, LiType=LiType, normalise=1,
                 RecipFlag=True, MODISSPARSE=True)
    K = np.ones([3, len(vza)])
    K[1, :] = kk.Ross[:]
    K[2, :] = kk.Li[:]
    fwd = np.array(P * K)
    if (ofile == True):
        aofile = file + '.kernelPplane'
        FILE = open(aofile, 'w')
        FILE.write('# pplane plot at mean sza of observations: sza = ' + str(sza[0]) + '\n')
        FILE.write('# 1:vza(pplane -ve = hs) 2:r(443) 3:r(565) 4:r(670) 5:r(765) 6:r(865)\n')
        for i in range(len(vza)):
            ostr = str(vza[i]) + ' '
            for j in range(5):
                ostr = ostr + str(fwd[j, i]) + ' '
            ostr = ostr + '\n'
            FILE.write(ostr)
        FILE.close()
    return P, rmse, xi


# w = np.where(SZA > 52)
# S = SZA[w]
# w1 = np.where(S < 55)
# s = S[w1]
# pylab.plot(phi[w][w1],REFL[0,:][w][w1],'o')
# pylab.show()

def invertData(VZA, SZA, RAA, REFL, RossType='Thick', LiType='Dense', RossHS=False, normalise=1, RecipFlag=True,
               MODISSPARSE=True):
    # invert
    kk = Kernels(VZA, SZA, RAA, RossHS=RossHS, MODISSPARSE=MODISSPARSE, RecipFlag=RecipFlag, normalise=normalise,
                 doIntegrals=False, LiType=LiType, RossType=RossType)
    K = np.ones([3, len(VZA)])
    K[1, :] = kk.Ross[:]
    K[2, :] = kk.Li[:]
    # form matrix
    K = np.matrix(K)
    M = K * K.transpose()
    MI = M.I
    nBands = len(REFL[:, 0])
    P = np.matrix(np.zeros([nBands, 3]))
    for i in range(nBands):
        R = np.matrix(REFL[i, :])
        V = K * R.transpose()
        P[i, :] = (MI * V).transpose()
    # rmse
    FWD = P * K
    d = FWD - REFL
    e = 0.0
    for i in range(nBands):
        e = e + d[i] * d[i].transpose()
    rmse = np.sqrt(e[0, 0] / (len(VZA) * nBands))
    phaseAngle = np.arctan2(kk._Kernels__sinphaang, kk._Kernels__cosphaang) * 180. / np.pi
    phaseAngle = phaseAngle[0:len(VZA)]
    return rmse, P, FWD, phaseAngle


# test function
def mimic(doPrint=False, doPlot=False, RossHS=False, RecipFlag=False):
    '''
    A test method to reproduce the results in Wanner et al. 1995.
    There are no parameters and a single option:
            doPrint=True    : print results to stdout (default doPrint=False)

    The method returns:
    VZA,SZA,RAA,RossThick,RossThin,LiSparse,LiDense,Roujean,LiTransit
    where all are numy arrays of dimensions 3 x nSamples
    so:
    VZA[0,:],RossThick[0,:] are the results for sza = 0.0
    VZA[1,:],RossThick[1,:] are the results for sza = 30.0
        VZA[2,:],RossThick[2,:] are the results for sza = 60.0

    '''
    # set up the angles
    r = 89  # do results for +/- r degrees)
    SZAS = np.array([0.0, -30.0, -60.0])  # sza
    vza = np.array(range(2 * r + 1)) * 1.0 - r
    # set up storage info
    RossThick = np.zeros([3, len(vza)])
    RossThin = np.zeros([3, len(vza)])
    LiSparse = np.zeros([3, len(vza)])
    LiDense = np.zeros([3, len(vza)])
    Roujean = np.zeros([3, len(vza)])
    LiTransit = np.zeros([3, len(vza)])
    SZA = np.zeros([3, len(vza)])
    VZA = np.zeros([3, len(vza)])
    RAA = np.zeros([3, len(vza)])
    # fill the angle info
    RossHS = RossHS
    for i in range(len(SZAS)):
        SZA[i, :] = SZAS[i]
        VZA[i, :] = vza[:]
        RAA[i, :] = 0.0
        # do the kernels
        kk = Kernels(VZA[i, :], SZA[i, :], RAA[i, :], RossHS=RossHS, MODISSPARSE=True, RecipFlag=RecipFlag, normalise=1,
                     doIntegrals=False, LiType='Dense', RossType='Thick')
        RossThick[i, :] = kk.Ross[:]
        LiDense[i, :] = kk.Li[:]
        if doPrint == True:
            kk.printKernels(file='RossThickLiDense.' + str(SZAS[i]) + '.dat')
            kk.printer('')
        kk = Kernels(VZA[i, :], SZA[i, :], RAA[i, :], RossHS=RossHS, MODISSPARSE=True, RecipFlag=RecipFlag, normalise=1,
                     doIntegrals=False, LiType='Sparse', RossType='Thin')
        RossThin[i, :] = kk.Ross[:]
        LiSparse[i, :] = kk.Li[:]
        if doPrint == True:
            kk.printKernels(file='RossThinLiSparse.' + str(SZAS[i]) + '.dat')
            kk.printer('')
        kk = Kernels(VZA[i, :], SZA[i, :], RAA[i, :], RossHS=RossHS, MODISSPARSE=True, RecipFlag=RecipFlag, normalise=1,
                     doIntegrals=False, LiType='Roujean', RossType='Thin')
        Roujean[i, :] = kk.Li[:]
        if doPrint == True:
            kk.printKernels(file='RossThinRoujean.' + str(SZAS[i]) + '.dat')
            kk.printer('')
        kk = Kernels(VZA[i, :], SZA[i, :], RAA[i, :], RossHS=RossHS, MODISSPARSE=True, RecipFlag=RecipFlag, normalise=1,
                     doIntegrals=False, LiType='Transit', RossType='Thin')
        LiTransit[i, :] = kk.Li[:]
        if doPrint == True:
            kk.printKernels(file='RossThinLiTransit.' + str(SZAS[i]) + '.dat')
            kk.printer('')
    if (doPlot == True):
        x = [-90.0, 90.0]
        y = [0.0, 0.0]
        for i in range(len(SZAS)):
            sza = SZAS[i]
            pylab.clf()
            pylab.xlabel('View Zenith Angle')
            pylab.ylabel('Kernel Value')
            pylab.title('Solar Zenith Angle ' + str(sza) + ' Degrees')
            pylab.plot(x, y)
            pylab.plot(kk.vzaDegrees, RossThick[i, :], label='RThick')
            pylab.plot(kk.vzaDegrees, RossThin[i, :], label='RThin')
            pylab.plot(kk.vzaDegrees, LiSparse[i, :], label='LiSp')
            pylab.plot(kk.vzaDegrees, LiDense[i, :], label='LiDen')
            pylab.plot(kk.vzaDegrees, Roujean[i, :], label='Roujean')
            pylab.plot(kk.vzaDegrees, LiTransit[i, :], label='LiTrans')
            pylab.axis([-90.0, 90.0, -3.0, 3.0])
            pylab.legend(loc=0)
            pylab.show()

    return VZA, SZA, RAA, RossThick, RossThin, LiSparse, LiDense, Roujean, LiTransit


class RelativeBRDFNorm(object):

    @staticmethod
    def sza(central_latitude):

        """
        Returns the mean sun zenith angle (SZA) as a function of the central latitude

        Args:
            central_latitude (float)

        Returns:
            ``float``
        """

        return 31.0076 + \
               -0.1272*central_latitude + \
               0.01187*(central_latitude**2) + \
               2.40e-05*(central_latitude**3) + \
               -9.48e-07*(central_latitude**4) + \
               -1.95e-09*(central_latitude**5) + \
               6.15e-11*(central_latitude**6)


class RossLiKernels(object):

    def get_kernels(self, central_latitude, solar_za, solar_az, sensor_za, sensor_az):

        # if isinstance(central_latitude, np.ndarray) or isinstance(central_latitude, xr.DataArray):
        #     delayed = True
        # else:
        #     delayed = False

        # Get the geometric scattering kernel.
        #
        # HLS uses a constant (per location) sun zenith angle (`solar_za`).
        # HLS uses 0 for sun azimuth angle (`solar_az`).
        # theta_v, theta_s, delta_gamma
        kl = Kernels(0.0, self.sza(central_latitude), 0.0, delayed=False, doIntegrals=False)

        # Copy the geometric scattering
        #   coefficients so they are
        #   not overwritten by the
        #   volume scattering coefficients.
        self.geo_norm = copy(kl.Li)
        self.vol_norm = copy(kl.Ross)

        # Get the volume scattering kernel.
        #
        # theta_v=0 for nadir view zenith angle, theta_s, delta_gamma
        kl = Kernels(sensor_za.data,
                     solar_za.data,
                     relative_azimuth(solar_az, sensor_az).data,
                     delayed=True,
                     doIntegrals=False)

        self.geo_sensor = kl.Li
        self.vol_sensor = kl.Ross


class BRDF(RelativeBRDFNorm, RossLiKernels):

    """
    A class for Nadir BRDF (NBAR) normalization
    """

    def __init__(self):

        self.geo_norm = None
        self.vol_norm = None
        self.geo_sensor = None
        self.vol_sensor = None

        # Setup the c-factor equation.
        #
        # `SA` = the sensor array
        self.c_equation = 'SA * ((fiso + fvol*vol_norm + fgeo*geo_norm) / (fiso + fvol*vol_sensor + fgeo*geo_sensor))'

        # A dictionary of BRDF kernel coefficients
        self.coeff_dict = dict(blue=dict(fiso=0.0774,
                                         fgeo=0.0079,
                                         fvol=0.0372),
                               green=dict(fiso=0.1306,
                                          fgeo=0.0178,
                                          fvol=0.058),
                               red=dict(fiso=0.169,
                                        fgeo=0.0227,
                                        fvol=0.0574),
                               nir=dict(fiso=0.3093,
                                        fgeo=0.033,
                                        fvol=0.1535),
                               swir1=dict(fiso=0.343,
                                          fgeo=0.0453,
                                          fvol=0.1154),
                               swir2=dict(fiso=0.2658,
                                          fgeo=0.0387,
                                          fvol=0.0639),
                               pan=dict(fiso=0.12567,
                                        fgeo=0.01613,
                                        fvol=0.0509))

    def get_coeffs(self, sensor_band):
        return self.coeff_dict[sensor_band]

    def norm_brdf(self,
                  data,
                  solar_za,
                  solar_az,
                  sensor_za,
                  sensor_az,
                  central_latitude=None,
                  sensor=None,
                  wavelengths=None,
                  nodata=0,
                  mask=None,
                  scale_factor=1.0,
                  out_range=None,
                  scale_angles=True):

        """
        Applies Nadir Bidirectional Reflectance Distribution Function (BRDF) normalization
        using the global c-factor method (see Roy et al. (2016))

        Args:
            data (2d or 3d array): The data to normalize.
            solar_za (2d array): The solar zenith angles (degrees).
            solar_az (2d array): The solar azimuth angles (degrees).
            sensor_za (2d array): The sensor azimuth angles (degrees).
            sensor_az (2d array): The sensor azimuth angles (degrees).
            central_latitude (Optional[float or 2d array]): The central latitude.
            sensor (Optional[str]): The satellite sensor.
            wavelengths (str list): Choices are ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'].
            nodata (Optional[int or float]): A 'no data' value to fill NAs with.
            mask (Optional[DataArray]): A data mask, where clear values are 0.
            scale_factor (Optional[float]): A scale factor to apply to the input data.
            out_range (Optional[float]): The out data range. If not given, the output data are return in a 0-1 range.
            scale_angles (Optional[bool]): Whether to scale the pixel angle arrays.

        References:

            See :cite:`roy_etal_2016` for the c-factor method.

            For further background on BRDF:

                :cite:`li_strahler_1992`

                :cite:`roujean_etal_1992`

                :cite:`schaaf_etal_2002`

        Examples:
            >>> import geowombat as gw
            >>>
            >>> # Example where pixel angles are stored in separate GeoTiff files
            >>> with gw.config.update(sensor='l7', scale_factor=0.0001, nodata=0):
            >>>
            >>>     with gw.open('solarz.tif') as solarz, gw.open('solara.tif') as solara, gw.open('sensorz.tif') as sensorz, gw.open('sensora.tif') as sensora:
            >>>
            >>>         with gw.open('landsat.tif') as ds:
            >>>             ds_brdf = gw.norm_brdf(ds, solarz, solara, sensorz, sensora)

        Returns:
            ``xarray.DataArray``
        """

        if not wavelengths:

            if sensor:
                wavelengths = list(data.gw.wavelengths[sensor]._fields)
            else:

                if not data.gw.sensor:
                    logger.exception('  The sensor must be supplied.')

                wavelengths = list(data.gw.wavelengths[data.gw.sensor]._fields)

        if not wavelengths:
            logger.exception('  The sensor or wavelength must be supplied.')

        if not isinstance(nodata, int) and not isinstance(nodata, float):
            nodata = data.gw.nodata

        # ne.set_num_threads(num_threads)

        if not isinstance(central_latitude, np.ndarray):
            if not isinstance(central_latitude, xr.DataArray):
                if not isinstance(central_latitude, float):

                    central_latitude = \
                    project_coords(np.array([data.x.values[int(data.x.shape[0] / 2)]], dtype='float64'),
                                   np.array([data.y.values[int(data.y.shape[0] / 2)]], dtype='float64'),
                                   data.crs,
                                   {'init': 'epsg:4326'})[1][0]

                    # TODO: rasterio.warp.reproject does not seem to be working
                    #
                    # Create the 2d latitudes
                    # central_latitude = project_coords(data.x.values,
                    #                                   data.y.values,
                    #                                   data.crs,
                    #                                   {'init': 'epsg:4326'},
                    #                                   num_threads=1,
                    #                                   warp_mem_limit=512)

        attrs = data.attrs

        if not nodata:
            nodata = data.gw.nodata

        if scale_factor == 1.0:
            scale_factor = data.gw.scale_factor

        # Scale the reflectance data
        if scale_factor != 1:
            data = data * scale_factor

        if scale_angles:

            # Scale the angle data to degrees

            solar_za = solar_za * 0.01
            solar_za.coords['band'] = [1]

            solar_az = solar_az * 0.01
            solar_az.coords['band'] = [1]

            sensor_za = sensor_za * 0.01
            sensor_za.coords['band'] = [1]

            sensor_az = sensor_az * 0.01
            sensor_az.coords['band'] = [1]

        # Get the Ross and Li coefficients
        self.get_kernels(central_latitude,
                         solar_za,
                         solar_az,
                         sensor_za,
                         sensor_az)

        # if len(wavelengths) == 1:
        #
        #     # Get the band iso, geo, and vol coefficients.
        #     coeffs = self.get_coeffs(wavelengths[0])
        #
        #     # Apply the adjustment.
        #     data = dask.delayed(ne.evaluate)(self.c_equation,
        #                                      local_dict=dict(fiso=coeffs['fiso'],
        #                                                      fgeo=coeffs['fgeo'],
        #                                                      fvol=coeffs['fvol'],
        #                                                      SA=data,
        #                                                      geo_norm=self.geo_norm,
        #                                                      geo_sensor=self.geo_sensor,
        #                                                      vol_norm=self.vol_norm,
        #                                                      vol_sensor=self.vol_sensor))
        #
        # else:

        results = list()

        for si, wavelength in enumerate(wavelengths):

            # Get the band iso, geo,
            #   and vol coefficients.
            coeffs = self.get_coeffs(wavelength)

            # c-factor
            c_factor = ((coeffs['fiso'] +
                         coeffs['fvol']*self.vol_norm +
                         coeffs['fgeo']*self.geo_norm) /
                        (coeffs['fiso'] +
                         coeffs['fvol']*self.vol_sensor +
                         coeffs['fgeo']*self.geo_sensor))

            p_norm = data.sel(band=wavelength).data * c_factor

            # Apply the adjustment to the current layer.
            results.append(p_norm)

        data = xr.DataArray(data=da.concatenate(results),
                            dims=('band', 'y', 'x'),
                            coords={'band': data.band.values,
                                    'y': data.y,
                                    'x': data.x},
                            attrs=data.attrs)

        if isinstance(out_range, float) or isinstance(out_range, int):

            if out_range <= 1:
                dtype = 'float64'
            elif 1 < out_range <= 255:
                dtype = 'uint8'
            else:
                dtype = 'uint16'

            drange = (0, out_range)

            data = (data * out_range).clip(0, out_range)

        else:

            drange = (0, 1)
            dtype = 'float64'

        # Mask data
        if isinstance(mask, xr.DataArray):
            data = xr.where((mask.sel(band=1) == 0) & (solar_za.sel(band=1) != -32768), data, nodata)
        else:
            data = xr.where(solar_za.sel(band=1) != -32768, data, nodata)

        data = data.transpose('band', 'y', 'x').astype(dtype)

        attrs['sensor'] = sensor
        attrs['calibration'] = 'Nadir BRDF-adjusted (NBAR) surface reflectance'
        attrs['nodata'] = nodata
        attrs['drange'] = drange

        data.attrs = attrs

        return data
