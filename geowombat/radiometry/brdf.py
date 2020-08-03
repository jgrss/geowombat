from copy import copy
from collections import namedtuple
import logging

from .angles import relative_azimuth
from ..core.util import project_coords

import numpy as np
import xarray as xr
import dask
import dask.array as da


logger = logging.getLogger(__name__)


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


class Special(object):

    @staticmethod
    def get_distance(tan1, tan2, cos3):

        """
        Gets distance component of Li kernels
        """

        temp = tan1 * tan1 + tan2 * tan2 - 2.0 * tan1 * tan2 * cos3

        temp = da.where(temp < 0, 0, temp)

        return da.sqrt(temp)

    @staticmethod
    def get_overlap(cos1, cos2, tan1, tan2, sin3, distance, hb, m_pi):

        """
        Applies HB ratio transformation
        """

        OverlapInfo = namedtuple('OverlapInfo', 'tvar sint overlap temp')

        temp = (1.0 / cos1) + (1.0 / cos2)

        cost = da.clip(hb * da.sqrt(distance * distance + tan1 * tan1 * tan2 * tan2 * sin3 * sin3) / temp, -1, 1)

        tvar = da.arccos(cost)
        sint = da.sin(tvar)
        overlap = 1.0 / m_pi * (tvar - sint * cost) * temp

        overlap = da.where(overlap < 0, 0, overlap)

        return OverlapInfo(tvar=tvar, sint=sint, overlap=overlap, temp=temp)


class Angles(object):

    @staticmethod
    def get_phaang(cos_vza, cos_sza, sin_vza, sin_sza, cos_raa):

        """
        Gets the phase angle
        """

        cos_phase_angle = da.clip(cos_vza * cos_sza + sin_vza * sin_sza * cos_raa, -1, 1)
        phase_angle = da.arccos(cos_phase_angle)
        sin_phase_angle = da.sin(phase_angle)

        return cos_phase_angle, phase_angle, sin_phase_angle

    @staticmethod
    def get_pangles(tan1, br, nearly_zero):

        """
        Get the prime angles
        """

        tanp = br * tan1

        tanp = da.where(tanp < 0, 0, tanp)

        angp = da.arctan(tanp)
        sinp = da.sin(angp)
        cosp = da.cos(angp)

        # have to make sure c is not 0
        cosp = da.where(cosp == 0, nearly_zero, cosp)

        return cosp, sinp, tanp

    @staticmethod
    def get_angle_info(vza, sza, raa, m_pi):

        """
        Gets the angle information
        """

        AngleInfo = namedtuple('AngleInfo', 'vza sza raa vza_rad sza_rad raa_rad')

        # View zenith angle
        vza_rad = da.deg2rad(vza)

        # Solar zenith angle
        sza_rad = da.deg2rad(sza)

        # Relative azimuth angle
        raa_rad = da.deg2rad(raa)

        vza_abs = da.fabs(vza_rad)
        sza_abs = da.fabs(sza_rad)

        raa_abs = da.where((vza_rad < 0) | (sza_rad < 0), m_pi, raa_rad)

        return AngleInfo(vza=vza,
                         sza=sza,
                         raa=raa,
                         vza_rad=vza_abs,
                         sza_rad=sza_abs,
                         raa_rad=raa_abs)


class LiKernel(Special, Angles):

    def get_li(self, kernel_type, recip_flag):

        # relative azimuth angle
        # ensure it is in a [0,2] pi range
        phi = da.fabs((self.angle_info.raa_rad % (2.0 * self.global_args.m_pi)))

        cos_phi = da.cos(phi)
        sin_phi = da.sin(phi)

        tanti = da.tan(self.angle_info.sza_rad)
        tantv = da.tan(self.angle_info.vza_rad)

        cos1, sin1, tan1 = self.get_pangles(tantv, self.global_args.br, self.global_args.nearly_zero)
        cos2, sin2, tan2 = self.get_pangles(tanti, self.global_args.br, self.global_args.nearly_zero)

        # sets cos & sin phase angle terms
        cos_phaang, phaang, sin_phaang = self.get_phaang(cos1, cos2, sin1, sin2, cos_phi)
        distance = self.get_distance(tan1, tan2, cos_phi)
        overlap_info = self.get_overlap(cos1, cos2, tan1, tan2, sin_phi, distance, self.global_args.hb, self.global_args.m_pi)

        if kernel_type.lower() == 'sparse':

            if recip_flag:
                li = overlap_info.overlap - overlap_info.temp + 0.5 * (1.0 + cos_phaang) / cos1 / cos2
            else:
                li = overlap_info.overlap - overlap_info.temp + 0.5 * (1.0 + cos_phaang) / cos1

        else:

            if kernel_type.lower() == 'dense':

                if recip_flag:
                    li = (1.0 + cos_phaang) / (cos1 * cos2 * (overlap_info.temp - overlap_info.overlap)) - 2.0
                else:
                    li = (1.0 + cos_phaang) / (cos1 * (overlap_info.temp - overlap_info.overlap)) - 2.0

        return li


class RossKernel(Special, Angles):

    def ross_part(self, angle_info, global_args):

        """
        Calculates the main part of Ross kernel
        """

        RossKernelOutputs = namedtuple('RossKernelOutputs',
                                       'cos_vza cos_sza sin_vza sin_sza cos_raa ross_element cos_phase_angle phase_angle sin_phase_angle ross')

        cos_vza = da.cos(angle_info.vza_rad)
        cos_sza = da.cos(angle_info.sza_rad)
        sin_vza = da.sin(angle_info.vza_rad)
        sin_sza = da.sin(angle_info.sza_rad)
        cos_raa = da.cos(angle_info.raa_rad)

        cos_phase_angle, phase_angle, sin_phase_angle = self.get_phaang(cos_vza, cos_sza, sin_vza, sin_sza, cos_raa)

        ross_element = (global_args.m_p / 2.0 - phase_angle) * cos_phase_angle + sin_phase_angle

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

    def ross_thin(self, angle_info, global_args):

        RossThinOutputs = namedtuple('RossThinOutputs', 'ross phase_angle')

        ross_kernel_outputs = self.ross_part(angle_info, global_args)

        ross_ = ross_kernel_outputs.ross_element / (ross_kernel_outputs.cos_vza * ross_kernel_outputs.cos_sza)

        return RossThinOutputs(ross=ross_, phase_angle=ross_kernel_outputs.phase_angle)

    def ross_thick(self, angle_info, global_args):

        RossThickOutputs = namedtuple('RossThickOutputs', 'ross phase_angle')

        ross_kernel_outputs = self.ross_part(angle_info, global_args)

        ross_ = ross_kernel_outputs.ross_element / (ross_kernel_outputs.cos_vza + ross_kernel_outputs.cos_sza)

        return RossThickOutputs(ross=ross_, phase_angle=ross_kernel_outputs.phase_angle)

    def get_ross(self, kernel_type):

        if kernel_type.lower() == 'thin':
            ross_kernel_outputs = self.ross_thin(self.angle_info, self.global_args)
        else:
            ross_kernel_outputs = self.ross_thick(self.angle_info, self.global_args)

        if self.global_args.hs:
            ross = ross_kernel_outputs.ross * (1.0 + 1.0 / (1.0 + ross_kernel_outputs.phase_angle / 0.25))
        else:
            ross = ross_kernel_outputs.ross - self.global_args.m_pi / 4.0

        return ross


class _Kernels(LiKernel, RossKernel):

    def __init__(self,
                 vza,
                 sza,
                 raa,
                 li_type='sparse',
                 ross_type='thick',
                 recip_flag=True,
                 br=1.0,
                 hb=2.0,
                 hs=True):

        GlobalArgs = namedtuple('GlobalArgs', 'br m_pi hb hs nearly_zero')

        self.global_args = GlobalArgs(br=br, m_pi=np.pi, hb=hb, hs=hs, nearly_zero=1e-20)
        self.angle_info = self.get_angle_info(vza, sza, raa, self.global_args.m_pi)

        self.li_k = self.get_li(li_type, recip_flag)
        self.ross_k = self.get_ross(ross_type)


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

            self.ross_k = ross
            self.li_k = li

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


class RossLiKernels(object):

    def _get_kernels(self, central_latitude, solar_za, solar_az, sensor_za, sensor_az):

        # if isinstance(central_latitude, np.ndarray) or isinstance(central_latitude, xr.DataArray):
        #     delayed = True
        # else:
        #     delayed = False

        # Get the geometric scattering kernel.
        #
        # HLS uses a constant (per location) sun zenith angle (`solar_za`).
        # HLS uses 0 for sun azimuth angle (`solar_az`).
        # theta_v, theta_s, delta_gamma
        kl = Kernels(0.0, self.get_mean_sza(central_latitude), 0.0, delayed=False, doIntegrals=False)

        # Copy the geometric scattering
        #   coefficients so they are
        #   not overwritten by the
        #   volume scattering coefficients.
        self.geo_norm = copy(kl.li_k)
        self.vol_norm = copy(kl.ross_k)

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


class BRDF(RossLiKernels):

    """
    A class for Bidirectional Reflectance Distribution Function (BRDF) normalization
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

    def _get_coeffs(self, sensor_band):
        return self.coeff_dict[sensor_band]

    @staticmethod
    def get_mean_sza(central_latitude):

        """
        Returns the mean solar zenith angle (SZA) as a function of the central latitude

        Args:
            central_latitude (float): The central latitude.

        Reference:

            See :cite:`zhang_etal_2016`

        Returns:
            ``float``
        """

        return 31.0076 + \
               -0.1272 * central_latitude + \
               0.01187 * (central_latitude ** 2) + \
               2.40e-05 * (central_latitude ** 3) + \
               -9.48e-07 * (central_latitude ** 4) + \
               -1.95e-09 * (central_latitude ** 5) + \
               6.15e-11 * (central_latitude ** 6)

    def norm_brdf(self,
                  data,
                  solar_za,
                  solar_az,
                  sensor_za,
                  sensor_az,
                  central_latitude=None,
                  sensor=None,
                  wavelengths=None,
                  src_nodata=-32768,
                  dst_nodata=-32768,
                  mask=None,
                  scale_factor=1.0,
                  out_range=None,
                  scale_angles=True):

        r"""
        Applies Nadir Bidirectional Reflectance Distribution Function (BRDF) normalization
        using the global c-factor method

        Args:
            data (2d or 3d DataArray): The data to normalize.
            solar_za (2d DataArray): The solar zenith angles (degrees).
            solar_az (2d DataArray): The solar azimuth angles (degrees).
            sensor_za (2d DataArray): The sensor azimuth angles (degrees).
            sensor_az (2d DataArray): The sensor azimuth angles (degrees).
            central_latitude (Optional[float or 2d DataArray]): The central latitude.
            sensor (Optional[str]): The satellite sensor.
            wavelengths (str list): The wavelength(s) to normalize.
            src_nodata (Optional[int or float]): The input 'no data' value.
            dst_nodata (Optional[int or float]): The output 'no data' value.
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

        Returns:
            ``xarray.DataArray``

        Examples:
            >>> import geowombat as gw
            >>> from geowombat.radiometry import BRDF
            >>>
            >>> brdf = BRDF()
            >>>
            >>> # Example where pixel angles are stored in separate GeoTiff files
            >>> with gw.config.update(sensor='l7', scale_factor=0.0001):
            >>>
            >>>     with gw.open('solarz.tif') as solarz,
            >>>         gw.open('solara.tif') as solara,
            >>>             gw.open('sensorz.tif') as sensorz,
            >>>                 gw.open('sensora.tif') as sensora:
            >>>
            >>>         with gw.open('landsat.tif') as src:
            >>>             src_norm = brdf.norm_brdf(src, solarz, solara, sensorz, sensora)
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

        if not isinstance(dst_nodata, int) and not isinstance(dst_nodata, float):
            dst_nodata = data.gw.nodata

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

        attrs = data.attrs.copy()

        # Set 'no data' as nans
        data = data.where(data != src_nodata)

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
        self._get_kernels(central_latitude,
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
            coeffs = self._get_coeffs(wavelength)

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
                            attrs=data.attrs).fillna(src_nodata)

        if isinstance(out_range, float) or isinstance(out_range, int):

            if out_range <= 1:
                dtype = 'float64'
            elif 1 < out_range <= 255:
                dtype = 'uint8'
            else:
                dtype = 'uint16'

            drange = (0, out_range)

            data = xr.where(data == src_nodata, src_nodata, (data * out_range).clip(0, out_range))

        else:

            drange = (0, 1)
            dtype = 'float64'

        # Mask data
        if isinstance(mask, xr.DataArray):
            data = xr.where((mask.sel(band=1) == 1) | (solar_za.sel(band=1) == -32768*0.01) | (data == src_nodata), dst_nodata, data)
        else:
            data = xr.where((solar_za.sel(band=1) == -32768*0.01) | (data == src_nodata), dst_nodata, data)

        data = data.transpose('band', 'y', 'x').astype(dtype)

        attrs['sensor'] = sensor
        attrs['calibration'] = 'BRDF-adjusted surface reflectance'
        attrs['nodata'] = dst_nodata
        attrs['drange'] = drange

        data.attrs = attrs

        return data
