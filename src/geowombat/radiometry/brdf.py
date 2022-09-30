from copy import copy
from collections import namedtuple
import logging

from ..handler import add_handler
from ..core.util import project_coords
from .angles import relative_azimuth

import numpy as np
import xarray as xr
import dask.array as da


logger = logging.getLogger(__name__)
logger = add_handler(logger)


class Special(object):
    @staticmethod
    def get_distance(tan1, tan2, cos3):

        """Gets distance component of Li kernels."""

        temp = tan1 * tan1 + tan2 * tan2 - 2.0 * tan1 * tan2 * cos3

        return da.sqrt(da.maximum(temp, 0))

    @staticmethod
    def get_overlap(cos1, cos2, tan1, tan2, sin3, distance, hb, m_pi):

        """Applies the HB ratio transformation."""

        OverlapInfo = namedtuple('OverlapInfo', 'tvar sint overlap temp')

        temp = (1.0 / cos1) + (1.0 / cos2)

        cost = da.clip(
            hb
            * da.sqrt(distance * distance + tan1 * tan1 * tan2 * tan2 * sin3 * sin3)
            / temp,
            -1,
            1,
        )

        tvar = da.arccos(cost)
        sint = da.sin(tvar)

        overlap = 1.0 / m_pi * (tvar - sint * cost) * temp
        overlap = da.maximum(overlap, 0)

        return OverlapInfo(tvar=tvar, sint=sint, overlap=overlap, temp=temp)


class Angles(object):
    @staticmethod
    def get_phaang(cos_vza, cos_sza, sin_vza, sin_sza, cos_raa):

        """Gets the phase angle."""

        cos_phase_angle = da.clip(
            cos_vza * cos_sza + sin_vza * sin_sza * cos_raa, -1, 1
        )
        phase_angle = da.arccos(cos_phase_angle)
        sin_phase_angle = da.sin(phase_angle)

        return cos_phase_angle, phase_angle, sin_phase_angle

    @staticmethod
    def get_pangles(tan1, br, nearly_zero):

        """Get the prime angles."""

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

        """Gets the angle information."""

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

        return AngleInfo(
            vza=vza, sza=sza, raa=raa, vza_rad=vza_abs, sza_rad=sza_abs, raa_rad=raa_abs
        )


class LiKernel(Special, Angles):
    def get_li(self, kernel_type, li_recip):

        # relative azimuth angle
        # ensure it is in a [0,2] pi range
        phi = da.fabs((self.angle_info.raa_rad % (2.0 * self.global_args.m_pi)))

        cos_phi = da.cos(phi)
        sin_phi = da.sin(phi)

        tanti = da.tan(self.angle_info.sza_rad)
        tantv = da.tan(self.angle_info.vza_rad)

        cos1, sin1, tan1 = self.get_pangles(
            tantv, self.global_args.br, self.global_args.nearly_zero
        )
        cos2, sin2, tan2 = self.get_pangles(
            tanti, self.global_args.br, self.global_args.nearly_zero
        )

        # sets cos & sin phase angle terms
        cos_phaang, phaang, sin_phaang = self.get_phaang(
            cos1, cos2, sin1, sin2, cos_phi
        )
        distance = self.get_distance(tan1, tan2, cos_phi)
        overlap_info = self.get_overlap(
            cos1,
            cos2,
            tan1,
            tan2,
            sin_phi,
            distance,
            self.global_args.hb,
            self.global_args.m_pi,
        )

        if kernel_type.lower() == 'sparse':

            if li_recip:
                li = (
                    overlap_info.overlap
                    - overlap_info.temp
                    + 0.5 * (1.0 + cos_phaang) / cos1 / cos2
                )
            else:
                li = (
                    overlap_info.overlap
                    - overlap_info.temp
                    + 0.5 * (1.0 + cos_phaang) / cos1
                )

        else:

            if kernel_type.lower() == 'dense':

                if li_recip:
                    li = (1.0 + cos_phaang) / (
                        cos1 * cos2 * (overlap_info.temp - overlap_info.overlap)
                    ) - 2.0
                else:
                    li = (1.0 + cos_phaang) / (
                        cos1 * (overlap_info.temp - overlap_info.overlap)
                    ) - 2.0

        return li


class RossKernel(Special, Angles):
    def ross_part(self, angle_info, global_args):

        """Calculates the main part of Ross kernel."""

        RossKernelOutputs = namedtuple(
            'RossKernelOutputs',
            'cos_vza cos_sza sin_vza sin_sza cos_raa ross_element cos_phase_angle phase_angle sin_phase_angle ross',
        )

        cos_vza = da.cos(angle_info.vza_rad)
        cos_sza = da.cos(angle_info.sza_rad)
        sin_vza = da.sin(angle_info.vza_rad)
        sin_sza = da.sin(angle_info.sza_rad)
        cos_raa = da.cos(angle_info.raa_rad)

        cos_phase_angle, phase_angle, sin_phase_angle = self.get_phaang(
            cos_vza, cos_sza, sin_vza, sin_sza, cos_raa
        )

        ross_element = (
            global_args.m_pi / 2.0 - phase_angle
        ) * cos_phase_angle + sin_phase_angle

        return RossKernelOutputs(
            cos_vza=cos_vza,
            cos_sza=cos_sza,
            sin_vza=sin_vza,
            sin_sza=sin_sza,
            cos_raa=cos_raa,
            ross_element=ross_element,
            cos_phase_angle=cos_phase_angle,
            phase_angle=phase_angle,
            sin_phase_angle=sin_phase_angle,
            ross=None,
        )

    @staticmethod
    def ross_thin(ross_outputs):

        RossThinOutputs = namedtuple('RossThinOutputs', 'ross phase_angle')

        ross_ = ross_outputs.ross_element / (
            ross_outputs.cos_vza * ross_outputs.cos_sza
        )

        return RossThinOutputs(ross=ross_, phase_angle=ross_outputs.phase_angle)

    @staticmethod
    def ross_thick(ross_outputs):

        RossThickOutputs = namedtuple('RossThickOutputs', 'ross phase_angle')

        ross_ = ross_outputs.ross_element / (
            ross_outputs.cos_vza + ross_outputs.cos_sza
        )

        return RossThickOutputs(ross=ross_, phase_angle=ross_outputs.phase_angle)

    def get_ross(self, kernel_type):

        ross_outputs = self.ross_part(self.angle_info, self.global_args)

        if kernel_type.lower() == 'thin':
            ross_kernel_outputs = self.ross_thin(ross_outputs)
        else:
            ross_kernel_outputs = self.ross_thick(ross_outputs)

        if self.global_args.hs:
            ross = ross_kernel_outputs.ross * (
                1.0 + 1.0 / (1.0 + ross_kernel_outputs.phase_angle / 0.25)
            )
        else:
            ross = ross_kernel_outputs.ross - self.global_args.m_pi / 4.0

        return ross


class BRDFKernels(LiKernel, RossKernel):

    """A class for the Li and Ross BRDF kernels.

    Args:
        vza (dask.array): The view zenith angle.
        sza (dask.array): The solar zenith angle.
        raa (dask.array): The relative azimuth angle.
        li_type (Optional[str]): The Li kernel type. Choices are ['sparse', 'dense'].
        ross_type (Optional[str]): The Ross kernel type. Choices are ['thin', 'thick'].
        br (Optional[float]): The BR ratio.
        hb (Optional[float]): The HB ratio.
    """

    def __init__(
        self,
        vza,
        sza,
        raa,
        li_type='sparse',
        ross_type='thick',
        li_recip=True,
        br=1.0,
        hb=2.0,
        hs=False,
    ):

        GlobalArgs = namedtuple('GlobalArgs', 'br m_pi hb hs nearly_zero')

        self.global_args = GlobalArgs(
            br=br, m_pi=np.pi, hb=hb, hs=hs, nearly_zero=1e-20
        )
        self.angle_info = self.get_angle_info(vza, sza, raa, self.global_args.m_pi)

        self.li_k = self.get_li(li_type, li_recip)
        self.ross_k = self.get_ross(ross_type)


class GeoVolKernels(object):
    @staticmethod
    def get_mean_sza(central_latitude):

        """Returns the mean solar zenith angle (SZA) as a function of the
        central latitude.

        Args:
            central_latitude (float): The central latitude.

        Reference:

            See :cite:`zhang_etal_2016`

        Returns:
            ``float``
        """

        return (
            31.0076
            + -0.1272 * central_latitude
            + 0.01187 * (central_latitude**2)
            + 2.40e-05 * (central_latitude**3)
            + -9.48e-07 * (central_latitude**4)
            + -1.95e-09 * (central_latitude**5)
            + 6.15e-11 * (central_latitude**6)
        )

    def get_kernels(self, central_latitude, solar_za, solar_az, sensor_za, sensor_az):

        # Get the geometric scattering kernel.
        #
        # HLS uses a constant (per location) sun zenith angle (`solar_za`).
        # HLS uses 0 for sun azimuth angle (`solar_az`).
        # theta_v, theta_s, delta_gamma
        kl = BRDFKernels(0.0, self.get_mean_sza(central_latitude), 0.0)

        # Copy the geometric scattering
        #   coefficients so they are
        #   not overwritten by the
        #   volume scattering coefficients.
        self.geo_norm = copy(kl.li_k)
        self.vol_norm = copy(kl.ross_k)

        # Get the volume scattering kernel.
        #
        # theta_v=0 for nadir view zenith angle, theta_s, delta_gamma
        kl = BRDFKernels(
            sensor_za.data, solar_za.data, relative_azimuth(solar_az, sensor_az).data
        )

        self.geo_sensor = kl.li_k
        self.vol_sensor = kl.ross_k


class BRDF(GeoVolKernels):

    """A class for Bidirectional Reflectance Distribution Function (BRDF)
    normalization."""

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
        self.coeff_dict = dict(
            blue=dict(fiso=0.0774, fgeo=0.0079, fvol=0.0372),
            green=dict(fiso=0.1306, fgeo=0.0178, fvol=0.058),
            red=dict(fiso=0.169, fgeo=0.0227, fvol=0.0574),
            nir=dict(fiso=0.3093, fgeo=0.033, fvol=0.1535),
            swir1=dict(fiso=0.343, fgeo=0.0453, fvol=0.1154),
            swir2=dict(fiso=0.2658, fgeo=0.0387, fvol=0.0639),
            pan=dict(fiso=0.12567, fgeo=0.01613, fvol=0.0509),
        )

    def _get_coeffs(self, sensor_band):
        return self.coeff_dict[sensor_band]

    def norm_brdf(
        self,
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
        scale_angles=True,
    ):
        r"""Applies Nadir Bidirectional Reflectance Distribution Function (BRDF) normalization
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

        if not isinstance(dst_nodata, (int, float)):
            dst_nodata = data.gw.nodataval

        if not isinstance(central_latitude, np.ndarray):
            if not isinstance(central_latitude, xr.DataArray):
                if not isinstance(central_latitude, float):
                    central_latitude = project_coords(
                        np.array(
                            [data.x.values[int(data.x.shape[0] / 2)]], dtype='float64'
                        ),
                        np.array(
                            [data.y.values[int(data.y.shape[0] / 2)]], dtype='float64'
                        ),
                        data.crs,
                        {'init': 'epsg:4326'},
                    )[1][0]

        attrs = data.attrs.copy()
        # Set 'no data' as nans and scale the reflectance data
        data = data.gw.set_nodata(
            src_nodata,
            np.nan,
            out_range=(0, 1),
            dtype='float64',
            scale_factor=scale_factor,
            offset=0,
        )

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
        self.get_kernels(central_latitude, solar_za, solar_az, sensor_za, sensor_az)

        results = []
        for si, wavelength in enumerate(wavelengths):
            # Get the band iso, geo, and vol coefficients.
            coeffs = self._get_coeffs(wavelength)
            # c-factor
            c_factor = (
                coeffs['fiso']
                + coeffs['fvol'] * self.vol_norm
                + coeffs['fgeo'] * self.geo_norm
            ) / (
                coeffs['fiso']
                + coeffs['fvol'] * self.vol_sensor
                + coeffs['fgeo'] * self.geo_sensor
            )

            p_norm = data.sel(band=wavelength).data * c_factor
            # Apply the adjustment to the current layer.
            results.append(p_norm)

        data = xr.DataArray(
            data=da.concatenate(results),
            dims=('band', 'y', 'x'),
            coords={'band': data.band.values, 'y': data.y, 'x': data.x},
            attrs=data.attrs,
        ).fillna(src_nodata)

        if isinstance(out_range, (int, float, tuple)):
            if isinstance(out_range, (int, float)):
                range_max = out_range
            else:
                range_max = out_range[1]

            if range_max <= 1:
                dtype = 'float64'
            elif 1 < range_max <= 255:
                dtype = 'uint8'
            else:
                dtype = 'uint16'

            drange = (0, range_max)
            data = xr.where(
                data == src_nodata, src_nodata, (data * range_max).clip(0, range_max)
            )

        else:
            drange = (0, 1)
            dtype = 'float64'

        # Mask data
        if isinstance(mask, xr.DataArray):
            data = xr.where(
                (mask.sel(band=1) == 1)
                | (solar_za.sel(band=1) == -32768 * 0.01)
                | (data == src_nodata),
                dst_nodata,
                data,
            )
        else:
            data = xr.where(
                (solar_za.sel(band=1) == -32768 * 0.01) | (data == src_nodata),
                dst_nodata,
                data,
            )

        data = data.transpose('band', 'y', 'x').astype(dtype)
        attrs['sensor'] = sensor
        attrs['calibration'] = 'BRDF-adjusted surface reflectance'
        attrs['drange'] = drange

        return data.assign_attrs(**attrs).gw.assign_nodata_attrs(dst_nodata)
