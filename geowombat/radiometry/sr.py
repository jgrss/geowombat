import math
from collections import namedtuple
from datetime import datetime as dtime
import datetime
import logging

from ..handler import add_handler
from .angles import relative_azimuth

import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da


logger = logging.getLogger(__name__)
logger = add_handler(logger)


def coeffs_to_array(coeffs, band_names):
    """Converts coefficients to a DataArray"""
    return xr.DataArray(data=[coeffs[bi] for bi in band_names], coords={'band': band_names}, dims='band')


def p_r(m, r, rphase, cos_solar_za, cos_sensor_za):

    """
    Calculates atmospheric reflectance due to Rayleigh scattering

    Args:
        m (float): The air mass.
        r (float): The Rayleigh optical depth.
        rphase (float): The Rayleigh phase function.
        cos_solar_za (DataArray): The cosine of the solar zenith angle.
        cos_sensor_za (DataArray): The cosine of the sensor zenith angle.

    Returns:
        ``xarray.DataArray``
    """

    rphase_stack = xr.concat([rphase] * len(m.band), dim='band')
    rphase_stack.coords['band'] = m.band.values

    cos_solar_za_stack = xr.concat([cos_solar_za] * len(m.band), dim='band')
    cos_solar_za_stack.coords['band'] = m.band.values

    cos_sensor_za_stack = xr.concat([cos_sensor_za] * len(m.band), dim='band')
    cos_sensor_za_stack.coords['band'] = m.band.values

    return rphase_stack * ((1.0 - xr.ufuncs.exp(-m*r)) / (4.0 * (cos_solar_za_stack + cos_sensor_za_stack)))


def t_sv(r, cos_zenith):

    """
    Calculates atmospheric transmittance of sun-surface path

    Args:
        r (float): The Rayleigh optical depth.
        cos_zenith (DataArray): The cosine of the zenith angle.

    Returns:
        ``xarray.DataArray``
    """

    cos_zenith_stack = xr.concat([cos_zenith] * len(r.band), dim='band')
    cos_zenith_stack.coords['band'] = r.band.values

    cose1 = xr.ufuncs.exp(-r / cos_zenith_stack)
    cose2 = xr.ufuncs.exp(0.52*r / cos_zenith_stack)

    return cose1 + cose1 * (cose2 - 1.0)


def s_atm(r):

    """
    Calculates atmospheric backscattering ratio to count multiple reflections between the surface and atmosphere

    Args:
        r (float): The Rayleigh optical depth.

    Returns:
        ``float``
    """

    return (0.92*r) * xr.ufuncs.exp(-r)


def _format_coeff(dataframe, sensor, key):

    bands_dict = dict(l5={'1': 'blue', '2': 'green', '3': 'red', '4': 'nir', '5': 'swir1', '6': 'th', '7': 'swir2'},
                      l7={'1': 'blue', '2': 'green', '3': 'red', '4': 'nir', '5': 'swir1', '6VCID1': 'th1',
                          '6VCID2': 'th2', '7': 'swir2', '8': 'pan'},
                      l8={'1': 'coastal', '2': 'blue', '3': 'green', '4': 'red', '5': 'nir', '6': 'swir1',
                          '7': 'swir2', '8': 'pan', '9': 'cirrus', '10': 'th1', '11': 'th2'})

    sensor_dict = bands_dict[sensor]

    dataframe_ = dataframe[dataframe.iloc[:, 0].str.startswith(key)].values

    pairs = {}

    for di in range(dataframe_.shape[0]):

        bd = dataframe_[di, 0]
        cf = dataframe_[di, 1]

        # e.g., REFLECTANCE_ADD_BAND_1 -> 1
        band_name = ''.join(bd.split('_')[3:])

        if band_name in sensor_dict:

            var_band = sensor_dict[band_name]
            pairs[var_band] = float(cf)

    if not pairs:
        logger.warning('  No metadata coefficients were acquired.')

    return pairs


class MetaData(object):

    """
    A class for sensor metadata
    """

    @staticmethod
    def get_landsat_coefficients(meta_file):

        """
        Gets coefficients from a Landsat metadata file

        Args:
            meta_file (str): A metadata file.

        Returns:

            ``namedtuple``:

                sensor, m_l, a_l, m_p, a_p
        """

        associations = {'LANDSAT_5': 'l5',
                        'LANDSAT_7': 'l7',
                        'LANDSAT_8': 'l8'}

        MetaCoeffs = namedtuple('MetaCoeffs', 'sensor m_l a_l m_p a_p date_acquired')

        df = pd.read_csv(meta_file, sep='=')

        df.iloc[:, 0] = df.iloc[:, 0].str.strip()
        df.iloc[:, 1] = df.iloc[:, 1].str.strip()

        spacecraft_id = dict(df[df.iloc[:, 0].str.startswith('SPACECRAFT_ID')].values)
        spacecraft_id['SPACECRAFT_ID'] = spacecraft_id['SPACECRAFT_ID'].replace('"', '')
        sensor = associations[spacecraft_id['SPACECRAFT_ID']]

        m_l = _format_coeff(df, sensor, 'RADIANCE_MULT_BAND_')
        a_l = _format_coeff(df, sensor, 'RADIANCE_ADD_BAND_')
        m_p = _format_coeff(df, sensor, 'REFLECTANCE_MULT_BAND_')
        a_p = _format_coeff(df, sensor, 'REFLECTANCE_ADD_BAND_')

        date_acquired_ = dict(df[df.iloc[:, 0].str.startswith('DATE_ACQUIRED')].values)
        date_acquired_ = date_acquired_['DATE_ACQUIRED'].replace('"', '')
        year, month, day = date_acquired_.split('-')

        year = int(year)
        month = int(month)
        day = int(day)

        scene_center_time = dict(df[df.iloc[:, 0].str.startswith('SCENE_CENTER_TIME')].values)
        scene_center_time = scene_center_time['SCENE_CENTER_TIME'].replace('"', '')
        hour = int(scene_center_time.split(':')[0])

        date_acquired = dtime(year, month, day, hour, tzinfo=datetime.timezone.utc)

        return MetaCoeffs(sensor=sensor,
                          m_l=m_l,
                          a_l=a_l,
                          m_p=m_p,
                          a_p=a_p,
                          date_acquired=date_acquired)


class LinearAdjustments(object):

    """
    A class for linear bandpass adjustments
    """

    def __init__(self):

        self.coefficients = dict(s2a=dict(l5=None,
                                          l7=None,
                                          l8=dict(alphas=dict(coastal=-0.0002,
                                                              blue=-0.004,
                                                              green=-0.0009,
                                                              red=0.0009,
                                                              nir=-0.0001,
                                                              swir1=-0.0011,
                                                              swir2=-0.0012),
                                                  betas=dict(coastal=0.9959,
                                                             blue=0.9778,
                                                             green=1.0053,
                                                             red=0.9765,
                                                             nir=0.9983,
                                                             swir1=0.9987,
                                                             swir2=1.003))),
                                 s2b=dict(l5=None,
                                          l7=None,
                                          l8=dict(alphas=dict(coastal=-0.0002,
                                                              blue=-0.004,
                                                              green=-0.0008,
                                                              red=0.001,
                                                              nir=0.0,
                                                              swir1=-0.0003,
                                                              swir2=0.0004),
                                                  betas=dict(coastal=0.9959,
                                                             blue=0.9778,
                                                             green=1.0075,
                                                             red=0.9761,
                                                             nir=0.9966,
                                                             swir1=1.0,
                                                             swir2=0.9867))),
                                 l5=dict(l7=None,
                                         l8=dict(alphas=dict(blue=-0.0095,
                                                             green=-0.0016,
                                                             red=-0.0022,
                                                             nir=-0.0021,
                                                             swir1=-0.003,
                                                             swir2=0.0029,
                                                             pan=-0.00443),
                                                 betas=dict(blue=0.9785,
                                                            green=0.9542,
                                                            red=0.9825,
                                                            nir=1.0073,
                                                            swir1=1.0171,
                                                            swir2=0.9949,
                                                            pan=0.9717)),
                                         s2=None),
                                 l7=dict(l5=None,
                                         l8=dict(alphas=dict(blue=-0.0095,
                                                             green=-0.0016,
                                                             red=-0.0022,
                                                             nir=-0.0021,
                                                             swir1=-0.003,
                                                             swir2=0.0029,
                                                             pan=-0.00443),
                                                 betas=dict(blue=0.9785,
                                                            green=0.9542,
                                                            red=0.9825,
                                                            nir=1.0073,
                                                            swir1=1.0171,
                                                            swir2=0.9949,
                                                            pan=0.9717)),
                                         s2=None))

    def bandpass(self,
                 data,
                 sensor=None,
                 to='l8',
                 band_names=None,
                 scale_factor=1,
                 src_nodata=0,
                 dst_nodata=0):

        """
        Applies a bandpass adjustment by applying a linear function to surface reflectance values

        Args:
            data (DataArray): The data to adjust.
            sensor (Optional[str]): The sensor to adjust.
            to (Optional[str]): The sensor to adjust to.
            band_names (Optional[list]): The bands to adjust. If not given, all bands are adjusted.
            scale_factor (Optional[float]): A scale factor to apply to the input data.
            src_nodata (Optional[int or float]): The input 'no data' value.
            dst_nodata (Optional[int or float]): The output 'no data' value.

        Reference:

            Sentinel-2 and Landsat 8:

                https://hls.gsfc.nasa.gov/algorithms/bandpass-adjustment/

                See :cite:`chastain_etal_2019` for further details

            Landsat 7 and Landsat 8:

                See :cite:`roy_etal_2016` (Table 2)

        Returns:
            ``xarray.DataArray``

        Examples:
            >>> import geowombat as gw
            >>> from geowombat.radiometry import LinearAdjustments
            >>>
            >>> la = LinearAdjustments()
            >>>
            >>> # Adjust all Sentinel-2 bands to Landsat 8
            >>> with gw.config.update(sensor='s2'):
            >>>     with gw.open('sentinel-2.tif') as ds:
            >>>         ds_adjusted = la.bandpass(ds, to='l8')
        """

        attrs = data.attrs.copy()

        # Set 'no data' as nans
        data = data.where(data != src_nodata)

        if scale_factor == 1.0:
            scale_factor = data.gw.scale_factor

        # Scale the reflectance data
        if scale_factor != 1:
            data = data * scale_factor

        if not band_names:

            band_names = data.band.values.tolist()

            if band_names[0] == 1:
                band_names = list(data.gw.wavelengths[sensor]._fields)

        coeff_dict = self.coefficients[sensor][to]

        alphas = np.array([coeff_dict['alphas'][bd] for bd in band_names], dtype='float64')
        betas = np.array([coeff_dict['betas'][bd] for bd in band_names], dtype='float64')

        alphas = xr.DataArray(data=alphas,
                              coords={'band': band_names},
                              dims='band')

        betas = xr.DataArray(data=betas,
                             coords={'band': band_names},
                             dims='band')

        # Apply the linear bandpass adjustment
        data = alphas + betas * data

        if scale_factor != 1:
            data = data / scale_factor

        data = data.fillna(dst_nodata)

        data.attrs['adjustment'] = '{} to {}'.format(sensor, to)
        data.attrs['alphas'] = alphas.data.tolist()
        data.attrs['betas'] = betas.data.tolist()

        data.attrs = attrs

        return data


class RadTransforms(MetaData):

    """
    A class for radiometric transformations
    """

    def dn_to_sr(self,
                 dn,
                 solar_za,
                 solar_az,
                 sensor_za,
                 sensor_az,
                 src_nodata=-32768,
                 dst_nodata=-32768,
                 sensor=None,
                 method='srem',
                 angle_factor=0.01,
                 meta=None):

        """
        Converts digital numbers to surface reflectance

        Args:
            dn (DataArray): The digital number data to calibrate.
            solar_za (DataArray): The solar zenith angle.
            solar_az (DataArray): The solar azimuth angle.
            sensor_za (DataArray): The sensor, or view, zenith angle.
            sensor_az (DataArray): The sensor, or view, azimuth angle.
            src_nodata (Optional[int or float]): The input 'no data' value.
            dst_nodata (Optional[int or float]): The output 'no data' value.
            sensor (Optional[str]): The data's sensor.
            method (Optional[str]): The method to use. Only 'srem' is supported.
            angle_factor (Optional[float]): The scale factor for angles.
            meta (Optional[namedtuple]): A metadata object with gain and bias coefficients.

        References:
            https://www.usgs.gov/land-resources/nli/landsat/using-usgs-landsat-level-1-data-product

        Returns:

            ``xarray.DataArray``:

                Data range: 0-1

        Examples:
            >>> from geowombat.radiometry import RadTransforms
            >>>
            >>> sr = RadTransforms()
            >>> meta = sr.get_landsat_coefficients('file.MTL')
            >>>
            >>> # Convert DNs to surface reflectance using Landsat metadata
            >>> with gw.open('dn.tif') as ds:
            >>>     sr_data = sr.dn_to_sr(ds, solar_za, sensor_za, meta=meta)
        """

        attrs = dn.attrs.copy()

        # Get the data band names and positional indices
        band_names = dn.band.values.tolist()

        if meta:

            if not sensor:
                sensor = meta.sensor

            # Ensure that the metadata holder has matching bands
            for bi in band_names:

                if bi not in meta.m_p:
                    logger.warning(meta.m_p)
                    logger.warning(band_names)
                    logger.exception('  The metadata holder does not have matching bands.')
                    raise ValueError

                if bi not in meta.a_p:
                    logger.warning(meta.a_p)
                    logger.warning(band_names)
                    logger.exception('  The metadata holder does not have matching bands.')
                    raise ValueError

            # Get the gain and offsets and
            #   convert the gain and offsets
            #   to named coordinates.
            m_p = coeffs_to_array(meta.m_p, band_names)
            a_p = coeffs_to_array(meta.a_p, band_names)

            # TOAR with sun angle correction
            toar = self.dn_to_toar(dn, m_p, a_p, solar_za=solar_za, angle_factor=angle_factor, sun_angle=True)

        else:

            if not sensor:
                sensor = dn.gw.sensor

            # d = distance between the Earth and Sun in the astronomical unit
            # ESUN = mean solar exoatmospheric radiation
            GlobalArgs = namedtuple('GlobalArgs', 'pi d esun')

            # TODO: set global arguments
            global_args = GlobalArgs(pi=math.pi, d=None, esun=None)

            radiance = self.dn_to_radiance(dn, None, None)

            toar = self.radiance_to_toar(radiance, solar_za, global_args)

        sr_data = self.toar_to_sr(toar,
                                  solar_za,
                                  solar_az,
                                  sensor_za,
                                  sensor_az,
                                  sensor,
                                  src_nodata=src_nodata,
                                  dst_nodata=dst_nodata)

        attrs['sensor'] = sensor
        attrs['nodata'] = dst_nodata
        attrs['calibration'] = 'surface reflectance'
        attrs['method'] = method
        attrs['drange'] = (0, 1)

        sr_data.attrs = attrs

        return sr_data

    @staticmethod
    def _linear_transform(data, gain, bias):
        return gain * data + bias

    def dn_to_radiance(self, dn, gain, bias):

        """
        Converts digital numbers to radiance

        Args:
            dn (DataArray): The digital number data to calibrate.
            gain (DataArray): A gain value.
            bias (DataArray): A bias value.

        Returns:
            ``xarray.DataArray``
        """

        attrs = dn.attrs.copy()
        attrs['calibration'] = 'radiance'

        return self._linear_transform(dn, gain, bias).assign_attrs(**attrs)

    def dn_to_toar(self, dn, gain, bias, solar_za=None, angle_factor=0.01, sun_angle=True):

        """
        Converts digital numbers to top-of-atmosphere reflectance

        Args:
            dn (DataArray): The digital number data to calibrate.
            gain (DataArray): A gain value.
            bias (DataArray): A bias value.
            solar_za (DataArray): The solar zenith angle.
            angle_factor (Optional[float]): The scale factor for angles.
            sun_angle (Optional[bool]): Whether to correct for the sun angle.

        Returns:
            ``xarray.DataArray``
        """

        attrs = dn.attrs.copy()

        toar = self._linear_transform(dn, gain, bias)

        if sun_angle:

            # TOA reflectance with sun angle correction
            cos_sza = xr.concat([xr.ufuncs.cos(xr.ufuncs.deg2rad(solar_za * angle_factor))] * len(toar.band), dim='band')
            cos_sza.coords['band'] = toar.band.values
            toar = toar / cos_sza

        attrs['calibration'] = 'top-of-atmosphere reflectance'

        return toar.assign_attrs(**attrs)

    @staticmethod
    def radiance_to_toar(radiance, solar_za, global_args):

        """
        Converts digital numbers to top-of-atmosphere reflectance

        Args:
            radiance (DataArray): The radiance data to calibrate.
            solar_za (DataArray): The solar zenith angle.
            global_args (namedtuple): Global arguments.

        Returns:
            ``xarray.DataArray``
        """

        attrs = radiance.attrs.copy()

        solar_zenith_angle = solar_za * 0.01

        toar_data = (global_args.pi * radiance * global_args.d**2) / (global_args.esun * da.cos(solar_zenith_angle))

        attrs['calibration'] = 'top-of-atmosphere reflectance'

        toar_data.attrs = attrs

        return toar_data

    @staticmethod
    def toar_to_sr(toar,
                   solar_za,
                   solar_az,
                   sensor_za,
                   sensor_az,
                   sensor,
                   src_nodata=-32768,
                   dst_nodata=-32768,
                   method='srem'):

        """
        Converts top-of-atmosphere reflectance to surface reflectance

        Args:
            toar (DataArray): The top-of-atmosphere reflectance.
            solar_za (DataArray): The solar zenith angle.
            solar_az (DataArray): The solar azimuth angle.
            sensor_za (DataArray): The sensor zenith angle.
            sensor_az (DataArray): The sensor azimuth angle.
            sensor (str): The satellite sensor.
            src_nodata (Optional[int or float]): The input 'no data' value.
            dst_nodata (Optional[int or float]): The output 'no data' value.
            method (Optional[str]): The method to use. Currently, only 'srem' is supported and ``method`` is not used.

                Choices:
                    'srem': A Simplified and Robust Surface Reflectance Estimation Method (SREM)

        References:

            See :cite:`bilal_etal_2019` for the SREM method

        Returns:

            ``xarray.DataArray``:

                Data range: 0-1
        """

        attrs = toar.attrs.copy()

        # Set 'no data' as nans
        toar = toar.where(toar != src_nodata)

        # Convert micrometers to nanometers
        central_nm = toar.gw.central_um[sensor] * 1000.0
        band_names = list(toar.gw.wavelengths[sensor]._fields)
        band_nm = [getattr(central_nm, p) for p in band_names]
        nm = xr.DataArray(data=band_nm, coords={'band': band_names}, dims='band')

        # Scale the angles to degrees
        sza = solar_za * 0.01
        sza.coords['band'] = [1]

        saa = solar_az * 0.01
        saa.coords['band'] = [1]

        vza = sensor_za * 0.01
        vza.coords['band'] = [1]

        vaa = sensor_az * 0.01
        vaa.coords['band'] = [1]

        # Convert to radians
        rad_sza = xr.ufuncs.deg2rad(sza)
        rad_vza = xr.ufuncs.deg2rad(vza)

        # Cosine(deg2rad(angles)) = angles x (pi / 180)
        cos_sza = xr.ufuncs.cos(rad_sza)
        cos_sza.coords['band'] = [1]
        cos_vza = xr.ufuncs.cos(rad_vza)
        cos_vza.coords['band'] = [1]

        sin_sza = xr.ufuncs.sin(rad_sza)
        sin_sza.coords['band'] = [1]
        sin_vza = xr.ufuncs.sin(rad_vza)
        sin_vza.coords['band'] = [1]

        # air mass
        m = (1.0 / cos_sza.sel(band=1)) + (1.0 / cos_vza.sel(band=1))
        m = m.expand_dims(dim='band')
        m = m.assign_coords(band=[1])

        m = xr.concat([m]*len(toar.band), dim='band')
        m.coords['band'] = toar.band.values

        # Rayleigh optical depth
        # Hansen, JF and Travis, LD (1974) LIGHT SCATTERING IN PLANETARY ATMOSPHERES
        r = 0.008569*nm**-4 * (1.0 + 0.0113*nm**-2 + 0.0013*nm**-4)

        # Relative azimuth angle
        # TODO: doesn't work if the band coordinate is named
        raa = relative_azimuth(saa, vaa)
        rad_raa = xr.ufuncs.deg2rad(raa)
        cos_raa = xr.ufuncs.cos(rad_raa)

        # scattering angle = the angle between the direction of incident and scattered radiation
        # Liu, CH and Liu GR (2009) AEROSOL OPTICAL DEPTH RETRIEVAL FOR SPOT HRV IMAGES, Journal of Marine Science and Technology
        # http://stcorp.github.io/harp/doc/html/algorithms/derivations/scattering_angle.html
        # cos_sza = cos(pi/180 x sza)
        # cos_vza = cos(pi/180 x vza)
        # sin_sza = sin(pi/180 x sza)
        # sin_vza = sin(pi/180 x vza)
        scattering_angle = xr.ufuncs.arccos(-cos_sza * cos_vza - sin_sza * sin_vza * cos_raa)
        cos2_scattering_angle = xr.ufuncs.cos(scattering_angle)**2

        # Rayleigh phase function
        rayleigh_a = 0.9587256
        rayleigh_b = 1.0 - rayleigh_a
        rphase = ((3.0 * rayleigh_a) / (4.0 + rayleigh_b)) * (1.0 + cos2_scattering_angle)

        # Get the air mass
        pr_data = p_r(m, r, rphase, cos_sza, cos_vza)

        toar_diff = toar - pr_data

        # Total transmission = downward x upward
        transmission = t_sv(r, cos_sza) * t_sv(r, cos_vza)

        # Atmospheric backscattering ratio
        ab_ratio = s_atm(r)

        sr_data = (toar_diff / (toar_diff * ab_ratio + transmission)).fillna(src_nodata)

        # Create a 'no data' mask
        mask = sr_data.where((sr_data != src_nodata) & (sr_data > 0)).count(dim='band').astype('uint8')

        # Mask 'no data' values
        sr_data = xr.where(mask < sr_data.gw.nbands,
                           dst_nodata,
                           sr_data.clip(0, 1))\
            .transpose('band', 'y', 'x').astype('float64')

        attrs['sensor'] = sensor
        attrs['calibration'] = 'surface reflectance'
        attrs['nodata'] = dst_nodata
        attrs['drange'] = (0, 1)

        sr_data.attrs = attrs

        return sr_data
