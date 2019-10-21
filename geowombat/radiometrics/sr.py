import math
from collections import namedtuple

import pandas as pd
import xarray as xr
import dask.array as da


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

    return rphase * ((1.0 - math.exp(-m*r)) / (4.0 * cos_solar_za + cos_sensor_za))


def t_s(r, cos_solar_za):

    """
    Calculates atmospheric transmittance of sun-surface path (downward)

    Args:
        r (float): The Rayleigh optical depth.
        cos_solar_za (DataArray): The cosine of the solar zenith angle.

    Returns:
        ``xarray.DataArray``
    """

    cose = da.exp(-r / cos_solar_za)

    return cose + cose * (da.exp(0.52*r / cos_solar_za) - 1.0)


def t_v(r, cos_sensor_za):

    """
    Calculates atmospheric transmittance of surface-sensor path (upward)

    Args:
        r (float): The Rayleigh optical depth.
        cos_sensor_za (DataArray): The cosine of the sensor zenith angle.

    Returns:
        ``xarray.DataArray``
    """

    cose = da.exp(-r / cos_sensor_za)

    return cose + cose * (da.exp(0.52 * r / cos_sensor_za) - 1.0)


def s_atm(r):

    """
    Calculates atmospheric backscattering ratio to count multiple reflections between the surface and atmosphere

    Args:
        r (float): The Rayleigh optical depth.

    Returns:
        ``float``
    """

    return (0.92*r) * math.exp(-r)


def _format_coeff(dataframe, key):

    dataframe_ = dataframe[dataframe.iloc[:, 0].str.startswith(key)].values
    dataframe_[:, 1] = dataframe_[:, 1].astype(float)
    dataframe_[:, 0] = list(range(1, dataframe_.shape[0]+1))

    return dict(dataframe_)


class SurfaceReflectance(object):

    """
    A general class for surface reflectance calibration methods
    """

    @staticmethod
    def get_coefficients(meta_file):

        """
        Gets coefficients from a metadata file

        Args:
            meta_file (str): A metadata file.

        Returns:
            ``namedtuple``
        """

        associations = {'LANDSAT_8': 'l8'}

        MetaCoeffs = namedtuple('MetaCoeffs', 'sensor m_l a_l m_p a_p')

        df = pd.read_csv(meta_file, sep='=')

        df.iloc[:, 0] = df.iloc[:, 0].str.strip()
        df.iloc[:, 1] = df.iloc[:, 1].str.strip()

        m_l = _format_coeff(df, 'RADIANCE_MULT_BAND_')
        a_l = _format_coeff(df, 'RADIANCE_ADD_BAND_')
        m_p = _format_coeff(df, 'REFLECTANCE_MULT_BAND_')
        a_p = _format_coeff(df, 'REFLECTANCE_ADD_BAND_')

        spacecraft_id = dict(df[df.iloc[:, 0].str.startswith('SPACECRAFT_ID')].values)
        spacecraft_id['SPACECRAFT_ID'] = spacecraft_id['SPACECRAFT_ID'].replace('"', '')
        sensor = associations[spacecraft_id['SPACECRAFT_ID']]

        return MetaCoeffs(sensor=sensor, m_l=m_l, a_l=a_l, m_p=m_p, a_p=a_p)

    def dn_to_sr(self, dn, solar_za, sensor_za, sensor=None, method='srem', meta=None):

        """
        Converts digital numbers to surface reflectance

        Args:
            dn (DataArray): The digital number data to calibrate.
            solar_za (DataArray): The solar zenith angle.
            sensor_za (DataArray): The sensor zenith angle.
            sensor (Optional[str]): The data's sensor.
            method (Optional[str]): The method to use. Only 'srem' is supported.
            meta (Optional[namedtuple]): A metadata object with gain and bias coefficients.

        References:

            See :cite:`bilal_etal_2019`

            TODO: add to .bib
            Bilal et al. (2019) A Simplified and Robust Surface Reflectance Estimation Method (SREM) for Use over Diverse
                Land Surfaces Using Multi-Sensor Data, Remote Sensing, 11(1344) doi:10.3390/rs11111344.

            https://www.usgs.gov/land-resources/nli/landsat/using-usgs-landsat-level-1-data-product

        Examples:
            >>> from geowombat.radiometrics import SurfaceReflectance
            >>>
            >>> sr = SurfaceReflectance()
            >>> meta = sr.get_coefficients('file.MTL')
            >>>
            >>> # Convert DNs to surface reflectance using Landsat metadata
            >>> with gw.open('dn.tif') as ds:
            >>>     sr_data = sr.dn_to_sr(ds, solar_za, sensor_za, meta=meta)

        Returns:
            ``xarray.DataArray``
        """

        if meta:

            # Get the sensor wavelengths
            wavelengths = dn.gw.wavelengths[meta.sensor]

            # Get the data band names and positional indices
            band_names = dn.band.values.tolist()
            band_indices = [getattr(wavelengths, p) for p in band_names]

            # Get the gain and offsets and
            #   convert the gain and offsets
            #   to named coordinates.
            m_p = xr.DataArray(data=[meta.m_p[bi] for bi in band_indices], coords={'band': band_names}, dims='band')
            a_p = xr.DataArray(data=[meta.a_p[bi] for bi in band_indices], coords={'band': band_names}, dims='band')

            toar = self.dn_to_toar(dn, m_p, a_p)

            # TOAR with sun angle correction
            # toar / da.cos(solar_za)

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

        return self.toar_to_sr(toar, solar_za, sensor_za)

    @staticmethod
    def dn_to_radiance(dn, gain, bias):

        """
        Converts digital numbers to radiance

        Args:
            dn (DataArray): The digital number data to calibrate.
            gain (Optional[float]): A gain value.
            bias (Optional[float]): A bias value.

        Returns:
            ``xarray.DataArray``
        """

        # TODO: get gain and bias from metadata
        return gain * dn + bias

    @staticmethod
    def dn_to_toar(dn, gain, bias):

        """
        Converts digital numbers to radiance

        Args:
            dn (DataArray): The digital number data to calibrate.
            gain (Optional[float]): A gain value.
            bias (Optional[float]): A bias value.

        Returns:
            ``xarray.DataArray``
        """

        # TODO: get gain and bias from metadata
        return gain * dn + bias

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

        solar_zenith_angle = solar_za * 0.01

        return (global_args.pi * radiance * global_args.d**2) / (global_args.esun * da.cos(solar_zenith_angle))

    @staticmethod
    def toar_to_sr(toar, solar_za, sensor_za):

        """
        Converts top-of-atmosphere reflectance to surface reflectance

        Args:
            toar (DataArray): The top-of-atmosphere reflectance.
            solar_za (DataArray): The solar zenith angle.
            sensor_za (DataArray): The sensor zenith angle.

        Returns:
            ``xarray.DataArray``
        """

        solar_zenith_angle = solar_za * 0.01
        sensor_zenith_angle = sensor_za * 0.01

        cos_solar_zenith_angle = da.cos(solar_zenith_angle)
        cos_sensor_zenith_angle = da.cos(sensor_zenith_angle)

        # air mass
        m = (1.0 / da.cos(solar_zenith_angle)) + (1.0 / da.cos(sensor_zenith_angle))

        # Rayleigh optical depth
        r = 0.008569**-4 * (1.0 + 0.0113**-2 + 0.0013**-4)

        # Rayleigh phase function
        # TODO: get scattering angle
        rphase = ((3.0 * 0.9587256) / (4.0 + 1.0 - 0.9587256)) * (1.0 + da.cos(scattering_angle)**2)

        pr_data = p_r(m, r, rphase, cos_solar_zenith_angle, cos_sensor_zenith_angle)

        toar_diff = toar - pr_data

        return toar_diff / (toar_diff * s_atm(r) + t_s(r, cos_solar_zenith_angle) * t_v(r, cos_sensor_zenith_angle))
