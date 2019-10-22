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

    rphase_stack = xr.concat([rphase] * len(m.band), dim='band')
    rphase_stack.coords['band'] = m.band.values

    cos_solar_za_stack = xr.concat([cos_solar_za] * len(m.band), dim='band')
    cos_solar_za_stack.coords['band'] = m.band.values

    cos_sensor_za_stack = xr.concat([cos_sensor_za] * len(m.band), dim='band')
    cos_sensor_za_stack.coords['band'] = m.band.values

    return rphase_stack * ((1.0 - xr.ufuncs.exp(-m*r)) / (4.0 * (cos_solar_za_stack + cos_sensor_za_stack)))


def t_s(r, cos_solar_za):

    """
    Calculates atmospheric transmittance of sun-surface path (downward)

    Args:
        r (float): The Rayleigh optical depth.
        cos_solar_za (DataArray): The cosine of the solar zenith angle.

    Returns:
        ``xarray.DataArray``
    """

    cos_solar_za_stack = xr.concat([cos_solar_za] * len(r.band), dim='band')
    cos_solar_za_stack.coords['band'] = r.band.values

    cose = xr.ufuncs.exp(-r / cos_solar_za_stack)

    return cose + cose * (xr.ufuncs.exp(0.52*r / cos_solar_za_stack) - 1.0)


def t_v(r, cos_sensor_za):

    """
    Calculates atmospheric transmittance of surface-sensor path (upward)

    Args:
        r (float): The Rayleigh optical depth.
        cos_sensor_za (DataArray): The cosine of the sensor zenith angle.

    Returns:
        ``xarray.DataArray``
    """

    cos_sensor_za_stack = xr.concat([cos_sensor_za] * len(r.band), dim='band')
    cos_sensor_za_stack.coords['band'] = r.band.values

    cose = xr.ufuncs.exp(-r / cos_sensor_za_stack)

    return cose + cose * (xr.ufuncs.exp(0.52 * r / cos_sensor_za_stack) - 1.0)


def s_atm(r):

    """
    Calculates atmospheric backscattering ratio to count multiple reflections between the surface and atmosphere

    Args:
        r (float): The Rayleigh optical depth.

    Returns:
        ``float``
    """

    return (0.92*r) * xr.ufuncs.exp(-r)


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

    def dn_to_sr(self, dn, solar_za, sensor_za, solar_az, sensor_az, sensor=None, method='srem', meta=None):

        """
        Converts digital numbers to surface reflectance

        Args:
            dn (DataArray): The digital number data to calibrate.
            solar_za (DataArray): The solar zenith angle.
            sensor_za (DataArray): The sensor zenith angle.
            solar_az (DataArray): The solar azimuth angle.
            sensor_az (DataArray): The sensor azimuth angle.
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
            >>> from geowombat.radiometry import SurfaceReflectance
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

        attrs = dn.attrs

        # Get the data band names and positional indices
        band_names = dn.band.values.tolist()

        if meta:

            # Get the sensor wavelengths
            wavelengths = dn.gw.wavelengths[meta.sensor]

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

        micrometers = dn.gw.micrometers[meta.sensor]
        band_um = [getattr(micrometers, p) for p in band_names]
        um = xr.DataArray(data=band_um, coords={'band': band_names}, dims='band')

        sr_data = self.toar_to_sr(toar, solar_za, sensor_za, solar_az, sensor_az, um)
        sr_data.attrs = attrs

        return sr_data

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
    def toar_to_sr(toar, solar_za, sensor_za, solar_az, sensor_az, um):

        """
        Converts top-of-atmosphere reflectance to surface reflectance

        Args:
            toar (DataArray): The top-of-atmosphere reflectance.
            solar_za (DataArray): The solar zenith angle.
            sensor_za (DataArray): The sensor zenith angle.
            solar_az (DataArray): The solar azimuth angle.
            sensor_az (DataArray): The sensor azimuth angle.
            um (namedtuple): The sensor wavelength micrometers.

        Returns:
            ``xarray.DataArray``
        """

        # Scale the angles
        solar_zenith_angle = solar_za * 0.01
        sensor_zenith_angle = sensor_za * 0.01
        solar_azimuth_angle = solar_az * 0.01
        sensor_azimuth_angle = sensor_az * 0.01

        # Cosine(angles)
        cos_solar_zenith_angle = xr.ufuncs.cos(solar_zenith_angle)
        cos_sensor_zenith_angle = xr.ufuncs.cos(sensor_zenith_angle)

        # air mass
        m = (1.0 / cos_solar_zenith_angle) + (1.0 / cos_sensor_zenith_angle)

        m = xr.concat([m]*len(toar.band), dim='band')
        m.coords['band'] = toar.band.values

        # Rayleigh optical depth
        # Hansen, JF and Travis, LD (1974) LIGHT SCATTERING IN PLANETARY ATMOSPHERES
        r = 0.008569*um**-4 * (1.0 + 0.0113*um**-2 + 0.0013*um**-4)

        # Rayleigh phase function
        # scattering angle = the angle between the direction of incident and scattered radiation
        # Liu, CH and Liu GR (2009) AEROSOL OPTICAL DEPTH RETRIEVAL FOR SPOT HRV IMAGES, Journal of Marine Science and Technology

        # Relative azimuth angle
        # http://stcorp.github.io/harp/doc/html/algorithms/derivations/relative_azimuth_angle.html
        rel_azimuth_angle = xr.ufuncs.fabs(solar_azimuth_angle - sensor_azimuth_angle - 180.0)

        # http://stcorp.github.io/harp/doc/html/algorithms/derivations/scattering_angle.html
        scattering_angle = xr.ufuncs.arccos(-xr.ufuncs.cos(solar_zenith_angle)*xr.ufuncs.cos(sensor_zenith_angle) -
                                            xr.ufuncs.sin(solar_zenith_angle)*xr.ufuncs.sin(sensor_zenith_angle)*xr.ufuncs.cos(rel_azimuth_angle))

        rphase = ((3.0 * 0.9587256) / (4.0 + 1.0 - 0.9587256)) * (1.0 + xr.ufuncs.cos(scattering_angle)**2)

        pr_data = p_r(m, r, rphase, cos_solar_zenith_angle, cos_sensor_zenith_angle)

        # da.nan_to_num(pr_data).max().compute()

        toar_diff = toar - pr_data

        total_transmission = t_s(r, cos_solar_zenith_angle) * t_v(r, cos_sensor_zenith_angle)

        return toar_diff / (toar_diff * s_atm(r) + total_transmission)
