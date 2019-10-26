import math
from collections import namedtuple
from datetime import datetime as dtime
import datetime

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


class MetaData(object):

    @staticmethod
    def get_landsat_coefficients(meta_file):

        """
        Gets coefficients from a Landsat metadata file

        Args:
            meta_file (str): A metadata file.

        Returns:
            ``namedtuple``
        """

        associations = {'LANDSAT_8': 'l8'}

        MetaCoeffs = namedtuple('MetaCoeffs', 'sensor m_l a_l m_p a_p date_acquired')

        df = pd.read_csv(meta_file, sep='=')

        df.iloc[:, 0] = df.iloc[:, 0].str.strip()
        df.iloc[:, 1] = df.iloc[:, 1].str.strip()

        m_l = _format_coeff(df, 'RADIANCE_MULT_BAND_')
        a_l = _format_coeff(df, 'RADIANCE_ADD_BAND_')
        m_p = _format_coeff(df, 'REFLECTANCE_MULT_BAND_')
        a_p = _format_coeff(df, 'REFLECTANCE_ADD_BAND_')

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

        spacecraft_id = dict(df[df.iloc[:, 0].str.startswith('SPACECRAFT_ID')].values)
        spacecraft_id['SPACECRAFT_ID'] = spacecraft_id['SPACECRAFT_ID'].replace('"', '')
        sensor = associations[spacecraft_id['SPACECRAFT_ID']]

        return MetaCoeffs(sensor=sensor,
                          m_l=m_l,
                          a_l=a_l,
                          m_p=m_p,
                          a_p=a_p,
                          date_acquired=date_acquired)


class RadTransforms(MetaData):

    """
    A general class for radiometric transformations
    """

    def dn_to_sr(self,
                 dn,
                 solar_za,
                 solar_az,
                 sensor_za,
                 sensor_az,
                 nodata=-32768,
                 sensor=None,
                 method='srem',
                 meta=None):

        """
        Converts digital numbers to surface reflectance

        Args:
            dn (DataArray): The digital number data to calibrate.
            solar_za (DataArray): The solar zenith angle.
            solar_az (DataArray): The solar azimuth angle.
            sensor_za (DataArray): The sensor, or view, zenith angle.
            sensor_az (DataArray): The sensor, or view, azimuth angle.
            nodata (Optional[int or float]): The 'no data' value from the pixel angle data.
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
            >>> from geowombat.radiometry import RadTransforms
            >>>
            >>> sr = RadTransforms()
            >>> meta = sr.get_landsat_coefficients('file.MTL')
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

        sr_data = self.toar_to_sr(toar, solar_za, sensor_za, solar_az, sensor_az, um).fillna(nodata).clip(0, 1).astype('float64')
        sr_data = sr_data.where(sr_data != nodata)

        attrs['sensor'] = sensor
        attrs['nodata'] = nodata
        attrs['method'] = method
        attrs['drange'] = (0, 1)

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

        # kwargs = dict(readxsize=1024, readysize=1024, n_workers=4, n_threads=4)

        # Scale the angles to degrees
        sza = solar_za * 0.01
        saa = solar_az * 0.01
        vza = sensor_za * 0.01
        vaa = sensor_az * 0.01

        # Convert to radians
        rad_sza = xr.ufuncs.deg2rad(sza)
        rad_vza = xr.ufuncs.deg2rad(vza)

        # sza.attrs = solar_za.attrs
        # saa.attrs = solar_az.attrs
        # vza.attrs = sensor_za.attrs
        # vaa.attrs = sensor_az.attrs
        # sza.gw.to_raster('/media/jcgr/data/imagery/temp/outputs/sza.tif', **kwargs)
        # saa.gw.to_raster('/media/jcgr/data/imagery/temp/outputs/saa.tif', **kwargs)
        # vza.gw.to_raster('/media/jcgr/data/imagery/temp/outputs/vza.tif', **kwargs)
        # vaa.gw.to_raster('/media/jcgr/data/imagery/temp/outputs/vaa.tif', **kwargs)

        # Cosine(deg2rad(angles)) = angles x (pi / 180)
        cos_sza = xr.ufuncs.cos(rad_sza)
        cos_vza = xr.ufuncs.cos(rad_vza)

        # cos_sza.attrs = solar_za.attrs
        # cos_vza.attrs = solar_za.attrs
        # cos_sza.gw.to_raster('/media/jcgr/data/imagery/temp/outputs/cos_sza.tif', **kwargs)
        # cos_vza.gw.to_raster('/media/jcgr/data/imagery/temp/outputs/cos_vza.tif', **kwargs)

        # air mass
        m = (1.0 / cos_sza) + (1.0 / cos_vza)

        m = xr.concat([m]*len(toar.band), dim='band')
        m.coords['band'] = toar.band.values

        # m.attrs = solar_za.attrs
        # m.gw.to_raster('/media/jcgr/data/imagery/temp/outputs/m.tif', **kwargs)

        # Rayleigh optical depth
        # Hansen, JF and Travis, LD (1974) LIGHT SCATTERING IN PLANETARY ATMOSPHERES
        r = 0.008569*um**-4 * (1.0 + 0.0113*um**-2 + 0.0013*um**-4)

        # Rayleigh phase function
        # scattering angle = the angle between the direction of incident and scattered radiation
        # Liu, CH and Liu GR (2009) AEROSOL OPTICAL DEPTH RETRIEVAL FOR SPOT HRV IMAGES, Journal of Marine Science and Technology

        # Relative azimuth angle
        # http://stcorp.github.io/harp/doc/html/algorithms/derivations/relative_azimuth_angle.html
        rel_azimuth_angle = xr.ufuncs.fabs(saa - vaa - 180.0)

        # rel_azimuth_angle.attrs = solar_za.attrs
        # rel_azimuth_angle.gw.to_raster('/media/jcgr/data/imagery/temp/outputs/rel_azimuth_angle.tif', **kwargs)

        # http://stcorp.github.io/harp/doc/html/algorithms/derivations/scattering_angle.html
        scattering_angle = xr.ufuncs.arccos(-xr.ufuncs.cos(rad_sza)*xr.ufuncs.cos(rad_vza) -
                                            xr.ufuncs.sin(rad_sza)*xr.ufuncs.sin(rad_vza) *
                                            xr.ufuncs.cos(xr.ufuncs.deg2rad(rel_azimuth_angle)))

        rphase = ((3.0 * 0.9587256) / (4.0 + 1.0 - 0.9587256)) * (1.0 + xr.ufuncs.cos(scattering_angle)**2)

        # rphase.attrs = solar_za.attrs
        # rphase.gw.to_raster('/media/jcgr/data/imagery/temp/outputs/rphase.tif', **kwargs)

        pr_data = p_r(m, r, rphase, cos_sza, cos_vza)

        # pr_data.attrs = solar_za.attrs
        # pr_data.gw.to_raster('/media/jcgr/data/imagery/temp/outputs/pr_data.tif', **kwargs)

        # da.nan_to_num(pr_data).max().compute()

        toar_diff = toar - pr_data

        total_transmission = t_s(r, cos_sza) * t_v(r, cos_vza)

        # total_transmission.attrs = solar_za.attrs
        # total_transmission.gw.to_raster('/media/jcgr/data/imagery/temp/outputs/total_transmission.tif', **kwargs)

        return toar_diff / (toar_diff * s_atm(r) + total_transmission)
