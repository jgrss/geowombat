import math
from collections import namedtuple
from datetime import datetime as dtime
import datetime

from .angles import relative_azimuth

import numpy as np
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

    bands_dict = dict(l5={'1': 'blue', '2': 'green', '3': 'red', '4': 'nir', '5': 'swir1', '6': 'swir2'},
                      l7={'1': 'blue', '2': 'green', '3': 'red', '4': 'nir', '5': 'swir1', '6VCID1': 'th1',
                          '6VCID2': 'th2', '7': 'swir2', '8': 'pan'},
                      l8={'1': 'coastal', '2': 'blue', '3': 'green', '4': 'red', '5': 'nir', '6': 'swir1',
                          '7': 'swir2', '8': 'pan', '9': 'cirrus', '10': 'th1', '11': 'th2'})

    sensor_dict = bands_dict[sensor]

    dataframe_ = dataframe[dataframe.iloc[:, 0].str.startswith(key)].values

    pairs = dict()

    for di in range(dataframe_.shape[0]):

        bd = dataframe_[di, 0]
        cf = dataframe_[di, 1]

        try:
            pairs[sensor_dict[''.join(bd.split('_')[3:])]] = float(cf)
        except:
            pass

    # dataframe_[:, 1] = dataframe_[:, 1].astype(float)
    # dataframe_[:, 0] = list(range(1, dataframe_.shape[0]+1))

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
            ``namedtuple``
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
                 band_names=None):

        """
        Applies a bandpass adjustment by applying a linear function to surface reflectance values

        Args:
            data (DataArray): The data to adjust.
            sensor (Optional[str]): The sensor to adjust.
            to (Optional[str]): The sensor to adjust to.
            band_names (Optional[list]): The bands to adjust. If not given, all bands are adjusted.

        Reference:

            Sentinel-2 and Landsat 8:

                https://hls.gsfc.nasa.gov/algorithms/bandpass-adjustment/

                See :cite:`chastain_etal_2019` for further details

            Landsat 7 and Landsat 8:

                See :cite:`roy_etal_2016` (Table 2)

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

        Returns:
            ``xarray.DataArray``
        """

        attrs = data.attrs.copy()

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

        data.attrs['adjustment'] = '{} to {}'.format(sensor, to)
        data.attrs['alphas'] = alphas.data.compute().tolist()
        data.attrs['betas'] = betas.data.compute().tolist()

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
                 nodata=-32768,
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
            nodata (Optional[int or float]): The 'no data' value from the pixel angle data.
            sensor (Optional[str]): The data's sensor.
            method (Optional[str]): The method to use. Only 'srem' is supported.
            angle_factor (Optional[float]): The scale factor for angles.
            meta (Optional[namedtuple]): A metadata object with gain and bias coefficients.

        References:
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

        attrs = dn.attrs.copy()

        # Get the data band names and positional indices
        band_names = dn.band.values.tolist()

        if meta:

            # Get the sensor wavelengths
            # wavelengths = dn.gw.wavelengths[meta.sensor]

            # band_indices = [getattr(wavelengths, p) for p in band_names]

            # Get the gain and offsets and
            #   convert the gain and offsets
            #   to named coordinates.
            m_p = xr.DataArray(data=[meta.m_p[bi] for bi in band_names], coords={'band': band_names}, dims='band')
            a_p = xr.DataArray(data=[meta.a_p[bi] for bi in band_names], coords={'band': band_names}, dims='band')

            toar = self.dn_to_toar(dn, m_p, a_p)

            # TOAR with sun angle correction
            cos_sza = xr.concat([xr.ufuncs.cos(xr.ufuncs.deg2rad(solar_za*angle_factor))] * len(toar.band), dim='band')
            cos_sza.coords['band'] = toar.band.values
            toar = toar / cos_sza
            toar.attrs = attrs

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
                                  meta.sensor,
                                  nodata=nodata)

        sr_data = sr_data.where(sr_data != nodata)

        attrs['sensor'] = sensor
        attrs['nodata'] = nodata
        attrs['calibration'] = 'surface reflectance'
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

        attrs = dn.attrs.copy()

        # TODO: get gain and bias from metadata
        rad_data = gain * dn + bias

        attrs['calibration'] = 'radiance'

        rad_data.attrs = attrs

        return rad_data

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

        attrs = dn.attrs.copy()

        # TODO: get gain and bias from metadata
        toar_data = gain * dn + bias

        attrs['calibration'] = 'top-of-atmosphere reflectance'

        toar_data.attrs = attrs

        return toar_data

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
                   nodata=-32768):

        """
        Converts top-of-atmosphere reflectance to surface reflectance

        Args:
            toar (DataArray): The top-of-atmosphere reflectance.
            solar_za (DataArray): The solar zenith angle.
            solar_az (DataArray): The solar azimuth angle.
            sensor_za (DataArray): The sensor zenith angle.
            sensor_az (DataArray): The sensor azimuth angle.
            sensor (str): The satellite sensor.
            nodata (Optional[int or float]): The 'no data' value from the pixel angle data.

        References:

            See :cite:`bilal_etal_2019`

        Returns:
            ``xarray.DataArray``
        """

        attrs = toar.attrs.copy()

        central_um = toar.gw.central_um[sensor]
        band_names = list(toar.gw.wavelengths[sensor]._fields)
        band_um = [getattr(central_um, p) for p in band_names]
        um = xr.DataArray(data=band_um, coords={'band': band_names}, dims='band')

        # Scale the angles to degrees
        sza = solar_za * 0.01
        saa = solar_az * 0.01
        vza = sensor_za * 0.01
        vaa = sensor_az * 0.01

        # Convert to radians
        rad_sza = xr.ufuncs.deg2rad(sza)
        rad_vza = xr.ufuncs.deg2rad(vza)

        # Cosine(deg2rad(angles)) = angles x (pi / 180)
        cos_sza = xr.ufuncs.cos(rad_sza)
        cos_vza = xr.ufuncs.cos(rad_vza)

        sin_sza = xr.ufuncs.sin(rad_sza)
        sin_vza = xr.ufuncs.sin(rad_vza)

        # air mass
        m = (1.0 / cos_sza) + (1.0 / cos_vza)

        m = xr.concat([m]*len(toar.band), dim='band')
        m.coords['band'] = toar.band.values

        # Rayleigh optical depth
        # Hansen, JF and Travis, LD (1974) LIGHT SCATTERING IN PLANETARY ATMOSPHERES
        r = 0.008569*um**-4 * (1.0 + 0.0113*um**-2 + 0.0013*um**-4)

        # Relative azimuth angle
        raa = relative_azimuth(saa, vaa)
        rad_raa = xr.ufuncs.deg2rad(raa)
        cos_raa = xr.ufuncs.cos(rad_raa)

        # scattering angle = the angle between the direction of incident and scattered radiation
        # Liu, CH and Liu GR (2009) AEROSOL OPTICAL DEPTH RETRIEVAL FOR SPOT HRV IMAGES, Journal of Marine Science and Technology
        # http://stcorp.github.io/harp/doc/html/algorithms/derivations/scattering_angle.html
        scattering_angle = xr.ufuncs.arccos(-cos_sza * cos_vza - sin_sza * sin_vza * cos_raa)
        cos2_scattering_angle = xr.ufuncs.cos(scattering_angle)**2

        # Rayleigh phase function
        rphase = ((3.0 * 0.9587256) / (4.0 + 1.0 - 0.9587256)) * (1.0 + cos2_scattering_angle)

        pr_data = p_r(m, r, rphase, cos_sza, cos_vza)

        # da.nan_to_num(pr_data).max().compute()

        toar_diff = toar - pr_data

        # Total transmission = downward x upward
        transmission = t_sv(r, cos_sza) * t_sv(r, cos_vza)

        # Atmospheric backscattering ratio
        ab_ratio = s_atm(r)

        sr_data = (toar_diff / (toar_diff * ab_ratio + transmission)).fillna(nodata).clip(0, 1).astype('float64')

        attrs['sensor'] = sensor
        attrs['calibration'] = 'surface reflectance'
        attrs['nodata'] = nodata
        attrs['drange'] = (0, 1)

        sr_data.attrs = attrs

        return sr_data
