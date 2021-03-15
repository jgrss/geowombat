"""
interpolated_LUTs.py

The Interpolated_LUTs class handles loading, downloading and interpolating
of LUTs (look up tables) used by the 6S emulator

Reference:
    https://github.com/samsammurphy/6S_emulator/blob/master/bin/interpolated_LUTs.py
"""

import shutil
import string
import random
from pathlib import Path
import logging
from collections import namedtuple

from ..handler import add_handler
from ..core import ndarray_to_xarray
from ..data import LUTDownloader, NASAEarthdataDownloader
from ..data import srtm30m_bounding_boxes

import geowombat as gw

import numpy as np
import geopandas as gpd
import xarray as xr
import joblib


logger = logging.getLogger(__name__)
logger = add_handler(logger)

LUTNames = namedtuple('LUTNames', 'name path')

DATA_PATH = Path.home() / '.geowombat/datasets'


def _set_names(sensor_name):

    lut_path = DATA_PATH / 'lut'

    lut_path.mkdir(parents=True, exist_ok=True)

    return LUTNames(name=sensor_name,
                    path=lut_path)


SENSOR_LOOKUP = {'l5': _set_names('l5'),
                 'l7': _set_names('l7'),
                 'l8': _set_names('l8'),
                 's2a': _set_names('s2a'),
                 's2b': _set_names('s2b')}


def _random_id(string_length):

    """
    Generates a random string of letters and digits
    """

    letters_digits = string.ascii_letters + string.digits

    return ''.join(random.choice(letters_digits) for i in range(string_length))


class Altitude(object):

    @staticmethod
    def get_mean_altitude(data,
                          out_dir,
                          username=None,
                          key_file=None,
                          code_file=None,
                          n_jobs=1,
                          delete_downloaded=False):

        if not username:
            username = data.gw.nasa_earthdata_user

        if not key_file:
            key_file = data.gw.nasa_earthdata_key

        if not code_file:
            code_file = data.gw.nasa_earthdata_code

        if not username or not key_file or not code_file:
            logger.exception('  The NASA EarthData username, secret key file, and secret code file must be provided to download SRTM data.')
            raise AttributeError

        if not Path(out_dir).is_dir():
            Path(out_dir).mkdir(parents=True, exist_ok=True)

        srtm_grid_path_temp = Path(out_dir) / f'srtm30m_bounding_boxes_{_random_id(9)}.gpkg'

        shutil.copy(str(srtm30m_bounding_boxes), str(srtm_grid_path_temp))

        srtm_df = gpd.read_file(srtm_grid_path_temp)

        srtm_grid_path_temp.unlink()

        srtm_df_int = srtm_df[srtm_df.geometry.intersects(data.gw.geodataframe.to_crs(epsg=4326).geometry.values[0])]

        nedd = NASAEarthdataDownloader(username, key_file, code_file)

        hgt_files = []
        zip_paths = []

        for dfn in srtm_df_int.dataFile.values.tolist():

            zip_file = f"{out_dir}/NASADEM_HGT_{dfn.split('.')[0].lower()}.zip"

            nedd.download_srtm(dfn.split('.')[0].lower(), zip_file)

            src_zip = f"zip+file://{zip_file}!/{Path(zip_file).stem.split('_')[-1]}.hgt"

            hgt_files.append(Path(zip_file))
            zip_paths.append(src_zip)

        if len(zip_paths) == 1:
            zip_paths = zip_paths[0]
            mosaic = False
        else:
            mosaic = True

        with gw.open(zip_paths, mosaic=mosaic) as src:

            mean_elev = src.transpose('band', 'y', 'x')\
                                .mean().data\
                                .compute(num_workers=n_jobs)

        if delete_downloaded:

            for fn in hgt_files:
                fn.unlink()

        return mean_elev


class SixS(Altitude):

    """
    A class to handle loading, downloading and interpolating
    of LUTs (look up tables) used by the 6S emulator.

    Args:
        sensor (str): The sensor to adjust.
        rad_scale (Optional[float]): The radiance scale factor. Scaled values should be in the range [0,1000].
        angle_factor (Optional[float]): The angle scale factor.

    Example:
        >>> sixs = SixS('l5', verbose=1)
        >>>
        >>> with gw.config.update(sensor='l7'):
        >>>     with gw.open('image.tif') as src, gw.open('solar_za') as sza:
        >>>         sixs.rad_to_sr(src, 'blue', sza, doy, h2o=1.0, o3=0.4, aot=0.3)
    """

    @staticmethod
    def _load(sensor, wavelength, interp_method, from_toar=False):

        if from_toar:
            raise NotImplementedError('Lookup tables from top of atmosphere reflectance are not supported.')
            lut_path = SENSOR_LOOKUP[sensor].path / f'{sensor}_{wavelength}_from_toar.lut'
        else:
            lut_path = SENSOR_LOOKUP[sensor].path / f'{sensor}_{wavelength}.lut'

        if not lut_path.is_file():

            logger.info(f'  Downloading {lut_path.name} into {SENSOR_LOOKUP[sensor].path}.')

            lutd = LUTDownloader()

            lutd.download(f'https://s3geowombat.s3.amazonaws.com/{sensor}_{wavelength}.lut',
                          str(lut_path),
                          safe_download=False)

        lut_ = joblib.load(str(lut_path))

        return lut_[interp_method]

    @staticmethod
    def _rad_to_sr_from_coeffs(rad, xa, xb, xc):

        """
        Transforms radiance to surface reflectance using 6S coefficients

        Args:
            rad (float | DataArray): The radiance.
            xa (float | DataArray): The inverse of the transmittance.
            xb (float | DataArray): The scattering term of the atmosphere.
            xc (float | DataArray): The spherical albedo (atmospheric reflectance for isotropic light).

        Returns:
            ``float`` | ``xarray.DataArray``

        References:
            https://py6s.readthedocs.io/en/latest/installation.html
                y = xa * (measured radiance) - xb
                acr = y / (1. + xc * y)
        """

        y = xa * rad - xb

        return y / (1.0 + xc * y)

    @staticmethod
    def _toar_to_sr_from_coeffs(toar, t_g, p_alpha, s, t_s, t_v):

        """
        Transforms top of atmosphere reflectance to surface reflectance using 6S coefficients

        Args:
            toar (float | DataArray): The top of atmosphere reflectance.
            t_g (float): The total gaseous transmission of the atmosphere.
            p_alpha (float): The atmospheric reflectance.
            s (float): The spherical albedo of the atmosphere.
            t_s (float): The atmospheric transmittance from sun to target.
            t_v (float): The atmospheric transmittance from target to satellite.

        Returns:
            ``float`` | ``xarray.DataArray``
        """

        sr_s = ((toar / t_g) - p_alpha) / (t_s * t_v)

        return sr_s / (1.0 + s * sr_s)

    @staticmethod
    def prepare_coeff(band_data, coeffs, cindex):
        return ndarray_to_xarray(band_data, coeffs[:, :, cindex], ['coeff'])

    @staticmethod
    def _mask_nodata(data, other_data, src_nodata, dst_nodata):

        # Create a 'no data' mask
        mask = data.where((data != src_nodata) & (other_data != src_nodata))\
                        .count(dim='band')\
                        .astype('uint8')

        # Mask 'no data' values
        return xr.where(mask < data.gw.nbands,
                        dst_nodata,
                        data.clip(0, 1))\
                    .transpose('band', 'y', 'x')

    def toar_to_sr(self,
                   data,
                   sensor,
                   wavelength,
                   sza,
                   doy,
                   src_nodata=-32768,
                   dst_nodata=-32768,
                   angle_factor=0.01,
                   interp_method='fast',
                   h2o=1.0,
                   o3=0.4,
                   aot=0.3,
                   altitude=0.0,
                   n_jobs=1):

        """
        Converts top of atmosphere reflectance to surface reflectance using 6S outputs

        Args:
            data (DataArray): The top of atmosphere reflectance.
            sensor (str): The sensor name.
            wavelength (str): The band wavelength to process.
            sza (float | DataArray): The solar zenith angle.
            doy (int): The day of year.
            src_nodata (Optional[int or float]): The input 'no data' value.
            dst_nodata (Optional[int or float]): The output 'no data' value.
            angle_factor (Optional[float]): The scale factor for angles.
            interp_method (Optional[str]): The LUT interpolation method. Choices are ['fast', 'slow'].
                'fast': Uses nearest neighbor lookup with ``scipy.interpolate.NearestNDInterpolator``.
                'slow': Uses linear interpolation with ``scipy.interpolate.LinearNDInterpolator``.
            h2o (Optional[float]): The water vapor (g/m^2). [0,8.5].
            o3 (Optional[float]): The ozone (cm-atm). [0,8].
            aot (Optional[float | DataArray]): The aerosol optical thickness (unitless). [0,3].
            altitude (Optional[float]): The altitude over the sensor acquisition location.
            n_jobs (Optional[int]): The number of parallel jobs for ``dask.compute``.

        6S model outputs:
            t_g (float): The total gaseous transmission of the atmosphere.
                s.run() --> s.outputs.total_gaseous_transmittance
            p_alpha (float): The atmospheric reflectance.
                s.run() --> s.outputs.atmospheric_intrinsic_reflectance
            s (float): The spherical albedo of the atmosphere.
                s.run() --> s.outputs.spherical_albedo
            t_s (float): The atmospheric transmittance from sun to target.
                s.run() --> s.outputs.transmittance_total_scattering.downward
            t_v (float): The atmospheric transmittance from target to satellite.
                s.run() --> s.outputs.transmittance_total_scattering.upward
        """

        attrs = data.attrs.copy()

        # Load the LUT
        lut = self._load(sensor, wavelength, interp_method, from_toar=True)

        band_data = data.sel(band=wavelength)

        if isinstance(sza, xr.DataArray):
            sza = sza.squeeze().astype('float64').data.compute(num_workers=n_jobs)

        sza *= angle_factor

        if not isinstance(aot, xr.DataArray):
            aot = xr.zeros_like(data[0]).squeeze() + aot

        # t_g, p_alpha, s, t_s, t_v
        coeffs = lut(sza, h2o, o3, aot, altitude)

        elliptical_orbit_correction = 0.03275104 * np.cos(doy / 59.66638337) + 0.96804905

        coeffs *= elliptical_orbit_correction

        t_g = self.prepare_coeff(band_data, coeffs, 0)
        p_alpha = self.prepare_coeff(band_data, coeffs, 1)
        s = self.prepare_coeff(band_data, coeffs, 2)
        t_s = self.prepare_coeff(band_data, coeffs, 3)
        t_v = self.prepare_coeff(band_data, coeffs, 4)

        sr = self._toar_to_sr_from_coeffs(band_data,
                                          t_g.sel(band='coeff'),
                                          p_alpha.sel(band='coeff'),
                                          s.sel(band='coeff'),
                                          t_s.sel(band='coeff'),
                                          t_v.sel(band='coeff'))\
                    .fillna(src_nodata)\
                    .expand_dims(dim='band')\
                    .assign_coords(coords={'band': [wavelength]})\
                    .astype('float64')

        sr = self._mask_nodata(sr, band_data, src_nodata, dst_nodata)

        attrs['sensor'] = sensor
        attrs['nodata'] = dst_nodata
        attrs['calibration'] = 'surface reflectance'
        attrs['method'] = '6s radiative transfer model'
        attrs['drange'] = (0, 1)

        return sr.assign_attrs(**attrs)

    def rad_to_sr(self,
                  data,
                  sensor,
                  wavelength,
                  sza,
                  doy,
                  src_nodata=-32768,
                  dst_nodata=-32768,
                  angle_factor=0.01,
                  interp_method='fast',
                  h2o=1.0,
                  o3=0.4,
                  aot=0.3,
                  altitude=0.0,
                  n_jobs=1):

        """
        Converts radiance to surface reflectance using a 6S radiative transfer model lookup table

        Args:
            data (DataArray): The data to correct, in radiance.
            sensor (str): The sensor name.
            wavelength (str): The band wavelength to process.
            sza (float | DataArray): The solar zenith angle.
            doy (int): The day of year.
            src_nodata (Optional[int or float]): The input 'no data' value.
            dst_nodata (Optional[int or float]): The output 'no data' value.
            angle_factor (Optional[float]): The scale factor for angles.
            interp_method (Optional[str]): The LUT interpolation method. Choices are ['fast', 'slow'].
                'fast': Uses nearest neighbor lookup with ``scipy.interpolate.NearestNDInterpolator``.
                'slow': Uses linear interpolation with ``scipy.interpolate.LinearNDInterpolator``.
            h2o (Optional[float]): The water vapor (g/m^2). [0,8.5].
            o3 (Optional[float]): The ozone (cm-atm). [0,8].
            aot (Optional[float | DataArray]): The aerosol optical thickness (unitless). [0,3].
            altitude (Optional[float]): The altitude over the sensor acquisition location.
            n_jobs (Optional[int]): The number of parallel jobs for ``dask.compute``.

        Returns:

            ``xarray.DataArray``:

                Data range: 0-1
        """

        # Load the LUT
        lut = self._load(sensor, wavelength, interp_method)

        attrs = data.attrs.copy()

        band_data = data.sel(band=wavelength)

        if isinstance(sza, xr.DataArray):
            sza = sza.squeeze().astype('float64').data.compute(num_workers=n_jobs)

        if isinstance(aot, xr.DataArray):
            aot = aot.squeeze().astype('float64').data.compute(num_workers=n_jobs)

        sza *= angle_factor

        coeffs = lut(sza, h2o, o3, aot, altitude)

        elliptical_orbit_correction = 0.03275104 * np.cos(doy / 59.66638337) + 0.96804905

        coeffs *= elliptical_orbit_correction

        xa = self.prepare_coeff(band_data, coeffs, 0)
        xb = self.prepare_coeff(band_data, coeffs, 1)
        xc = self.prepare_coeff(band_data, coeffs, 2)

        sr = self._rad_to_sr_from_coeffs(band_data,
                                         xa.sel(band='coeff'),
                                         xb.sel(band='coeff'),
                                         xc.sel(band='coeff'))\
                    .fillna(src_nodata)\
                    .expand_dims(dim='band')\
                    .assign_coords(coords={'band': [wavelength]})\
                    .astype('float64')

        sr = self._mask_nodata(sr, band_data, src_nodata, dst_nodata)

        attrs['sensor'] = sensor
        attrs['nodata'] = dst_nodata
        attrs['calibration'] = 'surface reflectance'
        attrs['method'] = '6s radiative transfer model'
        attrs['drange'] = (0, 1)

        return sr.assign_attrs(**attrs)


class AOT(object):

    def get_optimized_aot(self,
                          blue_rad_dark,
                          blue_p_dark,
                          sensor,
                          wavelength,
                          interp_method,
                          sza,
                          doy,
                          h2o,
                          o3,
                          altitude,
                          max_aot=0.5):

        """
        Gets the optimal aerosol optical thickness

        Args:
            blue_rad_dark (DataArray)
            blue_p_dark (DataArray)
            sensor (str)
            wavelength (str)
            interp_method (str)
            sza (float): The solar zenith angle (in degrees).
            doy (int): The day of year.
            h2o (float): The water vapor (g/m^2). [0,8.5].
            o3 (float): The ozone (cm-atm). [0,8].
            altitude (float)
            max_aot (float)
        """

        # Load the LUT
        lut = self._load(sensor, wavelength, interp_method)

        min_score = np.zeros(blue_rad_dark.shape, dtype='float64') + 1e9
        aot = np.zeros(blue_rad_dark.shape, dtype='float64')

        elliptical_orbit_correction = 0.03275104 * np.cos(doy / 59.66638337) + 0.96804905

        for aot_iter in np.arange(0.01, max_aot+0.01, 0.01):

            xa, xb, xc = lut(sza, h2o, o3, aot_iter, altitude)

            xa *= elliptical_orbit_correction
            xb *= elliptical_orbit_correction
            xc *= elliptical_orbit_correction

            res = self._rad_to_sr_from_coeffs(blue_rad_dark, xa, xb, xc)

            score = np.abs(res - blue_p_dark)

            aot = np.where(score < min_score, aot_iter, aot)
            min_score = np.where(score < min_score, score, min_score)

        return aot.clip(0, max_aot)
