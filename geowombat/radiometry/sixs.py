"""
interpolated_LUTs.py

The Interpolated_LUTs class handles loading, downloading and interpolating
of LUTs (look up tables) used by the 6S emulator

Reference:
    https://github.com/samsammurphy/6S_emulator/blob/master/bin/interpolated_LUTs.py
"""

import os
from pathlib import Path
import logging
from collections import namedtuple
from getpass import getpass
import math

from ..handler import add_handler
from ..core import ndarray_to_xarray

import geowombat as gw

import numpy as np
import geopandas as gpd
import xarray as xr
import joblib
import requests
import yaml
from tqdm import tqdm
from cryptography.fernet import Fernet


logger = logging.getLogger(__name__)
logger = add_handler(logger)

LUTNames = namedtuple('LUTNames', 'name path')

p = Path(os.path.abspath(os.path.dirname(__file__)))
DATA_PATH = p / '../data'


def _set_names(sensor_name):

    lut_path = DATA_PATH / 'lut'

    return LUTNames(name=sensor_name,
                    path=lut_path)


SENSOR_LOOKUP = {'l5': _set_names('l5'),
                 'l7': _set_names('l7'),
                 'l8': _set_names('l8'),
                 's2a': _set_names('s2a'),
                 's2b': _set_names('s2b')}


class PassKey(object):

    @staticmethod
    def create_key(key_file):

        key = Fernet.generate_key()

        with open(key_file, mode='w') as pf:
            yaml.dump({'key': key}, pf, default_flow_style=False)

    @staticmethod
    def create_passcode(key_file, passcode_file):

        """
        Args:
            key_file (str)
            passcode_file (str)
        """

        passcode = getpass()

        with open(key_file, mode='r') as pf:
            key = yaml.load(pf, Loader=yaml.FullLoader)

        cipher_suite = Fernet(key['key'])

        ciphered_text = cipher_suite.encrypt(passcode.encode())

        with open(passcode_file, mode='w') as pf:
            yaml.dump({'passcode': ciphered_text}, pf, default_flow_style=False)

    @staticmethod
    def load_passcode(key_file, passcode_file):

        with open(key_file, mode='r') as pf:
            key = yaml.load(pf, Loader=yaml.FullLoader)

        cipher_suite = Fernet(key['key'])

        with open(passcode_file, mode='r') as pf:
            ciphered_text = yaml.load(pf, Loader=yaml.FullLoader)

        return cipher_suite.decrypt(ciphered_text['passcode'])


class EarthDataDownloader(PassKey):

    def __init__(self, username, key_file, code_file):

        self.username = username
        self.key_file = key_file
        self.code_file = code_file

        self.outpath = None

    def download(self, url, outfile):

        self.outpath = Path(outfile)

        if self.outpath.is_file():
            logger.warning(f'  The file {outfile} is already downloaded.')
            return

        base64_password = self.load_passcode(self.key_file, self.code_file).decode()

        chunk_size = 256 * 10240

        with requests.Session() as session:

            session.auth = (self.username, base64_password)

            # Open
            req = session.request('get', url)
            response = session.get(req.url, auth=(self.username, base64_password))

            if not response.ok:
                logger.exception('  Could not retrieve the page.')
                raise NameError

            if 'Content-Length' in response.headers:

                content_length = float(response.headers['Content-Length'])
                content_iters = int(math.ceil(content_length / chunk_size))
                chunk_size_ = chunk_size * 1

            else:

                content_iters = 1
                chunk_size_ = chunk_size * 1000

            with open(str(outfile), 'wb') as ofn:

                for data in tqdm(response.iter_content(chunk_size=chunk_size_), total=content_iters):
                    ofn.write(data)


class Altitude(object):

    @staticmethod
    def get_mean_altitude(data, username, key_file, code_file, outdir, n_jobs=1, delete_downloaded=False):

        srtm_df = gpd.read_file(DATA_PATH / 'srtm30m_bounding_boxes.gpkg')

        srtm_df_int = srtm_df[srtm_df.geometry.intersects(data.gw.geodataframe.to_crs(epsg=4326).geometry.values[0])]

        edd = EarthDataDownloader(username, key_file, code_file)

        hgt_files = []
        zip_paths = []

        for dfn in srtm_df_int.dataFile.values.tolist():

            zip_file = f"{outdir}/NASADEM_HGT_{dfn.split('.')[0].lower()}.zip"

            edd.download(f"https://e4ftl01.cr.usgs.gov/MEASURES/NASADEM_HGT.001/2000.02.11/NASADEM_HGT_{dfn.split('.')[0].lower()}.zip",
                         zip_file)

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
    def _load(sensor, wavelength, interp_method):

        lut_path = SENSOR_LOOKUP[sensor].path / f'{sensor}_{wavelength}.lut'

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
            ``float``

        References:
            https://py6s.readthedocs.io/en/latest/installation.html
                y = xa * (measured radiance) - xb
                acr = y / (1. + xc * y)
        """

        y = xa * rad - xb

        return y / (1.0 + xc * y)

    @staticmethod
    def prepare_coeff(band_data, coeffs, cindex):
        return ndarray_to_xarray(band_data, coeffs[:, :, cindex], ['coeff'])

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
                    .astype('float64')\
                    .clip(0, 1)

        # Create a 'no data' mask
        mask = sr.where((sr != src_nodata) & (band_data != src_nodata))\
                    .count(dim='band')\
                    .astype('uint8')

        # Mask 'no data' values
        sr = xr.where(mask < sr.gw.nbands,
                      dst_nodata,
                      sr)\
                .transpose('band', 'y', 'x')

        attrs['sensor'] = sensor
        attrs['nodata'] = dst_nodata
        attrs['calibration'] = 'surface reflectance'
        attrs['method'] = '6s radiative transfer model'
        attrs['drange'] = (0, 1)

        return sr.assign_attrs(**attrs)

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
                          altitude):

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
        """

        # Load the LUT
        lut = self._load(sensor, wavelength, interp_method)

        min_score = np.zeros(blue_rad_dark.shape, dtype='float64') + 1e9
        aot = np.zeros(blue_rad_dark.shape, dtype='float64')

        elliptical_orbit_correction = 0.03275104 * np.cos(doy / 59.66638337) + 0.96804905

        for aot_iter in np.arange(0.01, 1.1, 0.01):

            xa, xb, xc = lut(sza, h2o, o3, aot_iter, altitude)

            xa *= elliptical_orbit_correction
            xb *= elliptical_orbit_correction
            xc *= elliptical_orbit_correction

            res = self._rad_to_sr_from_coeffs(blue_rad_dark, xa, xb, xc)

            score = np.abs(res - blue_p_dark)

            aot = np.where(score < min_score, aot_iter, aot)
            min_score = np.where(score < min_score, score, min_score)

        return aot
