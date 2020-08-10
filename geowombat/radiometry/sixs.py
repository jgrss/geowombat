"""
interpolated_LUTs.py

The Interpolated_LUTs class handles loading, downloading and interpolating
of LUTs (look up tables) used by the 6S emulator

Reference:
    https://github.com/samsammurphy/6S_emulator/blob/master/bin/interpolated_LUTs.py
"""

import os
from pathlib import Path
import pickle
import zipfile
import tarfile
import itertools
import logging
from collections import namedtuple

from ..handler import add_handler
from ..core import ndarray_to_xarray

import numpy as np
from scipy.interpolate import LinearNDInterpolator
import xarray as xr
import joblib


logger = logging.getLogger(__name__)
logger = add_handler(logger)

LUTNames = namedtuple('LUTNames', 'name path')

p = Path(os.path.abspath(os.path.dirname(__file__)))
DATA_PATH = p / '../data'


def _set_names(sensor_name):

    lut_path = DATA_PATH / 'lut' / '{}.bz2'.format(sensor_name)

    return LUTNames(name=sensor_name,
                    path=lut_path)


class SixS(object):

    """
    A class to handle loading, downloading and interpolating
    of LUTs (look up tables) used by the 6S emulator.

    Args:
        sensor (str): The sensor to adjust.
        band (int): The band to adjust.
        rad_scale (Optional[float]): The radiance scale factor. Scaled values should be in the range [0,1000].
        angle_scale (Optional[float]): The angle scale factor.
        verbose (Optional[int]): The verbosity level.

    Example:
        >>> sixs = SixS('l5', verbose=1)
        >>>
        >>> with gw.config.update(sensor='l7'):
        >>>     with gw.open('image.tif') as src, gw.open('solar_za') as sza:
        >>>         sixs.rad_to_sr(src, 'blue', sza, doy, h2o=1.0, o3=0.4, aot=0.3)
    """

    def __init__(self, sensor, rad_scale=1.0, angle_scale=0.01, verbose=0):

        self.sensor = sensor
        self.rad_scale = rad_scale
        self.angle_scale = angle_scale
        self.verbose = verbose

        self.lut = None

        self.sensor_lookup = {'l5': _set_names('l5'),
                              'l7': _set_names('l7'),
                              'l8': _set_names('l8'),
                              's2': _set_names('s2')}

        # Load the lookup tables
        self._load()

    def _load(self):
        self.lut = joblib.load(str(self.sensor_lookup[self.sensor].path))

    def rad_to_sr(self, data, band, sza, doy, h2o=1.0, o3=0.4, aot=0.3):

        """
        Gets 6s coefficients

        Args:
            data (DataArray): The data to correct, in radiance.
            band (str): The band wavelength to process.
            sza (float | DataArray): The solar zenith angle.
            doy (int): The day of year.
            h2o (Optional[float]): The water vapor (g/m^2). [0,8.5].
            o3 (Optional[float]): The ozone (cm-atm). [0,8].
            aot (Optional[float | DataArray]): The aerosol optical thickness (unitless). [0,3].

        Returns:
            ``xarray.DataArray``
        """

        if not self.lut:
            logger.exception('  The lookup table does not exist. Be sure to load it.')
            raise NameError

        if band not in self.lut:
            logger.exception('  The band {} does not exist in the LUT.'.format(band))

        band_interp = self.lut[band]

        attrs = data.attrs.copy()

        band_data = data.sel(band=band)

        altitude = 0.0

        if isinstance(sza, xr.DataArray):
            sza = sza.squeeze().data.compute()

        if isinstance(aot, xr.DataArray):
            aot = aot.squeeze().data.compute()

        if isinstance(sza, np.ndarray):
            sza = sza.squeeze().flatten()

        if isinstance(aot, np.ndarray):
            aot = aot.squeeze().flatten()

        if isinstance(sza, np.ndarray) and isinstance(aot, np.ndarray):
            valid_idx = np.where(~np.isnan(aot) & ~np.isnan(sza))
            sza = sza[valid_idx]
            aot = aot[valid_idx]
        elif isinstance(sza, np.ndarray) and not isinstance(aot, np.ndarray):
            valid_idx = np.where(~np.isnan(sza))
            sza = sza[valid_idx]
        elif not isinstance(sza, np.ndarray) and isinstance(aot, np.ndarray):
            valid_idx = np.where(~np.isnan(aot))
            aot = aot[valid_idx]

        ab = band_interp(sza*self.angle_scale, h2o, o3, aot, altitude)

        a = np.zeros(band_data.gw.nrows*band_data.gw.ncols, dtype='float64')
        b = np.zeros(band_data.gw.nrows*band_data.gw.ncols, dtype='float64')

        a[:] = np.nan
        b[:] = np.nan

        a[valid_idx] = ab[:, 0].flatten()
        b[valid_idx] = ab[:, 1].flatten()

        elliptical_orbit_correction = 0.03275104 * np.cos(doy / 59.66638337) + 0.96804905
        a *= elliptical_orbit_correction
        b *= elliptical_orbit_correction

        a = a.reshape(band_data.gw.nrows, band_data.gw.ncols)
        b = b.reshape(band_data.gw.nrows, band_data.gw.ncols)

        a = ndarray_to_xarray(band_data, a, ['coeff'])
        b = ndarray_to_xarray(band_data, b, ['coeff'])

        return ((band_data*self.rad_scale - a.sel(band='coeff')) / b.sel(band='coeff'))\
                    .expand_dims(dim='band')\
                    .assign_coords(coords={'band': [band]})\
                    .assign_attrs(**attrs)

    @staticmethod
    def get_optimized_aot(blue_rad_dark, blue_p_dark, sensor, blue_band, meta, h2o, o3):

        sxs = SixS(sensor, blue_band)
        sxs.load()

        doy = meta.date_acquired.timetuple().tm_yday
        altitude = 0.0

        min_score = np.zeros(blue_rad_dark.shape, dtype='float64') + 1e9
        aot = np.zeros(blue_rad_dark.shape, dtype='float64')

        for aot_iter in np.linspace(0.1, 1.0, 10):

            # sza, h2o, o3, aot, alt
            a, b = sxs.lut(meta.sza, h2o, o3, aot_iter, altitude)

            elliptical_orbit_correction = 0.03275104 * math.cos(doy / 59.66638337) + 0.96804905
            a *= elliptical_orbit_correction
            b *= elliptical_orbit_correction

            res = (blue_rad_dark - a) / b

            score = np.abs(res - blue_p_dark)

            aot = np.where(score < min_score, aot_iter, aot)
            min_score = np.where(score < min_score, score, min_score)

        return aot

    # def interpolate(self):
    #
    #     """
    #     Interpolates look-up tables
    #     """
    #
    #     if self.verbose > 0:
    #         logger.info('  Interpolating the LUT ...')
    #
    #     lut_dict = self._load('lut')
    #
    #     in_vars = lut_dict['config']['invars']
    #
    #     # all permutations
    #     inputs = list(itertools.product(in_vars['solar_zs'],
    #                                     in_vars['H2Os'],
    #                                     in_vars['O3s'],
    #                                     in_vars['AOTs'],
    #                                     in_vars['alts']))
    #
    #     # output variables (6S correction coefficients)
    #     outputs = lut_dict['outputs']
    #
    #     interpolator = LinearNDInterpolator(inputs, outputs)
    #
    #     self._dump(interpolator)
    #
    #     if self.verbose > 0:
    #
    #         logger.info('  Finished interpolating for sensor {SENSOR}, band {BAND}.'.format(SENSOR=self.sensor,
    #                                                                                         BAND=self.band))
