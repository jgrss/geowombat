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

LUTNames = namedtuple('LUTNames', 'base lut name zip')

p = Path(os.path.abspath(os.path.dirname(__file__)))
DATA_PATH = p / '../data'


def _set_names(sensor_name):

    s_base = DATA_PATH / 'lut' / sensor_name
    s_lut = DATA_PATH / 'lut' / sensor_name / sensor_name / 'Continental' / 'view_zenith_0'
    s_zip = DATA_PATH / 'lut' / '{}.zip'.format(sensor_name)

    return LUTNames(base=s_base,
                    lut=s_lut,
                    name=sensor_name,
                    zip=s_zip)


def interp_lut(sensors, bands):

    """
    Interpolates a list of sensors and bands

    Args:
        sensors (list): A list of sensors.
        bands (list): A list of bands.

    Returns:
        ``None``

    Example:
        >>> from geowombat.radiometry.lut import interp_lut
        >>>
        >>> interp_lut(['l8'], [2, 3, 4, 5])
    """

    for sensor in sensors:
        for band in bands:
            sxs = SixS(sensor, band, verbose=1)
            sxs.load()


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
        >>> ilut = InterpLUT('l5', 1, verbose=1)
        >>> ilut.load()
        >>>
        >>> with gw.open('image.tif') as src:
        >>>     ilut.rad_to_sr(src.sel(band=1), sza, doy, h2o=1.0, o3=0.4, aot=0.3)
    """

    def __init__(self, sensor, band, rad_scale=1.0, angle_scale=0.01, verbose=0):

        self.sensor = sensor
        self.band = band
        self.rad_scale = rad_scale
        self.angle_scale = angle_scale
        self.verbose = verbose
        self.lut = None

        self.sensor_lookup = {'l5': _set_names('LANDSAT_TM'),
                              'l7': _set_names('LANDSAT_ETM'),
                              'l8': _set_names('LANDSAT_OLI'),
                              's2': _set_names('S2A_MSI')}

        if sensor.lower() == 's2':

            lut_format = '{NAME}_B{BAND:02d}.lut'.format(NAME=self.sensor_lookup[self.sensor].name, BAND=self.band)
            ilut_format = '{NAME}_B{BAND:02d}.ilut'.format(NAME=self.sensor_lookup[self.sensor].name, BAND=self.band)

        else:

            lut_format = '{NAME}_B{BAND:d}.lut'.format(NAME=self.sensor_lookup[self.sensor].name, BAND=self.band)
            ilut_format = '{NAME}_B{BAND:d}.ilut'.format(NAME=self.sensor_lookup[self.sensor].name, BAND=self.band)

        self.lut_path = self.sensor_lookup[self.sensor].lut / lut_format
        self.ilut_path = self.sensor_lookup[self.sensor].lut / ilut_format

        self._setup()

        if not self.ilut_path.is_file():
            self.interpolate()

    def load(self, which='ilut'):
        self.lut = self._load(which)

    def rad_to_sr(self, data, sza, doy, h2o=1.0, o3=0.4, aot=0.3):

        """
        Gets 6s coefficients

        Args:
            data (DataArray): The data to correct, in radiance.
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

        attrs = data.attrs.copy()
        band_name = data.band.values

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

        ab = self.lut(sza*self.angle_scale, h2o, o3, aot, altitude)

        a = np.zeros(data.gw.nrows*data.gw.ncols, dtype='float64')
        b = np.zeros(data.gw.nrows*data.gw.ncols, dtype='float64')

        a[:] = np.nan
        b[:] = np.nan

        a[valid_idx] = ab[:, 0].flatten()
        b[valid_idx] = ab[:, 1].flatten()

        elliptical_orbit_correction = 0.03275104 * np.cos(doy / 59.66638337) + 0.96804905
        a *= elliptical_orbit_correction
        b *= elliptical_orbit_correction

        a = a.reshape(data.gw.nrows, data.gw.ncols)
        b = b.reshape(data.gw.nrows, data.gw.ncols)

        a = ndarray_to_xarray(data, a, ['coeff'])
        b = ndarray_to_xarray(data, b, ['coeff'])

        return ((data*self.rad_scale - a.sel(band='coeff')) / b.sel(band='coeff'))\
                    .expand_dims(dim='band')\
                    .assign_coords(coords={'band': [band_name]})\
                    .assign_attrs(**attrs)

    def _setup(self):

        if not (DATA_PATH / 'lut.tar.gz').is_dir():

            with tarfile.open(DATA_PATH / 'lut.tar.gz', mode='r:gz') as tf:
                tf.extractall(path=DATA_PATH)

        if not self.sensor_lookup[self.sensor].base.is_dir():

            with zipfile.ZipFile(self.sensor_lookup[self.sensor].zip, mode='r') as uzip:
                uzip.extractall(path=self.sensor_lookup[self.sensor].base)

    def _load(self, which):

        lut_path_ = self.lut_path if which == 'lut' else self.ilut_path

        return joblib.load(lut_path_)

    def _dump(self, interp_obj):

        with open(self.ilut_path, 'wb') as lut_file:
            pickle.dump(interp_obj, lut_file)

    def interpolate(self):

        """
        Interpolates look-up tables
        """

        if self.verbose > 0:
            logger.info('  Interpolating the LUT ...')

        lut_dict = self._load('lut')

        in_vars = lut_dict['config']['invars']

        # all permutations
        inputs = list(itertools.product(in_vars['solar_zs'],
                                        in_vars['H2Os'],
                                        in_vars['O3s'],
                                        in_vars['AOTs'],
                                        in_vars['alts']))

        # output variables (6S correction coefficients)
        outputs = lut_dict['outputs']

        interpolator = LinearNDInterpolator(inputs, outputs)

        self._dump(interpolator)

        if self.verbose > 0:

            logger.info('  Finished interpolating for sensor {SENSOR}, band {BAND}.'.format(SENSOR=self.sensor,
                                                                                            BAND=self.band))
