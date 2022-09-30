import logging
from contextlib import contextmanager

from ..handler import add_handler

logger = logging.getLogger(__name__)
logger = add_handler(logger)


@contextmanager
def _executor_dummy(processes=1):
    yield None


@contextmanager
def _cluster_dummy(**kwargs):
    yield None


@contextmanager
def _client_dummy(**kwargs):
    yield None


class PropertyMixin(object):
    @staticmethod
    def check_sensor(data, sensor=None, return_error=True):
        """Checks if a sensor name is provided."""
        if sensor is None:
            if hasattr(data.gw, 'sensor'):
                sensor = data.gw.sensor
            else:
                if return_error:
                    logger.exception('  A sensor must be provided.')

        return sensor

    @staticmethod
    def check_sensor_band_names(data, sensor, band_names):

        """Checks if band names can be collected from a sensor's wavelength
        names."""

        if not band_names:

            if isinstance(sensor, str):
                band_names = list(data.gw.wavelengths[sensor]._fields)
            elif isinstance(data.gw.sensor, str):
                band_names = list(data.gw.wavelengths[data.gw.sensor]._fields)

        if not band_names:
            band_names = list(range(1, data.gw.nbands + 1))

        return band_names
