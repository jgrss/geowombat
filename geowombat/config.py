import os
from contextlib import contextmanager
import configparser


config = {}

ASSOCIATIONS = {'sensor': 'satellite'}


class set(object):

    """
    Args:

    >>> with gw.config.set(sensor='l8'):
    >>>
    >>>     with gw.open('image.tif') as ds:
    >>>         print(ds.gw.sensor)
    """

    def __init__(self, config=config, **kwargs):

        self.config_file = os.path.join(os.path.dirname(__file__), 'config.ini')

        self.config = config
        self._set_defaults(config)

        if kwargs:
            self._assign(config, **kwargs)

    def __enter__(self):
        return self.config

    def __exit__(self, type, value, traceback):
        d = self.config
        self._set_defaults(d)

    def _set_defaults(self, d):

        config_parser = configparser.ConfigParser()

        config_parser.read(self.config_file)

        for section in config_parser.sections():

            for k, v in config_parser[section].items():
                d[k] = v

    def _assign(self, d, **kwargs):

        config_parser = configparser.ConfigParser()

        config_parser.read(self.config_file)

        if kwargs:

            for k, v in kwargs.items():
                config_parser[ASSOCIATIONS[k]][k] = v

        for section in config_parser.sections():

            for k, v in config_parser[section].items():
                d[k] = v
