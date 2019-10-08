import os
import configparser


config = {}

config_file = os.path.join(os.path.dirname(__file__), 'config.ini')

config_parser = configparser.ConfigParser()

config_parser.read(config_file)

ASSOCIATIONS = {}

for section in config_parser.sections():
    for k, v in config_parser[section].items():
        ASSOCIATIONS[k] = section

for section in config_parser.sections():

    for k, v in config_parser[section].items():

        if v in ['True', 'False']:

            if v.lower() == 'true':
                config[k] = True
            else:
                config[k] = False
        else:

            try:
                config[k] = float(v)
            except:
                config[k] = v


class update(object):

    """
    >>> with gw.config.update(sensor='l8'):
    >>>
    >>>     with gw.open('image.tif') as ds:
    >>>         print(ds.gw.config)
    """

    def __init__(self, config=config, **kwargs):

        self.config = config
        self._set_defaults(config)

        if kwargs:
            self._assign(config, **kwargs)

    def __enter__(self):
        return self.config

    def __exit__(self, type, value, traceback):
        d = self.config
        self._set_defaults(d)

    @staticmethod
    def _set_defaults(d):

        config_parser.read(config_file)

        for section in config_parser.sections():

            for k, v in config_parser[section].items():

                if v in ['True', 'False']:

                    if v.lower() == 'true':
                        d[k] = True
                    else:
                        d[k] = False
                else:

                    try:
                        d[k] = float(v)
                    except:
                        d[k] = v

    @staticmethod
    def _assign(d, **kwargs):

        config_parser.read(config_file)

        if kwargs:

            for k, v in kwargs.items():
                config_parser[ASSOCIATIONS[k]][k] = str(v)

        for section in config_parser.sections():

            for k, v in config_parser[section].items():

                if v in ['True', 'False']:

                    if v.lower() == 'true':
                        d[k] = True
                    else:
                        d[k] = False
                else:

                    try:
                        d[k] = float(v)
                    except:
                        d[k] = v
