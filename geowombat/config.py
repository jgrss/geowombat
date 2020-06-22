import os
import ast
import configparser


config = {}
_config_under_context = False

config_file = os.path.join(os.path.dirname(__file__), 'config.ini')

config_parser = configparser.ConfigParser()

config_parser.read(config_file)

ASSOCIATIONS = {}

for section in config_parser.sections():
    for k, v in config_parser[section].items():
        ASSOCIATIONS[k] = section


def _update_config(config_parser, config_dict):

    for section in config_parser.sections():

        for k, v in config_parser[section].items():

            if v in ['True', 'False']:
                config_dict[k] = True if v == 'True' else False

            elif v == 'None':
                config_dict[k] = None
            else:

                try:
                    config_dict[k] = ast.literal_eval(v)
                except:
                    config_dict[k] = v

    return config_dict


config = _update_config(config_parser, config)


def _set_defaults(d):

    config_parser.read(config_file)
    d = _update_config(config_parser, d)


class update(object):

    """
    >>> with gw.config.update(sensor='l8'):
    >>>
    >>>     with gw.open('image.tif') as ds:
    >>>         print(ds.gw.config)
    """

    def __init__(self, config=config, **kwargs):

        self.config = config
        self._config_under_context = _config_under_context
        self.__set_defaults(config)

        if kwargs:
            self._assign(config, **kwargs)

    def __enter__(self):
        self._config_under_context = True
        return self.config

    def __exit__(self, type, value, traceback):
        d = self.config
        self._config_under_context = False
        self.__set_defaults(d)

    @staticmethod
    def __set_defaults(d):
        _set_defaults(d)

    @staticmethod
    def _assign(d, **kwargs):

        if kwargs:

            for k, v in kwargs.items():
                config_parser[ASSOCIATIONS[k]][k] = str(v)

        d = _update_config(config_parser, d)
