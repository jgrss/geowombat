import logging


_FORMAT = '%(asctime)s:%(levelname)s:%(lineno)s:%(module)s.%(funcName)s:%(message)s'

_handler = logging.StreamHandler()
_formatter = logging.Formatter(_FORMAT, '%H:%M:%S')
_handler.setFormatter(_formatter)


def add_handler(logger):

    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

    return logger
