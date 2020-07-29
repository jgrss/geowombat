import logging

_FORMAT = '%(asctime)s:%(levelname)s:%(lineno)s:%(module)s.%(funcName)s:%(message)s'
_formatter = logging.Formatter(_FORMAT, '%H:%M:%S')
_handler = logging.NullHandler()
_handler.setFormatter(_formatter)

logger = logging.getLogger(__name__)
logger.addHandler(_handler)
