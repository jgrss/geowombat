import logging


_FORMAT = '%(asctime)s:%(levelname)s:%(lineno)s:%(module)s.%(funcName)s:%(message)s'

_handler = logging.StreamHandler()
_formatter = logging.Formatter(_FORMAT, '%H:%M:%S')
_handler.setFormatter(_formatter)


def add_handler(logger):

    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

    return logger


GDAL_INSTALL_MSG = (
    "GDAL Python bindings (osgeo) are required for this feature "
    "but are not installed.\n"
    "Most geowombat features work without GDAL. "
    "To enable this feature, install GDAL:\n\n"
    "  Conda (easiest, all platforms):\n"
    "    conda install -c conda-forge gdal\n\n"
    "  Linux (Ubuntu/Debian):\n"
    "    sudo apt install gdal-bin libgdal-dev python3-gdal\n\n"
    "  macOS (Homebrew):\n"
    "    brew install gdal\n"
    '    pip install gdal[numpy]=="$(gdal-config --version).*"\n\n'
    "  Windows:\n"
    "    conda install -c conda-forge gdal\n\n"
    "  See https://geowombat.readthedocs.io for full instructions."
)
