.. _install_post:

Post installation
=================

Install optional extras
#######################

Install GeoWombat with libraries for co-registration::

    (gwenv) pip install git+https://github.com/jgrss/geowombat.git#egg=project[coreg]

Install GeoWombat with libraries for co-registration and geo-performance enhancements::

    (gwenv) pip install git+https://github.com/jgrss/geowombat.git#egg=project[coreg,perf]

Install GeoWombat with all extra libraries::

    (gwenv) pip install git+https://github.com/jgrss/geowombat.git#egg=project[all]

.. _test-install:

Test the installation
---------------------

1. Test the import
##################

If GeoWombat installed correctly, you should be able to run the following command::

    python -c "import geowombat as gw;print(gw.__version__)"

or in Python:

.. ipython:: python

    import geowombat as gw
    print(gw.__version__)

2. Unittests
############

Install `testfixtures` (used to test logging outputs in `test_config.py`)::

    pip install testfixtures

Run all unittests inside GeoWombat's `/tests` directory::

    cd geowombat/tests
    python -m unittest

Run an individual test::

    python test_open.py

An example of a full installation on Ubuntu with Conda
------------------------------------------------------

The following installs a working anaconda environment including GDAL::

    pip install pip-tools
    conda create -n geowombat python=3.7  cython scipy numpy zarr requests -c conda-forge
    conda activate geowombat
    sudo apt install libspatialindex-dev libgdal-dev
    conda install -c conda-forge libspatialindex zarr requests
    pip install git+https://github.com/jgrss/geowombat
    python -c "import geowombat as gw;print(gw.__version__)"