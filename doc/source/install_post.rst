.. _install_post:

Post installation
=================

Install optional extras
#######################

Install GeoWombat with libraries for co-registration::

    (gwenv) pip install "geowombat[coreg]@git+https://github.com/jgrss/geowombat.git"

Install GeoWombat with libraries for stream from STAC::

    (gwenv) pip install "geowombat[stac]@git+https://github.com/jgrss/geowombat.git"

Install GeoWombat with all extra libraries::

    (gwenv) pip install "geowombat[all]@git+https://github.com/jgrss/geowombat.git"

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

    python -m unittest test_open.py
