.. _installing:

Installation
============

1. Virtual environments
-----------------------

We recommend using virtual environments, but this section can be skipped.

Virtual environments with `virtualenv`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the `virtualenv` Python package::

    $ pip install virtualenv

Create a virtual environment with a specific Python version::

    $ virtualenv -p python3.7 gwenv

Activate the virtual environment::

    $ source gwenv/bin/activate

Virtual environments with `Conda`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install Conda following the `online instructions <https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html>`_.

Create a virtual Conda environment with a specific Python version::

    $ conda create --name gwenv python=3.7 cython numpy

Activate the virtual environment::

    $ conda activate gwenv

Using the virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With an environment activated, the command line should look something like::

    (gwenv) $

2. `GeoWombat` requirements
---------------------------

Cython, NumPy, and GDAL must be installed prior to installing GeoWombat. Cython and NumPy can be installed via pip (note that the conda example above installs Cython and NumPy)::

    $ pip install cython numpy

GDAL can be installed via pip or conda, but it requires the GDAL binaries.

Installing non-Python `GeoWombat` prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GeoWombat requires `GDAL <https://gdal.org/>`_ and `libspatialindex <https://libspatialindex.org/>`_.

.. note::

    This GDAL requirement is a prerequisite itself for the `Python GDAL bindings <https://pypi.org/project/GDAL/>`_.

Installing `GDAL` on Ubuntu
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install Python 3 (if not installed)::

    $ # Ubuntu>=16.10
    $ sudo apt install python3

If Python 3 is already installed, but pip is not, install pip::

    $ apt install python3-pip
    $
    $ # or
    $ # pip install pip-tools

Install the GDAL binaries::

    $ # 1) Add GDAL and update Ubuntu
    $ sudo add-apt-repository ppa:ubuntugis/ppa && sudo apt-get update
    $ # Optional--add the latest unstable version (for GDAL 3.0)
    $ # sudo add-apt-repository ppa:ubuntugis/ubuntugis-unstable
    $
    $ # 2) Install the GDAL binaries
    $ sudo apt install gdal-bin
    $ sudo apt install libgdal-dev
    $
    $ # 3) Update environment paths
    $ export CPLUS_INCLUDE_PATH=/usr/include/gdal
    $ export C_INCLUDE_PATH=/usr/include/gdal
    $ export LD_LIBRARY_PATH=/usr/local/lib
    $
    $ # Optional, add permanently to a .profile file
    $ # echo 'export CPLUS_INCLUDE_PATH="/usr/include/gdal"' >> ~/.profile
    $ # echo 'export C_INCLUDE_PATH="/usr/include/gdal"' >> ~/.profile
    $ # echo 'export LD_LIBRARY_PATH="/usr/local/lib"' >> ~/.profile
    $
    $ # 4) Check the installed GDAL version
    $ gdalinfo --version

.. note::

    If you install GDAL 3.x on Ubuntu, when importing rasterio you may receive an error saying `CPLE_OpenFailedError: Unable to open EPSG support file gcs.csv`. This error is documented on rasterio's [ReadTheDocs page](https://rasterio.readthedocs.io/en/latest/faq.html) and [GitHub page](https://github.com/mapbox/rasterio/issues/1787). If the suggested solutions do not fix the issue, you can try setting the `GDAL_DATA` environment variable to point to Fiona (which will be installed automatically when installing GeoWombat). For example, if you have setup a virtual environment, the `GDAL_DATA` variable can point to `/path/to/myenv/lib/python3.7/site-packages/fiona/gdal_data`, where `/path/to/myenv` is the name of your virtual environment path directory. Change 3.7 if using a different Python version.

Installing `libspatialindex` on Ubuntu
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install `libspatialindex` (which is a requirement for `Rtree <https://pypi.org/project/Rtree/>`_)::

    $ apt install libspatialindex-c4v5

Installing `GDAL` with `Conda`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See `conda-forge examples <https://anaconda.org/conda-forge/gdal>`_.

Installing `libspatialindex` with `Conda`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See `conda-forge examples <https://anaconda.org/conda-forge/libspatialindex>`_.

Install the `GDAL` Python bindings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GeoWombat will attempt to install the GDAL Python package if the GDAL binaries are installed. To install Python GDAL manually, use pip::

    $ # match the GDAL binaries
    $ pip install GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}')
    $ # which is the same as the following if the GDAL binary version is 2.4.0
    $ # pip install GDAL==2.4.0 --no-binary=gdal

3. Install `GeoWombat`
----------------------

Install the most recent stable version from GitHub.com::

    $ pip install https://github.com/jgrss/geowombat/archive/1.2.4.tar.gz

Or, install the latest development version from GitHub.com::

    $ pip install git+https://github.com/jgrss/geowombat

Install GeoWombat with libraries for co-registration::

    $ pip install git+https://github.com/jgrss/geowombat.git#egg=project[coreg]

4. Testing the installation
---------------------------

If GeoWombat installed correctly, you should be able to run the following command::

    $ python -c "import geowombat as gw;print(gw.__version__)"

or in Python:

.. ipython:: python

    import geowombat as gw
    print(gw.__version__)
