.. _install:

Installation
============

Quick install
-------------

Install with pip from the GitHub repository::

    pip install git+https://github.com/jgrss/geowombat

Install dependencies
--------------------

If you already have GDAL and libspatialindex installed, geowombat should work. To test it, see the :ref:`section on testing <test-install>`. If GDAL or libspatialindex are not installed, you will need to do so before having a functional geowombat.

Install GDAL
############

If all you need to install is GDAL, see :ref:`Installing GDAL on Ubuntu <gdal-install-ubuntu>` or :ref:`Installing GDAL on Conda <gdal-install-conda>`.

Step-by-step guide
##################

If you are starting from scratch and wish to walk through the full installation procedure, see the following installation steps. We recommend using `virtual environments <https://docs.python.org/3/tutorial/venv.html>`_ and provide examples with `virtualenv <https://packaging.python.org/key_projects/#virtualenv>`_ and `conda <https://docs.conda.io/en/latest/>`_.

1. Install with a virtual environment
#####################################

Virtual environments with virtualenv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the `virtualenv` Python package::

    pip install virtualenv

Create a virtual environment with a specific Python version::

    virtualenv -p python3.7 gwenv

Activate the virtual environment::

    source gwenv/bin/activate

Virtual environments with Conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install Conda following the `online instructions <https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html>`_.

Create a virtual Conda environment with a specific Python version::

    conda create --name gwenv python=3.7 cython numpy

Activate the virtual environment::

    conda activate gwenv

Using the virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With an environment activated, the command line should look something like::

    (gwenv) $

2. GeoWombat requirements
#########################

Cython, NumPy, and GDAL must be installed prior to installing GeoWombat. Cython and NumPy can be installed via pip (note that the conda example above installs Cython and NumPy)::

    pip install cython numpy

GDAL can be installed via pip or conda, but it requires the GDAL binaries.

Installing non-Python GeoWombat prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GeoWombat requires `GDAL <https://gdal.org/>`_ and `libspatialindex <https://libspatialindex.org/>`_.

.. note::

    This GDAL requirement is a prerequisite itself for the `Python GDAL bindings <https://pypi.org/project/GDAL/>`_.

.. _gdal-install-ubuntu:

Installing GDAL on Ubuntu
~~~~~~~~~~~~~~~~~~~~~~~~~

Install Python 3 (if not installed)::

    # Ubuntu>=16.10
    sudo apt install python3

If Python 3 is already installed, but pip is not, install pip::

    apt install python3-pip

    # or
    # pip install pip-tools

Install the GDAL binaries::

    # Add GDAL and update Ubuntu
    sudo add-apt-repository ppa:ubuntugis/ppa && sudo apt-get update

    # Optional--add the latest unstable version (for GDAL 3.0)
    # sudo add-apt-repository ppa:ubuntugis/ubuntugis-unstable

    # Install the GDAL binaries
    sudo apt install gdal-bin
    sudo apt install libgdal-dev
    
    # Ensure g++ cc1plus is installed for geowombat compile
    sudo apt-get install g++

Update environment paths::

    export CPLUS_INCLUDE_PATH=/usr/include/gdal
    export C_INCLUDE_PATH=/usr/include/gdal
    export LD_LIBRARY_PATH=/usr/local/lib

    # Optional, add permanently to a .profile file
    # echo 'export CPLUS_INCLUDE_PATH="/usr/include/gdal"' >> ~/.profile
    # echo 'export C_INCLUDE_PATH="/usr/include/gdal"' >> ~/.profile
    # echo 'export LD_LIBRARY_PATH="/usr/local/lib"' >> ~/.profile

Check the installed GDAL version::

    gdalinfo --version

.. note::

    If you install GDAL 3.x on Ubuntu, when importing rasterio you may receive an error saying

    - `CPLE_OpenFailedError: Unable to open EPSG support file gcs.csv`.

    This error is documented on rasterio's `ReadTheDocs page <https://rasterio.readthedocs.io/en/latest/faq.html>`_ and `GitHub page <https://github.com/mapbox/rasterio/issues/1787>`_. If the suggested solutions do not fix the issue, you can try setting the `GDAL_DATA` environment variable to point to Fiona (which will be installed automatically when installing GeoWombat). For example, if you have setup a virtual environment, the `GDAL_DATA` variable can point to `/path/to/myenv/lib/python3.7/site-packages/fiona/gdal_data`, where `/path/to/myenv` is the name of your virtual environment path directory. Change 3.7 if using a different Python version.

Installing libspatialindex on Ubuntu
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install `libspatialindex` (which is a requirement for `Rtree <https://pypi.org/project/Rtree/>`_)::

    sudo apt install libspatialindex-dev

.. _gdal-install-conda:

Installing GDAL with Conda
~~~~~~~~~~~~~~~~~~~~~~~~~~

See `conda-forge examples <https://anaconda.org/conda-forge/gdal>`_.

Installing libspatialindex with Conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See `conda-forge examples <https://anaconda.org/conda-forge/libspatialindex>`_.

Install the GDAL Python bindings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GeoWombat will attempt to install the GDAL Python package if the GDAL binaries are installed. However, to install Python GDAL manually, use pip::

    # match the GDAL binaries
    pip install GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}') --no-binary=gdal
    # which is the same as the following if the GDAL binary version is 2.4.0
    # pip install GDAL==2.4.0 --no-binary=gdal

3. Install GeoWombat
####################

Install the latest version from GitHub.com::

    pip install git+https://github.com/jgrss/geowombat

4. Install optional extras
##########################

Install GeoWombat with libraries for co-registration::

    pip install git+https://github.com/jgrss/geowombat.git#egg=project[coreg]

Install GeoWombat with libraries for co-registration and geo-performance enhancements::

    pip install git+https://github.com/jgrss/geowombat.git#egg=project[coreg,perf]

Install GeoWombat with all extra libraries::

    pip install git+https://github.com/jgrss/geowombat.git#egg=project[all]

.. _test-install:

Test the installation
---------------------

If GeoWombat installed correctly, you should be able to run the following command::

    python -c "import geowombat as gw;print(gw.__version__)"

or in Python:

.. ipython:: python

    import geowombat as gw
    print(gw.__version__)
    
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

