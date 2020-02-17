.. _installing:

Installation
============

Virtual environments
--------------------

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

GeoWombat requirements
----------------------

Cython, NumPy, and GDAL must be installed prior to installing GeoWombat.

Cython and NumPy can be installed via pip::

    $ pip install cython numpy

GDAL can be installed via pip or conda, but it requires the GDAL binaries.

Installing GDAL on Ubuntu
~~~~~~~~~~~~~~~~~~~~~~~~~

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

(Optional) Install libspatialindex::

    $ apt install libspatialindex-c4v5

Install the GDAL Python API::

    $ # match the GDAL binaries
    $ pip install GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}')
    $ # which is the same as the following if the GDAL binary version is 2.4.x
    $ # pip install GDAL==2.4

Installing GDAL with `Conda`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See `conda-forge examples <https://anaconda.org/conda-forge/gdal>`_.

Install GeoWombat
-----------------

Install the most recent stable version from GitHub.com::

    $ pip install https://github.com/jgrss/geowombat/archive/1.2.0.tar.gz

Or, install the latest development version from GitHub.com::

    $ pip install git+https://github.com/jgrss/geowombat

Install optional libraries::

    $ pip install arosics earthpy pyfftw bottleneck

Testing the installation
------------------------

If GeoWombat installed correctly, you should be able to run the following command::

    $ python -c "import geowombat as gw;print(gw.__version__)"

or in Python:

.. ipython:: python

    import geowombat as gw
    print(gw.__version__)
