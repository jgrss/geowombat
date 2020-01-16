.. _installing:

Installation
============

General Instructions
--------------------

Install the stable version from GitHub.com::

    $ pip install https://github.com/jgrss/geowombat/archive/1.0.3.tar.gz

Install the latest development version from GitHub.com::

    $ pip install git+https://github.com/jgrss/geowombat

Linux Instructions
------------------

Install pip tools and ensure a working version of GDAL and Python bindings::

    $ pip install pip-tools

    $ sudo add-apt-repository ppa:ubuntugis/ppa && sudo apt-get update
    $ sudo apt-get update
    $ sudo apt-get install gdal-bin
    $ sudo apt-get install libgdal-dev
    $ export CPLUS_INCLUDE_PATH=/usr/include/gdal
    $ export C_INCLUDE_PATH=/usr/include/gdal
    $ pip install GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}')

Virtual environments
--------------------

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

Install `GeoWombat`
~~~~~~~~~~~~~~~~~~~

With an environment activated, install `GeoWombat`::

    $ (gwenv) pip install git+https://github.com/jgrss/geowombat
