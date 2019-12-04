.. _installing:

Installation
============

Mac Instructions
------------

Install the latest development version from GitHub using pip::

    $ pip install --user git+https://github.com/jgrss/geowombat

Linux Instructions
-----------

Install pip tools and ensure a working version of GDAL and python bindings:: 

    $ pip install pip-tools

    $ sudo add-apt-repository ppa:ubuntugis/ppa && sudo apt-get update
    $ sudo apt-get update
    $ sudo apt-get install gdal-bin
    $ sudo apt-get install libgdal-dev
    $ export CPLUS_INCLUDE_PATH=/usr/include/gdal
    $ export C_INCLUDE_PATH=/usr/include/gdal
    $ pip install GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}')

Build conda environment (not required) to isolate geowombat::

    $ conda create --name geowombat python=3.7 cython numpy
    $ conda activate geowombat

Install the latest development version from GitHub using pip::

    $ pip install --user git+https://github.com/jgrss/geowombat

