.. _install:

Installing geowombat
====================

Easiest Install
---------------

Conda provides an easy and consistent installation method regardless of operating system. 

Installing `geowombat` from the `conda-forge` channel can be achieved by adding `conda-forge` to your channels with::

    conda config --add channels conda-forge
    conda config --set channel_priority strict

Once the `conda-forge` channel has been enabled, `geowombat` can be installed with `conda`::

    conda install geowombat

or with `mamba`::

    mamba install geowombat


Quick Install
-------------

If you system already has all the core dependencies installed. Install the latest version from the GitHub repository::

    pip install git+https://github.com/jgrss/geowombat


Detailed Install
----------------

.. tabs::

   .. tab:: OSx

      .. tabs::

         .. tab:: 1 - Dependencies

            Installing on a mac takes a few extra steps. In particular you will need to install `gcc` to compile and `gdal`. Both of these are easiest to install via homebrew: `homebrew Install <https://docs.brew.sh/Installation>`_.
            
            From the terminal window, update brew and install::

                brew update
                brew upgrade
                brew install gdal openssl gcc

            Check the installed GDAL version::

                gdalinfo --version

         .. tab:: 2 - Envrionments

            **Virtual environments with virtualenv**

            Install the `virtualenv` Python package::

                pip install virtualenv

            Create a virtual environment with a specific Python version::

                virtualenv -p python3.7 gwenv

            Activate the virtual environment::

                source gwenv/bin/activate

            Install geowombat requirements::

                pip install Cython numpy


            **Virtual environments with Conda**

            Install Conda following the `online instructions <https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html>`_.

            Create a virtual Conda environment with a specific Python version::

                conda create --name gwenv python=3.8 cython numpy

            Activate the virtual environment::

                conda activate gwenv
            
            Install geowombat requirements via conda-forge::

                conda config --env --add channels conda-forge
                conda config --env --set channel_priority strict
                conda install cython gdal numpy -y

            **Using the virtual environment**

            With an environment activated, the command line should look something like::

                (gwenv) $

         .. tab:: 3 - Geowombat
            **Install GeoWombat**

            Install the latest version from GitHub.com::

                pip install git+https://github.com/jgrss/geowombat

         .. tab:: (Extras)

            **Install optional extras**

            Geowombat has a lot of additional capabilities, some of which you may or may not want to use. For this reason we allow the user to decide which dependencies they want to install. 

            Install GeoWombat with libraries for building sphinx docs::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[docs]

            Install GeoWombat with libraries for co-registration::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[coreg]

            Install GeoWombat with libraries for savings zarr files::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[zarr]

            Install GeoWombat with libraries for machine learning and classification::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[ml]

            Install GeoWombat with libraries for pygeos, opencv, netcdf and ray support::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[perf]

            Install GeoWombat with libraries for parsing dates automatically::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[time]

            Install GeoWombat with libraries with mapping making dependencies::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[view]
            
            Install GeoWombat with libraries for accessing hosted data::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[web]

            Install GeoWombat with libraries for co-registration and geo-performance enhancements::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[coreg,perf]

            Install GeoWombat with all extra libraries::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[all]

        

   .. tab:: Linux (Debian)

      .. tabs::

         .. tab:: 1 - Dependencies

            Installing on a linux requires you to have a few underlying dependencies such as the gdal binaries::

                apt update -y && apt upgrade -y && \
                apt install -y software-properties-common && \
                add-apt-repository ppa:ubuntugis/ppa && \
                apt update -y && apt install -y \
                gdal-bin \
                geotiff-bin \
                git \
                libgdal-dev \
                libgl1 \
                libspatialindex-dev \ 
                wget \
                python3 \
                python3-pip \
                pip \
                g++ 
  
            Check the installed GDAL version::

                gdalinfo --version
                    
         .. tab:: 2 - Envrionments

            **Virtual environments with virtualenv**

            Install the `virtualenv` Python package::

                pip install virtualenv

            Create a virtual environment with a specific Python version::

                virtualenv -p python3.7 gwenv

            Activate the virtual environment::

                source gwenv/bin/activate

            Install geowombat requirements::

                pip install Cython numpy


            **Virtual environments with Conda**

            Install Conda following the `online instructions <https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html>`_.

            Create a virtual Conda environment with a specific Python version::

                conda create --name gwenv python=3.8 cython numpy

            Activate the virtual environment::

                conda activate gwenv
            
            Install geowombat requirements via conda-forge::

                conda config --env --add channels conda-forge
                conda config --env --set channel_priority strict
                conda install cython gdal numpy -y

            **Using the virtual environment**

            With an environment activated, the command line should look something like::

                (gwenv) $

         .. tab:: 3 - Geowombat
            **Install GeoWombat**

            Install the latest version from GitHub.com::

                pip install git+https://github.com/jgrss/geowombat

         .. tab:: (Extras)

            **Install optional extras**
            
            Geowombat has a lot of additional capabilities, some of which you may or may not want to use. For this reason we allow the user to decide which dependencies they want to install. 

            Install GeoWombat with libraries for building sphinx docs::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[docs]

            Install GeoWombat with libraries for co-registration::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[coreg]

            Install GeoWombat with libraries for savings zarr files::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[zarr]

            Install GeoWombat with libraries for machine learning and classification::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[ml]

            Install GeoWombat with libraries for pygeos, opencv, netcdf and ray support::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[perf]

            Install GeoWombat with libraries for parsing dates automatically::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[time]

            Install GeoWombat with libraries with mapping making dependencies::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[view]
            
            Install GeoWombat with libraries for accessing hosted data::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[web]

            Install GeoWombat with libraries for co-registration and geo-performance enhancements::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[coreg,perf]

            Install GeoWombat with all extra libraries::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[all]

        
   .. tab:: Windows

      .. tabs::

         .. tab:: 1 - Dependencies

            Although we are not 100% sure, if you use `conda` we do not currently believe you need to install gdal binaries or any other dependencies with windows. Please let us know if this is not the case!             

         .. tab:: 2 - Envrionments
            In windows we strongly recommend the use of Conda since pip often requires the use of precompiled binaries which can get tricky. 
            There may be some cases where pip installing packages will not be successful in Windows. In these cases please refer to our instructions on using `Christoph Gohlke's website <https://opensourceoptions.com/blog/how-to-install-gdal-for-python-with-pip-on-windows/>`_.

            **Virtual environments with Conda**

            Install Conda following the `online instructions <https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html>`_.

            Create a virtual Conda environment with a specific Python version::

                conda create --name gwenv python=3.8 cython numpy

            Activate the virtual environment::

                conda activate gwenv
            
            Install geowombat requirements via conda-forge::

                conda config --env --add channels conda-forge
                conda config --env --set channel_priority strict
                conda install cython gdal numpy -y

            **Using the virtual environment**

            With an environment activated, the command line should look something like::

                (gwenv) $

         .. tab:: 3 - Geowombat
            **Install GeoWombat**

            Install the latest version from GitHub.com::

                pip install git+https://github.com/jgrss/geowombat

         .. tab:: (Extras)

            **Install optional extras**
            
            Geowombat has a lot of additional capabilities, some of which you may or may not want to use. For this reason we allow the user to decide which dependencies they want to install. 

            Install GeoWombat with libraries for building sphinx docs::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[docs]

            Install GeoWombat with libraries for co-registration::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[coreg]

            Install GeoWombat with libraries for savings zarr files::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[zarr]

            Install GeoWombat with libraries for machine learning and classification::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[ml]

            Install GeoWombat with libraries for pygeos, opencv, netcdf and ray support::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[perf]

            Install GeoWombat with libraries for parsing dates automatically::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[time]

            Install GeoWombat with libraries with mapping making dependencies::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[view]
            
            Install GeoWombat with libraries for accessing hosted data::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[web]

            Install GeoWombat with libraries for co-registration and geo-performance enhancements::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[coreg,perf]

            Install GeoWombat with all extra libraries::

                pip install git+https://github.com/jgrss/geowombat.git#egg=project[all]

   .. tab:: Docker
        It is also possible to pull or build an image with geowombat already install from Dockerhub. Please refer to the `online instructions <https://pygis.io/docs/b_conda_started.html#docker-for-spatial-python-gdal-included>`_.


Test the installation
---------------------

1. Test the import
##################

If GeoWombat installed correctly, you should be able to run the following command from the terminal::

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
    
 
 
Installation Notes
~~~~~~~~~~~~~~~~~~~

.. note::
    **GDAL Install**
    GeoWombat requires `GDAL <https://gdal.org/>`_ and `libspatialindex <https://libspatialindex.org/>`_.

    This GDAL requirement is a prerequisite itself for the `Python GDAL bindings <https://pypi.org/project/GDAL/>`_.

  
.. note::
    **GDAL Paths in Linux**
    Although we think this is now resolved, earlier gdal installs had issues with paths. If this is the case:
    
    Update environment paths::

        export CPLUS_INCLUDE_PATH=/usr/include/gdal
        export C_INCLUDE_PATH=/usr/include/gdal
        export LD_LIBRARY_PATH=/usr/local/lib

        # Optional, add permanently to a .profile file
        # echo 'export CPLUS_INCLUDE_PATH="/usr/include/gdal"' >> ~/.profile
        # echo 'export C_INCLUDE_PATH="/usr/include/gdal"' >> ~/.profile
        # echo 'export LD_LIBRARY_PATH="/usr/local/lib"' >> ~/.profile

.. note::
    **EPSG File Missing in Linux**
    If you install GDAL 3.x on Ubuntu, when importing rasterio you may receive an error saying

    - `CPLE_OpenFailedError: Unable to open EPSG support file gcs.csv`.

    This error is documented on rasterio's `ReadTheDocs page <https://rasterio.readthedocs.io/en/latest/faq.html>`_ and `GitHub page <https://github.com/mapbox/rasterio/issues/1787>`_. If the suggested solutions do not fix the issue, you can try setting the `GDAL_DATA` environment variable to point to Fiona (which will be installed automatically when installing GeoWombat). For example, if you have setup a virtual environment, the `GDAL_DATA` variable can point to `/path/to/myenv/lib/python3.7/site-packages/fiona/gdal_data`, where `/path/to/myenv` is the name of your virtual environment path directory. Change 3.7 if using a different Python version.
 

.. note::

    **Install the GDAL Python bindings seperately**
    GeoWombat will attempt to install the GDAL Python package if the GDAL binaries are installed. However, to install Python GDAL manually, use pip::

        # match the GDAL binaries
        pip install GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}') --no-binary=gdal
        # which is the same as the following if the GDAL binary version is 2.4.0
        # pip install GDAL==2.4.0 --no-binary=gdal
 

 
.. toctree::
   :maxdepth: 2

   install_gdal
   install_pip
   install_conda
   install_docker
   install_post
