.. _install_gdal:

GDAL installation
=================

Installing non-Python GeoWombat prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GeoWombat requires `GDAL <https://gdal.org/>`_ and `libspatialindex <https://libspatialindex.org/>`_.

.. note::

    This GDAL requirement is a prerequisite itself for the `Python GDAL bindings <https://pypi.org/project/GDAL/>`_.

If all you need to install is GDAL, see :ref:`Installing GDAL on Ubuntu <gdal-install-ubuntu>` or :ref:`Installing GDAL on Conda <gdal-install-conda>`.

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
    sudo add-apt-repository ppa:ubuntugis/ppa && sudo apt update && sudo apt upgrade -y

    # Install the GDAL binaries
    sudo apt install gdal-bin libgdal-dev geotiff-bin libgl1 -y

    # Ensure g++ cc1plus is installed for geowombat compile
    sudo apt install g++ -y

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

    sudo apt install libspatialindex-dev -y
