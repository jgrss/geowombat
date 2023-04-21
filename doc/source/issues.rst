.. _issues:

Issues
======

GDAL 3.0 and GDAL_DATA
----------------------

First, see `here <https://rasterio.readthedocs.io/en/latest/faq.html>`_ for issues with the GDAL data path.

If the GDAL_DATA directory is missing the gcs.csv file, point the GDAL_DATA environment variable to the Fiona path.

For example::

    GDAL_DATA=/envs/ts3.7/lib/python3.7/site-packages/fiona/gdal_data
