GeoWombat: geo-utilities for overhead air- and space-borne imagery
==================================================================

Like a wombat, **GeoWombat** has a simple interface with a strong backend. GeoWombat is inspired by, and built on, several key
libraries for large-scale data processing:

.. _xarray: http://xarray.pydata.org
.. _dask: http://dask.org
.. _rasterio: https://rasterio.readthedocs.io
.. _pandas: http://pandas.pydata.org
.. _geopandas: http://geopandas.org

GeoWombat is designed to provide specialized "geo-functionality" to Xarray and Dask data, using Rasterio for raster I/O.

Documentation
-------------

**Getting Started**

* :doc:`installing`
* :doc:`quick-overview`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   installing
   quick-overview

**User Guide**

* :doc:`io`
* :doc:`attributes`
* :doc:`extraction`
* :doc:`band-math`
* :doc:`machine-learning`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: User Guide

   io
   attributes
   extraction
   band-math
   machine-learning
