########################################
GeoWombat: Utilities for geospatial data
########################################

**Like a wombat, GeoWombat has a simple interface (for raster I/O) with a strong backend (for data processing at scale).**

GeoWombat provides utilities to process geospatial raster data. The package is inspired by, and built on, several key libraries for large-scale data processing, such as `Dask <http://dask.org>`_, `Geopandas <http://geopandas.org>`_, `Pandas <http://pandas.pydata.org>`_, `Rasterio <https://rasterio.readthedocs.io>`_, and `Xarray <http://xarray.pydata.org>`_. GeoWombat interfaces directly with Xarray for raster I/O, which uses Rasterio to open raster files such as satellite images or aerial photos as `Dask arrays <https://docs.dask.org/en/latest/array.html>`_. GeoWombat uses the `Xarray register <http://xarray.pydata.org/en/stable/internals.html>`_ to extend the functionality of `Xarray DataArrays <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html>`_.

One of the key features of GeoWombat is the on-the-fly handling of multiple files. In particular, GeoWombat leverages Rasterio to transform and align rasters with varying projections and spatial resolutions. In addition to simplifying the process of data alignment, GeoWombat utilizes the `task graphs <https://docs.dask.org/en/latest/graphs.html>`_ of Dask arrays. By default, GeoWombat loads a raster as a DataArray, which points to the raster data on file using a chunked Dask array. This task graph feature simplifies parallel computations of one or more raster files of any size.

*************
Documentation
*************

**Getting Started**

* :doc:`installing`
* :doc:`quick-overview`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   installing
   issues
   quick-overview

**User Guide**

* :doc:`io`
* :doc:`io-distributed`
* :doc:`attributes`
* :doc:`config`
* :doc:`crs`
* :doc:`extraction`
* :doc:`band-math`
* :doc:`machine-learning`
* :doc:`apply`
* :doc:`moving`
* :doc:`radiometry`
* :doc:`web`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: User Guide

   io
   io-distributed
   attributes
   config
   crs
   extraction
   band-math
   machine-learning
   apply
   moving
   radiometry
   web

**Change Log**

* :doc:`changelog`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Change Log

   changelog

**Reference**

* :doc:`api`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Reference

   api
