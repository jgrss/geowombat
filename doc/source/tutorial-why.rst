.. _tutorial-why:

Why GeoWombat?
==============

GeoWombat uses Xarray and Rasterio to open geospatial rasters as chunked Dask arrays. So, why not just use these libraries directly?

GeoWombat is built on top of these libraries, with some key additions for added flexibility with rasterio I/O and data processing. Examples include:

- On-the-fly CRS transformation
- Automatic data alignment
- Multi-image mosaicking
- Flexible data writing over parallel tasks

Why not Rasterio?
-----------------

Rasterio is the engine behind much of the data transformations and I/O. However, rasterio loads data in memory as NumPy arrays. We wanted to build a library that could open satellite imagery of any size.

Why not Xarray?
---------------

Xarray's :func:`open_rasterio` is the interface to rasterio. It opens a raster lazily as a Dask array, which means data are not loaded into memory until requested. However, Xarray is intended to be a `general purpose library <http://xarray.pydata.org/en/stable/internals.html#extending-xarray>`_. Therefore, it does not include geo-specific tasks included in GeoWombat, such as radiometric adjustments, vector/raster interaction, and special purpose I/O such as on-the-fly mosaicking.

Related projects
----------------

The following libraries use Xarray or allow for processing over large datasets in similar ways to GeoWombat.

- `rioxarray <https://corteva.github.io/rioxarray/stable/>`_
- `Satpy <https://satpy.readthedocs.io/en/latest/>`_
- `RIOS <http://www.rioshome.org/en/latest/>`_
