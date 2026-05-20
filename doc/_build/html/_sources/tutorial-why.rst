.. _tutorial-why:

Why GeoWombat?
==============

GeoWombat uses ``xarray`` and ``rasterio`` to open geospatial rasters as chunked ``dask`` arrays. So, why not just use
these libraries directly?

GeoWombat is built on top of these libraries, with some key additions for added flexibility with raster I/O and data processing.
Examples include:

- On-the-fly CRS transformation
- Automatic data alignment
- Multi-image mosaicking
- Flexible data writing over parallel tasks

Why not Rasterio?
-----------------

Rasterio is the engine behind much of the data transformations and I/O. However, ``rasterio`` loads data in memory as ``numpy`` arrays. We
wanted to build a library that could process rasters of any size.

Why not Xarray?
---------------

Xarray's `open_rasterio <https://docs.xarray.dev/en/v2022.10.0/generated/xarray.open_rasterio.html>`_ function is the interface
to ``rasterio``. It opens a delayed raster as a `dask.array.Array <https://docs.dask.org/en/stable/generated/dask.array.Array.html#dask.array.Array>`_,
which means data are not loaded into memory until requested. However, ``xarray`` is intended to be a
`general purpose library <http://xarray.pydata.org/en/stable/internals.html#extending-xarray>`_. Therefore,
it does not include geo-specific tasks included in ``geowombat``, such as radiometric adjustments, vector/raster interaction,
and special purpose I/O such as on-the-fly mosaicking.

Related projects
----------------

The following libraries use ``xarray`` or allow for processing over large datasets in similar ways to ``geowombat``.

- `rioxarray <https://corteva.github.io/rioxarray/stable/>`_
- `Satpy <https://satpy.readthedocs.io/en/latest/>`_
- `RIOS <http://www.rioshome.org/en/latest/>`_
