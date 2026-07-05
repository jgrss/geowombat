.. _tutorial-why:

Why GeoWombat?
==============

Satellites like Sentinel-2, Landsat, and PlanetScope hand you petabytes of free
imagery, but turning that imagery into results usually means gluing together a
whole stack of Python libraries. You might reach for ``rasterio`` to read and
write, GDAL to warp, ``numpy`` for the math, ``xarray`` for labeled dimensions,
``geopandas`` for vectors, and ``scikit-learn`` for modeling. Each of these has
its own conventions for coordinate reference systems, nodata values, affine
transforms, and chunked computation, and it is up to you to make them agree.

Even a seemingly simple task, like mosaicking two adjacent Landsat tiles, expects
you to understand affine transformations, CRS alignment, and resampling. More
involved workflows such as multi-temporal classification, BRDF-adjusted surface
reflectance, or continental-scale time series quickly turn into a lot of
boilerplate and demand deep familiarity with the underlying data structures. If
you are an ecologist, agronomist, or urban planner rather than a geospatial
software engineer, that is a real barrier.

GeoWombat's goal is to get out of your way. It uses ``xarray`` and ``rasterio`` to
open geospatial rasters as chunked ``dask`` arrays, then adds a consistent,
high-level API on top so you can work with large datasets without writing
low-level geospatial code. Key additions include:

- On-the-fly CRS transformation
- Automatic data alignment
- Multi-image mosaicking
- Flexible data writing over parallel tasks

So why not just use the underlying libraries directly?

Why not Rasterio?
-----------------

Rasterio is the engine behind much of GeoWombat's data transformation and I/O.
However, ``rasterio`` loads data into memory as ``numpy`` arrays. We wanted a
library that could process rasters of any size, so GeoWombat keeps your data lazy
with ``dask`` and only computes when you ask it to.

Why not Xarray?
---------------

Xarray's interface to ``rasterio`` opens a delayed raster as a
`dask.array.Array <https://docs.dask.org/en/stable/generated/dask.array.Array.html#dask.array.Array>`_,
so data are not loaded into memory until you request them. But ``xarray`` is a
`general purpose library <http://xarray.pydata.org/en/stable/internals.html#extending-xarray>`_ —
it does not include the geo-specific tasks GeoWombat adds, such as radiometric
adjustments, vector/raster interaction, and special-purpose I/O like on-the-fly
mosaicking.

How GeoWombat compares
----------------------

Several other projects tackle parts of the same workflow. Here is how they differ
from GeoWombat:

- `rioxarray <https://corteva.github.io/rioxarray/stable/>`_ extends ``xarray`` with
  rasterio-backed I/O and CRS-aware operations, but does not include remote sensing
  features like vegetation indices, QA masking, or classification pipelines.
- `Google Earth Engine <https://earthengine.google.com/>`_ offers cloud-based
  processing of global archives, but needs an internet connection, runs inside a
  proprietary platform, and limits how much control you have over the computation.
  `Xee <https://github.com/google/Xee>`_ bridges Earth Engine with ``xarray`` but
  inherits the same platform constraints.
- `Open Data Cube <https://www.opendatacube.org/>`_ provides a database-indexed
  approach to analysis-ready data, but expects you to set up infrastructure and
  ingest data first.
- `Raster Vision <https://rastervision.io/>`_ focuses specifically on deep learning
  for geospatial imagery rather than the broader analysis workflow.

GeoWombat aims to be a local-first toolkit that spans the whole chain, from data
access through modeling, behind a single API. Its context-manager pattern for
on-the-fly reprojection and alignment, built-in sensor profiles for automatic band
naming and scaling, and tight integration with ``scikit-learn`` pipelines and
PyTorch models are what set it apart.

Related projects
----------------

These libraries also build on ``xarray`` or support processing over large datasets
in ways similar to GeoWombat:

- `Satpy <https://satpy.readthedocs.io/en/latest/>`_
- `RIOS <http://www.rioshome.org/en/latest/>`_
