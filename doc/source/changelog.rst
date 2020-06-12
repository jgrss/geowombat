.. _changelog:

Change Log
==========

1.2.15 ()
---------

New
~~~

- Added :func:`set_nodata` function for `DataArrays`.

1.2.14 (12 June 2020)
---------------------

Bug fixes
~~~~~~~~~

- Fixed 'no data' clipping error in :func:`geowombat.util.GeoDownloads.download_cube`.

New
~~~

- Added `file_list` to :func:`geowombat.core.sort_images_by_date`.
- Added `nodata` keyword argument to :func:`geowombat.open`.

1.2.13 (2 June 2020)
--------------------

New
~~~

- Added `cloud_height` option in :func:`download_cube`.
- Added first attempt at supporting HDF reads.

1.2.12 (3 May 2020)
-------------------

New
~~~

- Added :func:`geowombat.radiometry.angles.estimate_cloud_shadows` to estimate cloud shadows from a cloud mask.

Bug fixes
~~~~~~~~~

- Fixed month range parsing in :func:`geowombat.util.web.download_cube`.

1.2.11 (1 May 2020)
-------------------

Enhancements
~~~~~~~~~~~~

- Added micrometer values for Sentinel-2 2A and 2B sensors.

1.2.10 (27 April 2020)
----------------------

New
~~~

- Added support for :func:`rasterio.windows.Window` and :func:`rasterio.coords.BoundingBox` objects in the :func:`geowombat.config.update` manager.

1.2.9 (1 April 2020)
--------------------

- Removed f-string requirement in setup.py script.
- Added keyword argument in :func:`download_cube` function.

1.2.8 (1 April 2020)
--------------------

- Modified sample iteration overhead in :func:`geowombat.moving.moving_window`.

New
~~~

- Added `weights` option in :func:`geowombat.moving.moving_window`.

1.2.7 (31 March 2020)
---------------------

New
~~~

- Added window weights for moving mean.
- Changed :func:`geowombat.moving.moving_window` parallelism from raster rows to raster samples.

1.2.6 (15 March 2020)
---------------------

- Fixed missing `gw.filename` attribute in :func:`geowombat.to_vrt`.

1.2.5 (4 March 2020)
--------------------

- Added attribute updating for band math.

1.2.4 (26 February 2020)
------------------------

- Added CRS check for `pyproj` CRS instances.

1.2.3 (23 February 2020)
------------------------

- Added check to support new CRS object in `geowombat`.
- Added padding to image edges when using the `padding` option in :func:`geowombat.to_raster`.
- Added checks for empty CRS objects.
- Added the Advanced Vegetation Index.
- Added :func:`geowombat.core.lonlat_to_xy` and :func:`geowombat.core.xy_to_lonlat` functions.

1.2.2 (12 February 2020)
------------------------

- Added `padding` option to :func:`geowombat.to_raster`.
- Added half cell adjustment to transformed samples in :func:`geowombat.sample`.

1.2.1 (12 February 2020)
------------------------

- Fixed a error in checking the spatial index.

1.2.0 (11 February 2020)
------------------------

- Rearranged and renamed various functions.

    - :func:`geowombat.to_crs` is deprecated in favor of :func:`geowombat.transform_crs`.
    - :func:`geowombat.geodataframe_to_array` is deprecated in favor of :func:`geowombat.polygon_to_array`.
    - :func:`geowombat.to_geodataframe` is deprecated in favor of :func:`geowombat.array_to_polygon`.

- Added `lazy_wombat` decorator to allow the user to apply in-memory functions lazily. See :ref:`apply` for examples.

1.1.6 (9 February 2020)
-----------------------

- Added new property `geodataframe`.
- Fixed error with pass `bounds_by` argument option to :func:`mosaic`.
- Modified :func:`geowombat.to_crs` to handle grid resampling.

1.1.5 (7 February 2020)
-----------------------

- Fixed error that caused the deletion of `ref_tar` when multiple raster files were opened.

1.1.4 (7 February 2020)
-----------------------

- Added configuration option `ref_tar` to target align outputs to a reference raster. Example usage looks like:

.. code:: python

    # Subset a raster but align to a target grid
    with gw.config.update(ref_bounds=bounds, ref_tar='image.tif'):
        with gw.open() as src:
            ...

1.1.3 (6 February 2020)
-----------------------

- Added new function :func:`geowombat.geodataframe_to_array` to convert a `geopandas.GeoDataFrame` to an `xarray.DataArray`.

1.1.2 (5 February 2020)
-----------------------

- Added 'empirical-rotation' method to :func:`geowombat.Topo().norm_topo` function in :class:`geowombat.Topo`

1.1.1 (28 January 2020)
-----------------------

- Fixed error reading by a extent bounds with `dask.delayed`

1.1.0 (24 January 2020)
-----------------------

- Added new class :class:`geowombat.radiometry.Topo` for topographic corrections.
- Added new `xarray.DataArray` `geowombat` accessor :func:`to_crs` for CRS transformations.
- Added new function :func:`geowombat.core.sort_images_by_date`.
- Added `geowombat.radiometry` module to the documentation.
- Added new `xarray.DataArray` `geowombat` property `bounds_as_namedtuple`.
- Rearranged documentation and fixed minor docstring issues.

1.0.7 (23 January 2020)
-----------------------

- Added new functions :func:`geowombat.core.indices_to_coords` and :func:`geowombat.core.coords_to_indices` to replace :func:`geowombat.core.ij_to_xy` and :func:`geowombat.core.xy_to_ij`.

1.0.6 (21 January 2020)
-----------------------

- Added missing imports for :func:`geowombat.sample`.

1.0.5 (21 January 2020)
-----------------------

Changes
~~~~~~~

- Modified :func:`geowombat.sample`. New functionality includes:

    - Systematic sampling
    - Random sampling
    - Stratified random sampling

1.0.4 (19 January 2020)
-----------------------

Changes
~~~~~~~

- Removed `DataArray` list option from :func:`geowombat.coregister`.

Bug fixes
~~~~~~~~~

- Fixed an error with global configuration settings that occurred when `ref_image` was used and subsequently overwritten.
- Removed `band_names` argument from :func:`imshow`.

1.0.3 (17 January 2020)
-----------------------

Bug fixes
~~~~~~~~~

- Added workaround example in the documentation for :func:`geowombat.moving`.

1.0.2 (16 January 2020)
-----------------------

Bug fixes
~~~~~~~~~

- Fixed a problem with :func:`geowombat.moving` block overlaps when requested window sizes were larger than the smallest Dask chunk size.
- Fixed :func:`geowombat.moving` percentile quantile sorting of a full moving window.

1.0.1 (15 January 2020)
-----------------------

New
~~~

- Added a check for even or odd window sizes with :func:`geowombat.moving`.
- Added an option to co-register a list of images.
- Added percentiles to :func:`geowombat.moving`.

Bug fixes
~~~~~~~~~

- Fixed missing `DataArray` attributes after changing data type.

1.0.0 (13 January 2020)
-----------------------

- First release

1.3.7b (12 January 2020)
------------------------

New
~~~

- Added :func:`geowombat.radiometry.pan_sharpen` function.
- Added properties for multi-spectral + panchromatic band stacks.

1.3.0b (9 January 2020)
-----------------------

New
~~~

- Added :func:`geowombat.to_vrt` function.

1.2.0b (29 December 2019)
-------------------------

New
~~~

- Added :func:`geowombat.to_geodataframe` function.

Bug fixes
~~~~~~~~~

- Fixed GeoDataFrame CRS check in :func:`geowombat.extract` function.

1.0.7b (20 December 2019)
-------------------------

New
~~~

- Added user argument `dtype` in :func:`geowombat.open` function.

Bug fixes
~~~~~~~~~

- Fixed time and band stacking error.
- Fixed dictionary string name error in CRF feature processing

1.0.0b (27 November 2019)
-------------------------

New
~~~

- Added :func:`geowombat.mask` function.

Bug fixes
~~~~~~~~~

- Fixed row/column offset error with :func:`warp` function.

1.0.0b (10 November 2019)
-------------------------

New
~~~

- Added :func:`download_cube` function in :class:`geowombat.util.web.GeoDownloads`.

1.0.0b (1 November 2019)
------------------------

Enhancements
~~~~~~~~~~~~

- Added `expand_by` user argument in :func:`geowombat.clip`.

1.0.0b (30 October 2019)
------------------------

New
~~~

- Added user functions as Xarray attributes. See :func:`geowombat.apply` for an example.

1.0.0b (24 October 2019)
------------------------

Enhancements
~~~~~~~~~~~~

- Implemented improvements from testing processes vs. threads for concurrent I/O in :func:`geowombat.to_raster`.

Bug fixes
~~~~~~~~~

- Changed BRDF normalization (:class:`geowombat.radiometry.BRDF`) from 1d to 2d in order to work with Dask arrays.

1.0.0b (23 October 2019)
------------------------

Changes
~~~~~~~

- Added fixes for surface reflectance

New
~~~

- Added support for band stacking (in addition to time stacking) in :func:`geowombat.open`. The new keyword argument is `stack_dim` and can be used like:

.. code:: python

    with gw.open(..., stack_dim='band') as ds:
        ...

1.0.0b (20 October 2019)
------------------------

Changes
~~~~~~~

- Block writing can now be done with `concurrent.futures` or with `dask.store`.

New
~~~

- Added automatic date parsing when concatenating a list of files.
- Added BRDF normalization using the c-factor method.

1.0.0a
------

History
~~~~~~~

- Examined concurrent writing workflows.
- Setup basic geo-spatial functionality.
