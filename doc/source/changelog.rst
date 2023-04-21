.. _changelog:

.. note::
    This page is deprecated. Please refer to `CHANGELOG.md <https://github.com/jgrss/geowombat/blob/main/CHANGELOG.md>`_ at https://github.com/jgrss/geowombat/ for changes >=1.8.0.

Changelog
=========

1.7.5 (20 May 2022)
----------------------

Enhancements
~~~~~~~~~~~~
- ML - added ability to train unsupervised sklearn predictors
- ML - Added ability to do Crossvalidation and hyperparameter tuning with sklearn predictors

Bug fixes
~~~~~~~~~
- resolved `.data` errors
- added fix to handle missing data in ML pipeline for prediction

1.7.3 (18 August 2021)
----------------------

Enhancements
~~~~~~~~~~~~

- Added check for corrupted .zip files in NASA data downloader.

1.7.2 (9 June 2021)
-------------------

Bug fixes
~~~~~~~~~

- Fixed time grouping in :func:`geowombat.load`.

1.7.1 (21 May 2021)
-------------------

Bug fixes
~~~~~~~~~

- Added a check for finite values in :func:`geowombat.load` to safely load ``numpy.inf`` values.

1.7.0 (17 May 2021)
-------------------

New
~~~

- Created :func:`geowombat.load` function.
    - This function is intended to support in-memory loading of time series data using :func:`xarray.open_mfdataset`, `dask <https://dask.org/>`_, and `ray <https://ray.io/>`_.

1.6.7 (12 May 2021)
-------------------

Bug fixes
~~~~~~~~~

- Added geowombat attribute lookup in :func:`geowombat.open` to support bounds as data slice objects.

1.6.6 (11 May 2021)
-------------------

Bug fixes
~~~~~~~~~

- Added geowombat attribute lookup in :func:`geowombat.open` to support windows as data slice objects.

1.6.5 (6 May 2021)
------------------

New
~~~

- Added :func:`DataArray.gw.read` method.

Bug fixes
~~~~~~~~~

- Added check for AOT object type to generate a NumPy array when a float is given.
- Fixed type conversion in :class:`geowombat.core.vi.TasseledCap` outputs.
- Fixed band lookup order in Landsat angle file creation.
- Improved masking in `geowombat.core.vi` module.

1.6.4 (16 March 2021)
---------------------

New
~~~

- Added kernel NDVI to vegetation indices.

Bug fixes
~~~~~~~~~

- Fixed ``has_band_dim`` and ``has_time_dim`` property methods.
    - Now uses ``xarray.DataArray.dims`` to check for existing dimensions.
- Fixed missing return value in vegetation index clip.
- Improved checks for ``DataArray`` band coordinate in vegetation index calculations.

Enhancements
~~~~~~~~~~~~

- Added automatic band dimension squeezing in :func:`{DataArray}.gw.imshow`.

1.6.3 (8 March 2021)
--------------------

New
~~~

- Added ``resampling`` and ``num_threads`` keyword arguments to :func:`geowombat.radiometry.landsat_pixel_angles` and :func:`geowombat.util.GeoDownloads().download_cube`.
- Added ``max_aot`` keyword argument to :func:`geowombat.radiometry.sixs.SixS().get_optimized_aot`.

1.6.2 (7 March 2021)
--------------------

New
~~~

- Added ``coords_only`` keyword argument in :func:`geowombat.transform_crs`.

Enhancements
~~~~~~~~~~~~

- Replaced ``joblib.Parallel`` with ``concurrent.futures.ThreadPoolExecutor`` in :func:`geowombat.util.web.DownloadMixin().download_gcp`.

1.6.1 (5 March 2021)
--------------------

Bug fixes
~~~~~~~~~

- Fixed Landsat angle .hdr file cell size in :func:`geowombat.radiometry.landsat_pixel_angles`.
    - When using the Landsat angle files with `subsample` > 1, the generated .img.hdr files retain the native 30 m cell size.
    - This fix edits the .img.hdr file to update the resample cell size.

1.6.0 (3 March 2021)
--------------------

New
~~~

- NetCDF support

    - NetCDF files can be opened following `rasterio dataset formatting <https://rasterio.readthedocs.io/en/latest/topics/datasets.html>`_.
    - Band variables for multiple datasets can be opened with the `netcdf_vars` argument. E.g.,

.. code:: python

    with gw.open('netcdf:dataset.nc', netcdf_vars=['blue', 'green', 'red'] as src:
        print(src)

- Atmospheric correction

    - Created :class:`geowombat.Radiometry.sixs.SixS`.
    - Created 6S lookup tables (LUTs) for Landsat 5, 7, and 8 and Sentinel 2A and 2B.
        - Includes LUTs for wavelengths blue, green, red, NIR, SWIR1, and SWIR2.

- Spatial-temporal fusion

    - Added StarFM method under :func:`geowombat.radiometry._fusion`.

.. warning::
    This implementation of StarFM is experimental and not tested against the original method.

- Added `prepend_str` argument to :func:`geowomat.core.sort_images_by_date`.

1.5.5 (28 Dec 2020)
-------------------

New
~~~

- Added ``min_frac_area`` keyword argument in :func:`geowombat.extract`.

1.5.4 (15 Dec 2020)
-------------------

Bug fixes
~~~~~~~~~

- Added check for ``stack_dim`` argument in :func:`geowombat.open`.
- Fixed ``data_are_separate`` and ``data_are_stacked`` properties.

Enhancements
~~~~~~~~~~~~

- Added ``persist_filenames`` keyword argument in :func:`geowombat.open`.

1.5.3 (13 Dec 2020)
-------------------

Bug fixes
~~~~~~~~~

- Added ``window_id`` return item in :func:`geowombat.calc_area`.

New
~~~

- Added :func:`geowombat.to_netcdf` function.

1.5.2 (9 Dec 2020)
------------------

Bug fixes
~~~~~~~~~

- Changed ``where`` clause from ``self`` to ``xarray`` in :func:`geowombat.wi`.

1.5.1 (3 Dec 2020)
------------------

Bug fixes
~~~~~~~~~

- Added check for empty polygon list in :func:`geowombat.array_to_polygon`.

1.5.0 (1 Dec 2020)
------------------

Enhancements
~~~~~~~~~~~~

- Added optional ``dask.distributed`` ``Client`` in :func:`geowombat.extract`
- :class:`geowombat.core.parallel.ParallelTask`.
    - Added 'ray' as an optional scheduler with ``@ray.remote`` user functions.
    - Added the option of iterating over a list of raster files.
- Added option to pass ``chunks=None`` in :class:`geowombat.open`.

Bug fixes
~~~~~~~~~

- Fixed conflict with global configuration and a user-provided reference ``DataArray`` in :func:`geowombat.polygon_to_array`.

1.4.6 (13 Nov 2020)
-------------------

Bug fixes
~~~~~~~~~

- Added `shapely.geometry.MultiPolygon` as an accepted data type in :func:`geowombat.core.prepare_points`.
- Added missing `GeoDataFrame` columns in :func:`geowombat.core.polygons_to_points`.

1.4.5 (9 Nov 2020)
------------------

Bug fixes
~~~~~~~~~

- Added check for config reference resolution type in :func:`geowombat.polygon_to_array`.

1.4.4 (8 Nov 2020)
------------------

Enhancements
~~~~~~~~~~~~

- Add global configuration options for :func:`geowombat.polygon_to_array`.

1.4.3 (2 Nov 2020)
-------------------

Bug fixes
~~~~~~~~~

- Added batch id to the window count id in :func:`geowombat.to_raster` and :class:`geowombat.core.parallel.ParallelTask`.

1.4.2 (25 Oct 2020)
-------------------

Bug fixes
~~~~~~~~~

- Fixed check for compression when using ``dask.store``.
- Updated the resampling methods to account for rasterio updates.

1.4.1 (22 Oct 2020)
-------------------

Bug fixes
~~~~~~~~~

- Fixed slicing error in ``DataArray`` window generator with arrays of >2 dimensions.

Enhancements
~~~~~~~~~~~~

- Added window id in :class:`geowombat.core.parallel.ParallelTask`.

1.4.0 (22 Oct 2020)
-------------------

Bug fixes
~~~~~~~~~

- Fixed error in destination bounds transformation when ``ref_crs`` was used in a configuration context without ``ref_bounds``, like:

.. code:: python

    # Previously did transform correctly
    with gw.config.update(ref_crs=crs, ref_bounds=bounds):
        with gw.open() as src:
            ...

    # Previously did not transform correctly
    with gw.config.update(ref_crs=crs):
        with gw.open() as src:
            ...

New
~~~

- Created :func:`geowombat.bounds_to_coords` function.

Enhancements
~~~~~~~~~~~~

- Added object types in the the data window generator. Yield type options now include ``DataArrays``, ``slice`` objects, and ``rasterio.windows.Window`` objects.

.. code:: python

    with gw.open() as src:
        for w in src.gw.windows():
            ...

1.3.1 (26 Aug 2020)
-------------------

Bug fixes
~~~~~~~~~

- Added a block type check to avoid compressing `zarr` files when data are written into separate files.

1.3.0 (25 Aug 2020)
-------------------

New
~~~

- Added machine learning module `geowombat.ml` with :func:`geowombat.ml.fit` and :func:`geowombat.ml.fit_predict` methods.
    - Model fitting is performed with `sklearn-xarray <https://phausamann.github.io/sklearn-xarray/>`_
    - Requires an installation with:
        - `pip install git+https://github.com/jgrss/geowombat.git#egg=project[ml]`.
- Added tasks visualization in `geowombat.tasks`.
    - Tasks can be setup and visualized with the :class:`geowombat.tasks.GeoTask` class.
        - Visualization graph implementation borrowed from `xarray-simlab <https://xarray-simlab.readthedocs.io/en/latest/index.html>`_
    - Requires an installation with:
        - `pip install git+https://github.com/jgrss/geowombat.git#egg=project[view]`.

1.2.33 (25 Aug 2020)
--------------------

Bug fixes
~~~~~~~~~

- Fixed coordinate length mismatch with converted polygons in :func:`geowombat.polygon_to_array`.

1.2.32 (24 Aug 2020)
--------------------

Bug fixes
~~~~~~~~~

- Added check/replace for Landsat 7 with thermal band for web downloads from the Google Cloud Platform.

1.2.31 (20 Aug 2020)
--------------------

Enhancements
~~~~~~~~~~~~

- Added support for parallel downloads in :func:`download_cube`, from the :class:`geowombat.util.GeoDownloads` class.

New
~~~

- Added Landsat and Sentinel 2 URL parsing in :class:`geowombat.util.GeoDownloads` for data on the Google Cloud Platform.

1.2.30 (17 Aug 2020)
--------------------

Enhancements
~~~~~~~~~~~~

- Added array value to polygon `GeoDataFrame` output in :func:`geowombat.array_to_polygon`.

1.2.29 (15 Aug 2020)
--------------------

Enhancements
~~~~~~~~~~~~

- Added `col` keyword argument in :func:`geowombat.polygon_to_array`.

1.2.28 (14 Aug 2020)
--------------------

Bug fixes
~~~~~~~~~

- Fixed error when using :func:`geowombat.polygon_to_array` with multiple opened rasters.

1.2.27 (8 Aug 2020)
-------------------

Bug fixes
~~~~~~~~~

- Fixed error with Google Cloud Platform query updates in :func:`geowombat.util.web.GeoDownloads().download_cube`.

1.2.26 (7 Aug 2020)
-------------------

Bug fixes
~~~~~~~~~

- Changed the default 'no data' value `nodata=None` to `nodata=0` and added `int` and `float` type checks.
    - This fix addressed `Issue #41 <https://github.com/jgrss/geowombat/issues/41>`_.

1.2.25 (4 Aug 2020)
-------------------

Bug fixes
~~~~~~~~~

- Changed Landsat 5 metadata flag for SWIR2 from 6 to 7 in :class:`geowombat.radiometry.sr.MetaData`.
- Removed logger file writing, which conflicted with read-only containers.

Enhancements
~~~~~~~~~~~~

- Reorganized the :class:`geowombat.radiometry.brdf.BRDF()` module.

1.2.24 (30 July 2020)
---------------------

Bug fixes
~~~~~~~~~

- Removed forced file writing of the log and replaced with per-module logging. `6579eb8 <https://github.com/jgrss/geowombat/commit/6579eb8e059ad8ef4e4b34e3793051104ee9bc39>`_

1.2.23 (27 July 2020)
---------------------

Bug fixes
~~~~~~~~~

- Fixed padded block writing with user functions in :func:`geowombat.to_raster`.
- Added check for existing metadata file in :func:`geowombat.util.web.download_cube`.

Enhancements
~~~~~~~~~~~~

- Reorganized the ReadTheDocs pages.

1.2.22 (21 July 2020)
---------------------

Bug fixes
~~~~~~~~~

- Added missing 'l5' download flag in :func:`geowombat.util.web.download_cube`.

Enhancements
~~~~~~~~~~~~

- Added 'l5' to :func:`geowombat.radiometry.sr.bandpass`.
- Modified support for writing GeoTiffs as separate files in :func:`geowombat.to_raster`.
- The previous version used the `DataArray.transform` property, which was derived from the full raster extent. The latest version of GeoWombat uses a `DataArray.gw.transform`, which is an updated transform property for each raster chunk.

1.2.21 (8 July 2020)
--------------------

Bug fixes
~~~~~~~~~

- Added check for None row/column chunks in :class:`geowombat.core.parallel.ParallelTask`.

Enhancements
~~~~~~~~~~~~

- Added `affine` and `transform` properties.

1.2.20 (7 July 2020)
--------------------

Bug fixes
~~~~~~~~~

- Fixed conditional value replacement in :func:`geowombat.calc_area`.

Enhancements
~~~~~~~~~~~~

- Added `return_binary` argument to :func:`geowombat.core.geoxarray.GeoWombatAccessor.compare`.

New
~~~

- Created :class:`geowombat.core.parallel.ParallelTask`.

1.2.19 (6 July 2020)
--------------------

Enhancements
~~~~~~~~~~~~

- Added source attributes to return object in :func:`geowombat.core.geoxarray.GeoWombatAccessor.compare`.

New
~~~

- Created :func:`geowombat.core.geoxarray.GeoWombatAccessor.replace` function.
- Created :func:`geowombat.replace` function.
- Created :func:`geowombat.core.geoxarray.GeoWombatAccessor.recode` function.
- Created :func:`geowombat.recode` function.

1.2.18 (1 July 2020)
--------------------

Bug fixes
~~~~~~~~~

- Fixed call to :func:`geowombat.sample` from ``DataArray`` method.

New
~~~

- Added image metadata tags to ``DataArray`` attributes in :class:`geowombat.open`.
- Added support for VRT creation from multiple files.
- Created :func:`geowombat.calc_area` function.
- Created :func:`geowombat.core.geoxarray.GeoWombatAccessor.compare` function.
- Created :func:`geowombat.core.geoxarray.GeoWombatAccessor.match_data` function.

1.2.17 (25 June 2020)
---------------------

Bug fixes
~~~~~~~~~

- Added missing tag update in file compression stage.
- Fixed issue with compression being triggered with ``compress=None`` or ``compress=False``.

1.2.16 (22 June 2020)
---------------------

Bug fixes
~~~~~~~~~

- Fixed an issue with lingering configuration reference bounds.

New
~~~

- Added metadata tags keyword argument to :func:`geowombat.to_raster`.
- Added `chunk_grid` and `footprint_grid` `DataArray` properties.

1.2.15 (15 June 2020)
---------------------

New
~~~

- Added :func:`set_nodata` function for `DataArrays`.
- Added :func:`bounds_overlay` function for `DataArrays`.

1.2.14 (12 June 2020)
---------------------

Bug fixes
~~~~~~~~~

- Fixed 'no data' clipping error in :func:`geowombat.util.GeoDownloads.download_cube`.

New
~~~

- Added `file_list` to :func:`geowombat.core.sort_images_by_date`.
- Added `nodata` keyword argument to :class:`geowombat.open`.

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

- Added user argument `dtype` in :class:`geowombat.open`.

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

- Added support for band stacking (in addition to time stacking) in :class:`geowombat.open`. The new keyword argument is `stack_dim` and can be used like:

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
