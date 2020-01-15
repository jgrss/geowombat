.. _changelog:

Change Log
==========

1.0.2 (16 January 2020)
-----------------------

Bug fixes
~~~~~~~~~

- Fixed a problem with :func:`geowombat.moving_window` block overlaps when requested window sizes were larger than the smallest Dask chunk size.
- Fixed :func:`geowombat.moving_window` percentile quantile sorting of a full moving window.

1.0.1 (15 January 2020)
-----------------------

New
~~~

- Added a check for even or odd window sizes with :func:`geowombat.moving_window`.
- Added an option to co-register a list of images.
- Added percentiles to :func:`geowombat.moving_window`.

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
