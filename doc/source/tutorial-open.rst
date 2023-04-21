.. _tutorial-open:

Opening rasters
===============

GeoWombat's file opening is meant to mimic Xarray and Rasterio. That is, rasters are typically opened with
a context manager using the function :func:`geowombat.open`. GeoWombat uses :func:`xarray.open_rasterio` to
load data into an `xarray.DataArray`. In GeoWombat, the data are always chunked, meaning the data are always
loaded as Dask arrays. As with :func:`xarray.open_rasterio`, the opened DataArrays always have at least 1 band.

Opening a single image
----------------------

Opening an image with default settings looks similar to :func:`xarray.open_rasterio` and :func:`rasterio.open`.
:func:`geowombat.open` expects a file name (`str` or `pathlib.Path`).

.. ipython:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518

    with gw.open(l8_224078_20200518) as src:
        print(src)

In the example above, `src` is an `xarray.DataArray`. Thus, printing the object will display the underlying Dask array dimensions and chunks, the DataArray named coordinates, and the DataArray attributes.

Opening multiple bands as a stack
---------------------------------

Often, satellite bands will be stored in separate raster files. To open the files as one DataArray, specify a list instead of a file name.

.. ipython:: python

    from geowombat.data import l8_224078_20200518_B2, l8_224078_20200518_B3, l8_224078_20200518_B4

    with gw.open([l8_224078_20200518_B2, l8_224078_20200518_B3, l8_224078_20200518_B4]) as src:
        print(src)

By default, GeoWombat will stack multiple files by time. So, to stack multiple bands with the same timestamp, change the **stack_dim** keyword.

.. ipython:: python

    from geowombat.data import l8_224078_20200518_B2, l8_224078_20200518_B3, l8_224078_20200518_B4

    with gw.open(
        [
            l8_224078_20200518_B2, l8_224078_20200518_B3, l8_224078_20200518_B4
        ],
        stack_dim='band'
    ) as src:
        print(src)

.. note::

    If time names are not specified with ``stack_dim`` = 'time', GeoWombat will attempt to parse dates from the file names. This could incur significant overhead when the file list is long. Therefore, it is good practice to specify the time names.

Overhead required to parse file names

.. code:: python

    with gw.open(long_file_list, stack_dim='time') as src:
        ...

No file parsing overhead

.. code:: python

    with gw.open(long_file_list, time_names=my_time_names, stack_dim='time') as src:
        ...

Opening multiple bands as a mosaic
----------------------------------

When a list of files are given, GeoWombat will stack the data by default. To mosaic multiple files into the same band coordinate, use the **mosaic** keyword.

.. ipython:: python

    from geowombat.data import l8_224077_20200518_B2, l8_224078_20200518_B2

    with gw.open(
        [
            l8_224077_20200518_B2, l8_224078_20200518_B2
        ],
        mosaic=True
    ) as src:
        print(src)

See :ref:`io` for more examples illustrating file opening.
