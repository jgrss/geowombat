.. _io:

Reading and writing files
=========================

File opening with GeoWombat uses the :func:`geowombat.open` function to open raster files.

.. ipython:: python

    # Import GeoWombat
    import geowombat as gw

.. ipython:: python

    # Load a 4-band test image
    from geowombat.data import rgbn

.. ipython:: python

    # Load two images that partially overlap
    from geowombat.data import rgbn_suba, rgbn_subb

To open individual images, GeoWombat wraps the :func:`xarray.open_rasterio` and :func:`xarray.open_dataset` functions.

Open a raster as a DataArray.

.. ipython:: python

    with gw.open(rgbn) as src:
        print(src)

Force the output data type.

.. ipython:: python

    with gw.open(rgbn, dtype='float32') as src:
        print(src.dtype)

Specify band names.

.. ipython:: python

    with gw.open(rgbn, band_names=['blue', 'green', 'red', 'nir']) as src:
        print(src.band)

Use the sensor name to set band names.

.. ipython:: python

    with gw.config.update(sensor='qb'):
        with gw.open(rgbn) as src:
            print(src.band)

To open multiple images stacked by bands, use a list of files with ``stack_dim='band'``.

Open a list of files as a DataArray, with all bands stacked.

.. ipython:: python

    with gw.open([rgbn, rgbn],
                 band_names=['b1', 'g1', 'r1', 'n1', 'b2', 'g2', 'r2', 'n2'],
                 stack_dim='band') as src:
        print(src)

To open multiple images as a time stack, change the input to a list of files.

Open a list of files as a DataArray.

.. ipython:: python

    with gw.open([rgbn, rgbn],
                 band_names=['blue', 'green', 'red', 'nir'],
                 time_names=['t1', 't2']) as src:
        print()

If `time_names` is not provided, GeoWombat will attempt to parse date strings using `dateparser.search.search_dates <https://dateparser.readthedocs.io/en/latest/>`_.

.. ipython:: python

    import os
    from geowombat.data import rgbn_time_list

    print('\n', ', '.join([os.path.basename(fn) for fn in rgbn_time_list]))

    with gw.config.update(sensor='rgbn'):
        with gw.open(rgbn_time_list) as src:
            print(src.time)

.. note::

    Xarray will handle alignment of images of varying sizes as long as the the resolutions are "target aligned". If images are not target aligned, Xarray might not concatenate a stack of images. With GeoWombat, we can use a context manager and a reference image to handle image alignment.

In the example below, we specify a reference image using GeoWombat's configuration manager.

.. note::

    The two images in this example are identical. The point here is just to illustrate the use of the configuration manager.

.. ipython:: python

    # Use an image as a reference for grid alignment and CRS-handling
    #
    # Within the configuration context, every image
    # in concat_list will conform to the reference grid.
    concat_list = [rgbn, rgbn]
    with gw.config.update(ref_image=rgbn):
        with gw.open(concat_list,
                     band_names=['blue', 'green', 'red', 'nir'],
                     time_names=['t1', 't2']) as src:
            print(src)

Stack the intersection of all images.

.. ipython:: python

    concat_list = [rgbn, rgbn_subb, rgbn_suba]
    with gw.open(concat_list,
                 band_names=['blue', 'green', 'red', 'nir'],
                 time_names=['t1', 't2', 't3'],
                 bounds_by='intersection') as src:
        print(src)

Stack the union of all images.

.. ipython:: python

    concat_list = [rgbn, rgbn_subb, rgbn_suba]
    with gw.open(concat_list,
                 band_names=['blue', 'green', 'red', 'nir'],
                 time_names=['t1', 't2', 't3'],
                 bounds_by='union') as src:
        print(src)

Keyword arguments always overwrite config settings. In this example, the reference image 'rgbn' is used to set the
CRS, bounds, and cell size. Using ``bounds_by='intersection'`` overrides the reference image bounds.

.. ipython:: python

    concat_list = [rgbn, rgbn_subb, rgbn_suba]
    with gw.config.update(ref_image=rgbn):
        with gw.open(concat_list,
                     band_names=['blue', 'green', 'red', 'nir'],
                     time_names=['t1', 't2', 't3'],
                     bounds_by='intersection') as src:
            print(src)

When multiple images have matching dates, the arrays are merged into one layer.

.. ipython:: python

    concat_list = [rgbn_suba, rgbn_subb, rgbn_suba]
    with gw.open(concat_list,
                 band_names=['blue', 'green', 'red', 'nir'],
                 time_names=['t1', 't1', 't2']) as src:
        print(src)

Use search wildcards to open a list of images.

.. code:: python

    with gw.open('*sub*.tif', band_names=['blue', 'green', 'red', 'nir']) as src:
        print(src)

Image mosaicking
----------------

Mosaic the two subsets into a single DataArray. If the images in the mosaic list have the same CRS, no configuration
is needed.

.. ipython:: python

    with gw.open([rgbn_suba, rgbn_subb],
                 band_names=['b', 'g', 'r', 'n'],
                 mosaic=True) as src:
        print(src)

If the images in the mosaic list have different CRSs, use a context manager to warp to a common grid.

.. note::

    The two images in this example have the same CRS. The point here is just to illustrate the use of the configuration manager.

.. ipython:: python

    # Use a reference CRS
    with gw.config.update(ref_image=rgbn):
        with gw.open([rgbn_suba, rgbn_subb],
                     band_names=['b', 'g', 'r', 'n'],
                     mosaic=True,
                     chunks=512) as src:
            print(src)

Mosaicking by the intersection of images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. ipython:: python

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(dpi=200)

    with gw.open([rgbn_suba, rgbn_subb],
                 chunks=64,
                 mosaic=True,
                 bounds_by='intersection') as src:

        src.sel(band=[4, 3, 2]).plot.imshow(ax=ax)
        src.gw.chunk_grid.plot(color='none', edgecolor='white', ls='-', lw=0.5, ax=ax)
        src.gw.footprint_grid.plot(color='none', edgecolor='r', ax=ax)

        ax.set_ylim(src.gw.footprint_grid.total_bounds[1]-10, src.gw.footprint_grid.total_bounds[3]+10)
        ax.set_xlim(src.gw.footprint_grid.total_bounds[0]-10, src.gw.footprint_grid.total_bounds[2]+10)

    ax.set_title('Image union', size=12)
    plt.tight_layout(pad=1)

Mosaicking by the union of images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. ipython:: python

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(dpi=200)

    with gw.open([rgbn_suba, rgbn_subb],
                 chunks=64,
                 mosaic=True,
                 bounds_by='union') as src:

        src.sel(band=[4, 3, 2]).plot.imshow(ax=ax)
        src.gw.chunk_grid.plot(color='none', edgecolor='white', ls='-', lw=0.5, ax=ax)
        src.gw.footprint_grid.plot(color='none', edgecolor='r', ax=ax)

        ax.set_ylim(src.gw.footprint_grid.total_bounds[1]-10, src.gw.footprint_grid.total_bounds[3]+10)
        ax.set_xlim(src.gw.footprint_grid.total_bounds[0]-10, src.gw.footprint_grid.total_bounds[2]+10)

    ax.set_title('Image union', size=12)
    plt.tight_layout(pad=1)

Writing DataArrays to file
--------------------------

GeoWombat's I/O can be accessed through the :func:`to_vrt` and :func:`to_raster` functions. These functions use
Rasterio's :func:`write` and Dask.array :func:`store` functions as I/O backends. In the examples below,
``src`` is an ``xarray.DataArray`` with the necessary transform information to write to an image file.

Write to a VRT file.

.. code:: python

    import geowombat as gw

    # Transform the data to lat/lon
    with gw.config.update(ref_crs=4326):

        with gw.open(rgbn, chunks=1024) as src:

            # Write the data to a VRT
            src.gw.to_vrt('lat_lon_file.vrt')

Write to a raster file.

.. code:: python

    import geowombat as gw

    with gw.open(rgbn, chunks=1024) as src:

        # Xarray drops attributes
        attrs = src.attrs.copy()

        # Apply operations on the DataArray
        src = src * 10.0

        src.attrs = attrs

        # Write the data to a GeoTiff
        src.gw.to_raster('output.tif',
                         verbose=1,
                         n_workers=4,    # number of process workers sent to ``concurrent.futures``
                         n_threads=2,    # number of thread workers sent to ``dask.compute``
                         n_chunks=200)   # number of window chunks to send as concurrent futures
