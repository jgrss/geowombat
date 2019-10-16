.. _io:

Reading and writing files
=========================

File opening with GeoWombat uses the :func:`geowombat.open` function to open raster files
-----------------------------------------------------------------------------------------

.. ipython:: python

    # Import GeoWombat
    import geowombat as gw

.. ipython:: python

    # Load a 4-band test image
    from geowombat.data import rgbn

.. ipython:: python

    # Load two images that partially overlap
    from geowombat.data import rgbn_suba, rgbn_subb

To open individual images, GeoWombat wraps :func:`xarray.open_rasterio` and :func:`xarray.open_dataset`:
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Open a raster as a DataArray

.. ipython:: python

    with gw.open(rgbn) as ds:
        print(ds)

Open a raster as a Dataset

.. ipython:: python

    with gw.open(rgbn, return_as='dataset') as ds:
        print(ds)

Specify band names

.. ipython:: python

    with gw.open(rgbn, band_names=['blue', 'green', 'red', 'nir']) as ds:
        print(ds)

To open multiple images as a time stack, change the input to a list of files
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Open a list of files as a DataArray

.. ipython:: python

    with gw.open([rgbn, rgbn],
                 band_names=['blue', 'green', 'red', 'nir'],
                 time_names=['t1', 't2']) as ds:
        print(ds)

Xarray will handle alignment of images of varying sizes as long as the the resolutions are "target aligned". If images are
not target aligned, Xarray might not concatenate a stack of images. With GeoWombat, we can use a context manager and
a reference image to handle image alignment.

In the example below, we specify a reference image using GeoWombat's configuration manager:

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
                     time_names=['t1', 't2']) as ds:
            print(ds)

Stack the intersection of all images

.. ipython:: python

    concat_list = [rgbn, rgbn_subb, rgbn_suba]
    with gw.open(concat_list,
                 band_names=['blue', 'green', 'red', 'nir'],
                 time_names=['t1', 't2', 't3'],
                 how='intersection') as ds:
        print(ds)

Stack the union of all images

.. ipython:: python

    concat_list = [rgbn, rgbn_subb, rgbn_suba]
    with gw.open(concat_list,
                 band_names=['blue', 'green', 'red', 'nir'],
                 time_names=['t1', 't2', 't3'],
                 how='union') as ds:
        print(ds)

Keyword arguments always overwrite config settings. In this example, the reference image 'rgbn' is used to set the
CRS, bounds, and cell size. Using how='intersection' overrides the reference image bounds.

.. ipython:: python

    concat_list = [rgbn, rgbn_subb, rgbn_suba]
    with gw.config.update(ref_image=rgbn):
        with gw.open(concat_list,
                     band_names=['blue', 'green', 'red', 'nir'],
                     time_names=['t1', 't2', 't3'],
                     how='intersection') as ds:
            print(ds)

When multiple images have matching dates, the arrays are merged into one layer

.. ipython:: python

    concat_list = [rgbn_suba, rgbn_subb, rgbn_suba]
    with gw.open(concat_list,
                 band_names=['blue', 'green', 'red', 'nir'],
                 time_names=['t1', 't1', 't2']) as ds:
        print(ds)

Use search wildcards to open a list of images

.. ipython:: python

    import os
    search = os.path.join(os.path.dirname(rgbn), '*sub*.tif')

.. ipython:: python

    with gw.open(search,
                 band_names=['blue', 'green', 'red', 'nir']) as ds:
        print(ds)

Image mosaicking
++++++++++++++++

Mosaic the two subsets into a single DataArray. If the images in the mosaic list have the same CRS, no configuration
is needed.

.. ipython:: python

    with gw.open([rgbn_suba, rgbn_subb],
                 band_names=['b', 'g', 'r', 'n'],
                 mosaic=True) as ds:
        print(ds)

If the images in the mosaic list have different CRSs, use a context manager to warp to a common grid.

.. note::

    The two images in this example have the same CRS. The point here is just to illustrate the use of the configuration manager.

.. ipython:: python

    # Use a reference CRS
    with gw.config.update(ref_image=rgbn):
        with gw.open([rgbn_suba, rgbn_subb],
                     band_names=['b', 'g', 'r', 'n'],
                     mosaic=True,
                     chunks=512) as ds:
            print(ds)

Writing DataArrays to file
++++++++++++++++++++++++++

In the example below, ``ds`` is an ``xarray.DataArray``. Using Rasterio's :func:`write` and Dask.array :func:`store`
as backends, we use the Xarray accessor :func:`to_raster` to write array chunks in parallel.

.. code:: python

    import geowombat as gw

    with gw.open(rgbn, chunks=1024) as ds:

        dss = ds * 10.0

        # Xarray drops attributes
        dss.attrs = ds.attrs

        # Write the data
        dss.gw.to_raster('output.tif',
                         verbose=1,
                         n_worker=4,
                         n_threads=2,
                         use_client=True)
