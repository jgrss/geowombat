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
    with gw.config.update(ref_image=rgbn):
        concat_list = [rgbn, rgbn]
        with gw.open(concat_list,
                     band_names=['blue', 'green', 'red', 'nir'],
                     time_names=['t1', 't2']) as ds:
            print(ds)

Image mosaicking
++++++++++++++++

Images can be mosaicked into a single DataArray

.. ipython:: python

    # Load two images that partially overlap
    from geowombat.data import rgbn_suba, rgbn_subb

Examine the first image subset

.. ipython:: python

    with gw.open(rgbn_suba, band_names=['b', 'g', 'r', 'n']) as ds:
        print(ds)

Examine the second image subset

.. ipython:: python

    with gw.open(rgbn_subb, band_names=['b', 'g', 'r', 'n']) as ds:
        print(ds)

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
                     mosaic=True) as ds:
            print(ds)

Writing DataArrays to file
++++++++++++++++++++++++++

In the example below, ``ds_res`` is an :func:`xarray.DataArray`. Using Rasterio as a backend, we use the
Xarray accessor :func:`to_raster` to write chunks in parallel.

.. ipython:: python

    #with gw.open(rgbn, chunks=(1, 1024, 1024)) as ds:
    #    ds_res = <do something>
    #    ds_res.gw.to_raster('output.tif', n_jobs=4)

Write to GeoTiff on a Dask distributed cluster

.. ipython:: python

    from geowombat.util import Cluster

    cluster = Cluster(n_workers=8,
                      threads_per_worker=1,
                      scheduler_port=0,
                      processes=False)

    #cluster.start()
    #
    #with joblib.parallel_backend('dask'):
    #
    #    with gw.open('example.tif') as ds:
    #
    #        # ds = <do something>
    #
    #        ds.gw.to_raster('output.tif',
    #                        n_jobs=8,
    #                        row_chunks=512,
    #                        col_chunks=512,
    #                        pool_chunksize=50,
    #                        tiled=True,
    #                        blockxsize=2048,
    #                        blockysize=2048,
    #                        compress='lzw')
    #
    #cluster.stop()
