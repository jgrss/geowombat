.. _io:

Reading and writing files
=========================

File opening with GeoWombat
---------------------------

Import GeoWombat

.. ipython:: python

    import geowombat as gw
    from geowombat.data import rgbn

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

Open a list of files as a DataArray

.. ipython:: python

    with gw.open([rgbn, rgbn],
                 band_names=['blue', 'green', 'red', 'nir'],
                 time_names=['t1', 't2']) as ds:
        print(ds)

Write to GeoTiff on a Dask distributed cluster

.. ipython:: python

    from geowombat.util import Cluster
    import joblib

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
