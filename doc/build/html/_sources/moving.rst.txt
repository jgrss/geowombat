.. _moving:

Two-dimensional moving windows
==============================

Examine the :func:`geowombat.moving` help
+++++++++++++++++++++++++++++++++++++++++

.. ipython:: python

    import geowombat as gw

    print(help(gw.moving))

Calculate the local average
+++++++++++++++++++++++++++

.. code:: python

    import geowombat as gw
    from geowombat.data import rgbn

    with gw.open(rgbn, chunks=512) as src:

        res = src.gw.moving(stat='mean', w=5, n_jobs=4, nodata=0)

        # Compute results
        res.data.compute()

The moving window function uses Dask to partition chunks and calculate a statistic for the chunk. Calling :func:`geowombat.to_raster` on the Xarray object will result in `concurrent.futures` being unable to pickle the underlying worker function.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

A workaround is to compute the results before writing to file

.. code:: python

    import xarray as xr
    import dask.array as da

    with gw.open(rgbn, chunks=512) as src:

        res = src.gw.moving(stat='mean', w=5, n_jobs=4, nodata=0)

        # Compute the moving window and save as an Xarray
        res = xr.DataArray(data=da.from_array(res.data.compute(num_workers=4), chunks=src.data.chunks),
                           dims=('band', 'y', 'x'),
                           coords={'band': src.band.values.tolist(),
                                   'y': src.y,
                                   'x': src.x},
                           attrs=src.attrs)

        # Write the results to file
        res.gw.to_raster('output.tif', n_workers=4, n_threads=1)
