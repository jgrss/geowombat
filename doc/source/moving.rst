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
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

A workaround is to pass :func:`geowombat.moving` as an Xarray attribute.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

    Note that there are three levels of parallelism in the example below: 1) The `n_jobs` argument passed as an Xarray attribute determines the number of image chunk rows to process in parallel. 2) The `n_workers` argument in the :func:`geowombat.to_raster` function determines the number of chunks to process in parallel. 3) The `n_threads` argument in the :func:`geowombat.to_raster` determines the number of thread workers to use for the Dask computation. The threads are set at 1 here because the computation happens in the :func:`geowombat.moving` function.

.. code:: python

    with gw.open(rgbn, chunks=512) as src:

        src.attrs['apply'] = gw.moving
        src.attrs['apply_kwargs'] = {'stat': 'mean', 'w': 5, 'n_jobs': 4, 'nodata': 0}

        # Write the results to file
        src.gw.to_raster('output.tif', n_workers=4, n_threads=1)
