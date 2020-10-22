.. _io-distributed:

Distributed processing
======================

One of the key features of GeoWombat is the ability to write Dask/Xarray tasks to file in a concurrent workflow. Below are
several examples illustrating this process.

Import GeoWombat and Dask

.. ipython:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518, l8_224078_20200518_B2, l8_224078_20200518_B3, l8_224078_20200518_B4

Dask diagnostics

.. ipython:: python

    from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler
    from dask.diagnostics import visualize

Use Dask to compute with parallel workers
-----------------------------------------

.. note::

    These examples illustrate the Dask concurrency over a DataArray task. An example showing how to write results
    to file in parallel are shown at the bottom of the page.

Chunk sizes of 64x64 require many reads for a simple calculation.

.. code:: python

    with Profiler() as prof, \
        ResourceProfiler(dt=0.25) as rprof, \
            CacheProfiler() as cprof:

        # Set the sensor name
        with gw.config.update(sensor='bgr'):

            with gw.open(l8_224078_20200518, chunks=64) as src:

                # Normalized difference index
                ndi = gw.norm_diff(src, 'green', 'red', nodata=0, scale_factor=0.0001)

                # Send the task to dask
                results = ndi.data.compute(num_workers=4)

.. code:: python

    prof.visualize()

.. raw:: html

    <iframe src="_static/profile_chunks64_4workers.html"
            marginwidth="0" marginheight="0" scrolling="no"
            width="650" height="300" style="border:none"></iframe>

Chunk sizes of 256x256 reduce the number of file reads.

.. code:: python

    with Profiler() as prof, \
        ResourceProfiler(dt=0.25) as rprof, \
            CacheProfiler() as cprof:

        # Set the sensor name
        with gw.config.update(sensor='bgr'):

            with gw.open(l8_224078_20200518, chunks=256) as src:

                # Normalized difference index
                ndi = gw.norm_diff(src, 'green', 'red', nodata=0, scale_factor=0.0001)

                # Send the task to dask
                results = ndi.data.compute(num_workers=4)

.. code:: python

    prof.visualize()

.. raw:: html

    <iframe src="_static/profile_chunks256_4workers.html"
            marginwidth="0" marginheight="0" scrolling="no"
            width="650" height="300" style="border:none"></iframe>

Increase the number of parallel workers

.. note::

     The appropriate choice of chunk size is challenging and takes some practice. Start by reading `Dask Best Practices <https://docs.dask.org/en/latest/array-best-practices.html#select-a-good-chunk-size>`_. We find, however, that with some experimentation you can find a good chunk size for common tasks. One simple approach is to choose a chunk size that fills around 75-95% of memory on your system. Accidentally exceeding 100% of memory leads to significant slow-downs.

     If you decide to manually calculate how large chunks should be to utilize all resources, keep in mind that "Dask will often have as many chunks in memory as twice the number of active threads"
     `Orientation of chunks <https://docs.dask.org/en/latest/array-best-practices.html#select-a-good-chunk-size>`_ is also critical, especially if dealing with multiple bands or a time series of images. Chunks in this case should have three dimensions ([bands, y, x] or [time, bands, y, x]). So, a five-period image stack with a single band might have a chunk size of [5, 1, 256, 256]. Proper orientation will reduce the need to read the same data more than once.

.. code:: python

    with Profiler() as prof, \
        ResourceProfiler(dt=0.25) as rprof, \
            CacheProfiler() as cprof:

        # Set the sensor name
        with gw.config.update(sensor='bgr'):

            with gw.open(l8_224078_20200518, chunks=256) as src:

                # Normalized difference index
                ndi = gw.norm_diff(src, 'green', 'red', nodata=0, scale_factor=0.0001)

                # Send the task to dask
                results = ndi.data.compute(num_workers=8)

.. code:: python

    prof.visualize()

.. raw:: html

    <iframe src="_static/profile_chunks256_8workers.html"
            marginwidth="0" marginheight="0" scrolling="no"
            width="650" height="300" style="border:none"></iframe>

Increase the complexity of the parallel task
--------------------------------------------

Open bands as separate files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. ipython:: python

    chunks = 256

    with Profiler() as prof, \
        ResourceProfiler(dt=0.25) as rprof, \
            CacheProfiler() as cprof:
        with gw.open(l8_224078_20200518_B2, band_names=['blue'], chunks=chunks) as src_b2, \
            gw.open(l8_224078_20200518_B3, band_names=['green'], chunks=chunks) as src_b3, \
                gw.open(l8_224078_20200518_B4, band_names=['red'], chunks=chunks) as src_b4:
            # Mask 'no data' values and scale the data
            t2 = src_b2.gw.set_nodata(0, 65535, (0, 1), 'float64', scale_factor=0.0001)
            t3 = src_b3.gw.set_nodata(0, 65535, (0, 1), 'float64', scale_factor=0.0001)
            t4 = src_b4.gw.set_nodata(0, 65535, (0, 1), 'float64', scale_factor=0.0001)
            task = (t2.sel(band='blue') * t3.sel(band='green') * t4.sel(band='red')).expand_dims(dim='band').assign_coords({'band': ['results']})
            print(task)
            results = task.data.compute(num_workers=8)

.. code:: python

    prof.visualize()

.. raw:: html

    <iframe src="_static/multi-band_task.html"
            marginwidth="0" marginheight="0" scrolling="no"
            width="650" height="300" style="border:none"></iframe>

Open bands as a stacked array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. ipython:: python

    chunks = 256

    with Profiler() as prof, \
        ResourceProfiler(dt=0.25) as rprof, \
            CacheProfiler() as cprof:
        with gw.config.update(sensor='bgr'):
            with gw.open([l8_224078_20200518_B2, l8_224078_20200518_B3, l8_224078_20200518_B4], stack_dim='band', chunks=chunks) as src:
                attrs = src.attrs.copy()
                # Mask 'no data' values and scale the data
                t = src.gw.set_nodata(0, 65535, (0, 1), 'float64', scale_factor=0.0001)
                task = (t.sel(band='blue') * t.sel(band='green') * t.sel(band='red')).expand_dims(dim='band').assign_coords({'band': ['results']})
                task.attrs = attrs
                print(task)
                results = task.data.compute(num_workers=8)

.. code:: python

    prof.visualize()

.. raw:: html

    <iframe src="_static/multi-band_stack_task.html"
            marginwidth="0" marginheight="0" scrolling="no"
            width="650" height="300" style="border:none"></iframe>

Use GeoWombat to write a task to file
-------------------------------------

In the previous examples, the call to ``dask`` :func:`compute` lets ``dask`` manage the task distribution. When writing results
to file with :func:`geowombat.to_raster`, individual chunks are managed in a parallel process using `concurrent.futures <https://docs.python.org/3/library/concurrent.futures.html>`_.
While there are many argument options to consider when calling :func:`geowombat.to_raster`, some of the key ones
are the :func:`geowombat.open` size of ``chunks`` and the :func:`to_raster` number of parallel ``n_workers`` and ``n_threads``.

.. note::

    When do I use workers versus threads? This probably depends on the problem being executed. If the computation task
    is mainly performing many reads at the chunk level (i.e., I/O bound) and the chunk-level process is relatively simple (i.e., the worker
    is not spending much time on each chunk) or the process can release the GIL, more ``n_threads`` might be more efficient. If the chunk-level computation is
    complex (i.e., CPU bound) and is the main bottleneck, more ``n_workers`` might be more efficient. See `Dask single-machine <https://docs.dask.org/en/latest/setup/single-machine.html>`_ for more details about threads vs. processes.

Writing results to file in a parallel environment can be performed on a laptop or a distributed compute system. With the
former, a call to :func:`geowombat.to_raster` is all that is needed. On a distributed compute system, one might instead use
a `distributed client <https://distributed.dask.org/en/latest/client.html>`_ to manage the task concurrency.

The code block below is a simple example that would use 8 threads within 1 process to write the task to a GeoTiff.

.. code:: python

    with gw.config.update(sensor='bgr'):
        with gw.open([l8_224078_20200518_B2, l8_224078_20200518_B3, l8_224078_20200518_B4], stack_dim='band', chunks=chunks) as src:

            attrs = src.attrs.copy()

            # Mask 'no data' values and scale the data
            t = src.gw.set_nodata(0, 65535, (0, 1), 'float64', scale_factor=0.0001)

            task = (t.sel(band='blue') * t.sel(band='green') * t.sel(band='red')).expand_dims(dim='band').assign_coords({'band': ['results']})
            task.attrs = attrs

            # The previous example using dask compute returns
            #   the results as a numpy array.
            # results = task.data.compute(num_workers=8)

            # Use geowombat to write the task to file where
            #   chunks are processed concurrently.
            task.gw.to_raster('results.tif', n_workers=1, n_threads=8, compress='lzw')

The same task might be executed on a distributed system in the following way.

.. code:: python

    with gw.config.update(sensor='bgr'):
        with gw.open([l8_224078_20200518_B2, l8_224078_20200518_B3, l8_224078_20200518_B4], stack_dim='band', chunks=chunks) as src:

            attrs = src.attrs.copy()

            # Mask 'no data' values and scale the data
            t = src.gw.set_nodata(0, 65535, (0, 1), 'float64', scale_factor=0.0001)

            task = (t.sel(band='blue') * t.sel(band='green') * t.sel(band='red')).expand_dims(dim='band').assign_coords({'band': ['results']})
            task.attrs = attrs

            # The previous example using dask compute returns
            #   the results as a numpy array.
            # results = task.data.compute(num_workers=8)

            # Use geowombat to write the task to file where
            #   chunks are processed concurrently.
            #
            # The results will be written under a distributed cluster environment.
            task.gw.to_raster('results.tif', use_client=True, n_workers=1, n_threads=8, compress='lzw')

Use GeoWombat to gather block-level results in parallel
-------------------------------------------------------

With :func:`geowombat.to_raster`, a Xarray/Dask task graph is executed in parallel and written to a raster file. If, however, you wish to retrieve values for each block without writing the entire blocks to file, use :class:`geowombat.core.parallel.ParallelTask`. In the example below, a custom function (`user_func`) is processed in parallel over each raster chunk/block.

.. code:: python

    import itertools
    import geowombat as gw
    from geowombat.core.parallel import ParallelTask

    def user_func(*args):

        """
        Block-level function to be executed in parallel. The first argument is the block data,
        the second argument is the block id, and the third argument is the number of parallel
        worker threads for dask.compute().
        """

        # Gather function arguments
        data, window_id, num_workers = list(itertools.chain(*args))

        # Send the computation to Dask
        return data.data.sum().compute(scheduler='threads', num_workers=num_workers)

    # Process 8 windows in parallel using threads
    # Process 1 dask chunks in parallel using threads
    # 8 total workers are needed
    with gw.open('image.tif', chunks=512) as src:

        # Each block is a 512x512 dask array
        # with chunks of 512x512
        pt = ParallelTask(src,
                          scheduler='threads',
                          n_workers=8)

        # There is only 1 chunk per block, so no
        # point in using multiple threads here
        res = pt.map(user_func, 1)

In the example above, :class:`geowombat.core.parallel.ParallelTask` reads row and column chunks of `src.gw.row_chunks` and `src.gw.col_chunks` size (which is set with :func:`geowombat.open`). Let's say we open a raster with chunks of 512x512. In the above example, the `data.data.sum().compute(scheduler='threads', num_workers=num_workers)` dask computation only has 1 chunk to process because the chunk sizes are the same size as the blocks being passed to `user_func`. We can specify a larger block size to read in parallel (the dask chunk size will remain the same) with **row_chunks** and **col_chunks**.

.. code:: python

    # Process 8 windows in parallel using threads
    # Process 4 dask chunks in parallel using threads
    # 32 total workers are needed
    with gw.open('image.tif', chunks=512) as src:

        # Each block is a 1024x1024 dask array
        # with chunks of 512x512
        pt = ParallelTask(src,
                          row_chunks=1024,
                          col_chunks=1024,
                          scheduler='threads',
                          n_workers=8)

        # Now, each block has 4 chunks, so we can use dask
        # to process them in parallel
        res = pt.map(user_func, 4)
