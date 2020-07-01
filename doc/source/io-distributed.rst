.. _io-distributed:

Distributed processing
======================

One of the key features of GeoWombat is the ability to write Dask/Xarray tasks to file in a concurrent workflow. Below are
several examples illustrating this process.

Import GeoWombat and Dask
-------------------------

.. ipython:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518, l8_224078_20200518_B2, l8_224078_20200518_B3, l8_224078_20200518_B4

Dask diagnostics

.. ipython:: python

    from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler
    from dask.diagnostics import visualize

Calculate a normalized difference index
---------------------------------------

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

Increase the complexity of the task
-----------------------------------

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

Writing computation results to file
-----------------------------------

In the previous examples, the call to ``dask`` :func:`compute` lets ``dask`` manage the task distribution. When writing results
to file with :func:`geowombat.to_raster`, individual chunks are managed in a parallel process using `concurrent.futures <https://docs.python.org/3/library/concurrent.futures.html>`_.
While there are many argument options to consider when calling :func:`geowombat.to_raster`, some of the key ones
are the :func:`geowombat.open` size of ``chunks`` and the :func:`to_raster` number of parallel ``n_workers`` and ``n_threads``.

.. note::

    When do I use workers versus threads? This probably depends on the problem being executed. If the computation task
    is mainly performing many reads at the chunk level and the chunk-level process is relatively simple (i.e., the worker
    is not spending much time on each chunk), more ``n_workers`` might be more efficient. If the chunk-level computation is
    complex and is the main bottleneck, more ``n_threads`` might be more efficient. See `Dask single-machine <https://docs.dask.org/en/latest/setup/single-machine.html>`_ for more details about threads vs. processes.

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