.. _apply:

Applying custom user functions
==============================

With functions that release the GIL (e.g., many NumPy functions, Cython), one can bypass Xarray and use Rasterio to write concurrently.
---------------------------------------------------------------------------------------------------------------------------------------

The example below applies a custom function concurrently over an image.

.. note::

    GeoWombat will not handle image alignment with the :func:`geowombat.apply` function.

.. code:: python

    def my_func(block, arg):
        return block * arg

.. code:: python

    import geowombat as gw

    gw.apply('input.tif', 'output.tif', my_func, args=(10.0,), n_jobs=4)

User functions as attributes
----------------------------

User functions that do not use a Dask task graph can be passed as attributes. Unlike the example above, the example below has guaranteed image alignment. Functions and arguments can be passed as `Xarray` attributes. Here is an example that uses one user argument.

.. code:: python

    import geowombat as gw

    # Function with one argument
    def user_func(block, n):
        return block * n

    with gw.open('input.tif') as ds:

        # Functions are given as 'apply'
        ds.attrs['apply'] = user_func

        # Function arguments are given as 'apply_args'
        ds.attrs['apply_args'] = [10.0]

        ds.gw.to_raster('output.tif',
                        n_workers=4,
                        n_threads=2,
                        separate=True,
                        overwrite=True,
                        compress='lzw')

In this example, a keyword argument is also used.

.. code:: python

    # Function with one argument and one keyword argument
    def user_func(block, n, divider=1.0):
        return (block * n) / divider

    with gw.open('input.tif') as ds:

        # Functions are given as 'apply'
        ds.attrs['apply'] = user_func

        # Function arguments are given as 'apply_args'
        # *Note that arguments should always be a list
        ds.attrs['apply_args'] = [10.0]

        # Function keyword arguments are given as 'apply_kwargs'
        # *Note that keyword arguments should always be a dictionary
        ds.attrs['apply_kwargs'] = {'divide': 2.3}

        ds.gw.to_raster('output.tif',
                        n_workers=4,
                        n_threads=2,
                        separate=True,
                        overwrite=True,
                        compress='lzw')

Applying in-memory GeoWombat functions lazily
---------------------------------------------

Several GeoWombat functions execute in-memory, and are therefore not optimized for large datasets. These GeoWombat functions can be applied at the block level.

.. code:: python

    import geowombat as gw
    import geopandas as gpd

    # Confirm that the function is supported for block-level processing
    print(hasattr(gw.polygon_to_array, 'wombat_func_'))

    # We can load the geometry spatial index once and pass it to the block level
    sindex = gpd.read_file('vector.gpkg').sindex

    with gw.open('input.tif') as src:

        src.attrs['apply'] = gw.polygon_to_array

        # All arguments must be passed as keyword arguments
        src.attrs['apply_kwargs'] = {'polygon': 'vector.gpkg',
                                     'sindex': sindex,
                                     'all_touched': False}

        src.gw.to_raster('output.tif',
                         n_workers=4,
                         n_threads=2,
                         compress='lzw')
