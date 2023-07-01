.. _apply:

User functions
==============

User apply
----------

With functions that release the GIL (e.g., many NumPy functions, Cython), one can use ``rasterio`` to write concurrently.

The example below applies a custom function concurrently over an image, where each block of data is multiplied by ``arg``.

.. note::

    GeoWombat will not handle image alignment with the :func:`geowombat.apply` function.

.. code:: python

    def my_func(w, block, arg):
        return w, block * arg

.. code:: python

    import geowombat as gw

    gw.apply('input.tif', 'output.tif', my_func, args=(10.0,), n_jobs=4)

User functions as DataArray attributes
--------------------------------------

User functions that do not use a ``dask`` task graph can be passed as attributes. Unlike the example above, the
example below has guaranteed image alignment. Functions and arguments can be passed as Xarray attributes.
Here is an example that uses one user argument.

.. code:: python

    import geowombat as gw

    # Function with one argument
    def user_func(block, n):
        return block * n

    with gw.open('input.tif') as ds:

        # Functions are given as 'apply'
        ds.attrs['apply'] = user_func

        # Function arguments (n) are given as 'apply_args'
        ds.attrs['apply_args'] = [10.0]

        ds.gw.save(
            'output.tif',
            num_workers=2,
            overwrite=True,
            compress='lzw'
        )

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
        ds.attrs['apply_kwargs'] = {'divider': 2.3}

        ds.gw.save(
            'output.tif',
            num_workers=2,
            overwrite=True,
            compress='lzw'
        )

Applying in-memory GeoWombat functions lazily
---------------------------------------------

Several ``geowombat`` functions execute in-memory, and are therefore not optimized for large datasets. However, these
functions can be applied at the block level for ``dask``-like out-of-memory processing using the user function framework.
In the example below, :func:`geowombat.polygon_to_array` is applied at the raster block level.

.. code:: python

    import geowombat as gw
    import geopandas as gpd

    # Confirm that the GeoWombat function is supported for block-level lazy processing
    print(hasattr(gw.polygon_to_array, 'wombat_func_'))

    with gw.open('input.tif') as src:

        # We can load the geometry spatial index once and pass it to the block level.
        # However, be sure that the CRS matches the raster CRS.
        df = gpd.gpd.read_file('vector.gpkg').to_crs(src.crs)
        sindex = df.sindex

        src.attrs['apply'] = gw.polygon_to_array

        # All arguments must be passed as keyword arguments
        src.attrs['apply_kwargs'] = {
            'polygon': df,
            'sindex': sindex,
            'all_touched': False
        }

        src.gw.save(
            'output.tif',
            num_workers=2,
            compress='lzw'
        )

By default, user functions expect a NumPy array as the first argument. It might be desirable to combine a ``geowombat``
function that operates on an :class:`xarray.DataArray`. To achieve this, we can decorate the function.

.. code:: python

    import geowombat as gw
    from geowombat.core.util import lazy_wombat

    @lazy_wombat
    def user_func(data=None, polygon=None, sindex=None, all_touched=None):
        """Converts a polygon to an array and then masks the array"""
        mask = gw.polygon_to_array(polygon=polygon, data=data, sindex=sindex, all_touched=all_touched)
        return (mask * data).astype('float64')

    with gw.open('input.tif') as src:

        df = gpd.gpd.read_file('vector.gpkg').to_crs(src.crs)
        sindex = df.sindex

        src.attrs['apply'] = user_func

        # All arguments must be passed as keyword arguments
        src.attrs['apply_kwargs'] = {
            'polygon': df,
            'sindex': sindex,
            'all_touched': False
        }

        src.gw.save(
            'output.tif',
            num_workers=2,
            compress='lzw'
        )

The above example is similar to the following with the :func:`geowombat.mask` function.

.. code:: python

    import geowombat as gw

    with gw.open('input.tif') as src:

        df = gpd.gpd.read_file('vector.gpkg').to_crs(src.crs)
        sindex = df.sindex

        src.attrs['apply'] = gw.mask

        # All arguments must be passed as keyword arguments
        src.attrs['apply_kwargs'] = {
            'dataframe': df,
            'keep': 'in'
        }

        src.gw.save(
            'output.tif',
            num_workers=2,
            compress='lzw'
        )
