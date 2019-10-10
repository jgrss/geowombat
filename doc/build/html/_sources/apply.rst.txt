.. _apply:

Applying custom user functions
==============================

With functions that release the GIL (e.g., many NumPy functions, Cython), one can bypass Xarray and use Rasterio to write concurrently
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The example below applies a custom function concurrently over a large image.

.. code:: python

    def my_func(block, arg):
        return block * arg


.. code:: python

    import geowombat as gw

    gw.apply('input.tif', 'output.tif', my_func, args=(10.0,), n_jobs=4)
