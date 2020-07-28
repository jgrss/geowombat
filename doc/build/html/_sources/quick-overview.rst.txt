.. _quick-overview:

Quick overview
==============

Here are some quick examples of what you can do with :py:class:`GeoWombat` objects.

To begin, import `GeoWombat` and `Xarray`:

.. ipython:: python

    import geowombat as gw
    import numpy as np
    import xarray as xr

Open a DataArray with a GeoWombat accessor
------------------------------------------

Any DataArray or Dataset will have the GeoWombat accessors appended:

.. ipython:: python

    data = xr.DataArray(np.random.randn(2, 3),
                        dims=('x', 'y'),
                        coords={'x': [10, 20]})
    print(data)
    print(data.gw)

Open a file
-----------

:py:class:`GeoWombat` has its own file opening API (:ref:`io`):

.. ipython:: python

    from geowombat.data import rgbn

    with gw.open(rgbn) as src:
        print(src)
        print(src.gw)

Write a raster
--------------

.. code:: python

    import geowombat as gw

    with gw.open(l8_224077_20200518_B4, chunks=1024) as src:

        # Xarray drops attributes
        attrs = src.attrs.copy()

        # Apply operations on the DataArray
        src = src * 10.0
        src.attrs = attrs

        # Write the data to a GeoTiff
        src.gw.to_raster('output.tif',
                         verbose=1,
                         n_workers=4,    # number of process workers sent to ``concurrent.futures``
                         n_threads=2,    # number of thread workers sent to ``dask.compute``
                         n_chunks=200)   # number of window chunks to send as concurrent futures
