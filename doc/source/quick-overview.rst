.. _quick-overview:

Quick overview
==============

Here are some quick examples of what you can do with ``geowombat`` objects.

To begin, import GeoWombat and Xarray:

.. ipython:: python

    import geowombat as gw
    import numpy as np
    import xarray as xr

Instantiate an xarray.DataArray with a GeoWombat accessor
---------------------------------------------------------

Any :class:`xarray.DataArray`` will have the ``geowombat`` accessors appended. That is, the :class:`xarray.DataArray` will have
an :class:`xarray.DataArray.gw` object.

.. ipython:: python

    data = xr.DataArray(
        np.random.randn(2, 3),
        dims=('x', 'y'),
        coords={'x': [10, 20]}
    )
    print(data)
    print(data.gw)

Open a raster file
------------------

GeoWombat has its own file opening API through :func:`geowombat.open` (see :ref:`io` for details).

.. ipython:: python

    from geowombat.data import rgbn

    with gw.open(rgbn) as src:
        print(src)
        print(src.gw)

Write a raster
--------------

Save an :class:`xarray.DataArray` to file with :func:`geowombat.save`.

.. code:: python

    import geowombat as gw

    with gw.open(l8_224077_20200518_B4, chunks=1024) as src:

        # Xarray drops attributes
        attrs = src.attrs.copy()

        # Apply operations on the DataArray
        src = src * 10.0
        src.attrs = attrs

        # Write the data to a GeoTiff
        src.gw.save('output.tif', num_workers=4)
