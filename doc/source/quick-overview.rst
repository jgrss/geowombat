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

File opening with GeoWombat
---------------------------

:py:class:`GeoWombat` has its own file opening API (:ref:`io`):

.. ipython:: python

    from geowombat.data import rgbn

    with gw.open(rgbn) as ds:
        print(ds)
        print(ds.gw)
