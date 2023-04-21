.. _quick-overview:

Quick overview
==============

Here are some quick examples of what you can do with GeoWombat objects.

To begin, import GeoWombat and Xarray:

.. ipython:: python

    import geowombat as gw
    import numpy as np
    import xarray as xr

Open a DataArray with a GeoWombat accessor
------------------------------------------

Any DataArray or Dataset will have the GeoWombat accessors appended:

.. ipython:: python

    data = xr.DataArray(
        np.random.randn(2, 3),
        dims=('x', 'y'),
        coords={'x': [10, 20]}
    )
    print(data)
    print(data.gw)

Open a file
-----------

GeoWombat has its own file opening API (:ref:`io`):

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
        src.gw.save('output.tif', num_workers=4)
