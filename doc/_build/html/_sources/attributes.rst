.. _attributes:

Raster attributes
=================

GeoWombat has additional raster attributes on top of Xarray/Rasterio attributes
-------------------------------------------------------------------------------

.. ipython:: python

    import geowombat as gw
    from geowombat.data import rgbn

Access raster attributes using the file opening API (:ref:`io`).

.. ipython:: python

    with gw.open(rgbn) as ds:
        print(ds.gw.meta)
        print(ds.gw.ndims)
        print(ds.gw.nbands)
        print(ds.gw.nrows)
        print(ds.gw.ncols)
        print(ds.gw.left)
        print(ds.gw.right)
        print(ds.gw.top)
        print(ds.gw.bottom)
        print(ds.gw.bounds)
        print(ds.gw.geometry)
