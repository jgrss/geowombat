.. _tutorial-gw:

GeoWombat DataArray accessor
============================

GeoWombat extends :class:`xarray.DataArray` through a registered ``.gw`` accessor.
Once an array is opened with :func:`geowombat.open`, every GeoWombat method and
property is reached through this accessor (for example ``src.gw.transform`` or
``src.gw.ndvi()``), which keeps the DataArray namespace clean while making the
functionality discoverable via tab-completion.

The properties below describe the geospatial attributes of the image. For the
complete list of accessor methods and properties, see the
:class:`~geowombat.core.geoxarray.GeoWombatAccessor` entry in the :ref:`api`.

DataArray GeoWombat attributes
------------------------------

.. ipython:: python

    import geowombat as gw
    from geowombat.data import rgbn

    with gw.open(rgbn) as src:
        print(src.gw.transform)
        print(src.gw.geodataframe)
        print(src.gw.meta)
        print(src.gw.ndims)
        print(src.gw.nbands)
        print(src.gw.nrows)
        print(src.gw.ncols)
        print(src.gw.row_chunks)
        print(src.gw.col_chunks)
        print(src.gw.left)
        print(src.gw.right)
        print(src.gw.top)
        print(src.gw.bottom)
        print(src.gw.bounds)
