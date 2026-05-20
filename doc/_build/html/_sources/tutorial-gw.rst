.. _tutorial-gw:

GeoWombat DataArray accessor
============================

The ``geowombat`` class has a number of attributes that define the properties of the image.

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
