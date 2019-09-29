.. _band_math:

Use named coordinates for band math
===================================

.. ipython:: python

    import geowombat as gw

.. ipython:: python

    #with gw.open('example.nc', chunks={'wavelength': 1, 'y': 500, 'x': 500}) as ds:
    #
    #    ds.gw.set_sensor('l8')
    #
    #    evi2 = ds.gw.evi2()
    #    ndvi = ds.gw.ndvi().compute()