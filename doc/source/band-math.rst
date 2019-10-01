.. _band_math:

Use named coordinates for band math
===================================

.. ipython:: python

    import geowombat as gw

.. ipython:: python

    with gw.config.update(sensor='rgb'):
        with gw.open('example.nc', chunks={'wavelength': 1, 'y': 500, 'x': 500}) as ds:
            print(ds.gw.wavelengths)
            print(ds.gw.sensor)
        #
        #    evi2 = ds.gw.evi2()
        #    ndvi = ds.gw.ndvi().compute()