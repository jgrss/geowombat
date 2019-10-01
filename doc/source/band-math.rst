.. _band_math:

Use named coordinates for band math
===================================

.. ipython:: python

    import geowombat as gw
    from geowombat.data import rgbn

Calculate a vegetation index, returning an `Xarray.DataArray`

.. ipython:: python

    with gw.open(rgbn) as ds:
        print(ds)
        evi = ds.gw.evi(sensor='rgbn', scale_factor=0.0001)
        print(evi)

Use the configuration context to set parameters

.. ipython:: python

    with gw.config.update(sensor='rgbn', scale_factor=0.0001):
        with gw.open(rgbn) as ds:
            evi = ds.gw.evi()
            print(evi)

Load the data to return a `NumPy` array

.. ipython:: python

    with gw.config.update(sensor='rgbn', scale_factor=0.0001):
        with gw.open(rgbn, band_names=['blue', 'green', 'red', 'nir']) as ds:
            evi2 = ds.gw.evi2().squeeze().load().data
            print(evi2)

Use the generic :func:`norm_diff` function with any two-band combination

.. ipython:: python

    with gw.config.update(sensor='rgbn'):
        with gw.open(rgbn, band_names=['blue', 'green', 'red', 'nir']) as ds:
            d = ds.gw.norm_diff('red', 'nir')
            print(d)
