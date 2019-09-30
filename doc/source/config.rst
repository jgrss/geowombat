.. _config:

GeoWombat configuration
=======================

GeoWombat has a context manager (:func:`geowombat.config`) to assist in configurations
------------------------------------------------------------------------------------

Import GeoWombat

.. ipython:: python

    import geowombat as gw

Wrap functions in a context manager to control global arguments

.. ipython:: python

    with gw.config.set(sensor='planetscope'):
        with gw.open(rgbn) as ds:
            print(ds.gw.config)
