.. _config:

GeoWombat configuration
=======================

GeoWombat has a context manager (:class:`geowombat.config`) to assist in configurations
---------------------------------------------------------------------------------------

.. ipython:: python

    import geowombat as gw

Wrap functions in a context manager to control global arguments.

.. ipython:: python

    with gw.config.update(sensor='qb', scale_factor=0.0001):
        with gw.open(rgbn) as ds:
            for k, v in ds.gw.config.items():
                print(k, v)

    with gw.config.update(sensor='ps', tiled=False):
        with gw.open(rgbn) as ds:
            for k, v in ds.gw.config.items():
                print(k, v)

    with gw.open(rgbn) as ds:
        for k, v in ds.gw.config.items():
            print(k, v)
