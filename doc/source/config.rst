.. _config:

Configuration
=============

GeoWombat has a context manager (:class:`geowombat.config`) to assist in configurations
---------------------------------------------------------------------------------------

.. ipython:: python

    import geowombat as gw

Wrap functions in a context manager to control global arguments for particular sensors.

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


Available Configurations
------------------------

The following is a list of configurations for all sensors. This documentation may become out of date, if so
please refer to geowombat/core/properties.py for the full list.

.. list-table:: Title
   :widths: 25 75
   :header-rows: 1

   * - Abreviated Name
     - Description

   * - 'rgb'
     - red, green, and blue

   * - 'rgbn'
     - red, green, blue, and NIR

   * - 'bgr'
     - blue, green, and red

   * - 'bgrn'
     - blue, green, red, and NIR

   * - 'l5'
     - Landsat 5 Thematic Mapper (TM)

   * - 'l7'
     - Landsat 7 Enhanced Thematic Mapper Plus (ETM+) without panchromatic and thermal bands

   * - 'l7th'
     - Landsat 7 Enhanced Thematic Mapper Plus (ETM+) with thermal band

   * - 'l7mspan'
     - Landsat 7 Enhanced Thematic Mapper Plus (ETM+) with panchromatic band

   * - 'l7pan'
     - Landsat 7 panchromatic band

   * - 'l8'
     - Landsat 8 Operational Land Imager (OLI) and Thermal Infrared Sensor (TIRS) without panchromatic and thermal bands

   * - 'l8l7'
     - Landsat 8 Operational Land Imager (OLI) and Thermal Infrared Sensor (TIRS) with 6 Landsat 7-like bands

   * - 'l8l7mspan'
     - Landsat 8 Operational Land Imager (OLI) and panchromatic band with 6 Landsat 7-like bands

   * - 'l8th'
     - Landsat 8 Operational Land Imager (OLI) and Thermal Infrared Sensor (TIRS) with thermal band

   * - 'l8pan'
     - Landsat 8 panchromatic band

   * - 's2'
     - Sentinel 2 Multi-Spectral Instrument (MSI) without 3 60m bands (coastal, water vapor, cirrus)

   * - 's2f'
     - Sentinel 2 Multi-Spectral Instrument (MSI) with 3 60m bands (coastal, water vapor, cirrus)

   * - 's2l7'
     - Sentinel 2 Multi-Spectral Instrument (MSI) with 6 Landsat 7-like bands

   * - 's210'
     - Sentinel 2 Multi-Spectral Instrument (MSI) with 4 10m (visible + NIR) bands

   * - 's220'
     - Sentinel 2 Multi-Spectral Instrument (MSI) with 6 20m bands

   * - 's2cloudless'
     - Sentinel 2 Multi-Spectral Instrument (MSI) with 10 bands for s2cloudless

   * - 'ps'
     - PlanetScope with 4 (visible + NIR) bands

   * - 'qb'
     - Quickbird with 4 (visible + NIR) bands

   * - 'ik'
     - IKONOS with 4 (visible + NIR) bands

