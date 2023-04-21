.. _tutorial-config:

Configuration manager
=====================

What is a context manager?
--------------------------

In short, a context manager ensures proper file closing using `with statements <https://docs.python.org/2.5/whatsnew/pep-343.html>`_.

What is the purpose of GeoWombat's context manager?
---------------------------------------------------

The examples shown in :ref:`tutorial-open` opened the entire rasters as DataArrays as they were stored on file. The configuration manager allows easy control over opened raster dimensions, alignment, and transformations.

How do I use it?
----------------

To use GeoWombat's configuration manager, just call :class:`geowombat.config.update` before opening a file. For example,

.. code:: python

    import geowombat as gw

    with gw.config.update(<keywords>...):

        # Every file opened within the configuration block will use
        # configuration keywords
        with gw.open('image.tif') as src:
            # do something

:class:`geowombat.config.update` stores keywords in a dictionary. To see all GeoWombat configuration keywords, just
iterate over the dictionary.

.. ipython:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518

    # Using the manager without keywords will set defaults
    with gw.config.update():
        with gw.open(l8_224078_20200518) as src:
            for k, v in src.gw.config.items():
                print('Keyword:', k.ljust(15), 'Value:', v)

Reference settings: CRS
-----------------------

Configuration keywords beginning with **ref** are the most important commands when opening rasters. For example,
to transform the CRS of the data on-the-fly, use **ref_crs**. For more on Coordinate Reference Systems, see :ref:`here <tutorial-crs>`.

.. ipython:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518

    proj4 = "+proj=aea +lat_1=-5 +lat_2=-42 +lat_0=-32 +lon_0=-60 +x_0=0 +y_0=0 +ellps=aust_SA +units=m +no_defs "

    # Without the manager
    with gw.open(l8_224078_20200518) as src:
        print(src.crs)

    # With the manager
    with gw.config.update(ref_crs=proj4):
        with gw.open(l8_224078_20200518) as src:
            print(src.crs)

Reference settings: Cell size
-----------------------------

It is possible to combine multiple configuration keywords. In the example below, the raster CRS is transformed from
UTM to Albers Equal Area with a resampled cell size of 100m x 100m.

.. ipython:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518

    # Without the manager
    with gw.open(l8_224078_20200518) as src:
        print(src.gw.celly, src.gw.cellx)

    # With the manager
    with gw.config.update(ref_crs=proj4, ref_res=(100, 100)):
        with gw.open(l8_224078_20200518) as src:
            print(src.gw.celly, src.gw.cellx)

Reference settings: Bounds
--------------------------

To subset an image, specify bounds as a ``tuple`` of (left, bottom, right, top) or a
:class:`rasterio.coords.BoundingBox <https://rasterio.readthedocs.io/en/stable/api/rasterio.coords.html#rasterio.coords.BoundingBox>`_ object.

.. ipython:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518
    from rasterio.coords import BoundingBox

    bounds = BoundingBox(left=724634.17, bottom=-2806501.39, right=737655.48, top=-2796221.42)

    # or
    # bounds = (724634.17, -2806501.39, 737655.48, -2796221.42)

    # Without the manager
    with gw.open(l8_224078_20200518) as src:
        print(src.gw.bounds)

    # With the manager
    with gw.config.update(ref_bounds=bounds):
        with gw.open(l8_224078_20200518) as src:
            print(src.gw.bounds)

Reference settings: Image
-------------------------

To use another image as a reference, just set **ref_image**. Then, the opened file's bounds, CRS, and cell size
will be transformed to match those of the reference image.

.. ipython:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518, l8_224077_20200518_B2

    # Without the manager
    with gw.open(l8_224078_20200518) as src:
        print(src.gw.bounds)

    with gw.open(l8_224077_20200518_B2) as src:
        print(src.gw.bounds)

    # With the manager
    with gw.config.update(ref_image=l8_224077_20200518_B2):
        with gw.open(l8_224078_20200518) as src:
            print(src.gw.bounds)

Reference settings: Sensors
---------------------------

Because rasters are opened as DataArrays, the band coordinates will be named. By default, the bands
will be named by their index position (starting at 1). It might, however, be more intuitive to store
the band names as strings, where the names correspond to the sensor wavelengths. In GeoWombat, you
can set the band names explicitly upon opening a file by using the :func:`geowombat.open` **band_names**
keyword. Alternatively, if the sensor is known (and supported by GeoWombat), then you can set the band
names by specifying the sensor name in the configuration settings.

.. note::

    In the example below, the example raster comes from a Landsat image. However, only the visible
    (blue, green, and red) wavelengths are stored. Thus, we use 'rgb' as the sensor name. If we had
    a full 6-band Landsat 7 image, for example, we could use the 'l7' sensor flag.

.. ipython:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518

    # Without the manager
    with gw.open(l8_224078_20200518) as src:
        print(src.band)

    # With the manager
    with gw.config.update(sensor='bgr'):
        with gw.open(l8_224078_20200518) as src:
            print(src.band)

To see all available sensor names, use the **avail_sensors** property.

.. ipython:: python

    with gw.open(l8_224078_20200518) as src:
        for sensor_name in src.gw.avail_sensors:
            print(sensor_name)

For a short description of the sensor, use the **sensor_names** property.

.. ipython:: python

    with gw.open(l8_224078_20200518) as src:
        for sensor_name, description in src.gw.sensor_names.items():
            print('{}: {}'.format(sensor_name.ljust(15), description))

The following is a list of all available sensor names. This documentation may become out of date, if so please
refer to ``geowombat/core/properties.py`` for the full list.

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

