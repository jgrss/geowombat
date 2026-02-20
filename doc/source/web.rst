.. _web:

Streaming data from cloud sources
=================================

GeoWombat integrates easy access to Spatial Temporal Asset Catalog (`STAC <https://stacspec.org/en>`_) APIs.
STAC is a standardized way to expose collections of spatial temporal data. Instead of downloading
large files, STAC lets you search for exactly the imagery you need (by location, date, and cloud cover)
and stream it directly into your analysis. For a full list of public STAC APIs, refer to the
`STAC datasets page <https://stacspec.org/en/about/datasets/>`_.

Installation
------------

To install ``geowombat`` with STAC functionality::

    pip install "geowombat[stac]"

This installs the required dependencies: ``pystac``, ``pystac_client``, ``stackstac``,
and ``planetary_computer``.

Supported catalogs and collections
----------------------------------

:func:`geowombat.core.stac.open_stac` supports the following STAC catalogs:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Catalog
     - Collections
   * - ``element84_v0``
     - ``sentinel_s2_l2a_cogs``
   * - ``element84_v1``
     - ``cop_dem_glo_30``, ``landsat_c2_l2``, ``sentinel_s2_l2a``, ``sentinel_s2_l1c``, ``sentinel_s1_l1c``, ``naip``
   * - ``microsoft_v1``
     - ``cop_dem_glo_30``, ``landsat_c2_l1``, ``landsat_c2_l2``, ``landsat_l8_c2_l2``, ``sentinel_s2_l2a``, ``sentinel_s1_l1c``, ``sentinel_3_lst``, ``io_lulc``, ``usda_cdl``, ``hls``, ``esa_worldcover``

STAC examples
-------------

Stream Sentinel-2 data from Element 84
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example streams Sentinel-2 Level 2A (surface reflectance) bands for
the Washington, D.C. area using `Element 84's <https://www.element84.com/>`_ STAC catalog.

.. code:: python

    from geowombat.core.stac import open_stac

    data, df = open_stac(
        stac_catalog="element84_v1",
        bounds=(-77.1, 38.85, -76.95, 38.95),  # DC area (left, bottom, right, top)
        epsg=32618,  # UTM Zone 18N
        collection="sentinel_s2_l2a",  # Sentinel-2 Level 2A (surface reflectance)
        bands=["blue", "green", "red", "nir"],
        cloud_cover_perc=20,
        start_date="2023-06-01",
        end_date="2023-07-31",
        resolution=10.0,
        chunksize=512,
    )

    # data is a lazy dask-backed xarray DataArray with dims (time, band, y, x)
    print(data)

Plot the results:

.. code:: python

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(dpi=200, figsize=(3, 3))
    data.sel(time=data.time[0], band=["red", "green", "blue"]).plot.imshow(
        robust=True, ax=ax
    )
    ax.set_title("Sentinel-2 RGB - Washington, D.C.")
    plt.tight_layout(pad=1)

.. image:: _static/stac_sentinel2_rgb.png
   :width: 400
   :alt: Sentinel-2 RGB composite of Washington, D.C.

Stream Landsat data from Microsoft Planetary Computer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from geowombat.core.stac import open_stac

    data_l, df_l = open_stac(
        stac_catalog='microsoft_v1',
        collection='landsat_c2_l2',
        bounds=(-77.1, 38.85, -76.95, 38.95),
        epsg=32618,
        bands=['red', 'green', 'blue', 'qa_pixel'],
        mask_data=True,
        start_date='2023-06-01',
        end_date='2023-07-31',
        resolution=30.0,
        chunksize=512,
    )

    print(data_l)

.. code:: python

    fig, ax = plt.subplots(dpi=200, figsize=(3, 3))
    data_l.sel(time=data_l.time[0], band=["red", "green", "blue"]).plot.imshow(
        robust=True, ax=ax
    )
    ax.set_title("Landsat RGB - Washington, D.C.")
    plt.tight_layout(pad=1)

.. note::

    When using ``mask_data=True`` with a ``qa_pixel`` band, cloud and shadow pixels
    are automatically masked. The ``qa_pixel`` band is removed from the output after masking.
    Add ``max_items=10`` to cap the number of scenes returned.


Merge multiple collections
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :func:`geowombat.core.stac.merge_stac` to combine data from different sensors
into a single time series:

.. code:: python

    from geowombat.core.stac import open_stac, merge_stac
    from rasterio.enums import Resampling

    # Load Landsat
    data_l, df_l = open_stac(
        stac_catalog="microsoft_v1",
        collection="landsat_c2_l2",
        bounds=(-77.1, 38.85, -76.95, 38.95),
        bands=["red", "green", "blue", "qa_pixel"],
        mask_data=True,
        start_date="2023-01-01",
        end_date="2023-12-31",
        epsg=32618,
        resolution=30.0,
    )

    # Load Sentinel-2, reprojected to match Landsat
    data_s2, df_s2 = open_stac(
        stac_catalog="element84_v1",
        collection="sentinel_s2_l2a",
        bounds=(-77.1, 38.85, -76.95, 38.95),
        bands=["blue", "green", "red"],
        resampling=Resampling.cubic,
        epsg=32618,
        start_date="2023-01-01",
        end_date="2023-12-31",
        resolution=30.0,
    )

    # Merge into a single time series
    stack = merge_stac(data_l, data_s2)
    print(stack)

Other examples
--------------

Rasterio makes it easy to read URLs from cloud sources. The examples below show other approaches to
reading imagery from sources such as AWS or Google Cloud Platform buckets.

Download data from Google Cloud Platform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here, a Landsat 7 panchromatic image is downloaded.

.. code:: python

    from geowombat.util.web import GeoDownloads

    gdl = GeoDownloads()

    gdl.list_gcp('l7', '225/083/*225083_201901*_T*')

    del_keys = [k for k, v in gdl.search_dict.items() if 'gap_mask' in k]

    for dk in del_keys:
        del gdl.search_dict[dk]

    # Results are saved as a dictionary
    print(list(gdl.search_dict.keys()))

    search_wildcards = ['ANG.txt', 'MTL.txt', 'B8.TIF']

    file_info = gdl.download_gcp(
        'l7',
        downloads='LE07/01/225/083/LE07_L1TP_225083_20190208_20190306_01_T1',
        search_wildcards=search_wildcards,
        verbose=1
    )

Download and cube data
~~~~~~~~~~~~~~~~~~~~~~

In this example, data are downloaded and processed for a given time range and geographic extent.

.. code:: python

    # Download Landsat 7 data
    sensors = ['l7']

    # Specify the date range
    date_range = ['2010-01-01', '2010-02-01']

    # Specify the geographic extent
    # left, bottom, right, top (in WGS84 lat/lon)
    bounds = (-91.57, 40.37, -91.46, 40.42)

    # Download the panchromatic band
    bands = ['pan']

    # Cube into an Albers Equal Area projection
    crs = "+proj=aea +lat_1=-5 +lat_2=-42 +lat_0=-32 +lon_0=-60 +x_0=0 +y_0=0 +ellps=aust_SA +units=m +no_defs"

    # Download a Landsat 7 panchromatic, BRDF-adjusted cube
    gdl.download_cube(
        sensors,
        date_range,
        bounds,
        bands,
        crs=crs
    )

In the example above, the bounds can also be taken directly from a file, as shown below.

.. code:: python

    import geopandas as gpd

    bounds = gpd.read_file('file.gpkg')

    # The CRS should be WGS84 lat/long
    bounds = bounds.to_crs('epsg:4326')

Read from virtual Cloud Optimized GeoTiffs
------------------------------------------

Using ``rasterio`` as a backend, we can read supported files directly from their respective cloud servers. In the example below,
we query a Landsat scene and open the blue, green, red, and NIR band metadata.

.. code:: python

    import os
    import geowombat as gw
    from geowombat.util import GeoDownloads

    os.environ['CURL_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'

    gdl = GeoDownloads()

    # This part is not necessary if you already know the scene id
    path = 42
    row = 34
    year = 2018
    month = 1

    # Query GCP
    gdl.list_gcp('l8', f'{path:03d}/{row:03d}/*{path:03d}{row:03d}_{year:04d}{month:02d}*_T1*')

    # Get the results
    print(gdl.get_gcp_results)

.. code:: python

    from geowombat.util import GeoDownloads
    gdl = GeoDownloads()

    # Select a scene id from the query
    scene_id = 'LC08_L1TP_042034_20180110_20180119_01_T1'

    # Set a list of bands to read
    bands = ['blue', 'green', 'red', 'nir']

    # Get the GCP URLs
    urls, meta_url = gdl.get_landsat_urls(scene_id, bands=bands)

    for url in urls:
        print(url)

Use the URLs to read the Landsat bands

.. code:: python

    # Open the images
    with gw.config.update(sensor='l8bgrn'):
        with gw.open(urls) as src:
            print(src)

The setup for Sentinel 2 is slightly different because of the SAFE directory storage format. Instead of a scene id, we need
a longer SAFE id.

.. note::

    Note that the Sentinel 2 data are not cloud optimized because they are stored in the .jp2 format. Therefore, the read performance
    could be much slower compared to the Landsat GeoTiffs.

.. code:: python

    gdl.list_gcp('s2b', '21/H/UD/*201801*.SAFE/GRANULE/*')

.. code:: python

    from geowombat.util import GeoDownloads
    gdl = GeoDownloads()

    safe_id = 'S2B_MSIL1C_20180124T135109_N0206_R024_T21HUD_20180124T153339.SAFE/GRANULE/L1C_T21HUD_A004626_20180124T135105'

    # We will read the blue, green, red, and NIR 10m bands
    bands = ['blue', 'green', 'red', 'nir']

    urls, meta_url = gdl.get_sentinel2_urls(safe_id, bands=bands)

    for url in urls:
        print(url)

Use the URLs to read the Sentinel 2 bands

.. code:: python

    # Open the images
    with gw.config.update(sensor='s2b10'):
        with gw.open(urls) as src:
            print(src)
