.. _web:

Web
===

Download data from Google Cloud Platform
----------------------------------------

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

    file_info = gdl.download_gcp('l7',
                                 downloads='LE07/01/225/083/LE07_L1TP_225083_20190208_20190306_01_T1',
                                 search_wildcards=search_wildcards,
                                 verbose=1)

Download and cube data
----------------------

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
    gdl.download_cube(sensors,
                      date_range,
                      bounds,
                      bands,
                      crs=crs)

In the example above, the bounds can also be taken directly from a file, as shown below.

.. code:: python

    import geopandas as gpd

    bounds = gpd.read_file('file.gpkg')

    # The CRS should be WGS84 lat/long
    bounds = bounds.to_crs('epsg:4326')

Read from virtual Cloud Optimized GeoTiffs
------------------------------------------

Using `rasterio` as a backend, we can read supported files directly from their respective cloud servers. In the example below,
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

.. ipython:: python

    from geowombat.util import GeoDownloads
    gdl = GeoDownloads()

    # Select a scene id from the query
    scene_id = 'LC08_L1TP_042034_20180110_20180119_01_T1'

    # Set a list of bands to read
    bands = ['blue', 'green', 'red', 'nir']

    # Get the GCP URLs
    urls = gdl.get_landsat_urls(scene_id, bands=bands)

    print(urls)

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

.. ipython:: python

    from geowombat.util import GeoDownloads
    gdl = GeoDownloads()

    safe_id = 'S2B_MSIL1C_20180124T135109_N0206_R024_T21HUD_20180124T153339.SAFE/GRANULE/L1C_T21HUD_A004626_20180124T135105'

    # We will read the blue, green, red, and NIR 10m bands
    bands = ['blue', 'green', 'red', 'nir']

    urls = gdl.get_sentinel2_urls(safe_id, bands=bands)

    print(urls)

Use the URLs to read the Sentinel 2 bands

.. code:: python

    # Open the images
    with gw.config.update(sensor='s2b10'):
        with gw.open(urls) as src:
            print(src)
