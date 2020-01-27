.. _web:

Web
===

GeoWombat can be used to download directly from the Google Cloud Platform.
--------------------------------------------------------------------------

Here, a Landsat 7 panchromatic image is downloaded.

.. code:: python

    from geowombat.web import GeoDownloads

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

GeoWombat provides a convenience function to download, radiometrically adjust, and cube data.
---------------------------------------------------------------------------------------------

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
    bounds = bounds.to_crs({'init': 'epsg:4326'})
