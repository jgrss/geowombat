.. _extraction:

Data extraction
===============

.. code:: python

    import geowombat as gw
    from geowombat.data import rgbn

Create a Rasterio window

.. code:: python

    from rasterio.windows import Window
    w = Window(row_off=0, col_off=0, height=100, width=100)

Slice a subset using a Rasterio window

.. code:: python

    with gw.open(rgbn,
                 band_names=['blue', 'green', 'red'],
                 num_workers=8,
                 indexes=[1, 2, 3],
                 window=w,
                 out_dtype='float32') as ds:
        print(ds)

Slice a subset using a tuple of bounded coordinates

.. code:: python

    bounds = (793000.0, 2049000.0, 794000.0, 2050000.0)

    with gw.open(rgbn,
                 band_names=['green', 'red', 'nir'],
                 num_workers=8,
                 indexes=[2, 3, 4],
                 bounds=bounds,
                 out_dtype='float32') as ds:
        print(ds)

Extract values at point locations

.. code:: python

    with gw.open(rgbn)
        df = ds.gw.extract('point.shp')

Extract values within polygons

.. code:: python

    import geopandas as gpd

    df = gpd.read_file('poly.gpkg')

    with gw.open(rgbn)

        df = ds.gw.extract(df,
                           bands=3,
                           band_names=['red'],
                           frac=0.1,
                           n_jobs=8,
                           num_workers=8,
                           verbose=1)
