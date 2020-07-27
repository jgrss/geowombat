.. _extraction:

Data extraction
===============

.. code:: python

    import geowombat as gw
    from geowombat.data import rgbn

Subsetting rasters
------------------

Either a `rasterio.window.Window` object or tuple can be used with :func:`geowombat.open`.

.. code:: python

    from rasterio.windows import Window
    w = Window(row_off=0, col_off=0, height=100, width=100)

    bounds = (793475.76, 2049033.03, 794222.03, 2049527.24)

Slice a subset using a `rasterio.window.Window`.

.. code:: python

    with gw.open(rgbn,
                 band_names=['blue', 'green', 'red'],
                 num_workers=8,
                 indexes=[1, 2, 3],
                 window=w,
                 out_dtype='float32') as src:
        print(src)

Slice a subset using a tuple of bounded coordinates.

.. code:: python

    with gw.open(rgbn,
                 band_names=['green', 'red', 'nir'],
                 num_workers=8,
                 indexes=[2, 3, 4],
                 bounds=bounds,
                 out_dtype='float32') as src:
        print(src)

The configuration manager provides an alternative method to subset rasters.

.. code:: python

    with gw.config.update(ref_bounds=bounds):

        with gw.open(rgbn) as src:
            print(src)

By default, the subset will be returned by the upper left coordinates of the bounds, potentially shifting cell alignment with the reference raster. To subset a raster and align it to the same grid, use the `ref_tar` keyword.

.. code:: python

    with gw.config.update(ref_bounds=bounds, ref_tar=rgbn):

        with gw.open(rgbn) as src:
            print(src)

Extracting data at coordinates
------------------------------

Extract values at point locations.

.. code:: python

    with gw.open(rgbn) as src:
        df = src.gw.extract('point.shp')

Extract values within polygons.

.. code:: python

    import geopandas as gpd

    df = gpd.read_file('poly.gpkg')

    with gw.open(rgbn) as src:

        df = src.gw.extract(df,
                            bands=3,
                            band_names=['red'],
                            frac=0.1,
                            n_jobs=8,
                            num_workers=8,
                            verbose=1)
