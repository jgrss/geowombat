.. _extraction:

Data extraction
===============

.. code:: python

    import geowombat as gw
    from geowombat.data import rgbn

Subsetting rasters
------------------

Either a `rasterio.window.Window <https://rasterio.readthedocs.io/en/stable/api/rasterio.windows.html#rasterio.windows.Window>`_
object or tuple can be used with :func:`geowombat.open`.

Slice a subset using a `rasterio.window.Window <https://rasterio.readthedocs.io/en/stable/api/rasterio.windows.html#rasterio.windows.Window>`_.

.. code:: python

    from rasterio.windows import Window
    w = Window(row_off=0, col_off=0, height=100, width=100)

    bounds = (793475.76, 2049033.03, 794222.03, 2049527.24)

    with gw.open(
        rgbn,
        band_names=['blue', 'green', 'red'],
        num_workers=8,
        indexes=[1, 2, 3],
        window=w,
        out_dtype='float32'
    ) as src:
        print(src)

Slice a subset using a tuple of bounded coordinates.

.. code:: python

    with gw.open(
        rgbn,
        band_names=['green', 'red', 'nir'],
        num_workers=8,
        indexes=[2, 3, 4],
        bounds=bounds,
        out_dtype='float32'
    ) as src:
        print(src)

The configuration manager provides an alternative method to subset rasters. See :ref:`tutorial-config` for more details.

.. code:: python

    with gw.config.update(ref_bounds=bounds):

        with gw.open(rgbn) as src:
            print(src)

By default, the subset will be returned by the upper left coordinates of the bounds, potentially shifting cell alignment
with the reference raster. To subset a raster and align it to the same grid, use the **ref_tar** keyword.

.. code:: python

    with gw.config.update(ref_bounds=bounds, ref_tar=rgbn):

        with gw.open(rgbn) as src:
            print(src)

Clipping to bounds
------------------

GeoWombat's :func:`geowombat.clip_by_polygon` is an alternative method to :class:`geowombat.config.update`. The
:func:`geowombat.clip_by_polygon` method limits the bounds of the image to match a polygon, where the polygon
can be a `geopandas.GeoDataFrame <https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.html#geopandas.GeoDataFrame>`_,
or a path to a file readable with `geopandas.read_file <https://geopandas.org/en/stable/docs/reference/api/geopandas.read_file.html#geopandas.read_file>`_.
You can augment the clip by using the argument ``query`` on the polygon attributes, and if multiple polygons
are present you can use ``mask_data`` to fill ``nans`` where polygons are not present, or expand the clip
array bounds by setting ``expand_by=<n pixels>`` on each side.

.. code:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518, l8_224078_20200518_polygons
    import geopandas as gpd

    polys = gpd.read_file(l8_224078_20200518_polygons)

    with gw.open(l8_224078_20200518) as src:
        print(src)
        clipped = src.gw.clip_by_polygon(
            df,
            query="name == water",
            mask_data=True,
            expand_by=1
        )
        print(clipped)

Extracting data with coordinates
--------------------------------

To extract values at a coordinate pair, translate the coordinates into array indices.

.. ipython:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518

    # Coordinates in map projection units
    y, x = -2823031.15, 761592.60

    with gw.open(l8_224078_20200518) as src:
        # Transform the map coordinates to data indices
        j, i = gw.coords_to_indices(x, y, src)
        data = src[:, i, j].data.compute()

    print(data.flatten())

A latitude/longitude pair can be extracted after converting to the map projection.

.. ipython:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518

    # Coordinates in latitude/longitude
    lat, lon = -25.50142964, -54.39756038

    with gw.open(l8_224078_20200518) as src:
        # Transform the coordinates to map units
        x, y = gw.lonlat_to_xy(lon, lat, src)
        # Transform the map coordinates to data indices
        j, i = gw.coords_to_indices(x, y, src)
        data = src[:, i, j].data.compute()

    print(data.flatten())

Extracting data with point geometry
-----------------------------------

In the example below, ``l8_224078_20200518_points`` is a `GeoPackage <https://www.geopackage.org/>`_ of point
locations, and the output ``df`` is a `geopandas.GeoDataFrame <https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.html#geopandas.GeoDataFrame>`_.
To extract the raster values at the point locations, use :func:`geowombat.extract`.

.. ipython:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518, l8_224078_20200518_points

    with gw.open(l8_224078_20200518) as src:
        df = src.gw.extract(l8_224078_20200518_points)

    print(df)

.. note::

    The line ``df = src.gw.extract(l8_224078_20200518_points)`` could also have been written as
    ``df = gw.extract(src, l8_224078_20200518_points)``.

In the previous example, the point vector had a CRS that matched the raster (i.e., EPSG=32621, or UTM zone 21N).
If the CRS had not matched, the :func:`geowombat.extract` function would have transformed the CRS on-the-fly.

.. ipython:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518, l8_224078_20200518_points
    import geopandas as gpd

    point_df = gpd.read_file(l8_224078_20200518_points)
    print(point_df.crs)

    # Transform the CRS to WGS84 lat/lon
    point_df = point_df.to_crs('epsg:4326')
    print(point_df.crs)

    with gw.open(l8_224078_20200518) as src:
        df = src.gw.extract(point_df)

    print(df)

Set the data band names.

.. ipython:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518, l8_224078_20200518_points

    with gw.config.update(sensor='bgr'):
        with gw.open(l8_224078_20200518) as src:
            df = src.gw.extract(
                l8_224078_20200518_points,
                band_names=src.band.values.tolist()
            )

    print(df)

Extracting data with polygon geometry
-------------------------------------

To extract values within polygons, use the same :func:`geowombat.extract` function.

.. ipython:: python

    from geowombat.data import l8_224078_20200518, l8_224078_20200518_polygons

    with gw.config.update(sensor='bgr'):
        with gw.open(l8_224078_20200518) as src:
            df = src.gw.extract(
                l8_224078_20200518_polygons,
                band_names=src.band.values.tolist()
            )

    print(df)
