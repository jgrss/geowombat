.. _tutorial-crs:

Coordinate Reference Systems
============================

Image projections can be transformed in GeoWombat using the configuration manager (see :ref:`tutorial-config`).
With the configuration manager, the CRS is transformed using
`pyproj.crs.CRS <https://pyproj4.github.io/pyproj/dev/api/crs/crs.html>`_
and `virtual warping <https://rasterio.readthedocs.io/en/latest/topics/virtual-warping.html>`_. For references,
see `Spatial Reference <https://spatialreference.org/>`_ and `epsg.io <http://epsg.io/>`_.

The CRS can be accessed from the `xarray.DataArray <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html>`_ attributes.

.. ipython:: python

    import geowombat as gw
    from geowombat.data import rgbn

    with gw.open(rgbn) as src:
        print(src.transform)
        print(src.gw.transform)
        print(src.crs)
        print(src.resampling)
        print(src.res)
        print(src.gw.cellx, src.gw.celly)

Transforming a CRS on-the-fly
-----------------------------

To transform the CRS, use the context manager. In this example, a proj4 string is used.

.. ipython:: python

    proj4 = "+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"

    with gw.config.update(ref_crs=proj4):
        with gw.open(rgbn) as src:
            print(src.transform)
            print(src.crs)
            print(src.resampling)
            print(src.res)

Other formats supported by rasterio, (e.g., crs codes) can be used.

.. ipython:: python

    with gw.config.update(ref_crs='ESRI:102008'):
        with gw.open(rgbn) as src:
            print(src.transform)
            print(src.crs)
            print(src.resampling)
            print(src.res)

Resampling the cell size
------------------------

The resampling algorithm can be specified in the :func:`geowombat.open` function. Here, we use cubic convolution resampling
to warp the data to EPSG code 102008.

.. ipython:: python

    with gw.config.update(ref_crs='ESRI:102008'):
        with gw.open(rgbn, resampling='cubic') as src:
            print(src.transform)
            print(src.crs)
            print(src.resampling)
            print(src.res)

The transformed cell resolution can be added in the context manager. Here, we resample the data to 10m x 10m spatial resolution.

.. ipython:: python

    with gw.config.update(ref_crs=proj4, ref_res=(10, 10)):
        with gw.open(rgbn, resampling='cubic') as src:
            print(src.transform)
            print(src.crs)
            print(src.resampling)
            print(src.res)

To transform an :class:`xarray.DataArray` outside of a configuration context, use the :func:`geowombat.transform_crs` function.

.. ipython:: python

    with gw.open(rgbn, resampling='cubic') as src:
        print(help(src.gw.transform_crs))

.. ipython:: python

    with gw.open(rgbn) as src:
        print(src.transform)
        print(src.crs)
        print(src.resampling)
        print(src.res)
        print('')
        src_tr = src.gw.transform_crs(proj4, dst_res=(10, 10), resampling='bilinear')
        print(src_tr.transform)
        print(src_tr.crs)
        print(src_tr.resampling)
        print(src_tr.res)
