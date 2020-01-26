.. _crs:

Coordinate Reference Systems
============================

Image projections can be transformed in `geowombat` using the configuration manager (see :ref:`config`). With the configuration manager, the CRS is transformed using `rasterio` `CRS <https://rasterio.readthedocs.io/en/latest/api/rasterio.crs.html>`_ and `virtual warping <https://rasterio.readthedocs.io/en/latest/topics/virtual-warping.html>`_.

The transform can be controlled from the `xarray.DataArray <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html>`_ attributes.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. ipython:: python

    import geowombat as gw
    from geowombat.data import rgbn

    with gw.open(rgbn) as src:
        print(src.transform)
        print(src.crs)
        print(src.resampling)
        print(src.res)

To transform the CRS, use the context manager. In this example, an EPSG code is used.

.. ipython:: python

    with gw.config.update(ref_crs=102008):
        with gw.open(rgbn) as src:
            print(src.transform)
            print(src.crs)
            print(src.resampling)
            print(src.res)

Other formats supported by `rasterio` can be used.

.. ipython:: python

    proj4 = "+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"

    with gw.config.update(ref_crs=proj4):
        with gw.open(rgbn) as src:
            print(src.transform)
            print(src.crs)
            print(src.resampling)
            print(src.res)

The resampling algorithm can be specified in the :func:`geowombat.open` function.

.. ipython:: python

    with gw.config.update(ref_crs=102008):
        with gw.open(rgbn, resampling='cubic') as src:
            print(src.transform)
            print(src.crs)
            print(src.resampling)
            print(src.res)

The transformed cell resolution can be added in the context manager.

.. ipython:: python

    with gw.config.update(ref_crs=102008, ref_res=(10, 10)):
        with gw.open(rgbn, resampling='cubic') as src:
            print(src.transform)
            print(src.crs)
            print(src.resampling)
            print(src.res)
