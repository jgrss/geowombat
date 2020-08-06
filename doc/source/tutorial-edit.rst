.. _tutorial-edit:

Editing rasters
===============

Setting 'no data' values
------------------------

The :func:`xarray.DataArray.where` function masks data by setting nans, as demonstrated by the example below.

.. code:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518

    # Zeros are replaced with nans
    with gw.open(l8_224078_20200518) as src:
        data = src.where(src != 0)

Setting 'no data' values with scaling
-------------------------------------

In GeoWombat, we use :func:`xarray.where` and :func:`xarray.DataArray.where` along with optional scaling in the :func:`set_nodata` function. In this example, we set zeros as 65535 and scale all other values from a [0,10000] range to [0,1].

.. code:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518

    # Set the 'no data' value and scale all other values
    with gw.open(l8_224078_20200518) as src:
        data = src.gw.set_nodata(0, 65535, (0, 1), 'float64', scale_factor=0.0001)

Replace values
--------------

The GeoWombat :func:`replace` function mimics :func:`pandas.DataFrame.replace`.

.. code:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518

    # Replace 1 with 10
    with gw.open(l8_224078_20200518) as src:
        data = src.gw.replace({1: 10})

.. note::

    The :func:`replace` function is typically used with thematic data.
