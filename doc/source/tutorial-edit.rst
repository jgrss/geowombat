.. _tutorial-edit:

Editing rasters
===============

Setting 'no data' values
------------------------

By default, ``geowombat`` (using ``rasterio`` and ``xarray``) will load the 'no data' value from the
file metadata, if it is available. For example:

.. ipython:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518

    with gw.open(l8_224078_20200518) as src:
        print(src)

Note the ``DataArray`` attributes 'nodatavals' and '_FillValue'. The former, 'nodatavals', is
:func:`geowombat.backends.xarray_rasterio_.open_rasterio` (originally from :func:`xarray.open_rasterio`)
convention. This attribute is a ``tuple`` of length ``DataArray.gw.nbands``, describing the 'no data' value
for each band. Typically, satellite imagery will have the same 'no data' value across all bands. The other
'no data' attribute, '_FillValue', is an attribute used by :func:`xarray.open_dataset` to flag 'no data' values.
This attribute is an ``int`` or ``float``. We store both attributes when opening data.

We can see in the opened image that the 'no data' value is ``nan`` (i.e., 'nodatavals' = (``nan``, ``nan``, ``nan``)
and '_FillValue' = ``nan``).

.. ipython:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518

    with gw.open(l8_224078_20200518) as src:
        print('dtype =', src.dtype)
        print(src.squeeze().values[0])

However, ``nan`` being set as the 'no data' is actually an error because this particular raster file does not
contain information about 'no data' values. If there is no existing 'no data' information ``rasterio`` will
set 'no data' as ``nan``. In this image, ``nans`` do not exist, and we can see that because the 'dtype' is
'uint16', whereas ``nans`` require data as floating point numbers.

Let's save a temporary file below and specify the 'no data' value as 0. Then, when we open the temporary
file the 'no data' attributes should be set as 0.

.. ipython:: python

    import tempfile
    from pathlib import Path
    import geowombat as gw
    from geowombat.data import l8_224078_20200518

    with gw.open(l8_224078_20200518) as src:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmp:
            # Set the temporary file name
            tmp_file = Path(tmp) / 'tmp_raster.tif'
            # Save the data to file, setting 'no data' as 0
            src.gw.save(tmp_file, nodata=0)
            # Open the temporary file and view the data
            with gw.open(tmp_file) as src_nodata:
                print(src_nodata)
                print(src_nodata.squeeze().values[0])

.. note::

    We are not modifying any data -- we are only updating the ``DataArray`` metadata. Thus, the printout of
    the data above reflect changes in the ``DataArray`` 'no data' attributes but the printed array values
    remained unchanged.

But what if we want to modify the 'no data' value when opening the file (instead of re-saving)? We can
pass ``nodata`` to the opener as shown below.

.. ipython:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518

    with gw.open(l8_224078_20200518, nodata=0) as src:
        print(src)
        print(src.squeeze().values[0])

We can also set 'no data' using the configuration manager like:

.. ipython:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518

    with gw.config.update(nodata=0):
        with gw.open(l8_224078_20200518) as src:
            print(src)
            print(src.squeeze().values[0])

Masking 'no data' values
------------------------

As mentioned above, the array data are not automatically modified by the 'no data' value. If we want to
mask our 'no data' values (i.e., exclude them from any calculations), we simply need to convert the
array values to ``nans``. GeoWombat provides a method called :func:`xarray.DataArray.gw.mask_nodata` to do this
that uses the metadata.

.. ipython:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518

    with gw.open(l8_224078_20200518, nodata=0) as src:
        # Print the first band with no masking
        print('No masking:')
        print(src.sel(band=1).values)
        # Print the first band with masked 'no data'
        print("\n'No data' values masked:")
        print(src.gw.mask_nodata().sel(band=1).values)

The :func:`xarray.DataArray.gw.mask_nodata` function uses :func:`xarray.DataArray.where` logic, as
demonstrated by the example below.

.. code:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518

    # Zeros are replaced with nans
    with gw.open(l8_224078_20200518) as src:
        data = src.where(src != 0)

Setting 'no data' values with scaling
-------------------------------------

In GeoWombat, we use :func:`xarray.DataArray.where` along with optional
scaling in the :func:`xarray.DataArray.gw.set_nodata` function. In this example, we set zeros as
``nan`` and scale all other values from a [0,10000] range to [0,1] (i.e., x 1e-4).

.. ipython:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518
    import numpy as np

    # Set the 'no data' value and scale all other values
    with gw.open(l8_224078_20200518, dtype='float64') as src:
        print(src.sel(band=1).values)
        data = src.gw.set_nodata(
            src_nodata=0, dst_nodata=np.nan, dtype='float64', scale_factor=1e-4
        )
        print(data.sel(band=1).values)

Replace values
--------------

The :func:`xarray.DataArray.gw.replace` function mimics :func:`pandas.DataFrame.replace`.

.. code:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518

    # Replace 1 with 10
    with gw.open(l8_224078_20200518) as src:
        data = src.gw.replace({1: 10})

.. note::

    The :func:`xarray.DataArray.gw.replace` function is typically used with thematic data.
