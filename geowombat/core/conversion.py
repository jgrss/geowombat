import xarray as xr


def dask_to_xarray(xarray_data, dask_data, band_name):

    """
    Converts a Dask array to an Xarray DataArray

    Args:
        xarray_data (DataArray): The DataArray with attribute information.
        dask_data (Dask Array): The Dask array to convert.
        band_name (str): The band name.

    Returns:
        DataArray
    """

    return xr.DataArray(dask_data.reshape(1, dask_data.shape[0], dask_data.shape[1]),
                        dims=['band', 'y', 'x'],
                        coords={'band': [band_name],
                                'y': xarray_data.y,
                                'x': xarray_data.x},
                        attrs=xarray_data.attrs)


def xarray_to_xdataset(data_array, band_names, time_names, ycoords=None, xcoords=None, attrs=None):

    """
    Converts an Xarray DataArray to a Xarray Dataset

    Args:
        data_array (DataArray)
        band_names (list)
        time_names (list)
        ycoords (1d array-like)
        xcoords (1d array-like)
        attrs (dict)

    Returns:
        Dataset
    """

    if len(data_array.shape) == 2:
        data_array = data_array.expand_dims('band')

    if len(data_array.shape) == 4:
        n_bands = data_array.shape[1]
    else:
        n_bands = data_array.shape[0]

    if not band_names:

        if n_bands == 1:
            band_names = ['1']
        else:
            band_names = list(map(str, range(1, n_bands+1)))

    if time_names:

        return xr.Dataset({'bands': (['date', 'band', 'y', 'x'], data_array)},
                             coords={'date': time_names,
                                     'band': band_names,
                                     'y': ('y', ycoords),
                                     'x': ('x', xcoords)},
                             attrs=attrs)

    else:

        return xr.Dataset({'bands': (['band', 'y', 'x'], data_array.data)},
                          coords={'band': band_names,
                                  'y': ('y', data_array.y),
                                  'x': ('x', data_array.x)},
                          attrs=data_array.attrs)
