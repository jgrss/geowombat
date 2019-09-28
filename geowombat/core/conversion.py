import xarray as xr


def xarray_to_xdataset(data_array, band_names, time_names, ycoords=None, xcoords=None, attrs=None):

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
