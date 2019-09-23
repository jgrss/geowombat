import os
from collections import namedtuple

from rasterio.windows import Window
import xarray as xr


def get_file_extension(filename):

    FileNames = namedtuple('FileNames', 'd_name f_name f_base f_ext')

    d_name, f_name = os.path.splitext(filename)
    f_base, f_ext = os.path.split(f_name)

    return FileNames(d_name=d_name, f_name=f_name, f_base=f_base, f_ext=f_ext)


def n_rows_cols(pixel_index, block_size, rows_cols):

    """
    Adjusts block size for the end of image rows and columns.

    Args:
        pixel_index (int): The current pixel row or column index.
        block_size (int): The image block size.
        rows_cols (int): The total number of rows or columns in the image.

    Returns:
        Adjusted block size as int.
    """

    return block_size if (pixel_index + block_size) < rows_cols else rows_cols - pixel_index


def setup_windows(n_rows, n_cols, row_chunks, col_chunks):

    window_list = list()

    for i in range(0, n_rows, row_chunks):

        height = n_rows_cols(i, row_chunks, n_rows)

        for j in range(0, n_cols, col_chunks):

            width = n_rows_cols(j, col_chunks, n_cols)

            window_list.append(Window(col_off=j, row_off=i, width=width, height=height))

    return window_list


def xarray_to_xdataset(data_array, band_names, dates, ycoords=None, xcoords=None, attrs=None):

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

    if dates:

        return xr.Dataset({'bands': (['date', 'band', 'y', 'x'], data_array)},
                             coords={'date': dates,
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
