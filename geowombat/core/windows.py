from .util import n_rows_cols

from rasterio.windows import from_bounds, Window


def get_window_offsets(n_rows, n_cols, row_chunks, col_chunks, return_as='list'):

    """
    Gets window offset indices from image dimensions and chunk sizes

    Args:
        n_rows (int): The number of rows to iterate over.
        n_cols (int): The number of columns to iterate over.
        row_chunks (int): The row chunk size.
        col_chunks (int): The column chunk size.
        return_as (Optional[str]): How to return the window information. Choices are ['dict', 'list'].

    Returns:
        Window information (list or dict)
    """

    if return_as == 'list':
        window_info = list()
    else:
        window_info = dict()

    i = 0

    for row_off in range(0, n_rows, row_chunks):

        height = n_rows_cols(row_off, row_chunks, n_rows)

        j = 0

        for col_off in range(0, n_cols, col_chunks):

            width = n_rows_cols(col_off, col_chunks, n_cols)

            if return_as == 'list':

                window_info.append(Window(col_off=col_off,
                                          row_off=row_off,
                                          width=width,
                                          height=height))

            else:

                window_info['{:d}{:d}'.format(i, j)] = Window(col_off=col_off,
                                                              row_off=row_off,
                                                              width=width,
                                                              height=height)

            j += 1

        i += 1

    return window_info
