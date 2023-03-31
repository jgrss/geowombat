import typing as T

from .util import n_rows_cols

from rasterio.windows import Window


def get_window_offsets(
    n_rows: int,
    n_cols: int,
    row_chunks: int,
    col_chunks: int,
    return_as: T.Optional[str] = 'list',
    padding: T.Optional[T.Sequence[float]] = None,
) -> T.Union[
    T.List[Window], T.List[T.Tuple[Window, Window]], T.Dict[str, Window]
]:
    """Gets window offset indices from image dimensions and chunk sizes.

    Args:
        n_rows (int): The number of rows to iterate over.
        n_cols (int): The number of columns to iterate over.
        row_chunks (int): The row chunk size.
        col_chunks (int): The column chunk size.
        return_as (Optional[str]): How to return the window information. Choices are ['dict', 'list'].
        padding (Optional[tuple]): Padding for each window. ``padding`` should be given as a tuple
            of (left pad, bottom pad, right pad, top pad). If ``padding`` is given, the returned list will contain
            a tuple of ``rasterio.windows.Window`` objects as (w1, w2), where w1 contains the normal window offsets
            and w2 contains the padded window offsets.

    Returns:
        Window information as ``list`` or ``dict``.
    """

    if return_as == 'list':
        window_info = []
    else:
        window_info = {}

    i = 0
    for row_off in range(0, n_rows, row_chunks):
        height = n_rows_cols(row_off, row_chunks, n_rows)

        j = 0

        for col_off in range(0, n_cols, col_chunks):
            width = n_rows_cols(col_off, col_chunks, n_cols)

            if (return_as == 'list') and not padding:
                window_info.append(
                    Window(
                        col_off=col_off,
                        row_off=row_off,
                        width=width,
                        height=height,
                    )
                )

            elif (return_as == 'list') and padding:

                lpad, bpad, rpad, tpad = padding

                padded_row_off = row_off - tpad if row_off - tpad >= 0 else 0
                padded_col_off = col_off - lpad if col_off - lpad >= 0 else 0

                padded_height = n_rows_cols(
                    padded_row_off,
                    abs(row_off - padded_row_off) + row_chunks + bpad,
                    n_rows,
                )
                padded_width = n_rows_cols(
                    padded_col_off,
                    abs(col_off - padded_col_off) + col_chunks + rpad,
                    n_cols,
                )

                window_info.append(
                    (
                        Window(
                            col_off=col_off,
                            row_off=row_off,
                            width=width,
                            height=height,
                        ),
                        Window(
                            col_off=padded_col_off,
                            row_off=padded_row_off,
                            width=padded_width,
                            height=padded_height,
                        ),
                    )
                )

            else:

                window_info['{:d}{:d}'.format(i, j)] = Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=width,
                    height=height,
                )

            j += 1
        i += 1

    return window_info
