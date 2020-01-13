from pathlib import Path

import zarr
import numcodecs


if hasattr(numcodecs, 'blosc'):

    numcodecs.blosc.use_threads = False

    compressor = numcodecs.Blosc(cname='zstd',
                                 clevel=2,
                                 shuffle=numcodecs.Blosc.BITSHUFFLE)


def to_zarr(filename, data, window, chunks, root=None):

    """
    Writes data to a zarr file

    Args:
        filename (str): The output file name.
        data (ndarray): The data to write.
        window (namedtuple): A ``rasterio.window.Window`` object.
        chunks (int or tuple): The ``zarr`` chunks.
        root (Optional[object]): The ``zarr`` root.

    Returns:
        ``str``
    """

    p = Path(filename)

    f_base = p.name.split('.')[0]
    d_name = p.parent
    sub_dir = d_name.joinpath('sub_tmp_')
    zarr_file = sub_dir.joinpath('data.zarr').as_posix()

    # sub_dir.mkdir(parents=True, exist_ok=True)

    if not root:
        root = zarr.open(zarr_file, mode='r+')

    group_name = '{BASE}_y{Y:09d}_x{X:09d}_h{H:09d}_w{W:09d}'.format(BASE=f_base,
                                                                     Y=window.row_off,
                                                                     X=window.col_off,
                                                                     H=window.height,
                                                                     W=window.width)

    group = root.create_group(group_name)

    synchronizer = zarr.ProcessSynchronizer('data.sync')

    z = group.array('data',
                    data,
                    compressor=compressor,
                    dtype=data.dtype.name,
                    chunks=chunks,
                    synchronizer=synchronizer)

    group.attrs['row_off'] = window.row_off
    group.attrs['col_off'] = window.col_off
    group.attrs['height'] = window.height
    group.attrs['width'] = window.width

    return zarr_file
