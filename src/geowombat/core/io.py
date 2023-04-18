import concurrent.futures
import itertools
import logging
import multiprocessing as multi
import os
import random
import shutil
import string
import threading
import typing as T
import warnings
from pathlib import Path

import dask
import numpy as np
import pyproj
import rasterio as rio
import xarray as xr
from affine import Affine
from dask import is_dask_collection
from dask.distributed import Client, progress
from osgeo import gdal
from rasterio import shutil as rio_shutil
from rasterio.drivers import driver_from_extension
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from threadpoolctl import threadpool_limits
from tqdm import tqdm
from tqdm.dask import TqdmCallback

try:
    import zarr

    from ..backends.zarr_ import to_zarr

    ZARR_INSTALLED = True
except ImportError:
    ZARR_INSTALLED = False

from ..backends.rasterio_ import RasterioStore, to_gtiff
from ..handler import add_handler
from .windows import get_window_offsets

logger = logging.getLogger(__name__)
logger = add_handler(logger)


def get_norm_indices(n_bands, window_slice, indexes_multi):
    # Prepend the band position index to the window slice
    if n_bands == 1:
        window_slice = tuple([slice(0, 1)] + list(window_slice))
        indexes = 1
    else:
        window_slice = tuple([slice(0, n_bands)] + list(window_slice))
        indexes = indexes_multi

    return window_slice, indexes


def _window_worker(w):
    """Helper to return window slice."""
    return slice(w.row_off, w.row_off + w.height), slice(
        w.col_off, w.col_off + w.width
    )


def _window_worker_time(w, n_bands, tidx, n_time):
    """Helper to return window slice."""
    window_slice = (
        slice(w.row_off, w.row_off + w.height),
        slice(w.col_off, w.col_off + w.width),
    )

    # Prepend the band position index to the window slice
    if n_bands == 1:
        window_slice = tuple(
            [slice(tidx, n_time)] + [slice(0, 1)] + list(window_slice)
        )
    else:
        window_slice = tuple(
            [slice(tidx, n_time)] + [slice(0, n_bands)] + list(window_slice)
        )

    return window_slice


def _block_read_func(fn_, g_, t_):
    """Function for block writing with ``concurrent.futures``"""
    if t_ == 'zarr':
        group_node = zarr.open(fn_, mode='r')[g_]
        w_ = Window(
            row_off=group_node.attrs['row_off'],
            col_off=group_node.attrs['col_off'],
            height=group_node.attrs['height'],
            width=group_node.attrs['width'],
        )
        out_data_ = np.squeeze(group_node['data'][:])
    else:
        w_ = Window(
            row_off=int(
                os.path.splitext(os.path.basename(fn_))[0].split('_')[-4][1:]
            ),
            col_off=int(
                os.path.splitext(os.path.basename(fn_))[0].split('_')[-3][1:]
            ),
            height=int(
                os.path.splitext(os.path.basename(fn_))[0].split('_')[-2][1:]
            ),
            width=int(
                os.path.splitext(os.path.basename(fn_))[0].split('_')[-1][1:]
            ),
        )
        out_data_ = np.squeeze(rio.open(fn_).read(window=w_))

    out_indexes_ = (
        1
        if len(out_data_.shape) == 2
        else list(range(1, out_data_.shape[0] + 1))
    )

    return w_, out_indexes_, out_data_


def _check_offsets(
    block, out_data_, window_, oleft, otop, ocols, orows, left_, top_
):
    # Check if the data were read at larger
    # extents than the write bounds.

    obottom = otop - (orows * abs(block.gw.celly))
    oright = oleft + (ocols * abs(block.gw.cellx))

    bottom_ = top_ - (window_.height * abs(block.gw.celly))
    right_ = left_ - (window_.width * abs(block.gw.cellx))

    left_diff = 0
    right_diff = 0
    top_diff = 0
    bottom_diff = 0

    if left_ < oleft:
        left_diff = int(abs(oleft - left_) / abs(block.gw.cellx))
        right_diff = out_data_.shape[-1]
    elif right_ > oright:
        left_diff = 0
        right_diff = int(abs(oright - right_) / abs(block.gw.cellx))

    if bottom_ < obottom:
        bottom_diff = int(abs(obottom - bottom_) / abs(block.gw.celly))
        top_diff = 0
    elif top_ > otop:
        bottom_diff = out_data_.shape[-2]
        top_diff = int(abs(otop - top_) / abs(block.gw.celly))

    if (
        (left_diff != 0)
        or (top_diff != 0)
        or (bottom_diff != 0)
        or (right_diff != 0)
    ):

        dshape = out_data_.shape

        if len(dshape) == 2:
            out_data_ = out_data_[top_diff:bottom_diff, left_diff:right_diff]
        elif len(dshape) == 3:
            out_data_ = out_data_[
                :, top_diff:bottom_diff, left_diff:right_diff
            ]
        elif len(dshape) == 4:
            out_data_ = out_data_[
                :, :, top_diff:bottom_diff, left_diff:right_diff
            ]

        window_ = Window(
            col_off=window_.col_off,
            row_off=window_.row_off,
            width=out_data_.shape[-1],
            height=out_data_.shape[-2],
        )

    return out_data_, window_


def _compute_block(
    block, wid, window_, padded_window_, n_workers, num_workers
):
    """Computes a DataArray window block of data.

    Args:
        block (DataArray): The ``xarray.DataArray`` to compute.
        wid (int): The window id.
        window_ (namedtuple): The window ``rasterio.windows.Window`` object.
        padded_window_ (namedtuple): A padded window ``rasterio.windows.Window`` object.
        n_workers (int): The number of parallel workers for chunks.
        num_workers (int): The number of parallel workers for ``dask.compute``.
        oleft (float): The output image left coordinate.
        otop (float): The output image top coordinate.
        ocols (int): The output image columns.
        orows (int): The output image rows.

    Returns:
        ``numpy.ndarray`` | ``rasterio.windows.Window`` | ``int`` | ``list``
    """
    out_data_ = None
    if 'apply' in block.attrs:
        attrs = block.attrs.copy()
        # Update the block transform
        attrs['transform'] = Affine(*block.gw.transform)
        attrs['window_id'] = wid

        block = block.assign_attrs(**attrs)

    if ('apply' in block.attrs) and hasattr(
        block.attrs['apply'], 'wombat_func_'
    ):
        if padded_window_:
            logger.warning('  Padding is not supported with lazy functions.')

        if block.attrs['apply'].wombat_func_:

            # Add the data to the keyword arguments
            block.attrs['apply_kwargs']['data'] = block

            out_data_ = block.attrs['apply'](**block.attrs['apply_kwargs'])

            if n_workers == 1:
                out_data_ = out_data_.data.compute(
                    scheduler='threads', num_workers=num_workers
                )
            else:

                with threading.Lock():
                    out_data_ = out_data_.data.compute(
                        scheduler='threads', num_workers=num_workers
                    )

        else:
            logger.exception('  The lazy wombat function is turned off.')

    else:
        # Get the data as a NumPy array
        if n_workers == 1:
            out_data_ = block.data.compute(
                scheduler='threads', num_workers=num_workers
            )
        else:
            with threading.Lock():
                out_data_ = block.data.compute(
                    scheduler='threads', num_workers=num_workers
                )

        if ('apply' in block.attrs) and not hasattr(
            block.attrs['apply'], 'wombat_func_'
        ):
            if padded_window_:
                # Add extra padding on the image borders
                rspad = (
                    padded_window_.height - window_.height
                    if window_.row_off == 0
                    else 0
                )
                cspad = (
                    padded_window_.width - window_.width
                    if window_.col_off == 0
                    else 0
                )
                repad = (
                    padded_window_.height - window_.height
                    if (window_.row_off != 0)
                    and (window_.height < block.gw.row_chunks)
                    else 0
                )
                cepad = (
                    padded_window_.width - window_.width
                    if (window_.col_off != 0)
                    and (window_.width < block.gw.col_chunks)
                    else 0
                )

                dshape = out_data_.shape

                if (rspad > 0) or (cspad > 0) or (repad > 0) or (cepad > 0):

                    if len(dshape) == 2:
                        out_data_ = np.pad(
                            out_data_,
                            ((rspad, repad), (cspad, cepad)),
                            mode='reflect',
                        )
                    elif len(dshape) == 3:
                        out_data_ = np.pad(
                            out_data_,
                            ((0, 0), (rspad, repad), (cspad, cepad)),
                            mode='reflect',
                        )
                    elif len(dshape) == 4:
                        out_data_ = np.pad(
                            out_data_,
                            ((0, 0), (0, 0), (rspad, repad), (cspad, cepad)),
                            mode='reflect',
                        )

            # Apply the user function
            if ('apply_args' in block.attrs) and (
                'apply_kwargs' in block.attrs
            ):
                out_data_ = block.attrs['apply'](
                    out_data_,
                    *block.attrs['apply_args'],
                    **block.attrs['apply_kwargs'],
                )
            elif ('apply_args' in block.attrs) and (
                'apply_kwargs' not in block.attrs
            ):
                out_data_ = block.attrs['apply'](
                    out_data_, *block.attrs['apply_args']
                )
            elif ('apply_args' not in block.attrs) and (
                'apply_kwargs' in block.attrs
            ):
                out_data_ = block.attrs['apply'](
                    out_data_, **block.attrs['apply_kwargs']
                )
            else:
                out_data_ = block.attrs['apply'](out_data_)

            if padded_window_:
                # Remove the extra padding
                dshape = out_data_.shape

                if len(dshape) == 2:
                    out_data_ = out_data_[
                        rspad : rspad + padded_window_.height,
                        cspad : cspad + padded_window_.width,
                    ]
                elif len(dshape) == 3:
                    out_data_ = out_data_[
                        :,
                        rspad : rspad + padded_window_.height,
                        cspad : cspad + padded_window_.width,
                    ]
                elif len(dshape) == 4:
                    out_data_ = out_data_[
                        :,
                        :,
                        rspad : rspad + padded_window_.height,
                        cspad : cspad + padded_window_.width,
                    ]

                dshape = out_data_.shape

                # Remove the padding
                # Get the non-padded array slice
                row_diff = abs(window_.row_off - padded_window_.row_off)
                col_diff = abs(window_.col_off - padded_window_.col_off)

                if len(dshape) == 2:
                    out_data_ = out_data_[
                        row_diff : row_diff + window_.height,
                        col_diff : col_diff + window_.width,
                    ]
                elif len(dshape) == 3:
                    out_data_ = out_data_[
                        :,
                        row_diff : row_diff + window_.height,
                        col_diff : col_diff + window_.width,
                    ]
                elif len(dshape) == 4:
                    out_data_ = out_data_[
                        :,
                        :,
                        row_diff : row_diff + window_.height,
                        col_diff : col_diff + window_.width,
                    ]

        else:
            if padded_window_:
                logger.warning(
                    '  Padding is only supported with user functions.'
                )

    if not isinstance(out_data_, np.ndarray):
        logger.exception(
            '  The data were not computed properly for block {:,d}'.format(wid)
        )

    dshape = out_data_.shape

    if len(dshape) > 2:
        out_data_ = out_data_.squeeze()

    if len(dshape) == 2:
        indexes_ = 1
    else:
        indexes_ = 1 if dshape[0] == 1 else list(range(1, dshape[0] + 1))

    return out_data_, indexes_, window_


@threadpool_limits.wrap(limits=1, user_api='blas')
def _write_xarray(*args):
    """Writes a DataArray to file.

    Args:
        args (iterable): A tuple from the window generator.

    Reference:
        https://github.com/dask/dask/issues/3600

    Returns:
        ``str`` | ``None``
    """
    zarr_file = None
    (
        block,
        filename,
        wid,
        block_window,
        padded_window,
        n_workers,
        n_threads,
        separate,
        chunks,
        root,
        out_block_type,
        tags,
        oleft,
        otop,
        ocols,
        orows,
        kwargs,
    ) = list(itertools.chain(*args))
    output, out_indexes, block_window = _compute_block(
        block, wid, block_window, padded_window, n_workers, n_threads
    )

    if separate and (out_block_type.lower() == 'zarr'):
        zarr_file = to_zarr(filename, output, block_window, chunks, root=root)
    else:
        to_gtiff(
            filename,
            output,
            block_window,
            out_indexes,
            block.gw.transform,
            n_workers,
            separate,
            tags,
            kwargs,
        )

    return zarr_file


def to_vrt(
    data,
    filename,
    overwrite=False,
    resampling=None,
    nodata=None,
    init_dest_nodata=True,
    warp_mem_limit=128,
):
    """Writes a file to a VRT file.

    Args:
        data (DataArray): The ``xarray.DataArray`` to write.
        filename (str): The output file name to write to.
        overwrite (Optional[bool]): Whether to overwrite an existing VRT file.
        resampling (Optional[object]): The resampling algorithm for ``rasterio.vrt.WarpedVRT``. Default is 'nearest'.
        nodata (Optional[float or int]): The 'no data' value for ``rasterio.vrt.WarpedVRT``.
        init_dest_nodata (Optional[bool]): Whether or not to initialize output to ``nodata`` for ``rasterio.vrt.WarpedVRT``.
        warp_mem_limit (Optional[int]): The GDAL memory limit for ``rasterio.vrt.WarpedVRT``.

    Returns:
        ``None``, writes to ``filename``

    Examples:
        >>> import geowombat as gw
        >>> from rasterio.enums import Resampling
        >>>
        >>> # Transform a CRS and save to VRT
        >>> with gw.config.update(ref_crs=102033):
        >>>     with gw.open('image.tif') as src:
        >>>         gw.to_vrt(
        >>>             src,
        >>>             'output.vrt',
        >>>             resampling=Resampling.cubic,
        >>>             warp_mem_limit=256
        >>>         )
        >>>
        >>> # Load multiple files set to a common geographic extent
        >>> bounds = (left, bottom, right, top)
        >>> with gw.config.update(ref_bounds=bounds):
        >>>     with gw.open(
        >>>         ['image1.tif', 'image2.tif'], mosaic=True
        >>>     ) as src:
        >>>         gw.to_vrt(src, 'output.vrt')
    """
    if Path(filename).is_file():
        if overwrite:
            Path(filename).unlink()
        else:
            logger.warning(f'  The VRT file {filename} already exists.')
            return

    if not resampling:
        resampling = Resampling.nearest

    if isinstance(data.attrs['filename'], str) or isinstance(
        data.attrs['filename'], Path
    ):
        # Open the input file on disk
        with rio.open(data.attrs['filename']) as src:
            with WarpedVRT(
                src,
                src_crs=src.crs,  # the original CRS
                crs=data.crs,  # the transformed CRS
                src_transform=src.gw.transform,  # the original transform
                transform=data.gw.transform,  # the new transform
                dtype=data.dtype,
                resampling=resampling,
                nodata=nodata,
                init_dest_nodata=init_dest_nodata,
                warp_mem_limit=warp_mem_limit,
            ) as vrt:
                rio_shutil.copy(vrt, filename, driver='VRT')

    else:
        if not data.gw.filenames:
            logger.exception(
                '  The data filenames attribute is empty. Use gw.open(..., persist_filenames=True).'
            )
            raise KeyError

        separate = (
            True
            if data.gw.data_are_separate and data.gw.data_are_stacked
            else False
        )

        vrt_options = gdal.BuildVRTOptions(
            outputBounds=data.gw.bounds,
            xRes=data.gw.cellx,
            yRes=data.gw.celly,
            separate=separate,
            outputSRS=data.crs,
        )
        ds = gdal.BuildVRT(filename, data.gw.filenames, options=vrt_options)
        ds = None


def to_netcdf(
    data: xr.DataArray,
    filename: T.Union[str, Path],
    overwrite: T.Optional[bool] = False,
    compute: T.Optional[bool] = True,
    *args,
    **kwargs,
):
    """Writes an Xarray DataArray to a NetCDF file.

    Args:
        data (DataArray): The ``xarray.DataArray`` to write.
        filename (str): The output file name to write to.
        overwrite (Optional[bool]): Whether to overwrite an existing file. Default is ``False``.
        compute (Optinoal[bool]): Whether to compute and write to ``filename``. Otherwise, return
            the ``dask`` task graph. Default is ``True``.
        args (DataArray): Additional ``DataArrays`` to stack.
        kwargs (dict): Encoding arguments.

    Return:
        ``None``, writes to ``filename``

    Examples:
        >>> import geowombat as gw
        >>> import xarray as xr
        >>>
        >>> # Write a single DataArray to a .nc file
        >>> with gw.config.update(sensor='l7'):
        >>>     with gw.open('LC08_L1TP_225078_20200219_20200225_01_T1.tif') as src:
        >>>         gw.to_netcdf(src, 'filename.nc', zlib=True, complevel=5)
        >>>
        >>> # Add extra layers
        >>> with gw.config.update(sensor='l7'):
        >>>     with gw.open(
        >>>         'LC08_L1TP_225078_20200219_20200225_01_T1.tif'
        >>>     ) as src, gw.open(
        >>>         'LC08_L1TP_225078_20200219_20200225_01_T1_angles.tif',
        >>>         band_names=['zenith', 'azimuth']
        >>>     ) as ang:
        >>>         src = (
        >>>             xr.where(
        >>>                 src == 0, -32768, src
        >>>             )
        >>>             .astype('int16')
        >>>             .assign_attrs(**src.attrs)
        >>>         )
        >>>
        >>>         gw.to_netcdf(
        >>>             src,
        >>>             'filename.nc',
        >>>             ang.astype('int16'),
        >>>             zlib=True,
        >>>             complevel=5,
        >>>             _FillValue=-32768
        >>>         )
        >>>
        >>> # Open the data and convert to a DataArray
        >>> with xr.open_dataset(
        >>>     'filename.nc', engine='h5netcdf', chunks=256
        >>> ) as ds:
        >>>     src = ds.to_array(dim='band')
    """
    if Path(filename).is_file():
        if overwrite:
            Path(filename).unlink()
        else:
            logger.warning(f'The file {str(filename)} already exists.')
            return

    encodings = {}
    chunksize = min(data.gw.row_chunks, data.gw.col_chunks)

    for band_name in data.band.values.tolist():
        encode_dict = {
            'chunksizes': (chunksize, chunksize),
            'dtype': data.dtype.name
            if isinstance(data.dtype, np.dtype)
            else data.dtype,
        }
        encode_dict.update(**kwargs)
        encodings[band_name] = encode_dict

    for other_data in args:
        chunksize = min(other_data.gw.row_chunks, other_data.gw.col_chunks)
        for band_name in other_data.band.values.tolist():
            encode_dict = {
                'chunksizes': (chunksize, chunksize),
                'dtype': other_data.dtype.name
                if isinstance(other_data.dtype, np.dtype)
                else other_data.dtype,
            }
            encode_dict.update(**kwargs)
            encodings[band_name] = encode_dict

        data = xr.concat((data, other_data), dim='band')

    attrs = data.attrs.copy()
    attrs['crs'] = f"epsg:{data.gw.crs_to_pyproj.to_epsg()}"
    ds = data.to_dataset(dim='band').assign_attrs(**attrs)
    if compute:
        (
            ds.to_netcdf(
                path=filename,
                mode='w',
                format='NETCDF4',
                engine='h5netcdf',
                encoding=encodings,
                compute=True,
            )
        )
    else:
        return ds.to_netcdf(
            path=filename,
            mode='w',
            format='NETCDF4',
            engine='h5netcdf',
            encoding=encodings,
            compute=False,
        )


def save(
    data: xr.DataArray,
    filename: T.Union[str, Path],
    mode: T.Optional[str] = 'w',
    nodata: T.Optional[T.Union[float, int]] = None,
    overwrite: T.Optional[bool] = False,
    client: T.Optional[Client] = None,
    compute: T.Optional[bool] = True,
    tags: T.Optional[dict] = None,
    compress: T.Optional[str] = 'none',
    compression: T.Optional[str] = None,
    num_workers: T.Optional[int] = 1,
    log_progress: T.Optional[bool] = True,
    tqdm_kwargs: T.Optional[dict] = None,
):
    """Saves a DataArray to raster using rasterio/dask.

    Args:
        filename (str | Path): The output file name to write to.
        overwrite (Optional[bool]): Whether to overwrite an existing file. Default is False.
        mode (Optional[str]): The file storage mode. Choices are ['w', 'r+'].
        nodata (Optional[float | int]): The 'no data' value. If ``None`` (default), the 'no data'
            value is taken from the ``DataArray`` metadata.
        client (Optional[Client object]): A ``dask.distributed.Client`` client object to persist data.
            Default is None.
        compute (Optinoal[bool]): Whether to compute and write to ``filename``. Otherwise, return
            the ``dask`` task graph. If ``True``, compute and write to ``filename``. If ``False``,
            return the ``dask`` task graph. Default is ``True``.
        tags (Optional[dict]): Metadata tags to write to file. Default is None.
        compress (Optional[str]): The file compression type. Default is 'none', or no compression.
        compression (Optional[str]): The file compression type. Default is 'none', or no compression.

            .. deprecated:: 2.1.4
                Use 'compress' -- 'compression' will be removed in >=2.2.0.

        num_workers (Optional[int]): The number of dask workers (i.e., chunks) to write concurrently.
            Default is 1.
        log_progress (Optional[bool]): Whether to log the progress bar during writing. Default is True.
        tqdm_kwargs (Optional[dict]): Keyword arguments to pass to ``tqdm``.

    Returns:
        ``None``, writes to ``filename``

    Example:
        >>> import geowombat as gw
        >>>
        >>> with gw.open('file.tif') as src:
        >>>     result = ...
        >>>     gw.save(result, 'output.tif', compress='lzw', num_workers=8)
        >>>
        >>> # Create delayed write tasks and compute later
        >>> tasks = [gw.save(array, 'output.tif', compute=False) for array in array_list]
        >>> # Write and close files
        >>> dask.compute(tasks, num_workers=8)
    """
    if compression is not None:
        warnings.warn(
            "The argument 'compression' will be deprecated in >=2.2.0. Use 'compress'.",
            DeprecationWarning,
            stacklevel=2,
        )
        compress = compression

    if mode not in ['w', 'r+']:
        raise AttributeError("The mode must be either 'w' or 'r+'.")

    if Path(filename).is_file():
        if overwrite:
            Path(filename).unlink()
        else:
            logger.warning(f'The file {str(filename)} already exists.')
            return

    if nodata is None:
        if hasattr(data, '_FillValue'):
            nodata = data.attrs['_FillValue']
        else:
            if hasattr(data, 'nodatavals'):
                nodata = data.attrs['nodatavals'][0]
            else:
                raise AttributeError(
                    "The DataArray does not have any 'no data' attributes."
                )
    dtype = data.dtype.name if isinstance(data.dtype, np.dtype) else data.dtype
    if isinstance(nodata, float):
        if dtype != 'float32':
            dtype = 'float64'

    blockxsize = (
        data.gw.check_chunksize(512, data.gw.ncols)
        if not data.gw.array_is_dask
        else data.gw.col_chunks
    )
    blockysize = (
        data.gw.check_chunksize(512, data.gw.nrows)
        if not data.gw.array_is_dask
        else data.gw.row_chunks
    )

    kwargs = dict(
        driver=driver_from_extension(filename),
        width=data.gw.ncols,
        height=data.gw.nrows,
        count=data.gw.nbands,
        dtype=dtype,
        nodata=nodata,
        blockxsize=blockxsize,
        blockysize=blockysize,
        crs=data.gw.crs_to_pyproj,
        transform=data.gw.transform,
        compress=compress,
        tiled=True if max(blockxsize, blockysize) >= 16 else False,
        sharing=False,
    )
    if tqdm_kwargs is None:
        tqdm_kwargs = {}

    if not compute:
        return (
            RasterioStore(filename, mode=mode, tags=tags, **kwargs)
            .open()
            .write_delayed(data)
        )

    else:
        with RasterioStore(
            filename, mode=mode, tags=tags, **kwargs
        ) as rio_store:
            # Store the data and return a lazy evaluator
            res = rio_store.write(data)

            if client is not None:
                results = client.persist(res)
                if log_progress:
                    progress(results)
                dask.compute(results)
            else:
                if log_progress:
                    with TqdmCallback(**tqdm_kwargs):
                        dask.compute(res, num_workers=num_workers)
                else:
                    dask.compute(res, num_workers=num_workers)

    return None


def to_raster(
    data,
    filename,
    readxsize=None,
    readysize=None,
    separate=False,
    out_block_type='gtiff',
    keep_blocks=False,
    verbose=0,
    overwrite=False,
    gdal_cache=512,
    scheduler='mpool',
    n_jobs=1,
    n_workers=None,
    n_threads=None,
    n_chunks=None,
    padding=None,
    tags=None,
    tqdm_kwargs=None,
    **kwargs,
):
    """Writes a ``dask`` array to a raster file.

    .. note::

        We advise using :func:`save` in place of this method.

    Args:
        data (DataArray): The ``xarray.DataArray`` to write.
        filename (str): The output file name to write to.
        readxsize (Optional[int]): The size of column chunks to read. If not given, ``readxsize`` defaults to Dask
            chunk size.
        readysize (Optional[int]): The size of row chunks to read. If not given, ``readysize`` defaults to Dask
            chunk size.
        separate (Optional[bool]): Whether to write blocks as separate files. Otherwise, write to a single file.
        out_block_type (Optional[str]): The output block type. Choices are ['gtiff', 'zarr'].
            Only used if ``separate`` = ``True``.
        keep_blocks (Optional[bool]): Whether to keep the blocks stored on disk. Only used if ``separate`` = ``True``.
        verbose (Optional[int]): The verbosity level.
        overwrite (Optional[bool]): Whether to overwrite an existing file.
        gdal_cache (Optional[int]): The ``GDAL`` cache size (in MB).
        scheduler (Optional[str]): The parallel task scheduler to use. Choices are ['processes', 'threads', 'mpool'].

            mpool: process pool of workers using ``multiprocessing.Pool``
            processes: process pool of workers using ``concurrent.futures``
            threads: thread pool of workers using ``concurrent.futures``

        n_jobs (Optional[int]): The total number of parallel jobs.
        n_workers (Optional[int]): The number of process workers.
        n_threads (Optional[int]): The number of thread workers.
        n_chunks (Optional[int]): The chunk size of windows. If not given, equal to ``n_workers`` x 50.
        overviews (Optional[bool or list]): Whether to build overview layers.
        resampling (Optional[str]): The resampling method for overviews when ``overviews`` is ``True`` or a ``list``.
            Choices are ['average', 'bilinear', 'cubic', 'cubic_spline', 'gauss', 'lanczos', 'max', 'med', 'min', 'mode', 'nearest'].
        padding (Optional[tuple]): Padding for each window. ``padding`` should be given as a tuple
            of (left pad, bottom pad, right pad, top pad). If ``padding`` is given, the returned list will contain
            a tuple of ``rasterio.windows.Window`` objects as (w1, w2), where w1 contains the normal window offsets
            and w2 contains the padded window offsets.
        tags (Optional[dict]): Image tags to write to file.
        tqdm_kwargs (Optional[dict]): Additional keyword arguments to pass to ``tqdm``.
        kwargs (Optional[dict]): Additional keyword arguments to pass to ``rasterio.write``.

    Returns:
        ``None``, writes to ``filename``

    Examples:
        >>> import geowombat as gw
        >>>
        >>> # Use 8 parallel workers
        >>> with gw.open('input.tif') as ds:
        >>>     gw.to_raster(ds, 'output.tif', n_jobs=8)
        >>>
        >>> # Use 4 process workers and 2 thread workers
        >>> with gw.open('input.tif') as ds:
        >>>     gw.to_raster(ds, 'output.tif', n_workers=4, n_threads=2)
        >>>
        >>> # Control the window chunks passed to concurrent.futures
        >>> with gw.open('input.tif') as ds:
        >>>     gw.to_raster(ds, 'output.tif', n_workers=4, n_threads=2, n_chunks=16)
        >>>
        >>> # Compress the output and build overviews
        >>> with gw.open('input.tif') as ds:
        >>>     gw.to_raster(ds, 'output.tif', n_jobs=8, overviews=True, compress='lzw')
    """
    if separate and not ZARR_INSTALLED and (out_block_type.lower() == 'zarr'):
        logger.exception('  zarr must be installed to write separate blocks.')
        raise ImportError

    pfile = Path(filename)

    if scheduler.lower() == 'mpool':
        pool_executor = multi.Pool
    else:
        pool_executor = (
            concurrent.futures.ProcessPoolExecutor
            if scheduler.lower() == 'processes'
            else concurrent.futures.ThreadPoolExecutor
        )

    if overwrite:
        if pfile.is_file():
            pfile.unlink()

    if pfile.is_file():
        logger.warning('  The output file already exists.')
        return

    if not is_dask_collection(data.data):
        logger.exception('  The data should be a dask array.')

    if isinstance(n_workers, int) and isinstance(n_threads, int):
        n_jobs = n_workers * n_threads
    else:
        n_workers = n_jobs
        n_threads = 1

    if not isinstance(n_chunks, int):
        n_chunks = n_workers * 50

    if not isinstance(readxsize, int):
        readxsize = data.gw.col_chunks

    if not isinstance(readysize, int):
        readysize = data.gw.row_chunks

    chunksize = (data.gw.row_chunks, data.gw.col_chunks)

    if tqdm_kwargs is None:
        tqdm_kwargs = {}

    # Force tiled outputs with no file sharing
    kwargs['sharing'] = False

    if data.gw.tiled:
        kwargs['tiled'] = True

    if 'compress' in kwargs:
        # boolean True or '<>'
        if kwargs['compress']:
            if (
                isinstance(kwargs['compress'], str)
                and kwargs['compress'].lower() == 'none'
            ):
                compress = False
            else:
                if 'num_threads' in kwargs:
                    compress = False
                else:
                    compress = True

                if compress:
                    # Store the compression type because
                    #   it is removed in concurrent writing
                    compress_type = kwargs['compress']
                    del kwargs['compress']

        else:
            compress = False

    elif isinstance(data.gw.compress, str) and (
        data.gw.compress.lower() in ['lzw', 'deflate']
    ):
        compress = True
        compress_type = data.gw.compress

    else:
        compress = False

    if 'nodata' not in kwargs:
        if isinstance(data.gw.nodata, int) or isinstance(
            data.gw.nodata, float
        ):
            kwargs['nodata'] = data.gw.nodata

    if 'blockxsize' not in kwargs:
        kwargs['blockxsize'] = data.gw.col_chunks

    if 'blockysize' not in kwargs:
        kwargs['blockysize'] = data.gw.row_chunks

    if 'bigtiff' not in kwargs:
        kwargs['bigtiff'] = data.gw.bigtiff

    if 'driver' not in kwargs:
        kwargs['driver'] = data.gw.driver

    if 'count' not in kwargs:
        kwargs['count'] = data.gw.nbands

    if 'width' not in kwargs:
        kwargs['width'] = data.gw.ncols

    if 'height' not in kwargs:
        kwargs['height'] = data.gw.nrows

    if 'transform' not in kwargs:
        kwargs['transform'] = data.gw.transform

    if 'num_threads' in kwargs:
        if isinstance(kwargs['num_threads'], str):
            kwargs['num_threads'] = 'all_cpus'

    if 'crs' in kwargs:
        crs = kwargs['crs']
    else:
        crs = data.crs

    if str(crs).lower().startswith('epsg:'):
        kwargs['crs'] = pyproj.CRS.from_user_input(crs)
    else:
        try:
            kwargs['crs'] = pyproj.CRS.from_epsg(int(crs))
        except ValueError:
            kwargs['crs'] = pyproj.CRS.from_user_input(crs)
    kwargs['crs'] = kwargs['crs'].to_wkt()

    root = None

    if separate and (out_block_type.lower() == 'zarr'):
        d_name = pfile.parent
        sub_dir = d_name.joinpath('sub_tmp_')
        sub_dir.mkdir(parents=True, exist_ok=True)
        zarr_file = str(sub_dir.joinpath('data.zarr'))
        root = zarr.open(zarr_file, mode='w')

    else:
        if not separate:
            if verbose > 0:
                logger.info('  Creating the file ...\n')

            with rio.open(filename, mode='w', **kwargs) as rio_dst:
                if tags:
                    rio_dst.update_tags(**tags)

    if verbose > 0:
        logger.info('  Writing data to file ...\n')

    with rio.Env(GDAL_CACHEMAX=gdal_cache):
        windows = get_window_offsets(
            data.gw.nrows,
            data.gw.ncols,
            readysize,
            readxsize,
            return_as='list',
            padding=padding,
        )

        n_windows = len(windows)

        oleft, otop = kwargs['transform'][2], kwargs['transform'][5]
        ocols, orows = kwargs['width'], kwargs['height']

        # Iterate over the windows in chunks
        for wchunk in range(0, n_windows, n_chunks):
            window_slice = windows[wchunk : wchunk + n_chunks]
            n_windows_slice = len(window_slice)

            if verbose > 0:
                logger.info(
                    '  Windows {:,d}--{:,d} of {:,d} ...'.format(
                        wchunk + 1, wchunk + n_windows_slice, n_windows
                    )
                )

            if padding:
                # Read the padded window
                if len(data.shape) == 2:
                    data_gen = (
                        (
                            data[
                                w[1].row_off : w[1].row_off + w[1].height,
                                w[1].col_off : w[1].col_off + w[1].width,
                            ],
                            filename,
                            widx + wchunk,
                            w[0],
                            w[1],
                            n_workers,
                            n_threads,
                            separate,
                            chunksize,
                            root,
                            out_block_type,
                            tags,
                            oleft,
                            otop,
                            ocols,
                            orows,
                            kwargs,
                        )
                        for widx, w in enumerate(window_slice)
                    )

                elif len(data.shape) == 3:
                    data_gen = (
                        (
                            data[
                                :,
                                w[1].row_off : w[1].row_off + w[1].height,
                                w[1].col_off : w[1].col_off + w[1].width,
                            ],
                            filename,
                            widx + wchunk,
                            w[0],
                            w[1],
                            n_workers,
                            n_threads,
                            separate,
                            chunksize,
                            root,
                            out_block_type,
                            tags,
                            oleft,
                            otop,
                            ocols,
                            orows,
                            kwargs,
                        )
                        for widx, w in enumerate(window_slice)
                    )

                else:
                    data_gen = (
                        (
                            data[
                                :,
                                :,
                                w[1].row_off : w[1].row_off + w[1].height,
                                w[1].col_off : w[1].col_off + w[1].width,
                            ],
                            filename,
                            widx + wchunk,
                            w[0],
                            w[1],
                            n_workers,
                            n_threads,
                            separate,
                            chunksize,
                            root,
                            out_block_type,
                            tags,
                            oleft,
                            otop,
                            ocols,
                            orows,
                            kwargs,
                        )
                        for widx, w in enumerate(window_slice)
                    )

            else:
                if len(data.shape) == 2:
                    data_gen = (
                        (
                            data[
                                w.row_off : w.row_off + w.height,
                                w.col_off : w.col_off + w.width,
                            ],
                            filename,
                            widx + wchunk,
                            w,
                            None,
                            n_workers,
                            n_threads,
                            separate,
                            chunksize,
                            root,
                            out_block_type,
                            tags,
                            oleft,
                            otop,
                            ocols,
                            orows,
                            kwargs,
                        )
                        for widx, w in enumerate(window_slice)
                    )

                elif len(data.shape) == 3:
                    data_gen = (
                        (
                            data[
                                :,
                                w.row_off : w.row_off + w.height,
                                w.col_off : w.col_off + w.width,
                            ],
                            filename,
                            widx + wchunk,
                            w,
                            None,
                            n_workers,
                            n_threads,
                            separate,
                            chunksize,
                            root,
                            out_block_type,
                            tags,
                            oleft,
                            otop,
                            ocols,
                            orows,
                            kwargs,
                        )
                        for widx, w in enumerate(window_slice)
                    )

                else:
                    data_gen = (
                        (
                            data[
                                :,
                                :,
                                w.row_off : w.row_off + w.height,
                                w.col_off : w.col_off + w.width,
                            ],
                            filename,
                            widx + wchunk,
                            w,
                            None,
                            n_workers,
                            n_threads,
                            separate,
                            chunksize,
                            root,
                            out_block_type,
                            tags,
                            oleft,
                            otop,
                            ocols,
                            orows,
                            kwargs,
                        )
                        for widx, w in enumerate(window_slice)
                    )

            if n_workers == 1:
                for __ in tqdm(
                    map(_write_xarray, data_gen),
                    total=n_windows_slice,
                    **tqdm_kwargs,
                ):
                    pass

            else:
                with pool_executor(n_workers) as executor:
                    if scheduler == 'mpool':
                        for __ in tqdm(
                            executor.imap_unordered(_write_xarray, data_gen),
                            total=n_windows_slice,
                            **tqdm_kwargs,
                        ):
                            pass

                    else:
                        for __ in tqdm(
                            executor.map(_write_xarray, data_gen),
                            total=n_windows_slice,
                            **tqdm_kwargs,
                        ):
                            pass

        if compress:
            if separate:
                if out_block_type.lower() == 'zarr':
                    group_keys = list(root.group_keys())
                    n_groups = len(group_keys)

                    if out_block_type.lower() == 'zarr':
                        open_file = zarr_file

                    kwargs['compress'] = compress_type
                    n_windows = len(group_keys)

                    # Compress into one file
                    with rio.open(filename, mode='w', **kwargs) as dst_:
                        if tags:
                            dst_.update_tags(**tags)

                        # Iterate over the windows in chunks
                        for wchunk in range(0, n_groups, n_chunks):
                            group_keys_slice = group_keys[
                                wchunk : wchunk + n_chunks
                            ]
                            n_windows_slice = len(group_keys_slice)

                            if verbose > 0:
                                logger.info(
                                    '  Windows {:,d}--{:,d} of {:,d} ...'.format(
                                        wchunk + 1,
                                        wchunk + n_windows_slice,
                                        n_windows,
                                    )
                                )

                            ################################################
                            data_gen = (
                                (open_file, group, 'zarr')
                                for group in group_keys_slice
                            )

                            with concurrent.futures.ProcessPoolExecutor(
                                max_workers=n_workers
                            ) as executor:
                                # Submit all the tasks as futures
                                futures = [
                                    executor.submit(_block_read_func, f, g, t)
                                    for f, g, t in data_gen
                                ]

                                for f in tqdm(
                                    concurrent.futures.as_completed(futures),
                                    total=n_windows_slice,
                                    **tqdm_kwargs,
                                ):
                                    (
                                        out_window,
                                        out_indexes,
                                        out_block,
                                    ) = f.result()
                                    dst_.write(
                                        out_block,
                                        window=out_window,
                                        indexes=out_indexes,
                                    )

                            futures = None

                    if not keep_blocks:
                        shutil.rmtree(sub_dir)

            else:

                if verbose > 0:
                    logger.info('  Compressing output file ...')

                p = Path(filename)

                d_name = p.parent
                f_base, f_ext = os.path.splitext(p.name)

                ld = string.ascii_letters + string.digits
                rstr = ''.join(random.choice(ld) for i in range(0, 9))

                temp_file = d_name.joinpath(
                    '{f_base}_temp_{rstr}{f_ext}'.format(
                        f_base=f_base, rstr=rstr, f_ext=f_ext
                    )
                )

                compress_raster(
                    filename,
                    str(temp_file),
                    n_jobs=n_jobs,
                    gdal_cache=gdal_cache,
                    compress=compress_type,
                    tags=tags,
                )

                temp_file.replace(filename)

            if verbose > 0:
                logger.info('  Finished compressing')

    if verbose > 0:
        logger.info('\nFinished writing the data.')


def _arg_gen(arg_, iter_):
    for i_ in iter_:
        yield arg_


def apply(
    infile,
    outfile,
    block_func,
    args=None,
    count=1,
    scheduler='processes',
    gdal_cache=512,
    n_jobs=4,
    overwrite=False,
    tags=None,
    **kwargs,
):
    """Applies a function and writes results to file.

    Args:
        infile (str): The input file to process.
        outfile (str): The output file.
        block_func (func): The user function to apply to each block. The function should always return the window,
            the data, and at least one argument. The block data inside the function will be a 2d array if the
            input image has 1 band, otherwise a 3d array.
        args (Optional[tuple]): Additional arguments to pass to ``block_func``.
        count (Optional[int]): The band count for the output file.
        scheduler (Optional[str]): The ``concurrent.futures`` scheduler to use. Choices are ['threads', 'processes'].
        gdal_cache (Optional[int]): The ``GDAL`` cache size (in MB).
        n_jobs (Optional[int]): The number of blocks to process in parallel.
        overwrite (Optional[bool]): Whether to overwrite an existing output file.
        tags (Optional[dict]): Image tags to write to file.
        kwargs (Optional[dict]): Additional keyword arguments to pass to ``rasterio.open``.

    Returns:
        ``None``, writes to ``outfile``

    Examples:
        >>> import geowombat as gw
        >>>
        >>> # Here is a function with no arguments
        >>> def my_func0(w, block, arg):
        >>>     return w, block
        >>>
        >>> gw.apply('input.tif',
        >>>          'output.tif',
        >>>           my_func0,
        >>>           n_jobs=8)
        >>>
        >>> # Here is a function with 1 argument
        >>> def my_func1(w, block, arg):
        >>>     return w, block * arg
        >>>
        >>> gw.apply('input.tif',
        >>>          'output.tif',
        >>>           my_func1,
        >>>           args=(10.0,),
        >>>           n_jobs=8)
    """

    if not args:
        args = (None,)

    if overwrite:

        if os.path.isfile(outfile):
            os.remove(outfile)

    kwargs['sharing'] = False
    kwargs['tiled'] = True

    io_mode = 'r+' if os.path.isfile(outfile) else 'w'
    out_indexes = 1 if count == 1 else list(range(1, count + 1))
    futures_executor = (
        concurrent.futures.ThreadPoolExecutor
        if scheduler == 'threads'
        else concurrent.futures.ProcessPoolExecutor
    )

    with rio.Env(GDAL_CACHEMAX=gdal_cache):
        with rio.open(infile) as src:
            profile = src.profile.copy()

            if 'dtype' not in kwargs:
                kwargs['dtype'] = profile['dtype']

            if 'nodata' not in kwargs:
                kwargs['nodata'] = profile['nodata']

            if 'blockxsize' not in kwargs:
                kwargs['blockxsize'] = profile['blockxsize']

            if 'blockxsize' not in kwargs:
                kwargs['blockysize'] = profile['blockysize']

            # Create a destination dataset based on source params. The
            # destination will be tiled, and we'll process the tiles
            # concurrently.
            profile.update(count=count, **kwargs)

            with rio.open(outfile, io_mode, **profile) as dst:
                if tags:
                    dst.update_tags(**tags)

                # This generator comprehension gives us raster data
                # arrays for each window. Later we will zip a mapping
                # of it with the windows list to get (window, result)
                # pairs.
                data_gen = (
                    src.read(window=w, out_dtype=profile['dtype'])
                    for ij, w in src.block_windows(1)
                )

                if args:
                    args = [
                        _arg_gen(arg, src.block_windows(1)) for arg in args
                    ]

                with futures_executor(max_workers=n_jobs) as executor:
                    # Submit all of the tasks as futures
                    futures = [
                        executor.submit(block_func, iter_[0][1], *iter_[1:])
                        for iter_ in zip(
                            list(src.block_windows(1)), data_gen, *args
                        )
                    ]

                    for f in tqdm(
                        concurrent.futures.as_completed(futures),
                        total=len(futures),
                    ):
                        out_window, out_block = f.result()
                        dst.write(
                            np.squeeze(out_block),
                            window=out_window,
                            indexes=out_indexes,
                        )


def _compress_dummy(w, block, dummy):
    """Dummy function to pass to concurrent writing."""
    return w, block


def compress_raster(
    infile, outfile, n_jobs=1, gdal_cache=512, compress='lzw', tags=None
):
    """Compresses a raster file.

    Args:
        infile (str): The file to compress.
        outfile (str): The output file.
        n_jobs (Optional[int]): The number of concurrent blocks to write.
        gdal_cache (Optional[int]): The ``GDAL`` cache size (in MB).
        compress (Optional[str]): The compression method.
        tags (Optional[dict]): Image tags to write to file.

    Returns:
        None
    """
    with rio.open(infile) as src:
        profile = src.profile.copy()
        profile.update(compress=compress)
        apply(
            infile,
            outfile,
            _compress_dummy,
            scheduler='processes',
            args=(None,),
            gdal_cache=gdal_cache,
            n_jobs=n_jobs,
            tags=tags,
            count=src.count,
            dtype=src.profile['dtype'],
            nodata=src.profile['nodata'],
            tiled=src.profile['tiled'],
            blockxsize=src.profile['blockxsize'],
            blockysize=src.profile['blockysize'],
            compress=compress,
        )
