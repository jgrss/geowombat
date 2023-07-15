import contextlib
import logging
import os
import typing as T
import warnings
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
from dask.delayed import Delayed
from rasterio import open as rio_open
from rasterio.coords import BoundingBox
from rasterio.windows import Window

from ..config import config
from ..core.util import parse_filename_dates
from ..core.windows import get_window_offsets
from ..handler import add_handler
from .rasterio_ import get_file_bounds, get_ref_image_meta
from .rasterio_ import transform_crs as rio_transform_crs
from .rasterio_ import (
    unpack_bounding_box,
    unpack_window,
    warp,
    warp_images,
    window_to_bounds,
)
from .xarray_rasterio_ import open_rasterio

logger = logging.getLogger(__name__)
logger = add_handler(logger)


def _update_kwarg(ref_obj, ref_kwargs, key):
    """Updates keyword arguments for global config parameters.

    Args:
        ref_obj (str or object)
        ref_kwargs (dict)
        key (str)

    Returns:
        ``dict``
    """
    if isinstance(ref_obj, str) and Path(ref_obj).is_file():
        # Get the metadata from the reference image
        ref_meta = get_ref_image_meta(ref_obj)
        ref_kwargs[key] = getattr(ref_meta, key)

    else:
        if ref_obj is not None:
            ref_kwargs[key] = ref_obj

    return ref_kwargs


def _get_raster_coords(filename):
    with open_rasterio(filename) as src:
        x = src.x.values - src.res[0] / 2.0
        y = src.y.values + src.res[1] / 2.0

    return x, y


def _check_config_globals(filenames, bounds_by, ref_kwargs):
    """Checks global configuration parameters.

    Args:
        filenames (str or str list)
        bounds_by (str)
        ref_kwargs (dict)
    """
    if config['nodata'] is not None:
        ref_kwargs = _update_kwarg(config['nodata'], ref_kwargs, 'nodata')
    # Check if there is a reference image
    if config['ref_image']:
        if isinstance(config['ref_image'], str) and os.path.isfile(
            config['ref_image']
        ):
            # Get the metadata from the reference image
            ref_meta = get_ref_image_meta(config['ref_image'])
            ref_kwargs['bounds'] = ref_meta.bounds
            ref_kwargs['crs'] = ref_meta.crs
            ref_kwargs['res'] = ref_meta.res

        else:
            if not config['ignore_warnings']:
                logger.warning('  The reference image does not exist')

    else:
        if config['ref_bounds']:
            if isinstance(config['ref_bounds'], str) and config[
                'ref_bounds'
            ].startswith('Window'):
                ref_bounds_ = window_to_bounds(
                    filenames, unpack_window(config['ref_bounds'])
                )
            elif isinstance(config['ref_bounds'], str) and config[
                'ref_bounds'
            ].startswith('BoundingBox'):
                ref_bounds_ = unpack_bounding_box(config['ref_bounds'])
            elif isinstance(config['ref_bounds'], Window):
                ref_bounds_ = window_to_bounds(filenames, config['ref_bounds'])
            elif isinstance(config['ref_bounds'], BoundingBox):

                ref_bounds_ = (
                    config['ref_bounds'].left,
                    config['ref_bounds'].bottom,
                    config['ref_bounds'].right,
                    config['ref_bounds'].top,
                )

            else:
                ref_bounds_ = config['ref_bounds']

            ref_kwargs = _update_kwarg(ref_bounds_, ref_kwargs, 'bounds')

        else:
            if isinstance(filenames, str) or isinstance(filenames, Path):
                # Use the bounds of the image
                ref_kwargs['bounds'] = get_file_bounds(
                    [filenames],
                    bounds_by='reference',
                    crs=ref_kwargs['crs'],
                    res=ref_kwargs['res'],
                    return_bounds=True,
                )

            else:
                # Replace the bounds keyword, if needed
                if bounds_by.lower() == 'intersection':
                    # Get the intersecting bounds of all images
                    ref_kwargs['bounds'] = get_file_bounds(
                        filenames,
                        bounds_by='intersection',
                        crs=ref_kwargs['crs'],
                        res=ref_kwargs['res'],
                        return_bounds=True,
                    )

                elif bounds_by.lower() == 'union':
                    # Get the union bounds of all images
                    ref_kwargs['bounds'] = get_file_bounds(
                        filenames,
                        bounds_by='union',
                        crs=ref_kwargs['crs'],
                        res=ref_kwargs['res'],
                        return_bounds=True,
                    )

                elif bounds_by.lower() == 'reference':
                    # Use the bounds of the first image
                    ref_kwargs['bounds'] = get_file_bounds(
                        filenames,
                        bounds_by='reference',
                        crs=ref_kwargs['crs'],
                        res=ref_kwargs['res'],
                        return_bounds=True,
                    )

                else:
                    logger.exception(
                        "  Choose from 'intersection', 'union', or 'reference'."
                    )

                config['ref_bounds'] = ref_kwargs['bounds']

        if config['ref_crs'] is not None:
            ref_kwargs = _update_kwarg(config['ref_crs'], ref_kwargs, 'crs')

        if config['ref_res'] is not None:
            ref_kwargs = _update_kwarg(config['ref_res'], ref_kwargs, 'res')

        if config['ref_tar'] is not None:
            if isinstance(config['ref_tar'], str):
                if os.path.isfile(config['ref_tar']):
                    ref_kwargs = _update_kwarg(
                        _get_raster_coords(config['ref_tar']),
                        ref_kwargs,
                        'tac',
                    )
                else:

                    if not config['ignore_warnings']:
                        logger.warning(
                            '  The target aligned raster does not exist.'
                        )

            else:
                if not config['ignore_warnings']:
                    logger.warning(
                        '  The target aligned raster must be an image.'
                    )

    return ref_kwargs


def delayed_to_xarray(
    delayed_data: Delayed,
    shape: tuple,
    dtype: T.Union[str, np.dtype],
    chunks: tuple,
    coords: dict,
    attrs: dict,
) -> xr.DataArray:
    """Converts a dask.Delayed array to a Xarray DataArray."""
    return xr.DataArray(
        da.from_delayed(delayed_data, shape=shape, dtype=dtype).rechunk(
            chunks
        ),
        dims=('band', 'y', 'x'),
        coords=coords,
        attrs=attrs,
    )


def warp_open(
    filename,
    band_names=None,
    resampling='nearest',
    dtype=None,
    netcdf_vars=None,
    nodata=None,
    return_windows=False,
    warp_mem_limit=512,
    num_threads=1,
    tap=False,
    **kwargs,
):
    """Warps and opens a file.

    Args:
        filename (str): The file to open.
        band_names (Optional[int, str, or list]): The band names.
        resampling (Optional[str]): The resampling method.
        dtype (Optional[str]): A data type to force the output to. If not given, the data type is extracted
            from the file.
        netcdf_vars (Optional[list]): NetCDF variables to open as a band stack.
        nodata (Optional[float | int]): A 'no data' value to set. Default is ``None``.
        return_windows (Optional[bool]): Whether to return block windows.
        warp_mem_limit (Optional[int]): The memory limit (in MB) for the ``rasterio.vrt.WarpedVRT`` function.
        num_threads (Optional[int]): The number of warp worker threads.
        tap (Optional[bool]): Whether to target align pixels.
        kwargs (Optional[dict]): Keyword arguments passed to ``open_rasterio``.

    Returns:
        ``xarray.DataArray``
    """
    ref_kwargs = {
        'bounds': None,
        'crs': None,
        'res': None,
        'nodata': nodata,
        'warp_mem_limit': warp_mem_limit,
        'num_threads': num_threads,
        'tap': tap,
        'tac': None,
    }

    ref_kwargs_netcdf_stack = ref_kwargs.copy()
    ref_kwargs_netcdf_stack['bounds_by'] = 'union'
    del ref_kwargs_netcdf_stack['tap']

    ref_kwargs = _check_config_globals(filename, 'reference', ref_kwargs)
    filenames = None

    # Create a list of variables to open
    if filename.lower().startswith('netcdf:') and netcdf_vars:
        filenames = (f'{filename}:' + f',{filename}:'.join(netcdf_vars)).split(
            ','
        )

    if filenames:
        ref_kwargs_netcdf_stack = _check_config_globals(
            filenames[0], 'reference', ref_kwargs_netcdf_stack
        )
        with rio_open(filenames[0]) as src:
            tags = src.tags()

    else:
        ref_kwargs_netcdf_stack = _check_config_globals(
            filename, 'reference', ref_kwargs_netcdf_stack
        )

        with rio_open(filename) as src:
            tags = src.tags()

    @contextlib.contextmanager
    def warp_netcdf_vars():
        # Warp all images to the same grid.
        warped_objects = warp_images(
            filenames, resampling=resampling, **ref_kwargs_netcdf_stack
        )

        yield xr.concat(
            (
                open_rasterio(
                    wobj, nodata=ref_kwargs['nodata'], **kwargs
                ).assign_coords(
                    band=[band_names[wi]] if band_names else [netcdf_vars[wi]]
                )
                for wi, wobj in enumerate(warped_objects)
            ),
            dim='band',
        )

    with open_rasterio(
        warp(filename, resampling=resampling, **ref_kwargs),
        nodata=ref_kwargs['nodata'],
        **kwargs,
    ) if not filenames else warp_netcdf_vars() as src:
        if band_names:
            if len(band_names) > src.gw.nbands:
                src.coords['band'] = band_names[: src.gw.nbands]
            else:
                src.coords['band'] = band_names

        else:
            if src.gw.sensor:
                if src.gw.sensor not in src.gw.avail_sensors:
                    if not src.gw.config['ignore_warnings']:
                        logger.warning(
                            '  The {} sensor is not currently supported.\nChoose from [{}].'.format(
                                src.gw.sensor, ', '.join(src.gw.avail_sensors)
                            )
                        )

                else:
                    new_band_names = list(
                        src.gw.wavelengths[src.gw.sensor]._fields
                    )
                    # Avoid nested opens within a `config` context
                    if len(new_band_names) != len(src.band.values.tolist()):
                        if not src.gw.config['ignore_warnings']:
                            logger.warning(
                                '  The new bands, {}, do not match the sensor bands, {}.'.format(
                                    new_band_names, src.band.values.tolist()
                                )
                            )

                    else:
                        src = src.assign_coords(**{'band': new_band_names})
                        src = src.assign_attrs(
                            **{'sensor': src.gw.sensor_names[src.gw.sensor]}
                        )

        if return_windows:
            if isinstance(kwargs['chunks'], tuple):
                chunksize = kwargs['chunks'][-1]
            else:
                chunksize = kwargs['chunks']

            src.attrs['block_windows'] = get_window_offsets(
                src.shape[-2],
                src.shape[-1],
                chunksize,
                chunksize,
                return_as='list',
            )

        src = src.assign_attrs(
            **{'filename': filename, 'resampling': resampling}
        )

        if tags:
            attrs = src.attrs.copy()
            attrs.update(tags)
            src = src.assign_attrs(**attrs)

        if dtype:

            attrs = src.attrs.copy()
            return src.astype(dtype).assign_attrs(**attrs)

        else:
            return src


def mosaic(
    filenames,
    overlap='max',
    bounds_by='reference',
    resampling='nearest',
    band_names=None,
    nodata=None,
    dtype=None,
    warp_mem_limit=512,
    num_threads=1,
    **kwargs,
):
    """Mosaics a list of images.

    Args:
        filenames (list): A list of file names to mosaic.
        overlap (Optional[str]): The keyword that determines how to handle overlapping data.
            Choices are ['min', 'max', 'mean'].
        bounds_by (Optional[str]): How to concatenate the output extent. Choices are ['intersection', 'union', 'reference'].

            * reference: Use the bounds of the reference image
            * intersection: Use the intersection (i.e., minimum extent) of all the image bounds
            * union: Use the union (i.e., maximum extent) of all the image bounds

        resampling (Optional[str]): The resampling method.
        band_names (Optional[1d array-like]): A list of names to give the band dimension.
        nodata (Optional[float | int]): A 'no data' value to set. Default is ``None``.
        dtype (Optional[str]): A data type to force the output to. If not given, the data type is extracted
            from the file.
        warp_mem_limit (Optional[int]): The memory limit (in MB) for the ``rasterio.vrt.WarpedVRT`` function.
        num_threads (Optional[int]): The number of warp worker threads.
        kwargs (Optional[dict]): Keyword arguments passed to ``open_rasterio``.

    Returns:
        ``xarray.DataArray``
    """
    if overlap not in ['min', 'max', 'mean']:
        logger.exception(
            "  The overlap argument must be one of ['min', 'max', 'mean']."
        )

    ref_kwargs = {
        'bounds': None,
        'crs': None,
        'res': None,
        'nodata': nodata,
        'warp_mem_limit': warp_mem_limit,
        'num_threads': num_threads,
        'tac': None,
    }

    ref_kwargs = _check_config_globals(filenames, bounds_by, ref_kwargs)

    # Warp all images to the same grid.
    warped_objects = warp_images(
        filenames, bounds_by=bounds_by, resampling=resampling, **ref_kwargs
    )

    geometries = []

    with rio_open(filenames[0]) as src_:
        tags = src_.tags()

    # Combine the data
    with open_rasterio(
        warped_objects[0], nodata=ref_kwargs['nodata'], **kwargs
    ) as darray:
        attrs = darray.attrs.copy()

        # Get the original bounds, unsampled
        with open_rasterio(
            filenames[0], nodata=ref_kwargs['nodata'], **kwargs
        ) as src_:
            geometries.append(src_.gw.geometry)

        for fidx, fn in enumerate(warped_objects[1:]):
            with open_rasterio(
                fn, nodata=ref_kwargs['nodata'], **kwargs
            ) as darrayb:
                with open_rasterio(
                    filenames[fidx + 1], nodata=ref_kwargs['nodata'], **kwargs
                ) as src_:
                    geometries.append(src_.gw.geometry)
                src_ = None

                if overlap == 'min':
                    if isinstance(ref_kwargs['nodata'], float) or isinstance(
                        ref_kwargs['nodata'], int
                    ):
                        darray = xr.where(
                            (darray.mean(dim='band') == ref_kwargs['nodata'])
                            & (
                                darrayb.mean(dim='band')
                                != ref_kwargs['nodata']
                            ),
                            darrayb,
                            xr.where(
                                (
                                    darray.mean(dim='band')
                                    != ref_kwargs['nodata']
                                )
                                & (
                                    darrayb.mean(dim='band')
                                    == ref_kwargs['nodata']
                                ),
                                darray,
                                np.minimum(darray, darrayb),
                            ),
                        )

                    else:
                        darray = np.minimum(darray, darrayb)

                elif overlap == 'max':
                    if isinstance(ref_kwargs['nodata'], float) or isinstance(
                        ref_kwargs['nodata'], int
                    ):
                        darray = xr.where(
                            (darray.mean(dim='band') == ref_kwargs['nodata'])
                            & (
                                darrayb.mean(dim='band')
                                != ref_kwargs['nodata']
                            ),
                            darrayb,
                            xr.where(
                                (
                                    darray.mean(dim='band')
                                    != ref_kwargs['nodata']
                                )
                                & (
                                    darrayb.mean(dim='band')
                                    == ref_kwargs['nodata']
                                ),
                                darray,
                                np.maximum(darray, darrayb),
                            ),
                        )

                    else:
                        darray = np.maximum(darray, darrayb)

                elif overlap == 'mean':
                    if isinstance(ref_kwargs['nodata'], float) or isinstance(
                        ref_kwargs['nodata'], int
                    ):

                        darray = xr.where(
                            (darray.mean(dim='band') == ref_kwargs['nodata'])
                            & (
                                darrayb.mean(dim='band')
                                != ref_kwargs['nodata']
                            ),
                            darrayb,
                            xr.where(
                                (
                                    darray.mean(dim='band')
                                    != ref_kwargs['nodata']
                                )
                                & (
                                    darrayb.mean(dim='band')
                                    == ref_kwargs['nodata']
                                ),
                                darray,
                                (darray + darrayb) / 2.0,
                            ),
                        )

                    else:
                        darray = (darray + darrayb) / 2.0

        darray = darray.assign_attrs(**attrs)

        if band_names:
            darray.coords['band'] = band_names
        else:

            if darray.gw.sensor:

                if darray.gw.sensor not in darray.gw.avail_sensors:

                    if not darray.gw.config['ignore_warnings']:

                        logger.warning(
                            '  The {} sensor is not currently supported.\nChoose from [{}].'.format(
                                darray.gw.sensor,
                                ', '.join(darray.gw.avail_sensors),
                            )
                        )

                else:

                    new_band_names = list(
                        darray.gw.wavelengths[darray.gw.sensor]._fields
                    )

                    if len(new_band_names) != len(darray.band.values.tolist()):

                        if not darray.gw.config['ignore_warnings']:
                            logger.warning(
                                '  The band list length does not match the sensor bands.'
                            )

                    else:
                        darray = darray.assign_coords(
                            **{'band': new_band_names}
                        )
                        darray = darray.assign_attrs(
                            **{
                                'sensor': darray.gw.sensor_names[
                                    darray.gw.sensor
                                ]
                            }
                        )

        darray = darray.assign_attrs(
            **{'resampling': resampling, 'geometries': geometries}
        )

        if tags:
            attrs = darray.attrs.copy()
            attrs.update(tags)
            darray = darray.assign_attrs(**attrs)

        if dtype:
            attrs = darray.attrs.copy()

            return darray.astype(dtype).assign_attrs(**attrs)

        else:
            return darray


def concat(
    filenames,
    stack_dim='time',
    bounds_by='reference',
    resampling='nearest',
    time_names=None,
    band_names=None,
    nodata=None,
    dtype=None,
    netcdf_vars=None,
    overlap='max',
    warp_mem_limit=512,
    num_threads=1,
    tap=False,
    **kwargs,
):
    """Concatenates a list of images.

    Args:
        filenames (list): A list of file names to concatenate.
        stack_dim (Optional[str]): The stack dimension. Choices are ['time', 'band'].
        bounds_by (Optional[str]): How to concatenate the output extent. Choices are ['intersection', 'union', 'reference'].

            * reference: Use the bounds of the reference image
            * intersection: Use the intersection (i.e., minimum extent) of all the image bounds
            * union: Use the union (i.e., maximum extent) of all the image bounds

        resampling (Optional[str]): The resampling method.
        time_names (Optional[1d array-like]): A list of names to give the time dimension.
        band_names (Optional[1d array-like]): A list of names to give the band dimension.
        nodata (Optional[float | int]): A 'no data' value to set. Default is ``None``.
        dtype (Optional[str]): A data type to force the output to. If not given, the data type is extracted
            from the file.
        netcdf_vars (Optional[list]): NetCDF variables to open as a band stack.
        overlap (Optional[str]): The keyword that determines how to handle overlapping data.
            Choices are ['min', 'max', 'mean']. Only used when mosaicking arrays from the same timeframe.
        warp_mem_limit (Optional[int]): The memory limit (in MB) for the ``rasterio.vrt.WarpedVRT`` function.
        num_threads (Optional[int]): The number of warp worker threads.
        tap (Optional[bool]): Whether to target align pixels.
        kwargs (Optional[dict]): Keyword arguments passed to ``open_rasterio``.

    Returns:
        ``xarray.DataArray``
    """
    if stack_dim.lower() not in ['band', 'time']:
        logger.exception("  The stack dimension should be 'band' or 'time'.")

    with rio_open(filenames[0]) as src_:
        tags = src_.tags()

    src_ = warp_open(
        f'{filenames[0]}:{netcdf_vars[0]}' if netcdf_vars else filenames[0],
        resampling=resampling,
        band_names=[netcdf_vars[0]] if netcdf_vars else band_names,
        nodata=nodata,
        warp_mem_limit=warp_mem_limit,
        num_threads=num_threads,
        **kwargs,
    )

    attrs = src_.attrs.copy()
    src_.close()
    src_ = None

    if time_names and not (str(filenames[0]).lower().startswith('netcdf:')):

        concat_list = []
        new_time_names = []

        # Check the time names for duplicates
        for tidx in range(0, len(time_names)):

            if list(time_names).count(time_names[tidx]) > 1:

                if time_names[tidx] not in new_time_names:

                    # Get the file names to mosaic
                    filenames_mosaic = [
                        filenames[i]
                        for i in range(0, len(time_names))
                        if time_names[i] == time_names[tidx]
                    ]

                    # Mosaic the images into a single-date array
                    concat_list.append(
                        mosaic(
                            filenames_mosaic,
                            overlap=overlap,
                            bounds_by=bounds_by,
                            resampling=resampling,
                            band_names=band_names,
                            nodata=nodata,
                            warp_mem_limit=warp_mem_limit,
                            num_threads=num_threads,
                            **kwargs,
                        )
                    )

                    new_time_names.append(time_names[tidx])

            else:

                new_time_names.append(time_names[tidx])

                # Warp the date
                concat_list.append(
                    warp_open(
                        filenames[tidx],
                        resampling=resampling,
                        band_names=band_names,
                        nodata=nodata,
                        warp_mem_limit=warp_mem_limit,
                        num_threads=num_threads,
                        **kwargs,
                    )
                )

        # Warp all images and concatenate along the 'time' axis into a DataArray
        src = xr.concat(concat_list, dim=stack_dim.lower()).assign_coords(
            time=new_time_names
        )

    else:
        warp_list = [
            warp_open(
                fn,
                resampling=resampling,
                band_names=band_names,
                netcdf_vars=netcdf_vars,
                nodata=nodata,
                warp_mem_limit=warp_mem_limit,
                num_threads=num_threads,
                **kwargs,
            )
            for fn in filenames
        ]
        # Check dimensions
        try:
            for fidx in range(0, len(warp_list) - 1):
                xr.align(warp_list[fidx], warp_list[fidx + 1], join='exact')
        except ValueError:
            if not warp_list[0].gw.config['ignore_warnings']:
                warning_message = 'The stacked dimensions are not aligned. If this was not intentional, use gw.config.update to align coordinates.'
                warnings.warn(warning_message, UserWarning)
                logger.warning(warning_message)

        src = xr.concat(warp_list, dim=stack_dim.lower())

    src = src.assign_attrs(**{'filename': [Path(fn).name for fn in filenames]})

    if tags:
        attrs = src.attrs.copy()
        attrs.update(tags)

    src = src.assign_attrs(**attrs)

    if stack_dim == 'time':

        if str(filenames[0]).lower().startswith('netcdf:'):

            if time_names:
                src.coords['time'] = time_names
            else:
                src.coords['time'] = parse_filename_dates(filenames)

            try:
                src = src.groupby('time').max().assign_attrs(**attrs)
            except ValueError:
                pass

        else:
            if not time_names:
                src.coords['time'] = parse_filename_dates(filenames)

    if band_names:
        src.coords['band'] = band_names
    else:
        if src.gw.sensor:
            if src.gw.sensor not in src.gw.avail_sensors:
                if not src.gw.config['ignore_warnings']:
                    logger.warning(
                        '  The {} sensor is not currently supported.\nChoose from [{}].'.format(
                            src.gw.sensor, ', '.join(src.gw.avail_sensors)
                        )
                    )

            else:
                new_band_names = list(
                    src.gw.wavelengths[src.gw.sensor]._fields
                )
                if len(new_band_names) != len(src.band.values.tolist()):
                    if not src.gw.config['ignore_warnings']:
                        logger.warning(
                            '  The new bands, {}, do not match the sensor bands, {}.'.format(
                                new_band_names, src.band.values.tolist()
                            )
                        )

                else:
                    src = src.assign_coords(**{'band': new_band_names})
                    src = src.assign_attrs(
                        **{'sensor': src.gw.sensor_names[src.gw.sensor]}
                    )

    if dtype:
        attrs = src.attrs.copy()
        return src.astype(dtype).assign_attrs(**attrs)

    else:
        return src


def transform_crs(
    data_src,
    dst_crs=None,
    dst_res=None,
    dst_width=None,
    dst_height=None,
    dst_bounds=None,
    src_nodata=None,
    dst_nodata=None,
    coords_only=False,
    resampling='nearest',
    warp_mem_limit=512,
    num_threads=1,
):
    """Transforms a DataArray to a new coordinate reference system.

    Args:
        data_src (DataArray): The data to transform.
        dst_crs (Optional[CRS | int | dict | str]): The destination CRS.
        dst_res (Optional[tuple]): The destination resolution.
        dst_width (Optional[int]): The destination width. Cannot be used with ``dst_res``.
        dst_height (Optional[int]): The destination height. Cannot be used with ``dst_res``.
        dst_bounds (Optional[BoundingBox | tuple]): The destination bounds, as a ``rasterio.coords.BoundingBox``
            or as a tuple of (left, bottom, right, top).
        src_nodata (Optional[int | float]): The source nodata value. Pixels with this value will not be used for
            interpolation. If not set, it will default to the nodata value of the source image if a masked ndarray
            or rasterio band, if available.
        dst_nodata (Optional[int | float]): The nodata value used to initialize the destination; it will remain in
            all areas not covered by the reprojected source. Defaults to the nodata value of the destination
            image (if set), the value of src_nodata, or 0 (GDAL default).
        coords_only (Optional[bool]): Whether to return transformed coordinates. If ``coords_only`` = ``True`` then
            the array is not warped and the size is unchanged. It also avoids in-memory computations.
        resampling (Optional[str]): The resampling method if ``filename`` is a ``list``.
            Choices are ['average', 'bilinear', 'cubic', 'cubic_spline', 'gauss', 'lanczos', 'max', 'med', 'min', 'mode', 'nearest'].
        warp_mem_limit (Optional[int]): The warp memory limit.
        num_threads (Optional[int]): The number of parallel threads.

    Returns:
        ``xarray.DataArray``
    """
    # Transform the input
    data_dict = rio_transform_crs(
        data_src,
        dst_crs=dst_crs,
        dst_res=dst_res,
        dst_width=dst_width,
        dst_height=dst_height,
        dst_bounds=dst_bounds,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
        coords_only=coords_only,
        resampling=resampling,
        warp_mem_limit=warp_mem_limit,
        num_threads=num_threads,
        return_as_dict=True,
        delayed_array=True,
    )
    dst_transform = data_dict['transform']
    dst_crs = data_dict['crs']
    dst_height = data_dict['height']
    dst_width = data_dict['width']

    # Get the transformed coordinates
    left = dst_transform[2]
    cellx = abs(dst_transform[0])
    x = np.arange(
        left + cellx / 2.0, left + cellx / 2.0 + (cellx * dst_width), cellx
    )[:dst_width]
    top = dst_transform[5]
    celly = abs(dst_transform[4])
    y = np.arange(
        top - celly / 2.0, top - celly / 2.0 - (celly * dst_height), -celly
    )[:dst_height]

    attrs = data_src.attrs.copy()

    if coords_only:
        attrs.update(
            {
                'crs': dst_crs,
                'transform': dst_transform[:6],
                'res': (cellx, celly),
            }
        )

        return data_src.assign_coords(x=x, y=y).assign_attrs(**attrs)

    else:
        # The transformed array is a dask Delayed object
        data_dst = data_dict['array']

        if not dst_res:
            dst_res = (abs(x[1] - x[0]), abs(y[0] - y[1]))

        attrs.update(
            {
                'transform': tuple(dst_transform)[:6],
                'crs': dst_crs,
                'res': dst_res,
                'resampling': resampling,
            }
        )
        # Get the DataArray from the delayed object
        data_dst = delayed_to_xarray(
            data_dst,
            shape=(data_src.gw.nbands, dst_height, dst_width),
            dtype=data_src.dtype,
            chunks=data_src.data.chunksize,
            coords={'band': data_src.band.values.tolist(), 'y': y, 'x': x},
            attrs=attrs,
        )

        return data_dst
