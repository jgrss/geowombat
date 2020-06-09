import os

from ..core.windows import get_window_offsets
from ..core.util import parse_filename_dates
from ..errors import logger
from ..config import config
from .rasterio_ import get_ref_image_meta, warp, warp_images, get_file_bounds, window_to_bounds, unpack_bounding_box, unpack_window
from .rasterio_ import transform_crs as rio_transform_crs

import numpy as np
from rasterio.windows import Window
from rasterio.coords import BoundingBox
import dask.array as da
import xarray as xr
from xarray.ufuncs import maximum as xr_maximum
from xarray.ufuncs import minimum as xr_mininum
from deprecated import deprecated


def _update_kwarg(ref_obj, ref_kwargs, key):

    """
    Updates keyword arguments for global config parameters

    Args:
        ref_obj (str or object)
        ref_kwargs (dict)
        key (str)

    Returns:
        ``dict``
    """

    if isinstance(ref_obj, str) and os.path.isfile(ref_obj):

        # Get the metadata from the reference image
        ref_meta = get_ref_image_meta(ref_obj)
        ref_kwargs[key] = getattr(ref_meta, key)

    else:

        if ref_obj:
            ref_kwargs[key] = ref_obj

    return ref_kwargs


def _get_raster_coords(filename):

    with xr.open_rasterio(filename) as src:

        x = src.x.values - src.res[0] / 2.0
        y = src.y.values + src.res[1] / 2.0

    return x, y


def _check_config_globals(filenames, bounds_by, ref_kwargs):

    """
    Checks global configuration parameters

    Args:
        filenames (str or str list)
        bounds_by (str)
        ref_kwargs (dict)
    """

    # Check if there is a reference image
    if config['ref_image']:

        if isinstance(config['ref_image'], str) and os.path.isfile(config['ref_image']):

            # Get the metadata from the reference image
            ref_meta = get_ref_image_meta(config['ref_image'])

            ref_kwargs['bounds'] = ref_meta.bounds
            ref_kwargs['crs'] = ref_meta.crs
            ref_kwargs['res'] = ref_meta.res

        else:
            logger.warning('  The reference image does not exist')

    else:

        if config['ref_bounds']:

            if isinstance(config['ref_bounds'], str) and config['ref_bounds'].startswith('Window'):
                ref_bounds_ = window_to_bounds(filenames, unpack_window(config['ref_bounds']))
            elif isinstance(config['ref_bounds'], str) and config['ref_bounds'].startswith('BoundingBox'):
                ref_bounds_ = unpack_bounding_box(config['ref_bounds'])
            elif isinstance(config['ref_bounds'], Window):
                ref_bounds_ = window_to_bounds(filenames, config['ref_bounds'])
            elif isinstance(config['ref_bounds'], BoundingBox):

                ref_bounds_ = (config['ref_bounds'].left,
                               config['ref_bounds'].bottom,
                               config['ref_bounds'].right,
                               config['ref_bounds'].top)

            else:
                ref_bounds_ = config['ref_bounds']

            ref_kwargs = _update_kwarg(ref_bounds_, ref_kwargs, 'bounds')

        else:

            if isinstance(filenames, str):

                # Use the bounds of the image
                ref_kwargs['bounds'] = get_file_bounds([filenames],
                                                       bounds_by='reference',
                                                       crs=ref_kwargs['crs'],
                                                       res=ref_kwargs['res'],
                                                       return_bounds=True)

            else:

                # Replace the bounds keyword, if needed
                if bounds_by.lower() == 'intersection':

                    # Get the intersecting bounds of all images
                    ref_kwargs['bounds'] = get_file_bounds(filenames,
                                                           bounds_by='intersection',
                                                           crs=ref_kwargs['crs'],
                                                           res=ref_kwargs['res'],
                                                           return_bounds=True)

                elif bounds_by.lower() == 'union':

                    # Get the union bounds of all images
                    ref_kwargs['bounds'] = get_file_bounds(filenames,
                                                           bounds_by='union',
                                                           crs=ref_kwargs['crs'],
                                                           res=ref_kwargs['res'],
                                                           return_bounds=True)

                elif bounds_by.lower() == 'reference':

                    # Use the bounds of the first image
                    ref_kwargs['bounds'] = get_file_bounds(filenames,
                                                           bounds_by='reference',
                                                           crs=ref_kwargs['crs'],
                                                           res=ref_kwargs['res'],
                                                           return_bounds=True)

                else:
                    logger.exception("  Choose from 'intersection', 'union', or 'reference'.")

                config['ref_bounds'] = ref_kwargs['bounds']

        if config['ref_crs']:
            ref_kwargs = _update_kwarg(config['ref_crs'], ref_kwargs, 'crs')

        if config['ref_res']:
            ref_kwargs = _update_kwarg(config['ref_res'], ref_kwargs, 'res')

        if config['ref_tar']:

            if isinstance(config['ref_tar'], str):

                if os.path.isfile(config['ref_tar']):
                    ref_kwargs = _update_kwarg(_get_raster_coords(config['ref_tar']), ref_kwargs, 'tac')
                else:
                    logger.warning('  The target aligned raster does not exist.')

            else:
                logger.warning('  The target aligned raster must be an image.')

    return ref_kwargs


def warp_open(filename,
              band_names=None,
              nodata=None,
              resampling='nearest',
              dtype=None,
              return_windows=False,
              warp_mem_limit=512,
              num_threads=1,
              tap=False,
              **kwargs):

    """
    Warps and opens a file

    Args:
        filename (str): The file to open.
        band_names (Optional[int, str, or list]): The band names.
        nodata (Optional[float | int]): A 'no data' value to set. Default is None.
        resampling (Optional[str]): The resampling method.
        dtype (Optional[str]): A data type to force the output to. If not given, the data type is extracted
            from the file.
        return_windows (Optional[bool]): Whether to return block windows.
        warp_mem_limit (Optional[int]): The memory limit (in MB) for the ``rasterio.vrt.WarpedVRT`` function.
        num_threads (Optional[int]): The number of warp worker threads.
        tap (Optional[bool]): Whether to target align pixels.
        kwargs (Optional[dict]): Keyword arguments passed to ``xarray.open_rasterio``.

    Returns:
        ``xarray.DataArray``
    """

    ref_kwargs = {'bounds': None,
                  'crs': None,
                  'res': None,
                  'nodata': nodata,
                  'warp_mem_limit': warp_mem_limit,
                  'num_threads': num_threads,
                  'tap': tap,
                  'tac': None}

    ref_kwargs = _check_config_globals(filename, 'reference', ref_kwargs)

    with xr.open_rasterio(warp(filename,
                               resampling=resampling,
                               **ref_kwargs),
                          **kwargs) as src:

        if band_names:
            src.coords['band'] = band_names
        else:

            if src.gw.sensor:

                if src.gw.sensor not in src.gw.avail_sensors:

                    logger.warning('  The {} sensor is not currently supported.\nChoose from [{}].'.format(src.gw.sensor,
                                                                                                           ', '.join(src.gw.avail_sensors)))

                else:

                    new_band_names = list(src.gw.wavelengths[src.gw.sensor]._fields)

                    # Avoid nested opens within a `config` context
                    if len(new_band_names) != len(src.band.values.tolist()):

                        logger.warning('  The new bands, {}, do not match the sensor bands, {}.'.format(new_band_names,
                                                                                                        src.band.values.tolist()))

                    else:

                        src.coords['band'] = new_band_names
                        src.attrs['sensor'] = src.gw.sensor_names[src.gw.sensor]

        if return_windows:

            if isinstance(kwargs['chunks'], tuple):
                chunksize = kwargs['chunks'][-1]
            else:
                chunksize = kwargs['chunks']

            src.attrs['block_windows'] = get_window_offsets(src.shape[-2],
                                                            src.shape[-1],
                                                            chunksize,
                                                            chunksize,
                                                            return_as='list')

        src.attrs['filename'] = filename
        src.attrs['resampling'] = resampling

        if dtype:

            attrs = src.attrs.copy()
            return src.astype(dtype).assign_attrs(**attrs)

        else:
            return src


def mosaic(filenames,
           overlap='max',
           bounds_by='reference',
           resampling='nearest',
           band_names=None,
           nodata=None,
           dtype=None,
           warp_mem_limit=512,
           num_threads=1,
           **kwargs):

    """
    Mosaics a list of images

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
        nodata (Optional[float | int]): A 'no data' value to set. Default is None.
        dtype (Optional[str]): A data type to force the output to. If not given, the data type is extracted
            from the file.
        warp_mem_limit (Optional[int]): The memory limit (in MB) for the ``rasterio.vrt.WarpedVRT`` function.
        num_threads (Optional[int]): The number of warp worker threads.
        kwargs (Optional[dict]): Keyword arguments passed to ``xarray.open_rasterio``.

    Returns:
        ``xarray.DataArray``
    """

    if overlap not in ['min', 'max', 'mean']:
        logger.exception("  The overlap argument must be one of ['min', 'max', 'mean'].")

    ref_kwargs = {'bounds': None,
                  'crs': None,
                  'res': None,
                  'nodata': nodata,
                  'warp_mem_limit': warp_mem_limit,
                  'num_threads': num_threads,
                  'tac': None}

    ref_kwargs = _check_config_globals(filenames, bounds_by, ref_kwargs)

    # Warp all images to the same grid.
    warped_objects = warp_images(filenames,
                                 bounds_by=bounds_by,
                                 resampling=resampling,
                                 **ref_kwargs)

    # Combine the data
    with xr.open_rasterio(warped_objects[0], **kwargs) as ds:

        attrs = ds.attrs.copy()

        for fn in warped_objects[1:]:

            with xr.open_rasterio(fn, **kwargs) as dsb:

                if overlap == 'min':

                    if isinstance(nodata, float) or isinstance(nodata, int):
                        ds = xr.where((ds == nodata) | (dsb == nodata), nodata, xr_mininum(ds, dsb))
                    else:
                        ds = xr_mininum(ds, dsb)

                elif overlap == 'max':

                    if isinstance(nodata, float) or isinstance(nodata, int):
                        ds = xr.where((ds == nodata) | (dsb == nodata), nodata, xr_maximum(ds, dsb))
                    else:
                        ds = xr_maximum(ds, dsb)

                elif overlap == 'mean':

                    if isinstance(nodata, float) or isinstance(nodata, int):
                        ds = xr.where((ds == nodata) | (dsb == nodata), nodata, (ds + dsb) / 2.0)
                    else:
                        ds = (ds + dsb) / 2.0

                # ds = ds.combine_first(dsb)

        ds = ds.assign_attrs(**attrs)

        if band_names:
            ds.coords['band'] = band_names
        else:

            if ds.gw.sensor:

                if ds.gw.sensor not in ds.gw.avail_sensors:

                    logger.warning('  The {} sensor is not currently supported.\nChoose from [{}].'.format(ds.gw.sensor,
                                                                                                           ', '.join(ds.gw.avail_sensors)))

                else:

                    new_band_names = list(ds.gw.wavelengths[ds.gw.sensor]._fields)

                    if len(new_band_names) != len(ds.band.values.tolist()):
                        logger.warning('  The band list length does not match the sensor bands.')
                    else:

                        ds.coords['band'] = new_band_names
                        ds.attrs['sensor'] = ds.gw.sensor_names[ds.gw.sensor]

        ds.attrs['resampling'] = resampling

        if dtype:

            attrs = ds.attrs.copy()
            return ds.astype(dtype).assign_attrs(**attrs)

        else:
            return ds


def concat(filenames,
           stack_dim='time',
           bounds_by='reference',
           resampling='nearest',
           time_names=None,
           band_names=None,
           nodata=None,
           dtype=None,
           overlap='max',
           warp_mem_limit=512,
           num_threads=1,
           tap=False,
           **kwargs):

    """
    Concatenates a list of images

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
        nodata (Optional[float | int]): A 'no data' value to set. Default is None.
        dtype (Optional[str]): A data type to force the output to. If not given, the data type is extracted
            from the file.
        overlap (Optional[str]): The keyword that determines how to handle overlapping data.
            Choices are ['min', 'max', 'mean']. Only used when mosaicking arrays from the same timeframe.
        warp_mem_limit (Optional[int]): The memory limit (in MB) for the ``rasterio.vrt.WarpedVRT`` function.
        num_threads (Optional[int]): The number of warp worker threads.
        tap (Optional[bool]): Whether to target align pixels.
        kwargs (Optional[dict]): Keyword arguments passed to ``xarray.open_rasterio``.

    Returns:
        ``xarray.DataArray``
    """

    if stack_dim.lower() not in ['band', 'time']:
        logger.exception("  The stack dimension should be 'band' or 'time'")

    ref_kwargs = {'bounds': None,
                  'crs': None,
                  'res': None,
                  'nodata': nodata,
                  'warp_mem_limit': warp_mem_limit,
                  'num_threads': num_threads,
                  'tap': tap,
                  'tac': None}

    ref_kwargs = _check_config_globals(filenames, bounds_by, ref_kwargs)

    # Keep a copy of the transformed attributes.
    with xr.open_rasterio(warp(filenames[0],
                               resampling=resampling,
                               **ref_kwargs), **kwargs) as ds_:

        attrs = ds_.attrs.copy()

    if time_names:

        concat_list = list()
        new_time_names = list()

        # Check the time names for duplicates
        for tidx in range(0, len(time_names)):

            if list(time_names).count(time_names[tidx]) > 1:

                if time_names[tidx] not in new_time_names:

                    # Get the file names to mosaic
                    filenames_mosaic = [filenames[i] for i in range(0, len(time_names))
                                        if time_names[i] == time_names[tidx]]

                    # Mosaic the images into a single-date array
                    concat_list.append(mosaic(filenames_mosaic,
                                              overlap=overlap,
                                              bounds_by=bounds_by,
                                              resampling=resampling,
                                              band_names=band_names,
                                              nodata=nodata,
                                              warp_mem_limit=warp_mem_limit,
                                              num_threads=num_threads,
                                              **kwargs))

                    new_time_names.append(time_names[tidx])

            else:

                new_time_names.append(time_names[tidx])

                # Warp the date
                concat_list.append(warp_open(filenames[tidx],
                                             resampling=resampling,
                                             band_names=band_names,
                                             nodata=nodata,
                                             warp_mem_limit=warp_mem_limit,
                                             num_threads=num_threads,
                                             **kwargs))

        # Warp all images and concatenate along the 'time' axis into a DataArray
        output = xr.concat(concat_list, dim=stack_dim.lower())

        # Assign the new time band names
        ds = output.assign_coords(time=new_time_names)

    else:

        # Warp all images and concatenate along
        #   the 'time' axis into a DataArray.
        ds = xr.concat([xr.open_rasterio(warp(fn,
                                              resampling=resampling,
                                              **ref_kwargs), **kwargs)
                        for fn in filenames], dim=stack_dim.lower())

    ds = ds.assign_attrs(**attrs)

    if not time_names and (stack_dim == 'time'):
        ds.coords['time'] = parse_filename_dates(filenames)

    if band_names:
        ds.coords['band'] = band_names
    else:

        if ds.gw.sensor:

            if ds.gw.sensor not in ds.gw.avail_sensors:

                logger.warning('  The {} sensor is not currently supported.\nChoose from [{}].'.format(ds.gw.sensor,
                                                                                                       ', '.join(ds.gw.avail_sensors)))

            else:

                new_band_names = list(ds.gw.wavelengths[ds.gw.sensor]._fields)

                if len(new_band_names) != len(ds.band.values.tolist()):

                    logger.warning('  The new bands, {}, do not match the sensor bands, {}.'.format(new_band_names,
                                                                                                    ds.band.values.tolist()))

                else:

                    ds.coords['band'] = new_band_names
                    ds.attrs['sensor'] = ds.gw.sensor_names[ds.gw.sensor]

    if dtype:
        
        attrs = ds.attrs.copy()
        return ds.astype(dtype).assign_attrs(**attrs)

    else:
        return ds


def transform_crs(data_src,
                  dst_crs=None,
                  dst_res=None,
                  dst_width=None,
                  dst_height=None,
                  dst_bounds=None,
                  resampling='nearest',
                  warp_mem_limit=512,
                  num_threads=1):

    """
    Transforms a DataArray to a new coordinate reference system

    Args:
        data_src (DataArray): The data to transform.
        dst_crs (Optional[CRS | int | dict | str]): The destination CRS.
        dst_res (Optional[tuple]): The destination resolution.
        dst_width (Optional[int]): The destination width. Cannot be used with ``dst_res``.
        dst_height (Optional[int]): The destination height. Cannot be used with ``dst_res``.
        dst_bounds (Optional[BoundingBox | tuple]): The destination bounds, as a ``rasterio.coords.BoundingBox``
            or as a tuple of (left, bottom, right, top).
        resampling (Optional[str]): The resampling method if ``filename`` is a ``list``.
            Choices are ['average', 'bilinear', 'cubic', 'cubic_spline', 'gauss', 'lanczos', 'max', 'med', 'min', 'mode', 'nearest'].
        warp_mem_limit (Optional[int]): The warp memory limit.
        num_threads (Optional[int]): The number of parallel threads.

    Returns:
        ``xarray.DataArray``
    """

    data_dst, dst_transform, dst_crs = rio_transform_crs(data_src,
                                                         dst_crs=dst_crs,
                                                         dst_res=dst_res,
                                                         dst_width=dst_width,
                                                         dst_height=dst_height,
                                                         dst_bounds=dst_bounds,
                                                         resampling=resampling,
                                                         warp_mem_limit=warp_mem_limit,
                                                         num_threads=num_threads)

    nrows, ncols = data_dst.shape[-2], data_dst.shape[-1]

    left = dst_transform[2]
    cellx = abs(dst_transform[0])
    x = np.arange(left + cellx / 2.0, left + cellx / 2.0 + (cellx * ncols), cellx)

    top = dst_transform[5]
    celly = abs(dst_transform[4])
    y = np.arange(top - celly / 2.0, top - celly / 2.0 - (celly * nrows), -celly)

    if not dst_res:
        dst_res = (abs(x[1] - x[0]), abs(y[0] - y[1]))

    data_dst = xr.DataArray(data=da.from_array(data_dst,
                                               chunks=data_src.data.chunksize),
                            coords={'band': data_src.band.values.tolist(),
                                    'y': y,
                                    'x': x},
                            dims=('band', 'y', 'x'),
                            attrs=data_src.attrs)

    data_dst.attrs['transform'] = tuple(dst_transform)[:6]
    data_dst.attrs['crs'] = dst_crs
    data_dst.attrs['res'] = dst_res
    data_dst.attrs['resampling'] = resampling

    if 'sensor' in data_src.attrs:
        data_dst.attrs['sensor'] = data_src.attrs['sensor']

    if 'filename' in data_src.attrs:
        data_dst.attrs['filename'] = data_src.attrs['filename']

    return data_dst


@deprecated('Deprecated since 1.2.0. Use geowombat.transform_crs() instead.')
def to_crs(data_src,
           dst_crs=None,
           dst_res=None,
           dst_width=None,
           dst_height=None,
           dst_bounds=None,
           resampling='nearest',
           warp_mem_limit=512,
           num_threads=1):

    """
    .. deprecated:: 1.2.0
        Use :func:`geowombat.transform_crs()` instead.

    Transforms a DataArray to a new coordinate reference system

    Args:
        data_src (DataArray): The data to transform.
        dst_crs (Optional[CRS | int | dict | str]): The destination CRS.
        dst_res (Optional[tuple]): The destination resolution.
        dst_width (Optional[int]): The destination width. Cannot be used with ``dst_res``.
        dst_height (Optional[int]): The destination height. Cannot be used with ``dst_res``.
        dst_bounds (Optional[BoundingBox | tuple]): The destination bounds, as a ``rasterio.coords.BoundingBox``
            or as a tuple of (left, bottom, right, top).
        resampling (Optional[str]): The resampling method if ``filename`` is a ``list``.
            Choices are ['average', 'bilinear', 'cubic', 'cubic_spline', 'gauss', 'lanczos', 'max', 'med', 'min', 'mode', 'nearest'].
        warp_mem_limit (Optional[int]): The warp memory limit.
        num_threads (Optional[int]): The number of parallel threads.

    Returns:
        ``xarray.DataArray``
    """

    data_dst, dst_transform, dst_crs = rio_transform_crs(data_src,
                                                         dst_crs=dst_crs,
                                                         dst_res=dst_res,
                                                         dst_width=dst_width,
                                                         dst_height=dst_height,
                                                         dst_bounds=dst_bounds,
                                                         resampling=resampling,
                                                         warp_mem_limit=warp_mem_limit,
                                                         num_threads=num_threads)

    nrows, ncols = data_dst.shape[-2], data_dst.shape[-1]

    left = dst_transform[2]
    cellx = abs(dst_transform[0])
    x = np.arange(left + cellx / 2.0, left + cellx / 2.0 + (cellx * ncols), cellx)

    top = dst_transform[5]
    celly = abs(dst_transform[4])
    y = np.arange(top - celly / 2.0, top - celly / 2.0 - (celly * nrows), -celly)

    if not dst_res:
        dst_res = (abs(x[1] - x[0]), abs(y[0] - y[1]))

    data_dst = xr.DataArray(data=da.from_array(data_dst,
                                               chunks=data_src.data.chunksize),
                            coords={'band': data_src.band.values.tolist(),
                                    'y': y,
                                    'x': x},
                            dims=('band', 'y', 'x'),
                            attrs=data_src.attrs)

    data_dst.attrs['transform'] = tuple(dst_transform)[:6]
    data_dst.attrs['crs'] = dst_crs
    data_dst.attrs['res'] = dst_res
    data_dst.attrs['resampling'] = resampling

    if 'sensor' in data_src.attrs:
        data_dst.attrs['sensor'] = data_src.attrs['sensor']

    if 'filename' in data_src.attrs:
        data_dst.attrs['filename'] = data_src.attrs['filename']

    return data_dst
