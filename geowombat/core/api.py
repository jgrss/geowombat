# https://github.com/pydata/xarray/issues/2560
import netCDF4
import h5netcdf

from contextlib import contextmanager

from ..errors import logger

from . import geoxarray
from .conversion import xarray_to_xdataset
from .util import Chunks, get_file_extension
from .windows import from_bounds

import numpy as np
import xarray as xr
import rasterio as rio
from rasterio.windows import Window
import dask
import dask.array as da

ch = Chunks()

IO_DICT = dict(rasterio=['.tif', '.img'],
               xarray=['.nc'])


@dask.delayed
def read_delayed(fname, **kwargs):

    with rio.open(fname) as src:

        data_slice = src.read(**kwargs)

        single_band = True if len(data_slice.shape) == 2 else False

        yblock = src.block_shapes[0][0]
        xblock = src.block_shapes[0][1]

        if single_band:

            # Expand to 1 band and the z dimension
            data_slice = da.from_array(data_slice[np.newaxis, np.newaxis, :, :],
                                       chunks=(1, yblock, xblock))

        else:

            # Expand the z dimension
            data_slice = da.from_array(data_slice[np.newaxis, :, :, :],
                                       chunks=(1, 1, yblock, xblock))

        ycoords = np.linspace(src.bounds.top - (kwargs['window'].row_off * src.res[0]),
                              src.bounds.top - (kwargs['window'].row_off * src.res[0]) - (kwargs['window'].height * src.res[0]), kwargs['window'].height)

        xcoords = np.linspace(src.bounds.left + (kwargs['window'].col_off * src.res[0]),
                              src.bounds.left + (kwargs['window'].col_off * src.res[0]) + (kwargs['window'].width * src.res[0]), kwargs['window'].width)

        attrs = dict()

        attrs['transform'] = tuple(src.transform)[:6]

        if hasattr(src, 'crs') and src.crs:

            try:
                attrs['crs'] = src.crs.to_proj4()
            except:
                attrs['crs'] = src.crs.to_string()

        if hasattr(src, 'res'):
            attrs['res'] = src.res

        if hasattr(src, 'is_tiled'):
            attrs['is_tiled'] = np.uint8(src.is_tiled)

        if hasattr(src, 'nodatavals'):
            attrs['nodatavals'] = tuple(np.nan if nodataval is None else nodataval for nodataval in src.nodatavals)

        if hasattr(src, 'offsets'):
            attrs['offsets'] = src.scales

        if hasattr(src, 'offsets'):
            attrs['offsets'] = src.offsets

        if hasattr(src, 'descriptions') and any(src.descriptions):
            attrs['descriptions'] = src.descriptions

        if hasattr(src, 'units') and any(src.units):
            attrs['units'] = src.units

        return xr.DataArray(data_slice,
                            dims=('time', 'band', 'y', 'x'),
                            coords={'time': np.arange(1, data_slice.shape[0]+1),
                                    'band': np.arange(1, data_slice.shape[1]+1),
                                    'y': ycoords,
                                    'x': xcoords},
                            attrs=attrs)


def read_list(file_list, **kwargs):
    return [read_delayed(fn, **kwargs) for fn in file_list]


def read(filename,
         band_names=None,
         time_names=None,
         bounds=None,
         num_workers=1,
         **kwargs):

    """
    Reads a window slice in-memory

    Args:
        filename (str or list): A file name or list of file names to open read.
        band_names (Optional[list]): A list of names to give the output band dimension.
        time_names (Optional[list]): A list of names to give the time dimension.
        bounds (Optional[1d array-like]): A bounding box to subset to, given as
            [minx, miny, maxx, maxy] or [left, bottom, right, top]. Default is None.
        num_workers (Optional[int]): The number of parallel `dask` workers.
        kwargs (Optional[dict]): Keyword arguments to pass to `Rasterio`.

    Returns:
        Stacked data at the window slice (Xarray DataArray)
    """

    # Cannot pass 'chunks' to `rasterio`
    if 'chunks' in kwargs:
        del kwargs['chunks']

    if bounds and ('window' not in kwargs):
        kwargs['window'] = from_bounds(*bounds)

    if isinstance(filename, str):

        data = dask.compute(read_delayed(filename, **kwargs), num_workers=num_workers)

        # if not band_names:
        #     band_names = np.arange(1, data.shape[0]+1)

    else:

        if 'indexes' in kwargs:

            if isinstance(kwargs['indexes'], int):
                count = 1
            elif isinstance(kwargs['indexes'], list) or isinstance(kwargs['indexes'], np.ndarray):
                count = len(kwargs['indexes'])
            else:
                logger.exception("  Unknown `rasterio.open.read` `indexes` value")

        else:

            # If no `indexes` is given, all bands are read
            with rio.open(filename[0]) as src:
                count = src.count

        data = xr.concat(dask.compute(read_list(filename,
                                                **kwargs),
                                      num_workers=num_workers)[0], dim='time')

        if not band_names:
            band_names = np.arange(1, count+1)

        if not time_names:
            time_names = np.arange(1, len(filename)+1)

        data.coords['band'] = band_names
        data.coords['time'] = time_names

    return data


@contextmanager
def open(filename,
         use='xarray',
         return_as='array',
         band_names=None,
         time_names=None,
         bounds=None,
         num_workers=1,
         **kwargs):

    """
    Opens a file

    Args:
        filename (str or list): The file name or list of files to open.
        use (Optional[str]): The package to use for file opening backend. Default is 'xarray'.
            Choices are ['xarray', 'rasterio'].
        return_as (Optional[str]): When `use`='xarray', return `Xarray.DataArray` or `Xarray.Dataset`.
        band_names (Optional[1d array-like]): A list of band names if `return_as`='dataset' or
            `bounds` is given or `window` is given. Default is None.
        time_names (Optional[1d array-like]): A list of names to give the time dimension if `bounds` is given.
            Default is None.
        bounds (Optional[1d array-like]): A bounding box to subset to, given as [minx, maxy, miny, maxx].
            Default is None.
        num_workers (Optional[int]): The number of parallel workers for `dask` if `bounds` is given or
            `window` is given. Default is 1.
        kwargs (Optional[dict]): Keyword arguments passed to the file opener.

    Examples:
        >>> import geowombat as gw
        >>>
        >>> # Open an image
        >>> with gw.open('image.tif') as ds:
        >>>     print(ds)
        >>>
        >>> # Open a list of images
        >>> with gw.open(['image1.tif', 'image2.tif']) as ds:
        >>>     print(ds)
        >>>
        >>> # Open a list of images at a window slice
        >>> from rasterio.windows import Window
        >>> w = Window(row_off=0, col_off=0, height=100, width=100)
        >>>
        >>> # Stack two images, opening band 3
        >>> with gw.open(['image1.tif', 'image2.tif'],
        >>>     band_names=['date1', 'date2'],
        >>>     num_workers=8,
        >>>     indexes=3,
        >>>     window=w,
        >>>     out_dtype='float32') as ds:
        >>>
        >>>     print(ds)
    """

    if use not in ['xarray', 'rasterio']:
        logger.exception("  The `use` backend must be one of ['xarray', 'rasterio']")

    if return_as not in ['array', 'dataset']:
        logger.exception("  The `Xarray` object must be one of ['array', 'dataset']")

    if 'chunks' in kwargs:
        ch.check_chunktype(kwargs['chunks'], output='3d')
    else:

        # GDAL's default chunk size is typically 256
        kwargs['chunks'] = (1, 256, 256)

    if bounds or ('window' in kwargs and isinstance(kwargs['window'], Window)):

        yield read(filename,
                   band_names=band_names,
                   time_names=time_names,
                   bounds=bounds,
                   num_workers=num_workers,
                   **kwargs)

    else:

        if isinstance(filename, list):

            darray = xr.concat([xr.open_rasterio(fn, **kwargs) for fn in filename], dim='time')

            if return_as == 'array':

                if band_names:
                    darray.coords['band'] = band_names

                if time_names:
                    darray.coords['time'] = time_names
                else:
                    darray.coords['time'] = list(range(1, darray.shape[0]+1))

                yield darray

            else:

                # The Dataset variable 'bands' has 4 named dimensions
                #   --time, component, y, x
                yield xarray_to_xdataset(darray,
                                         band_names,
                                         time_names,
                                         ycoords=darray.y,
                                         xcoords=darray.x,
                                         attrs=darray.attrs)

        else:

            file_names = get_file_extension(filename)

            if file_names.f_ext.lower() not in IO_DICT['rasterio'] + IO_DICT['xarray']:
                logger.exception('  The file format is not recognized.')

            if file_names.f_ext.lower() in IO_DICT['rasterio']:

                if use == 'xarray':

                    with xr.open_rasterio(filename, **kwargs) as src:

                        if return_as == 'dataset':
                            yield xarray_to_xdataset(src, band_names, time_names)
                        else:

                            if band_names:
                                src.coords['band'] = band_names

                            yield src

                else:

                    with rio.open(filename, **kwargs) as src:
                        yield src

            else:

                if 'chunks' in kwargs and not isinstance(kwargs['chunks'], dict):
                    logger.exception('  The chunks should be a dictionary.')

                with xr.open_dataset(filename, **kwargs) as src:
                    yield src
