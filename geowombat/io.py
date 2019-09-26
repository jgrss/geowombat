from .errors import logger

import numpy as np
import xarray as xr
import dask
import dask.array as da
import rasterio as rio


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


def read(filename, band_names=None, time_names=None, num_workers=1, **kwargs):

    """
    Reads a window slice in-memory

    Args:
        filename (str or list): A file name or list of file names to open read.
        band_names (Optional[list]): A list of names to give the output band dimension.
        time_names (Optional[list]): A list of names to give the time dimension.
        num_workers (Optional[int]): The number of parallel `dask` workers.
        kwargs (Optional[dict]): Keyword arguments to pass to `Rasterio`.

    Returns:
        Stacked data at the window slice (Xarray DataArray)
    """

    if 'chunks' in kwargs:
        del kwargs['chunks']

    if isinstance(filename, str):

        data = dask.compute(read_delayed(filename, **kwargs), num_workers=num_workers)

        if not band_names:
            band_names = np.arange(1, data.shape[0]+1)

        data.coords['band'] = band_names

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