from contextlib import contextmanager

from . import geoxarray
from .errors import logger
from . import helpers

import rasterio as rio
import xarray as xr
import dask.array as da


IO_DICT = dict(rasterio=['.tif', '.img'],
               xarray=['.nc'])


@contextmanager
def open(filename,
         use='xarray',
         xarray_return='array',
         band_names=None,
         dates=None,
         **kwargs):

    """
    Opens a file

    Args:
        filename (str): The file name to open.
        use (Optional[str]): The package to use for file opening backend. Default is 'xarray'.
        xarray_return (Optional[str]): When `use`='xarray', return `Xarray.DataArray` or `Xarray.Dataset`.
        band_names (Optional[array-like]): A list of band names if `xarray_return`='dataset'. Default is None.
        dates (Optional[array-like]): A list of dates if `filename`=`list` and `xarray_return`='dataset'. Default is None.
        kwargs (Optional[dict]): Keyword arguments passed to the file opener.
    """

    if use not in ['xarray', 'rasterio']:
        logger.exception("  The `use` backend must be one of ['xarray', 'rasterio']")

    if xarray_return not in ['array', 'dataset']:
        logger.exception("  The `Xarray` object must be one of ['array', 'dataset']")

    if 'chunks' not in kwargs:
        kwargs['chunks'] = (1, 512, 512)

    if isinstance(filename, list):

        darray = [xr.open_rasterio(fn, **kwargs) for fn in filename]

        # The Dataset variable 'bands' has 4 named dimensions
        #   --time, component, y, x
        yield helpers.xarray_to_xdataset(da.stack(darray),
                                         band_names,
                                         dates,
                                         ycoords=darray[0].y,
                                         xcoords=darray[0].x,
                                         attrs=darray[0].attrs)

    else:

        file_names = helpers.get_file_extension(filename)

        if file_names.f_ext.lower() not in IO_DICT['rasterio'] + IO_DICT['xarray']:
            logger.exception('  The file format is not recognized.')

        if file_names.f_ext.lower() in IO_DICT['rasterio']:

            if use == 'xarray':

                with xr.open_rasterio(filename, **kwargs) as src:

                    if xarray_return == 'dataset':
                        yield helpers.xarray_to_xdataset(src, band_names, dates)
                    else:
                        yield src

            else:

                with rio.open(filename, **kwargs) as src:
                    yield src

        else:

            with xr.open_dataset(filename, **kwargs) as src:
                yield src
