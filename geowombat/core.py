from contextlib import contextmanager

from . import geoxarray
from . import conversion
from . import helpers
from .errors import logger
from .io import read

import rasterio as rio
from rasterio.windows import Window
import xarray as xr


IO_DICT = dict(rasterio=['.tif', '.img'],
               xarray=['.nc'])


@contextmanager
def open(filename,
         use='xarray',
         xarray_return='array',
         band_names=None,
         dates=None,
         num_workers=1,
         **kwargs):

    """
    Opens a file

    Args:
        filename (str): The file name to open.
        use (Optional[str]): The package to use for file opening backend. Default is 'xarray'.
        xarray_return (Optional[str]): When `use`='xarray', return `Xarray.DataArray` or `Xarray.Dataset`.
        band_names (Optional[array-like]): A list of band names if `xarray_return`='dataset'. Default is None.
        dates (Optional[array-like]): A list of dates if `filename`=`list` and `xarray_return`='dataset'. Default is None.
        num_workers (Optional[int]): The number of parallel workers.
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
        >>>     window=w
        >>>     out_dtype='float32') as ds:
        >>>
        >>>     print(ds)
    """

    if use not in ['xarray', 'rasterio']:
        logger.exception("  The `use` backend must be one of ['xarray', 'rasterio']")

    if xarray_return not in ['array', 'dataset']:
        logger.exception("  The `Xarray` object must be one of ['array', 'dataset']")

    if 'chunks' not in kwargs:
        kwargs['chunks'] = (1, 512, 512)

    if 'window' in kwargs and isinstance(kwargs['window'], Window):
        yield read(filename, band_names=band_names, num_workers=num_workers, **kwargs)
    else:

        if isinstance(filename, list):

            if xarray_return == 'array':
                yield xr.concat([xr.open_rasterio(fn, **kwargs) for fn in filename], dim='band')
            else:

                darray = xr.concat([xr.open_rasterio(fn, **kwargs) for fn in filename], dim='band')

                # The Dataset variable 'bands' has 4 named dimensions
                #   --time, component, y, x
                yield conversion.xarray_to_xdataset(darray,
                                                    band_names,
                                                    dates,
                                                    ycoords=darray.y,
                                                    xcoords=darray.x,
                                                    attrs=darray.attrs)

        else:

            file_names = helpers.get_file_extension(filename)

            if file_names.f_ext.lower() not in IO_DICT['rasterio'] + IO_DICT['xarray']:
                logger.exception('  The file format is not recognized.')

            if file_names.f_ext.lower() in IO_DICT['rasterio']:

                if use == 'xarray':

                    with xr.open_rasterio(filename, **kwargs) as src:

                        if xarray_return == 'dataset':
                            yield conversion.xarray_to_xdataset(src, band_names, dates)
                        else:
                            yield src

                else:

                    with rio.open(filename, **kwargs) as src:
                        yield src

            else:

                with xr.open_dataset(filename, **kwargs) as src:
                    yield src
