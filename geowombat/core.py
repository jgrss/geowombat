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

    if 'chunks' not in kwargs:
        kwargs['chunks'] = (1, 512, 512)

    if bounds or ('window' in kwargs and isinstance(kwargs['window'], Window)):

        yield read(filename,
                   band_names=band_names,
                   time_names=time_names,
                   bounds=bounds,
                   num_workers=num_workers,
                   **kwargs)

    else:

        if isinstance(filename, list):

            if return_as == 'array':
                yield xr.concat([xr.open_rasterio(fn, **kwargs) for fn in filename], dim='band')
            else:

                darray = xr.concat([xr.open_rasterio(fn, **kwargs) for fn in filename], dim='band')

                # The Dataset variable 'bands' has 4 named dimensions
                #   --time, component, y, x
                yield conversion.xarray_to_xdataset(darray,
                                                    band_names,
                                                    time_names,
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

                        if return_as == 'dataset':
                            yield conversion.xarray_to_xdataset(src, band_names, time_names)
                        else:

                            if band_names:
                                src.coords['band'] = band_names

                            yield src

                else:

                    with rio.open(filename, **kwargs) as src:
                        yield src

            else:

                with xr.open_dataset(filename, **kwargs) as src:
                    yield src
