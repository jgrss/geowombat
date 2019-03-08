from .methods import GeoMethods

import mpglight.raster_tools as gl
import mpglight.vector_tools as vl

import numpy as np
import rasterio
from osgeo import gdal


def _wrap_gdal(src):
    pass


def _wrap_rasterio(src):
    pass


def _wrap_mpglue(src):

    # Wrap the array as a GeoArray
    geo_src = src.copy()

    geo_src.update_info(left=src.left+(src.j*src.cellY),
                        top=src.top-(src.i*src.cellY))

    geo_src.update_info(right=geo_src.left+(src.ccols*src.cellY),
                        bottom=geo_src.top-(src.rrows*src.cellY))

    geo_src._create = gl.create_raster
    geo_src._warp = gl.warp
    geo_src._transform = vl.Transform
    geo_src._get_xy_offsets = vl.get_xy_offsets
    geo_src._nd_to_rgb = gl.nd_to_rgb

    return geo_src


class GeoArray(GeoMethods, np.ndarray):

    """
    >>> import mpglue as gl
    >>> from geoarray as GeoArray
    >>>
    >>> with gl.ropen('image.tif') as src:
    >>>
    >>>     array = src.read()
    >>>     garray = GeoArray(array, src)
    """

    def __new__(cls, array, src, info=None):

        obj = np.asarray(array).view(cls)

        if isinstance(src, gl.ropen):

            obj.lib = 'mpglue'
            obj.src = _wrap_mpglue(src)

        elif isinstance(src, rasterio.io.DatasetReader):
            obj.lib = 'rasterio'
            # TODO: set self.attrs for rasterio
        elif isinstance(src, gdal.Dataset):
            obj.lib = 'gdal'
            # TODO: set self.attrs for GDAL

        obj.no_data_ = 0

        obj.original_layers = 1

        if len(array.shape) == 2:
            obj.original_rows, obj.original_columns = array.shape
        else:
            obj.original_layers, obj.original_rows, obj.original_columns = array.shape

        obj.info = info

        return obj

    # TODO: geo-aware math operations
    # def __add__(self):
    #     return

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.lib = getattr(obj, 'lib', None)
        self.no_data_ = getattr(obj, 'no_data_', None)
        self.info = getattr(obj, 'info', None)
        self.src = getattr(obj, 'src', None)
        self.original_layers = getattr(obj, 'original_layers', None)
        self.original_rows = getattr(obj, 'original_rows', None)
        self.original_columns = getattr(obj, 'original_columns', None)


class GeoWombat(object):

    """
    Args:
        file_name (str)
        backend (Optional[str])

    Example:
        >>> with GeoWombat('image.tif', backend='rasterio') as src:
        >>>     garray = src.read(bands=-1)
    """

    def __init__(self, file_name, backend='rasterio'):

        self.file_name = file_name
        self.backend = backend
        self.src = None

    def read(self, lazy=False, **kwargs):

        if self.backend == 'mpglue':

            with gl.ropen(self.file_name) as self.src:
                garray = GeoArray(src.read(**kwargs), self.src)

        elif self.backend == 'rasterio':

            with rasterio.open(self.file_name) as self.src:
                garray = GeoArray(src.read(**kwargs), self.src)

        return garray

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.src = None

    def __del__(self):
        self.__exit__(None, None, None)
