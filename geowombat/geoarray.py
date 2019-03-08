from .methods import GeoMethods

# TODO: replace light version with real mpglue
try:
    from mpglue import raster_tools as gl
    from mpglue import vector_tools as vl
except:
    import mpglight.raster_tools as gl
    import mpglight.vector_tools as vl

import numpy as np
import rasterio
from osgeo import gdal


def _warp_rasterio(src):
    return


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


def _wrap_gdal(src):

    with gl.ropen(file_name=src.GetFileList()[0]) as glsrc:
        geo_src = _wrap_mpglue(glsrc)

    return geo_src


class GeoArray(GeoMethods, np.ndarray):

    """
    >>> import mpglue as gl
    >>> import geowombat as gwb
    >>>
    >>> with gl.ropen('image.tif') as src:
    >>>
    >>>     array = src.read()
    >>>     garray = gwb.GeoArray(array, src)
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
            obj.src = _wrap_gdal(src)

        obj.original_layers = 1

        if len(array.shape) == 2:
            obj.original_rows, obj.original_columns = array.shape
        else:
            obj.original_layers, obj.original_rows, obj.original_columns = array.shape

        obj.no_data_ = 0
        obj.info = info
        obj.layer_names = map(str, list(range(1, obj.original_layers+1)))

        return obj

    # def __add__(self, other):
    #     return self._rc(self + np.asarray(other))
    #
    # def _rc(self, a):
    #
    #     if len(shape(a)) == 0:
    #         return a
    #     else:
    #         return self.__class__(a)

    # TODO: geo-aware math operations
    # def __add__(self, x):
    #
    #     try:
    #         return self.geo_add(x)
    #     except:
    #         return self + x

    # def __array_wrap__(self, result):
    #     return GeoArray(result, self.src)

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.lib = getattr(obj, 'lib', None)
        self.no_data_ = getattr(obj, 'no_data_', None)
        self.layer_names = getattr(obj, 'layer_names', None)
        self.src = getattr(obj, 'src', None)
        self.original_layers = getattr(obj, 'original_layers', None)
        self.original_rows = getattr(obj, 'original_rows', None)
        self.original_columns = getattr(obj, 'original_columns', None)


class open(GeoMethods):

    """
    A class to open Wombat GeoArrays

    Args:
        file_name (str)
        backend (Optional[str])

    Attributes:
        read
        file_name
        backend

    Example:
        >>> import geowombat as gwb
        >>>
        >>> with gwb.GeoOpen('image.tif', backend='mpglue') as src:
        >>>     garray = src.read(bands=-1)
    """

    def __init__(self, file_name, backend='mpglue'):

        self.file_name = file_name
        self.backend = backend

    def read(self, names=None, **kwargs):

        """
        Args:
            names (Optional[list])
            kwargs (Optional[dict])
        """

        if self.backend == 'mpglue':

            with gl.ropen(self.file_name) as src:
                garray = GeoArray(src.read(**kwargs), src)

        elif self.backend == 'rasterio':

            with rasterio.open(self.file_name) as src:
                garray = GeoArray(src.read(**kwargs), src)

        src = None

        if names:
            garray.set_names(names)

        return garray

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.src = None

    def __del__(self):
        self.__exit__(None, None, None)
