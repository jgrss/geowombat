from .methods import GeoMethods

import mpglight.raster_tools as gl
import mpglight.vector_tools as vl

import numpy as np
import rasterio
from osgeo import gdal


def _wrap_mpglue(src):

    # Wrap the array as a GeoArray
    geo_src = src.copy()

    geo_src.update_info(left=src.left+(src.j*src.cellY),
                        top=src.top-(src.i*src.cellY))

    geo_src.update_info(right=geo_src.left+(src.ccols*src.cellY),
                        bottom=geo_src.top-(src.rrows*src.cellY))

    geo_src._create = create_raster
    geo_src._warp = warp
    geo_src._transform = vl.Transform
    geo_src._get_xy_offsets = vl.get_xy_offsets
    geo_src._nd_to_rgb = nd_to_rgb

    return src


class GeoArray(GeoMethods, np.ndarray):

    """
    >>> import mpglue as gl
    >>> from geoarray as GeoArray
    >>>
    >>> src = gl.ropen('image.tif')
    >>> array = src.read()
    >>>
    >>> garray = GeoArray(array, src)
    """

    def __new__(cls, array, src, info=None):

        obj = np.asarray(array).view(cls)

        if isinstance(src, gl.ropen):

            obj.lib = 'mpglue'
            src = _wrap_mpglue(src)

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

        obj.src = src
        obj.info = info

        return obj

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

    # def __array_wrap__(self, out_arr, src=None):
    #     return super(GeoArray, self).__array_wrap__(self, out_arr, src)


# class GeoArray(GeoMethods):
#
#     def __init__(self, data, src):
#         self.values = NumPyView(data, src)
