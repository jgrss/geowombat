from typing import List
from abc import abstractmethod
import itertools

import numpy as np
from affine import Affine
import rasterio as rio
from rasterio.coords import BoundingBox
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

try:
    import torch
    PYTORCH_INSTALLED = True
except:
    PYTORCH_INSTALLED = False

try:
    import tensorflow as tf
    TENSORFLOW_INSTALLED = True
except:
    TENSORFLOW_INSTALLED = False

try:
    import jax.numpy as jnp
    from jax import device_put as jax_put
    JAX_INSTALLED = True
except:
    JAX_INSTALLED = False


class GPULib(object):

    def __init__(self, gpu_lib):
        self.gpu_lib = gpu_lib

    @staticmethod
    def jax(array):
        return jax_put(array)

    @staticmethod
    def pytorch(array):
        return torch.from_numpy(array).float().to('cuda:0')

    @staticmethod
    def tensorflow(array):
        return tf.convert_to_tensor(array, tf.float64)

    def __call__(self, array):
        return getattr(self, self.gpu_lib)


class _Warp(object):

    def warp(self,
             dst_crs=None,
             dst_res=None,
             dst_bounds=None,
             resampling='nearest',
             nodata=None,
             warp_mem_limit=None,
             num_threads=None,
             window_size=None):

        if dst_crs is None:
            dst_crs = self.srcs_[0].crs

        if dst_res is None:
            dst_res = self.srcs_[0].res

        if dst_bounds is None:
            dst_bounds = self.srcs_[0].bounds
        else:

            if isinstance(dst_bounds, list) or isinstance(dst_bounds, tuple):
                dst_bounds = BoundingBox(left=dst_bounds[0], bottom=dst_bounds[1], right=dst_bounds[2], top=dst_bounds[3])

        if nodata is None:
            nodata = self.srcs_[0].nodata

        if warp_mem_limit is None:
            warp_mem_limit = 256

        if num_threads is None:
            num_threads = 1

        dst_transform = Affine(dst_res[0], 0.0, dst_bounds.left, 0.0, -dst_res[1], dst_bounds.top)

        dst_width = int((dst_bounds.right - dst_bounds.left) / dst_res[0])
        dst_height = int((dst_bounds.top - dst_bounds.bottom) / dst_res[1])

        vrt_options = {'resampling': getattr(Resampling, resampling),
                       'crs': dst_crs,
                       'transform': dst_transform,
                       'height': dst_height,
                       'width': dst_width,
                       'nodata': nodata,
                       'warp_mem_limit': warp_mem_limit,
                       'warp_extras': {'multi': True,
                                       'warp_option': f'NUM_THREADS={num_threads}'}}

        self.vrts_ = [WarpedVRT(src,
                                src_crs=src.crs,
                                src_transform=src.transform,
                                **vrt_options) for src in self.srcs_]

        if window_size:

            def adjust_window(pixel_index, block_size, rows_cols):
                return block_size if (pixel_index + block_size) < rows_cols else rows_cols - pixel_index

            self.windows_ = []

            for row_off in range(0, dst_height, window_size[0]):
                wheight = adjust_window(row_off, window_size[0], dst_height)
                for col_off in range(0, dst_width, window_size[1]):
                    wwidth = adjust_window(col_off, window_size[1], dst_width)
                    self.windows_.append(Window(row_off=row_off, col_off=col_off, height=wheight, width=wwidth))

        else:
            self.windows_ = [[w[1] for w in src.block_windows(1)] for src in self.vrts_][0]


class _SeriesProps(object):

    @property
    def crs(self):
        return self.vrts_[0].crs

    @property
    def transform(self):
        return self.vrts_[0].transform

    @property
    def count(self):
        return self.vrts_[0].count

    @property
    def width(self):
        return self.vrts_[0].width

    @property
    def height(self):
        return self.vrts_[0].height

    @property
    def blockxsize(self):
        return self.windows_[0].width

    @property
    def blockysize(self):
        return self.windows_[0].height

    @property
    def nchunks(self):
        return len(self.windows_)

    @property
    def band_dict(self):
        return dict(zip(self.band_names, range(0, self.count))) if self.band_names else None


class BaseSeries(_SeriesProps, _Warp):

    def open(self, filenames):
        self.srcs_ = [rio.open(fn) for fn in filenames]


class TimeModule(object):

    def __init__(self):

        self.dtype = 'float64'
        self.count = 1
        self.compress = 'lzw'
        self.bigtiff = 'NO'

    def __call__(self, *args):

        w, array, band_dict = list(itertools.chain(*args))

        return w, self.calculate(array, band_dict=band_dict)

    def __repr__(self):
        return f"{self.__class__.__name__}():\n    self.dtype='{self.dtype}'\n    self.count={self.count}\n    self.compress='{self.compress}'\n    self.bigtiff='{self.bigtiff}'"

    def __str__(self):
        return "jax.numpy.DeviceArray()[bands x height x width]"

    def __add__(self, other):

        if isinstance(other, TimeModulePipeline):
            return TimeModulePipeline([self] + other.modules)
        else:
            return TimeModulePipeline([self, other])

    @abstractmethod
    def calculate(self, array: jnp.DeviceArray, band_dict: dict = None) -> jnp.DeviceArray:
        raise NotImplementedError


class TimeModulePipeline(object):

    def __init__(self, module_list: List[TimeModule]):

        self.modules = module_list

        self.count = 0
        for module in self.modules:
            self.count += module.count

        self.dtype = self.modules[-1].dtype
        self.compress = self.modules[-1].compress
        self.bigtiff = self.modules[-1].bigtiff

    def __add__(self, other):

        if isinstance(other, TimeModulePipeline):
            return TimeModulePipeline(self.modules + other.modules)
        else:
            return TimeModulePipeline(self.modules + [other])

    def __call__(self, *args):

        w = list(itertools.chain(*args))[0]

        results = []
        for module in self.modules:

            res = module(*args)[1]

            if len(res.shape) == 2:
                res = res[np.newaxis]

            results.append(res)

        return w, jnp.vstack(results).squeeze()


class SeriesStats(TimeModule):

    def __init__(self, time_stats):

        super(SeriesStats, self).__init__()

        self.time_stats = time_stats

        if isinstance(self.time_stats, str):
            self.count = 1
        else:
            self.count = len(list(self.time_stats))

    def calculate(self, array, band_dict=None):

        if isinstance(self.time_stats, str):
            return getattr(self, self.time_stats)(array)
        else:
            return self._stack(array, self.time_stats)

    @staticmethod
    def _scale_min_max(xv, mni, mxi, mno, mxo):
        return ((((mxo - mno) * (xv - mni)) / (mxi - mni)) + mno).clip(mno, mxo)

    @staticmethod
    def _lstsq(data):

        ndims, nbands, nrows, ncols = data.shape

        M = data.squeeze().transpose(1, 2, 0).reshape(nrows*ncols, ndims).T

        x = jnp.arange(0, M.shape[0])

        # Fit a least squares solution to each sample
        return jnp.linalg.lstsq(jnp.c_[x, jnp.ones_like(x)], M, rcond=None)[0]

    @staticmethod
    def amp(array):
        """Calculates the amplitude"""
        return jnp.nanmax(array, axis=0).squeeze() - jnp.nanmin(array, axis=0).squeeze()

    @staticmethod
    def cv(array):
        """Calculates the coefficient of variation"""
        return jnp.nanstd(array, axis=0).squeeze() / (jnp.nanmean(array, axis=0).squeeze() + 1e-9)

    @staticmethod
    def max(array):
        """Calculates the max"""
        return jnp.nanmax(array, axis=0).squeeze()

    @staticmethod
    def mean(array):
        """Calculates the mean"""
        return jnp.nanmean(array, axis=0).squeeze()

    def mean_abs_diff(self, array):
        """Calculates the mean absolute difference"""
        d = jnp.nanmean(jnp.fabs(jnp.diff(array, n=1, axis=0)), axis=0).squeeze()
        return self._scale_min_max(d, 0.0, 0.05, 0.0, 1.0)

    @staticmethod
    def min(array):
        """Calculates the min"""
        return jnp.nanmin(array, axis=0).squeeze()

    @staticmethod
    def norm_abs_energy(array):
        """Calculates the normalized absolute energy"""
        return jnp.nansum(array**2, axis=0).squeeze() / (jnp.nanmax(array, axis=0)**2 * array.shape[0]).squeeze()

    def _stack(self, array, stats):
        """Calculates a stack of statistics"""
        return jnp.vstack([getattr(self, stat)(array)[np.newaxis] for stat in stats]).squeeze()
