from typing import Any, List
from abc import abstractmethod
import itertools

from .windows import get_window_offsets

import numpy as np
from affine import Affine
import rasterio as rio
from rasterio.coords import BoundingBox
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

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
    JAX_INSTALLED = True
except:
    JAX_INSTALLED = False


class TransferLib(object):

    """
    Device transfers

    Args:
        transfer_lib (str): The device library to transfer to.
            Choices are ['jax', 'keras', 'numpy', 'pytorch', 'tensorflow'].

            'jax' -> GPU
            'keras' -> GPU
            'numpy' -> CPU
            'pytorch' -> GPU
            'tensorflow' -> GPU
    """

    def __init__(self, transfer_lib: str):
        self.transfer_lib = transfer_lib

    @staticmethod
    def jax(array):
        return jnp.asarray(array, dtype='float32')

    @staticmethod
    def keras(array):
        raise NotImplementedError

    @staticmethod
    def numpy(array):
        return np.asarray(array, dtype='float64')

    @staticmethod
    def pytorch(array):
        return torch.from_numpy(array).float().to('cuda:0')

    @staticmethod
    def tensorflow(array):
        return tf.convert_to_tensor(array, tf.float64)

    def __call__(self, array):
        return getattr(self, self.transfer_lib)(array)


class _Warp(object):

    def warp(self,
             dst_crs=None,
             dst_res=None,
             dst_bounds=None,
             resampling='nearest',
             nodata=None,
             warp_mem_limit=None,
             num_threads=None,
             window_size=None,
             padding=None):

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

            self.windows_ = get_window_offsets(dst_height,
                                               dst_width,
                                               window_size[0],
                                               window_size[1],
                                               return_as='list',
                                               padding=padding)

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
    def nodata(self):
        return self.vrts_[0].nodata

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
        self.band_dict = None

    def __call__(self, w, array, band_dict):

        self.band_dict = band_dict

        return w, self.calculate(array)

    def __repr__(self):

        return f"{self.__class__.__name__}():\n    " \
               f"self.dtype='{self.dtype}'\n    " \
               f"self.count={self.count}\n    " \
               f"self.compress='{self.compress}'\n    " \
               f"self.bigtiff='{self.bigtiff}'"

    def __str__(self):

        return f"{self.__class__.__name__}():\n    " \
               f"self.dtype='{self.dtype}'\n    " \
               f"self.count={self.count}\n    " \
               f"self.compress='{self.compress}'\n    " \
               f"self.bigtiff='{self.bigtiff}'\n    " \
               f"-> Array(numpy.ndarray | jax.numpy.DeviceArray | torch.Tensor | tensorflow.Tensor)[bands x height x width]"

    def __add__(self, other):

        if isinstance(other, TimeModulePipeline):
            return TimeModulePipeline([self] + other.modules)
        else:
            return TimeModulePipeline([self, other])

    @abstractmethod
    def calculate(self, data: Any) -> Any:

        """
        Calculates the user function

        Args:
            data (``numpy.ndarray`` |
                  ``jax.numpy.DeviceArray`` |
                  ``torch.Tensor`` |
                  ``tensorflow.Tensor``): The input array, shaped [time x bands x rows x columns].

        Returns:
            ``numpy.ndarray`` |
            ``jax.numpy.DeviceArray`` |
            ``torch.Tensor`` |
            ``tensorflow.Tensor``:
                Shaped (time|bands x rows x columns)
        """

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

    def __call__(self, w, array, band_dict):

        results = []
        for module in self.modules:

            res = module(w, array, band_dict)[1]

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

    def calculate(self, array):

        if isinstance(self.time_stats, str):
            return np.asarray(getattr(self, self.time_stats)(array))
        else:
            return np.asarray(self._stack(array, self.time_stats))

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

    def abs_slope_q1(self, data):
        """Calculates the absolute slope of the first quarter"""
        b1 = self._lstsq(data[:int(0.25*data.shape[0])])[0]
        b1[np.isnan(b1) | np.isinf(b1)] = 0
        return self._scale_min_max(jnp.fabs(b1), 0.0, 0.05, 0.0, 1.0)

    def abs_slope_q2(self, data):
        """Calculates the absolute slope of the second quarter"""
        b1 = self._lstsq(data[int(0.25*data.shape[0]):int(0.5*data.shape[0])])[0]
        b1[np.isnan(b1) | np.isinf(b1)] = 0
        return self._scale_min_max(jnp.fabs(b1), 0.0, 0.05, 0.0, 1.0)

    def abs_slope_q3(self, data):
        """Calculates the absolute slope of the third quarter"""
        b1 = self._lstsq(data[int(0.5*data.shape[0]):int(0.75*data.shape[0])])[0]
        b1[np.isnan(b1) | np.isinf(b1)] = 0
        return self._scale_min_max(jnp.fabs(b1), 0.0, 0.05, 0.0, 1.0)

    def abs_slope_q4(self, data):
        """Calculates the absolute slope of the fourth quarter"""
        b1 = self._lstsq(data[int(0.75*data.shape[0]):])[0]
        b1[np.isnan(b1) | np.isinf(b1)] = 0
        return self._scale_min_max(jnp.fabs(b1), 0.0, 0.05, 0.0, 1.0)

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

    @staticmethod
    def median(array):
        """Calculates the median"""
        return jnp.nanmedian(array, axis=0).squeeze()

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

    @staticmethod
    def percentile(array, p):
        """Calculates the nth percentile"""
        return jnp.nanpercentile(array, p, axis=0).squeeze()

    def _stack(self, array, stats):
        """Calculates a stack of statistics"""
        return jnp.vstack([getattr(self, 'percentile')(array, int(stat[10:]))[np.newaxis] if
                           stat.startswith('percentile') else
                           getattr(self, stat)(array)[np.newaxis] for stat in stats]).squeeze()
