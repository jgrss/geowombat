"""MiniRocket time-series classifier for geowombat.

Wraps the MiniRocket transform of Dempster, Schmidt and Webb (2021) so
that it can be used through the ``geowombat.ml.fit()``, ``predict()``
and ``fit_predict()`` API on multi-temporal ``(time, band, y, x)``
data. The implementation comes from `aeon
<https://www.aeon-toolkit.org/>`_, a BSD-licensed port of the original
code made with the authors' permission.

Requires: ``pip install "geowombat[ml]"`` (installs ``aeon``).

Example
-------
>>> import geowombat as gw
>>> from geowombat.ml import fit_predict, MiniRocketClassifier
>>>
>>> # One file per acquisition date; at least 9 dates are required
>>> with gw.open(image_list, stack_dim='time') as src:
...     y = fit_predict(src, MiniRocketClassifier(), labels, col='lc')

References
----------
Dempster, A., Schmidt, D.F. and Webb, G.I. (2021). MiniRocket: A very
fast (almost) deterministic transform for time series classification.
*Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery
& Data Mining*, 248-257. https://arxiv.org/abs/2012.08791
"""

import dask.array as da
import numpy as np
import xarray as xr
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

try:
    from aeon.transformations.collection.convolution_based import MiniRocket
except ImportError as e:
    raise ImportError(
        "MiniRocketClassifier requires aeon. "
        "Install with: pip install aeon "
        "or: pip install geowombat[ml]"
    ) from e

from ._base import GeoWombatDLClassifier, _get_nodata_mask, _rasterize_labels

#: MiniRocket kernels have length 9, so series must be at least this long.
MIN_TIMESTEPS = 9


def _to_series(block):
    """Reshape ``(time, band, y, x)`` to aeon's ``(cases, channels, time)``.

    Returns a float32 array of shape ``(y * x, band, time)`` with NaN and
    infinite values replaced by 0 (MiniRocket does not support missing values).
    """
    nt, nb, ny, nx = block.shape
    series = block.transpose(2, 3, 1, 0).reshape(ny * nx, nb, nt)
    return np.nan_to_num(
        series.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )


class MiniRocketClassifier(GeoWombatDLClassifier):
    """MiniRocket classifier for satellite image time series.

    Each pixel's spectral trajectory ``(band, time)`` is transformed with
    MiniRocket (fixed dilated convolution kernels producing ~10,000
    "proportion of positive values" features), scaled, and classified
    with a linear model. The transform is CPU-only, almost deterministic
    and fast, which makes it a strong baseline for per-pixel
    classification of multi-date imagery.

    Works through :func:`geowombat.ml.fit`,
    :func:`geowombat.ml.predict` and :func:`geowombat.ml.fit_predict`
    like the other classifiers.

    Parameters
    ----------
    n_kernels : int
        Number of convolutional kernels (rounded down to a multiple of
        84). Default 10,000.
    max_dilations_per_kernel : int
        Maximum number of dilations per kernel. Default 32.
    estimator : sklearn classifier or None
        Classifier fitted on the scaled MiniRocket features. ``None``
        uses ``RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))``. Pass
        e.g. ``LogisticRegression()`` if probabilities are needed.
    class_weight : dict, 'balanced' or None
        Class weights for the default ridge classifier only. Default
        ``None``.
    max_train_samples : int or None
        If set, randomly subsample at most this many labeled pixels for
        training. Guards memory when label polygons cover many pixels.
        Default ``None`` (use all labeled pixels).
    batch_size : int
        Number of pixels transformed at a time during prediction. Each
        pixel expands to ``n_kernels`` float32 features, so the peak
        memory per dask block is roughly
        ``batch_size * n_kernels * 4`` bytes. Default 2048.
    n_jobs : int
        Threads used by the MiniRocket transform. ``-1`` uses all
        processors. Default 1.
    random_state : int, RandomState or None
        Seed for the kernel biases and the optional subsampling.
    verbose : int
        0 = silent. Default 0.

    Notes
    -----
    - Requires multi-temporal data opened with ``stack_dim='time'`` and
      at least 9 time steps. The fitted model only predicts series of
      the same length it was trained on.
    - NaN values are replaced by 0 before the transform. Fill or
      interpolate gaps (e.g. cloud masks) beforehand for best results.
    - Implementation: ``aeon.transformations.collection.convolution_based.MiniRocket``.

    References
    ----------
    Dempster, A., Schmidt, D.F. and Webb, G.I. (2021). MiniRocket: A
    very fast (almost) deterministic transform for time series
    classification. *KDD '21*, 248-257.
    https://arxiv.org/abs/2012.08791

    Example
    -------
    >>> with gw.open(image_list, stack_dim='time') as src:
    ...     y = fit_predict(
    ...         src, MiniRocketClassifier(random_state=42), labels, col='lc'
    ...     )
    """

    def __init__(
        self,
        n_kernels=10_000,
        max_dilations_per_kernel=32,
        estimator=None,
        class_weight=None,
        max_train_samples=None,
        batch_size=2048,
        n_jobs=1,
        random_state=None,
        verbose=0,
    ):
        self.n_kernels = n_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.estimator = estimator
        self.class_weight = class_weight
        self.max_train_samples = max_train_samples
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self._transformer = None
        self._scaler = None
        self._estimator = None
        self._n_classes = None
        self._n_bands = None
        self._n_timesteps = None
        self._n_train_samples = None
        self.fitted_ = False

    @staticmethod
    def _check_series(data, method):
        if data.ndim != 4 or 'time' not in data.dims:
            raise ValueError(
                f"MiniRocketClassifier.{method}() requires multi-temporal "
                "data with a 'time' dimension (time, band, y, x). "
                "Open data with stack_dim='time'."
            )
        if data.sizes['time'] < MIN_TIMESTEPS:
            raise ValueError(
                f"MiniRocketClassifier requires at least {MIN_TIMESTEPS} "
                f"time steps; found {data.sizes['time']}."
            )

    def _make_estimator(self):
        if self.estimator is not None:
            return self.estimator
        return RidgeClassifierCV(
            alphas=np.logspace(-3, 3, 10), class_weight=self.class_weight
        )

    def fit(self, data, labels=None, col=None, targ_name="targ", **kwargs):
        self._check_series(data, 'fit')
        if labels is None or col is None:
            raise ValueError(
                "MiniRocketClassifier is supervised: pass labels and col."
            )

        label_arr, self._n_classes = _rasterize_labels(data, labels, col)
        flat_labels = label_arr.ravel()
        mask = flat_labels > 0
        if not mask.any():
            raise ValueError("No labeled pixels overlap the raster.")

        series = _to_series(data.values)[mask]
        y = (flat_labels[mask] - 1).astype(np.int64)

        if (
            self.max_train_samples is not None
            and len(y) > self.max_train_samples
        ):
            rng = check_random_state(self.random_state)
            idx = rng.choice(len(y), self.max_train_samples, replace=False)
            series, y = series[idx], y[idx]

        if self.verbose:
            print(
                f"MiniRocket: fitting on {len(y)} pixels, "
                f"{data.sizes['band']} bands, {data.sizes['time']} time steps"
            )

        self._transformer = MiniRocket(
            n_kernels=self.n_kernels,
            max_dilations_per_kernel=self.max_dilations_per_kernel,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        feats = self._transformer.fit_transform(series)
        self._scaler = StandardScaler(with_mean=False).fit(feats)
        self._estimator = self._make_estimator()
        self._estimator.fit(self._scaler.transform(feats), y)

        self._n_bands = data.sizes['band']
        self._n_timesteps = data.sizes['time']
        self._n_train_samples = len(y)
        self.fitted_ = True
        return self

    def predict(self, data, **kwargs):
        if not self.fitted_:
            raise RuntimeError("Call fit() before predict().")
        self._check_series(data, 'predict')
        if data.sizes['time'] != self._n_timesteps:
            raise ValueError(
                f"MiniRocketClassifier was fitted on {self._n_timesteps} "
                f"time steps but the data has {data.sizes['time']}. "
                "MiniRocket features depend on the series length."
            )
        if data.sizes['band'] != self._n_bands:
            raise ValueError(
                f"MiniRocketClassifier was fitted on {self._n_bands} bands "
                f"but the data has {data.sizes['band']}."
            )

        # Time and band must be single chunks inside each block
        if not hasattr(data.data, 'dask'):
            data = data.chunk(
                {
                    'time': -1,
                    'band': -1,
                    'y': min(512, data.sizes['y']),
                    'x': min(512, data.sizes['x']),
                }
            )
        else:
            data = data.chunk({'time': -1, 'band': -1})

        transformer = self._transformer
        scaler = self._scaler
        estimator = self._estimator
        batch_size = self.batch_size

        def _predict_block(block):
            # block: (time, band, y, x) numpy array
            nt, nb, cy, cx = block.shape
            series = _to_series(block)
            preds = np.empty(len(series), dtype=np.float32)
            for i in range(0, len(series), batch_size):
                feats = transformer.transform(series[i : i + batch_size])
                preds[i : i + batch_size] = estimator.predict(
                    scaler.transform(feats)
                )
            # Back to 1-based class ids
            return (preds + 1).reshape(1, cy, cx)

        result_da = da.map_blocks(
            _predict_block,
            data.data,
            dtype=np.float32,
            drop_axis=[0, 1],
            new_axis=0,
        )

        result = xr.DataArray(
            result_da,
            dims=("band", "y", "x"),
            coords={
                "band": ["targ"],
                "y": data.y,
                "x": data.x,
            },
        )

        nd_mask = _get_nodata_mask(data)
        if nd_mask is not None:
            # xr.where puts the mask dims first; restore (band, y, x)
            result = xr.where(nd_mask, np.nan, result).transpose(
                'band', 'y', 'x'
            )
            # xr.where names the result after the mask; drop it
            result.name = None

        return result.assign_attrs(**data.attrs)
