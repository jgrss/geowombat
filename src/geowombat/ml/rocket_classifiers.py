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

from ._base import GeoWombatDLClassifier, _to_series

#: MiniRocket kernels have length 9, so series must be at least this long.
MIN_TIMESTEPS = 9


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
        audit='warn',
        fill='train_median',
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
        self.audit = audit
        self.fill = fill
        self._transformer = None
        self._scaler = None
        self._estimator = None
        self._n_classes = None
        self._n_bands = None
        self._n_timesteps = None
        self._n_train_samples = None
        self.fitted_ = False

    def _make_estimator(self):
        if self.estimator is not None:
            return self.estimator
        return RidgeClassifierCV(
            alphas=np.logspace(-3, 3, 10), class_weight=self.class_weight
        )

    def fit(self, data, labels=None, col=None, targ_name="targ", **kwargs):
        self._check_series(data, 'fit', min_timesteps=MIN_TIMESTEPS)
        filled = self._prepare_fit_data(data)
        X, y = self._extract_training_series(filled, data, labels, col)
        # aeon expects (cases, channels, timepoints)
        series = np.ascontiguousarray(X.transpose(0, 2, 1))

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

        self._n_train_samples = len(y)
        self.fitted_ = True
        return self

    def predict(self, data, **kwargs):
        filled, residual = self._prepare_predict_data(data)

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
            filled.data,
            dtype=np.float32,
            drop_axis=[0, 1],
            new_axis=0,
        )

        return self._finalize_prediction(result_da, filled, residual)
