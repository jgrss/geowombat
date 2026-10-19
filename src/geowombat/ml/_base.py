"""Shared, torch-free base utilities for geowombat classifiers.

These helpers are used by the classifiers that handle their own xarray
``fit()`` / ``predict()`` (deep-learning models in
:mod:`geowombat.ml.dl_classifiers` and the MiniRocket time-series
classifier in :mod:`geowombat.ml.rocket_classifiers`). They are kept
here so that classifiers which do not need PyTorch can import them
without triggering the torch import guard.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr

from .. import polygon_to_array
from ..core.timeseries import (
    AUDIT_POLICIES,
    FILL_STRATEGIES,
    TimeSeriesFiller,
    audit_input,
    to_float_with_nan,
)


def _get_nodata_mask(data):
    """Return a 2-D boolean mask (y, x) that is True for nodata pixels.

    Prefers the ``_nodata_mask`` coordinate (warped from the original file)
    when available.  Falls back to value comparison using ``nodatavals`` from
    attributes.
    """
    if '_nodata_mask' in data.coords:
        return data.coords['_nodata_mask']

    nodatavals = data.attrs.get("nodatavals")
    if not nodatavals or len(nodatavals) == 0:
        return None

    src_nd = nodatavals[0]
    if isinstance(src_nd, float) and np.isnan(src_nd):

        def _is_nd(a):
            return a.isnull()

    else:

        def _is_nd(a):
            return a == src_nd

    if data.ndim == 4:
        return _is_nd(data.isel(time=0)).any(dim='band')
    else:
        return _is_nd(data).any(dim='band')


def _rasterize_labels(data, labels, col):
    """Rasterize vector labels onto the data grid.

    Parameters
    ----------
    data : xarray.DataArray
        Reference raster (band, y, x) or (time, band, y, x).
    labels : str, Path, or GeoDataFrame
        Vector labels.
    col : str
        Column name with class values.

    Returns
    -------
    label_array : numpy.ndarray
        2D array (y, x) with integer class labels (0 = nodata).
    n_classes : int
        Number of unique non-zero classes.
    """
    if isinstance(labels, (str, Path)):
        labels = gpd.read_file(labels)

    labels = labels.copy()

    # Encode labels to 1-based integers (0 = nodata)
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(labels[col])
    labels[col] = le.transform(labels[col]) + 1

    # Ensure data is dask-backed (polygon_to_array needs .chunksize)
    if not hasattr(data.data, 'chunksize'):
        data = data.chunk(
            {'y': min(256, data.sizes['y']), 'x': min(256, data.sizes['x'])}
        )

    label_raster = polygon_to_array(labels, col=col, data=data)
    label_arr = label_raster.values.squeeze()

    n_classes = len(le.classes_)

    return label_arr, n_classes


def _to_series(block):
    """Reshape ``(time, band, y, x)`` to ``(y * x, band, time)`` float32.

    NaN and infinite values are replaced by 0 as a last-resort guard; the
    classifiers fill gaps before reaching this point.
    """
    nt, nb, ny, nx = block.shape
    series = block.transpose(2, 3, 1, 0).reshape(ny * nx, nb, nt)
    return np.nan_to_num(
        series.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )


class GeoWombatDLClassifier:
    """Base class for classifiers that handle their own xarray fit/predict.

    Subclasses implement ``fit()`` and ``predict()``. The
    ``_is_gw_dl_classifier`` attribute makes ``geowombat.ml.fit()`` /
    ``predict()`` / ``fit_predict()`` delegate to those methods instead of
    the sklearn_xarray pipeline.

    The base provides the shared input hygiene: declared nodata is turned
    into NaN, inputs are audited (see
    :func:`geowombat.core.timeseries.audit_input`), gaps are filled with
    the ``fill`` strategy (see
    :class:`geowombat.core.timeseries.TimeSeriesFiller`) and predictions
    are assembled into a ``(band='targ', y, x)`` DataArray with nodata and
    unfillable pixels set to NaN.

    Attributes set by subclasses' ``__init__``:

    - ``audit``: ``'warn'`` (default), ``'raise'`` or ``'off'``.
    - ``fill``: ``'train_median'`` (default), ``'interpolate'``,
      ``'zero'`` or ``'none'``.

    Attributes available after ``fit()``:

    - ``fit_report_``: the missing-value report of the training data.
    - ``filler_``: the fitted :class:`TimeSeriesFiller`.
    """

    _is_gw_dl_classifier = True
    audit = 'warn'
    fill = 'train_median'
    fitted_ = False
    fit_report_ = None
    filler_ = None
    _n_bands = None
    _n_timesteps = None
    _n_classes = None

    def fit(self, data, labels=None, col=None, targ_name="targ", **kwargs):
        raise NotImplementedError

    def predict(self, data, **kwargs):
        raise NotImplementedError

    # -- shared helpers ---------------------------------------------------

    def _check_series(self, data, method, min_timesteps=1, require_time=True):
        name = type(self).__name__
        has_time = 'time' in data.dims
        if require_time and (data.ndim != 4 or not has_time):
            raise ValueError(
                f"{name}.{method}() requires multi-temporal data with a "
                "'time' dimension (time, band, y, x). Open data with "
                "stack_dim='time'."
            )
        if not require_time and data.ndim not in (3, 4):
            raise ValueError(
                f"{name}.{method}() expects (band, y, x) or "
                "(time, band, y, x) data."
            )
        if has_time and data.sizes['time'] < min_timesteps:
            raise ValueError(
                f"{name} requires at least {min_timesteps} time steps; "
                f"found {data.sizes['time']}."
            )

    def _validate_hygiene(self):
        if self.audit not in AUDIT_POLICIES:
            raise ValueError(f"audit must be one of {AUDIT_POLICIES}")
        if self.fill not in FILL_STRATEGIES:
            raise ValueError(f"fill must be one of {FILL_STRATEGIES}")

    def _prepare_fit_data(self, data):
        """Audit and fill the training stack; returns the filled array."""
        self._validate_hygiene()
        floated = to_float_with_nan(data)
        self.fit_report_ = audit_input(
            floated, stage='fit', policy=self.audit, fill=self.fill
        )
        self.filler_ = TimeSeriesFiller(self.fill).fit(floated)
        filled = self.filler_.transform(floated)
        self._n_bands = data.sizes['band']
        self._n_timesteps = data.sizes['time'] if 'time' in data.dims else None
        return filled

    def _extract_training_series(self, filled, data, labels, col):
        """Labelled pixels as ``(n, time, band)`` float32 and 0-based ``y``."""
        name = type(self).__name__
        if labels is None or col is None:
            raise ValueError(f"{name} is supervised: pass labels and col.")
        label_arr, self._n_classes = _rasterize_labels(data, labels, col)
        arr = filled.values
        if arr.ndim == 3:
            arr = arr[None]
        nt, nb, ny, nx = arr.shape
        pixels = arr.transpose(2, 3, 0, 1).reshape(ny * nx, nt, nb)
        flat_labels = label_arr.ravel()
        mask = flat_labels > 0
        if not mask.any():
            raise ValueError("No labeled pixels overlap the raster.")
        X = pixels[mask].astype(np.float32)
        y = (flat_labels[mask] - 1).astype(np.int64)
        bad = np.isnan(X).any(axis=(1, 2))
        if bad.any():
            if self.fill == 'none':
                raise ValueError(
                    f"{int(bad.sum())} training pixels contain missing "
                    "values and fill='none'. Fill gaps first "
                    "(src.gw.fill_time()) or use fill='train_median' or "
                    "'interpolate'."
                )
            X, y = X[~bad], y[~bad]
        if not len(y):
            raise ValueError(
                "No usable training pixels remain: every labelled pixel "
                f"has gaps that fill='{self.fill}' could not fill."
            )
        return X, y

    def _prepare_predict_data(self, data, require_time=True):
        """Check, audit, fill and chunk a stack for prediction.

        Returns the filled DataArray (``time`` and ``band`` in single chunks)
        and a lazy ``(y, x)`` mask of pixels that must be NaN in the output
        (nodata everywhere, or unfillable).
        """
        name = type(self).__name__
        if not self.fitted_:
            raise RuntimeError("Call fit() before predict().")
        self._check_series(data, 'predict', require_time=require_time)
        has_time = 'time' in data.dims
        if has_time != (self._n_timesteps is not None):
            raise ValueError(
                f"{name} was fitted on "
                f"{'multi-temporal' if self._n_timesteps else 'single-date'}"
                " data; the data to predict must match."
            )
        if has_time and data.sizes['time'] != self._n_timesteps:
            raise ValueError(
                f"{name} was fitted on {self._n_timesteps} time steps but "
                f"the data has {data.sizes['time']}. The fitted model only "
                "applies to series of the same length."
            )
        if data.sizes['band'] != self._n_bands:
            raise ValueError(
                f"{name} was fitted on {self._n_bands} bands but the data "
                f"has {data.sizes['band']}."
            )
        floated = to_float_with_nan(data)
        audit_input(
            floated,
            stage='predict',
            reference=self.fit_report_,
            policy=self.audit,
            fill=self.fill,
        )
        series_dims = [d for d in ('time', 'band') if d in floated.dims]
        nodata_px = floated.isnull().all(dim=series_dims)
        filled = self.filler_.transform(floated)
        residual = self.filler_.residual_mask(filled) | nodata_px
        chunks = {'band': -1}
        if has_time:
            chunks['time'] = -1
        if not hasattr(filled.data, 'dask'):
            chunks['y'] = min(512, filled.sizes['y'])
            chunks['x'] = min(512, filled.sizes['x'])
        filled = filled.chunk(chunks)
        return filled, residual

    def _finalize_prediction(self, result_da, data, residual_mask=None):
        """Wrap a ``(1, y, x)`` dask array as the ``(band, y, x)`` result."""
        result = xr.DataArray(
            result_da,
            dims=("band", "y", "x"),
            coords={"band": ["targ"], "y": data.y, "x": data.x},
        )
        for mask in (_get_nodata_mask(data), residual_mask):
            if mask is not None:
                result = xr.where(mask, np.nan, result)
        result = result.transpose('band', 'y', 'x')
        # xr.where names the result after the mask; drop it
        result.name = None
        return result.assign_attrs(**data.attrs)
