"""Input hygiene for multi-temporal rasters.

Helpers to inspect, mask and fill missing values in ``(time, band, y, x)``
stacks before they reach a classifier. Everything here is torch-free and
scikit-learn-free so the ``gw`` accessor can expose it without the ``ml``
extra.
"""

import warnings

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    'GeoWombatInputWarning',
    'TimeSeriesFiller',
    'audit_input',
    'fill_time',
    'mask_clouds',
    'missing_report',
    'to_float_with_nan',
]

FILL_STRATEGIES = ('train_median', 'interpolate', 'zero', 'none')
AUDIT_POLICIES = ('warn', 'raise', 'off')
_PIXEL_DIMS = ('y', 'x')


class GeoWombatInputWarning(UserWarning):
    """Warning about missing, suspicious or inconsistent input values."""


def _declared_nodata(data):
    """Return the declared 'no data' value, or None."""
    if '_FillValue' in data.attrs:
        return data.attrs['_FillValue']
    nodata = data.attrs.get('nodatavals')
    if nodata is None:
        return None
    if isinstance(nodata, (tuple, list, np.ndarray)):
        if len(nodata) == 0:
            return None
        nodata = nodata[0]
    return nodata


def _nodata_mask(data):
    """Boolean DataArray of declared nodata pixels, or None."""
    nodata = _declared_nodata(data)
    if nodata is None:
        return None
    if isinstance(nodata, float) and np.isnan(nodata):
        return None
    return data == nodata


def to_float_with_nan(data):
    """Cast to float32 with declared 'no data' values set to NaN.

    Uses the ``nodatavals`` / ``_FillValue`` attributes and, when present, the
    ``_nodata_mask`` coordinate attached by :func:`geowombat.open`. Attributes
    are preserved.
    """
    attrs = dict(data.attrs)
    out = data if data.dtype == np.float32 else data.astype('float32')
    nd = _nodata_mask(data)
    if nd is not None:
        out = out.where(~nd)
    if '_nodata_mask' in data.coords:
        out = out.where(~data.coords['_nodata_mask'].astype(bool))
    out.attrs = attrs
    return out


def _series_dims(data):
    return tuple(d for d in ('time', 'band') if d in data.dims)


def missing_report(data):
    """Fraction of missing values per time step and band.

    Args:
        data (DataArray): ``(time, band, y, x)`` or ``(band, y, x)``.

    Returns:
        ``pandas.DataFrame`` with one row per ``(time, band)`` and columns
        ``time, band, n_pixels, frac_nan, frac_nodata, frac_zero,
        frac_missing``. ``frac_missing`` is NaN or nodata; ``frac_zero``
        counts exact zeros that are *not* the declared nodata value.
    """
    is_nan = data.isnull()
    nd = _nodata_mask(data)
    if nd is None:
        nd = xr.zeros_like(is_nan, dtype=bool)
    is_zero = (data == 0) & ~nd
    stats = xr.Dataset(
        {
            'frac_nan': is_nan.mean(dim=_PIXEL_DIMS),
            'frac_nodata': nd.mean(dim=_PIXEL_DIMS),
            'frac_zero': is_zero.mean(dim=_PIXEL_DIMS),
            'frac_missing': (is_nan | nd).mean(dim=_PIXEL_DIMS),
        }
    ).compute()
    keep = [c for c in stats.coords if c in _series_dims(data)]
    stats = stats.reset_coords(drop=True)[list(stats.data_vars)]
    stats = stats.drop_vars([c for c in stats.coords if c not in keep])
    df = stats.to_dataframe().reset_index()
    if 'time' not in df.columns:
        df['time'] = np.nan
    df['n_pixels'] = int(data.sizes['y'] * data.sizes['x'])
    cols = [
        'time',
        'band',
        'n_pixels',
        'frac_nan',
        'frac_nodata',
        'frac_zero',
        'frac_missing',
    ]
    return df[cols].reset_index(drop=True)


def _label(row):
    if pd.isna(row['time']):
        return f"band={row['band']!r}"
    return f"time={row['time']!r}, band={row['band']!r}"


def _pct(x):
    return f"{100 * float(x):.0f}%"


def audit_input(
    data,
    *,
    stage,
    reference=None,
    policy='warn',
    warn_missing_frac=0.05,
    warn_zero_frac=0.01,
    warn_shift_frac=0.10,
    fill=None,
):
    """Inspect a stack for missing or suspicious values and warn.

    Args:
        data (DataArray): The stack to inspect.
        stage (str): ``'fit'`` or ``'predict'``.
        reference (DataFrame): The report returned at ``fit`` time. When
            given at ``stage='predict'``, a rise in missingness relative
            to training triggers a warning.
        policy (str): ``'warn'`` (default), ``'raise'`` or ``'off'``.
        warn_missing_frac (float): Missing fraction per ``(time, band)``
            above which to warn.
        warn_zero_frac (float): Fraction of exact zeros (when zero is not
            the nodata value) above which to warn.
        warn_shift_frac (float): Increase in missing fraction relative to
            ``reference`` above which to warn.
        fill (str): The fill strategy that will be applied, for messages.

    Returns:
        ``pandas.DataFrame``: the :func:`missing_report`.

    Raises:
        ValueError: when a ``(time, band)`` slice is entirely missing at
            ``stage='fit'`` (unless ``fill='interpolate'``), or for any
            finding when ``policy='raise'``.
    """
    if policy not in AUDIT_POLICIES:
        raise ValueError(f"policy must be one of {AUDIT_POLICIES}")
    if stage not in ('fit', 'predict'):
        raise ValueError("stage must be 'fit' or 'predict'")
    report = missing_report(data)
    if policy == 'off':
        return report

    def emit(msg):
        if policy == 'raise':
            raise ValueError(msg)
        warnings.warn(msg, GeoWombatInputWarning, stacklevel=3)

    fill_hint = (
        f"Gaps will be filled with fill='{fill}'."
        if fill
        else "Fill them first (e.g. src.gw.fill_time()) or set fill= on "
        "the classifier."
    )

    if stage == 'fit' and fill != 'interpolate':
        full = report[report['frac_missing'] >= 1.0]
        if len(full):
            where = ', '.join(_label(r) for _, r in full.iterrows())
            raise ValueError(
                f"These slices are entirely missing in the training data: "
                f"{where}. No training statistic can be computed for them. "
                "Drop those time steps (data.isel(time=...)) or use "
                "fill='interpolate'."
            )

    heavy = report[report['frac_missing'] > warn_missing_frac]
    if len(heavy):
        worst = heavy.sort_values('frac_missing', ascending=False).head(5)
        where = '; '.join(
            f"{_label(r)}: {_pct(r['frac_missing'])}"
            for _, r in worst.iterrows()
        )
        emit(
            f"{len(heavy)} (time, band) slice(s) have more than "
            f"{_pct(warn_missing_frac)} missing values (NaN or nodata). "
            f"Worst: {where}. {fill_hint}"
        )

    zeros = report[report['frac_zero'] > warn_zero_frac]
    if len(zeros):
        where = '; '.join(
            f"{_label(r)}: {_pct(r['frac_zero'])}" for _, r in zeros.iterrows()
        )
        emit(
            f"Exact zeros found that are not declared as nodata ({where}). "
            "Zeros in imagery usually mean unmasked nodata or fill values. "
            "Open the data with nodata=0, or mask them, so they are not "
            "treated as real measurements."
        )

    if stage == 'predict' and reference is not None:
        ref = reference[['time', 'band', 'frac_missing']].rename(
            columns={'frac_missing': 'frac_missing_fit'}
        )
        merged = report.merge(ref, on=['time', 'band'], how='left')
        merged['frac_missing_fit'] = merged['frac_missing_fit'].fillna(0.0)
        delta = merged['frac_missing'] - merged['frac_missing_fit']
        shifted = merged[delta > warn_shift_frac]
        if len(shifted):
            where = '; '.join(
                f"{_label(r)}: {_pct(r['frac_missing_fit'])} missing at fit, "
                f"{_pct(r['frac_missing'])} now"
                for _, r in shifted.iterrows()
            )
            emit(
                "Missing data differs from the training data: "
                f"{where}. The model never saw gaps there, so predictions "
                f"in those pixels rely on the fill ({fill or 'none'}). Check "
                "that clouds and nodata were masked the same way for the "
                "training and prediction data."
            )

    if (
        stage == 'fit'
        and 'time' in data.dims
        and data.sizes['time'] >= 3
        and not data.attrs.get('cloud_masked')
        and bool((report['frac_missing'] == 0).all())
    ):
        emit(
            f"No missing values in any of the {data.sizes['time']} time "
            "steps. Multi-date optical stacks normally have cloud gaps; make "
            "sure clouds and shadows were masked (src.gw.mask_clouds()). If "
            "they were, set data.attrs['cloud_masked'] = True to silence "
            "this warning."
        )

    return report


def _nan_run_length(arr):
    """Length of the contiguous NaN run each element belongs to.

    Operates along the last axis. Valid elements get 0.
    """
    nan = np.isnan(arr)
    n = nan.shape[-1]
    fwd = np.zeros(nan.shape, dtype=np.int32)
    bwd = np.zeros(nan.shape, dtype=np.int32)
    for i in range(n):
        prev = fwd[..., i - 1] if i > 0 else 0
        fwd[..., i] = (prev + 1) * nan[..., i]
    for i in range(n - 1, -1, -1):
        nxt = bwd[..., i + 1] if i < n - 1 else 0
        bwd[..., i] = (nxt + 1) * nan[..., i]
    return np.where(nan, fwd + bwd - 1, 0).astype(np.int32)


def _interpolate_time(data, max_gap=None):
    """Linear interpolation along time with nearest-value end filling.

    ``max_gap`` is the maximum number of consecutive missing steps to fill;
    longer gaps (including gaps at either end) stay NaN.
    """
    if 'time' not in data.dims:
        raise ValueError("Interpolation requires a 'time' dimension.")
    data = data.chunk({'time': -1})
    out = data.interpolate_na(
        dim='time', method='linear', use_coordinate=False
    )
    out = out.interpolate_na(
        dim='time',
        method='nearest',
        use_coordinate=False,
        fill_value='extrapolate',
    )
    if max_gap is not None:
        run = xr.apply_ufunc(
            _nan_run_length,
            data,
            input_core_dims=[['time']],
            output_core_dims=[['time']],
            dask='parallelized',
            output_dtypes=[np.int32],
        ).transpose(*data.dims)
        out = out.where(run <= int(max_gap))
    out.attrs = dict(data.attrs)
    return out


class TimeSeriesFiller:
    """Fill missing values with training statistics or interpolation.

    Args:
        strategy (str): ``'train_median'`` (per band and time step over
            the raster seen at ``fit``), ``'interpolate'`` (linear along
            time, ends filled with the nearest value), ``'zero'`` (legacy,
            warns) or ``'none'``.
    """

    def __init__(self, strategy='train_median'):
        if strategy not in FILL_STRATEGIES:
            raise ValueError(f"strategy must be one of {FILL_STRATEGIES}")
        self.strategy = strategy
        self.fill_values_ = None

    def fit(self, data):
        if self.strategy == 'train_median':
            med = data.median(dim=_PIXEL_DIMS, skipna=True).compute()
            med = med.reset_coords(drop=True)
            self.fill_values_ = med.drop_vars(
                [c for c in med.coords if c not in _series_dims(data)]
            )
        return self

    def transform(self, data):
        if self.strategy == 'train_median':
            if self.fill_values_ is None:
                raise RuntimeError("Call fit() before transform().")
            out = data.fillna(self.fill_values_)
        elif self.strategy == 'interpolate':
            out = _interpolate_time(data)
        elif self.strategy == 'zero':
            warnings.warn(
                "Filling missing values with 0. Zeros are far outside the "
                "range of real measurements and distort scaling and model "
                "inputs; prefer fill='train_median' or 'interpolate'.",
                GeoWombatInputWarning,
                stacklevel=2,
            )
            out = data.fillna(0)
        else:
            out = data
        out.attrs = dict(data.attrs)
        return out

    @staticmethod
    def residual_mask(filled):
        """``(y, x)`` mask of pixels that still contain NaN after filling."""
        return filled.isnull().any(dim=list(_series_dims(filled)))


def fill_time(data, method='interpolate', max_gap=None):
    """Fill missing values along the time dimension.

    Args:
        data (DataArray): ``(time, band, y, x)`` stack.
        method (str): ``'interpolate'`` (linear along time, ends filled
            with the nearest value) or ``'median'`` (per band and time
            step median over the raster).
        max_gap (int): For ``'interpolate'``, the maximum number of
            consecutive missing steps to fill.

    Returns:
        ``xarray.DataArray``
    """
    if method == 'interpolate':
        return _interpolate_time(data, max_gap=max_gap)
    if method == 'median':
        return TimeSeriesFiller('train_median').fit(data).transform(data)
    raise ValueError("method must be 'interpolate' or 'median'")


def mask_clouds(data, mask, clear_values=(0,), broadcast_time=False):
    """Set cloudy pixels to NaN and tag the data as cloud masked.

    Args:
        data (DataArray): The imagery.
        mask (DataArray): Either a boolean array (``True`` = cloud) or an
            integer code array such as the output of
            :class:`geowombat.radiometry.QAMasker` or
            :func:`geowombat.radiometry.mask_s2` (a single ``band='mask'``
            dimension is dropped).
        clear_values (tuple): Codes treated as clear in an integer mask.
        broadcast_time (bool): Allow a mask without a ``time`` dimension to
            be applied to every time step of multi-temporal data.

    Returns:
        ``xarray.DataArray`` (float32, NaN where cloudy) with
        ``attrs['cloud_masked'] = True``.
    """
    if mask.dtype == bool:
        cloud = mask
    else:
        cloud = ~mask.isin(list(clear_values))
    if 'band' in cloud.dims:
        if cloud.sizes['band'] != 1:
            raise ValueError("The mask must have a single 'band'.")
        cloud = cloud.squeeze('band', drop=True)
    if 'time' in data.dims and 'time' not in cloud.dims:
        if not broadcast_time:
            raise ValueError(
                "The mask has no 'time' dimension but the data does. Pass "
                "broadcast_time=True to apply the same mask to every date."
            )
    cloud = cloud.assign_coords(
        {d: data.coords[d] for d in cloud.dims if d in data.coords}
    )
    out = to_float_with_nan(data).where(~cloud)
    out.attrs = {**data.attrs, 'cloud_masked': True}
    return out
