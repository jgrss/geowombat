"""Integration tests: input audit and fill inside the classifiers."""

import unittest
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import LabelEncoder

try:
    import aeon  # noqa: F401

    AEON_AVAILABLE = True
except ImportError:
    AEON_AVAILABLE = False

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pytorch_tabnet  # noqa: F401

    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False

import geowombat as gw
from geowombat.core.timeseries import GeoWombatInputWarning
from geowombat.data import l8_224078_20200518, l8_224078_20200518_polygons
from geowombat.ml import fit, fit_predict, predict

aoi_poly = gpd.read_file(l8_224078_20200518_polygons)
aoi_poly["lc"] = LabelEncoder().fit_transform(aoi_poly.name)
aoi_poly = aoi_poly.drop(columns=["name"])

TIME_STACK = [l8_224078_20200518] * 9
GAP_TIME = 4


def _open_stack():
    return gw.open(TIME_STACK, stack_dim='time', nodata=0)


def _with_gap(src, time_index=GAP_TIME, frac=0.5, seed=0):
    """Return a copy with ``frac`` of the pixels NaN at one time step."""
    rng = np.random.default_rng(seed)
    hit = rng.random((src.sizes['y'], src.sizes['x'])) < frac
    pix = xr.DataArray(hit, dims=('y', 'x'), coords={'y': src.y, 'x': src.x})
    if 'time' in src.dims:
        step = xr.DataArray(
            np.arange(src.sizes['time']) == time_index,
            dims='time',
            coords={'time': src.time},
        )
        cond = step & pix
    else:
        # Single image: a pixel missing in every band is footprint nodata,
        # so only knock out the first band to create a real gap.
        band = xr.DataArray(
            np.arange(src.sizes['band']) == 0,
            dims='band',
            coords={'band': src.band},
        )
        cond = band & pix
    out = src.where(~cond)
    out.attrs = dict(src.attrs)
    return out


def _gw_warnings(func, *args, **kwargs):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = func(*args, **kwargs)
    msgs = [
        str(w.message)
        for w in rec
        if issubclass(w.category, GeoWombatInputWarning)
    ]
    return out, msgs


def _quiet(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return func(*args, **kwargs)


def _assert_prediction(test, y, src):
    test.assertEqual(y.dims, ('band', 'y', 'x'))
    test.assertIsNone(y.name)
    test.assertEqual(list(y.band.values), ['targ'])
    vals = y.values
    nd = src.coords['_nodata_mask'].values
    test.assertTrue(np.isfinite(vals[0][~nd]).all())
    test.assertTrue(np.all(vals[0][~nd] >= 1))
    test.assertTrue(np.isnan(vals[0][nd]).all())


@unittest.skipUnless(AEON_AVAILABLE, "aeon not installed")
class TestMiniRocketHygiene(unittest.TestCase):
    def _clf(self, **kwargs):
        from geowombat.ml import MiniRocketClassifier

        params = dict(n_kernels=84, random_state=0)
        params.update(kwargs)
        return MiniRocketClassifier(**params)

    def test_shift_warning_and_finite_output(self):
        with gw.config.update(ref_res=300):
            with _open_stack() as src:
                X, Xy, clf = _quiet(fit, src, self._clf(), aoi_poly, col='lc')
                gappy = _with_gap(src)
                y, msgs = _gw_warnings(predict, gappy, X, clf)
                y = y.compute()
        shift = [m for m in msgs if 'training' in m.lower()]
        self.assertTrue(shift, msgs)
        self.assertIn(str(GAP_TIME), shift[0])
        _assert_prediction(self, y, src)

    def test_fit_report_and_filler_are_stored(self):
        with gw.config.update(ref_res=300):
            with _open_stack() as src:
                X, Xy, clf = _quiet(fit, src, self._clf(), aoi_poly, col='lc')
        self.assertIsInstance(clf.fit_report_, pd.DataFrame)
        self.assertEqual(len(clf.fit_report_), 9 * 3)
        self.assertEqual(clf.filler_.fill_values_.dims, ('time', 'band'))
        self.assertTrue(np.isfinite(clf.filler_.fill_values_.values).all())

    def test_fill_none_raises_on_training_gaps(self):
        with gw.config.update(ref_res=300):
            with _open_stack() as src:
                gappy = _with_gap(src)
                with self.assertRaisesRegex(ValueError, "fill"):
                    _quiet(
                        fit, gappy, self._clf(fill='none'), aoi_poly, col='lc'
                    )

    def test_fill_interpolate_runs(self):
        with gw.config.update(ref_res=300):
            with _open_stack() as src:
                gappy = _with_gap(src)
                y = _quiet(
                    fit_predict,
                    gappy,
                    self._clf(fill='interpolate'),
                    aoi_poly,
                    col='lc',
                ).compute()
        _assert_prediction(self, y, src)

    def test_audit_off_is_silent(self):
        with gw.config.update(ref_res=300):
            with _open_stack() as src:
                gappy = _with_gap(src)
                y, msgs = _gw_warnings(
                    fit_predict,
                    gappy,
                    self._clf(audit='off'),
                    aoi_poly,
                    col='lc',
                )
                y.compute()
        self.assertEqual(msgs, [])

    def test_unmasked_cloud_warning_on_clean_stack(self):
        with gw.config.update(ref_res=300):
            with _open_stack() as src:
                _, msgs = _gw_warnings(
                    fit, src, self._clf(), aoi_poly, col='lc'
                )
        self.assertTrue(any('cloud' in m.lower() for m in msgs), msgs)


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class TestLTAEHygiene(unittest.TestCase):
    def _clf(self, **kwargs):
        from geowombat.ml.dl_classifiers import LTAEClassifier

        params = dict(max_epochs=2, d_model=32, d_k=8, n_head=2, verbose=0)
        params.update(kwargs)
        return LTAEClassifier(**params)

    def test_shift_warning_and_finite_output(self):
        with gw.config.update(ref_res=300):
            with _open_stack() as src:
                X, Xy, clf = _quiet(fit, src, self._clf(), aoi_poly, col='lc')
                gappy = _with_gap(src)
                y, msgs = _gw_warnings(predict, gappy, X, clf)
                y = y.compute()
        self.assertTrue(any('training' in m.lower() for m in msgs), msgs)
        _assert_prediction(self, y, src)

    def test_training_gaps_do_not_poison_normalisation(self):
        with gw.config.update(ref_res=300):
            with _open_stack() as src:
                gappy = _with_gap(src)
                X, Xy, clf = _quiet(
                    fit, gappy, self._clf(), aoi_poly, col='lc'
                )
        self.assertTrue(np.isfinite(clf._feat_mean).all())
        self.assertTrue(np.isfinite(clf._feat_std).all())

    def test_fill_none_raises_on_training_gaps(self):
        with gw.config.update(ref_res=300):
            with _open_stack() as src:
                gappy = _with_gap(src)
                with self.assertRaisesRegex(ValueError, "fill"):
                    _quiet(
                        fit, gappy, self._clf(fill='none'), aoi_poly, col='lc'
                    )


@unittest.skipUnless(
    TORCH_AVAILABLE and TABNET_AVAILABLE, "torch/pytorch-tabnet not installed"
)
class TestTabNetHygiene(unittest.TestCase):
    def _clf(self, **kwargs):
        from geowombat.ml.dl_classifiers import TabNetClassifier

        params = dict(max_epochs=3, verbose=0)
        params.update(kwargs)
        return TabNetClassifier(**params)

    def test_single_image_shift_warning(self):
        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                X, Xy, clf = _quiet(fit, src, self._clf(), aoi_poly, col='lc')
                gappy = _with_gap(src, frac=0.3)
                y, msgs = _gw_warnings(predict, gappy, X, clf)
                y = y.compute()
        self.assertTrue(any('training' in m.lower() for m in msgs), msgs)
        _assert_prediction(self, y, src)

    def test_interpolate_requires_time(self):
        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                with self.assertRaisesRegex(ValueError, "time"):
                    _quiet(
                        fit,
                        src,
                        self._clf(fill='interpolate'),
                        aoi_poly,
                        col='lc',
                    )


if __name__ == '__main__':
    unittest.main()
