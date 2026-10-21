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


class TestBaseHelpers(unittest.TestCase):
    def test_to_series_layouts(self):
        from geowombat.ml._base import _to_series

        block = np.arange(2 * 3 * 4 * 5, dtype='float32').reshape(2, 3, 4, 5)
        block[0, 0, 0, 0] = np.nan
        tm = _to_series(block, 'time_major', fill_nan=False)
        bm = _to_series(block, 'band_major')
        self.assertEqual(tm.shape, (20, 2, 3))
        self.assertEqual(bm.shape, (20, 3, 2))
        np.testing.assert_array_equal(
            tm, block.transpose(2, 3, 0, 1).reshape(20, 2, 3)
        )
        self.assertTrue(np.isnan(tm[0, 0, 0]))
        self.assertEqual(bm[0, 0, 0], 0.0)  # NaN guard applied
        with self.assertRaises(ValueError):
            _to_series(block, 'bogus')

    def test_chunk_for_prediction_splits_single_block(self):
        # Finding 5: a single dask block must still be split for map_blocks
        from geowombat.ml._base import _chunk_for_prediction

        arr = xr.DataArray(
            np.zeros((2, 1, 1000, 1200), dtype='float32'),
            dims=('time', 'band', 'y', 'x'),
        ).chunk({'time': -1})
        out = _chunk_for_prediction(arr)
        self.assertEqual(out.chunks[0], (2,))
        self.assertLessEqual(max(out.chunks[2]), 512)
        self.assertLessEqual(max(out.chunks[3]), 512)
        # an already reasonably chunked array is left alone
        arr2 = arr.chunk({'time': -1, 'band': -1, 'y': 256, 'x': 256})
        out2 = _chunk_for_prediction(arr2)
        self.assertEqual(out2.chunks[2], arr2.chunks[2])

    def test_prepare_fit_data_materialises_once(self):
        # Finding 7: the training raster is read a single time
        from geowombat.ml import _base

        arr = xr.DataArray(
            np.random.default_rng(0).random((3, 2, 40, 40)).astype('float32'),
            dims=('time', 'band', 'y', 'x'),
            attrs={'cloud_masked': True},
        ).chunk({'y': 10, 'x': 10})
        clf = _base.GeoWombatDLClassifier()
        filled = clf._prepare_fit_data(arr)
        self.assertFalse(hasattr(filled.data, 'dask'))
        self.assertEqual(len(clf.fit_report_), 6)


class TestBaseExtraction(unittest.TestCase):
    def test_no_usable_training_pixels_raises(self):
        from unittest import mock

        from geowombat.ml import _base

        clf = _base.GeoWombatDLClassifier()
        clf.fill = 'interpolate'
        arr = np.full((4, 2, 5, 5), np.nan, dtype='float32')
        filled = xr.DataArray(arr, dims=('time', 'band', 'y', 'x'))
        labels = np.zeros((5, 5), dtype=int)
        labels[1, 1] = 1
        with mock.patch.object(
            _base, '_rasterize_labels', return_value=(labels, 1)
        ):
            with self.assertRaisesRegex(ValueError, 'training'):
                clf._extract_training_series(filled, filled, 'labels', 'lc')


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
                gap_label = int(src.time.values[GAP_TIME])
                y, msgs = _gw_warnings(predict, gappy, X, clf)
                y = y.compute()
        shift = [m for m in msgs if 'training' in m.lower()]
        self.assertTrue(shift, msgs)
        self.assertIn(f"time={gap_label}", shift[0])
        _assert_prediction(self, y, src)

    def test_three_d_nodata_mask_is_handled(self):
        # Findings 1 and 8: per-date nodata must not crash and must not
        # blank pixels that are valid on other dates
        with gw.config.update(ref_res=300):
            with _open_stack() as src:
                X, Xy, clf = _quiet(fit, src, self._clf(), aoi_poly, col='lc')
                mask2 = src.coords['_nodata_mask'].values
                mask3 = np.repeat(mask2[None], src.sizes['time'], axis=0)
                iy, ix = np.argwhere(~mask2)[0]
                mask3[0, iy, ix] = True  # nodata on the first date only
                src3 = src.assign_coords(
                    _nodata_mask=xr.DataArray(
                        mask3,
                        dims=('time', 'y', 'x'),
                        coords={'time': src.time, 'y': src.y, 'x': src.x},
                    )
                )
                y = _quiet(predict, src3, X, clf).compute()
        self.assertEqual(y.dims, ('band', 'y', 'x'))
        self.assertTrue(np.isfinite(y.values[0, iy, ix]))
        self.assertTrue(np.isnan(y.values[0][mask2]).all())

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
