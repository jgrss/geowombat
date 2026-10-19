"""Tests for TempCNNClassifier and CNNBiLSTMClassifier."""

import unittest
import warnings

import geopandas as gpd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xarray import DataArray as xr_da

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import geowombat as gw
from geowombat.data import l8_224078_20200518, l8_224078_20200518_polygons
from geowombat.ml import fit, fit_predict, predict

aoi_poly = gpd.read_file(l8_224078_20200518_polygons)
aoi_poly["lc"] = LabelEncoder().fit_transform(aoi_poly.name)
aoi_poly = aoi_poly.drop(columns=["name"])

TIME_STACK = [l8_224078_20200518, l8_224078_20200518]


def _open_stack(n=2):
    return gw.open([l8_224078_20200518] * n, stack_dim='time', nodata=0)


def _quiet(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return func(*args, **kwargs)


class _SequenceClassifierTests:
    """Shared tests; subclasses set ``make`` to build a small classifier."""

    def make(self, **kwargs):
        raise NotImplementedError

    def test_fit_predict(self):
        with gw.config.update(ref_res=300):
            with _open_stack() as src:
                y = _quiet(fit_predict, src, self.make(), aoi_poly, col='lc')
                y = y.compute()
                nd = src.coords['_nodata_mask'].values
        self.assertIsInstance(y, xr_da)
        self.assertEqual(y.dims, ('band', 'y', 'x'))
        self.assertIsNone(y.name)
        vals = y.values[0]
        self.assertTrue(np.isfinite(vals[~nd]).all())
        self.assertTrue(np.all(vals[~nd] >= 1))
        self.assertTrue(np.all(vals[~nd] <= 4))
        self.assertTrue(np.isnan(vals[nd]).all())

    def test_fit_then_predict(self):
        with gw.config.update(ref_res=300):
            with _open_stack() as src:
                X, Xy, clf = _quiet(fit, src, self.make(), aoi_poly, col='lc')
                y = predict(src, X, clf).compute()
        self.assertTrue(clf.fitted_)
        self.assertEqual(clf._n_classes, 4)
        self.assertEqual(clf._n_bands, 3)
        self.assertEqual(clf._n_timesteps, 2)
        self.assertEqual(len(clf._feat_mean), 3)
        self.assertTrue(len(y.values[np.isfinite(y.values)]) > 0)

    def test_raises_without_time(self):
        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                with self.assertRaisesRegex(ValueError, "time"):
                    _quiet(fit, src, self.make(), aoi_poly, col='lc')

    def test_predict_before_fit_raises(self):
        with gw.config.update(ref_res=300):
            with _open_stack() as src:
                with self.assertRaises(RuntimeError):
                    self.make().predict(src)

    def test_predict_timestep_mismatch_raises(self):
        with gw.config.update(ref_res=300):
            with _open_stack() as src:
                X, Xy, clf = _quiet(fit, src, self.make(), aoi_poly, col='lc')
            with _open_stack(3) as src3:
                with self.assertRaisesRegex(ValueError, "time steps"):
                    clf.predict(src3)

    def test_seed_makes_predictions_reproducible(self):
        with gw.config.update(ref_res=300):
            with _open_stack() as src:
                y1 = _quiet(
                    fit_predict, src, self.make(seed=0), aoi_poly, col='lc'
                ).compute()
                y2 = _quiet(
                    fit_predict, src, self.make(seed=0), aoi_poly, col='lc'
                ).compute()
        np.testing.assert_array_equal(y1.values, y2.values)

    def test_early_stopping_and_options_run(self):
        with gw.config.update(ref_res=300):
            with _open_stack() as src:
                clf = self.make(
                    max_epochs=6,
                    val_fraction=0.3,
                    patience=1,
                    loss='focal',
                    class_weight='balanced',
                    scheduler='cosine',
                    weight_decay=1e-4,
                    seed=1,
                )
                X, Xy, clf = _quiet(fit, src, clf, aoi_poly, col='lc')
        self.assertIsNotNone(clf.best_epoch_)
        self.assertGreaterEqual(clf.best_epoch_, 1)
        self.assertLessEqual(clf.best_epoch_, 6)

    def test_bad_option_values_raise(self):
        with gw.config.update(ref_res=300):
            with _open_stack() as src:
                with self.assertRaises(ValueError):
                    _quiet(
                        fit, src, self.make(loss='bogus'), aoi_poly, col='lc'
                    )
                with self.assertRaises(ValueError):
                    _quiet(
                        fit, src, self.make(fill='bogus'), aoi_poly, col='lc'
                    )


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class TestTempCNN(_SequenceClassifierTests, unittest.TestCase):
    def make(self, **kwargs):
        from geowombat.ml import TempCNNClassifier

        params = dict(max_epochs=2, hidden=(8, 16, 16), verbose=0)
        params.update(kwargs)
        return TempCNNClassifier(**params)

    def test_lone_last_batch_does_not_crash(self):
        # C2: 257 samples with batch_size 256 leaves a batch of one
        import torch

        from geowombat.ml._torch_base import _train_sequence_model
        from geowombat.ml.sequence_classifiers import _TempCNNModule

        X = np.random.default_rng(0).normal(size=(257, 4, 3)).astype('float32')
        y = np.random.default_rng(0).integers(0, 2, 257).astype('int64')
        mod = _TempCNNModule(3, 2, hidden=(8, 8, 8))
        _train_sequence_model(
            mod,
            X,
            y,
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.AdamW(mod.parameters()),
            max_epochs=1,
            batch_size=256,
            device=torch.device('cpu'),
        )

    def test_module_shape(self):
        from geowombat.ml.sequence_classifiers import _TempCNNModule

        mod = _TempCNNModule(in_channels=3, n_classes=4, hidden=(8, 16, 16))
        out = mod(torch.zeros(5, 9, 3))
        self.assertEqual(tuple(out.shape), (5, 4))


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class TestCNNBiLSTM(_SequenceClassifierTests, unittest.TestCase):
    def make(self, **kwargs):
        from geowombat.ml import CNNBiLSTMClassifier

        params = dict(max_epochs=2, conv_filters=8, lstm_hidden=8, verbose=0)
        params.update(kwargs)
        return CNNBiLSTMClassifier(**params)

    def test_module_shape(self):
        from geowombat.ml.sequence_classifiers import _CNNBiLSTMModule

        mod = _CNNBiLSTMModule(
            in_channels=3, n_classes=4, conv_filters=8, lstm_hidden=8
        )
        out = mod(torch.zeros(5, 9, 3))
        self.assertEqual(tuple(out.shape), (5, 4))


if __name__ == '__main__':
    unittest.main()
