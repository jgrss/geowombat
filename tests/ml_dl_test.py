import unittest
import warnings

import geopandas as gpd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xarray import DataArray as xr_da

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from pytorch_tabnet.tab_model import TabNetClassifier as _TabNet
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False

try:
    import segmentation_models_pytorch
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False

import geowombat as gw
from geowombat.data import (
    l8_224078_20200518,
    l8_224078_20200518_polygons,
)
from geowombat.ml import fit, fit_predict, predict

aoi_poly = gpd.read_file(l8_224078_20200518_polygons)
aoi_poly["lc"] = LabelEncoder().fit_transform(aoi_poly.name)
aoi_poly = aoi_poly.drop(columns=["name"])


# -------------------------------------------------------------------
# TabNet
# -------------------------------------------------------------------

@unittest.skipUnless(
    TORCH_AVAILABLE and TABNET_AVAILABLE,
    "torch/pytorch-tabnet not installed",
)
class TestTabNet(unittest.TestCase):

    def test_tabnet_fit_predict(self):
        """TabNet fit_predict produces valid xarray output."""
        from geowombat.ml.dl_classifiers import TabNetClassifier

        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y = fit_predict(
                        src,
                        TabNetClassifier(
                            max_epochs=3, verbose=0,
                        ),
                        aoi_poly,
                        col='lc',
                    )

        self.assertIsInstance(y, xr_da)
        self.assertIn('band', y.dims)
        self.assertIn('y', y.dims)
        self.assertIn('x', y.dims)
        # Check prediction values are valid (1-based or nan)
        vals = y.values[np.isfinite(y.values)]
        self.assertTrue(len(vals) > 0)
        self.assertTrue(np.all(vals >= 1))

    def test_tabnet_fit_then_predict(self):
        """Separate fit() then predict() works for TabNet."""
        from geowombat.ml.dl_classifiers import TabNetClassifier

        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf = TabNetClassifier(
                        max_epochs=3, verbose=0,
                    )
                    X, Xy, clf = fit(
                        src, clf, aoi_poly, col='lc',
                    )
                    y = predict(src, X, clf)

        self.assertIsInstance(y, xr_da)
        vals = y.values[np.isfinite(y.values)]
        self.assertTrue(len(vals) > 0)

    def test_tabnet_fitted_flag(self):
        """TabNet sets fitted_ attribute after fit()."""
        from geowombat.ml.dl_classifiers import TabNetClassifier

        clf = TabNetClassifier(max_epochs=2, verbose=0)
        self.assertFalse(clf.fitted_)

        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    X, Xy, clf = fit(
                        src, clf, aoi_poly, col='lc',
                    )
        self.assertTrue(clf.fitted_)
        self.assertIsNotNone(clf._n_classes)
        self.assertEqual(clf._n_classes, 4)

    def test_tabnet_predict_before_fit_raises(self):
        """TabNet raises RuntimeError if predict() before fit()."""
        from geowombat.ml.dl_classifiers import TabNetClassifier

        clf = TabNetClassifier()
        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                with self.assertRaises(RuntimeError):
                    clf.predict(src)

    def test_tabnet_multiple_classes(self):
        """TabNet predicts more than one class (not collapsed)."""
        from geowombat.ml.dl_classifiers import TabNetClassifier

        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y = fit_predict(
                        src,
                        TabNetClassifier(
                            max_epochs=20, verbose=0,
                        ),
                        aoi_poly,
                        col='lc',
                    )
        vals = y.values[np.isfinite(y.values)]
        unique = np.unique(vals)
        self.assertGreater(
            len(unique), 1,
            f"TabNet collapsed to single class: {unique}",
        )

    def test_tabnet_normalization_stored(self):
        """TabNet stores feature normalization stats after fit."""
        from geowombat.ml.dl_classifiers import TabNetClassifier

        clf = TabNetClassifier(max_epochs=2, verbose=0)
        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    X, Xy, clf = fit(
                        src, clf, aoi_poly, col='lc',
                    )
        self.assertIsNotNone(clf._feat_mean)
        self.assertIsNotNone(clf._feat_std)
        self.assertEqual(len(clf._feat_mean), src.sizes['band'])


# -------------------------------------------------------------------
# L-TAE
# -------------------------------------------------------------------

@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class TestLTAE(unittest.TestCase):

    def test_ltae_fit_predict(self):
        """L-TAE fit_predict works with time-stacked data."""
        from geowombat.ml.dl_classifiers import LTAEClassifier

        with gw.config.update(ref_res=300):
            with gw.open(
                [l8_224078_20200518, l8_224078_20200518],
                stack_dim='time',
                nodata=0,
            ) as src:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf = LTAEClassifier(
                        max_epochs=3, verbose=0, d_model=32,
                        d_k=8, n_head=2,
                    )
                    y = fit_predict(
                        src, clf, aoi_poly, col='lc',
                    )

        self.assertIsInstance(y, xr_da)
        self.assertIn('band', y.dims)
        self.assertIn('y', y.dims)
        self.assertIn('x', y.dims)
        vals = y.values[np.isfinite(y.values)]
        self.assertTrue(len(vals) > 0)
        self.assertTrue(np.all(vals >= 1))

    def test_ltae_fit_then_predict(self):
        """Separate fit() then predict() works for L-TAE."""
        from geowombat.ml.dl_classifiers import LTAEClassifier

        with gw.config.update(ref_res=300):
            with gw.open(
                [l8_224078_20200518, l8_224078_20200518],
                stack_dim='time',
                nodata=0,
            ) as src:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf = LTAEClassifier(
                        max_epochs=3, verbose=0, d_model=32,
                        d_k=8, n_head=2,
                    )
                    X, Xy, clf = fit(
                        src, clf, aoi_poly, col='lc',
                    )
                    y = predict(src, X, clf)

        self.assertIsInstance(y, xr_da)
        vals = y.values[np.isfinite(y.values)]
        self.assertTrue(len(vals) > 0)

    def test_ltae_raises_without_time(self):
        """L-TAE raises error when data has no time dimension."""
        from geowombat.ml.dl_classifiers import LTAEClassifier

        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                clf = LTAEClassifier()
                with self.assertRaises(ValueError):
                    fit(src, clf, aoi_poly, col='lc')

    def test_ltae_fitted_flag_and_metadata(self):
        """L-TAE sets fitted_, n_classes, n_bands, n_timesteps."""
        from geowombat.ml.dl_classifiers import LTAEClassifier

        clf = LTAEClassifier(
            max_epochs=2, verbose=0, d_model=32,
            d_k=8, n_head=2,
        )
        self.assertFalse(clf.fitted_)

        with gw.config.update(ref_res=300):
            with gw.open(
                [l8_224078_20200518, l8_224078_20200518],
                stack_dim='time',
                nodata=0,
            ) as src:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    X, Xy, clf = fit(
                        src, clf, aoi_poly, col='lc',
                    )

        self.assertTrue(clf.fitted_)
        self.assertEqual(clf._n_classes, 4)
        self.assertEqual(clf._n_bands, 3)
        self.assertEqual(clf._n_timesteps, 2)

    def test_ltae_predict_before_fit_raises(self):
        """L-TAE raises RuntimeError if predict() before fit()."""
        from geowombat.ml.dl_classifiers import LTAEClassifier

        clf = LTAEClassifier()
        with gw.config.update(ref_res=300):
            with gw.open(
                [l8_224078_20200518, l8_224078_20200518],
                stack_dim='time',
                nodata=0,
            ) as src:
                with self.assertRaises(RuntimeError):
                    clf.predict(src)

    def test_ltae_predict_without_time_raises(self):
        """L-TAE predict() raises error for non-temporal data."""
        from geowombat.ml.dl_classifiers import LTAEClassifier

        clf = LTAEClassifier(
            max_epochs=2, verbose=0, d_model=32,
            d_k=8, n_head=2,
        )
        with gw.config.update(ref_res=300):
            with gw.open(
                [l8_224078_20200518, l8_224078_20200518],
                stack_dim='time',
                nodata=0,
            ) as src:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    X, Xy, clf = fit(
                        src, clf, aoi_poly, col='lc',
                    )

            # Try predicting on non-temporal data
            with gw.open(l8_224078_20200518, nodata=0) as src2:
                with self.assertRaises(ValueError):
                    clf.predict(src2)

    def test_ltae_normalization_stored(self):
        """L-TAE stores per-band normalization stats after fit."""
        from geowombat.ml.dl_classifiers import LTAEClassifier

        clf = LTAEClassifier(
            max_epochs=2, verbose=0, d_model=32,
            d_k=8, n_head=2,
        )
        with gw.config.update(ref_res=300):
            with gw.open(
                [l8_224078_20200518, l8_224078_20200518],
                stack_dim='time',
                nodata=0,
            ) as src:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    X, Xy, clf = fit(
                        src, clf, aoi_poly, col='lc',
                    )
        self.assertIsNotNone(clf._feat_mean)
        self.assertIsNotNone(clf._feat_std)
        # Per-band stats: shape should be (n_bands,)
        self.assertEqual(len(clf._feat_mean), 3)


# -------------------------------------------------------------------
# TorchGeo
# -------------------------------------------------------------------

@unittest.skipUnless(
    TORCH_AVAILABLE and SMP_AVAILABLE,
    "torch/segmentation-models-pytorch not installed",
)
class TestTorchGeo(unittest.TestCase):

    def test_unet_fit_predict(self):
        """TorchGeo U-Net fit_predict with small patches."""
        from geowombat.ml.dl_classifiers import TorchGeoClassifier

        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf = TorchGeoClassifier(
                        model='unet',
                        backbone='resnet18',
                        weights=None,
                        patch_size=16,
                        max_patches=20,
                        max_epochs=2,
                        batch_size=4,
                        verbose=0,
                    )
                    y = fit_predict(
                        src, clf, aoi_poly, col='lc',
                    )

        self.assertIsInstance(y, xr_da)
        self.assertIn('band', y.dims)
        vals = y.values[np.isfinite(y.values)]
        self.assertTrue(len(vals) > 0)
        self.assertTrue(np.all(vals >= 1))

    def test_unet_fit_then_predict(self):
        """Separate fit() then predict() works for U-Net."""
        from geowombat.ml.dl_classifiers import TorchGeoClassifier

        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf = TorchGeoClassifier(
                        model='unet',
                        backbone='resnet18',
                        weights=None,
                        patch_size=16,
                        max_patches=20,
                        max_epochs=2,
                        batch_size=4,
                        verbose=0,
                    )
                    X, Xy, clf = fit(
                        src, clf, aoi_poly, col='lc',
                    )
                    y = predict(src, X, clf)

        self.assertIsInstance(y, xr_da)
        vals = y.values[np.isfinite(y.values)]
        self.assertTrue(len(vals) > 0)

    def test_torchgeo_fitted_flag(self):
        """TorchGeo sets fitted_ and n_classes after fit."""
        from geowombat.ml.dl_classifiers import TorchGeoClassifier

        clf = TorchGeoClassifier(
            model='unet', backbone='resnet18',
            weights=None, patch_size=16,
            max_patches=10, max_epochs=1,
            batch_size=4, verbose=0,
        )
        self.assertFalse(clf.fitted_)

        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    X, Xy, clf = fit(
                        src, clf, aoi_poly, col='lc',
                    )
        self.assertTrue(clf.fitted_)
        self.assertEqual(clf._n_classes, 4)

    def test_torchgeo_predict_before_fit_raises(self):
        """TorchGeo raises RuntimeError if predict() before fit()."""
        from geowombat.ml.dl_classifiers import TorchGeoClassifier

        clf = TorchGeoClassifier()
        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                with self.assertRaises(RuntimeError):
                    clf.predict(src)

    def test_torchgeo_expected_bands_no_weights(self):
        """expected_bands returns None in_chans when no weights."""
        from geowombat.ml.dl_classifiers import TorchGeoClassifier

        clf = TorchGeoClassifier(weights=None)
        info = clf.expected_bands
        self.assertIsNone(info['in_chans'])

    def test_deeplabv3plus(self):
        """DeepLabV3+ architecture works with fit_predict."""
        from geowombat.ml.dl_classifiers import TorchGeoClassifier

        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf = TorchGeoClassifier(
                        model='deeplabv3+',
                        backbone='resnet18',
                        weights=None,
                        patch_size=16,
                        max_patches=10,
                        max_epochs=1,
                        batch_size=4,
                        verbose=0,
                    )
                    y = fit_predict(
                        src, clf, aoi_poly, col='lc',
                    )

        self.assertIsInstance(y, xr_da)
        vals = y.values[np.isfinite(y.values)]
        self.assertTrue(len(vals) > 0)

    def test_non_chunked_data(self):
        """TorchGeo handles non-chunked (computed) data."""
        from geowombat.ml.dl_classifiers import TorchGeoClassifier

        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                # Force compute to remove dask chunks
                src_computed = src.compute()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf = TorchGeoClassifier(
                        model='unet',
                        backbone='resnet18',
                        weights=None,
                        patch_size=16,
                        max_patches=10,
                        max_epochs=1,
                        batch_size=4,
                        verbose=0,
                    )
                    y = fit_predict(
                        src_computed, clf, aoi_poly,
                        col='lc',
                    )

        self.assertIsInstance(y, xr_da)
        vals = y.values[np.isfinite(y.values)]
        self.assertTrue(len(vals) > 0)


# -------------------------------------------------------------------
# Shared / integration
# -------------------------------------------------------------------

@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class TestDLIntegration(unittest.TestCase):

    def test_dl_classifiers_importable_from_ml(self):
        """DL classifiers are importable from geowombat.ml."""
        from geowombat.ml import (
            TabNetClassifier,
            LTAEClassifier,
            TorchGeoClassifier,
        )
        self.assertTrue(
            hasattr(TabNetClassifier, '_is_gw_dl_classifier')
        )
        self.assertTrue(
            hasattr(LTAEClassifier, '_is_gw_dl_classifier')
        )
        self.assertTrue(
            hasattr(TorchGeoClassifier, '_is_gw_dl_classifier')
        )

    def test_resolve_device_auto(self):
        """_resolve_device('auto') returns a valid torch.device."""
        from geowombat.ml.dl_classifiers import _resolve_device

        dev = _resolve_device('auto')
        self.assertIsInstance(dev, torch.device)

    def test_resolve_device_cpu(self):
        """_resolve_device('cpu') returns CPU device."""
        from geowombat.ml.dl_classifiers import _resolve_device

        dev = _resolve_device('cpu')
        self.assertEqual(dev.type, 'cpu')

    def test_rasterize_labels_with_computed_data(self):
        """_rasterize_labels auto-chunks non-dask data."""
        from geowombat.ml.dl_classifiers import _rasterize_labels

        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                src_computed = src.compute()
                label_arr, n_classes = _rasterize_labels(
                    src_computed, aoi_poly, 'lc',
                )

        self.assertEqual(n_classes, 4)
        self.assertEqual(label_arr.ndim, 2)
        self.assertTrue(np.any(label_arr > 0))


if __name__ == '__main__':
    unittest.main()
