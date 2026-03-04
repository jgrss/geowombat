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


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class TestLTAE(unittest.TestCase):

    def test_ltae_temporal(self):
        """L-TAE works with time-stacked multi-date data."""
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

    def test_ltae_raises_without_time(self):
        """L-TAE raises error when data has no time dimension."""
        from geowombat.ml.dl_classifiers import LTAEClassifier

        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                clf = LTAEClassifier()
                with self.assertRaises(ValueError):
                    fit(src, clf, aoi_poly, col='lc')


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


if __name__ == '__main__':
    unittest.main()
