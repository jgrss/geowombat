import unittest
import warnings

import geopandas as gpd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xarray import DataArray as xr_da

try:
    import aeon  # noqa: F401

    AEON_AVAILABLE = True
except ImportError:
    AEON_AVAILABLE = False

import geowombat as gw
from geowombat.data import l8_224078_20200518, l8_224078_20200518_polygons
from geowombat.ml import fit, fit_predict, predict

aoi_poly = gpd.read_file(l8_224078_20200518_polygons)
aoi_poly["lc"] = LabelEncoder().fit_transform(aoi_poly.name)
aoi_poly = aoi_poly.drop(columns=["name"])

# MiniRocket needs at least 9 time steps; stack the same image for a demo.
TIME_STACK = [l8_224078_20200518] * 9


def _small_clf(**kwargs):
    from geowombat.ml.rocket_classifiers import MiniRocketClassifier

    params = dict(n_kernels=84, random_state=0)
    params.update(kwargs)
    return MiniRocketClassifier(**params)


@unittest.skipUnless(AEON_AVAILABLE, "aeon not installed")
class TestMiniRocket(unittest.TestCase):
    def test_minirocket_fit_predict(self):
        """fit_predict returns a (band, y, x) map of 1-based classes."""
        with gw.config.update(ref_res=300):
            with gw.open(TIME_STACK, stack_dim='time', nodata=0) as src:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y = fit_predict(src, _small_clf(), aoi_poly, col='lc')

        self.assertIsInstance(y, xr_da)
        self.assertEqual(y.dims, ('band', 'y', 'x'))
        self.assertEqual(list(y.band.values), ['targ'])
        # The nodata mask must not leak its name into the result
        self.assertIsNone(y.name)
        vals = y.values[np.isfinite(y.values)]
        self.assertTrue(len(vals) > 0)
        self.assertTrue(np.all(vals >= 1))
        self.assertTrue(np.all(vals <= 4))

    def test_minirocket_fit_then_predict(self):
        """Separate fit() then predict() through geowombat.ml works."""
        with gw.config.update(ref_res=300):
            with gw.open(TIME_STACK, stack_dim='time', nodata=0) as src:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    X, Xy, clf = fit(src, _small_clf(), aoi_poly, col='lc')
                    y = predict(src, X, clf)

        self.assertIsInstance(y, xr_da)
        vals = y.values[np.isfinite(y.values)]
        self.assertTrue(len(vals) > 0)

    def test_minirocket_raises_without_time(self):
        """Fit() raises ValueError when data has no time dimension."""
        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                with self.assertRaisesRegex(ValueError, "time"):
                    fit(src, _small_clf(), aoi_poly, col='lc')

    def test_minirocket_raises_too_few_timesteps(self):
        """Fit() raises ValueError with fewer than 9 time steps."""
        with gw.config.update(ref_res=300):
            with gw.open(
                [l8_224078_20200518] * 2, stack_dim='time', nodata=0
            ) as src:
                with self.assertRaisesRegex(ValueError, "9"):
                    fit(src, _small_clf(), aoi_poly, col='lc')

    def test_minirocket_predict_before_fit_raises(self):
        """Predict() before fit() raises RuntimeError."""
        with gw.config.update(ref_res=300):
            with gw.open(TIME_STACK, stack_dim='time', nodata=0) as src:
                with self.assertRaises(RuntimeError):
                    _small_clf().predict(src)

    def test_minirocket_predict_timestep_mismatch_raises(self):
        """Predict() on a different series length raises ValueError."""
        with gw.config.update(ref_res=300):
            with gw.open(TIME_STACK, stack_dim='time', nodata=0) as src:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    X, Xy, clf = fit(src, _small_clf(), aoi_poly, col='lc')
            with gw.open(
                [l8_224078_20200518] * 10, stack_dim='time', nodata=0
            ) as src10:
                with self.assertRaisesRegex(ValueError, "time steps"):
                    clf.predict(src10)

    def test_minirocket_fitted_metadata(self):
        """Fit() sets fitted_, _n_classes, _n_bands and _n_timesteps."""
        clf = _small_clf()
        self.assertFalse(clf.fitted_)
        with gw.config.update(ref_res=300):
            with gw.open(TIME_STACK, stack_dim='time', nodata=0) as src:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    X, Xy, clf = fit(src, clf, aoi_poly, col='lc')
        self.assertTrue(clf.fitted_)
        self.assertEqual(clf._n_classes, 4)
        self.assertEqual(clf._n_bands, 3)
        self.assertEqual(clf._n_timesteps, 9)

    def test_minirocket_custom_estimator(self):
        """A user-supplied sklearn estimator replaces the default ridge."""
        from sklearn.linear_model import LogisticRegression

        est = LogisticRegression(max_iter=200)
        with gw.config.update(ref_res=300):
            with gw.open(TIME_STACK, stack_dim='time', nodata=0) as src:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf = _small_clf(estimator=est)
                    y = fit_predict(src, clf, aoi_poly, col='lc')
        self.assertIs(clf._estimator, est)
        vals = y.values[np.isfinite(y.values)]
        self.assertTrue(np.all(vals >= 1))

    def test_minirocket_max_train_samples(self):
        """max_train_samples caps the number of training pixels."""
        with gw.config.update(ref_res=300):
            with gw.open(TIME_STACK, stack_dim='time', nodata=0) as src:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf = _small_clf(max_train_samples=10)
                    X, Xy, clf = fit(src, clf, aoi_poly, col='lc')
        self.assertEqual(clf._n_train_samples, 10)

    def test_minirocket_nodata_masked(self):
        """Nodata pixels are NaN in the prediction."""
        with gw.config.update(ref_res=300):
            with gw.open(TIME_STACK, stack_dim='time', nodata=0) as src:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y = fit_predict(src, _small_clf(), aoi_poly, col='lc')
                # Warped data carries source nodata in this coordinate
                nd = src.coords['_nodata_mask'].values
        self.assertTrue(nd.any())
        self.assertTrue(np.all(np.isnan(y.values[0][nd])))

    def test_minirocket_nan_input_does_not_propagate(self):
        """NaN values in the series are filled, not propagated."""
        with gw.config.update(ref_res=300):
            with gw.open(TIME_STACK, stack_dim='time', nodata=0) as src:
                src_nan = src.astype('float32').where(src > 0)
                src_nan.attrs = src.attrs
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y = fit_predict(src_nan, _small_clf(), aoi_poly, col='lc')
        vals = y.values[np.isfinite(y.values)]
        self.assertTrue(len(vals) > 0)

    def test_minirocket_exported_from_ml(self):
        """MiniRocketClassifier is exported from geowombat.ml."""
        import geowombat.ml as ml

        self.assertIn('MiniRocketClassifier', ml.__all__)
        from geowombat.ml import MiniRocketClassifier  # noqa: F401


if __name__ == '__main__':
    unittest.main()
