import unittest

import geowombat as gw
from geowombat.data import (
    l8_224078_20200518_points,
    l8_224078_20200518_polygons,
    l8_224078_20200518,
)
from geowombat.ml import fit, predict, fit_predict

import numpy as np
import geopandas as gpd
from sklearn.preprocessing import LabelEncoder
from sklearn_xarray.preprocessing import Featurizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB


aoi_point = gpd.read_file(l8_224078_20200518_points)
aoi_point["lc"] = LabelEncoder().fit_transform(aoi_point.name)
aoi_point = aoi_point.drop(columns=["name"])

aoi_poly = gpd.read_file(l8_224078_20200518_polygons)
aoi_poly["lc"] = LabelEncoder().fit_transform(aoi_poly.name)
aoi_poly = aoi_poly.drop(columns=["name"])


pl_wo_feat = Pipeline(
    [
        # ("featurizer", Featurizer()),
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        ("clf", GaussianNB()),
    ]
)

pl_w_feat = Pipeline(
    [
        ("featurizer", Featurizer()),
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        ("clf", GaussianNB()),
    ]
)


class TestConfig(unittest.TestCase):
    def test_fitpredict_eq_fit_predict_point(self):

        with gw.config.update(
            ref_res=300,
        ):
            with gw.open(l8_224078_20200518) as src:
                X, clf = fit(src, aoi_point, pl_w_feat, col="lc")
                y1 = predict(src, X, clf)
        with gw.config.update(
            ref_res=300,
        ):
            with gw.open(l8_224078_20200518) as src:
                y2 = fit_predict(src, aoi_point, pl_w_feat, col="lc")

        self.assertTrue(np.allclose(y1.values, y2.values))

    def test_w_wo_featurizer_point(self):

        with gw.config.update(
            ref_res=300,
        ):
            with gw.open(l8_224078_20200518) as src:
                y1 = fit_predict(src, aoi_point, pl_w_feat, col="lc")
        with gw.config.update(
            ref_res=300,
        ):
            with gw.open(l8_224078_20200518) as src:
                y2 = fit_predict(src, aoi_point, pl_wo_feat, col="lc")

        self.assertTrue(np.allclose(y1.values, y2.values))

    def test_fitpredict_eq_fit_predict_poly(self):

        with gw.config.update(
            ref_res=300,
        ):
            with gw.open(l8_224078_20200518) as src:
                X, clf = fit(src, aoi_poly, pl_w_feat, col="lc")
                y1 = predict(src, X, clf)
        with gw.config.update(
            ref_res=300,
        ):
            with gw.open(l8_224078_20200518) as src:
                y2 = fit_predict(src, aoi_poly, pl_w_feat, col="lc")

        self.assertTrue(np.allclose(y1.values, y2.values))

    def test_w_wo_featurizer_poly(self):

        with gw.config.update(
            ref_res=300,
        ):
            with gw.open(l8_224078_20200518) as src:
                y1 = fit_predict(src, aoi_poly, pl_w_feat, col="lc")
        with gw.config.update(
            ref_res=300,
        ):
            with gw.open(l8_224078_20200518) as src:
                y2 = fit_predict(src, aoi_poly, pl_wo_feat, col="lc")

        self.assertTrue(np.allclose(y1.values, y2.values))


if __name__ == "__main__":
    unittest.main()
