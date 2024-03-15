import unittest

import geowombat as gw
from geowombat.data import (
    l8_224078_20200518_points,
    l8_224078_20200518_polygons,
    l8_224078_20200518,
    l8_224078_20200518_B2,
)

import numpy as np
import geopandas as gpd
from sklearn.preprocessing import LabelEncoder
from pandas.testing import assert_frame_equal

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

aoi = gpd.read_file(l8_224078_20200518_points)
aoi["id"] = LabelEncoder().fit_transform(aoi.name)
aoi = aoi.drop(columns=["name"])

l8_224078_20200518_B2_values = np.array(
    [7966, 8030, 7561, 8302, 8277, 7398], dtype="float64"
)

l8_224078_20200518_values = np.array(
    [
        [7966, 8030, 7561, 8302, 8277, 7398],
        [7326, 7490, 6874, 8202, 7982, 6711],
        [6254, 8080, 6106, 8111, 7341, 6007],
    ],
    dtype="float64",
)


class TestExtract(unittest.TestCase):
    def test_single_image_single_band(self):
        with gw.open(l8_224078_20200518_B2) as src:
            df = gw.extract(src, aoi, band_names=["blue"])

        self.assertTrue(
            np.allclose(df.blue.values, l8_224078_20200518_B2_values)
        )

    def test_single_image_multi_band(self):
        with gw.open(l8_224078_20200518) as src:
            df = gw.extract(src, aoi, band_names=["blue", "green", "red"])

        self.assertTrue(
            np.allclose(df.blue.values, l8_224078_20200518_values[0, :])
        )
        self.assertTrue(
            np.allclose(df.green.values, l8_224078_20200518_values[1, :])
        )
        self.assertTrue(
            np.allclose(df.red.values, l8_224078_20200518_values[2, :])
        )

    def test_multi_image_single_band(self):
        with gw.open([l8_224078_20200518_B2, l8_224078_20200518_B2]) as src:
            df = gw.extract(src, aoi, band_names=["blue"])

        self.assertTrue(
            np.allclose(df.t1_blue.values, l8_224078_20200518_B2_values)
        )
        self.assertTrue(
            np.allclose(df.t2_blue.values, l8_224078_20200518_B2_values)
        )

    @unittest.skipUnless(
        RAY_AVAILABLE, "Ray is not available - skipping ray extract"
    )  #
    def test_clients_all_same_df(self):
        with gw.open(l8_224078_20200518_B2) as src:
            ray.init()
            df_ray = gw.extract(
                src,
                aoi,
                band_names=["blue", "green", "red"],
                use_ray_client=True,
            )
            ray.shutdown()
            df_none = gw.extract(
                src,
                aoi,
                band_names=["blue", "green", "red"],
                use_client=False,
            )
            df_true = gw.extract(
                src,
                aoi,
                band_names=["blue", "green", "red"],
                use_client=True,
            )
        self.assertTrue(
            assert_frame_equal(df_ray, df_none, check_exact=True) == None
        )
        self.assertTrue(
            assert_frame_equal(df_ray, df_true, check_exact=True) == None
        )

    def test_multi_image_multi_band(self):
        with gw.open([l8_224078_20200518, l8_224078_20200518]) as src:
            df = gw.extract(src, aoi, band_names=["blue", "green", "red"])

        self.assertTrue(
            np.allclose(df.t1_blue.values, l8_224078_20200518_values[0, :])
        )
        self.assertTrue(
            np.allclose(df.t1_green.values, l8_224078_20200518_values[1, :])
        )
        self.assertTrue(
            np.allclose(df.t1_red.values, l8_224078_20200518_values[2, :])
        )
        self.assertTrue(
            np.allclose(df.t2_blue.values, l8_224078_20200518_values[0, :])
        )
        self.assertTrue(
            np.allclose(df.t2_green.values, l8_224078_20200518_values[1, :])
        )
        self.assertTrue(
            np.allclose(df.t2_red.values, l8_224078_20200518_values[2, :])
        )

    def test_points(self):
        with gw.open(l8_224078_20200518_B2) as src:
            df = gw.extract(
                src, l8_224078_20200518_polygons, band_names=["blue"]
            )
            self.assertTrue(
                np.allclose(
                    df.blue.values[:10],
                    np.array(
                        [
                            7994,
                            8017,
                            8008,
                            8008,
                            8018,
                            8007,
                            7991,
                            7993,
                            7981,
                            7991,
                        ]
                    ),
                )
            )
            self.assertTrue(
                np.allclose(
                    df.geometry.x.values[:10],
                    np.array(
                        [
                            737559.50243024,
                            737589.50243024,
                            737619.50243024,
                            737649.50243024,
                            737679.50243024,
                            737709.50243024,
                            737739.50243024,
                            737769.50243024,
                            737799.50243024,
                            737829.50243024,
                        ]
                    ),
                )
            )
            self.assertTrue(
                np.allclose(
                    df.geometry.y.values[:10],
                    np.array(
                        [
                            -2795247.77178369,
                            -2795247.77178369,
                            -2795247.77178369,
                            -2795247.77178369,
                            -2795247.77178369,
                            -2795247.77178369,
                            -2795247.77178369,
                            -2795247.77178369,
                            -2795247.77178369,
                            -2795247.77178369,
                        ]
                    ),
                )
            )


if __name__ == "__main__":
    unittest.main()
