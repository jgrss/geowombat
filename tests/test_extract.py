import unittest

import geowombat as gw
from geowombat.data import l8_224078_20200518_points, l8_224078_20200518, l8_224078_20200518_B2
from geowombat.core.conversion import Converters

import numpy as np
import geopandas as gpd
from sklearn.preprocessing import LabelEncoder


converters = Converters()


aoi = gpd.read_file(l8_224078_20200518_points)
aoi['id'] = LabelEncoder().fit_transform(aoi.name)
aoi = aoi.drop(columns=['name'])

l8_224078_20200518_B2_values = np.array([7966, 8030, 7561, 8302, 8277, 7398], dtype='float64')

l8_224078_20200518_values = np.array([[7966, 8030, 7561, 8302, 8277, 7398],
                                      [7326, 7490, 6874, 8202, 7982, 6711],
                                      [6254, 8080, 6106, 8111, 7341, 6007]], dtype='float64')


class TestConfig(unittest.TestCase):

    def test_single_image_single_band(self):

        with gw.open(l8_224078_20200518_B2) as src:
            df = gw.extract(src, aoi, band_names=['blue'])

        self.assertTrue(np.allclose(df.blue.values, l8_224078_20200518_B2_values))

    def test_single_image_multi_band(self):

        with gw.open(l8_224078_20200518) as src:
            df = gw.extract(src, aoi, band_names=['blue', 'green', 'red'])

        self.assertTrue(np.allclose(df.blue.values, l8_224078_20200518_values[0, :]))
        self.assertTrue(np.allclose(df.green.values, l8_224078_20200518_values[1, :]))
        self.assertTrue(np.allclose(df.red.values, l8_224078_20200518_values[2, :]))

    def test_multi_image_single_band(self):

        with gw.open([l8_224078_20200518_B2, l8_224078_20200518_B2]) as src:
            df = gw.extract(src, aoi, band_names=['blue'])

        self.assertTrue(np.allclose(df.t1_blue.values, l8_224078_20200518_B2_values))
        self.assertTrue(np.allclose(df.t2_blue.values, l8_224078_20200518_B2_values))

    def test_multi_image_multi_band(self):

        with gw.open([l8_224078_20200518, l8_224078_20200518]) as src:
            df = gw.extract(src, aoi, band_names=['blue', 'green', 'red'])

        self.assertTrue(np.allclose(df.t1_blue.values, l8_224078_20200518_values[0, :]))
        self.assertTrue(np.allclose(df.t1_green.values, l8_224078_20200518_values[1, :]))
        self.assertTrue(np.allclose(df.t1_red.values, l8_224078_20200518_values[2, :]))
        self.assertTrue(np.allclose(df.t2_blue.values, l8_224078_20200518_values[0, :]))
        self.assertTrue(np.allclose(df.t2_green.values, l8_224078_20200518_values[1, :]))
        self.assertTrue(np.allclose(df.t2_red.values, l8_224078_20200518_values[2, :]))


if __name__ == '__main__':
    unittest.main()
