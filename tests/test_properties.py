import unittest

import geowombat as gw
from geowombat.data import l8_224078_20200518

import geopandas as gpd


class TestProperties(unittest.TestCase):
    def test_filenames_none(self):
        with gw.open(l8_224078_20200518, persist_filenames=False) as src:
            self.assertFalse(hasattr(src, 'filenames'))

    def test_filenames_single(self):
        with gw.open(l8_224078_20200518, persist_filenames=True) as src:
            self.assertTrue(hasattr(src, 'filenames'))
            self.assertEqual(len(src.filenames), 1)

    def test_filenames_multi(self):
        with gw.open(
            [l8_224078_20200518, l8_224078_20200518], persist_filenames=True
        ) as src:
            self.assertTrue(hasattr(src, 'filenames'))
            self.assertEqual(len(src.filenames), 2)

    def test_chunk_grid(self):
        with gw.open(l8_224078_20200518) as src:
            self.assertEqual(type(src.gw.chunk_grid), gpd.GeoDataFrame)

    def test_footprint_grid(self):
        with gw.open(l8_224078_20200518, persist_filenames=True) as src:
            self.assertEqual(type(src.gw.footprint_grid), gpd.GeoDataFrame)


if __name__ == '__main__':
    unittest.main()
