import unittest

import geowombat as gw
from geowombat.data import l8_224078_20200518

import geopandas as gpd


class TestProperties(unittest.TestCase):
    def test_filenames_none(self):
        with gw.open(l8_224078_20200518, persist_filenames=False) as src:
            self.assertFalse(hasattr(src, '_filenames'))

    def test_filenames_single(self):
        with gw.open(l8_224078_20200518, persist_filenames=True) as src:
            self.assertTrue(hasattr(src, '_filenames'))
            self.assertEqual(len(src.gw.filenames), 1)

    def test_filenames_multi(self):
        with gw.open(
            [l8_224078_20200518, l8_224078_20200518], persist_filenames=True
        ) as src:
            self.assertTrue(hasattr(src, '_filenames'))
            self.assertEqual(len(src.gw.filenames), 2)

    def test_chunk_grid(self):
        with gw.open(l8_224078_20200518) as src:
            self.assertEqual(type(src.gw.chunk_grid), gpd.GeoDataFrame)

    def test_footprint_grid(self):
        with gw.open(l8_224078_20200518, persist_filenames=True) as src:
            self.assertEqual(type(src.gw.footprint_grid), gpd.GeoDataFrame)

    def test_array_type(self):
        with gw.open(l8_224078_20200518) as src:
            self.assertTrue(src.gw.array_is_dask)
            src.load()
            self.assertFalse(src.gw.array_is_dask)

    def test_chunksize_check(self):
        with gw.open(l8_224078_20200518) as src:
            # Valid multiple of 16 within bounds
            self.assertEqual(src.gw.check_chunksize(64, 128), 64)
            # Valid multiple of 16 but exceeds bounds - should reduce
            self.assertEqual(src.gw.check_chunksize(64, 60), 32)
            # Non-multiple of 16 should round to nearest (250 -> 256)
            self.assertEqual(src.gw.check_chunksize(250, 512), 256)
            # Non-multiple of 16 should round to nearest (100 -> 96)
            self.assertEqual(src.gw.check_chunksize(100, 512), 96)
            # Invalid/zero should default to 512
            self.assertEqual(src.gw.check_chunksize(0, 1024), 512)
            # Negative should default to 512
            self.assertEqual(src.gw.check_chunksize(-10, 1024), 512)


if __name__ == '__main__':
    unittest.main()
