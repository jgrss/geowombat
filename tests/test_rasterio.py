import unittest
from dataclasses import dataclass

from geowombat.data import l8_224077_20200518_B2
from geowombat.backends.rasterio_ import get_dims_from_bounds, get_file_info

import rasterio as rio
from rasterio.coords import BoundingBox


class TestRasterio(unittest.TestCase):
    def test_get_dims_from_bounds(self):
        bounds = BoundingBox(
            left=-100,
            bottom=-100,
            right=100,
            top=100,
        )
        height, width = get_dims_from_bounds(bounds=bounds, res=(10, 10))
        self.assertEqual((height, width), (20, 20))

    def test_get_file_info(self):
        with rio.open(l8_224077_20200518_B2) as src:
            file_info = get_file_info(src)

            self.assertEqual(file_info.src_bounds, src.bounds)
            self.assertEqual(file_info.src_res, src.res)
            self.assertEqual(file_info.src_height, src.height)
            self.assertEqual(file_info.src_width, src.width)


if __name__ == '__main__':
    unittest.main()
