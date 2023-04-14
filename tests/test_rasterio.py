import unittest

import rasterio as rio
from rasterio.coords import BoundingBox
from rasterio.windows import Window

from geowombat.backends.rasterio_ import (
    check_res,
    get_dims_from_bounds,
    get_file_info,
    unpack_bounding_box,
    unpack_window,
    window_to_bounds,
)
from geowombat.data import l8_224077_20200518_B2


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

    def test_check_res(self):
        self.assertEqual(check_res(10), (10.0, 10.0))

    def test_unpack_bounding_box(self):
        bounds = 'BoundingBox(left=-100, bottom=-100, right=100, top=100)'
        ref_bounds = BoundingBox(
            left=-100,
            bottom=-100,
            right=100,
            top=100,
        )
        converted_bounds = unpack_bounding_box(bounds)
        self.assertEqual(ref_bounds, converted_bounds)

    def test_unpack_window(self):
        window = 'Window(col_off=0, row_off=0, width=100, height=100)'
        converted_window = unpack_window(window)
        ref_window = Window(col_off=0, row_off=0, width=100, height=100)
        self.assertEqual(ref_window, converted_window)

    def test_window_to_bounds(self):
        w = Window(col_off=0, row_off=0, width=100, height=100)
        with rio.open(l8_224077_20200518_B2) as src:
            bounds = window_to_bounds(l8_224077_20200518_B2, w=w)
            ref_bounds = (
                src.bounds.left,
                src.bounds.top - (100 * src.res[0]),
                src.bounds.left + (100 * src.res[0]),
                src.bounds.top,
            )
            self.assertEqual(bounds, ref_bounds)


if __name__ == '__main__':
    unittest.main()
