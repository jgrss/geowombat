import unittest

import numpy as np
import rasterio as rio
from affine import Affine
from pyproj import CRS
from rasterio.coords import BoundingBox
from rasterio.windows import Window

from geowombat.backends.rasterio_ import (
    align_bounds,
    check_crs,
    check_file_crs,
    check_res,
    get_file_info,
    get_window_from_bounds,
    unpack_bounding_box,
    unpack_window,
    window_to_bounds,
)
from geowombat.data import (
    l3b_s2b_00390821jxn0l2a_20210319_20220730_c01,
    l8_224077_20200518_B2,
)


class TestRasterio(unittest.TestCase):
    def test_align_bounds(self):
        transform, width, height = align_bounds(
            -100.0, -100.0, 100.0, 100.0, (2.0, 2.0)
        )
        self.assertTupleEqual(
            transform, Affine(2.0, 0.0, -100.0, 0.0, -2.0, 100.0)
        )

        self.assertRaises(
            TypeError, align_bounds, -100.0, -100.0, 100.0, 100.0, {2.0}
        )

        transform, width, height = align_bounds(
            -100.0, -100.0, 100.0, 100.0, 2.0
        )
        self.assertTupleEqual(
            transform, Affine(2.0, 0.0, -100.0, 0.0, -2.0, 100.0)
        )

        transform, width, height = align_bounds(
            -100.0, -100.0, 100.0, 100.0, 1.67
        )
        self.assertTupleEqual(
            transform,
            Affine(
                1.67, 0.0, -100.19999999999999, 0.0, -1.67, 100.19999999999999
            ),
        )

    def test_get_window_from_bounds(self):
        bounds = BoundingBox(
            left=-100,
            bottom=-100,
            right=100,
            top=100,
        )
        dst_window = get_window_from_bounds(bounds=bounds, res=(10, 10))
        self.assertEqual((dst_window.height, dst_window.width), (20, 20))

    def test_get_file_info(self):
        with rio.open(l8_224077_20200518_B2) as src:
            file_info = get_file_info(src)

            self.assertEqual(file_info.src_bounds, src.bounds)
            self.assertEqual(file_info.src_res, src.res)
            self.assertEqual(file_info.src_height, src.height)
            self.assertEqual(file_info.src_width, src.width)

    def test_check_crs(self):
        with rio.open(l8_224077_20200518_B2) as src:
            self.assertEqual(check_crs(src.crs), CRS.from_epsg(32621))
            self.assertEqual(
                check_crs(src.crs.to_dict()), CRS.from_epsg(32621)
            )
            self.assertEqual(
                check_crs(src.crs.to_proj4()), CRS.from_epsg(32621)
            )
            self.assertEqual(
                check_crs(
                    "+proj=utm +zone=21 +datum=WGS84 +units=m +no_defs +type=crs"
                ),
                CRS.from_epsg(32621),
            )

        self.assertEqual(check_crs(4326), CRS.from_epsg(4326))
        self.assertRaises(ValueError, check_crs, 4325)
        self.assertRaises(ValueError, check_crs, 'cat')
        self.assertRaises(TypeError, check_crs, np.ones(1))

    def test_check_file_crs(self):
        crs = check_file_crs(
            f"netcdf:{l3b_s2b_00390821jxn0l2a_20210319_20220730_c01}:blue"
        )
        self.assertEqual(crs, CRS.from_epsg(8858))

        crs = check_file_crs(l3b_s2b_00390821jxn0l2a_20210319_20220730_c01)
        self.assertEqual(crs, CRS.from_epsg(8858))

        crs = check_file_crs(l8_224077_20200518_B2)
        self.assertEqual(crs, CRS.from_epsg(32621))

    def test_check_res(self):
        self.assertEqual(check_res(10.0), (10.0, 10.0))
        self.assertEqual(check_res(10), (10.0, 10.0))
        self.assertEqual(check_res((10, 10)), (10.0, 10.0))
        self.assertRaises(TypeError, check_res, {10})

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
        bounds = window_to_bounds(l8_224077_20200518_B2, w=w)
        with rio.open(l8_224077_20200518_B2) as src:
            ref_bounds = (
                src.bounds.left,
                src.bounds.top - (100 * src.res[0]),
                src.bounds.left + (100 * src.res[0]),
                src.bounds.top,
            )
            self.assertEqual(bounds, ref_bounds)
        bounds = window_to_bounds([l8_224077_20200518_B2], w=w)
        with rio.open(l8_224077_20200518_B2) as src:
            ref_bounds = (
                src.bounds.left,
                src.bounds.top - (100 * src.res[0]),
                src.bounds.left + (100 * src.res[0]),
                src.bounds.top,
            )
            self.assertEqual(bounds, ref_bounds)


if __name__ == '__main__':
    unittest.main()
