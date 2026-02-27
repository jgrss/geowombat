"""Tests for issue #325: rounding in bounds dimension calculation.

When a GeoTIFF's bounds produce a width/height that is very close to
an integer (e.g. 255.9999), the result should be rounded rather than
truncated.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.transform import from_bounds

import geowombat as gw


class TestBoundsRounding(unittest.TestCase):
    def test_rounding_preserves_dimensions(self):
        """A 256x256 web mercator raster should stay 256x256 when opened.

        Reproduces issue #325 where float division of bounds by
        resolution yields 255.9999... which was truncated to 255.
        """
        height, width = 256, 256
        # Web Mercator bounds chosen so that
        # (right - left) / res = 255.9999...
        left = 0.0
        bottom = 0.0
        # Use a resolution that causes floating point imprecision
        res = 10.0
        right = left + width * res - 1e-10  # 2559.9999999999
        top = bottom + height * res - 1e-10

        transform = from_bounds(left, bottom, right, top, width, height)

        with tempfile.NamedTemporaryFile(suffix='.tif') as tmp:
            with rio.open(
                tmp.name,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype='uint8',
                crs='EPSG:3857',
                transform=transform,
            ) as dst:
                dst.write(np.ones((1, height, width), dtype='uint8'))

            with gw.open(tmp.name) as src:
                self.assertEqual(
                    src.shape[-2:],
                    (height, width),
                    f'Expected ({height}, {width}) but got {src.shape[-2:]}. '
                    f'Bounds division may be truncating instead of rounding.',
                )


if __name__ == '__main__':
    unittest.main()
