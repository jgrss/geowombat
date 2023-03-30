import unittest
import tempfile
from pathlib import Path

from geowombat.data import l8_224077_20200518_B2
from geowombat.backends.gdal_ import warp


class TestGDALWarp(unittest.TestCase):
    def test_gdal_warp(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / 'tmp.tif'
            warp(
                l8_224077_20200518_B2,
                out_path,
                overwrite=True,
                width=10,
                height=10,
            )


if __name__ == '__main__':
    unittest.main()
