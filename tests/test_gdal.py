import shutil
import tempfile
import unittest
from pathlib import Path

from geowombat.backends.gdal_ import warp
from geowombat.data import l8_224077_20200518_B2


class TestGDALWarp(unittest.TestCase):
    def test_gdal_warp(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / 'tmp.tif'
            out_path2 = Path(tmp) / 'tmp2.tif'
            if out_path.is_file():
                out_path.unlink()
            warp(
                l8_224077_20200518_B2,
                out_path,
                width=10,
                height=10,
            )
            self.assertTrue(out_path.is_file())
            shutil.copy(str(out_path), str(out_path2))
            warp(
                out_path,
                out_path2,
                overwrite=True,
                width=10,
                height=10,
                delete_input=True,
            )
            self.assertFalse(out_path.is_file())
            self.assertTrue(out_path2.is_file())


if __name__ == '__main__':
    unittest.main()
