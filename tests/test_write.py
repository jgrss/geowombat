import unittest
import tempfile
from pathlib import Path

import geowombat as gw
from geowombat.data import l8_224078_20200518


class TestConfig(unittest.TestCase):
    def test_to_netcdf(self):
        bands = ['blue', 'green', 'red']
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / 'test.nc'
            with gw.open(l8_224078_20200518, band_names=bands) as src:
                (
                    src
                    .fillna(32768)
                    .assign_attrs(nodatavals=(32768,))
                    .astype('uint16')
                    .gw.to_netcdf(
                        filename=out_path,
                        overwrite=True
                    )
                )

    def test_to_raster(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / 'test.tif'
            with gw.open(l8_224078_20200518) as src:
                (
                    src
                    .fillna(32768)
                    .assign_attrs(nodatavals=(32768,))
                    .astype('uint16')
                    .gw.to_raster(
                        filename=out_path,
                        overwrite=True,
                        tags={'TEST_METADATA': 'TEST_VALUE'},
                        compression='lzw',
                        num_workers=2
                    )
                )
                with gw.open(out_path) as tmp_src:
                    self.assertTrue(src.equals(tmp_src))
                    self.assertTrue(hasattr(tmp_src, 'TEST_METADATA'))
                    self.assertEqual(tmp_src.TEST_METADATA, 'TEST_VALUE')

    def test_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / 'test.tif'
            with gw.open(l8_224078_20200518) as src:
                (
                    src
                    .fillna(32768)
                    .assign_attrs(nodatavals=(32768,))
                    .astype('uint16')
                    .gw.save(
                        filename=out_path,
                        overwrite=True,
                        tags={'TEST_METADATA': 'TEST_VALUE'},
                        compression='lzw',
                        num_workers=2
                    )
                )
                with gw.open(out_path) as tmp_src:
                    self.assertTrue(src.equals(tmp_src))
                    self.assertTrue(hasattr(tmp_src, 'TEST_METADATA'))
                    self.assertEqual(tmp_src.TEST_METADATA, 'TEST_VALUE')


if __name__ == '__main__':
    unittest.main()
