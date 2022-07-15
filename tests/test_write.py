import unittest
import tempfile
from pathlib import Path

import geowombat as gw
from geowombat.data import l8_224078_20200518


class TestConfig(unittest.TestCase):
    def test_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / 'test.tif'
            with gw.open(l8_224078_20200518) as src:
                (
                    src.fillna(32768)
                    .assign_attrs(nodatavals=[32768])
                    .astype('uint16').gw.save(
                        filename=out_path,
                        overwrite=True,
                        tags={'TEST_METADATA': 'TEST_VALUE'},
                        compression='lzw',
                        num_workers=4
                    )
                )
                with gw.open(out_path) as tmp_src:
                    self.assertTrue(src.equals(tmp_src))
                    self.assertTrue(hasattr(tmp_src, 'TEST_METADATA'))
                    self.assertEqual(tmp_src.TEST_METADATA, 'TEST_VALUE')


if __name__ == '__main__':
    unittest.main()