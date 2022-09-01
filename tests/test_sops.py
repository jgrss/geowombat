import unittest

import geowombat as gw
from geowombat.data import l8_224078_20200518


class TestSOPs(unittest.TestCase):
    def test_sample(self):
        n = 100
        with gw.open(l8_224078_20200518) as src:
            df = gw.sample(src, n=n)
            self.assertEqual(len(df.index), n)
            self.assertIn(1, df.columns.tolist())
            self.assertIn(2, df.columns.tolist())
            self.assertIn(3, df.columns.tolist())

    def test_replace(self):
        replace = {7581: 20_000}
        with gw.open(l8_224078_20200518) as src:
            src_r = gw.replace(src, replace)
            self.assertIn(7581, src.drop_duplicates('band').values)
            self.assertNotIn(7581, src_r.drop_duplicates('band').values)
