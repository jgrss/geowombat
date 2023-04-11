import unittest

import geowombat as gw
from geowombat.data import l8_224078_20200518


class TestMoving(unittest.TestCase):
    def test_moving(self):
        with gw.open(l8_224078_20200518, chunks=128) as src:
            res = src.gw.moving(
                stat="mean",
                w=3,
                nodata=0,
            )
            ref_band1_mean = (
                src.sel(band=1)[100:103, 100:103].mean().data.compute()
            )
            tar_band1_mean = res.sel(band=1)[101, 101].data.compute()

            self.assertEqual(ref_band1_mean, tar_band1_mean)
            self.assertEqual(src.shape, res.shape)

            res = src.gw.moving(
                stat="mean",
                w=5,
                nodata=0,
            )
            ref_band1_mean = (
                src.sel(band=1)[100:105, 100:105].mean().data.compute()
            )
            tar_band1_mean = res.sel(band=1)[102, 102].data.compute()

            self.assertEqual(ref_band1_mean, tar_band1_mean)
            self.assertEqual(src.shape, res.shape)


if __name__ == '__main__':
    unittest.main()
