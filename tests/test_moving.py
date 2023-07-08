import unittest

import xarray as xr

import geowombat as gw
from geowombat.data import l8_224078_20200518


class TestMoving(unittest.TestCase):
    def moving_func(self, src: xr.DataArray, stat: str):
        # 3x3 window
        res_3x3 = src.gw.moving(
            stat=stat,
            w=3,
            nodata=0,
        )
        # 5x5 window
        res_5x5 = src.gw.moving(
            stat=stat,
            w=5,
            nodata=0,
        )

        for band in res_3x3.band.values:
            ref_array = src.where(lambda x: x != 0).sel(band=band)

            ref_band1_value = float(
                getattr(ref_array[100:103, 100:103], stat)(skipna=True)
                .fillna(0)
                .data.compute()
            )
            tar_band1_value = res_3x3.sel(band=band)[101, 101].data.compute()
            self.assertEqual(int(ref_band1_value), int(tar_band1_value))
            self.assertEqual(src.shape, res_3x3.shape)

        for band in res_5x5.band.values:
            ref_array = src.where(lambda x: x != 0).sel(band=band)
            ref_band1_value = float(
                getattr(ref_array[100:105, 100:105], stat)(skipna=True)
                .fillna(0)
                .data.compute()
            )
            tar_band1_value = res_5x5.sel(band=band)[102, 102].data.compute()

            self.assertEqual(int(ref_band1_value), int(tar_band1_value))
            self.assertEqual(src.shape, res_5x5.shape)

            # Check along chunk border
            ref_band1_value = float(
                getattr(ref_array[:5, 127:132], stat)(skipna=True)
                .fillna(0)
                .data.compute()
            )
            tar_band1_value = res_5x5.sel(band=band)[2, 129].data.compute()

            self.assertEqual(int(ref_band1_value), int(tar_band1_value))

    def test_moving_mean(self):
        with gw.open(l8_224078_20200518, chunks=128) as src:
            self.moving_func(src, stat="mean")

    def test_moving_min(self):
        with gw.open(l8_224078_20200518, chunks=128) as src:
            self.moving_func(src, stat="min")

    def test_moving_max(self):
        with gw.open(l8_224078_20200518, chunks=128) as src:
            self.moving_func(src, stat="max")

    def test_moving_var(self):
        with gw.open(l8_224078_20200518, chunks=128) as src:
            self.moving_func(src, stat="var")

    def test_moving_std(self):
        with gw.open(l8_224078_20200518, chunks=128) as src:
            self.moving_func(src, stat="std")


if __name__ == '__main__':
    unittest.main()
