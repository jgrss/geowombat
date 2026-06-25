import unittest

import numpy as np
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

    def test_moving_perc(self):
        # The percentile path was previously untested. Exercise it on real
        # data and confirm it runs to completion with the expected shape.
        with gw.open(l8_224078_20200518, chunks=128) as src:
            for perc in (10, 50, 90):
                res = src.gw.moving(
                    stat="perc", perc=perc, w=5, nodata=0
                ).compute()
                self.assertEqual(res.shape, src.shape)

    def test_moving_perc_all_nodata_window(self):
        # Regression: a window containing only nodata (nvalid == 0) used to
        # dereference an empty buffer in the C percentile routine and
        # segfault. It must now return nodata without crashing.
        nodata = 0.0
        arr = np.ones((1, 12, 12), dtype="float64") * 50.0
        arr[:, :6, :6] = nodata  # all-nodata quadrant larger than the window
        da = xr.DataArray(
            arr,
            dims=("band", "y", "x"),
            coords={
                "band": [1],
                "y": np.arange(12, 0, -1.0),
                "x": np.arange(12.0),
            },
        ).chunk({"band": 1, "y": 6, "x": 6})

        res = da.gw.moving(stat="perc", perc=50, w=3, nodata=nodata).compute()

        self.assertEqual(res.shape, da.shape)
        # Center of the all-nodata quadrant resolves to nodata.
        self.assertEqual(float(res[0, 2, 2].values), nodata)
        # A window fully inside the valid region resolves to the data value.
        self.assertEqual(float(res[0, 9, 9].values), 50.0)


if __name__ == '__main__':
    unittest.main()
