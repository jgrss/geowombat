import unittest
from pathlib import Path

import geowombat as gw
from geowombat.data import l8_224078_20200518, l3b_s2b_00390821jxn0l2a_20210319_20220730_c01

import numpy as np
import dask
import xarray as xr
import rasterio as rio
from pyproj import CRS


class TestOpen(unittest.TestCase):
    def test_open_netcdf(self):
        with gw.open(
            l3b_s2b_00390821jxn0l2a_20210319_20220730_c01,
            chunks={'band': -1, 'y': 256, 'x': 256},
            engine='h5netcdf'
        ) as src:
            self.assertEqual(src.shape, (6, 668, 668))
            with xr.open_dataset(
                l3b_s2b_00390821jxn0l2a_20210319_20220730_c01,
                chunks={'band': -1, 'y': 256, 'x': 256},
                engine='h5netcdf'
            ) as ds:
                self.assertTrue(np.allclose(src.y.values, ds.y.values))
                self.assertTrue(np.allclose(src.x.values, ds.x.values))

    def test_open(self):
        with gw.open(l8_224078_20200518) as src:
            self.assertEqual(src.gw.nbands, 3)

    def test_has_band_dim(self):
        with gw.open(l8_224078_20200518) as src:
            self.assertTrue(src.gw.has_band_dim)

    def test_has_band_coord(self):
        with gw.open(l8_224078_20200518) as src:
            self.assertTrue(src.gw.has_band_coord)

    def test_has_band(self):
        with gw.open(l8_224078_20200518) as src:
            self.assertTrue(src.gw.has_band)

    def test_has_no_band_coord(self):
        with gw.open(l8_224078_20200518) as src:
            self.assertFalse(src.drop_vars('band').gw.has_band_coord)

    def test_nodata(self):
        with gw.open(l8_224078_20200518) as src:
            self.assertEqual(src.gw.nodataval, np.nan)
        with gw.open(l8_224078_20200518, nodata=0) as src:
            self.assertEqual(src.gw.nodataval, 0)

    def test_open_multiple(self):
        with gw.open([l8_224078_20200518, l8_224078_20200518], stack_dim='time') as src:
            self.assertEqual(src.gw.ntime, 2)

    def test_open_multiple_same(self):
        with gw.open(
            [l8_224078_20200518, l8_224078_20200518],
            time_names=['20200518', '20200518'],
            stack_dim='time'
        ) as src:
            self.assertEqual(src.gw.ntime, 1)

    def test_open_multiple_same_max(self):
        with gw.open(
            [l8_224078_20200518, l8_224078_20200518],
            time_names=['20200518', '20200518'],
            stack_dim='time',
            overlap='max'
        ) as src:
            self.assertEqual(src.gw.ntime, 1)

    def test_open_multiple_same_min(self):
        with gw.open(
            [l8_224078_20200518, l8_224078_20200518],
            time_names=['20200518', '20200518'],
            stack_dim='time',
            overlap='min'
        ) as src:
            self.assertEqual(src.gw.ntime, 1)

    def test_open_multiple_same_mean(self):
        with gw.open(
            [l8_224078_20200518, l8_224078_20200518],
            time_names=['20200518', '20200518'],
            stack_dim='time',
            overlap='mean'
        ) as src:
            self.assertEqual(src.gw.ntime, 1)

    def test_has_time_dim(self):
        with gw.open([l8_224078_20200518, l8_224078_20200518], stack_dim='time') as src:
            self.assertTrue(src.gw.has_time_dim)

    def test_has_time_coord(self):
        with gw.open([l8_224078_20200518, l8_224078_20200518], stack_dim='time') as src:
            self.assertTrue(src.gw.has_time_coord)

    def test_has_time(self):
        with gw.open([l8_224078_20200518, l8_224078_20200518], stack_dim='time') as src:
            self.assertTrue(src.gw.has_time)

    def test_has_no_time_coord(self):
        with gw.open([l8_224078_20200518, l8_224078_20200518], stack_dim='time') as src:
            self.assertFalse(src.drop_vars('time').gw.has_time_coord)

    def test_open_path(self):
        with gw.open(Path(l8_224078_20200518)) as src:
            self.assertEqual(src.gw.nbands, 3)
            self.assertEqual(src.gw.band_chunks, 3)

    def test_open_type_xarray(self):
        with gw.open(l8_224078_20200518) as src:
            self.assertIsInstance(src, xr.DataArray)

    def test_open_type_dask(self):
        with gw.open(l8_224078_20200518) as src:
            self.assertIsInstance(src.data, dask.array.core.Array)

    def test_crs(self):
        with gw.open(l8_224078_20200518) as src:
            self.assertEqual(src.crs, 32621)

    def test_time_chunks(self):
        with gw.open(l8_224078_20200518) as src:
            self.assertEqual(src.gw.time_chunks, 1)

    def test_row_chunks(self):
        with gw.open(l8_224078_20200518) as src:
            self.assertEqual(src.gw.row_chunks, 256)

    def test_row_chunks_set(self):
        with gw.open(l8_224078_20200518, chunks=64) as src:
            self.assertEqual(src.gw.row_chunks, 64)

    def test_col_chunks(self):
        with gw.open(l8_224078_20200518) as src:
            self.assertEqual(src.gw.col_chunks, 256)

    def test_col_chunks_set(self):
        with gw.open(l8_224078_20200518, chunks=64) as src:
            self.assertEqual(src.gw.col_chunks, 64)

    def test_dtype(self):
        with gw.open(l8_224078_20200518, dtype='float64') as src:
            self.assertEqual(src.dtype, 'float64')

    def test_count(self):
        with gw.open(l8_224078_20200518) as src, rio.open(l8_224078_20200518) as rsrc:
            self.assertEqual(src.gw.nbands, rsrc.count)

    def test_width(self):
        with gw.open(l8_224078_20200518) as src, rio.open(l8_224078_20200518) as rsrc:
            self.assertEqual(src.gw.nrows, rsrc.height)

    def test_height(self):
        with gw.open(l8_224078_20200518) as src, rio.open(l8_224078_20200518) as rsrc:
            self.assertEqual(src.gw.ncols, rsrc.width)

    def test_transform(self):
        test_crs = CRS.from_user_input('epsg:4326')
        with gw.open(l8_224078_20200518) as src:
            result = src.gw.transform_crs(
                dst_crs=4326,
                dst_width=src.gw.ncols,
                dst_height=src.gw.nrows,
                coords_only=True
            )
            self.assertEqual(test_crs, result.crs)
            self.assertEqual(test_crs, result.gw.crs_to_pyproj)


if __name__ == '__main__':
    unittest.main()
