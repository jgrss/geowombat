import unittest
from pathlib import Path

import geowombat as gw
from geowombat.data import l8_224078_20200518

import dask
import xarray as xr
import rasterio as rio


class TestOpen(unittest.TestCase):

    def test_open(self):

        with gw.open(l8_224078_20200518) as src:
            self.assertEqual(src.gw.nbands, 3)

    def test_open_multiple(self):

        with gw.open([l8_224078_20200518, l8_224078_20200518], stack_dim='time') as src:
            self.assertEqual(src.gw.ntime, 2)

    def test_open_path(self):

        with gw.open(Path(l8_224078_20200518)) as src:
            self.assertEqual(src.gw.nbands, 3)

    def test_open_type_xarray(self):

        with gw.open(l8_224078_20200518) as src:
            self.assertIsInstance(src, xr.DataArray)

    def test_open_type_dask(self):

        with gw.open(l8_224078_20200518) as src:
            self.assertIsInstance(src.data, dask.array.core.Array)

    def test_crs(self):

        with gw.open(l8_224078_20200518) as src:
            self.assertEqual(src.crs, '+init=epsg:32621')

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


if __name__ == '__main__':
    unittest.main()
