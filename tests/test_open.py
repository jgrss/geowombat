import tempfile
import unittest
from pathlib import Path

import dask
import numpy as np
import rasterio as rio
import xarray as xr
from pyproj import CRS

import geowombat as gw
from geowombat.core import coords_to_indices, lonlat_to_xy
from geowombat.data import (
    l3b_s2b_00390821jxn0l2a_20210319_20220730_c01,
    l8_224077_20200518_B2,
    l8_224077_20200518_B2_60m,
    l8_224078_20200518,
    l8_224078_20200518_B2,
)


class TestOpen(unittest.TestCase):
    def test_open_netcdf(self):
        with gw.open(
            l3b_s2b_00390821jxn0l2a_20210319_20220730_c01,
            chunks={'band': -1, 'y': 256, 'x': 256},
            engine='h5netcdf',
        ) as src:
            self.assertEqual(src.shape, (6, 668, 668))
            with xr.open_dataset(
                l3b_s2b_00390821jxn0l2a_20210319_20220730_c01,
                chunks={'band': -1, 'y': 256, 'x': 256},
                engine='h5netcdf',
            ) as ds:
                self.assertTrue(np.allclose(src.y.values, ds.y.values))
                self.assertTrue(np.allclose(src.x.values, ds.x.values))

    def test_open(self):
        with gw.open(l8_224078_20200518) as src:
            self.assertEqual(src.gw.nbands, 3)

    def test_open_incompat_res(self):
        with gw.open(l8_224077_20200518_B2) as src30m:
            with gw.open(l8_224077_20200518_B2_60m) as src60m:
                with self.assertRaises(ValueError):
                    res = xr.align(src30m, src60m, join='exact')

        with self.assertWarns(UserWarning):
            with gw.open(
                [l8_224077_20200518_B2, l8_224077_20200518_B2_60m],
                stack_dim='band',
                band_names=[1, 2],
            ) as src:
                pass

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
            self.assertTrue(np.isnan(src.gw.nodataval))
        with gw.open(l8_224078_20200518, nodata=0) as src:
            self.assertEqual(src.gw.nodataval, 0)

    def test_open_multiple(self):
        with gw.open(
            [l8_224078_20200518, l8_224078_20200518],
            stack_dim='time',
        ) as src:
            self.assertEqual(src.gw.ntime, 2),
            self.assertTrue(src.gw.has_time_dim)
            self.assertTrue(src.gw.has_time_coord)

        with gw.open(
            [l8_224078_20200518_B2, l8_224078_20200518_B2],
            stack_dim='band',
        ) as src:
            self.assertEqual(src.gw.nbands, 2)
            self.assertTrue(src.gw.has_band_dim)
            self.assertTrue(src.gw.has_band_coord)

    def test_open_multiple_same(self):
        with gw.open(
            [l8_224078_20200518, l8_224078_20200518],
            time_names=['20200518', '20200518'],
            stack_dim='time',
        ) as src:
            self.assertEqual(src.gw.ntime, 1)
            self.assertTrue(src.gw.has_time_dim)
            self.assertTrue(src.gw.has_time_coord)

    def test_open_multiple_same_max(self):
        with gw.open(
            [l8_224078_20200518, l8_224078_20200518],
            time_names=['20200518', '20200518'],
            stack_dim='time',
            overlap='max',
        ) as src:
            self.assertEqual(src.gw.ntime, 1)
            self.assertTrue(src.gw.has_time_dim)
            self.assertTrue(src.gw.has_time_coord)

    def test_open_multiple_same_min(self):
        with gw.open(
            [l8_224078_20200518, l8_224078_20200518],
            time_names=['20200518', '20200518'],
            stack_dim='time',
            overlap='min',
        ) as src:
            self.assertEqual(src.gw.ntime, 1)
            self.assertTrue(src.gw.has_time_dim)
            self.assertTrue(src.gw.has_time_coord)

    def test_open_multiple_same_mean(self):
        with gw.open(
            [l8_224078_20200518, l8_224078_20200518],
            time_names=['20200518', '20200518'],
            stack_dim='time',
            overlap='mean',
        ) as src:
            self.assertEqual(src.gw.ntime, 1)
            self.assertTrue(src.gw.has_time_dim)
            self.assertTrue(src.gw.has_time_coord)

    def test_union_values(self):
        filenames = [l8_224077_20200518_B2, l8_224078_20200518_B2]
        with gw.open(
            filenames,
            band_names=['blue'],
            mosaic=True,
            bounds_by='union',
        ) as src:
            values = src.values[
                0,
                src.gw.nrows // 2,
                src.gw.ncols // 2 : src.gw.ncols // 2 + 10,
            ]
            self.assertTrue(
                np.allclose(
                    values,
                    np.array(
                        [
                            7524,
                            7538,
                            7573,
                            7625,
                            7683,
                            7661,
                            7643,
                            7773,
                            7697,
                            7566,
                        ]
                    ),
                )
            )

    def test_bounds_union(self):
        filenames = [l8_224077_20200518_B2, l8_224078_20200518_B2]
        with gw.open(
            filenames,
            band_names=['blue'],
            mosaic=True,
            bounds_by='union',
        ) as src:
            bounds = src.gw.bounds
            self.assertEqual(
                bounds, (693990.0, -2832810.0, 778590.0, -2766600.0)
            )

    def test_bounds_intersection(self):
        filenames = [l8_224077_20200518_B2, l8_224078_20200518_B2]
        with gw.open(
            filenames,
            band_names=['blue'],
            mosaic=True,
            bounds_by='intersection',
        ) as src:
            bounds = src.gw.bounds
            self.assertEqual(
                bounds, (717330.0, -2812080.0, 754200.0, -2776980.0)
            )

    def test_mosaic_max_bands(self):
        filenames = [l8_224077_20200518_B2, l8_224078_20200518_B2]
        with gw.open(
            filenames,
            band_names=['blue'],
            mosaic=True,
            overlap='max',
            bounds_by='intersection',
            nodata=0,
        ) as src:
            self.assertTrue(src.gw.has_band_dim)
            self.assertTrue(src.gw.has_band_coord)
            self.assertEqual(src.shape, (1, 1170, 1229))

    def test_mosaic_max(self):
        filenames = [l8_224077_20200518_B2, l8_224078_20200518_B2]
        with gw.open(
            filenames,
            band_names=['blue'],
            mosaic=True,
            overlap='max',
            bounds_by='intersection',
            nodata=0,
        ) as src:
            x, y = lonlat_to_xy(-54.78604601, -25.23023330, dst_crs=src)
            j, i = coords_to_indices(x, y, src)
            block = src[0, i : i + 3, j : j + 3].values
            self.assertTrue(
                np.allclose(
                    block,
                    np.array(
                        [
                            [8387, 8183, 8050],
                            [7938, 7869, 7889],
                            [7862, 7828, 7721],
                        ],
                        dtype='float32',
                    ),
                )
            )

    def test_mosaic_min(self):
        filenames = [l8_224077_20200518_B2, l8_224078_20200518_B2]
        with gw.open(
            filenames,
            band_names=['blue'],
            mosaic=True,
            overlap='min',
            bounds_by='intersection',
            nodata=0,
        ) as src:
            x, y = lonlat_to_xy(-54.78604601, -25.23023330, dst_crs=src)
            j, i = coords_to_indices(x, y, src)
            block = src[0, i : i + 3, j : j + 3].values
            self.assertTrue(
                np.allclose(
                    block,
                    np.array(
                        [
                            [8384, 8183, 8049],
                            [7934, 7867, 7885],
                            [7861, 7826, 7721],
                        ],
                        dtype='float32',
                    ),
                )
            )

    def test_mosaic_mean(self):
        filenames = [l8_224077_20200518_B2, l8_224078_20200518_B2]
        with gw.open(
            filenames,
            band_names=['blue'],
            mosaic=True,
            overlap='mean',
            bounds_by='intersection',
            nodata=0,
        ) as src:
            x, y = lonlat_to_xy(-54.78604601, -25.23023330, dst_crs=src)
            j, i = coords_to_indices(x, y, src)
            block = src[0, i : i + 3, j : j + 3].values
            self.assertTrue(
                np.allclose(
                    block,
                    np.array(
                        [
                            [8385.5, 8183, 8049.5],
                            [7936, 7868, 7887],
                            [7861.5, 7827, 7721],
                        ],
                        dtype='float32',
                    ),
                )
            )

    def test_has_time_dim(self):
        with gw.open(
            [l8_224078_20200518, l8_224078_20200518], stack_dim='time'
        ) as src:
            self.assertTrue(src.gw.has_time_dim)

    def test_has_time_coord(self):
        with gw.open(
            [l8_224078_20200518, l8_224078_20200518], stack_dim='time'
        ) as src:
            self.assertTrue(src.gw.has_time_coord)

    def test_has_time(self):
        with gw.open(
            [l8_224078_20200518, l8_224078_20200518], stack_dim='time'
        ) as src:
            self.assertTrue(src.gw.has_time)

    def test_has_no_time_coord(self):
        with gw.open(
            [l8_224078_20200518, l8_224078_20200518], stack_dim='time'
        ) as src:
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
        with gw.open(l8_224078_20200518) as src, rio.open(
            l8_224078_20200518
        ) as rsrc:
            self.assertEqual(src.gw.nbands, rsrc.count)

    def test_width(self):
        with gw.open(l8_224078_20200518) as src, rio.open(
            l8_224078_20200518
        ) as rsrc:
            self.assertEqual(src.gw.nrows, rsrc.height)

    def test_height(self):
        with gw.open(l8_224078_20200518) as src, rio.open(
            l8_224078_20200518
        ) as rsrc:
            self.assertEqual(src.gw.ncols, rsrc.width)

    def test_transform(self):
        test_crs = CRS.from_user_input('epsg:4326')
        with gw.open(l8_224078_20200518) as src:
            result = src.gw.transform_crs(
                dst_crs=4326,
                dst_width=src.gw.ncols,
                dst_height=src.gw.nrows,
                coords_only=True,
            )
            self.assertEqual(test_crs, result.crs)
            self.assertEqual(test_crs, result.gw.crs_to_pyproj)


if __name__ == '__main__':
    unittest.main()
