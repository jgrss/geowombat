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
    l8_224077_20200518_B2_nan,
    l8_224078_20200518_B2_nan,
)


class TestOpen(unittest.TestCase):
    def test_open_netcdf(self):
        with gw.open(
            l3b_s2b_00390821jxn0l2a_20210319_20220730_c01,
            chunks={"band": -1, "y": 256, "x": 256},
            engine="h5netcdf",
        ) as src:
            self.assertEqual(src.shape, (6, 668, 668))
            with xr.open_dataset(
                l3b_s2b_00390821jxn0l2a_20210319_20220730_c01,
                chunks={"band": -1, "y": 256, "x": 256},
                engine="h5netcdf",
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
                    res = xr.align(src30m, src60m, join="exact")

        with self.assertWarns(UserWarning):
            with gw.open(
                [l8_224077_20200518_B2, l8_224077_20200518_B2_60m],
                stack_dim="band",
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
            self.assertFalse(src.drop_vars("band").gw.has_band_coord)

    def test_nodata(self):
        with gw.open(l8_224078_20200518) as src:
            self.assertTrue(np.isnan(src.gw.nodataval))
        with gw.open(l8_224078_20200518, nodata=0) as src:
            self.assertEqual(src.gw.nodataval, 0)

    def test_open_multiple(self):
        with gw.open(
            [l8_224078_20200518, l8_224078_20200518],
            stack_dim="time",
        ) as src:
            self.assertEqual(src.gw.ntime, 2),
            self.assertTrue(src.gw.has_time_dim)
            self.assertTrue(src.gw.has_time_coord)

        with gw.open(
            [l8_224078_20200518_B2, l8_224078_20200518_B2],
            stack_dim="band",
        ) as src:
            self.assertEqual(src.gw.nbands, 2)
            self.assertTrue(src.gw.has_band_dim)
            self.assertTrue(src.gw.has_band_coord)

    def test_open_multiple_same(self):
        with gw.open(
            [l8_224078_20200518, l8_224078_20200518],
            time_names=["20200518", "20200518"],
            stack_dim="time",
        ) as src:
            self.assertEqual(src.gw.ntime, 1)
            self.assertTrue(src.gw.has_time_dim)
            self.assertTrue(src.gw.has_time_coord)

    def test_open_multiple_same_max(self):
        with gw.open(
            [l8_224078_20200518, l8_224078_20200518],
            time_names=["20200518", "20200518"],
            stack_dim="time",
            overlap="max",
        ) as src:
            self.assertEqual(src.gw.ntime, 1)
            self.assertTrue(src.gw.has_time_dim)
            self.assertTrue(src.gw.has_time_coord)

    def test_open_multiple_same_min(self):
        with gw.open(
            [l8_224078_20200518, l8_224078_20200518],
            time_names=["20200518", "20200518"],
            stack_dim="time",
            overlap="min",
        ) as src:
            self.assertEqual(src.gw.ntime, 1)
            self.assertTrue(src.gw.has_time_dim)
            self.assertTrue(src.gw.has_time_coord)

    def test_open_multiple_same_mean(self):
        with gw.open(
            [l8_224078_20200518, l8_224078_20200518],
            time_names=["20200518", "20200518"],
            stack_dim="time",
            overlap="mean",
        ) as src:
            self.assertEqual(src.gw.ntime, 1)
            self.assertTrue(src.gw.has_time_dim)
            self.assertTrue(src.gw.has_time_coord)

    def test_union_values(self):
        filenames = [l8_224077_20200518_B2, l8_224078_20200518_B2]
        with gw.open(
            filenames,
            band_names=["blue"],
            mosaic=True,
            bounds_by="union",
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
            band_names=["blue"],
            mosaic=True,
            bounds_by="union",
        ) as src:
            bounds = src.gw.bounds
            self.assertEqual(
                bounds, (693990.0, -2832810.0, 778590.0, -2766600.0)
            )

    def test_bounds_intersection(self):
        filenames = [l8_224077_20200518_B2, l8_224078_20200518_B2]
        with gw.open(
            filenames,
            band_names=["blue"],
            mosaic=True,
            bounds_by="intersection",
        ) as src:
            bounds = src.gw.bounds
            self.assertEqual(
                bounds, (717330.0, -2812080.0, 754200.0, -2776980.0)
            )

    def test_mosaic_max_bands(self):
        filenames = [l8_224077_20200518_B2, l8_224078_20200518_B2]
        with gw.open(
            filenames,
            band_names=["blue"],
            mosaic=True,
            overlap="max",
            bounds_by="intersection",
            nodata=0,
        ) as src:
            self.assertTrue(src.gw.has_band_dim)
            self.assertTrue(src.gw.has_band_coord)
            self.assertEqual(src.shape, (1, 1170, 1229))

    def test_mosaic_max(self):
        filenames = [l8_224077_20200518_B2, l8_224078_20200518_B2]
        with gw.open(
            filenames,
            band_names=["blue"],
            mosaic=True,
            overlap="max",
            bounds_by="intersection",
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
                        dtype="float32",
                    ),
                )
            )

    def test_mosaic_max_nan(self):
        filenames = [l8_224077_20200518_B2_nan, l8_224078_20200518_B2_nan]
        with gw.open(
            filenames,
            band_names=["blue"],
            mosaic=True,
            overlap="max",
            bounds_by="intersection",
            nodata=0,
        ) as src:
            start_values = src.values[
                0,
                0,
                0:10,
            ]
            end_values = src.values[
                0,
                -2,
                -10:,
            ]
            x, y = lonlat_to_xy(-54.78604601, -25.23023330, dst_crs=src)
            j, i = coords_to_indices(x, y, src)
            mid_values = src[0, i : i + 3, j : j + 3].values
            self.assertTrue(
                np.allclose(
                    start_values,
                    np.array(
                        [
                            8482.0,
                            8489.0,
                            8483.0,
                            8547.0,
                            8561.0,
                            8574.0,
                            8616.0,
                            8530.0,
                            8396.0,
                            8125.0,
                        ]
                    ),
                ),
            )
            self.assertTrue(
                np.allclose(
                    mid_values,
                    np.array(
                        [
                            [8387.0, 8183.0, 8050.0],
                            [7938.0, 7869.0, 7889.0],
                            [7862.0, 7828.0, 7721.0],
                        ]
                    ),
                )
            )
            self.assertTrue(
                np.allclose(
                    end_values,
                    np.array(
                        [
                            7409.0,
                            7427.0,
                            7490.0,
                            7444.0,
                            7502.0,
                            7472.0,
                            7464.0,
                            7443.0,
                            7406.0,
                            np.nan,
                        ]
                    ),
                    equal_nan=True,
                ),
            )

    def test_mosaic_min(self):
        filenames = [l8_224077_20200518_B2, l8_224078_20200518_B2]
        with gw.open(
            filenames,
            band_names=["blue"],
            mosaic=True,
            overlap="min",
            bounds_by="intersection",
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
                        dtype="float32",
                    ),
                )
            )

    def test_mosaic_min_nan(self):
        filenames = [l8_224077_20200518_B2_nan, l8_224078_20200518_B2_nan]
        with gw.open(
            filenames,
            band_names=["blue"],
            mosaic=True,
            overlap="min",
            bounds_by="intersection",
            nodata=0,
        ) as src:
            start_values = src.values[
                0,
                0,
                0:10,
            ]
            end_values = src.values[
                0,
                -2,
                -10:,
            ]
            x, y = lonlat_to_xy(-54.78604601, -25.23023330, dst_crs=src)
            j, i = coords_to_indices(x, y, src)
            mid_values = src[0, i : i + 3, j : j + 3].values
            self.assertTrue(
                np.allclose(
                    start_values,
                    np.array(
                        [
                            8482.0,
                            8489.0,
                            8483.0,
                            8547.0,
                            8561.0,
                            8574.0,
                            8616.0,
                            8530.0,
                            8396.0,
                            8125.0,
                        ]
                    ),
                ),
            )
            self.assertTrue(
                np.allclose(
                    mid_values,
                    np.array(
                        [
                            [8384.0, 8183.0, 8049.0],
                            [7934.0, 7867.0, 7885.0],
                            [7861.0, 7826.0, 7721.0],
                        ]
                    ),
                )
            )
            self.assertTrue(
                np.allclose(
                    end_values,
                    np.array(
                        [
                            7409.0,
                            7427.0,
                            7490.0,
                            7444.0,
                            7502.0,
                            7472.0,
                            7464.0,
                            7443.0,
                            7406.0,
                            np.nan,
                        ]
                    ),
                    equal_nan=True,
                ),
            )

    def test_mosaic_mean(self):

        filenames = [l8_224077_20200518_B2_nan, l8_224078_20200518_B2_nan]
        with gw.open(
            filenames,
            band_names=["blue"],
            mosaic=True,
            overlap="mean",
            bounds_by="union",
        ) as src:
            start_values = src.values[
                0,
                0,
                0:10,
            ]
            mid_values = src.values[
                0,
                src.gw.nrows // 2,
                src.gw.ncols // 2 : src.gw.ncols // 2 + 10,
            ]
            end_values = src.values[
                0,
                -2,
                -10:,
            ]
            self.assertTrue(
                np.allclose(
                    mid_values,
                    np.array(
                        [
                            7523,
                            7538,
                            7573,
                            7625,
                            7683,
                            7660,
                            7642,
                            7772,
                            7696,
                            7565,
                        ]
                    ),
                )
            )

            self.assertTrue(
                np.allclose(
                    start_values,
                    np.array(
                        [
                            8482.0,
                            8489.0,
                            8483.0,
                            8547.0,
                            8561.0,
                            8574.0,
                            8616.0,
                            8530.0,
                            8396.0,
                            8125.0,
                        ]
                    ),
                )
            )

            self.assertTrue(
                np.allclose(
                    end_values,
                    np.array(
                        [
                            7409.0,
                            7427.0,
                            7490.0,
                            7444.0,
                            7502.0,
                            7472.0,
                            7464.0,
                            7443.0,
                            7406.0,
                            np.nan,
                        ]
                    ),
                    equal_nan=True,
                )
            )

    def test_footprint_grid(self):
        filenames = [l8_224077_20200518_B2, l8_224078_20200518_B2]
        with gw.open(
            filenames,
            band_names=["blue"],
            mosaic=True,
            overlap="mean",
            bounds_by="intersection",
            nodata=0,
            persist_filenames=True,
        ) as src:
            self.assertEqual(len(src.gw.chunk_grid), 25)
            self.assertEqual(len(src.gw.footprint_grid), 2)
            self.assertEqual(
                filenames, src.gw.footprint_grid.footprint.values.tolist()
            )

        with gw.open(
            filenames,
            band_names=["blue"],
            mosaic=True,
            overlap="mean",
            bounds_by="intersection",
            nodata=0,
            chunks=128,
            persist_filenames=False,
        ) as src:
            self.assertEqual(len(src.gw.chunk_grid), 100)
            with self.assertRaises(AttributeError):
                src.gw.footprint_grid

    def test_has_time_dim(self):
        with gw.open(
            [l8_224078_20200518, l8_224078_20200518], stack_dim="time"
        ) as src:
            self.assertTrue(src.gw.has_time_dim)

    def test_has_time_coord(self):
        with gw.open(
            [l8_224078_20200518, l8_224078_20200518], stack_dim="time"
        ) as src:
            self.assertTrue(src.gw.has_time_coord)

    def test_has_time(self):
        with gw.open(
            [l8_224078_20200518, l8_224078_20200518], stack_dim="time"
        ) as src:
            self.assertTrue(src.gw.has_time)

    def test_has_no_time_coord(self):
        with gw.open(
            [l8_224078_20200518, l8_224078_20200518], stack_dim="time"
        ) as src:
            self.assertFalse(src.drop_vars("time").gw.has_time_coord)

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
        with gw.open(l8_224078_20200518, dtype="float64") as src:
            self.assertEqual(src.dtype, "float64")

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
        test_crs = CRS.from_user_input("epsg:4326")
        with gw.open(l8_224078_20200518) as src:
            result = src.gw.transform_crs(
                dst_crs=4326,
                dst_width=src.gw.ncols,
                dst_height=src.gw.nrows,
                coords_only=True,
            )
            self.assertEqual(test_crs, result.crs)
            self.assertEqual(test_crs, result.gw.crs_to_pyproj)


if __name__ == "__main__":
    unittest.main()
