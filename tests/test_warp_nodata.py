"""Nodata overrides must apply before interpolation during ``gw.open``."""

import gc
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.transform import from_origin

import geowombat as gw
from geowombat.data import l8_224078_20200518


class TestWarpNodata(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(gc.collect)
        self.raster_count = 0

    def _filename(self):
        self.raster_count += 1
        return str(
            Path(self.directory.name) / f'raster{self.raster_count}.tif'
        )

    def _raster(self, nodata=0, file_nodata=None):
        data = np.full((2, 20, 20), 100, dtype='float32')
        data[:, :6, :] = nodata
        data[0, 8:14, 8:14] = nodata
        data[1, 12:18, 12:18] = nodata
        filename = self._filename()
        with rio.open(
            filename,
            'w',
            driver='GTiff',
            height=20,
            width=20,
            count=2,
            dtype=data.dtype,
            transform=from_origin(500000, 4400000, 30, 30),
            crs='EPSG:32621',
            nodata=file_nodata,
        ) as dst:
            dst.write(data)
        return filename

    def _assert_warp(
        self, filename, effective_nodata, open_nodata, method, **config
    ):
        with rio.open(filename) as raw:
            data = raw.read()
            profile = raw.profile
        # Supplying nodata to gw.open must have the same result as declaring
        # it in the source file, including multi-band GDAL mask semantics.
        reference = self._filename()
        profile['nodata'] = effective_nodata
        with rio.open(reference, 'w', **profile) as dst:
            dst.write(data)
        with gw.config.update(**config):
            with gw.open(reference, resampling=method, chunks=128) as ref:
                expected = ref.values
            with gw.open(
                filename,
                nodata=open_nodata,
                resampling=method,
                chunks=128,
            ) as src:
                expected_mask = (
                    np.isnan(expected)
                    if np.isnan(effective_nodata)
                    else expected == effective_nodata
                )
                self.assertTrue(expected_mask.any())
                self.assertTrue((~expected_mask).any())
                np.testing.assert_allclose(src.values, expected)
                np.testing.assert_array_equal(
                    np.isnan(src.gw.mask_nodata().values), expected_mask
                )

    def test_override_without_file_metadata(self):
        for nodata in (0, -32768, np.nan):
            for method in ('nearest', 'bilinear', 'cubic'):
                with self.subTest(nodata=nodata, method=method):
                    self._assert_warp(
                        self._raster(nodata=nodata),
                        nodata,
                        nodata,
                        method,
                        ref_crs='EPSG:32618',
                    )

    def test_override_existing_file_metadata(self):
        for method in ('bilinear', 'cubic'):
            with self.subTest(method=method):
                self._assert_warp(
                    self._raster(file_nodata=-9999),
                    0,
                    0,
                    method,
                    ref_crs='EPSG:32618',
                )

    def test_config_override_takes_precedence(self):
        self._assert_warp(
            self._raster(file_nodata=-9999),
            0,
            -9999,
            'cubic',
            ref_crs='EPSG:32618',
            nodata=0,
        )

    def test_override_during_same_crs_resampling(self):
        self._assert_warp(self._raster(), 0, 0, 'cubic', ref_res=17)

    def test_file_metadata_used_without_override(self):
        self._assert_warp(
            self._raster(file_nodata=0),
            0,
            None,
            'cubic',
            ref_crs='EPSG:32618',
        )

    def test_same_grid_override_preserves_other_values(self):
        filename = self._raster(nodata=-9999, file_nodata=0)
        with rio.open(filename, 'r+') as dst:
            data = dst.read()
            data[:, 15:18, :3] = 0
            dst.write(data)
        with rio.open(filename) as raw:
            expected = raw.read()
        with gw.open(filename, nodata=-9999, chunks=8) as src:
            np.testing.assert_array_equal(src.values, expected)
            np.testing.assert_array_equal(
                np.isnan(src.gw.mask_nodata().values), expected == -9999
            )

    def test_issue_251_landsat_reprojection(self):
        self._assert_warp(
            l8_224078_20200518,
            0,
            0,
            'cubic',
            ref_crs='EPSG:32618',
        )


if __name__ == '__main__':
    unittest.main()
