import tempfile
import unittest
from pathlib import Path

import numpy as np
from osgeo import gdal, osr
from rasterio.coords import BoundingBox

import geowombat as gw

NODATA_VALUE = 255
RASTER_ORIGIN = (1238095, -2305756)


def create_nodata_raster(filename: str):
    ncol = 100
    nrow = 100

    egdata = np.random.randint(0, 10, (nrow, ncol))
    # add some nodata
    egdata[:10, :10] = NODATA_VALUE

    pixelWidth = 10
    pixelHeight = -10
    originX = RASTER_ORIGIN[0]
    originY = RASTER_ORIGIN[1]
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(filename, ncol, nrow, 1, gdal.GDT_Byte)
    outRaster.SetGeoTransform(
        (originX, pixelWidth, 0, originY, 0, pixelHeight)
    )
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(egdata)
    outband.SetNoDataValue(NODATA_VALUE)
    outband.FlushCache()
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(3577)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband = None
    outRaster = None


def load(filename, bounds=None, config_nodata=None, open_nodata=None):
    with gw.config.update(ref_bounds=bounds, nodata=config_nodata):
        with gw.open(filename, dtype='uint8', nodata=open_nodata) as src:
            nodata_block = src[:, :10, :10]

    return nodata_block


class TestNodata(unittest.TestCase):
    def test_nodata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_file = str(Path(tmp) / 'tmp_nodata.tif')
            create_nodata_raster(tmp_file)
            nodata_block = load(tmp_file)
            # The 'no data' values should be read
            self.assertTrue((nodata_block.values == NODATA_VALUE).all())
            self.assertEqual(
                nodata_block.attrs['nodatavals'],
                (NODATA_VALUE,) * nodata_block.gw.nbands,
            )
            self.assertEqual(nodata_block.attrs['_FillValue'], NODATA_VALUE)
            self.assertEqual(nodata_block.gw.nodataval, NODATA_VALUE)

    def test_nodata_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_file = str(Path(tmp) / 'tmp_nodata.tif')
            create_nodata_raster(tmp_file)
            nodata_block = load(tmp_file).gw.mask_nodata()
            # The 'no data' values should be nans
            self.assertTrue(np.isnan(nodata_block.values).all())
            self.assertEqual(
                nodata_block.attrs['nodatavals'],
                (NODATA_VALUE,) * nodata_block.gw.nbands,
            )
            self.assertEqual(nodata_block.attrs['_FillValue'], NODATA_VALUE)
            self.assertEqual(nodata_block.gw.nodataval, NODATA_VALUE)

    def test_nodata_user(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_file = str(Path(tmp) / 'tmp_nodata.tif')
            create_nodata_raster(tmp_file)
            # Setting 'no data' explicitly should leave 255s
            nodata_block = load(tmp_file, open_nodata=0)
            self.assertTrue((nodata_block.values == NODATA_VALUE).all())
            # ...but change the attributes
            self.assertEqual(
                nodata_block.attrs['nodatavals'], (0,) * nodata_block.gw.nbands
            )
            self.assertEqual(nodata_block.attrs['_FillValue'], 0)
            self.assertEqual(nodata_block.gw.nodataval, 0)

            # Masking should set 'no data'
            self.assertFalse(
                np.isnan(nodata_block.gw.mask_nodata().values).any()
            )
            # ...but not persist
            self.assertTrue((nodata_block.values == NODATA_VALUE).all())
            self.assertEqual(
                nodata_block.attrs['nodatavals'], (0,) * nodata_block.gw.nbands
            )
            self.assertEqual(nodata_block.attrs['_FillValue'], 0)
            self.assertEqual(nodata_block.gw.nodataval, 0)

            # Change the 'no data' value and attributes
            nodata_block = nodata_block.gw.set_nodata(
                src_nodata=0,
                dst_nodata=255,
            )
            self.assertTrue(
                np.isnan(nodata_block.gw.mask_nodata().values).all()
            )
            self.assertEqual(
                nodata_block.attrs['nodatavals'],
                (255,) * nodata_block.gw.nbands,
            )
            self.assertEqual(nodata_block.attrs['_FillValue'], 255)
            self.assertEqual(nodata_block.gw.nodataval, 255)

    def test_nodata_user_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_file = str(Path(tmp) / 'tmp_nodata.tif')
            create_nodata_raster(tmp_file)
            # Setting 'no data' explicitly should leave 255s
            nodata_block = load(tmp_file, config_nodata=0).gw.mask_nodata()
            self.assertTrue((nodata_block.values == NODATA_VALUE).all())
            # ...but change the attributes
            self.assertEqual(
                nodata_block.attrs['nodatavals'], (0,) * nodata_block.gw.nbands
            )
            self.assertEqual(nodata_block.attrs['_FillValue'], 0)
            self.assertEqual(nodata_block.gw.nodataval, 0)

            # Masking should set 'no data'
            self.assertFalse(
                np.isnan(nodata_block.gw.mask_nodata().values).any()
            )
            # ...but not persist
            self.assertTrue((nodata_block.values == NODATA_VALUE).all())
            self.assertEqual(
                nodata_block.attrs['nodatavals'], (0,) * nodata_block.gw.nbands
            )
            self.assertEqual(nodata_block.attrs['_FillValue'], 0)
            self.assertEqual(nodata_block.gw.nodataval, 0)

            # Change the 'no data' value and attributes
            nodata_block = nodata_block.gw.set_nodata(
                src_nodata=0,
                dst_nodata=255,
            )
            self.assertTrue(
                np.isnan(nodata_block.gw.mask_nodata().values).all()
            )
            self.assertEqual(
                nodata_block.attrs['nodatavals'],
                (255,) * nodata_block.gw.nbands,
            )
            self.assertEqual(nodata_block.attrs['_FillValue'], 255)
            self.assertEqual(nodata_block.gw.nodataval, 255)

    def test_nodata_bbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_file = str(Path(tmp) / 'tmp_nodata.tif')
            create_nodata_raster(tmp_file)
            bounds = BoundingBox(
                left=RASTER_ORIGIN[0],
                bottom=RASTER_ORIGIN[1] - 60,
                right=RASTER_ORIGIN[0] + 60,
                top=RASTER_ORIGIN[1],
            )
            nodata_block = load(tmp_file, bounds=bounds)
            self.assertEqual(nodata_block.mean().values, NODATA_VALUE)
            self.assertEqual(
                nodata_block.attrs['nodatavals'],
                (NODATA_VALUE,) * nodata_block.gw.nbands,
            )
            self.assertEqual(nodata_block.attrs['_FillValue'], NODATA_VALUE)
            self.assertEqual(nodata_block.gw.nodataval, NODATA_VALUE)


if __name__ == '__main__':
    unittest.main()
