import tempfile
import unittest
from pathlib import Path

import dask
import rasterio as rio

import geowombat as gw
from geowombat.data import (
    l8_224077_20200518_B2,
    l8_224077_20200518_B3,
    l8_224078_20200518,
    l8_224078_20200518_B2,
)

NODATA = 32768


class TestWrite(unittest.TestCase):
    def test_to_netcdf(self):
        bands = ["blue", "green", "red"]
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "test.nc"
            with gw.open(l8_224078_20200518, band_names=bands) as src:
                try:
                    (
                        src.gw.set_nodata(0, 32768)
                        .astype("uint16")
                        .gw.to_netcdf(filename=out_path, overwrite=True)
                    )
                except Exception as e:
                    # If any exception is raised, fail the test with a message
                    self.fail(f"An error occurred during saving: {e}")

    def test_to_raster(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "test.tif"

            with gw.open(l8_224078_20200518) as src:
                data = src.gw.set_nodata(0, 32768).astype("uint16")
                data.gw.to_raster(
                    filename=out_path,
                    overwrite=True,
                    tags={"TEST_METADATA": "TEST_VALUE"},
                    compress="lzw",
                    num_workers=2,
                )

                with gw.open(out_path) as tmp_src:
                    # Compare array values
                    self.assertTrue(data.equals(tmp_src))
                    # Compare attributes
                    self.assertTrue(
                        data.gw.nodataval == tmp_src.gw.nodataval == NODATA
                    )
                    self.assertEqual(data.gw.dtype, tmp_src.dtype)
                    self.assertTrue(hasattr(tmp_src, "TEST_METADATA"))
                    self.assertEqual(tmp_src.TEST_METADATA, "TEST_VALUE")

            with rio.open(out_path) as rio_src:
                self.assertTrue(rio_src.nodata == NODATA)

    def test_to_raster_multi(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "test.tif"
            with gw.open(
                [l8_224077_20200518_B2, l8_224077_20200518_B3],
                stack_dim="band",
                band_names=["B2", "B3"],
            ) as src:
                # Xarray drops attributes
                data = src.gw.set_nodata(0, 32768).astype("uint16")
                attrs = data.attrs.copy()
                # Apply operations on the DataArray
                data = (data * 10.0).assign_attrs(**attrs)
                # Write the data to a GeoTiff
                data.gw.to_raster(out_path, overwrite=True)
                self.assertTrue(out_path.is_file())
                with gw.open(out_path, band_names=["B2", "B3"]) as tmp_src:
                    # Compare array values
                    self.assertTrue(data.equals(tmp_src))
                    # Compare attributes
                    self.assertTrue(
                        data.gw.nodataval == tmp_src.gw.nodataval == NODATA
                    )
                    self.assertEqual(data.gw.dtype, tmp_src.dtype)

    def test_bigtiff(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "test.tif"
            with gw.config.update({"bigtiff": True}):
                with gw.open(l8_224078_20200518) as src:
                    data = src.gw.set_nodata(0, NODATA, dtype="uint16")
                    (
                        data.gw.save(
                            filename=out_path,
                            overwrite=True,
                            compress="lzw",
                            num_workers=2,
                        )
                    )

                    with gw.open(out_path) as tmp_src:
                        # Compare array values
                        self.assertTrue(data.equals(tmp_src))
                        # Compare attributes
                        self.assertTrue(
                            data.gw.nodataval == tmp_src.gw.nodataval == NODATA
                        )
                        self.assertEqual(data.gw.dtype, tmp_src.dtype)

            with rio.open(out_path) as rio_src:
                self.assertTrue(rio_src.nodata == NODATA)

    def test_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "test.tif"
            with gw.open(l8_224078_20200518) as src:
                data = src.gw.set_nodata(0, NODATA, dtype="uint16")
                (
                    data.gw.save(
                        filename=out_path,
                        overwrite=True,
                        tags={"TEST_METADATA": "TEST_VALUE"},
                        compress="lzw",
                        num_workers=2,
                    )
                )
                with gw.open(out_path) as tmp_src:
                    # Compare array values
                    self.assertTrue(data.equals(tmp_src))
                    # Compare attributes
                    self.assertTrue(
                        data.gw.nodataval == tmp_src.gw.nodataval == NODATA
                    )
                    self.assertEqual(data.gw.dtype, tmp_src.dtype)
                    self.assertTrue(hasattr(tmp_src, "TEST_METADATA"))
                    self.assertEqual(tmp_src.TEST_METADATA, "TEST_VALUE")

            with rio.open(out_path) as rio_src:
                self.assertTrue(rio_src.nodata == NODATA)

    def test_save_small(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "test.tif"
            with gw.open(l8_224078_20200518) as src:
                data = src.gw.set_nodata(0, NODATA, dtype="uint16")
                data = data[:, :1, :2]

                try:
                    data.gw.save(
                        filename=out_path,
                        overwrite=True,
                        tags={"TEST_METADATA": "TEST_VALUE"},
                        compress="none",
                        num_workers=1,
                    )
                except ValueError:
                    self.fail("The small array write test failed.")

    def test_delayed_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "test.tif"
            with gw.open(l8_224078_20200518) as src:
                data = src.gw.set_nodata(0, NODATA, dtype="uint16")
                tasks = [
                    gw.save(
                        data,
                        filename=out_path,
                        tags={"TEST_METADATA": "TEST_VALUE"},
                        compress="lzw",
                        num_workers=2,
                        compute=False,
                        overwrite=True,
                    )
                ]
            dask.compute(tasks, num_workers=2)
            with gw.open(out_path) as tmp_src:
                # Compare array values
                self.assertTrue(data.equals(tmp_src))
                # Compare attributes
                self.assertTrue(
                    data.gw.nodataval == tmp_src.gw.nodataval == NODATA
                )
                self.assertEqual(data.gw.dtype, tmp_src.dtype)
                self.assertTrue(hasattr(tmp_src, "TEST_METADATA"))
                self.assertEqual(tmp_src.TEST_METADATA, "TEST_VALUE")

            with rio.open(out_path) as rio_src:
                self.assertTrue(rio_src.nodata == NODATA)

    def test_mosaic_save_single_band(self):
        filenames = [l8_224077_20200518_B2, l8_224078_20200518_B2]

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                with gw.open(
                    filenames,
                    band_names=["blue"],
                    mosaic=True,
                    bounds_by="union",
                    nodata=0,
                ) as src:
                    src.gw.save(Path(temp_dir) / "test.tif", overwrite=True)

            except Exception as e:
                # If any exception is raised, fail the test with a message
                self.fail(f"An error occurred during saving: {e}")


if __name__ == "__main__":
    unittest.main()
