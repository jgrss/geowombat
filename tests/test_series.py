import unittest
import tempfile
from pathlib import Path
import warnings

import geowombat as gw
from geowombat.data import l8_224078_20200518
import rasterio as rio
import numpy as np

IMAGE_LIST = [l8_224078_20200518] * 2

try:
    import jax.numpy as jnp

    JAX_INSTALLED = True
except ImportError:
    JAX_INSTALLED = False


class TemporalMean(gw.TimeModule):
    def __init__(self):
        super(TemporalMean, self).__init__()

    def calculate(self, array):
        sl1 = (
            slice(0, None),
            slice(self.band_dict["red"], self.band_dict["red"] + 1),
            slice(0, None),
            slice(0, None),
        )
        sl2 = (
            slice(0, None),
            slice(self.band_dict["green"], self.band_dict["green"] + 1),
            slice(0, None),
            slice(0, None),
        )
        vi = (array[sl1] - array[sl2]) / ((array[sl1] + array[sl2]) + 1e-9)

        return vi.mean(axis=0).squeeze()


class my_maximum(gw.TimeModule):
    def __init__(self):
        super(my_maximum, self).__init__()

    def calculate(self, array):
        return np.max(array, axis=0).squeeze()


class my_minimum(gw.TimeModule):
    def __init__(self):
        super(my_minimum, self).__init__()

    def calculate(self, array):
        return np.min(array, axis=0).squeeze()


class TestSeries(unittest.TestCase):
    def test_series(self):
        with rio.open(l8_224078_20200518) as src:
            res = src.res
            bounds = src.bounds
        if JAX_INSTALLED:
            with tempfile.TemporaryDirectory() as tmp:
                out_path = Path(tmp) / "test.tif"
                with gw.series(
                    IMAGE_LIST,
                    band_names=["blue", "green", "red"],
                    crs="epsg:32621",
                    res=res,
                    bounds=bounds,
                    nodata=0,
                    num_threads=2,
                    window_size=(512, 512),
                ) as src:
                    src.apply(
                        TemporalMean(),
                        bands=-1,
                        gain=1e-4,
                        processes=False,
                        num_workers=2,
                        outfile=out_path,
                    )
                with gw.open(out_path) as dst:
                    self.assertEqual(dst.gw.nbands, 1)
        else:
            warnings.warn("Could not import jax. Defaulting to numpy.")
            with tempfile.TemporaryDirectory() as tmp:
                out_path = Path(tmp) / "test.tif"
                with gw.series(
                    IMAGE_LIST,
                    band_names=["blue", "green", "red"],
                    crs="epsg:32621",
                    res=res,
                    bounds=bounds,
                    nodata=0,
                    num_threads=2,
                    window_size=(512, 512),
                    transfer_lib="numpy",
                ) as src:
                    src.apply(
                        TemporalMean(),
                        bands=-1,
                        gain=1e-4,
                        processes=False,
                        num_workers=2,
                        outfile=out_path,
                    )
                with gw.open(out_path) as dst:
                    self.assertEqual(dst.gw.nbands, 1)

    def test_series_bigtiff(self):
        with rio.open(l8_224078_20200518) as src:
            res = src.res
            bounds = src.bounds

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "test.tif"
            with gw.series(
                IMAGE_LIST,
                band_names=["blue", "green", "red"],
                crs="epsg:32621",
                res=res,
                bounds=bounds,
                nodata=0,
                num_threads=2,
                window_size=(512, 512),
                transfer_lib="numpy",
            ) as src:
                src.apply(
                    TemporalMean(),
                    bands=-1,
                    gain=1e-4,
                    processes=False,
                    num_workers=2,
                    outfile=out_path,
                    kwargs={"BIGTIFF": "YES"},
                )
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)

    def test_series_multiple(self):
        with rio.open(l8_224078_20200518) as src:
            res = src.res
            bounds = src.bounds

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "test3.tif"
            with gw.series(
                IMAGE_LIST,
                band_names=["blue", "green", "red"],
                crs="epsg:32621",
                res=res,
                bounds=bounds,
                nodata=0,
                num_threads=2,
                window_size=(512, 512),
            ) as src:
                src.apply(
                    ["mean", "max", "cv"],
                    bands=-1,
                    gain=1e-4,
                    processes=False,
                    num_workers=2,
                    outfile=out_path,
                )
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 3)


if __name__ == "__main__":
    unittest.main()
