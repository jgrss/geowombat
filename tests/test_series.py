import unittest
import tempfile
from pathlib import Path

import geowombat as gw
from geowombat.data import l8_224078_20200518
import rasterio as rio


IMAGE_LIST = [l8_224078_20200518] * 2


class TemporalMean(gw.TimeModule):
    def __init__(self):
        super(TemporalMean, self).__init__()
    def calculate(self, array):
        sl1 = (slice(0, None), slice(self.band_dict['red'], self.band_dict['red']+1), slice(0, None), slice(0, None))
        sl2 = (slice(0, None), slice(self.band_dict['green'], self.band_dict['green']+1), slice(0, None), slice(0, None))
        vi = (array[sl1] - array[sl2]) / ((array[sl1] + array[sl2]) + 1e-9)

        return vi.mean(axis=0).squeeze()


class TestSeries(unittest.TestCase):
    def test_series(self):
        with rio.open(l8_224078_20200518) as src:
            res = src.res
            bounds = src.bounds

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / 'test.tif'
            with gw.series(
                IMAGE_LIST,
                band_names=['blue', 'green', 'red'],
                crs='epsg:32621',
                res=res,
                bounds=bounds,
                nodata=0,
                num_threads=2,
                window_size=(512, 512)
            ) as src:
                src.apply(
                    TemporalMean(),
                    bands=-1,
                    gain=1e-4,
                    processes=False,
                    num_workers=2,
                    outfile=out_path
                )
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)


if __name__ == '__main__':
    unittest.main()
