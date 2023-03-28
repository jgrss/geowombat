import unittest

import numpy as np
import geopandas as gpd

import geowombat as gw
from geowombat.data import (
    l8_224077_20200518_B2,
    l8_224077_20200518_B3,
    l8_224078_20200518_polygons,
)


class TestClip(unittest.TestCase):
    def test_clip_crs(self):
        with gw.open(
            [l8_224077_20200518_B2, l8_224077_20200518_B3],
            stack_dim="band",
            band_names=["B2", "B3"],
        ) as src:
            src_clip = gw.clip(src, l8_224078_20200518_polygons)
            self.assertEqual(src_clip.crs, src.crs)

    def test_clip_data(self):
        with gw.open(l8_224077_20200518_B2, chunks=64) as src:
            src_crs = src.crs
            src_clip = gw.clip(src, l8_224078_20200518_polygons)
        df = gpd.read_file(l8_224078_20200518_polygons)
        df = df.to_crs(src_crs)
        with gw.config.update(
            ref_bounds=df.total_bounds.tolist(), ref_tar=l8_224077_20200518_B2
        ):
            with gw.open(l8_224077_20200518_B2, chunks=64) as ref:
                self.assertTrue(
                    np.allclose(ref.data.compute(), src_clip.data.compute())
                )


if __name__ == '__main__':
    unittest.main()
