import unittest

import geopandas as gpd
import numpy as np

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
            src_clip = gw.clip_by_polygon(src, l8_224078_20200518_polygons)
            self.assertEqual(src_clip.crs, src.crs)

    def test_clip_data(self):
        with gw.open(l8_224077_20200518_B2, chunks=64) as src:
            src_crs = src.crs
            src_clip = gw.clip_by_polygon(src, l8_224078_20200518_polygons)
            src_clip_mask = gw.clip_by_polygon(
                src, l8_224078_20200518_polygons, mask_data=True
            )
        df = gpd.read_file(l8_224078_20200518_polygons)
        df = df.to_crs(src_crs)
        with gw.config.update(
            ref_bounds=df.total_bounds.tolist(),
            ref_tar=l8_224077_20200518_B2,
        ):
            with gw.open(l8_224077_20200518_B2, chunks=64) as ref:
                self.assertTrue(
                    np.allclose(ref.data.compute(), src_clip.data.compute())
                )
                # Masked data have different values
                self.assertFalse(
                    np.allclose(
                        ref.data.compute(), src_clip_mask.data.compute()
                    )
                )
                # Masked data null values should equal polygon null values
                poly = gw.polygon_to_array(
                    l8_224078_20200518_polygons, data=ref
                )
                poly_mask = poly.where(lambda x: x == 1)
                res = src_clip_mask * poly_mask
                self.assertTrue(
                    res.isnull().sum().data.compute()
                    == poly_mask.isnull().sum().data.compute()
                )


if __name__ == '__main__':
    unittest.main()
