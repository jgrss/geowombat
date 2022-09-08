import unittest

import geowombat as gw
from geowombat.data import l8_224077_20200518_B2, l8_224077_20200518_B3, l8_224078_20200518_polygons


class TestClip(unittest.TestCase):
    def test_clip(self):
        with gw.open(
            [l8_224077_20200518_B2, l8_224077_20200518_B3],
            stack_dim="band",
            band_names=["B2", "B3"]
        ) as src:
            src_clip = gw.clip(src, l8_224078_20200518_polygons)
            self.assertEqual(src_clip.crs, src.crs)


if __name__ == '__main__':
    unittest.main()
