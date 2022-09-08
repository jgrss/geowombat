import unittest

import geowombat as gw
from geowombat.data import l8_224077_20200518_B2
from geowombat.data import l8_224077_20200518_B4


class TestCOREG(unittest.TestCase):
    def test_coreg(self):
        # with gw.config.update(ref_crs="epsg:8858"):
        with gw.open(l8_224077_20200518_B2) as target, gw.open(l8_224077_20200518_B4) as reference:
            data = gw.coregister(
                target=target,
                reference=reference,
                ws=(256, 256),
                r_b4match=1,
                s_b4match=1,
                max_shift=5,
                resamp_alg_deshift='nearest',
                resamp_alg_calc='cubic',
                out_gsd=[30.0, 30.0],
                q=True,
                nodata=(0, 0),  # reference, data to be shifted
                CPUs=1
            )
            self.assertTrue(reference.shape == data.shape)

    def test_coreg_transform(self):
        chunks = {'band': -1, 'y': 512, 'x': 512}
        with gw.config.update(ref_crs="epsg:8858"):
            with gw.open(l8_224077_20200518_B2, chunks=chunks) as target, \
                gw.open(l8_224077_20200518_B4, chunks=chunks) as reference:
                data = gw.coregister(
                    target=target,
                    reference=reference,
                    wkt_version='WKT2_2019',
                    ws=(256, 256),
                    r_b4match=1,
                    s_b4match=1,
                    max_shift=5,
                    resamp_alg_deshift='nearest',
                    resamp_alg_calc='cubic',
                    out_gsd=[30.0, 30.0],
                    q=True,
                    nodata=(0, 0),  # reference, data to be shifted
                    CPUs=1
                )
                self.assertTrue(reference.shape == data.shape)


if __name__ == '__main__':
    unittest.main()
