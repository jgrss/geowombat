import unittest
import tempfile
from pathlib import Path

import geowombat as gw
from geowombat.data import l8_224077_20200518_B2
from geowombat.data import l8_224077_20200518_B4
import numpy as np
import xarray as xr


def shift(data: xr.DataArray, x: int, y: int) -> xr.DataArray:
    return (
        (
            data.astype('float64').shift(
                shifts={'y': y, 'x': x}, fill_value=data._FillValue
            )
        )
        .fillna(0)
        .astype('uint16')
    )


class TestCOREG(unittest.TestCase):
    def test_coreg_data(self):
        with gw.open(l8_224077_20200518_B2) as target, gw.open(
            l8_224077_20200518_B4
        ) as reference:
            data = gw.coregister(
                target=target,
                reference=reference,
                ws=(256, 256),
                r_b4match=1,
                s_b4match=1,
                max_shift=5,
                resamp_alg_deshift='nearest',
                resamp_alg_calc='cubic',
                out_gsd=[target.gw.celly, reference.gw.celly],
                q=True,
                nodata=(0, 0),
                CPUs=1,
            )
            self.assertTrue(reference.shape == data.shape)

    def test_coreg_transform_data(self):
        with gw.config.update(ref_crs='epsg:8858'):
            with gw.open(l8_224077_20200518_B2, chunks=512) as target, gw.open(
                l8_224077_20200518_B4, chunks=512
            ) as reference:
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
                    out_gsd=[target.gw.celly, reference.gw.celly],
                    q=True,
                    nodata=(0, 0),
                    CPUs=1,
                )
                self.assertTrue(reference.shape == data.shape)

    def test_coreg_shift(self):
        """Tests a 1-pixel shift."""
        with gw.open(l8_224077_20200518_B2) as target, gw.open(
            l8_224077_20200518_B4
        ) as reference:
            with tempfile.TemporaryDirectory() as tmp:
                # Shift by 1 pixel in each direction
                target_shifted = shift(target, x=1, y=1)
                tmp_file = Path(tmp) / '_tmp_shift.tif'
                target_shifted.gw.save(tmp_file, overwrite=True)
                with gw.open(tmp_file) as target_shifted:
                    # Co-register the shifted data
                    shifted = gw.coregister(
                        target=target_shifted,
                        reference=reference,
                        ws=(256, 256),
                        r_b4match=1,
                        s_b4match=1,
                        max_shift=5,
                        resamp_alg_deshift='nearest',
                        resamp_alg_calc='cubic',
                        out_gsd=[target_shifted.gw.celly, reference.gw.celly],
                        q=True,
                        nodata=(0, 0),
                        CPUs=1,
                    )
                    self.assertFalse(
                        np.allclose(
                            target.values[:, :-1, :-1],
                            target_shifted.values[:, :-1, :-1],
                        )
                    )
                    # Check if the corrected data match the original (unshifted) target
                    self.assertTrue(
                        np.allclose(
                            target.values[:, :-1, :-1], shifted.values[:, :-1, :-1]
                        )
                    )


if __name__ == '__main__':
    unittest.main()
