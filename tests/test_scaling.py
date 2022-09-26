import unittest

import geowombat as gw
from geowombat.data import l8_224078_20200518

import numpy as np


NODATA_LOC = (slice(0, None), 0, 0)
VALID_LOC = (slice(0, None), 100, 100)
SCALE_FACTOR = 1e-4
NODATA_VALUE = 0
VALID_DATA = np.array([7497, 6771, 6083])


class TestScaling(unittest.TestCase):
    def test_no_scaling(self):
        with gw.open(l8_224078_20200518) as src:
            self.assertTrue(
                np.allclose(src[NODATA_LOC].values, np.zeros(src.gw.nbands))
            )
            self.assertTrue(np.allclose(src[VALID_LOC].values, VALID_DATA))

    def test_scaling_apply(self):
        """The metadata scales and offsets are 1 and 0, respectively.

        Therefore, no scaling is applied even with the use of `scale_data`.
        """
        with gw.open(l8_224078_20200518, scale_data=True) as src:
            self.assertTrue(
                np.allclose(src[NODATA_LOC].values, np.zeros(src.gw.nbands))
            )
            self.assertTrue(np.allclose(src[VALID_LOC].values, VALID_DATA))

    def test_scaling_user(self):
        """The use of `scale_factor` overrides the raster metadata."""
        with gw.open(
            l8_224078_20200518, scale_data=True, scale_factor=SCALE_FACTOR
        ) as src:
            self.assertTrue(
                np.allclose(src[NODATA_LOC].values, np.zeros(src.gw.nbands))
            )
            self.assertTrue(
                np.allclose(src[VALID_LOC].values, VALID_DATA * SCALE_FACTOR)
            )

    def test_scaling_nodata_user(self):
        """The use of `scale_factor` and `nodata` overrides the raster
        metadata."""
        with gw.open(
            l8_224078_20200518,
            scale_data=True,
            scale_factor=SCALE_FACTOR,
            nodata=NODATA_VALUE,
        ) as src:
            self.assertTrue(np.all(np.isnan(src[NODATA_LOC].values)))
            self.assertTrue(
                np.allclose(src[VALID_LOC].values, VALID_DATA * SCALE_FACTOR)
            )

    def test_scaling_config(self):
        """The use of `scale_factor` in the config overrides all."""
        with gw.config.update(scale_factor=SCALE_FACTOR):
            with gw.open(
                l8_224078_20200518, scale_data=True, scale_factor=255.0
            ) as src:
                self.assertTrue(
                    np.allclose(src[NODATA_LOC].values, np.zeros(src.gw.nbands))
                )
                self.assertTrue(
                    np.allclose(src[VALID_LOC].values, VALID_DATA * SCALE_FACTOR)
                )

    def test_scaling_nodata_config(self):
        """The use of `scale_factor` and `nodata` in the config overrides
        all."""
        with gw.config.update(scale_factor=SCALE_FACTOR, nodata=NODATA_VALUE):
            with gw.open(
                l8_224078_20200518, scale_data=True, scale_factor=255.0, nodata=-999
            ) as src:
                self.assertTrue(np.all(np.isnan(src[NODATA_LOC].values)))
                self.assertTrue(
                    np.allclose(src[VALID_LOC].values, VALID_DATA * SCALE_FACTOR)
                )


if __name__ == '__main__':
    unittest.main()
