import unittest
from datetime import datetime, timezone

from geowombat.data import l8_224078_20200127_meta
from geowombat.radiometry import RadTransforms

import pandas as pd


RT = RadTransforms()


class TestRadiometrics(unittest.TestCase):
    def test_landsat_metadata(self):
        meta = RT.get_landsat_coefficients(str(l8_224078_20200127_meta))

        self.assertTrue(meta.sensor, 'l8')
        self.assertEqual(
            pd.Timestamp(meta.date_acquired).round('D').to_pydatetime(),
            datetime(2020, 1, 28, 0, 0, tzinfo=timezone.utc),
        )


if __name__ == '__main__':
    unittest.main()
