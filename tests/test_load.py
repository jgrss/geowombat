import unittest
import datetime

import geowombat as gw
from geowombat.data import l3b_s2b_00390821jxn0l2a_20210319_20220730_c01
import numpy as np


IMAGE_LIST = [
    l3b_s2b_00390821jxn0l2a_20210319_20220730_c01,
    l3b_s2b_00390821jxn0l2a_20210319_20220730_c01,
    l3b_s2b_00390821jxn0l2a_20210319_20220730_c01
]
IMAGE_DATES = [
    datetime.datetime(2020, 5, 18, 0, 0),
    datetime.datetime(2020, 6, 18, 0, 0),
    datetime.datetime(2020, 7, 18, 0, 0)
]


class TestLoad(unittest.TestCase):
    def test_load(self):
        data_slice = (
            slice(0, None),
            slice(0, None),
            slice(0, 64),
            slice(0, 64)
        )
        dates, y = gw.load(
            IMAGE_LIST,
            IMAGE_DATES,
            ['blue'],
            chunks={'band': -1, 'y': 64, 'x': 64},
            nodata=65535,
            data_slice=data_slice,
            num_workers=1
        )

        self.assertTrue(isinstance(y, np.ndarray))
        self.assertEqual(y.shape, (3, 1, 64, 64))


if __name__ == '__main__':
    unittest.main()
