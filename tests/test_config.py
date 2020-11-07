import unittest

import geowombat as gw
from geowombat.data import l8_224078_20200518


class TestConfig(unittest.TestCase):

    def test_config_bands(self):

        with gw.open(l8_224078_20200518) as src:
            self.assertEqual(','.join(map(str, src.band.values.tolist())), '1,2,3')

    def test_config_bands_set_none(self):

        with gw.config.update(sensor=None):

            with gw.open(l8_224078_20200518) as src:
                self.assertEqual(','.join(map(str, src.band.values.tolist())), '1,2,3')

    def test_config_bands_set(self):

        with gw.config.update(sensor='bgr'):

            with gw.open(l8_224078_20200518) as src:
                self.assertEqual(','.join(src.band.values.tolist()), 'blue,green,red')

    def test_config_bands_set_override(self):

        with gw.config.update(sensor='bgr'):

            with gw.open(l8_224078_20200518, band_names=['b1', 'b2', 'b3']) as src:
                self.assertEqual(','.join(src.band.values.tolist()), 'b1,b2,b3')


if __name__ == '__main__':
    unittest.main()
