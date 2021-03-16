import unittest

import geowombat as gw
from geowombat.data import l8_224078_20200518

from testfixtures import LogCapture


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

    def test_config_defaults(self):

        with gw.open(l8_224078_20200518) as src:

            self.assertEqual(src.gw.config['scale_factor'], 1)

            for config_default in ['with_config', 'ignore_warnings']:
                self.assertFalse(src.gw.config[config_default])

            for config_default in ['sensor', 'nodata', 'ref_image', 'ref_bounds', 'ref_crs', 'ref_res', 'ref_tar', 'compress']:
                self.assertIsNone(src.gw.config[config_default])

    def test_config_set_res(self):

        with gw.config.update(ref_res=100):

            with gw.open(l8_224078_20200518) as src:
                self.assertEqual(src.gw.celly, 100)

        with gw.open(l8_224078_20200518) as src:
            self.assertEqual(src.gw.celly, 30)

    def test_with_config(self):

        with gw.config.update():

            with gw.open(l8_224078_20200518) as src:
                self.assertTrue(src.gw.config['with_config'])

        with gw.open(l8_224078_20200518) as src:
            self.assertFalse(src.gw.config['with_config'])

    def test_warnings_ignore(self):

        with LogCapture() as log:

            with gw.config.update(sensor='l7', ignore_warnings=True):

                with gw.open(l8_224078_20200518) as src:
                    pass

            self.assertNotIn(('geowombat.backends.xarray_', 'WARNING', "  The new bands, ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'], do not match the sensor bands, [1, 2, 3]."),
                             log)


if __name__ == '__main__':
    unittest.main()
