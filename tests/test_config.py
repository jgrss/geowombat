import unittest

import geowombat as gw
from geowombat.data import l8_224078_20200518
from geowombat.data import l8_224077_20200518_B2, l8_224077_20200518_B3, l8_224077_20200518_B4
from geowombat.data import rgbn

from pyproj import CRS
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

        test_crs = CRS.from_user_input('epsg:32621')
        with gw.open(l8_224078_20200518) as src:
            self.assertTrue(src.crs, 32621)
            self.assertTrue(test_crs, src.gw.crs_to_pyproj)
            self.assertFalse(src.gw.config['with_config'])

    def test_config_crs_3857(self):
        """Test warp to Pseudo-Mercator / EPSG:3857
        """
        test_crs = CRS.from_user_input('epsg:3857')
        with gw.config.update(ref_crs='epsg:3857', ref_res=100):
            with gw.open(l8_224078_20200518) as src:
                self.assertTrue(src.resampling, 'nearest')
                self.assertTrue(src.crs, 3857)
                self.assertTrue(test_crs, src.gw.crs_to_pyproj)
                self.assertTrue(src.res, (100.0, 100.0))

    def test_config_crs_8858(self):
        """Test warp to Equal Earth Americas / EPSG:8858
        """
        test_crs = CRS.from_user_input('epsg:8858')
        with gw.config.update(ref_crs='epsg:8858', ref_res=100):
            with gw.open(l8_224078_20200518) as src:
                self.assertTrue(src.resampling, 'nearest')
                self.assertTrue(src.crs, 8858)
                self.assertTrue(test_crs, src.gw.crs_to_pyproj)
                self.assertTrue(src.res, (100.0, 100.0))

        test_crs = CRS.from_user_input('+proj=eqearth')
        with gw.config.update(ref_crs='+proj=eqearth', ref_res=100):
            with gw.open(l8_224078_20200518) as src:
                self.assertTrue(src.resampling, 'nearest')
                self.assertTrue(src.crs, 8858)
                self.assertTrue(test_crs, src.gw.crs_to_pyproj)
                self.assertTrue(src.res, (100.0, 100.0))

    def test_multiple_crs(self):
        test_crs = CRS.from_user_input('epsg:32621')
        filenames = [l8_224078_20200518, l8_224078_20200518]
        with gw.config.update(ref_image=l8_224077_20200518_B2):
            with gw.open(
                filenames,
                band_names=['blue', 'green', 'red'],
                time_names=['t1', 't2']
            ) as src:
                self.assertTrue(src.crs, 32621)
                self.assertTrue(test_crs, src.gw.crs_to_pyproj)

    def test_unique_crs(self):
        proj4 = "+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"
        with gw.open(rgbn) as src:
            dst = src.gw.transform_crs(proj4, dst_res=(10, 10), resampling='bilinear')
            self.assertEqual(dst.crs, proj4)
            self.assertEqual(dst.gw.crs_to_pyproj, proj4)

    def test_warnings_ignore(self):
        with LogCapture() as log:
            with gw.config.update(sensor='l7', ignore_warnings=True):
                with gw.open(l8_224078_20200518) as src:
                    pass

            self.assertNotIn(('geowombat.backends.xarray_', 'WARNING', "  The new bands, ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'], do not match the sensor bands, [1, 2, 3]."),
                             log)


if __name__ == '__main__':
    unittest.main()
