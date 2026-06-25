"""Tests for backend helpers in geowombat.backends.xarray_rasterio_."""

import unittest

import numpy as np

from geowombat.backends.xarray_rasterio_ import _parse_envi


class TestParseEnvi(unittest.TestCase):
    def test_parses_wavelength_and_fwhm_vectors(self):
        meta = {
            'wavelength': '{0.45, 0.56, 0.66}',
            'fwhm': '{0.02, 0.02, 0.02}',
            'description': '{some sensor}',
        }
        parsed = _parse_envi(meta)

        self.assertTrue(np.issubdtype(parsed['wavelength'].dtype, np.floating))
        self.assertTrue(np.allclose(parsed['wavelength'], [0.45, 0.56, 0.66]))
        self.assertTrue(np.allclose(parsed['fwhm'], [0.02, 0.02, 0.02]))

    def test_non_vector_keys_pass_through(self):
        # Keys without a registered parser go through ``default`` (strip
        # the braces, keep the string).
        meta = {'description': '{some sensor}'}
        parsed = _parse_envi(meta)
        self.assertEqual(parsed['description'], 'some sensor')

    def test_tolerates_trailing_comma_and_whitespace(self):
        meta = {'wavelength': '{1.0, 2.0,}'}
        parsed = _parse_envi(meta)
        self.assertTrue(np.allclose(parsed['wavelength'], [1.0, 2.0]))


if __name__ == '__main__':
    unittest.main()
