"""Tests for geowombat.core.util helpers."""

import unittest

from geowombat.core.util import get_file_extension


class TestGetFileExtension(unittest.TestCase):
    """``get_file_extension`` must recognize the extension of a URL even
    when a signed-cloud-storage query string trails it.
    """

    def test_local_path_unchanged(self):
        self.assertEqual(get_file_extension('/data/scene.tif').f_ext, '.tif')
        self.assertEqual(get_file_extension('scene.nc').f_ext, '.nc')
        self.assertEqual(get_file_extension('scene.tar.gz').f_ext, '.gz')

    def test_strips_url_query_string(self):
        url = (
            'https://example.blob.core.windows.net/naip/scene.tif'
            '?st=2024-01-01T00%3A00%3A00Z'
            '&se=2024-01-02T00%3A00%3A00Z'
            '&sig=ABC%2BDEF'
        )
        self.assertEqual(get_file_extension(url).f_ext, '.tif')

    def test_strips_url_fragment(self):
        self.assertEqual(
            get_file_extension('http://x/y.tif#section').f_ext, '.tif',
        )

    def test_strips_query_and_fragment(self):
        self.assertEqual(
            get_file_extension('http://x/y.tif?a=1#frag').f_ext, '.tif',
        )

    def test_accepts_pathlib(self):
        from pathlib import Path
        self.assertEqual(get_file_extension(Path('/x/y.tif')).f_ext, '.tif')


if __name__ == '__main__':
    unittest.main()
