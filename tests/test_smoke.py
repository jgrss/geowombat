"""Build / import smoke tests.

These guard the package's importability and, in particular, that the
compiled Cython extensions load against the installed NumPy. They are the
first line of defense for the NumPy 1.x / 2.x ABI compatibility story: a
mismatched build raises ``ImportError`` here before any functional test runs.
"""

import importlib
import unittest

import numpy as np


class TestSmoke(unittest.TestCase):
    def test_import_geowombat(self):
        import geowombat as gw

        self.assertIsInstance(gw.__version__, str)
        self.assertTrue(gw.__version__)

    def test_public_api_present(self):
        import geowombat as gw

        for name in ('open', 'config', 'series', 'load'):
            self.assertTrue(hasattr(gw, name), msg=f'gw.{name} is missing')

    def test_compiled_extensions_load(self):
        # A NumPy-ABI mismatch (extension built against a different NumPy
        # major) surfaces as ImportError on these imports.
        moving = importlib.import_module('geowombat.moving._moving')
        fusion = importlib.import_module('geowombat.radiometry._fusion')
        self.assertIsNotNone(moving)
        self.assertIsNotNone(fusion)

    def test_numpy_version_supported(self):
        # Informational in CI logs; also asserts we are within the
        # supported runtime range declared in pyproject.toml.
        major = int(np.__version__.split('.')[0])
        print(f'Running against NumPy {np.__version__}')
        self.assertGreaterEqual(major, 1)
        self.assertLess(major, 3)


class TestDataPackaging(unittest.TestCase):
    """Guard that bundled data files are actually installed.

    ``geowombat.data`` exposes module-level path constants (e.g.
    ``stac_training``) built from files that must be shipped by the build
    backend's data-install list. A missing entry passes in an editable
    install (which reads the source tree) but breaks a real wheel/sdist
    install, so check it explicitly here.
    """

    def test_bundled_data_paths_exist(self):
        import os

        from geowombat import data as gwdata

        data_dir = os.path.dirname(gwdata.__file__)
        exts = ('.tif', '.TIF', '.nc', '.gpkg', '.geojson', '.txt', '.gz')
        missing = []
        for name in dir(gwdata):
            if name.startswith('_'):
                continue
            val = getattr(gwdata, name)
            # Only inspect string path constants that point at a file
            # inside the data directory with a known data extension.
            if (
                isinstance(val, str)
                and os.path.abspath(val).startswith(os.path.abspath(data_dir))
                and val.endswith(exts)
            ):
                if not os.path.isfile(val):
                    missing.append(f'{name} -> {os.path.basename(val)}')
        self.assertEqual(
            missing, [], msg=f'Bundled data files not installed: {missing}'
        )


if __name__ == '__main__':
    unittest.main()
