"""Regression tests for GDAL_NUM_THREADS threading wiring.

The old `warp_extras={'multi': True, 'warp_option': ...}` plumbing produced
`Warning 6: warp options does not support option WARP_EXTRAS / MULTI`
spam (one or two lines per WarpedVRT construction) on rasterio>=1.4 /
GDAL>=3.12. These tests pin the warning-free behavior and confirm that
geowombat's internal rio.Env composes with a user's outer rio.Env.
"""

import contextlib
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import rasterio as rio

import geowombat as gw
from geowombat.data import l8_224078_20200518


@contextlib.contextmanager
def capture_c_stderr():
    """Redirect fd 2 to a temp file so GDAL C-level warnings are captured."""
    sys.stderr.flush()
    saved = os.dup(2)
    tmp = tempfile.TemporaryFile(mode="w+")
    os.dup2(tmp.fileno(), 2)
    try:
        yield tmp
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)
        tmp.close()


def _count_warning_lines(text: str) -> int:
    return sum(
        1 for line in text.splitlines() if line.strip().lower().startswith("warning")
    )


class TestWarpThreadingNoWarnings(unittest.TestCase):
    """gw.open + .compute() must not emit GDAL warp warnings."""

    def _run_compute(self, num_threads: int):
        with gw.config.update(ref_crs=4326, ref_res=0.0005):
            with gw.open(
                l8_224078_20200518,
                chunks=128,
                num_threads=num_threads,
            ) as src:
                src.data.sum().compute()

    def test_default_num_threads_no_warnings(self):
        with capture_c_stderr() as err:
            self._run_compute(num_threads=1)
            err.seek(0)
            out = err.read()
        self.assertEqual(_count_warning_lines(out), 0, out)

    def test_multi_thread_no_warnings(self):
        with capture_c_stderr() as err:
            self._run_compute(num_threads=4)
            err.seek(0)
            out = err.read()
        self.assertEqual(_count_warning_lines(out), 0, out)


class TestRioEnvComposition(unittest.TestCase):
    """A user's outer rio.Env composes with geowombat's inner rio.Env."""

    def test_user_env_visible_at_vrt_construction(self):
        captured = {}
        original_init = rio.vrt.WarpedVRT.__init__

        def spy_init(self, *args, **kwargs):
            # Record the active GDAL env at construction time.
            captured.setdefault("envs", []).append(dict(rio.env.getenv()))
            return original_init(self, *args, **kwargs)

        with patch.object(rio.vrt.WarpedVRT, "__init__", spy_init):
            with rio.Env(GDAL_CACHEMAX=256):
                with gw.config.update(ref_crs=4326, ref_res=0.0005):
                    with gw.open(
                        l8_224078_20200518,
                        chunks=128,
                        num_threads=2,
                    ) as src:
                        src.data.sum().compute()

        envs = captured.get("envs", [])
        self.assertGreater(len(envs), 0, "no WarpedVRT constructions observed")
        # At least one VRT construction should see BOTH the outer GDAL_CACHEMAX
        # and our inner GDAL_NUM_THREADS — i.e. the envs composed correctly.
        composed = [
            e for e in envs
            if e.get("GDAL_CACHEMAX") in (256, "256")
            and str(e.get("GDAL_NUM_THREADS")) == "2"
        ]
        self.assertGreater(
            len(composed),
            0,
            f"no env saw both outer GDAL_CACHEMAX and inner GDAL_NUM_THREADS; "
            f"sample envs: {envs[:3]}",
        )

    def test_default_does_not_set_num_threads(self):
        """num_threads=1 (default) must NOT inject GDAL_NUM_THREADS,
        so a user's outer rio.Env(GDAL_NUM_THREADS=...) flows through."""
        captured = {}
        original_init = rio.vrt.WarpedVRT.__init__

        def spy_init(self, *args, **kwargs):
            captured.setdefault("envs", []).append(dict(rio.env.getenv()))
            return original_init(self, *args, **kwargs)

        with patch.object(rio.vrt.WarpedVRT, "__init__", spy_init):
            with rio.Env(GDAL_NUM_THREADS="ALL_CPUS"):
                with gw.config.update(ref_crs=4326, ref_res=0.0005):
                    with gw.open(
                        l8_224078_20200518,
                        chunks=128,
                    ) as src:
                        src.data.sum().compute()

        envs = captured.get("envs", [])
        self.assertGreater(len(envs), 0)
        # Every VRT construction should see the user's ALL_CPUS untouched.
        overridden = [
            e for e in envs
            if str(e.get("GDAL_NUM_THREADS")) not in ("ALL_CPUS", "None", "")
            and e.get("GDAL_NUM_THREADS") is not None
        ]
        self.assertEqual(
            overridden, [],
            f"default num_threads=1 should not override user's outer "
            f"GDAL_NUM_THREADS=ALL_CPUS; overridden envs: {overridden[:3]}",
        )


if __name__ == "__main__":
    unittest.main()
