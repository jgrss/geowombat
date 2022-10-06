import unittest
from datetime import datetime, timezone
import tempfile
import tarfile
from pathlib import Path

import geowombat as gw
from geowombat.data import (
    l8_224078_20200127_meta,
    l7_225078_20110306_B1,
    l7_225078_20110306_ang,
    wrs2,
)
from geowombat.bin import extract_espa_tools
from geowombat.radiometry import RadTransforms, landsat_pixel_angles

import numpy as np
import pandas as pd
import geopandas as gpd


RT = RadTransforms()


def get_pathrow() -> gpd.GeoDataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(wrs2, mode='r:gz') as f:
            f.extractall(tmp)
            df = gpd.read_file(Path(tmp) / 'wrs2_descending.shp')
            df = df.query("PR == 225078")

    return df


class TestRadiometrics(unittest.TestCase):
    def test_landsat_metadata(self):
        meta = RT.get_landsat_coefficients(str(l8_224078_20200127_meta))

        self.assertTrue(meta.sensor, 'l8')
        self.assertEqual(
            pd.Timestamp(meta.date_acquired).round('D').to_pydatetime(),
            datetime(2020, 1, 28, 0, 0, tzinfo=timezone.utc),
        )

    def test_angles(self):
        df = get_pathrow()
        with tempfile.TemporaryDirectory() as gz_tmp:
            angle_paths = extract_espa_tools(gz_tmp)
            with tempfile.TemporaryDirectory() as tmp:
                angle_data = landsat_pixel_angles(
                    str(l7_225078_20110306_ang),
                    l7_225078_20110306_B1,
                    tmp,
                    'l7',
                    l57_angles_path=str(angle_paths.l57_angles_path),
                    l8_angles_path=str(angle_paths.l8_angles_path),
                    subsample=10,
                    resampling='bilinear',
                    num_workers=1,
                    verbose=0,
                    chunks=256,
                )
                sza = angle_data.sza.compute()
                with gw.open(l7_225078_20110306_B1) as src:
                    # Same CRS
                    self.assertEqual(src.gw.crs_to_pyproj, sza.gw.crs_to_pyproj)
                    # Same bounds
                    self.assertEqual(src.gw.affine, sza.gw.affine)
                self.assertTrue(
                    df.to_crs(src.gw.crs_to_pyproj)
                    .geometry.intersects(sza.gw.geometry)
                    .values[0]
                )
                # Valid data so not clipped
                self.assertFalse(np.all(sza.data.compute() == -32768))
                self.assertFalse(np.all(np.isnan(sza.data.compute())))


if __name__ == '__main__':
    unittest.main()
