"""Tests for issue #330: Sentinel-2 PSD namespace version handling.

Post-September 2024 Sentinel-2 metadata uses psd-15 namespace instead
of psd-14. The angle parsing functions should handle both versions.
"""

import tempfile
import unittest
from pathlib import Path

from geowombat.radiometry.angles import (
    _sentinel_geocoding_base,
    get_sentinel_angle_shape,
    get_sentinel_crs_transform,
)

# Minimal Sentinel-2 L2A tile metadata XML templates
_XML_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<n1:Level-2A_Tile_ID
    xmlns:n1="https://{psd}.sentinel2.eo.esa.int/PSD/S2_PDI_Level-2A_Tile_Metadata.xsd"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <n1:General_Info>
    <TILE_ID>S2B_OPER_MSI_L2A_TL_2BS1_20230701T000000_A000001_T35JNF_N05.10</TILE_ID>
  </n1:General_Info>
  <n1:Geometric_Info>
    <Tile_Geocoding>
      <HORIZONTAL_CS_CODE>EPSG:32635</HORIZONTAL_CS_CODE>
      <Size resolution="10">
        <NROWS>10980</NROWS>
        <NCOLS>10980</NCOLS>
      </Size>
      <Geoposition resolution="10">
        <ULX>300000</ULX>
        <ULY>4400040</ULY>
      </Geoposition>
    </Tile_Geocoding>
    <Tile_Angles>
      <ROW_STEP>5000</ROW_STEP>
      <COL_STEP>5000</COL_STEP>
      <Sun_Angles_Grid>
        <Zenith>
          <Values_List>
            <VALUES>30.0 30.0 30.0</VALUES>
            <VALUES>30.0 30.0 30.0</VALUES>
            <VALUES>30.0 30.0 30.0</VALUES>
          </Values_List>
        </Zenith>
        <Azimuth>
          <Values_List>
            <VALUES>150.0 150.0 150.0</VALUES>
            <VALUES>150.0 150.0 150.0</VALUES>
            <VALUES>150.0 150.0 150.0</VALUES>
          </Values_List>
        </Azimuth>
      </Sun_Angles_Grid>
    </Tile_Angles>
  </n1:Geometric_Info>
</n1:Level-2A_Tile_ID>
"""


def _write_xml(psd_version: str, path: str):
    """Write a minimal Sentinel-2 metadata XML with the given PSD
    version."""
    xml = _XML_TEMPLATE.format(psd=psd_version)
    Path(path).write_text(xml)


class TestSentinelNamespace(unittest.TestCase):
    def test_psd14_crs_transform(self):
        """psd-14 metadata should parse correctly."""
        with tempfile.NamedTemporaryFile(
            suffix='.xml', mode='w', delete=False
        ) as tmp:
            _write_xml('psd-14', tmp.name)
            crs, transform, nrows, ncols = get_sentinel_crs_transform(
                tmp.name
            )
            self.assertEqual(crs, 'EPSG:32635')
            self.assertEqual(nrows, 10980)
            self.assertEqual(ncols, 10980)

    def test_psd15_crs_transform(self):
        """psd-15 metadata (post-Sept 2024) should also parse."""
        with tempfile.NamedTemporaryFile(
            suffix='.xml', mode='w', delete=False
        ) as tmp:
            _write_xml('psd-15', tmp.name)
            crs, transform, nrows, ncols = get_sentinel_crs_transform(
                tmp.name
            )
            self.assertEqual(crs, 'EPSG:32635')
            self.assertEqual(nrows, 10980)
            self.assertEqual(ncols, 10980)

    def test_psd14_angle_shape(self):
        """psd-14 angle shape parsing should work."""
        with tempfile.NamedTemporaryFile(
            suffix='.xml', mode='w', delete=False
        ) as tmp:
            _write_xml('psd-14', tmp.name)
            angles_view_list, angle_nrows, angle_ncols = (
                get_sentinel_angle_shape(tmp.name)
            )
            self.assertGreater(angle_nrows, 0)
            self.assertGreater(angle_ncols, 0)

    def test_psd15_angle_shape(self):
        """psd-15 angle shape parsing should work."""
        with tempfile.NamedTemporaryFile(
            suffix='.xml', mode='w', delete=False
        ) as tmp:
            _write_xml('psd-15', tmp.name)
            angles_view_list, angle_nrows, angle_ncols = (
                get_sentinel_angle_shape(tmp.name)
            )
            self.assertGreater(angle_nrows, 0)
            self.assertGreater(angle_ncols, 0)

    def test_unsupported_psd_raises(self):
        """An unsupported PSD version should raise ValueError."""
        with tempfile.NamedTemporaryFile(
            suffix='.xml', mode='w', delete=False
        ) as tmp:
            _write_xml('psd-99', tmp.name)
            with self.assertRaises(ValueError):
                get_sentinel_crs_transform(tmp.name)


if __name__ == '__main__':
    unittest.main()
