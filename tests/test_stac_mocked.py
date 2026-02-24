# flake8: noqa

import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import requests
import dask.array as da
import xarray as xr

from geowombat.core.stac import (
    STAC_CATALOGS,
    STAC_COLLECTIONS,
    STACCollections,
    STACCollectionURLNames,
    STACNames,
    open_stac,
    composite_stac,
    _Client,
)

# Check if STAC dependencies are available
STAC_AVAILABLE = _Client is not None

CONNECTIVITY_TIMEOUT = 5  # seconds


class TestSTACConnectivity(unittest.TestCase):
    """Quick connectivity checks - always run with short timeout."""

    def test_element84_v0_reachable(self):
        """Verify Element84 v0 STAC catalog responds."""
        url = STAC_CATALOGS[STACNames.ELEMENT84_V0]
        response = requests.get(url, timeout=CONNECTIVITY_TIMEOUT)
        self.assertEqual(response.status_code, 200)

    def test_element84_v1_reachable(self):
        """Verify Element84 v1 STAC catalog responds."""
        url = STAC_CATALOGS[STACNames.ELEMENT84_V1]
        response = requests.get(url, timeout=CONNECTIVITY_TIMEOUT)
        self.assertEqual(response.status_code, 200)

    def test_microsoft_v1_reachable(self):
        """Verify Microsoft Planetary Computer responds."""
        url = STAC_CATALOGS[STACNames.MICROSOFT_V1]
        response = requests.get(url, timeout=CONNECTIVITY_TIMEOUT)
        self.assertEqual(response.status_code, 200)


def create_mock_data_array(
    shape=None,
    bands=['blue'],
    crs=8857,
    res=10.0,
    collection='sentinel_s2_l2a',
):
    """Create a mock xarray DataArray matching expected STAC output."""
    # Default shape based on bands length
    if shape is None:
        shape = (2, len(bands), 48, 64)
    data = da.random.random(shape, chunks=(1, 1, 32, 32))
    return xr.DataArray(
        data,
        dims=('time', 'band', 'y', 'x'),
        coords={
            'time': pd.date_range('2022-07-01', periods=shape[0]),
            'band': bands,
            'y': np.arange(shape[2]) * -res,
            'x': np.arange(shape[3]) * res,
        },
        attrs={
            'crs': f'EPSG:{crs}',
            'res': (res, res),
            'transform': (res, 0, 0, 0, -res, 0),
            '_FillValue': 32768,
            'resolution': res,
            'epsg': crs,
            'collection': collection,
        },
    )


class TestSTACCollectionRegistry(unittest.TestCase):
    """Test that STAC collections are properly registered."""

    def test_element84_v0_collections(self):
        """Verify Element84 v0 has expected collections."""
        self.assertIn(
            STACCollectionURLNames.SENTINEL_S2_L2A_COGS,
            STAC_COLLECTIONS[STACNames.ELEMENT84_V0],
        )

    def test_element84_v1_collections(self):
        """Verify Element84 v1 has expected collections."""
        expected = [
            STACCollectionURLNames.COP_DEM_GLO_30,
            STACCollectionURLNames.LANDSAT_C2_L2,
            STACCollectionURLNames.SENTINEL_S2_L2A,
            STACCollectionURLNames.SENTINEL_S2_L1C,
            STACCollectionURLNames.SENTINEL_S1_L1C,
            STACCollectionURLNames.NAIP,
        ]
        for collection in expected:
            self.assertIn(
                collection,
                STAC_COLLECTIONS[STACNames.ELEMENT84_V1],
            )

    def test_microsoft_v1_collections(self):
        """Verify Microsoft v1 has expected collections."""
        expected = [
            STACCollectionURLNames.COP_DEM_GLO_30,
            STACCollectionURLNames.LANDSAT_C2_L1,
            STACCollectionURLNames.LANDSAT_C2_L2,
            STACCollectionURLNames.SENTINEL_S2_L2A,
            STACCollectionURLNames.SENTINEL_S1_L1C,
            STACCollectionURLNames.SENTINEL_3_LST,
            STACCollectionURLNames.LANDSAT_L8_C2_L2,
            STACCollectionURLNames.USDA_CDL,
            STACCollectionURLNames.IO_LULC,
        ]
        for collection in expected:
            self.assertIn(
                collection,
                STAC_COLLECTIONS[STACNames.MICROSOFT_V1],
            )


class TestSTACMocked(unittest.TestCase):
    """Mocked tests that don't depend on external APIs."""

    @patch('geowombat.core.stac.stackstac')
    @patch('geowombat.core.stac._Client')
    @patch('geowombat.core.stac.pystac')
    def test_open_stac_element84_mocked(
        self, mock_pystac, mock_client_class, mock_stackstac
    ):
        """Test open_stac with mocked Element84 catalog."""
        # Setup mock catalog
        mock_catalog = MagicMock()
        mock_client_class.open.return_value = mock_catalog

        # Setup mock search results
        mock_search = MagicMock()
        mock_item = MagicMock()
        mock_item.id = 'test_item_1'
        mock_search.items.return_value = [mock_item]
        mock_catalog.search.return_value = mock_search

        # Setup mock pystac.ItemCollection
        mock_item_collection = MagicMock()
        mock_item_collection.__iter__ = lambda self: iter([mock_item])
        mock_pystac.ItemCollection.return_value = mock_item_collection

        # Setup mock stackstac output
        mock_data = create_mock_data_array()
        mock_stackstac.stack.return_value = mock_data

        # Call open_stac
        result, df = open_stac(
            stac_catalog='element84_v1',
            collection='sentinel_s2_l2a',
            bounds=(-86.6, 40.8, -86.5, 40.9),
            proj_bounds=(0, 0, 640, 480),
            epsg=8857,
            bands=['blue'],
            start_date='2022-07-01',
            end_date='2022-07-07',
            resolution=10.0,
        )

        # Verify catalog was opened with correct URL
        mock_client_class.open.assert_called_once_with(
            STAC_CATALOGS[STACNames.ELEMENT84_V1]
        )
        # Verify search was performed
        mock_catalog.search.assert_called_once()
        # Verify stackstac.stack was called
        mock_stackstac.stack.assert_called_once()

    @patch('geowombat.core.stac.stackstac')
    @patch('geowombat.core.stac.pc')
    @patch('geowombat.core.stac._Client')
    def test_open_stac_microsoft_mocked(
        self, mock_client_class, mock_pc, mock_stackstac
    ):
        """Test open_stac with mocked Microsoft catalog (includes pc.sign)."""
        # Setup mock catalog
        mock_catalog = MagicMock()
        mock_client_class.open.return_value = mock_catalog

        # Setup mock search results
        mock_search = MagicMock()
        mock_item = MagicMock()
        mock_item.id = 'test_item_1'
        mock_search.items.return_value = [mock_item]
        mock_catalog.search.return_value = mock_search

        # Setup mock planetary_computer.sign
        mock_signed_items = MagicMock()
        mock_signed_items.__iter__ = lambda self: iter([mock_item])
        mock_pc.sign.return_value = mock_signed_items

        # Setup mock stackstac output
        mock_data = create_mock_data_array(
            bands=['red', 'nir08'], collection='landsat_c2_l2'
        )
        mock_stackstac.stack.return_value = mock_data

        # Call open_stac with Microsoft catalog
        result, df = open_stac(
            stac_catalog='microsoft_v1',
            collection='landsat_c2_l2',
            bounds=(-86.6, 40.8, -86.5, 40.9),
            proj_bounds=(0, 0, 640, 480),
            epsg=8857,
            bands=['red', 'nir08'],
            start_date='2022-07-01',
            end_date='2022-07-07',
            resolution=10.0,
        )

        # Verify catalog was opened with correct URL
        mock_client_class.open.assert_called_once_with(
            STAC_CATALOGS[STACNames.MICROSOFT_V1]
        )
        # Verify pc.sign was called for Microsoft catalog
        mock_pc.sign.assert_called_once()
        # Verify stackstac.stack was called
        mock_stackstac.stack.assert_called_once()

    @patch('geowombat.core.stac.stackstac')
    @patch('geowombat.core.stac._Client')
    @patch('geowombat.core.stac.pystac')
    def test_open_stac_no_results(
        self, mock_pystac, mock_client_class, mock_stackstac
    ):
        """Test open_stac returns None when no items found."""
        # Setup mock catalog
        mock_catalog = MagicMock()
        mock_client_class.open.return_value = mock_catalog

        # Setup mock search with no results
        mock_search = MagicMock()
        mock_search.items.return_value = []
        mock_catalog.search.return_value = mock_search

        # Call open_stac
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result, df = open_stac(
                stac_catalog='element84_v1',
                collection='sentinel_s2_l2a',
                bounds=(-86.6, 40.8, -86.5, 40.9),
                proj_bounds=(0, 0, 640, 480),
                epsg=8857,
                bands=['blue'],
                start_date='2022-07-01',
                end_date='2022-07-07',
                resolution=10.0,
            )

        # Verify None is returned when no items found
        self.assertIsNone(result)
        self.assertIsNone(df)

    def test_invalid_collection_raises_error(self):
        """Test that invalid collection name raises ValueError."""
        with self.assertRaises(ValueError):
            open_stac(
                stac_catalog='element84_v1',
                collection='invalid_collection_name',
                bounds=(-86.6, 40.8, -86.5, 40.9),
                start_date='2022-07-01',
                end_date='2022-07-07',
            )

    def test_missing_collection_raises_error(self):
        """Test that missing collection parameter raises NameError."""
        with self.assertRaises(NameError):
            open_stac(
                stac_catalog='element84_v1',
                bounds=(-86.6, 40.8, -86.5, 40.9),
                start_date='2022-07-01',
                end_date='2022-07-07',
            )


class TestSTACMasking(unittest.TestCase):
    """Tests for pixel-level masking in open_stac."""

    def _setup_mocks(
        self, mock_pystac, mock_client_class, mock_stackstac, mock_data
    ):
        """Common mock setup for open_stac tests."""
        mock_catalog = MagicMock()
        mock_client_class.open.return_value = mock_catalog
        mock_search = MagicMock()
        mock_item = MagicMock()
        mock_item.id = 'test_item'
        mock_search.items.return_value = [mock_item]
        mock_catalog.search.return_value = mock_search
        mock_item_collection = MagicMock()
        mock_item_collection.__iter__ = lambda self: iter(
            [mock_item]
        )
        mock_pystac.ItemCollection.return_value = (
            mock_item_collection
        )
        mock_stackstac.stack.return_value = mock_data

    @patch('geowombat.core.stac.stackstac')
    @patch('geowombat.core.stac._Client')
    @patch('geowombat.core.stac.pystac')
    def test_s2_scl_masking(
        self, mock_pystac, mock_client_class, mock_stackstac
    ):
        """Test mask_data=True for Sentinel-2 with SCL."""
        # SCL=4 (vegetation, good), SCL=9 (cloud high, bad)
        spectral = np.ones((2, 2, 16, 16), dtype=np.float32)
        scl = np.full(
            (2, 1, 16, 16), 4, dtype=np.float32
        )
        scl[:, :, 5:10, 5:10] = 9  # cloud patch
        all_data = da.from_array(
            np.concatenate([spectral, scl], axis=1)
        )
        mock_data = xr.DataArray(
            all_data,
            dims=('time', 'band', 'y', 'x'),
            coords={
                'time': pd.date_range('2022-07-01', periods=2),
                'band': ['blue', 'green', 'scl'],
                'y': np.arange(16) * -10.0,
                'x': np.arange(16) * 10.0,
            },
            attrs={
                'crs': 'epsg:32618',
                'res': (10.0, 10.0),
                'transform': (10.0, 0, 0, 0, -10.0, 0),
                'resolution': 10.0,
                'epsg': 32618,
                'collection': 'sentinel_s2_l2a',
            },
        )
        self._setup_mocks(
            mock_pystac, mock_client_class,
            mock_stackstac, mock_data,
        )

        result, df = open_stac(
            stac_catalog='element84_v1',
            collection='sentinel_s2_l2a',
            bounds=(-77.1, 38.85, -76.95, 38.95),
            proj_bounds=(0, 0, 160, 160),
            epsg=32618,
            bands=['blue', 'green'],
            mask_data=True,
            start_date='2022-07-01',
            end_date='2022-07-07',
            resolution=10.0,
        )

        # SCL band should be removed
        self.assertNotIn('scl', result.band.values)
        self.assertEqual(
            list(result.band.values), ['blue', 'green']
        )
        # Cloud pixels (SCL=9) should be NaN
        vals = result.values
        self.assertTrue(
            np.isnan(vals[0, :, 5:10, 5:10]).all()
        )
        # Clear pixels (SCL=4) should NOT be NaN
        self.assertFalse(
            np.isnan(vals[0, :, 0:3, 0:3]).all()
        )

    @patch('geowombat.core.stac.stackstac')
    @patch('geowombat.core.stac._Client')
    @patch('geowombat.core.stac.pystac')
    def test_s2_auto_injects_scl(
        self, mock_pystac, mock_client_class, mock_stackstac
    ):
        """Test that mask_data=True auto-adds scl band."""
        mock_data = create_mock_data_array(
            bands=['blue', 'green', 'scl'],
            collection='sentinel_s2_l2a',
        )
        self._setup_mocks(
            mock_pystac, mock_client_class,
            mock_stackstac, mock_data,
        )

        open_stac(
            stac_catalog='element84_v1',
            collection='sentinel_s2_l2a',
            bounds=(-77.1, 38.85, -76.95, 38.95),
            proj_bounds=(0, 0, 160, 160),
            epsg=32618,
            bands=['blue', 'green'],
            mask_data=True,
            start_date='2022-07-01',
            end_date='2022-07-07',
            resolution=10.0,
        )

        # stackstac.stack should have been called with
        # scl added to the band list
        call_args = mock_stackstac.stack.call_args
        assets = call_args[1].get('assets') or call_args[0][1]
        self.assertIn('scl', assets)

    @patch('geowombat.core.stac.stackstac')
    @patch('geowombat.core.stac.pc')
    @patch('geowombat.core.stac._Client')
    def test_landsat_auto_injects_qa_pixel(
        self, mock_client_class, mock_pc, mock_stackstac
    ):
        """Test that mask_data=True auto-adds qa_pixel band
        for Landsat."""
        mock_catalog = MagicMock()
        mock_client_class.open.return_value = mock_catalog
        mock_search = MagicMock()
        mock_item = MagicMock()
        mock_item.id = 'test_item'
        mock_search.items.return_value = [mock_item]
        mock_catalog.search.return_value = mock_search
        mock_signed = MagicMock()
        mock_signed.__iter__ = lambda self: iter([mock_item])
        mock_pc.sign.return_value = mock_signed

        mock_data = create_mock_data_array(
            bands=['red', 'qa_pixel'],
            collection='landsat_c2_l2',
        )
        mock_stackstac.stack.return_value = mock_data

        open_stac(
            stac_catalog='microsoft_v1',
            collection='landsat_c2_l2',
            bounds=(-77.1, 38.85, -76.95, 38.95),
            proj_bounds=(0, 0, 160, 160),
            epsg=32618,
            bands=['red'],
            mask_data=True,
            start_date='2022-07-01',
            end_date='2022-07-07',
            resolution=30.0,
        )

        call_args = mock_stackstac.stack.call_args
        assets = call_args[1].get('assets') or call_args[0][1]
        self.assertIn('qa_pixel', assets)


class TestCompositeSTAC(unittest.TestCase):
    """Tests for composite_stac function."""

    @patch('geowombat.core.stac.open_stac')
    def test_monthly_composite(self, mock_open_stac):
        """Test composite_stac produces monthly medians."""
        times = pd.date_range('2022-01-01', periods=60, freq='3D')
        data = da.random.random(
            (len(times), 2, 16, 16), chunks=(10, 2, 16, 16)
        )
        mock_data = xr.DataArray(
            data,
            dims=('time', 'band', 'y', 'x'),
            coords={
                'time': times,
                'band': ['red', 'nir'],
                'y': np.arange(16) * -10.0,
                'x': np.arange(16) * 10.0,
            },
            attrs={
                'crs': 'epsg:32618',
                'res': (10.0, 10.0),
                'transform': (10.0, 0, 0, 0, -10.0, 0),
                'resolution': 10.0,
                'collection': 'sentinel_s2_l2a',
            },
        )
        mock_open_stac.return_value = (mock_data, pd.DataFrame())

        composite, df = composite_stac(
            collection='sentinel_s2_l2a',
            bounds=(-77.1, 38.85, -76.95, 38.95),
            bands=['red', 'nir'],
            start_date='2022-01-01',
            end_date='2022-06-30',
            frequency='MS',
        )

        # Should produce monthly composites
        self.assertGreater(composite.sizes['time'], 0)
        self.assertLessEqual(composite.sizes['time'], 7)
        # Dims should be (time, band, y, x)
        self.assertEqual(
            list(composite.dims), ['time', 'band', 'y', 'x']
        )
        # Attrs should be preserved
        self.assertEqual(composite.attrs['crs'], 'epsg:32618')
        self.assertIn('res', composite.attrs)
        self.assertIn('transform', composite.attrs)
        # open_stac called with mask_data=True, compute=False
        call_kwargs = mock_open_stac.call_args[1]
        self.assertTrue(call_kwargs['mask_data'])
        self.assertFalse(call_kwargs['compute'])

    @patch('geowombat.core.stac.open_stac')
    def test_composite_no_results(self, mock_open_stac):
        """Test composite_stac returns None when no data."""
        mock_open_stac.return_value = (None, None)

        result = composite_stac(
            collection='sentinel_s2_l2a',
            bounds=(-77.1, 38.85, -76.95, 38.95),
            bands=['red'],
            start_date='2022-01-01',
            end_date='2022-01-31',
        )

        self.assertEqual(result, (None, None))

    @patch('geowombat.core.stac.open_stac')
    def test_quarterly_frequency(self, mock_open_stac):
        """Test composite_stac with quarterly frequency."""
        times = pd.date_range(
            '2022-01-01', periods=120, freq='3D'
        )
        data = da.random.random(
            (len(times), 1, 8, 8), chunks=(30, 1, 8, 8)
        )
        mock_data = xr.DataArray(
            data,
            dims=('time', 'band', 'y', 'x'),
            coords={
                'time': times,
                'band': ['red'],
                'y': np.arange(8) * -10.0,
                'x': np.arange(8) * 10.0,
            },
            attrs={
                'crs': 'epsg:32618',
                'res': (10.0, 10.0),
                'transform': (10.0, 0, 0, 0, -10.0, 0),
                'resolution': 10.0,
                'collection': 'landsat_c2_l2',
            },
        )
        mock_open_stac.return_value = (mock_data, pd.DataFrame())

        composite, df = composite_stac(
            collection='landsat_c2_l2',
            bounds=(-77.1, 38.85, -76.95, 38.95),
            bands=['red'],
            start_date='2022-01-01',
            end_date='2022-12-31',
            frequency='QS',
        )

        # Quarterly: should have up to 5 quarters
        self.assertGreater(composite.sizes['time'], 0)
        self.assertLessEqual(composite.sizes['time'], 5)


if __name__ == '__main__':
    unittest.main()
