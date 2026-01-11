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
)

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
    shape=(2, 1, 48, 64),
    bands=['blue'],
    crs=8857,
    res=10.0,
    collection='sentinel_s2_l2a',
):
    """Create a mock xarray DataArray matching expected STAC output."""
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
        """Test that invalid collection name raises NameError."""
        with self.assertRaises(NameError):
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


if __name__ == '__main__':
    unittest.main()
