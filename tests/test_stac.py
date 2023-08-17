import unittest

import geopandas as gpd
import validators
from rasterio.enums import Resampling
from shapely.geometry import CAP_STYLE, JOIN_STYLE, shape

from geowombat.core.stac import (
    STAC_CATALOGS,
    STAC_COLLECTIONS,
    STACCollections,
    STACNames,
    open_stac,
)

geojson = {
    "type": "Polygon",
    "coordinates": [
        [
            [12.489159107208252, 41.88771017105127],
            [12.49619722366333, 41.88771017105127],
            [12.49619722366333, 41.8920390506062],
            [12.489159107208252, 41.8920390506062],
            [12.489159107208252, 41.88771017105127],
        ]
    ],
}

EPSG = 8857

DF = gpd.GeoDataFrame(geometry=[shape(geojson)], crs=4326)

SEARCH_DF = (
    DF.to_crs(f'epsg:{EPSG}')
    .buffer(100, cap_style=CAP_STYLE.square, join_style=JOIN_STYLE.mitre)
    .to_crs('epsg:4326')
    .to_frame(name='geometry')
)


def url_is_valid(url: str) -> bool:
    validation = validators.url(url)
    if validation:
        return True
    return False


class TestDownloadSingleBand(unittest.TestCase):
    # def test_download_sentinel_3_lst(self):
    #     stack = open_stac(
    #         stac_catalog='microsoft',
    #         bounds=SEARCH_DF,
    #         proj_bounds=tuple(
    #             DF.to_crs(f'epsg:{EPSG}').total_bounds.flatten().tolist()
    #         ),
    #         epsg=EPSG,
    #         collection='sentinel_3_lst',
    #         bands=['lst-in'],
    #         cloud_cover_perc=90,
    #         chunksize=64,
    #         start_date='2022-07-01',
    #         end_date='2022-07-07',
    #         resolution=300.0,
    #         nodata_fill=32768,
    #         resampling=Resampling.nearest,
    #         max_items=None,
    #     )[0]
    #     self.assertTrue(stack.shape == (20, 1, 3, 4))
    #     self.assertTrue(stack.crs == 'epsg:8857')
    #     self.assertTrue(stack.gw.celly == 300.0)
    #     self.assertTrue(stack.gw.cellx == 300.0)
    #     self.assertTrue(stack.gw.nodataval == 32768)

    # def test_download_blue_sentinel_s2_l1c(self):
    #     stack = open_stac(
    #         stac_catalog='element84_v1',
    #         bounds=SEARCH_DF,
    #         proj_bounds=tuple(
    #             DF.to_crs(f'epsg:{EPSG}').total_bounds.flatten().tolist()
    #         ),
    #         epsg=EPSG,
    #         collection='sentinel_s2_l1c',
    #         bands=['blue'],
    #         cloud_cover_perc=90,
    #         chunksize=64,
    #         start_date='2022-07-01',
    #         end_date='2022-07-07',
    #         resolution=10.0,
    #         nodata_fill=32768,
    #         resampling=Resampling.nearest,
    #         max_items=None,
    #     )[0]
    #     self.assertTrue(stack.shape == (2, 1, 48, 64))
    #     self.assertTrue(stack.crs == 'epsg:8857')
    #     self.assertTrue(stack.gw.celly == 10.0)
    #     self.assertTrue(stack.gw.cellx == 10.0)
    #     self.assertTrue(stack.gw.nodataval == 32768)

    def test_download_blue_sentinel_s2_l2a_cogs(self):
        stack = open_stac(
            stac_catalog='element84_v0',
            bounds=SEARCH_DF,
            proj_bounds=tuple(
                DF.to_crs(f'epsg:{EPSG}').total_bounds.flatten().tolist()
            ),
            epsg=EPSG,
            collection='sentinel_s2_l2a_cogs',
            bands=['B02'],
            cloud_cover_perc=90,
            chunksize=64,
            start_date='2022-07-01',
            end_date='2022-07-07',
            resolution=10.0,
            nodata_fill=32768,
            resampling=Resampling.nearest,
            max_items=None,
        )[0]
        self.assertTrue(stack.shape == (2, 1, 48, 64))
        self.assertTrue(stack.crs == 'epsg:8857')
        self.assertTrue(stack.gw.celly == 10.0)
        self.assertTrue(stack.gw.cellx == 10.0)
        self.assertTrue(stack.gw.nodataval == 32768)


class TestSTAC(unittest.TestCase):
    def test_unsupported(self):
        # Element84 v1 does not have Sentinel2 COGs
        collection = 'element84_v1'
        stac_catalog = 'sentinel_s2_l2a_cogs'
        collection_dict = STAC_COLLECTIONS[STACCollections(stac_catalog)]
        with self.assertRaises(KeyError):
            catalog_collections = [collection_dict[STACNames(collection)]]
        # Element84 does not have a usda_cdl collection
        collection = 'element84_v1'
        stac_catalog = 'usda_cdl'
        collection_dict = STAC_COLLECTIONS[STACCollections(stac_catalog)]
        with self.assertRaises(KeyError):
            catalog_collections = [collection_dict[STACNames(collection)]]

    def test_constants(self):
        self.assertEqual(STACNames('element84_v0'), STACNames.element84_v0)
        self.assertEqual(STACNames('element84_v1'), STACNames.element84_v1)
        self.assertEqual(STACNames('microsoft'), STACNames.microsoft)

    def test_urls(self):
        self.assertTrue(url_is_valid(STAC_CATALOGS[STACNames.element84_v0]))
        self.assertTrue(url_is_valid(STAC_CATALOGS[STACNames.element84_v1]))
        self.assertTrue(url_is_valid(STAC_CATALOGS[STACNames.microsoft]))


if __name__ == '__main__':
    unittest.main()
