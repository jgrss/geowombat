# flake8: noqa

import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import validators
from dask.distributed import Client, LocalCluster
from pyproj import CRS
from rasterio.enums import Resampling
from shapely.geometry import CAP_STYLE, JOIN_STYLE, shape

import geowombat as gw
from geowombat.core.stac import (
    STAC_CATALOGS,
    STAC_COLLECTIONS,
    STACCollections,
    STACCollectionURLNames,
    STACNames,
    open_stac,
)

search_geojson = {
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

naip_geojson = {
    "type": "Polygon",
    "coordinates": [
        [
            [-86.65852429273222, 40.869286853632445],
            [-86.65852429273222, 40.85430596003397],
            [-86.63824797576922, 40.85430596003397],
            [-86.63824797576922, 40.869286853632445],
            [-86.65852429273222, 40.869286853632445],
        ]
    ],
}


def geosjon_to_df(
    geojson: dict,
    epsg: int,
) -> tuple:
    latlon_df = gpd.GeoDataFrame(geometry=[shape(geojson)], crs=4326)
    proj_df = latlon_df.to_crs(f'epsg:{epsg}')
    buffer_df = (
        proj_df.buffer(
            100, cap_style=CAP_STYLE.square, join_style=JOIN_STYLE.mitre
        )
        .to_crs('epsg:4326')
        .to_frame(name='geometry')
    )

    return buffer_df, tuple(proj_df.total_bounds.flatten().tolist())


SEARCH_EPSG = 8857
SEARCH_DF, SEARCH_BOUNDS = geosjon_to_df(
    geojson=search_geojson,
    epsg=SEARCH_EPSG,
)

NAIP_EPSG = 8858
NAIP_DF, NAIP_PROJ_BOUNDS = geosjon_to_df(
    geojson=naip_geojson,
    epsg=NAIP_EPSG,
)


def url_is_valid(url: str) -> bool:
    validation = validators.url(url)
    if validation:
        return True
    return False


class TestSearchSingleBand(unittest.TestCase):
    def test_search_sentinel_3_lst(self):
        stack = open_stac(
            stac_catalog='microsoft_v1',
            bounds=SEARCH_DF,
            proj_bounds=SEARCH_BOUNDS,
            epsg=SEARCH_EPSG,
            collection='sentinel_3_lst',
            bands=['lst-in'],
            cloud_cover_perc=90,
            chunksize=64,
            start_date='2022-07-01',
            end_date='2022-07-07',
            resolution=300.0,
            nodata_fill=32768,
            resampling=Resampling.nearest,
            max_items=None,
        )[0]

        self.assertTrue(stack.shape == (20, 1, 3, 4))
        self.assertTrue(stack.gw.crs_to_pyproj == CRS.from_epsg(SEARCH_EPSG))
        self.assertTrue(stack.gw.celly == 300.0)
        self.assertTrue(stack.gw.cellx == 300.0)
        self.assertTrue(stack.gw.nodataval == 32768)

    def test_search_blue_sentinel_s2_l1c(self):
        stack = open_stac(
            stac_catalog='element84_v1',
            bounds=SEARCH_DF,
            proj_bounds=SEARCH_BOUNDS,
            epsg=SEARCH_EPSG,
            collection='sentinel_s2_l1c',
            bands=['blue'],
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
        self.assertTrue(stack.gw.crs_to_pyproj == CRS.from_epsg(SEARCH_EPSG))
        self.assertTrue(stack.gw.celly == 10.0)
        self.assertTrue(stack.gw.cellx == 10.0)
        self.assertTrue(stack.gw.nodataval == 32768)

    def test_search_sentinel_s1_l1c(self):
        stack = open_stac(
            stac_catalog='element84_v1',
            bounds=SEARCH_DF,
            proj_bounds=SEARCH_BOUNDS,
            epsg=SEARCH_EPSG,
            collection='sentinel_s1_l1c',
            bands=['blue'],
            cloud_cover_perc=90,
            chunksize=64,
            start_date='2022-07-01',
            end_date='2022-07-07',
            resolution=10.0,
            nodata_fill=32768,
            resampling=Resampling.nearest,
            max_items=None,
        )[0]

    def test_search_ms_landsat_c2_l2(self):
        stack, df = open_stac(
            stac_catalog='microsoft_v1',
            bounds=SEARCH_DF,
            proj_bounds=SEARCH_BOUNDS,
            epsg=SEARCH_EPSG,
            collection='landsat_c2_l2',
            bands=['red', 'nir'],
            cloud_cover_perc=90,
            chunksize=64,
            start_date='2022-07-01',
            end_date='2022-07-07',
            resolution=10.0,
            nodata_fill=32768,
            resampling=Resampling.nearest,
            max_items=None,
            extra_assets=['mtl.txt'],
            view_asset_keys=True,
        )
        self.assertTrue(stack is None)
        self.assertTrue(df is None)

        with tempfile.TemporaryDirectory() as tmp_path:
            stack, df = open_stac(
                stac_catalog='microsoft_v1',
                bounds=SEARCH_DF,
                proj_bounds=SEARCH_BOUNDS,
                epsg=SEARCH_EPSG,
                collection='landsat_c2_l2',
                bands=['red', 'nir08'],
                cloud_cover_perc=90,
                chunksize=32,
                start_date='2022-07-01',
                end_date='2022-07-07',
                resolution=10.0,
                nodata_fill=32768,
                resampling=Resampling.nearest,
                max_items=None,
                extra_assets=['mtl.txt'],
                out_path=Path(tmp_path),
            )

            self.assertFalse(
                set(stack.band.values).difference(['red', 'nir08'])
            )
            self.assertTrue(stack.shape == (2, 2, 48, 64))
            self.assertTrue(
                stack.gw.crs_to_pyproj == CRS.from_epsg(SEARCH_EPSG)
            )
            self.assertTrue(stack.gw.celly == 10.0)
            self.assertTrue(stack.gw.cellx == 10.0)
            self.assertTrue(stack.gw.nodataval == 32768)
            self.assertTrue(len(df.index) == 2)
            self.assertFalse(set(df.id.values).difference(df.id.values))

    def test_search_blue_sentinel_s2_l2a(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            stack, df = open_stac(
                stac_catalog='element84_v1',
                bounds=SEARCH_DF,
                proj_bounds=SEARCH_BOUNDS,
                epsg=SEARCH_EPSG,
                collection='sentinel_s2_l2a',
                bands=['blue'],
                cloud_cover_perc=90,
                chunksize=32,
                start_date='2022-07-01',
                end_date='2022-07-07',
                resolution=10.0,
                nodata_fill=32768,
                resampling=Resampling.nearest,
                max_items=None,
                extra_assets=['granule_metadata', 'scl'],
                out_path=Path(tmp_path),
            )

            self.assertTrue(stack.shape == (2, 1, 48, 64))
            self.assertTrue(
                stack.gw.crs_to_pyproj == CRS.from_epsg(SEARCH_EPSG)
            )
            self.assertTrue(stack.gw.celly == 10.0)
            self.assertTrue(stack.gw.cellx == 10.0)
            self.assertTrue(stack.gw.nodataval == 32768)
            self.assertTrue(len(df.index) == 2)
            self.assertFalse(set(df.id.values).difference(df.id.values))

            out_path = Path(tmp_path) / 'test.tif'
            time_mean = stack.mean(dim='time', keep_attrs=True)
            time_mean.gw.save(
                filename=out_path,
                overwrite=True,
            )
            with gw.open(out_path) as src:
                self.assertTrue(
                    np.allclose(
                        time_mean.data.compute(),
                        src.data.compute(),
                    )
                )

            out_path_client = Path(tmp_path) / 'test_client.tif'
            with LocalCluster(
                processes=True,
                n_workers=2,
                threads_per_worker=1,
                memory_limit="1GB",
            ) as cluster:
                with Client(cluster) as client:
                    time_mean.gw.save(
                        filename=out_path_client,
                        overwrite=True,
                        client=client,
                    )

            with gw.open(out_path_client) as src:
                self.assertTrue(
                    np.allclose(
                        time_mean.data.compute(),
                        src.data.compute(),
                    )
                )


class TestSTAC(unittest.TestCase):
    def test_unsupported(self):
        # Element84 v1 does have Sentinel2
        self.assertTrue(
            STACCollectionURLNames[STACCollections('sentinel_s2_l2a').name]
            in STAC_COLLECTIONS[STACNames.ELEMENT84_V1]
        )

        # Element84 v1 does not have Sentinel2 COGs
        self.assertFalse(
            STACCollectionURLNames[
                STACCollections('sentinel_s2_l2a_cogs').name
            ]
            in STAC_COLLECTIONS[STACNames.ELEMENT84_V1]
        )

        # Element84 does not have a USDA CDL collection
        self.assertFalse(
            STACCollectionURLNames[STACCollections('usda_cdl').name]
            in STAC_COLLECTIONS[STACNames.ELEMENT84_V1]
        )

    def test_urls(self):
        self.assertTrue(url_is_valid(STAC_CATALOGS[STACNames.ELEMENT84_V0]))
        self.assertTrue(url_is_valid(STAC_CATALOGS[STACNames.ELEMENT84_V1]))
        self.assertTrue(url_is_valid(STAC_CATALOGS[STACNames.MICROSOFT_V1]))


if __name__ == '__main__':
    unittest.main()
