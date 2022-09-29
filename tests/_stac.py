import unittest

from geowombat.core.stac import open_stac

import geopandas as gpd
from rasterio.enums import Resampling
from shapely.geometry import CAP_STYLE, JOIN_STYLE, shape


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


class TestSTAC(unittest.TestCase):
    def test_blue(self):
        stack = open_stac(
            stac_catalog='element84',
            bounds=SEARCH_DF,
            proj_bounds=tuple(
                DF.to_crs(f'epsg:{EPSG}').total_bounds.flatten().tolist()
            ),
            epsg=EPSG,
            collection='sentinel_s2_l2a',
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

        stack = (
            stack.groupby('time')
            .mean(dim='time', skipna=True)
            .assign_attrs(**stack.attrs)
            .assign_attrs(collection=None)
        )
        self.assertTrue(stack.shape == (1, 1, 48, 64))
        self.assertTrue(stack.gw.nodataval == 32768)

    def test_visible(self):
        stack = open_stac(
            stac_catalog='element84',
            bounds=SEARCH_DF,
            proj_bounds=tuple(
                DF.to_crs(f'epsg:{EPSG}').total_bounds.flatten().tolist()
            ),
            epsg=EPSG,
            collection='sentinel_s2_l2a',
            bands=['B02', 'B03', 'B04'],
            cloud_cover_perc=90,
            chunksize=64,
            start_date='2022-07-01',
            end_date='2022-07-07',
            resolution=10.0,
            nodata_fill=32768,
            resampling=Resampling.nearest,
            max_items=None,
        )[0]
        self.assertTrue(stack.shape == (2, 3, 48, 64))
        self.assertTrue(stack.crs == 'epsg:8857')
        self.assertTrue(stack.gw.celly == 10.0)
        self.assertTrue(stack.gw.cellx == 10.0)

        stack = (
            stack.groupby('time')
            .mean(dim='time', skipna=True)
            .assign_attrs(**stack.attrs)
            .assign_attrs(collection=None)
        )
        self.assertTrue(stack.shape == (1, 3, 48, 64))
        self.assertTrue(stack.gw.nodataval == 32768)


if __name__ == '__main__':
    unittest.main()
