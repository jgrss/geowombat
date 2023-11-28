import unittest

import dask.array as da
import numpy as np
import xarray as xr

import geowombat as gw


def create_data_twoband(band1: str, band2: str) -> xr.DataArray:
    bands = np.array([[[0.4]], [[0.1]]])

    return xr.DataArray(
        da.from_array(bands, chunks=(-1, 1, 1)),
        dims=('band', 'y', 'x'),
        coords={'band': [band1, band2], 'y': [1], 'x': [1]},
    )


def create_data_threeband(band1: str, band2: str, band3: str) -> xr.DataArray:
    bands = np.array([[[0.09]], [[0.11]], [[0.21]]])

    return xr.DataArray(
        da.from_array(bands, chunks=(-1, 1, 1)),
        dims=('band', 'y', 'x'),
        coords={'band': [band1, band2, band3], 'y': [1], 'x': [1]},
    )


class TestVI(unittest.TestCase):
    def test_wi(self):
        data = create_data_twoband('red', 'swir1')
        result = data.sel(band='red') + data.sel(band='swir1')
        result = (1.0 - (result.where(lambda x: x <= 0.5) / 0.5)).fillna(0)

        vi = data.gw.wi()

        self.assertTrue(np.allclose(vi.data.compute(), result.data.compute()))

    def test_kndvi(self):
        data = create_data_twoband('red', 'nir')
        result = (data.sel(band='nir') - data.sel(band='red')) / (
            data.sel(band='nir') + data.sel(band='red')
        )
        result = np.tanh(result**2)

        vi = data.gw.kndvi()

        self.assertTrue(np.allclose(vi.data.compute(), result.data.compute()))

    def test_ndvi(self):
        data = create_data_twoband('red', 'nir')
        result = (data.sel(band='nir') - data.sel(band='red')) / (
            data.sel(band='nir') + data.sel(band='red')
        )

        vi = data.gw.ndvi()

        self.assertTrue(np.allclose(vi.data.compute(), result.data.compute()))

    def test_nbr(self):
        data = create_data_twoband('nir', 'swir2')
        result = (data.sel(band='nir') - data.sel(band='swir2')) / (
            data.sel(band='nir') + data.sel(band='swir2')
        )

        vi = data.gw.nbr()

        self.assertTrue(np.allclose(vi.data.compute(), result.data.compute()))

    def test_gcvi(self):
        data = create_data_twoband('green', 'nir')
        result = (data.sel(band='nir') / data.sel(band='green') - 1.0).clip(
            0, 10
        ) / 10.0

        vi = data.gw.gcvi()

        self.assertTrue(np.allclose(vi.data.compute(), result.data.compute()))

    def test_evi(self):
        data = create_data_threeband('blue', 'red', 'nir')

        result = (
            2.5
            * (data.sel(band='nir') - data.sel(band='red'))
            / (
                (
                    data.sel(band='nir')
                    + 6.0 * data.sel(band='red')
                    - 7.5 * data.sel(band='blue')
                )
                + 1.0
            )
        )

        vi = data.gw.evi()

        self.assertTrue(np.allclose(vi.data.compute(), result.data.compute()))

    def test_evi2(self):
        data = create_data_twoband('red', 'nir')
        result = (
            2.5
            * (
                (data.sel(band='nir') - data.sel(band='red'))
                / (data.sel(band='nir') + 1.0 + (2.4 * (data.sel(band='red'))))
            )
        ).clip(0, 1)

        vi = data.gw.evi2()

        self.assertTrue(np.allclose(vi.data.compute(), result.data.compute()))

    def test_avi(self):
        data = create_data_twoband('red', 'nir')
        result = (
            (
                (
                    data.sel(band='nir')
                    * (1.0 - data.sel(band='red'))
                    * (data.sel(band='nir') - data.sel(band='red'))
                )
                ** 0.3334
            )
            .fillna(0)
            .clip(0, 1)
        )

        vi = data.gw.avi(nodata=0)

        self.assertTrue(np.allclose(vi.data.compute(), result.data.compute()))


if __name__ == '__main__':
    unittest.main()
