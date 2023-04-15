import unittest

import dask.array as da
import numpy as np
import xarray as xr

import geowombat as gw

BANDS = np.array([[[0.4]], [[0.1]]])


def create_data(band1: str, band2: str) -> xr.DataArray:
    return xr.DataArray(
        da.from_array(BANDS, chunks=(-1, 1, 1)),
        dims=('band', 'y', 'x'),
        coords={'band': [band1, band2], 'y': [1], 'x': [1]},
    )


class TestVI(unittest.TestCase):
    def test_wi(self):
        data = create_data('red', 'swir1')
        result = data.sel(band='red') + data.sel(band='swir1')
        result = (1.0 - (result.where(lambda x: x <= 0.5) / 0.5)).fillna(0)

        vi = data.gw.wi()

        self.assertTrue(np.allclose(vi.data.compute(), result.data.compute()))

    def test_kndvi(self):
        data = create_data('red', 'nir')
        result = (data.sel(band='nir') - data.sel(band='red')) / (
            data.sel(band='nir') + data.sel(band='red')
        )
        result = np.tanh(result**2)

        vi = data.gw.kndvi()

        self.assertTrue(np.allclose(vi.data.compute(), result.data.compute()))

    def test_ndvi(self):
        data = create_data('red', 'nir')
        result = (data.sel(band='nir') - data.sel(band='red')) / (
            data.sel(band='nir') + data.sel(band='red')
        )

        vi = data.gw.ndvi()

        self.assertTrue(np.allclose(vi.data.compute(), result.data.compute()))

    def test_nbr(self):
        data = create_data('nir', 'swir2')
        result = (data.sel(band='nir') - data.sel(band='swir2')) / (
            data.sel(band='nir') + data.sel(band='swir2')
        )

        vi = data.gw.nbr()

        self.assertTrue(np.allclose(vi.data.compute(), result.data.compute()))

    def test_gcvi(self):
        data = create_data('green', 'nir')
        result = (data.sel(band='nir') / data.sel(band='green') - 1.0).clip(
            0, 10
        ) / 10.0

        vi = data.gw.gcvi()

        self.assertTrue(np.allclose(vi.data.compute(), result.data.compute()))

    def test_evi2(self):
        data = create_data('red', 'nir')
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
        data = create_data('red', 'nir')
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
