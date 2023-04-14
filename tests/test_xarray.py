import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from affine import Affine
from dask import delayed

import geowombat as gw
from geowombat.backends.xarray_ import delayed_to_xarray
from geowombat.data import (
    l3b_s2b_00390821jxn0l2a_20210319_20220730_c01,
    l8_224077_20200518_B2,
)


def scale(data: xr.DataArray) -> xr.DataArray:
    return data * 1e-4


def expand_time(dataset):
    """`open_mfdataset` preprocess function."""
    attrs = dataset.attrs.copy()
    attrs['transform'] = Affine(*attrs['transform'])
    attrs['res'] = tuple(attrs['res'])
    # Get the date
    file_date = datetime.strptime(
        Path(dataset.encoding['source']).stem.split('_')[3], '%Y%m%d'
    )
    darray = (
        dataset.to_array()
        .rename({'variable': 'band'})
        .assign_coords(time=file_date, y=dataset.y, x=dataset.x)
        .expand_dims('time')
        .transpose('time', 'band', 'y', 'x')
        .where(lambda x: x != x.nodatavals[0])
    )

    return darray.map_blocks(scale, template=darray).assign_attrs(**attrs)


class TestXarray(unittest.TestCase):
    def test_open(self):
        """Test to ensure dask/xarray version changes do not break behavior."""
        n_time = 2
        n_bands = 6
        with xr.open_mfdataset(
            [l3b_s2b_00390821jxn0l2a_20210319_20220730_c01] * n_time,
            concat_dim='time',
            chunks={
                'time': -1,
                'band': -1,
                'y': 256,
                'x': 256,
            },
            combine='nested',
            engine='h5netcdf',
            preprocess=expand_time,
            parallel=True,
        ) as ds:
            self.assertEqual(ds.gw.ntime, n_time)
            self.assertEqual(ds.gw.nbands, n_bands)

    def test_delayed_to_xarray(self):
        def random_func(bands: int, height: int, width: int) -> np.ndarray:
            np.random.seed(100)
            return np.random.random((bands, height, width))

        with gw.open(l8_224077_20200518_B2) as src:
            data_dst = delayed_to_xarray(
                delayed(random_func)(
                    src.gw.nbands, src.gw.nrows, src.gw.ncols
                ),
                shape=(src.gw.nbands, src.gw.nrows, src.gw.ncols),
                dtype=src.dtype,
                chunks=src.data.chunksize,
                coords={
                    'band': src.band.values.tolist(),
                    'y': src.y,
                    'x': src.x,
                },
                attrs=src.attrs,
            )
            ref_data = random_func(src.gw.nbands, src.gw.nrows, src.gw.ncols)

            self.assertTrue(np.allclose(ref_data, data_dst.data.compute()))
            self.assertEqual(src.shape, data_dst.shape)
