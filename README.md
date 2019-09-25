[](#mit-license)[](#python-3.6)[](#package-version)

[![MIT license](https://img.shields.io/badge/License-MIT-black.svg)](https://lbesson.mit-license.org/)
[![Python 3.6](https://img.shields.io/badge/python-3.6-black.svg)](https://www.python.org/downloads/release/python-360/)
![Package version](https://img.shields.io/badge/version-0.1.0-blue.svg?cacheSeconds=2592000)

![](data/wombat.png)

## *GeoWombat* is a Python package to apply geo-functions to `Xarray` and `Dask` data

Like a wombat, [`GeoWombat`](https://github.com/jgrss/geowombat) has a simple interface with a strong backend. `GeoWombat` uses
[`Rasterio`](https://github.com/mapbox/rasterio), [`Xarray`](http://xarray.pydata.org/en/stable/) and [`Dask`](https://dask.org/) 
to apply geo-functions to satellite imagery. 

`GeoWombat` is designed to provide specialized functionality to `Xarray` and `Dask` data, using `Rasterio` for 
overhead space- or -airborne imagery I/O.

---

### Source code

```
git clone https://github.com/jgrss/geowombat.git
```

### Installation

```
python3 install --user git+https://github.com/jgrss/geowombat
```

---

### Example usage:

```python
>>> import geowombat as gw
>>> import xarray as xr
```

##### `GeoWombat` methods are automatically appended to `Xarray` `Datasets` and `DataArrays`

```python
>>> with xr.open_rasterio('example.tif') as ds:
>>>     print(ds.gw)
```

##### Use `GeoWombat` to open a file as an `Xarray.DataArray`

```python
>>> with gw.open('example.tif') as ds:
>>>     print(ds)    # `ds` is an `Xarray.DataArray`
```

##### Use `GeoWombat` to open a file as an `Xarray.Dataset`

```python
>>> # Open a 3-band image with blue, green, and red wavelengths
>>> with gw.open('example.tif',
>>>              xarray_return='dataset', 
>>>              band_names=['blue', 'green', 'red']) as ds:
>>>
>>>     print(ds)    # `ds` is an `Xarray.Dataset`
```

##### Extract values for every pixel intersecting point geometry (point.shp)

```python
>>> with gw.open('example.tif') as ds:
>>>     df = ds.gw.extract('point.shp')
```

##### Extract values for every pixel within polygon geometry (poly.gpkg)

```python
>>> with gw.open('example.tif') as ds:
>>>     df = ds.gw.extract('poly.gpkg', bands=3, band_names=['red'], frac=0.1)
```

##### Write to GeoTiff on a `Dask` distributed cluster

```python
>>> import joblib
>>> from dask.distributed import Client, LocalCluster
>>>
>>> cluster = LocalCluster(n_workers=8,
>>>                        threads_per_worker=1,
>>>                        scheduler_port=0,
>>>                        processes=False)
>>> 
>>> client = Client(cluster)
>>> 
>>> with joblib.parallel_backend('dask'):
>>>
>>>     with gw.open('example.tif') as ds:
>>>
>>>         # ds = <do something>
>>>
>>>         ds.gw.to_raster('output.tif',
>>>                         n_jobs=8,
>>>                         row_chunks=512,
>>>                         col_chunks=512,
>>>                         pool_chunksize=50,
>>>                         tiled=True,
>>>                         blockxsize=2048,
>>>                         blockysize=2048,
>>>                         compress='lzw')
>>>
>>> client.close()
>>> cluster.close()
```

##### Let `GeoWombat` handle the distributed cluster

```python
>>> def user_func(ds_):
>>>     """Function to mask a DataArray where values are equal to 1"""
>>>     return ds_.where(ds_ == 1)
>>>
>>> # Open a 3-band image with blue, green, and red wavelengths
>>> with gw.open('example.tif') as ds:
>>>
>>>     ds.gw.apply('output.tif',
>>>                 user_func,
>>>                 n_jobs=8,
>>>                 row_chunks=512,
>>>                 col_chunks=512,
>>>                 pool_chunksize=50,
>>>                 tiled=True,
>>>                 blockxsize=2048,
>>>                 blockysize=2048,
>>>                 compress='lzw')
```

##### Use `GeoWombat` for a machine learning pipeline

```python
>>> from geowombat.model import Model
>>>
>>> # Extract training samples
>>> with gw.open('example.tif') as ds:
>>>
>>>     df = ds.gw.extract('poly.gpkg', bands=3, band_names=['red', 'nir'], frac=0.1)
>>>
>>>     df['response'] = np.random.randint(0, high=4, size=df.shape[0])
>>>     df['weights'] = np.random.randn(df.shape[0])
>>>
>>>     # Fit a classifier
>>>     clf = Model(name='lightgbm', use_dask=True, n_jobs=4)
>>>     clf.fit(df, x=['red', 'nir'], y='response', sample_weight='weights')
>>>
>>>     # Save the model to file (optional)
>>>     clf.to_file('lgm.model')
>>>
>>>     ds.gw.predict(clf)
```

```python
>>> # Load the model from file
>>> clf.from_file('lgm.model')
>>>
>>> # Apply the model to an image
>>> with gw.open('example.tif') as ds:
>>>     ds.gw.predict(clf)
```

---

### Old GeoNumPy functionality

##### Convert NumPy arrays to GeoArrays

[Rasterio](https://github.com/mapbox/rasterio)

```python
>>> import rasterio
>>>
>>> with rasterio.open('example.tif') as src:
>>>
>>>     array = src.read(1)
>>>
>>>     # Wrap GeoWombat
>>>     garray = gwb.GeoArray(array, src)
```

[MpGlue](https://github.com/jgrss/mpglue)

```python
>>> import mpglue as gl
>>>
>>> with gl.ropen('example.tif') as src:
>>>
>>>     array = src.read(bands=-1)
>>>
>>>     # Wrap GeoWombat
>>>     garray = gwb.GeoArray(array, src)
```

GDAL

```python
>>> from osgeo import gdal
>>>
>>> src = gdal.Open('example.tif')
>>>
>>> array = src.GetRasterBand(1).ReadAsArray()
>>>
>>> # Wrap GeoWombat
>>> garray = gwb.GeoArray(array, src)
>>>
>>> src = None
```

##### GeoArray properties

GeoArrays maintain coordinates

```python
>>> geo_sub = garray.extract(row_start=500, rows=500, col_start=500, cols=200)  
>>>
>>> print(garray.extent)
>>> print(geo_sub.extent)
```

##### Transformations

Resample array and cell size

```python
>>> array_proj = garray.to_crs(cell_size=200.0, resample='near')
```

Re-project coordinates

```python
>>> array_proj = garray.to_crs(crs=4326)
```

##### Pandas-like window methods

5x5 moving average

```python
>>> mean = garray.moving(5).mean()
```

##### I/O

Write array to geo-referenced file

```python
>>> garray.to_raster('image.tif')
```

#### See the [notebooks](https://github.com/jgrss/geowombat/tree/master/notebooks) for more detailed examples
