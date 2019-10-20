![](data/logo.png)

## *GeoWombat* is a Python package for geo-utilities with `Xarray` and `Dask` arrays

[![MIT license](https://img.shields.io/badge/License-MIT-black.svg)](https://lbesson.mit-license.org/)
[![Python 3.6](https://img.shields.io/badge/python-3.6-black.svg)](https://www.python.org/downloads/release/python-360/)
![Package version](https://img.shields.io/badge/version-1.0.0-blue.svg?cacheSeconds=2592000)

Like a wombat, [`GeoWombat`](https://github.com/jgrss/geowombat) has a simple interface with a strong backend. `GeoWombat` uses
[`Rasterio`](https://github.com/mapbox/rasterio), [`Xarray`](http://xarray.pydata.org/en/stable/) and [`Dask`](https://dask.org/) 
to apply geo-functions to satellite imagery. 

`GeoWombat` is designed to provide specialized functionality to `Xarray` and `Dask` data, using `Rasterio` for 
overhead space- or -airborne imagery I/O.

## Basic usage

```python
>>> import geowombat as gw
>>>
>>> # Open images as Xarray DataArrays
>>> with gw.open('image.tif', chunks=(1, 512, 512)) as ds:
>>>
>>>     # Do Xarray and Dask operations
>>>     dss = ds * 10.0
>>>
>>>     # Write the computation task to file
>>>     dss.gw.to_raster('output.tif', n_jobs=8)
```

## Documentation (temporary location)
---

Read the documentation by opening the [HTML file](https://github.com/jgrss/geowombat/tree/master/doc/build/html/index.html).

#### See the [notebooks](https://github.com/jgrss/geowombat/tree/master/notebooks) for more detailed examples
