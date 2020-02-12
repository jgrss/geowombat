![](data/logo.png)

[![MIT license](https://img.shields.io/badge/License-MIT-black.svg)](https://lbesson.mit-license.org/)
[![Python 3.6](https://img.shields.io/badge/python-3.x-black.svg)](https://www.python.org/downloads/release/python-360/)
![Package version](https://img.shields.io/badge/version-1.2.2-blue.svg?cacheSeconds=2592000)

### *GeoWombat* is a Python package for geo-utilities applied to air- and space-borne imagery

Like a wombat, [`GeoWombat`](https://github.com/jgrss/geowombat) has a simple interface with a strong backend. GeoWombat uses
[`Rasterio`](https://github.com/mapbox/rasterio), [`Xarray`](http://xarray.pydata.org/en/stable/) and [`Dask`](https://dask.org/) 
for I/O and distributed computing with named coordinates.

## Basic usage

```python
>>> import geowombat as gw
>>>
>>> # Open images as Xarray DataArrays
>>> with gw.open('image.tif', chunks=512) as ds:
>>>
>>>     # Do Xarray and Dask operations
>>>     dss = ds * 10.0
>>>
>>>     # Write the computation task to file using 16 parallel jobs
>>>     dss.gw.to_raster('output.tif', n_workers=4, n_threads=4)
```

## Documentation
---

For more details, see [https://geowombat.readthedocs.io](https://geowombat.readthedocs.io).
