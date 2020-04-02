![](data/logo.png)

[![](https://img.shields.io/badge/License-MIT-black.svg)](https://lbesson.mit-license.org/)
[![](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)
![](https://img.shields.io/badge/version-1.2.9-blue.svg?cacheSeconds=2592000)

### *GeoWombat* is a Python package for geo-utilities applied to air- and space-borne imagery

Like a wombat, [`GeoWombat`](https://github.com/jgrss/geowombat) has a simple interface with a strong backend. GeoWombat uses
[`Rasterio`](https://github.com/mapbox/rasterio), [`Xarray`](http://xarray.pydata.org/en/stable/) and [`Dask`](https://dask.org/) 
for I/O and distributed computing with named coordinates.

## Basic usage

```python
>>> import geowombat as gw
```

Use a context manager and Xarray plotting to analyze processing chains

```python
>>> # Define geographic bounds to warp images to
>>> with gw.config.update(ref_bounds=bounds):
>>>
>>>     # Open images as Xarray DataArrays
>>>     with gw.open('image_a.tif') as srca, \
>>>         gw.open('image_b.tif') as srcb:
>>>
>>>         # Apply calculations using Xarray and Dask
>>>         results = srca.sel(band=1) * srcb.sel(band=[1, 2, 3]).mean(dim='band')
>>>
>>>         # Check results by computing the task and plotting
>>>         results.gw.imshow()
```

Computation scales easily over large datasets with minimal changes to the code.

```python
>>> # Set a reference image to align to
>>> with gw.config.update(ref_image='image_a.tif'):
>>>
>>>     # Open images as Xarray DataArrays
>>>     with gw.open('image_a.tif') as srca, \
>>>         gw.open('image_b.tif') as srcb:
>>>
>>>         # The size of srca, srcb, and results are determined by the configuration context
>>>         results = srca.sel(band=1) * srcb.sel(band=[1, 2, 3]).mean(dim='band')
>>>
>>>         # Initiate computation by writing the results to file. 
>>>         # Mix process and thread workers to execute the task in parallel. 
>>>         results.gw.to_raster('output.tif', 
>>>                              n_workers=4, 
>>>                              n_threads=4,
>>>                              compress='lzw')
```

## Documentation
---

For more details, see [https://geowombat.readthedocs.io](https://geowombat.readthedocs.io).
