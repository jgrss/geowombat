![](data/logo.png)

[![](https://img.shields.io/badge/License-MIT-black.svg)](https://lbesson.mit-license.org/)
[![](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)
![](https://img.shields.io/badge/version-1.2.29-blue.svg?cacheSeconds=2592000)

### GeoWombat: Utilities for geospatial data

Like a wombat, GeoWombat has a simple interface (for raster I/O) with a strong backend (for data processing at scale).

## Basic usage

```python
>>> import geowombat as gw
```

Use a context manager and Xarray plotting to analyze processing chains

```python
>>> # Define satellite sensors (here, Landsat 7)
>>> with gw.config.update(sensor='l7'):
>>>
>>>     # Open images as Xarray DataArrays
>>>     with gw.open('LT05_L1TP_227083_20110123_20161011_01_T1.tif') as src:
>>>
>>>         # Apply calculations using Xarray and Dask
>>>         results = src.sel(band=['blue', 'green', 'red']).mean(dim='band')
>>>
>>>         # Check results by computing the task and plotting
>>>         results.gw.imshow()
```

Use a context manager to pass sensor information to geowombat methods

```python
>>> # Set the sensor as Sentinel 2
>>> with gw.config.update(sensor='s2'):
>>>
>>>     # Open a Sentinel 2 image
>>>     with gw.open('L1C_T20HPH_A002352_20151204T141125_MTD.tif') as src:
>>>
>>>         # Use built-in normalization methods, such as the NDVI
>>>         ndvi = src.gw.ndvi(scale_factor=0.0001)
>>>
>>>         # Check results by computing the task and plotting
>>>         ndvi.gw.imshow()
```

Computation scales easily over large datasets with minimal changes to the code.

```python
>>> # Set a reference image to align to
>>> with gw.config.update(ref_image='ref_image.tif'):
>>>
>>>     # Open images as Xarray DataArrays
>>>     with gw.open('image_a.tif') as srca, gw.open('image_b.tif') as srcb:
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
