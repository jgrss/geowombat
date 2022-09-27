![](data/logo.png)

[![](https://img.shields.io/badge/License-MIT-black.svg)](https://github.com/jgrss/geowombat/blob/main/LICENSE.txt)
[![](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue)](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue)
[![](https://badge.fury.io/gh/jgrss%2Fgeowombat.svg)](https://badge.fury.io/gh/jgrss%2Fgeowombat)
[![](https://github.com/jgrss/geowombat/actions/workflows/ci.yml/badge.svg)](https://github.com/jgrss/geowombat/actions?query=workflow%3ACI)
[![](https://img.shields.io/github/repo-size/jgrss/geowombat)](https://shields.io/category/size)
[![](https://readthedocs.org/projects/geowombat/badge/?version=latest&style=flat)](https://readthedocs.org/projects/geowombat/)

### GeoWombat: Utilities for geospatial data

Like a wombat, GeoWombat has a simple interface (for raster I/O) with a strong backend (for data processing at scale).

## Common Remote Sensing Uses
* Simple read/write for a variety of sensors, including:
    * Sentinel 2
    * Landsat 5-8
    * PlanetScope
    * Others
* Image mosaicking
* On-the-fly image transformations (reprojection)
* Point / polygon raster sampling, extraction
* Time series analysis
* Band math (NDVI, Tasseled cap, EVI etc)
* Image classification and regression
* Radiometry (BRDF normalization)
* Distributed processing

## Basic usage - Sentinel & Landsat

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
>>>         results.gw.save(
>>>             'output.tif',
>>>             num_workers=4,
>>>             compress='lzw'
>>>         )
```


## Documentation

For more details, see [https://geowombat.readthedocs.io](https://geowombat.readthedocs.io).

## Installation

### Conda Install
To allow easy installation and build of all dependencies we recommend installing via conda-forge:

Installing geowombat from the conda-forge channel can be achieved by adding conda-forge to your channels with:

```commandline
conda config --add channels conda-forge
conda config --set channel_priority strict
```
Once the conda-forge channel has been enabled, geowombat can be installed with conda:

```commandline
conda install geowombat
```

or faster with mamba:

```commandline
mamba install geowombat
```

### Pip Install
GeoWombat is not on PyPi, but it can be installed with `pip`. We provide detailed instructions in our [documentation](https://geowombat.readthedocs.io/en/latest/install.html).

### Universal Install Via Docker
If you are having trouble installing geowombat, the surest way to get it up and running is with Docker containers.
See the `Dockerfile`, or for more details instructions, see the guide on [pygis.io](https://mmann1123.github.io/pyGIS/docs/b_conda_started.html).

## Learning

If you are new to geospatial programming in Python please refer to [pygis.io](https://pygis.io)
