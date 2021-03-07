![](data/logo.png)

[![](https://img.shields.io/badge/License-MIT-black.svg)](https://lbesson.mit-license.org/)
[![](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)
![](https://img.shields.io/badge/version-1.6.3-blue.svg?cacheSeconds=2592000)

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
>>>         results.gw.to_raster('output.tif', 
>>>                              n_workers=4, 
>>>                              n_threads=4,
>>>                              compress='lzw')
```

## Installation

GeoWombat is not on PyPi, but it can be installed with `pip`. Most dependencies will be automatically installed. However, the exception is GDAL, which needs to be installed prior to executing the `pip` command below. See [the installation documentation](https://geowombat.readthedocs.io/en/latest/install.html) for details on installing GDAL. With GDAL installed, GeoWombat can be installed like:

```commandline
pip install git+https://github.com/jgrss/geowombat
```

Alternatively, use `git` to clone and build the repository like:

```commandline
git clone https://github.com/jgrss/geowombat.git
cd geowombat/
python setup.py build && pip install --upgrade . && rm -rf build/
```

### Update

To update GeoWombat:

```shell script
pip install --upgrade git+https://github.com/jgrss/geowombat
```

Or, to update a cloned repository:

```shell script
cd geowombat/
git pull origin master
python setup.py build && pip install --upgrade . && rm -rf build/
```

### (Optional) Install into a virtual environment on Linux

With `virtualenv`

```shell script
# Create a Python 3.7 virtual environment named gwenv
virtualenv -p python=3.7 gwenv

# Activate the virtual environment
source gwenv/bin/activate

# Install the libspatialindex and GDAL libraries
sudo apt install libspatialindex-dev libgdal-dev

# Install GeoWombat with all extra libraries
pip install git+https://github.com/jgrss/geowombat.git#egg=project[all]
```

With `conda`

```shell script
# Create a Python 3.7 virtual environment named gwenv
conda create -n gwenv python=3.7 cython numpy -c conda-forge

# Activate the virtual environment
conda activate gwenv

# Install the libspatialindex and GDAL libraries
sudo apt install libspatialindex-dev libgdal-dev
# Alternatively, use conda-forge
# conda install -c conda-forge libspatialindex gdal

# Install GeoWombat with all extra libraries
pip install git+https://github.com/jgrss/geowombat.git#egg=project[all]
```

## Documentation

For more details, see [https://geowombat.readthedocs.io](https://geowombat.readthedocs.io).
