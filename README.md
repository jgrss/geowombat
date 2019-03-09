![](wombat.png)

## *GeoWombat* is a Python library to manage _geo-aware_ NumPy arrays

Like wombats, [`GeoWombat`](https://github.com/jgrss/geowombat) is a simple interface with a strong backend. `GeoWombat` utilizes libraries such as 
[`MpGlue`](https://github.com/jgrss/mpglue), [`GDAL`](https://pypi.org/project/GDAL/), and [`Rasterio`](https://github.com/mapbox/rasterio) to wrap geo-information from satellite imagery onto 
[`NumPy`](http://www.numpy.org/) n-dimensional arrays.

`GeoWombat` arrays contain geo-information on top of `NumPy` functionality. Therefore, coordinates persist through 
array slices, coordinate transforms can be applied at the array level, and `NumPy` vectorized math operations can occur
on arrays of varying sizes and overlap.

`GeoWombat` is designed to provide low-level functionality to `NumPy` arrays from overhead space- or -airborne imagery.
For large image operations, libraries such as [`RIOS`](http://www.rioshome.org/en/latest/) or 
[`rio-mucho`](https://github.com/mapbox/rio-mucho) are better suited for out-of-memory processing of satellite imagery.

---

### Source code

```
git clone https://github.com/jgrss/geowombat.git
```

### Installation

```
python3 setup.py install --process-dependency-links
```

#### Dependencies

- CartoPy
- GDAL
- GeoPandas
- MpGlue
- NumPy
- Rasterio

### Example usage:

```python
>>> import geowombat as gwb
```

##### Open directly from a file

```python
>>> with gwb.open('example.tif') as src:
>>>     garray = src.read(bands=-1)
```

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
