## `GeoWombat` is a Python library designed to wrap NumPy arrays with geo-information

---

### Requirements

* CartoPy
* GDAL
* GeoPandas
* MpGlue
* NumPy
* Rasterio

### Example usage:

```python
>>> import geowombat as gwb
```

### Open directly from a file

```python
>>> with gwb.GeoWombat('example.tif', backend='rasterio') as gsrc:
>>>     garray = gsrc.read(lazy=True)
```

### Convert NumPy arrays to GeoArrays

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

### GeoArray properties

GeoArrays maintain coordinates

```python
>>> print(garray.extent)
>>> geo_sub = garray.extract(row_start=500, rows=500, col_start=500, cols=200)  
>>> print(geo_sub.extent)
```

### Transformations

Resample array and cell size

```python
>>> array_proj = garray.to_crs(cell_size=200.0, resample='near')
```

Re-project coordinates

```python
>>> array_proj = garray.to_crs(crs=4326)
```

### I/O

Write array to geo-referenced file

```python
>>> garray.to_raster('image.tif')
```
