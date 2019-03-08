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
