![](wombat.png)

## *GeoWombat* is a Python library to manage _geo-aware_ NumPy arrays

Like wombats, `GeoWombat` is a simple interface with a powerful backend. `GeoWombat` utilizes libraries such as `NumPy` 
and `Rasterio` to wrap geo-information from satellite imagery onto n-dimensional arrays.

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

#### Example usage:

```python
>>> import geowombat as gwb
```

#### Open directly from a file

```python
>>> with gwb.open('example.tif') as gsrc:
>>>     garray = gsrc.read(bands=-1)
```

#### Convert NumPy arrays to GeoArrays

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

#### GeoArray properties

GeoArrays maintain coordinates

```python
>>> geo_sub = garray.extract(row_start=500, rows=500, col_start=500, cols=200)  
>>>
>>> print(garray.extent)
>>> print(geo_sub.extent)
```

#### Transformations

Resample array and cell size

```python
>>> array_proj = garray.to_crs(cell_size=200.0, resample='near')
```

Re-project coordinates

```python
>>> array_proj = garray.to_crs(crs=4326)
```

#### Pandas-like window methods

5x5 moving average

```python
>>> mean = garray.moving(5).mean()
```

#### I/O

Write array to geo-referenced file

```python
>>> garray.to_raster('image.tif')
```
