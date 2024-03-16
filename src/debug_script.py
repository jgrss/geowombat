#%% 
import rasterio 

#create raster 3000x3000 with 1 band and float64, set projection to WGS84, pin to 0,0
with rasterio.open(
      "../data/test.tif",
      "w",
      driver="GTiff",
      height=2000,
      width=2000,
      count=1,
      dtype="float64",
      crs="EPSG:4326",
      transform=rasterio.Affine(0.0001, 0.0, 0.0, 0.0, -0.0001, 0.0),
) as dst:
   dst.write(
      np.random.rand(3000, 3000),
      1,
   )
# use bounds to create random 500 points in test.tif
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point

with rasterio.open("../data/test.tif") as src:
   bounds = src.bounds

np.random.seed(0)
n = 500
x = np.random.uniform(bounds.left, bounds.right, n)
y = np.random.uniform(bounds.bottom, bounds.top, n)
df = pd.DataFrame({"x": x, "y": y})
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y),crs="EPSG:4326")
gdf.to_file("../data/test_points.shp")


 # %% 
import sys 
sys.path.append("/") 
import cProfile

sys.path.append("/geowombat") 
import ray

import geowombat as gw 

from geowombat.data import (
    l8_224078_20200518_points,
    l8_224078_20200518_polygons,
    l8_224078_20200518,
    l8_224078_20200518_B2,
)

import numpy as np
import geopandas as gpd
from sklearn.preprocessing import LabelEncoder
import timeit

aoi = gpd.read_file("../data/test_points.shp") 
aoi.geometry = aoi.geometry.buffer(0.125)
# aoi["id"] = LabelEncoder().fit_transform(aoi.name)
# aoi = aoi.drop(columns=["name"])

l8_224078_20200518_B2_values = np.array(
    [7966, 8030, 7561, 8302, 8277, 7398], dtype="float64"
)

l8_224078_20200518_values = np.array(
    [
        [7966, 8030, 7561, 8302, 8277, 7398],
        [7326, 7490, 6874, 8202, 7982, 6711],
        [6254, 8080, 6106, 8111, 7341, 6007],
    ],
    dtype="float64",
)
import cProfile
def test():
   with gw.open("../data/test.tif") as src:
      df_none = gw.extract(
         src, aoi, band_names=["blue", "green", "red"], use_client=False
      )
   print(df_none)

cProfile.run('test()',)

# %%
import cProfile
import ray
ray.init()

def test():

   with gw.open(l8_224078_20200518) as src:
      df_ray = gw.extract(
         src, aoi, band_names=["blue", "green", "red"], use_ray_client=True
      )
   print(df_ray)
cProfile.run('test()')

ray.shutdown()

#%%
import rasterio as rio
from rasterio import features
from rasterio import features
from rasterio.enums import MergeAlg
from rasterio.plot import show
from numpy import int16
from matplotlib import pyplot as plt
import numpy as np

with rio.open(l8_224078_20200518) as src:
   features.rasterize(
         aoi.geometry,
         out_shape=src.shape,
         transform=src.transform,
         all_touched=True,
         fill=0,
         default_value=1,
         dtype="uint8",
   )

# rasterize will cell values equal to index of aoi
from rasterio import features
import geopandas as gpd

aoi = gpd.read_file(l8_224078_20200518_polygons)
aoi.geometry = aoi.buffer(1000)

geom_value = ((geom,value) for geom, value in zip(aoi.geometry, range(1,len(aoi.index))))

with rio.open(l8_224078_20200518) as src:
      rasterized = features.rasterize(
         geom_value,
         out_shape=src.shape,
         transform=src.transform,
         all_touched=True,
         fill=0,
         default_value=1,
         dtype="uint16",
      )     
# get the xid location of all non-zero values
xid = np.where(rasterized > 0)

with gw.open([l8_224078_20200518]*5) as data:


   if isinstance(bands, list):
         bands_idx = (np.array(bands, dtype="int64") - 1).tolist()
   elif isinstance(bands, np.ndarray):
         bands_idx = (bands - 1).tolist()
   elif isinstance(bands, int):
         bands_idx = [bands]
   else:

         if shape_len > 2:
            bands_idx = list(range(0, data.gw.nbands))

   if not ray.is_initialized():
         ray.init()
   res = ray.get(
         self.extract_data_slice.remote(src, bands_idx, yidx, xidx)
   )

#%%
fig, ax = plt.subplots(1, figsize = (10, 10))
show(rasterized, ax = ax)
plt.gca().invert_yaxis()
#%%

with gw.open([l8_224078_20200518_B2]*5, stack_dim='band') as src:
   display(src)

# %%
from pandas.testing import assert_frame_equal
import ray

with gw.open(l8_224078_20200518) as src:
    ray.init()

    df_ray = gw.extract(
        src,
        aoi,
        band_names=["blue", "green", "red"],
        use_ray_client=True,
    )
    ray.shutdown()

    print(df_ray)

    df_none = gw.extract(
        src, aoi, band_names=["blue", "green", "red"], use_client=False
    )
    print(df_none)

    df_true = gw.extract(
        src, aoi, band_names=["blue", "green", "red"], use_client=True
    )
assert_frame_equal(df_ray, df_none)
assert_frame_equal(df_ray, df_true)
# %%
