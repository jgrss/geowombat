 # %% 
import sys 
sys.path.append("/") 
import cProfile

sys.path.append("/geowombat") 
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

aoi = gpd.read_file(l8_224078_20200518_polygons) 
aoi.geometry = aoi.geometry.buffer(1000)
aoi["id"] = LabelEncoder().fit_transform(aoi.name)
aoi = aoi.drop(columns=["name"])

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
   with gw.open(l8_224078_20200518) as src:
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
