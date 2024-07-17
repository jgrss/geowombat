# %%
import sys

sys.path.insert(0, "/home/mmann1123/Documents/git/geowombat")
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


aoi = gpd.read_file(l8_224078_20200518_polygons)
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

with gw.open(l8_224078_20200518) as src:
    df_none = gw.extract(
        src, aoi, band_names=["blue", "green", "red"], use_client=False
    )
print(df_none)
# %%
import ray

ray.init()
with gw.open(l8_224078_20200518) as src:
    df_ray = gw.extract(
        src, aoi, band_names=["blue", "green", "red"], use_ray_client=True
    )
print(df_ray)
ray.shutdown()


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
