# %%
import sys

sys.path.append("/")
sys.path.append("/geowombat")
import geowombat as gw

from geowombat import polygon_to_array
from geowombat.ml.transformers import Stackerizer
from geowombat.ml.transformers import Featurizer_GW as Featurizer

import xarray as xr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn_xarray import wrap, Target

# from sklearn_xarray.preprocessing import Featurizer
import numpy as np
from geopandas.geodataframe import GeoDataFrame

from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn_xarray import wrap, Target


import numpy as np
import geowombat as gw
from geowombat.data import (
    rgbn_20160101,
    rgbn_20160401,
    rgbn_20160517,
    l8_224078_20200518,
    l8_224078_20200518_polygons,
)
import geopandas as gpd

# https://github.com/phausamann/sklearn-xarray/issues/52

aoi_poly = gpd.read_file(l8_224078_20200518_polygons)
aoi_poly["lc"] = LabelEncoder().fit_transform(aoi_poly.name) + 1
aoi_poly = aoi_poly.drop(columns=["name"])


with gw.open(
    l8_224078_20200518,
    # [l8_224078_20200518, l8_224078_20200518, l8_224078_20200518],
    band_names=["blue", "green", "red"],
    # time_names=["t1", "t2", "t3"],
) as srcs:
    # srcs.load()
    print(srcs)

#%%
srcs, labels, col = srcs, aoi_poly, "lc"

#%%

if not hasattr(srcs, "time"):
    srcs = (
        srcs.assign_coords(coords={"time": "t1"})
        .expand_dims(dim="time")
        .transpose("time", "band", "y", "x")
    )

if isinstance(labels, str) or isinstance(labels, GeoDataFrame):
    labels = polygon_to_array(labels, col=col, data=srcs)

labels = xr.concat([labels] * srcs.gw.ntime, dim="band").assign_coords(
    coords={"band": srcs.time.values.tolist()}
)

srcs.coords["land_use"] = (["time", "y", "x"], labels.values)
#%%

Xs = srcs.stack(sample=("x", "y", "time")).T

#%%
from sklearn_xarray import Target
from sklearn.preprocessing import LabelEncoder

Xna = Xs[~Xs["land_use"].isnull()]

ys = Target(coord="land_use", transform_func=LabelEncoder().fit_transform)(Xna)

#%%
from sklearn_xarray import wrap
from sklearn_xarray.preprocessing import Sanitizer, Featurizer
from sklearn.naive_bayes import GaussianNB

#%%
wrapper = Pipeline(
    [("cls", wrap(GaussianNB(), reshapes="band"))]
    # [('feat',Featurizer(feature_dim='targ')),("cls", GaussianNB())]  # doesnt work
)

wrapper.fit(Xna, ys)

#%% clusters
from sklearn.cluster import KMeans

wrapper = Pipeline(
    [("cls", wrap(KMeans(n_clusters=6, random_state=0), reshapes="band"))]
)


wrapper.fit(Xs)


#%%

y = (
    wrapper.predict(Xs)
    .unstack("sample")
    .assign_coords(coords={"band": "land_use"})
    .expand_dims(dim="band")
    .transpose("time", "band", "y", "x")
)
if y.gw.ntime == 1:
    y = y.sel(time="t1")

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(dpi=200, figsize=(5, 5))

y.plot(robust=True, ax=ax)
plt.tight_layout(pad=1)


###############################################################


#%% WORKING not classifying though

import sys

sys.path.append("/")
sys.path.append("/geowombat")
import geowombat as gw


from geowombat.data import (
    l8_224078_20200518_points,
    l8_224078_20200518_polygons,
    l8_224078_20200518,
)
from geowombat.ml import fit, predict, fit_predict

import numpy as np
import geopandas as gpd
from sklearn.preprocessing import LabelEncoder
from sklearn_xarray.preprocessing import Featurizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, KFold
from sklearn_xarray.model_selection import CrossValidatorWrapper
from geopandas.geodataframe import GeoDataFrame
from geowombat import polygon_to_array
from sklearn_xarray import wrap, Target

aoi_point = gpd.read_file(l8_224078_20200518_points)
aoi_point["lc"] = LabelEncoder().fit_transform(aoi_point.name)
aoi_point = aoi_point.drop(columns=["name"])

aoi_poly = gpd.read_file(l8_224078_20200518_polygons)
aoi_poly["lc"] = LabelEncoder().fit_transform(aoi_poly.name)
aoi_poly = aoi_poly.drop(columns=["name"])


from sklearn_xarray.preprocessing import Sanitizer

pl_w_feat = Pipeline(
    [
        # ("san", Sanitizer()),
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        ("clf", GaussianNB()),
    ]
)

#%%

with gw.open(l8_224078_20200518) as src:
    X, Xy, clf = fit(src, pl_w_feat, aoi_point, col="lc")
    y1 = predict(src, X, clf)

#%%
data, clf, labels, col = src, pl_w_feat, aoi_point, "lc"
targ_name = "targ"
targ_dim_name = "sample"


# always reencode +1
le = LabelEncoder()
labels[col] = le.fit_transform(labels[col]) + 1


if isinstance(labels, str) or isinstance(labels, GeoDataFrame):
    labels = polygon_to_array(labels, col=col, data=data)

# data.load()
#%%
data = (
    data.assign_coords(coords={"time": "t1"})
    .expand_dims(dim="time")
    .transpose("time", "band", "y", "x")
)
data
#%%
data.coords["targ"] = (["time", "y", "x"], labels.values)
#%%
X = data.stack(sample=("x", "y", "time")).T
X
from geowombat.ml.transformers import Stackerizer

X = Stackerizer(stack_dims=("y", "x", "time"), direction="stack").fit_transform(data)
Xna = X[~X[targ_name].isnull()]
#%%
y = Target(coord="targ", transform_func=LabelEncoder().fit_transform)(X)
y
#%%
from sklearn.ensemble import RandomForestClassifier

wrapper = Pipeline(
    [("san", Sanitizer()), ("cls", wrap(RandomForestClassifier(), reshapes="band"))]
)

wrapper.fit(X, y)


#%%
y = (
    wrapper.predict(X)
    .unstack(targ_dim_name)
    .assign_coords(coords={"band": targ_name})
    .expand_dims(dim="band")
    .transpose("time", "band", "y", "x")
)
#%%
# if y.gw.ntime == 1:
#     y = y.isel(time=0)
# y


# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(dpi=200, figsize=(5, 5))

y.plot(robust=True, ax=ax)
plt.tight_layout(pad=1)

# %%
