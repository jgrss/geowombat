SEE
https://github.com/phausamann/sklearn-xarray/issues/52

# %%
import sys

sys.path.append("/")
sys.path.append("/geowombat")
import geowombat as gw


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


aoi_point = gpd.read_file(l8_224078_20200518_points)
aoi_point["lc"] = LabelEncoder().fit_transform(aoi_point.name)
aoi_point = aoi_point.drop(columns=["name"])

aoi_poly = gpd.read_file(l8_224078_20200518_polygons)
aoi_poly["lc"] = LabelEncoder().fit_transform(aoi_poly.name)
aoi_poly = aoi_poly.drop(columns=["name"])


pl_w_feat = Pipeline(
    [
        # ("featurizer", Featurizer()),
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        ("clf", GaussianNB()),
    ]
)


#%%

with gw.open(l8_224078_20200518) as src:
    X, Xy, clf = fit(src, pl_w_feat, aoi_point, col="lc")
    y1 = predict(src, X, clf)

#%%  try with wrap doesn't work
from sklearn_xarray import wrap

# pl_w_feat = Pipeline(
#     [
#         # ("featurizer", Featurizer()),
#         ("scaler", wrap(StandardScaler(), sample_dim="band")),
#         ("pca", wrap(wrap(PCA(), sample_dim="band"))),
#         ("clf", wrap(GaussianNB(), sample_dim="band")),
#     ]
# )

# with gw.open(l8_224078_20200518) as src:
#     X, Xy, clf = fit(src, pl_w_feat, aoi_point, col="lc")
#     y1 = predict(src, X, clf)

#%%fit(        self, data,clf,labels=None, col=None,
data, clf, labels, col = src, pl_w_feat, aoi_point, "lc"
targ_name = "targ"
targ_dim_name = "sample"


#%% data = self._prepare_labels(data, labels, col, targ_name)

if labels[col].dtype != int:
    le = LabelEncoder()
    labels[col] = le.fit_transform(labels.name)
    logger.warning(
        "target labels were not integers, applying LabelEncoder. Classes:",
        le.classes_,
        "Code:",
        le.transform(le.classes_),
    )

if isinstance(labels, str) or isinstance(labels, GeoDataFrame):
    labels = polygon_to_array(labels, col=col, data=data)

#%

# TODO: is this sufficient for single dates?
if not data.gw.has_time_coord:
    # data = self._add_time_dim(data)
    data = (
        data.assign_coords(coords={"time": "t1"})
        .expand_dims(dim="time")
        .transpose("time", "band", "y", "x")
    )


labels = xr.concat([labels] * data.gw.ntime, dim="band").assign_coords(
    coords={"band": data.time.values.tolist()}
)
# Mask 'no data' outside training data
labels = labels.where(labels != 0)

data.coords[targ_name] = (["time", "y", "x"], labels.data)

#%% X, Xna = self._prepare_predictors(data, targ_name)
def _stack_it(data):
    return Stackerizer(stack_dims=("y", "x", "time"), direction="stack").fit_transform(
        data
    )


X = _stack_it(data)
#%%
# drop nans
Xna = X[~X[targ_name].isnull()]


# clf = self._prepare_classifiers(clf)
#%%
# TODO: should we be using lazy=True?
y = Target(
    coord=targ_name,
    transform_func=LabelEncoder().fit_transform,
    dim=targ_dim_name,
)(Xna)
#%%
# TO DO: Validation checks
# Xna, y = check_X_y(Xna, y)

clf.fit(Xna, y)


# %%
check_is_fitted(clf)
clf.predict(X)

#########################################################################################

# %%
# X["targ"].values = clf.predict(X)

# y = X.reindex(band=[targ_name])
y = X.copy(deep=True)

#%%
# y[targ_name].values = clf.predict(X)
# y.coords[targ_name] = (["time", "y", "x"], clf.predict(X))
y.coords[targ_name] = (X.sample, clf.predict(X))

y.sel(band=1).valaues

#%%
y = X.reindex(band=[targ_name])
data.coords[targ_name] = (["time", "y", "x"], labels.data)
y = (
    y.unstack(targ_dim_name)
    # .reset(coords={"band": targ_name})
    # .expand_dims(dim="band")
    .transpose("time", "band", "y", "x")
)

y.sel(band="targ").values

#%%

da = xr.DataArray(
    clf.predict(X),
    [("x", X["x"].values), ("y", X["y"].values), ("time", X["time"].values)],
)

y = X.concat([X, da])
y
#%%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(dpi=200, figsize=(5, 5))
y.plot(robust=True, ax=ax)


# %%
clf.predict(X)

# %%
y = (
    clf.predict(X)
    .unstack(targ_dim_name)
    .assign_coords(coords={"band": targ_name})
    .expand_dims(dim="band")
    .transpose("time", "band", "y", "x")
)

# no point unit doesn't have nan
if mask_nodataval:
    y = self._mask_nodata(y=y, x=data)

xr.concat([data, y], dim="band").sel(band=targ_name)

# %%
