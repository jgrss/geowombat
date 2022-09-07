#%%


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
from geowombat.data import rgbn_20160101, rgbn_20160401, rgbn_20160517


# https://github.com/phausamann/sklearn-xarray/issues/52

with gw.open(
    [rgbn_20160101, rgbn_20160401, rgbn_20160517],
    band_names=["blue", "green", "red", "nir"],
    time_names=["t1", "t2", "t3"],
) as srcs:
    # srcs.load()
    print(srcs)
land_use = np.tile(
    "water", (srcs.sizes["time"], srcs.sizes["y"], srcs.sizes["x"])
).astype(object)
land_use[srcs.sel(band="green").values > 128] = 1
land_use[srcs.sel(band="green").values < 88] = np.NaN

land_use = land_use.astype(str)
srcs.coords["land_use"] = (["time", "y", "x"], land_use)


Xs = srcs.stack(sample=("x", "y", "time")).T

#%%
from sklearn_xarray import Target
from sklearn.preprocessing import LabelEncoder

Xna = Xs[~Xs["land_use"].isnull()]

ys = Target(coord="land_use", transform_func=LabelEncoder().fit_transform)(Xna)

#%%
from sklearn_xarray import wrap
from sklearn_xarray.preprocessing import Sanitizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


wrapper = Pipeline(
    [("cls", wrap(GaussianNB(), reshapes="band"))]  # ("san", Sanitizer()),
)

wrapper.fit(Xna, ys)
#%%
yp = wrapper.predict(Xs)
yp


y = (
    wrapper.predict(Xs)
    .unstack("sample")
    .assign_coords(coords={"band": "land_use"})
    .expand_dims(dim="band")
    .transpose("time", "band", "y", "x")
)

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(dpi=200, figsize=(5, 5))

y.sel(time="t1").plot(robust=True, ax=ax)
plt.tight_layout(pad=1)


#%%
###############

# https://github.com/phausamann/sklearn-xarray/issues/52

with gw.open(
    [rgbn_20160101],  # , rgbn_20160401, rgbn_20160517],
    band_names=["blue", "green", "red", "nir"],
    # time_names=["t1", "t2", "t3"],
) as src:
    src.load()


land_use = np.tile(
    "water", (src.sizes["y"], src.sizes["x"])  # src.sizes["time"],
).astype(object)
land_use[(src.sel(band="green").values > 128)[0]] = "forest"
land_use = land_use.astype(str)
src.coords["land_use"] = (["y", "x"], land_use)


X = src.stack(
    sample=(
        "x",
        "y",
    )
).T
X
#%%
from sklearn_xarray import Target
from sklearn.preprocessing import LabelBinarizer

y = Target(coord="land_use", transform_func=LabelBinarizer().fit_transform)(X)

#%%
from sklearn_xarray import wrap
from sklearn.linear_model import LogisticRegression

wrapper = wrap(LogisticRegression(), reshapes="band")
wrapper.fit(X, y)
#%%
yp = wrapper.predict(X)
yp

# %% test remove missing

Xna = X[~X[targ_name].isnull()]
