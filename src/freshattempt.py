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
) as src:
    src.load()

land_use = np.tile("water", (src.sizes["time"], src.sizes["y"], src.sizes["x"])).astype(
    object
)
land_use[src.sel(band="green").values > 128] = 1
land_use[src.sel(band="green").values < 88] = np.NaN

land_use = land_use.astype(str)
src.coords["land_use"] = (["time", "y", "x"], land_use)


X = src.stack(sample=("x", "y", "time")).T
X
#%%
from sklearn_xarray import Target
from sklearn.preprocessing import LabelEncoder

Xna = X[~X["land_use"].isnull()]

y = Target(coord="land_use", transform_func=LabelEncoder().fit_transform)(Xna)
print(y)
print(X)
#%%
from sklearn_xarray import wrap
from sklearn_xarray.preprocessing import Sanitizer
from sklearn.linear_model import LogisticRegression

wrapper = Pipeline(
    [("san", Sanitizer()), ("cls", wrap(LogisticRegression(), reshapes="band"))]
)

wrapper.fit(X, y)
#%%
yp = wrapper.predict(X)
yp


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
