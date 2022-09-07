#%%

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


if labels[col].dtype != int:
    le = LabelEncoder()
    labels[col] = le.fit_transform(labels.name)


if isinstance(labels, str) or isinstance(labels, GeoDataFrame):
    labels = polygon_to_array(labels, col=col, data=data)

#%%
labels












# %% old attempt 
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


#%%  THIS GETS PAST DATA ISSUE 

from sklearn_xarray import wrap, Target

wrapper = Pipeline(
    [("san", Sanitizer()), ("cls", wrap(GaussianNB(), reshapes="band"))]
)

with gw.open(l8_224078_20200518) as src:
    X, Xy, clf = fit(src, pl_w_feat, aoi_point, col="lc")
    y1 = predict(src, X, clf)


# %%
import functools
import logging

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








import numpy as np
import geowombat as gw
from geowombat.data import rgbn_20160101, rgbn_20160401, rgbn_20160517


https://github.com/phausamann/sklearn-xarray/issues/52

with gw.open(
    [rgbn_20160101, rgbn_20160401, rgbn_20160517],
    band_names=["blue", "green", "red", "nir"],
    time_names=["t1", "t2", "t3"],
) as src:
    src.load()

land_use = np.tile(
    "water", (src.sizes["time"], src.sizes["y"], src.sizes["x"])
).astype(object)
land_use[src.sel(band="green").values > 128] = "forest"
land_use = land_use.astype(str)
src.coords["land_use"] = (["time", "y", "x"], land_use)



X = src.stack(sample=("x", "y", "time")).T
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
















#%%
from sklearn_xarray.preprocessing import Sanitizer
from sklearn_xarray import wrap
from geopandas.geodataframe import GeoDataFrame
from geowombat import polygon_to_array

pipeline = Pipeline(
    [   ('san',Sanitizer()),
        ("pca", wrap(PCA(n_components=2), )),
        ("cls", wrap(GaussianNB(), )),
    ]
)
with gw.open([l8_224078_20200518,l8_224078_20200518], time_names=['t1','t2']) as src:
    data, clf, labels, col = src , pl_w_feat, aoi_point, "lc"
    targ_name = "targ"
    targ_dim_name = "sample"
 
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

    labels = labels.where(labels != 0)
 
    # add labels into targ dim
    data.coords[targ_name] = (['time', "y", "x"], labels.data)
    
    data = xr.concat([data,labels], dim='targ')
    #%% X, Xna = self._prepare_predictors(data, targ_name)
    def _stack_it(data):
        return Stackerizer(stack_dims=("y", "x", ), direction="stack").fit_transform(
            data
        )


    X = _stack_it(data)

    # drop nans

    y = Target(
        coord=targ_name,
        transform_func=LabelEncoder().fit_transform,
        dim=targ_dim_name,
    )(X)

    clf.fit(X, y)


############################################


#%%

from sklearn_xarray.preprocessing import Sanitizer

pipeline = Pipeline(
    [   ('san',Sanitizer()),
        ("pca", wrap(PCA(n_components=2), )),
        ("cls", wrap(GaussianNB(), )),
    ]
)
with gw.open([l8_224078_20200518,l8_224078_20200518], time_names=['t1','t2']) as src:
    data, clf, labels, col = src , pl_w_feat, aoi_point, "lc"
    targ_name = "targ"
    targ_dim_name = "sample"



#%%fit(        self, data,clf,labels=None, col=None,

data, clf, labels, col = src, pl_w_feat, aoi_point, "lc"
targ_name = "targ"
targ_dim_name = "sample"

#  _prepare_labels(data, labels, col, targ_name)
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

# TODO: is this sufficient for single dates?

#%%     datatime = self._add_time_dim(data)
if not data.gw.has_time_coord:
    data= data.assign_coords(coords={"time": "t1"}).expand_dims(dim="time").transpose(
        "time", "band", "y", "x"
    )
 
#%% back to _prepare_labels
# labels = xr.concat([labels] * data.gw.ntime, dim="band").assign_coords(
#     coords={"band": data.time.values.tolist()}
# )

if data.gw.has_time_coord:
    labels = xr.concat([labels] * data.gw.ntime, dim="band").assign_coords(
                coords={"band": data.time.values.tolist()}
            )
    data.coords[targ_name] = (["time", "y", "x"], labels.data)
# else:
#     labels = xr.concat([labels], dim="band").assign_coords(
#         coords={"band": 'targ'}
#     )

# Mask 'no data' outside training data
# labels = labels.where(labels != 0)
#%%
data.coords[targ_name] = (['time',"y", "x"], labels.data)
# xr.concat([data, labels], dim=targ_name)

# labels = labels.reindex(band=[targ_name]) # doesnt work
labels = labels.sel(band=1).assign_coords(band=(targ_name))


#%% _prepare_predictors

#%% X = self._stack_it(data)

# X = Stackerizer(stack_dims=("y", "x", "time"), direction="stack").fit_transform(data)
X = Stackerizer(stack_dims=("y", "x"), direction="stack").fit_transform(data)
X
# drop nans
Xna = X[~X.sel(band=targ_name).isnull().values]
Xna

# TODO: groupby as a user option?
# Xgp = Xna.groupby(targ_name).mean('sample')

# return X, Xna

#%% _prepare_classifiers

if isinstance(clf, Pipeline):

    cln = Pipeline(
        [
            (clf_name, clf_)
            for clf_name, clf_ in clf.steps
            if not isinstance(clf_, Featurizer)
        ]
    )

    cln.steps.insert(0, ("featurizer", Featurizer()))

    clf = Pipeline(
        [(cln_name, WrappedClassifier(cln_)) for cln_name, cln_ in cln.steps]
    )

else:
    clf = WrappedClassifier(clf)

# return clf

#%% fit
y = Target(
    coord=targ_name,
    transform_func=LabelEncoder().fit_transform,
    dim=targ_dim_name,
)(Xna)
clf.fit(Xna, y)
