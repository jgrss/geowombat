# %%
import sys

sys.path.append("/")
sys.path.append("/geowombat")
import geowombat as gw

# %%

from geowombat.data import l8_224078_20200518, l8_224078_20200518_polygons
from geowombat.ml import fit, predict, fit_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
import geopandas as gpd
import matplotlib.pyplot as plt

le = LabelEncoder()

# The labels are string names, so here we convert them to integers
labels = gpd.read_file(l8_224078_20200518_polygons)
labels["lc"] = le.fit(labels.name).transform(labels.name)
print(labels)

# Use a data pipeline
pl = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        ("clf", GaussianNB()),
    ]
)
#%%
# Fit the classifier
with gw.config.update(ref_res=100):
    with gw.open(l8_224078_20200518, chunks=128) as src:
        X, Xy, clf = fit(src, pl, labels, col="lc")
        y = predict(src, X, clf)
        print(y)
#%%

fig, ax = plt.subplots(dpi=200, figsize=(5, 5))

with gw.config.update(ref_res=300):
    with gw.open(l8_224078_20200518, chunks=128) as src:
        X, Xy, clf = fit(src, pl, labels, col="name")

        y = predict(src, X, clf)
        y.plot(robust=True, ax=ax)
plt.tight_layout(pad=1)

# %%
from sklearn.model_selection import GridSearchCV, KFold
from sklearn_xarray.model_selection import CrossValidatorWrapper
import matplotlib.pyplot as plt
fig, ax = plt.subplots(dpi=200)

cv = CrossValidatorWrapper(KFold())
gridsearch = GridSearchCV(
   pl, cv=cv, param_grid={'pca__n_components': [1,2,3]})

with gw.config.update(ref_res=300):
    with gw.open(l8_224078_20200518, chunks=128) as src:
         X, Xy, clf = fit(src, pl, labels, col="lc")
         gridsearch.fit(Xy[0],Xy[1])
         print(gridsearch.best_params_)
         print(gridsearch.best_score_)
         clf.set_params(**gridsearch.best_params_)
         y = predict(src, X, clf)
         y.plot(robust=True, ax=ax)

#%%
from sklearn.cluster import KMeans

from geowombat.data import (
    l8_224078_20200518_points,
    l8_224078_20200518_polygons,
    l8_224078_20200518,
)

aoi_point = gpd.read_file(l8_224078_20200518_points)
aoi_point["lc"] = LabelEncoder().fit_transform(aoi_point.name)
aoi_point = aoi_point.drop(columns=["name"])

aoi_poly = gpd.read_file(l8_224078_20200518_polygons)
aoi_poly["lc"] = LabelEncoder().fit_transform(aoi_poly.name)
aoi_poly = aoi_poly.drop(columns=["name"])

 
 
cl_w_feat = Pipeline(
    [
        ("pca", PCA()),
        ("clf", KMeans(3, random_state=0)),
    ]
)

from sklearn.model_selection import GridSearchCV, KFold
from sklearn_xarray.model_selection import CrossValidatorWrapper

cv = CrossValidatorWrapper(KFold())
gridsearch = GridSearchCV(pl, cv=cv, param_grid={"pca__n_components": [1, 2, 3]})

with gw.config.update(ref_res=300):
   with gw.open(l8_224078_20200518, chunks=128) as src:
      X, Xy, clf = fit(src, pl_wo_feat, aoi_poly, col="lc")
      gridsearch.fit(Xy[0], Xy[1])

      clf.set_params(**gridsearch.best_params_)
      y1 = predict(src, X, clf)

      pl_wo_feat.set_params(**gridsearch.best_params_)
      y2 = fit_predict(src, pl_wo_feat, aoi_poly, col="lc")

\

#%%        
>>> from sklearn_xarray.model_selection import CrossValidatorWrapper
>>> from sklearn.model_selection import GridSearchCV, KFold
>>>
>>> cv = CrossValidatorWrapper(KFold())
>>> pipeline = Pipeline([
...     ('pca', wrap(PCA(), reshapes='feature')),
...     ('cls', wrap(LogisticRegression(), reshapes='feature'))
... ])
>>>
>>> gridsearch = GridSearchCV(
...     pipeline, cv=cv, param_grid={'pca__n_components': [20, 40, 60]}
... )
>>>
>>> gridsearch.fit(X, y) 
GridSearchCV(...)
>>> gridsearch.best_params_
{'pca__n_components': 20}
>>> gridsearch.best_score_
0.9182110801609408



#%%


import geowombat as gw
from geowombat.data import l8_224078_20200518, l8_224078_20200518_polygons
from geowombat.ml import fit

import geopandas as gpd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB

le = LabelEncoder()

# The labels are string names, so here we convert them to integers
labels = gpd.read_file(l8_224078_20200518_polygons)
labels['lc'] = le.fit(labels.name).transform(labels.name)

# Use a data pipeline
pl = Pipeline([('scaler', StandardScaler()),
                ('pca', PCA()),
                ('clf', GaussianNB())])

# Fit the classifier
with gw.config.update(ref_res=100):
    with gw.open(l8_224078_20200518, chunks=128) as src:
        X, Xy, clf = fit(src, pl, labels, col='lc')

print(clf)
#%%

from geowombat.ml import fit_predict
import matplotlib.pyplot as plt

fig, ax = plt.subplots(dpi=200)

with gw.config.update(ref_res=100):
    with gw.open(l8_224078_20200518 ) as src:
        y = fit_predict(src, pl, labels, col='lc')
        y.plot(robust=True, ax=ax)
        print(y)

#%%

    with gw.config.update(ref_res=100):
        with gw.open([l8_224078_20200518, l8_224078_20200518], time_names=['t1', 't2'], stack_dim='time', chunks=128) as src:
            y = fit_predict(src, pl, labels, col='lc')
            print(y)

#%%
 

    fig, ax = plt.subplots(dpi=200,figsize=(5,5))

    # Fit the classifier
    with gw.config.update(ref_res=100):
        with gw.open(l8_224078_20200518, chunks=128) as src:
            X, Xy, clf = fit(src, pl, labels, col="lc")
            y = predict(src, X, clf)
            y.plot(robust=True, ax=ax)
    plt.tight_layout(pad=1)
#%%
from sklearn.cluster import KMeans

fig, ax = plt.subplots(dpi=200,figsize=(5,5))

cl = Pipeline([ ('scaler', StandardScaler()),
                ('pca', PCA()),
                ('clf', KMeans(n_clusters=3, random_state=0))])

# fit and predict unsupervised classifier
with gw.config.update(ref_res=300):
    with gw.open(l8_224078_20200518) as src:
        X, Xy, clf = fit(src, cl)
        y = predict(src, X, clf)
        y.plot(robust=True, ax=ax)
plt.tight_layout(pad=1)

# Fit_predict unsupervised classifier
with gw.config.update(ref_res=300):
    with gw.open(l8_224078_20200518) as src:
        y= fit_predict(src, cl)
        y.plot(robust=True, ax=ax)
plt.tight_layout(pad=1)


#%%


from sklearn.model_selection import GridSearchCV, KFold
from sklearn_xarray.model_selection import CrossValidatorWrapper

cv = CrossValidatorWrapper(KFold())
gridsearch = GridSearchCV(pl, cv=cv, scoring='balanced_accuracy',
                    param_grid={"pca__n_components": [1, 2, 3]})

fig, ax = plt.subplots(dpi=200,figsize=(5,5))


# Use a data pipeline
pl = Pipeline([('scaler', StandardScaler()),
                ('pca', PCA()),
                ('clf', GaussianNB())])

with gw.config.update(ref_res=300):
    with gw.open(l8_224078_20200518) as src:
        # fit a model to get Xy used to train model
        X, Xy, clf = fit(src, pl, labels, col="lc")

        # fit cross valiation and parameter tuning
        gridsearch.fit(*Xy)
        print(gridsearch.best_params_)
        print(gridsearch.best_score_)

        # get set tuned parameters
        # Note: predict(gridsearch.best_model_) not currently supported 
        clf.set_params(**gridsearch.best_params_)
        y1 = predict(src, X, clf)
        y.plot(robust=True, ax=ax)
plt.tight_layout(pad=1)

# %%
