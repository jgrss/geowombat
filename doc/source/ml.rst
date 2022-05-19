.. _ml:

Machine learning
================

Fit a classifier
----------------

.. ipython:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518, l8_224078_20200518_polygons
    from geowombat.ml import fit, predict, fit_predict

    import geopandas as gpd
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.naive_bayes import GaussianNB
    from sklearn.cluster import KMeans

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

Fit a classifier and predict on an array
----------------------------------------

.. ipython:: python

    from geowombat.ml import fit_predict
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(dpi=200)

    with gw.config.update(ref_res=100):
        with gw.open(l8_224078_20200518 ) as src:
            y = fit_predict(src, pl, labels, col='lc')
            y.plot(robust=True, ax=ax)
            print(y)

Fit a classifier with multiple dates
------------------------------------

.. ipython:: python

    with gw.config.update(ref_res=100):
        with gw.open([l8_224078_20200518, l8_224078_20200518], time_names=['t1', 't2'], stack_dim='time', chunks=128) as src:
            y = fit_predict(src, pl, labels, col='lc')
            print(y)

Train a supervised classifier and predict 
------------------------------


.. ipython:: python
    fig, ax = plt.subplots(dpi=200,figsize=(5,5))

    # Fit the classifier
    with gw.config.update(ref_res=100):
        with gw.open(l8_224078_20200518, chunks=128) as src:
            X, Xy, clf = fit(src, pl, labels, col="lc")
            y = predict(src, X, clf)
            y.plot(robust=True, ax=ax)
    plt.tight_layout(pad=1)

Train an unsupervised classifier and predict 
------------------------------
Unsupervised classifiers can also be used in a pipeline

.. ipython:: python

    cl = Pipeline([ ('scaler', StandardScaler()),
                    ('pca', PCA()),
                    ('clf', KMeans(n_clusters=3, random_state=0))])

    fig, ax = plt.subplots(dpi=200,figsize=(5,5))

    # fit and predict unsupervised classifier
    with gw.config.update(ref_res=300):
        with gw.open(l8_224078_20200518) as src:
            X, Xy, clf = fit(src, cl)
            y = predict(src, X, clf)
            y.plot(robust=True, ax=ax)
    plt.tight_layout(pad=1)

    fig, ax = plt.subplots(dpi=200,figsize=(5,5))

    # Fit_predict unsupervised classifier
    with gw.config.update(ref_res=300):
        with gw.open(l8_224078_20200518) as src:
            y= fit_predict(src, cl)
            y.plot(robust=True, ax=ax)
    plt.tight_layout(pad=1)


Predict with cross validation and parameter tuning
------------------------------
Crossvalidation and parameter tuning is now possible 
.. ipython:: python

    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn_xarray.model_selection import CrossValidatorWrapper

    cv = CrossValidatorWrapper(KFold())
    gridsearch = GridSearchCV(pl, cv=cv, scoring='balanced_accuracy',
                        param_grid={"pca__n_components": [1, 2, 3]})

    fig, ax = plt.subplots(dpi=200,figsize=(5,5))

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