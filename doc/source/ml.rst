.. _ml:

Machine learning
================

Fit a classifier
----------------

.. ipython:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518, l8_224078_20200518_polygons
    from geowombat.ml import fit

    import geopandas as gpd
    from sklearn_xarray.preprocessing import Featurizer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.naive_bayes import GaussianNB

    le = LabelEncoder()

    # The labels are string names, so here we convert them to integers
    labels = gpd.read_file(l8_224078_20200518_polygons)
    labels['lc'] = le.fit(labels.name).transform(labels.name)

    # Use a data pipeline
    pl = Pipeline([('featurizer', Featurizer()),
                   ('scaler', StandardScaler()),
                   ('pca', PCA()),
                   ('clf', GaussianNB())])

    # Fit the classifier
    with gw.config.update(ref_res=100):
        with gw.open(l8_224078_20200518, chunks=128) as src:
            X, clf = fit(src, labels, pl, col='lc')

    print(clf)

Fit a classifier and predict on an array
----------------------------------------

.. ipython:: python

    from geowombat.ml import fit_predict

    with gw.config.update(ref_res=100):
        with gw.open(l8_224078_20200518, chunks=128) as src:
            y = fit_predict(src, labels, pl, col='lc')
            print(y)

Fit a classifier with multiple dates
------------------------------------

.. ipython:: python

    with gw.config.update(ref_res=100):
        with gw.open([l8_224078_20200518, l8_224078_20200518], time_names=['t1', 't2'], stack_dim='time', chunks=128) as src:
            y = fit_predict(src, labels, pl, col='lc')
            print(y)

Train a classifier and predict 
------------------------------


.. ipython:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518, l8_224078_20200518_polygons
    from geowombat.ml import fit, predict
    from sklearn_xarray.preprocessing import Featurizer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.naive_bayes import GaussianNB

    le = LabelEncoder()

    # The labels are string names, so here we convert them to integers
    labels = gpd.read_file(l8_224078_20200518_polygons)
    labels["lc"] = le.fit(labels.name).transform(labels.name)
    print(labels)

    # Use a data pipeline
    pl = Pipeline(
        [
            ("featurizer", Featurizer()),
            ("scaler", StandardScaler()),
            ("pca", PCA()),
            ("clf", GaussianNB()),
        ]
    )

    # Fit the classifier
    with gw.config.update(ref_res=100):
        with gw.open(l8_224078_20200518, chunks=128) as src:
            X, clf = fit(src, labels, pl, col="lc")
            y = predict(X, clf)
            print(y)