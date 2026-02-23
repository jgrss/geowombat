.. _ml:

Machine learning
================

GeoWombat's ML module works with any scikit-learn compatible classifier or pipeline. Pass a classifier
to :func:`~geowombat.ml.fit`, :func:`~geowombat.ml.predict`, or :func:`~geowombat.ml.fit_predict`
and it will be applied to the raster data as an xarray DataArray.

To install ML dependencies::

    pip install "geowombat[ml]"

Recommended classifiers for remote sensing
------------------------------------------

Supervised
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Classifier
     - Module
     - Notes
   * - Random Forest
     - ``sklearn.ensemble.RandomForestClassifier``
     - Most widely used in remote sensing; handles high-dimensional data well, robust to noise
   * - LightGBM
     - ``lightgbm.LGBMClassifier``
     - Fast gradient boosting; strong accuracy with large datasets, supports categorical features
   * - Gradient Boosted Trees
     - ``sklearn.ensemble.GradientBoostingClassifier``
     - Strong accuracy; slower than LightGBM on large datasets
   * - Support Vector Machine
     - ``sklearn.svm.SVC``
     - Effective in high-dimensional spaces; works well with small training sets
   * - Gaussian Naive Bayes
     - ``sklearn.naive_bayes.GaussianNB``
     - Fast and simple baseline; assumes feature independence
   * - k-Nearest Neighbors
     - ``sklearn.neighbors.KNeighborsClassifier``
     - Non-parametric; useful for complex class boundaries

Unsupervised
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Classifier
     - Module
     - Notes
   * - K-Means
     - ``sklearn.cluster.KMeans``
     - Standard clustering; fast, works well for spectrally distinct classes
   * - Mini-Batch K-Means
     - ``sklearn.cluster.MiniBatchKMeans``
     - Faster K-Means variant for large rasters
   * - Gaussian Mixture Model
     - ``sklearn.mixture.GaussianMixture``
     - Soft clustering; models class overlap better than K-Means

Fit a classifier
----------------

.. ipython:: python

    import geowombat as gw
    from geowombat.data import l8_224078_20200518, l8_224078_20200518_polygons, l8_224078_20200518_points
    from geowombat.ml import fit, predict, fit_predict

    import geopandas as gpd
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.naive_bayes import GaussianNB
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    le = LabelEncoder()

    # The labels are string names, so here we convert them to integers
    labels = gpd.read_file(l8_224078_20200518_polygons)
    labels['lc'] = le.fit(labels.name).transform(labels.name)

    # Point labels (used for time-stacked examples below)
    labels_pts = gpd.read_file(l8_224078_20200518_points)
    labels_pts['lc'] = le.fit(labels_pts.name).transform(labels_pts.name)
    labels_pts = labels_pts.drop(columns=['name'])

    # Use a data pipeline
    pl = Pipeline([('scaler', StandardScaler()),
                    ('pca', PCA()),
                    ('clf', GaussianNB())])

    # Fit the classifier
    with gw.config.update(ref_res=100):
        with gw.open(l8_224078_20200518, nodata=0, chunks=128) as src:
            X, Xy, clf = fit(src, pl, labels, col='lc')

    print(clf)

Fit a classifier and predict on an array
----------------------------------------

.. ipython:: python

    from geowombat.ml import fit_predict
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(dpi=200)

    with gw.config.update(ref_res=100):
        with gw.open(l8_224078_20200518, nodata=0) as src:
            y = fit_predict(src, pl, labels, col='lc')
            print(y)

    @savefig ml_fit_predict.png
    y.plot(robust=True, ax=ax)

Train a supervised classifier and predict
-----------------------------------------

.. ipython:: python

    fig, ax = plt.subplots(dpi=200,figsize=(5,5))

    # Fit the classifier
    with gw.config.update(ref_res=100):
        with gw.open(l8_224078_20200518, nodata=0, chunks=128) as src:
            X, Xy, clf = fit(src, pl, labels, col="lc")
            y = predict(src, X, clf)
            y.plot(robust=True, ax=ax)
    @savefig ml_supervised_predict.png
    plt.tight_layout(pad=1)

Train an unsupervised classifier and predict
--------------------------------------------

Unsupervised classifiers can also be used in a pipeline

.. ipython:: python

    cl = Pipeline([ ('scaler', StandardScaler()),
                    ('pca', PCA()),
                    ('clf', KMeans(n_clusters=3, random_state=0))])

    fig, ax = plt.subplots(dpi=200,figsize=(5,5))

    # fit and predict unsupervised classifier
    with gw.config.update(ref_res=300):
        with gw.open(l8_224078_20200518, nodata=0) as src:
            X, Xy, clf = fit(src, cl)
            y = predict(src, X, clf)
            y.plot(robust=True, ax=ax)
    @savefig ml_unsupervised_predict.png
    plt.tight_layout(pad=1)

    fig, ax = plt.subplots(dpi=200,figsize=(5,5))

    # Fit_predict unsupervised classifier
    with gw.config.update(ref_res=300):
        with gw.open(l8_224078_20200518, nodata=0) as src:
            y = fit_predict(src, cl)
            y.plot(robust=True, ax=ax)
    @savefig ml_unsupervised_fit_predict.png
    plt.tight_layout(pad=1)

Predict with cross validation and parameter tuning
--------------------------------------------------

Cross-validation and parameter tuning is now possible

.. ipython:: python

    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn_xarray.model_selection import CrossValidatorWrapper

    cv = CrossValidatorWrapper(KFold())
    gridsearch = GridSearchCV(
        pl,
        cv=cv,
        scoring='balanced_accuracy',
        param_grid={"pca__n_components": [1, 2, 3]}
    )

    fig, ax = plt.subplots(dpi=200,figsize=(5,5))

    with gw.config.update(ref_res=300):
        with gw.open(l8_224078_20200518, nodata=0) as src:
            # fit a model to get Xy used to train model
            X, Xy, clf = fit(src, pl, labels, col="lc")

            # fit cross valiation and parameter tuning
            # NOTE: must unpack * object Xy
            gridsearch.fit(*Xy)
            print(gridsearch.best_params_)
            print(gridsearch.best_score_)

            # get set tuned parameters
            # Note: predict(gridsearch.best_model_) not currently supported
            clf.set_params(**gridsearch.best_params_)
            y = predict(src, X, clf)
            y.plot(robust=True, ax=ax)
    @savefig ml_cv_predict.png
    plt.tight_layout(pad=1)

Time-stacked classification with ``temporal_mode``
---------------------------------------------------

When opening multiple images with ``stack_dim='time'``, the data has shape
``(time, band, y, x)``. This is the format returned by STAC queries and
multi-date ``gw.open()`` calls. The ``temporal_mode`` parameter controls how
time is handled during classification:

- ``'panel'`` (default) — each pixel-time is an independent sample with B
  spectral features. Output retains the time dimension with one prediction
  per time step.
- ``'flatten'`` — all time steps are flattened into the band dimension,
  creating T×B features per pixel. Output has no time dimension.

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 30

   * - Input
     - ``temporal_mode``
     - Features
     - Output shape
   * - ``(time=T, band=B, y, x)``
     - ``'panel'``
     - B
     - ``(time=T, band='targ', y, x)``
   * - ``(time=T, band=B, y, x)``
     - ``'flatten'``
     - T × B
     - ``(band='targ', y, x)``

Panel mode
~~~~~~~~~~

Each pixel-time combination is treated as an independent sample. The output
retains the time dimension, giving one prediction map per time step.

.. ipython:: python

    import numpy as np

    pl_gmm = Pipeline([('scaler', StandardScaler()),
                        ('clf', GaussianMixture(n_components=4, random_state=0))])

    fig, axes = plt.subplots(1, 2, dpi=200, figsize=(10, 5))

    with gw.config.update(ref_res=300):
        with gw.open(
            [l8_224078_20200518, l8_224078_20200518],
            stack_dim='time',
            nodata=0,
        ) as src:
            y = fit_predict(
                data=src, clf=pl_gmm,
                temporal_mode='panel',
            )
            print(f"Dims:  {y.dims}")
            print(f"Shape: {y.shape}")

            y.isel(time=0).plot(robust=True, ax=axes[0])
            axes[0].set_title('Panel - time 0')
            y.isel(time=1).plot(robust=True, ax=axes[1])
            axes[1].set_title('Panel - time 1')

    @savefig ml_panel.png
    plt.tight_layout(pad=1)

Since both time steps use the same input image, the predictions are identical:

.. ipython:: python

    print("Time steps match:",
          np.allclose(y.isel(time=0).values, y.isel(time=1).values, equal_nan=True))


Flatten mode
~~~~~~~~~~~~

All time steps are flattened into the band dimension, creating T×B features
per pixel. This produces a single prediction map regardless of how many time
steps exist.

.. ipython:: python

    # Use PCA(n_components=1) since flattening doubles the feature count
    pl_gmm_flat = Pipeline([('scaler', StandardScaler()),
                            ('pca', PCA(n_components=1)),
                            ('clf', GaussianMixture(n_components=4, random_state=0))])

    fig, ax = plt.subplots(dpi=200, figsize=(5, 5))

    with gw.config.update(ref_res=300):
        with gw.open(
            [l8_224078_20200518, l8_224078_20200518],
            stack_dim='time',
            nodata=0,
        ) as src:
            y = fit_predict(
                data=src, clf=pl_gmm_flat,
                temporal_mode='flatten',
            )
            print(f"Dims:  {y.dims}")
            print(f"Shape: {y.shape}")
            y.plot(robust=True, ax=ax)

    @savefig ml_flatten.png
    plt.tight_layout(pad=1)


Supervised classification with time-stacked data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Supervised classifiers also work with time-stacked data. Pass training labels
just as you would for a single image.

.. ipython:: python

    pl_time = Pipeline([('scaler', StandardScaler()),
                        ('pca', PCA()),
                        ('clf', GaussianNB())])

    fig, axes = plt.subplots(1, 2, dpi=200, figsize=(10, 5))

    with gw.config.update(ref_res=300):
        with gw.open(
            [l8_224078_20200518, l8_224078_20200518],
            stack_dim='time',
            nodata=0,
        ) as src:
            y = fit_predict(
                src, pl_time, labels_pts, col='lc',
                temporal_mode='panel',
            )
            print(f"Dims:  {y.dims}")

            y.isel(time=0).plot(robust=True, ax=axes[0])
            axes[0].set_title('Supervised - time 0')
            y.isel(time=1).plot(robust=True, ax=axes[1])
            axes[1].set_title('Supervised - time 1')

    @savefig ml_supervised_time.png
    plt.tight_layout(pad=1)


Save prediction output
----------------------

.. code:: python

    y.gw.save('output.tif', overwrite=True)
