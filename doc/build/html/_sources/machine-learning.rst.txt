.. _machine_learning:

Use GeoWombat for a machine learning pipeline
=============================================

Use a Scikit-learn classifier
+++++++++++++++++++++++++++++

.. code:: python

    import geowombat as gw
    from sklearn import ensemble

    # Fit a Scikit-learn classifier
    clf = ensemble.RandomForestClassifier()
    clf.fit(X, y)

    # Apply the classifier to an image
    with gw.open('image.tif') as ds:
        pred = gw.predict(ds, clf, outname='predictions.tif')

Use GeoWombat and Dask to fit a classifier
++++++++++++++++++++++++++++++++++++++++++

.. code:: python

    import geowombat as gw
    from geowombat.models import GeoWombatClassifier

    # Fit a LightGBM classifier
    clf = GeoWombatClassifier(name='lightgbm')
    clf.fit(X, y)

    # Apply the classifier to an image
    with gw.open('image.tif') as ds:
        pred = gw.predict(ds, clf, outname='predictions.tif')