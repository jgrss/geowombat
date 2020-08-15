from .. import polygon_to_array

import xarray as xr


class Classifiers(object):

    @staticmethod
    def fit(data, labels, col=None):

        """
        Fits a classifier given class labels

        Args:
            data (DataArray): The data to predict on.
            labels (str | Path | GeoDataFrame): Class labels as polygon geometry.
            col (Optional[str]): The column in ``labels`` you want to assign values from.
                If ``None``, creates a binary raster.

        Returns:
            ``xarray.DataArray``

        Example:
            >>> from geowombat.ml import fit
            >>>
            >>> with gw.open('image.tif') as src:
            >>>     X, clf = fit(src, labels)
            >>>     y = clf.predict(X).unstack('sample')
        """

        if not isinstance(labels, xr.DataArray):
            labels = polygon_to_array(labels, col=col, data=data)

        # TODO: is this sufficient?
        if data.gw.has_time_coord:

            labels = xr.concat([labels] * data.gw.ntime, dim='band')\
                        .assign_coords({'band': data.time.values.tolist()})

        # Mask 'no data'
        labels = labels.where(labels != 0)

        # TODO: what is the structure for single dates?
        if data.gw.has_time_coord:

            data.coords['land_use'] = (["time", "y", "x"], labels)

            # TODO: where are we importing Stackerizer from?
            X = Stackerizer(stack_dims=('y', 'x', 'time'),
                            direction='stack').fit_transform(data)

            # drop nans from
            Xna = X[~X.land_use.isnull()]

            # TODO: groupby as a user option?
            # Xgp = Xna.groupby('land_use').mean('sample')

        # TODO: need imports
        pl = Pipeline(
            [
                ("featurizer", Featurizer()),  # Stack all dimensions and variables except for sample dimension.
                ("scaler", wrap(StandardScaler)),  # zscores , ?wrap if xarray.self required?
                ("pca", wrap(PCA, reshapes="feature")),
                ("cls", wrap(GaussianNB, reshapes="feature")),
            ]
        )

        cv = CrossValidatorWrapper(
            GroupShuffleSplit(n_splits=1, test_size=0.5), groupby=["time"]
        )

        clf = GridSearchCV(
            pl, cv=cv, verbose=1, param_grid={"pca__n_components": [5]}
        )

        y = Target(
            coord="land_use", transform_func=LabelEncoder().fit_transform, dim="sample")(Xna)

        clf.fit(Xna, y)

        return X, clf

    def fit_predict(self, data, labels, col=None):

        """
        Fits a classifier given class labels and predicts on a DataArray

        Args:
            data (DataArray): The data to predict on.
            labels (str | Path | GeoDataFrame): Class labels as polygon geometry.
            col (Optional[str]): The column in ``labels`` you want to assign values from.
                If ``None``, creates a binary raster.

        Returns:
            ``xarray.DataArray``

        Example:
            >>> from geowombat.ml import fit_predict
            >>>
            >>> with gw.open('image.tif') as src:
            >>>     y = fit_predict(src, labels)
        """

        X, clf = self.fit(data, labels, col=col)

        return clf.predict(X).unstack('sample')
