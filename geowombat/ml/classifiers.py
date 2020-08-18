from .. import polygon_to_array
from .transformers import Stackerizer

import xarray as xr
from sklearn_xarray import Target
from sklearn_xarray.model_selection import CrossValidatorWrapper
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder


class ClassifiersMixin(object):

    @staticmethod
    def grid_search_cv(pipeline):

        # TODO: groupby arg
        cv = CrossValidatorWrapper(GroupShuffleSplit(n_splits=1,
                                                     test_size=0.5),
                                   groupby=['time'])

        # TODO: param_grid arg
        clf = GridSearchCV(pipeline,
                           cv=cv,
                           verbose=1,
                           param_grid={"pca__n_components": [5]})

        return clf


class Classifiers(ClassifiersMixin):

    def fit(self,
            data,
            labels,
            clf,
            grid_search=False,
            targ_name='targ',
            targ_dim_name='sample',
            col=None):

        """
        Fits a classifier given class labels

        Args:
            data (DataArray): The data to predict on.
            labels (str | Path | GeoDataFrame): Class labels as polygon geometry.
            clf (object): The classifier or classification pipeline.
            grid_search (Optional[bool]): Whether to use cross-validation.
            targ_name (Optional[str]): The target name.
            targ_dim_name (Optional[str]): The target coordinate name.
            col (Optional[str]): The column in ``labels`` you want to assign values from.
                If ``None``, creates a binary raster.

        Returns:
            ``xarray.DataArray``

        Example:
            >>> import geowombat as gw
            >>> from geowombat.ml import fit
            >>>
            >>> from sklearn_xarray import wrap
            >>> from sklearn_xarray.preprocessing import Featurizer
            >>> from sklearn.pipeline import Pipeline
            >>> from sklearn.preprocessing import StandardScaler
            >>> from sklearn.decomposition import PCA
            >>> from sklearn.naive_bayes import GaussianNB
            >>>
            >>> pl = Pipeline(
            >>>     [("featurizer", Featurizer()), 
            >>>      ("scaler", wrap(StandardScaler)),
            >>>      ("pca", wrap(PCA, reshapes="feature")),
            >>>      ("cls", wrap(GaussianNB, reshapes="feature"))])
            >>>
            >>> with gw.open('image.tif') as src:
            >>>     X, clf = fit(src, labels, pl, grid_search=True, col='id')
            >>>     y = clf.predict(X).unstack('sample')
            >>>
            >>> from sklearn.neural_network import MLPClassifier
            >>> wrapped = wrap(MLPClassifier())
            >>>
            >>> with gw.open('image.tif') as src:
            >>>     X, clf = fit(src, labels, wrapped, col='id')
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

            data.coords[targ_name] = (['time', 'y', 'x'], labels)

            # TODO: where are we importing Stackerizer from?
            X = Stackerizer(stack_dims=('y', 'x', 'time'),
                            direction='stack').fit_transform(data)

            # drop nans from
            Xna = X[~X[targ_name].isnull()]

            # TODO: groupby as a user option?
            # Xgp = Xna.groupby(targ_name).mean('sample')
      
        # TODO: what is the structure for single dates?

        else:
            # labels = xr.concat([labels], dim='band') 

            # # Mask 'no data'
            # labels = labels.where(labels != 0)
            
            # data.coords[targ_name] = (['y', 'x'], labels)

            # # TODO: where are we importing Stackerizer from?
            # X = Stackerizer(stack_dims=('y', 'x' ),
            #                 direction='stack').fit_transform(data)

            # # drop nans from
            # Xna = X[~X[targ_name].isnull()]

            # # TODO: groupby as a user option?
            # # Xgp = Xna.groupby(targ_name).mean('sample')



        if grid_search:
            clf = self.grid_search_cv(clf)

        # TODO: should we be using lazy=True?
        y = Target(coord=targ_name,
                   transform_func=LabelEncoder().fit_transform,
                   dim=targ_dim_name)(Xna)

        clf.fit(Xna, y)

        return X, clf

    def fit_predict(self,
                    data,
                    labels,
                    clf,
                    grid_search=False,
                    targ_name='targ',
                    targ_dim_name='sample',
                    col=None):

        """
        Fits a classifier given class labels and predicts on a DataArray

        Args:
            data (DataArray): The data to predict on.
            labels (str | Path | GeoDataFrame): Class labels as polygon geometry.
            clf (object): The classifier or classification pipeline.
            grid_search (Optional[bool]): Whether to use cross-validation.
            targ_name (Optional[str]): The target name.
            targ_dim_name (Optional[str]): The target coordinate name.
            col (Optional[str]): The column in ``labels`` you want to assign values from.
                If ``None``, creates a binary raster.

        Returns:
            ``xarray.DataArray``

        Example:
            >>> import geowombat as gw
            >>> from geowombat.ml import fit_predict
            >>>
            >>> from sklearn_xarray import wrap
            >>> from sklearn.neural_network import MLPClassifier
            >>> wrapped = wrap(MLPClassifier())
            >>>
            >>> with gw.open('image.tif') as src:
            >>>     y = fit_predict(src, labels, wrapped)
        """

        X, clf = self.fit(data,
                          labels,
                          clf,
                          grid_search=grid_search,
                          targ_name=targ_name,
                          targ_dim_name=targ_dim_name,
                          col=col)

        return clf.predict(X).unstack(targ_dim_name)
