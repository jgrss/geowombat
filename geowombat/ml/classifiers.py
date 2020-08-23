import functools

from .. import polygon_to_array
from .transformers import Stackerizer

import xarray as xr
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn_xarray import wrap, Target
from sklearn_xarray.model_selection import CrossValidatorWrapper


def wrapped_cls(cls):

    @functools.wraps(cls)
    def wrapper(self):

        if self.__module__.split('.')[0] != 'sklearn_xarray':
            self = wrap(self, reshapes='feature')

        return self

    return wrapper


@wrapped_cls
class WrappedClassifier(object):
    pass


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

    @staticmethod
    def _prepare_labels(data, labels, col, targ_name):

        if not isinstance(labels, xr.DataArray):
            labels = polygon_to_array(labels, col=col, data=data)

        # TODO: is this sufficient for single dates?
        if not data.gw.has_time_coord:
            data = data.assign_coords({'time': [1]})

        labels = xr.concat([labels] * data.gw.ntime, dim='band')\
                    .assign_coords({'band': data.time.values.tolist()})

        # Mask 'no data'
        labels = labels.where(labels != 0)

        data.coords[targ_name] = (['time', 'y', 'x'], labels)

        return data

    @staticmethod
    def _prepare_predictors(data, targ_name):

        # TODO: where are we importing Stackerizer from?
        X = Stackerizer(stack_dims=('y', 'x', 'time'),
                        direction='stack').fit_transform(data)

        # drop nans
        Xna = X[~X[targ_name].isnull()]

        # TODO: groupby as a user option?
        # Xgp = Xna.groupby(targ_name).mean('sample')

        return X, Xna

    @staticmethod
    def _prepare_classifiers(clf):

        if isinstance(clf, Pipeline):
            clf = Pipeline([(clf_name, WrappedClassifier(clf_)) for clf_name, clf_ in clf.steps])
        else:
            clf = WrappedClassifier(clf)

        return clf

    @staticmethod
    def add_categorical(data, labels, col, variable_name='cat1'):

        """
        Writes xarray bands to disk by band

        Args:

            data (xarray.DataArray)
            labels (Path or GeoDataFrame): The labels with categorical data.
            col (Optional[str]): The column in ``labels`` you want to assign values from.
                If ``None``, creates a binary raster.
            variable_name (Optional[str]): The name assigned to the categorical data.

        Example:
            >>> from geowombat.ml.classifiers import Classifiers
            >>>
            >>> gwclf = Classifiers()
            >>>
            >>> climatecluster = ' ./ClusterEco15_Y5.shp'
            >>>
            >>> time_names = [str(x) for x in range(len(vrts))]
            >>>
            >>> with gw.open(vrts, time_names=time_names) as src:
            >>>     src.attrs['filename'] = vrts
            >>>     cats = gwclf.add_categorical(src, climatecluster, col='ClusterN_2', variable_name='clim_clust')
        """

        if not isinstance(labels, xr.DataArray):
            labels = polygon_to_array(labels, col=col, data=data)
            labels['band'] = [variable_name]

        # TODO: is this sufficient for single dates?
        if not data.gw.has_time_coord:
            data = data.assign_coords({'time': [1]})

        labels = xr.concat([labels] * data.gw.ntime, dim='time')\
                    .assign_coords({'time': data.time.values.tolist()})

        data = xr.concat([data, labels], dim='band')

        return data

    
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
            >>> from sklearn_xarray.preprocessing import Featurizer
            >>> from sklearn.pipeline import Pipeline
            >>> from sklearn.preprocessing import StandardScaler
            >>> from sklearn.decomposition import PCA
            >>> from sklearn.naive_bayes import GaussianNB
            >>>
            >>> # Use a data pipeline
            >>> pl = Pipeline(
            >>>     [("featurizer", Featurizer()),
            >>>      ("scaler", StandardScaler()),
            >>>      ("pca", PCA()),
            >>>      ("cls", GaussianNB()))])
            >>>
            >>> with gw.open('image.tif') as src:
            >>>     X, clf = fit(src, labels, pl, grid_search=True, col='id')
            >>>     y = clf.predict(X).unstack('sample')
            >>>
            >>> # Use a single classifier
            >>> from sklearn.neural_network import MLPClassifier
            >>> mlp = MLPClassifier()
            >>>
            >>> with gw.open('image.tif') as src:
            >>>     X, clf = fit(src, labels, mlp, col='id')
            >>>     y = clf.predict(X).unstack('sample')
        """

        data = self._prepare_labels(data, labels, col, targ_name)
        X, Xna = self._prepare_predictors(data, targ_name)
        clf = self._prepare_classifiers(clf)

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

        return clf.predict(X)\
                    .unstack(targ_dim_name)\
                    .assign_coords(coords={'band': targ_name})\
                    .expand_dims(dim='band')\
                    .transpose('time', 'band', 'y', 'x')
