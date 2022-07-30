import functools
import logging

from .. import polygon_to_array
from .transformers import Stackerizer

import xarray as xr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn_xarray import wrap, Target
from sklearn_xarray.preprocessing import Featurizer
import numpy as np
from geopandas.geodataframe import GeoDataFrame

from sklearn.utils.validation import check_X_y, check_is_fitted, check_array

logger = logging.getLogger(__name__)


def wrapped_cls(cls):
    @functools.wraps(cls)
    def wrapper(self):

        if self.__module__.split(".")[0] != "sklearn_xarray":
            self = wrap(self, reshapes="feature")

        return self

    return wrapper


@wrapped_cls
class WrappedClassifier(object):
    pass


class ClassifiersMixin(object):
    @staticmethod
    def _add_time_dim(data):
        return (
            data.assign_coords(coords={"time": "t1"})
            .expand_dims(dim="time")
            .transpose("time", "band", "y", "x")
        )

    # @staticmethod
    def _prepare_labels(self, data, labels, col, targ_name):

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
        if not data.gw.has_time_coord:
            data = self._add_time_dim(data)

        labels = xr.concat([labels] * data.gw.ntime, dim="band").assign_coords(
            coords={"band": data.time.values.tolist()}
        )

        # Mask 'no data' outside training data
        labels = labels.where(labels != 0)

        data.coords[targ_name] = (["time", "y", "x"], labels.data)

        return data

    @staticmethod
    def _stack_it(data):
        return Stackerizer(
            stack_dims=("y", "x", "time"), direction="stack"
        ).fit_transform(data)

    # @staticmethod
    def _prepare_predictors(self, data, targ_name):

        X = self._stack_it(data)

        # drop nans
        Xna = X[~X[targ_name].isnull()]

        # TODO: groupby as a user option?
        # Xgp = Xna.groupby(targ_name).mean('sample')

        return X, Xna

    @staticmethod
    def _prepare_classifiers(clf):

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

        return clf

    @staticmethod
    def add_categorical(data, labels, col, variable_name="cat1"):

        """
        Adds numeric categorical data to array based on polygon col values.
        For multiple time periods, multiple copies are made, one for each time period.

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
            labels["band"] = [variable_name]

        # TODO: is this sufficient for single dates?
        if not data.gw.has_time_coord:

            data = (
                data.assign_coords(coords={"time": "t1"})
                .expand_dims(dim="time")
                .transpose("time", "band", "y", "x")
            )

        labels = xr.concat([labels] * data.gw.ntime, dim="time").assign_coords(
            {"time": data.time.values.tolist()}
        )

        data = xr.concat([data, labels], dim="band")

        return data

    @staticmethod
    def _mask_nodata(y, x, src_nodata=None, dst_nodata=np.nan):
        """
        Remove missing data value and replace with another.

        Args:
            y (DataArray): The prediction
            x (DataArray): The data to used to predict.
            src_nodata (int,float): Value to replace , default is x.attrs["nodatavals"][0]
            dst_nodata (int,float): Replacement value, default is np.nan - but converts y to float

        """
        if src_nodata is None:
            src_nodata = x.attrs["nodatavals"][0]

        if len(x.shape) == 3:
            mask = np.any((x == src_nodata).values, 0)
            return xr.where(mask, dst_nodata, y)
        else:
            mask = np.any((x == src_nodata).values, 1, keepdims=True)
            return xr.where(mask, dst_nodata, y)


class Classifiers(ClassifiersMixin):
    def fit(
        self,
        data,
        clf,
        labels=None,
        col=None,
        targ_name="targ",
        targ_dim_name="sample",
    ):
        """Fits a classifier given class labels

        Args:
            data (DataArray): The data to predict on.
            clf (object): The classifier or classification pipeline.
            labels (Optional[str | Path | GeoDataFrame]): Class labels as polygon geometry.
            col (Optional[str]): The column in ``labels`` you want to assign values from.
                If ``None``, creates a binary raster.
            targ_name (Optional[str]): The target name.
            targ_dim_name (Optional[str]): The target coordinate name.

        Returns:
            X (xarray.DataArray): Original DataArray augmented to accept prediction dimension
            Xna if unsupervised classifier: tuple(xarray.DataArray, sklearn_xarray.Target): X:Reshaped feature data without NAs removed, y:None
            Xna if supervised classifier: tuple(xarray.DataArray, sklearn_xarray.Target): X:Reshaped feature data with NAs removed, y:Array holding target data
            clf, (sklearn pipeline): Fitted pipeline object

        Example:
            >>> import geowombat as gw
            >>> from geowombat.data import l8_224078_20200518, l8_224078_20200518_polygons
            >>> from geowombat.ml import fit
            >>>
            >>> import geopandas as gpd
            >>> from sklearn_xarray.preprocessing import Featurizer
            >>> from sklearn.pipeline import Pipeline
            >>> from sklearn.preprocessing import StandardScaler, LabelEncoder
            >>> from sklearn.decomposition import PCA
            >>> from sklearn.naive_bayes import GaussianNB
            >>>
            >>> le = LabelEncoder()
            >>>
            >>> labels = gpd.read_file(l8_224078_20200518_polygons)
            >>> labels['lc'] = le.fit(labels.name).transform(labels.name)
            >>>
            >>> # Use supervised classification pipeline
            >>> pl = Pipeline([('scaler', StandardScaler()),
            >>>                ('pca', PCA()),
            >>>                ('clf', GaussianNB())])
            >>>
            >>> with gw.open(l8_224078_20200518) as src:
            >>>   X, Xy, clf = fit(src, pl, labels, col='lc')

            >>> # Fit an unsupervised classifier
            >>> cl = Pipeline([('pca', PCA()),
            >>>                ('cst', KMeans()))])
            >>> with gw.open(l8_224078_20200518) as src:
            >>>    X, Xy, clf = fit(src, cl)
        """
        if clf._estimator_type == "clusterer":
            data = self._add_time_dim(data)
            X = self._stack_it(data)
            clf = self._prepare_classifiers(clf)

            # TODO: Validation checks
            # check_array(X)
            clf.fit(X)

            return X, (X, None), clf

        else:

            data = self._prepare_labels(data, labels, col, targ_name)
            X, Xna = self._prepare_predictors(data, targ_name)
            clf = self._prepare_classifiers(clf)

            # TODO: should we be using lazy=True?
            y = Target(
                coord=targ_name,
                transform_func=LabelEncoder().fit_transform,
                dim=targ_dim_name,
            )(Xna)

            # TO DO: Validation checks
            # Xna, y = check_X_y(Xna, y)

            clf.fit(Xna, y)

            return X, (Xna, y), clf

    def predict(
        self,
        data,
        X,
        clf,
        targ_name="targ",
        targ_dim_name="sample",
        mask_nodataval=True,
    ):
        """Fits a classifier given class labels and predicts on a DataArray

        Args:
            data (DataArray): The data to predict on.
            X (str | Path | DataArray): Data array generated by geowombat.ml.fit
            clf (object): The classifier or classification pipeline.
            targ_name (Optional[str]): The target name.
            targ_dim_name (Optional[str]): The target coordinate name.
            mask_nodataval (Optional[Bool]): If true, data.attrs["nodatavals"][0]
                are replaced with np.nan and the array is returned as type float


        Returns:
            ``xarray.DataArray``:

                Predictions shaped ('time' x 'band' x 'y' x 'x')

        Example:

            >>> import geowombat as gw
            >>> from geowombat.data import l8_224078_20200518, l8_224078_20200518_polygons
            >>> from geowombat.ml import fit, predict
            >>> import geopandas as gpd
            >>> from sklearn_xarray.preprocessing import Featurizer
            >>> from sklearn.pipeline import Pipeline
            >>> from sklearn.preprocessing import LabelEncoder, StandardScaler
            >>> from sklearn.decomposition import PCA
            >>> from sklearn.naive_bayes import GaussianNB

            >>> le = LabelEncoder()
            >>> labels = gpd.read_file(l8_224078_20200518_polygons)
            >>> labels["lc"] = le.fit(labels.name).transform(labels.name)

            >>> # Use a data pipeline
            >>> pl = Pipeline([('scaler', StandardScaler()),
            >>>                ('pca', PCA()),
            >>>                ('clf', GaussianNB()))])

            >>> # Fit and predict the classifier
            >>> with gw.config.update(ref_res=100):
            >>>     with gw.open(l8_224078_20200518, chunks=128) as src:
            >>>         X, Xy, clf = fit(src, pl, labels, col="lc")
            >>>         y = predict(src, X, clf)
            >>>         print(y)

            >>> # Fit and predict an unsupervised classifier
            >>> cl = Pipeline([('pca', PCA()),
            >>>                ('cst', KMeans()))])
            >>> with gw.open(l8_224078_20200518) as src:
            >>>    X, Xy, clf = fit(src, cl)
            >>>    y1 = predict(src, X, clf)
        """
        check_is_fitted(clf)

        # try:
        y = (
            clf.predict(X)
            .unstack(targ_dim_name)
            .assign_coords(coords={"band": targ_name})
            .expand_dims(dim="band")
            .transpose("time", "band", "y", "x")
        )

        # no point unit doesn't have nan
        if mask_nodataval:
            y = self._mask_nodata(y=y, x=data)

        return xr.concat([data, y], dim="band").sel(band=targ_name)

    def fit_predict(
        self,
        data,
        clf,
        labels=None,
        col=None,
        targ_name="targ",
        targ_dim_name="sample",
        mask_nodataval=True,
    ):
        """Fits a classifier given class labels and predicts on a DataArray

        Args:
            data (DataArray): The data to predict on.
            clf (object): The classifier or classification pipeline.
            labels (optional[str | Path | GeoDataFrame]): Class labels as polygon geometry.
            col (Optional[str]): The column in ``labels`` you want to assign values from.
                If ``None``, creates a binary raster.
            targ_name (Optional[str]): The target name.
            targ_dim_name (Optional[str]): The target coordinate name.
            mask_nodataval (Optional[Bool]): If true, data.attrs["nodatavals"][0]
                are replaced with np.nan and the array is returned as type float

        Returns:
            ``xarray.DataArray``:

                Predictions shaped ('time' x 'band' x 'y' x 'x')

        Example:
            >>> import geowombat as gw
            >>> from geowombat.data import l8_224078_20200518, l8_224078_20200518_polygons
            >>> from geowombat.ml import fit_predict
            >>>
            >>> import geopandas as gpd
            >>> from sklearn_xarray.preprocessing import Featurizer
            >>> from sklearn.pipeline import Pipeline
            >>> from sklearn.preprocessing import StandardScaler, LabelEncoder
            >>> from sklearn.decomposition import PCA
            >>> from sklearn.naive_bayes import GaussianNB
            >>> from sklearn.cluster import KMeans
            >>>
            >>> le = LabelEncoder()
            >>>
            >>> labels = gpd.read_file(l8_224078_20200518_polygons)
            >>> labels['lc'] = le.fit(labels.name).transform(labels.name)
            >>>
            >>> # Use a supervised classification pipeline
            >>> pl = Pipeline([('scaler', StandardScaler()),
            >>>                ('pca', PCA()),
            >>>                ('clf', GaussianNB()))])
            >>>
            >>> with gw.open(l8_224078_20200518) as src:
            >>>     y = fit_predict(src, pl, labels, col='lc')
            >>>     y.isel(time=0).sel(band='targ').gw.imshow()
            >>>
            >>> with gw.open([l8_224078_20200518,l8_224078_20200518]) as src:
            >>>     y = fit_predict(src, pl, labels, col='lc')
            >>>     y.isel(time=1).sel(band='targ').gw.imshow()
            >>>
            >>> # Use an unsupervised classification pipeline
            >>> cl = Pipeline([('pca', PCA()),
            >>>                ('cst', KMeans()))])
            >>> with gw.open(l8_224078_20200518) as src:
            >>>     y2 = fit_predict(src, cl)
        """

        X, Xy, clf = self.fit(
            data,
            clf,
            labels,
            col=col,
            targ_name=targ_name,
            targ_dim_name=targ_dim_name,
        )

        y = self.predict(data, X, clf, targ_name, targ_dim_name, mask_nodataval)

        return y
