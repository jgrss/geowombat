import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from geopandas.geodataframe import GeoDataFrame
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn_xarray import Target, wrap

from .. import polygon_to_array


def _is_unsupervised(estimator):
    """Check if an estimator is unsupervised (clusterer or density model).

    For Pipeline objects, checks the final estimator.
    Supports both old (_estimator_type) and new (__sklearn_tags__) sklearn APIs.
    Detects KMeans, MiniBatchKMeans, GaussianMixture, etc.
    """
    _UNSUPERVISED_TYPES = {'clusterer', 'DensityEstimator', 'density_estimator'}

    # Handle sklearn_xarray wrapped estimators
    if hasattr(estimator, 'estimator'):
        return _is_unsupervised(estimator.estimator)

    if isinstance(estimator, Pipeline):
        # Get the final estimator in the pipeline
        final_estimator = estimator.steps[-1][1]
        return _is_unsupervised(final_estimator)

    # Try new sklearn 1.6+ tags API first
    if hasattr(estimator, '__sklearn_tags__'):
        try:
            tags = estimator.__sklearn_tags__()
            # In sklearn 1.6+, tags has estimator_type attribute
            if hasattr(tags, 'estimator_type'):
                return tags.estimator_type in _UNSUPERVISED_TYPES
        except Exception:
            pass

    # Fall back to old _estimator_type attribute (sklearn < 1.6)
    estimator_type = getattr(estimator, '_estimator_type', None)
    if estimator_type is not None:
        return estimator_type in _UNSUPERVISED_TYPES

    # Last resort: check if it inherits from ClusterMixin
    try:
        from sklearn.cluster import ClusterMixin
        return isinstance(estimator, ClusterMixin)
    except ImportError:
        pass

    return False

def _wrap_classifier(clf):
    """Wrap a classifier with sklearn_xarray if not already wrapped."""
    if clf.__module__.split(".")[0] != "sklearn_xarray":
        clf = wrap(clf, reshapes="band")
    return clf


class ClassifiersMixin(object):

    def __init__(self):
        self.le = LabelEncoder()

    @staticmethod
    def _add_time_dim(data):
        if "time" not in data.dims:

            return (
                data.assign_coords(coords={"time": "t1"})
                .expand_dims(dim="time")
                .transpose("time", "band", "y", "x")
            )
        else:
            return data

    @property
    def classes_(self):
        return self.le.classes_ + 1

    def _fit_labels(self, labels: np.ndarray):
        self.le.fit(labels)

    def _transform_labels(self, labels: np.ndarray):
        return self.le.transform(labels) + 1

    def _prepare_labels(self, data, labels, col, targ_name):
        # Convert path to GeoDataFrame first
        if isinstance(labels, (str, Path)):
            labels = gpd.read_file(labels)

        # Now safe to access labels[col] and rasterize
        if isinstance(labels, GeoDataFrame):
            if (labels[col].dtype != int) or (labels[col].min() == 0):
                self._fit_labels(labels[col])
                labels[col] = self._transform_labels(labels[col])
                warnings.warn(
                    "target labels were not integers or min class is 0 (conflicts with missing), "
                    f"applying LabelEncoder and adding 1. Input classes = {','.join(self.le.classes_.astype(str).tolist())}; "
                    f"Transformed classes = {','.join(self._transform_labels(self.le.classes_).astype(str).tolist())}",
                    UserWarning,
                )
            labels = polygon_to_array(labels, col=col, data=data)

        labels = xr.concat([labels] * data.gw.ntime, dim="band").assign_coords(
            coords={"band": data.time.values.tolist()}
        )

        data.coords[targ_name] = (["time", "y", "x"], labels.values)

        return data, labels

    @staticmethod
    def _stack_it(data):
        return data.stack(sample=("x", "y", "time")).T

    @staticmethod
    def _clean_coords(data, targ_name="targ"):
        """Drop non-essential coordinates that break sklearn_xarray.

        STAC data (via stackstac) carries extra metadata coordinates
        (title, id, collection, etc.) as DataArray objects. These cause
        TypeError in sklearn_xarray when constructing xarray Variables.
        Keep only the dimension coordinates plus the target coordinate.
        """
        essential = set(data.dims) | {targ_name}
        drop = [c for c in data.coords if c not in essential]
        if drop:
            return data.drop_vars(drop)
        return data

    def _prepare_predictors(self, data, targ_name):

        data = self._clean_coords(data, targ_name)
        X = self._stack_it(data)

        # drop nans
        try:
            # prep target axis 
            Xna = X[~X[targ_name].isnull()]
            Xna = Xna[Xna[targ_name] != 0]  # filter zero targets (fill from polygon_to_array)
            # TODO: if X.gw.nodataval is not None:
            #     Xna = X[X!= X.gw.nodata ]  # changes here would have to be reflected in y as well
        except KeyError:
            Xna = X

        # TODO: groupby as a user option?
        # Xgp = Xna.groupby(targ_name).mean('sample')

        return X, Xna

    @staticmethod
    def _prepare_classifiers(clf):

        # ideally need to add wrap(clf, reshapes='band')
        if isinstance(clf, Pipeline):

            clf = Pipeline(
                [
                    (cln_name, wrap(cln_, reshapes="band"))
                    for cln_name, cln_ in clf.steps
                ]
            )

        else:
            clf = _wrap_classifier(clf)

        return clf

    @staticmethod
    def add_categorical(data, labels, col, variable_name="cat1"):

        """Adds numeric categorical data to array based on polygon col values.
        For multiple time periods, multiple copies are made, one for each time
        period.

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
    def _flatten_time(data):
        """Reshape (time, band, y, x) -> (time*band, y, x).

        Flattens all temporal bands into a single band dimension so
        each pixel is one sample with all spectral-temporal features.
        Dask-friendly (preserves lazy evaluation).
        """
        time_vals = data.time.values
        band_vals = data.band.values
        new_bands = [
            f"{t}_{b}" for t in time_vals for b in band_vals
        ]

        slices = [
            data.isel(time=i).drop_vars("time")
            for i in range(len(time_vals))
        ]
        result = xr.concat(slices, dim="band")
        result = result.assign_coords(band=new_bands)
        return result

    @staticmethod
    def _mask_nodata(y, x, src_nodata=None, dst_nodata=np.nan):
        """Remove missing data value and replace with another.

        Args:
            y (DataArray): The prediction
            x (DataArray): The data to used to predict.
            src_nodata (int,float): Value to replace , default is x.attrs["nodatavals"][0]
            dst_nodata (int,float): Replacement value, default is np.nan - but converts y to float
        """
        if src_nodata is None:
            nodatavals = x.attrs.get("nodatavals")
            if nodatavals is None or len(nodatavals) == 0:
                return y
            src_nodata = nodatavals[0]

        if len(x.shape) == 3:
            mask = np.any((x == src_nodata).values, 0)
            return xr.where(mask, dst_nodata, y)
        else:
            mask = np.any((x == src_nodata).values, 1, keepdims=True)
            return xr.where(mask, dst_nodata, y)

        # suggested by jordan
        # Set the 'no data' attribute
        # .gw.assign_nodata_attrs(0)
        # Convert 'no data' values to nans
        # .gw.mask_nodata()


class Classifiers(ClassifiersMixin):
    def fit(
        self,
        data,
        clf,
        labels=None,
        col=None,
        targ_name="targ",
        targ_dim_name="sample",
        temporal_mode="panel",
    ):
        """Fits a classifier given class labels.

        Args:
            data (DataArray): The data to predict on.
            clf (object): The classifier or classification pipeline.
            labels (Optional[str | Path | GeoDataFrame]): Class labels as polygon geometry.
            col (Optional[str]): The column in ``labels`` you want to assign values from.
                If ``None``, creates a binary raster.
            targ_name (Optional[str]): The target name.
            targ_dim_name (Optional[str]): The target coordinate name.
            temporal_mode (Optional[str]): How to handle time-dimensioned data.
                'panel' — each pixel-time is an independent sample (B features).
                    Output has time dimension with per-time predictions.
                'flatten' — flatten time into band (T*B features per pixel).
                    Output has no time dimension, one prediction per pixel.
                Ignored when data has no time dimension.

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
            >>>                ('cst', KMeans())])
            >>> with gw.open(l8_224078_20200518) as src:
            >>>    X, Xy, clf = fit(src, cl)
        """
        # Flatten time into band if requested
        if data.gw.has_time_coord and temporal_mode == 'flatten':
            data = self._flatten_time(data)

        if _is_unsupervised(clf):
            data = self._add_time_dim(data)
            X, Xna = self._prepare_predictors(data, targ_name)

            clf = self._prepare_classifiers(clf)

            # TODO: Validation checks
            # check_array(X)
            clf.fit(X)
            setattr(clf, "fitted_", True)

            return X, (X, None), clf

        else:
            data = self._add_time_dim(data)

            data, labels = self._prepare_labels(data, labels, col, targ_name)
            X, Xna = self._prepare_predictors(data, targ_name)
            clf = self._prepare_classifiers(clf)

            # TODO: should we be using lazy=True?
            y = Target(
                coord=targ_name,
            )(Xna)

            # TO DO: Validation checks
            # Xna, y = check_X_y(Xna, y)

            clf.fit(Xna, y)
            setattr(clf, "fitted_", True)

            return X, (Xna, y), clf

    def predict(
        self,
        data,
        X,
        clf,
        targ_name="targ",
        targ_dim_name="sample",
        mask_nodataval=True,
        temporal_mode="panel",
    ):
        """Predicts on a DataArray using a fitted classifier.

        Args:
            data (DataArray): The data to predict on.
            X (DataArray): Data array generated by geowombat.ml.fit.
            clf (object): The classifier or classification pipeline.
            targ_name (Optional[str]): The target name.
            targ_dim_name (Optional[str]): The target coordinate name.
            mask_nodataval (Optional[Bool]): If true, data.attrs["nodatavals"][0]
                are replaced with np.nan and the array is returned as type float
            temporal_mode (Optional[str]): How to handle time-dimensioned data.
                'panel' — each pixel-time is an independent sample.
                'flatten' — flatten time into band.
                Must match the temporal_mode used in fit().

        Returns:
            ``xarray.DataArray``:

                Predictions shaped ('time' x 'band' x 'y' x 'x')

        Example:

            >>> import geowombat as gw
            >>> from geowombat.data import l8_224078_20200518, l8_224078_20200518_polygons
            >>> from geowombat.ml import fit, predict
            >>> import geopandas as gpd
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
            >>>                ('clf', GaussianNB())])

            >>> # Fit and predict the classifier
            >>> with gw.config.update(ref_res=100):
            >>>     with gw.open(l8_224078_20200518, nodata=0) as src:
            >>>         X, Xy, clf = fit(src, pl, labels, col="lc")
            >>>         y = predict(src, X, clf)
            >>>         print(y)

            >>> # Fit and predict an unsupervised classifier
            >>> cl = Pipeline([('pca', PCA()),
            >>>                ('cst', KMeans())])
            >>> with gw.open(l8_224078_20200518) as src:
            >>>    X, Xy, clf = fit(src, cl)
            >>>    y1 = predict(src, X, clf)
        """
        # Flatten time into band if requested (for mask_nodata)
        if data.gw.has_time_coord and temporal_mode == 'flatten':
            data = self._flatten_time(data)

        check_is_fitted(clf)

        y = (
            clf.predict(X)
            .unstack(targ_dim_name)
            .assign_coords(coords={"band": targ_name})
            .expand_dims(dim="band")
            .transpose("time", "band", "y", "x")
        )

        if mask_nodataval:
            y = self._mask_nodata(y=y, x=data)

        if y.gw.ntime == 1:
            y = y.isel(time=0).drop_vars("time")

        # Convert to dask array
        chunks = {
            "band": -1,
            "y": data.gw.row_chunks,
            "x": data.gw.col_chunks,
        }
        if "time" in y.dims:
            chunks["time"] = -1

        y = y.chunk(chunks).assign_attrs(**data.attrs)

        return y

    def fit_predict(
        self,
        data,
        clf,
        labels=None,
        col=None,
        targ_name="targ",
        targ_dim_name="sample",
        mask_nodataval=True,
        temporal_mode="panel",
    ):
        """Fits a classifier given class labels and predicts on a DataArray.

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
            temporal_mode (Optional[str]): How to handle time-dimensioned data.
                'panel' — each pixel-time is an independent sample.
                'flatten' — flatten time into band.

        Returns:
            ``xarray.DataArray``:

                Predictions shaped ('time' x 'band' x 'y' x 'x')

        Example:
            >>> import geowombat as gw
            >>> from geowombat.data import l8_224078_20200518, l8_224078_20200518_polygons
            >>> from geowombat.ml import fit_predict
            >>>
            >>> import geopandas as gpd
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
            >>>                ('clf', GaussianNB())])
            >>>
            >>> with gw.open(l8_224078_20200518, nodata=0) as src:
            >>>     y = fit_predict(src, pl, labels, col='lc')
            >>>     y.sel(band='targ').gw.imshow()
            >>>
            >>> # Use an unsupervised classification pipeline
            >>> cl = Pipeline([('pca', PCA()),
            >>>                ('cst', KMeans())])
            >>> with gw.open(l8_224078_20200518, nodata=0) as src:
            >>>     y2 = fit_predict(src, cl)
        """
        # Flatten time into band once if requested,
        # so both fit() and predict() see the same data
        if data.gw.has_time_coord and temporal_mode == 'flatten':
            data = self._flatten_time(data)

        X, Xy, clf = self.fit(
            data,
            clf,
            labels=labels,
            col=col,
            targ_name=targ_name,
            targ_dim_name=targ_dim_name,
            temporal_mode=temporal_mode,
        )

        y = self.predict(
            data, X, clf, targ_name, targ_dim_name,
            mask_nodataval, temporal_mode,
        )

        return y
