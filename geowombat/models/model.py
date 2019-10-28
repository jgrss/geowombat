import os
from copy import copy
import ctypes
import inspect
from contextlib import contextmanager

from ..errors import logger
from ..backends import Cluster
from ..core.util import Chunks

import numpy as np
import joblib
import six

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import ensemble
# from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.multiclass import unique_labels

import xarray as xr
import dask.array as da
from dask_ml.wrappers import ParallelPostFit

try:
    import lightgbm as lgb
    LIGHTGBM_INSTALLED = True
except:
    LIGHTGBM_INSTALLED = False

try:
    from imblearn import ensemble as imblearn
    IMBLEARN_INSTALLED = True
except:
    IMBLEARN_INSTALLED = False

try:
    MKL_LIB = ctypes.CDLL('libmkl_rt.so')
except:
    MKL_LIB = None


@contextmanager
def _backend_dummy(**kwargs):
    yield None


class VotingClassifier(BaseEstimator, ClassifierMixin):

    """
    A voting classifier class to use prefit models instead of re-fitting

    Args:
        estimators (list of tuples): The fitted estimators.
        weights (Optional[list, 1d array-like): The estimator weights.
        y (1d array-like)
        classes (1d array-like)
    """

    def __init__(self, estimators, weights=None, y=None, classes=None):

        self.estimators = estimators
        self.weights = weights
        self.is_prefit_model = True
        self.y_ = y
        self.classes_ = None

        if isinstance(y, np.ndarray) or isinstance(y, list):
            self.classes_ = unique_labels(y)
        elif isinstance(classes, np.ndarray) or isinstance(classes, list):
            self.classes_ = classes

        if isinstance(self.weights, list):
            self.weights = np.array(self.weights, dtype='float32')

        if self.weights is None:
            self.weights = np.ones(len(self.estimators), dtype='float32')

        if len(self.weights) != len(self.estimators):
            logger.exception('  The length of the weights must match the length of the estimators.')

        if isinstance(self.classes_, np.ndarray) or isinstance(self.classes_, list):
            self.n_classes_ = len(self.classes_)
        else:
            self.n_classes_ = 0

    def predict(self, X):

        """
        Predicts discrete classes by soft probability averaging

        Args:
            X (2d array): The predictive variables.
        """

        # Get predictions as an index of the array position.
        probabilities_argmax = np.argmax(self.predict_proba(X), axis=1)

        predictions = np.zeros(probabilities_argmax.shape, dtype='int16')

        # Convert indices to classes.
        for class_index, real_class in enumerate(self.classes_):
            predictions[probabilities_argmax == class_index] = real_class

        return predictions

    def predict_proba(self, X):

        """
        Predicts class posterior probabilities by soft probability averaging

        Args:
            X (2d array): The predictive variables.
        """

        clf = self.estimators[0][1]

        X_probas = clf.predict_proba(X) * self.weights[0]

        for clf_idx in range(1, len(self.estimators)):

            clf = self.estimators[clf_idx][1]
            X_probas += clf.predict_proba(X) * self.weights[clf_idx]

        return X_probas / self.weights.sum()


class GeoWombatClassifier(object):

    """
    A class for model fitting models with Dask

    Args:
        name (str or list): A model or list of models to use. Choices are ['lightgbm', 'extra trees',
            'random forest', 'bal random forest', 'bag bagging'].
        lightgbm_kwargs (dict): Keyword arguments for LightGBM.
        extra_trees_kwargs (dict): Keyword arguments for Extra Trees.
        random_forest_kwargs (dict): Keyword arguments for Random Forest.
        bal_random_forest_kwargs (dict): Keyword arguments for Balanced Random Forest.
        bal_bagging_kwargs (dict): Keyword arguments for Balanced Bagging.
        n_jobs (Optional[int]): The number of parallel processes.

    Examples:
        >>> ml = GeoWombatClassifier(['lightgbm', 'random forest'], n_jobs=8)
        >>>
        >>> # Fit a model
        >>> ml.fit(X, y)
        >>>
        >>> # Save the model to file
        >>> ml.to_file('clf.model')
        >>>
        >>> # Load a model from file
        >>> ml.from_file('clf.model')

    Requires:
        pip install --upgrade dask[dataframe] dask-ml
    """

    def __init__(self,
                 name=None,
                 lightgbm_kwargs=None,
                 extra_trees_kwargs=None,
                 random_forest_kwargs=None,
                 bal_random_forest_kwargs=None,
                 bal_bagging_kwargs=None,
                 verbose=0,
                 backend='none',
                 n_jobs=-1):

        if MKL_LIB:
            __ = MKL_LIB.MKL_Set_Num_Threads(n_jobs)

        self.backend = backend
        self.n_jobs = n_jobs

        self.name = name
        self.lightgbm_kwargs = lightgbm_kwargs
        self.extra_trees_kwargs = extra_trees_kwargs
        self.random_forest_kwargs = random_forest_kwargs
        self.bal_random_forest_kwargs = bal_random_forest_kwargs
        self.bal_bagging_kwargs = bal_bagging_kwargs
        self.verbose = verbose

        self.x = None
        self.y = None

        self.clf_dict = dict()
        self.model = None

        self._setup_models()

    @property
    def classes_(self):

        """
        Get the class list from the first estimator
        """

        if len(self.name_list) == 1:
            return self.model.estimator.classes_
        else:

            try:
                return self.model.estimator.estimators[0][1].classes_
            except:
                return self.model.estimators[0][1].classes_

    @property
    def class_count(self):
        """Get the class count"""
        return self.classes_.shape[0]

    def _setup_models(self):

        if not self.lightgbm_kwargs:

            self.lightgbm_kwargs = dict(boosting_type='goss',
                                        num_leaves=100,
                                        max_depth=50,
                                        learning_rate=0.01,
                                        n_estimators=200,
                                        objective='multiclass',
                                        class_weight='balanced',
                                        bagging_fraction=0.5,
                                        feature_fraction=0.5,
                                        num_iterations=100,
                                        n_jobs=self.n_jobs)

        if not self.extra_trees_kwargs:

            self.extra_trees_kwargs = dict(n_estimators=200,
                                           max_depth=25,
                                           class_weight='balanced',
                                           n_jobs=self.n_jobs,
                                           verbose=0)

        if not self.random_forest_kwargs:

            self.random_forest_kwargs = dict(n_estimators=200,
                                             max_depth=25,
                                             class_weight='balanced',
                                             n_jobs=self.n_jobs,
                                             verbose=0)

        if not self.bal_random_forest_kwargs:

            self.bal_random_forest_kwargs = dict(n_estimators=50,
                                                 max_depth=25,
                                                 n_jobs=self.n_jobs,
                                                 verbose=0)

        if not self.bal_bagging_kwargs:

            self.bal_bagging_kwargs = dict(n_estimators=50,
                                           n_jobs=self.n_jobs,
                                           verbose=0)

        lgb_model_ = lgb.LGBMClassifier(**self.lightgbm_kwargs) if LIGHTGBM_INSTALLED else None
        ext_model_ = ensemble.ExtraTreesClassifier(**self.extra_trees_kwargs)
        rfr_model_ = ensemble.RandomForestClassifier(**self.random_forest_kwargs)
        imb_model_ = imblearn.BalancedRandomForestClassifier(**self.bal_random_forest_kwargs) if IMBLEARN_INSTALLED else None
        bal_model_ = imblearn.BalancedBaggingClassifier(**self.bal_bagging_kwargs) if IMBLEARN_INSTALLED else None

        model_dict = {'lightgbm': lgb_model_,
                      'extra trees': ext_model_,
                      'random forest': rfr_model_,
                      'bal random forest': imb_model_,
                      'bal bagging': bal_model_}

        if not self.name:

            self.name = ['lightgbm',
                         'extra trees',
                         'random forest',
                         'bal random forest',
                         'bal bagging']

        if isinstance(self.name, str):
            self.name_list = [self.name]
        else:
            self.name_list = self.name

        for model_name in self.name_list:
            self.clf_dict[model_name] = model_dict[model_name]

    def _concat_classifiers(self,
                            X,
                            y,
                            sample_weight,
                            X_calibrate,
                            y_calibrate,
                            sample_weight_calibrate,
                            skf_cv,
                            cv_calibrate):

        """
        Calibrates a list of classifiers
        """

        fitted_estimators = list()

        # Fit each estimator
        for clf_name, clf in six.iteritems(self.clf_dict):

            if cv_calibrate:

                clf_ = CalibratedClassifierCV(base_estimator=clf,
                                              method='isotonic',
                                              cv=skf_cv)

                if self.verbose > 0:
                    logger.info('  Fitting and calibrating a {} model ...'.format(clf_name))

                # Fit and calibrate the model
                clf_.fit(X, y, sample_weight=sample_weight)

            else:

                if self.verbose > 0:
                    logger.info('  Fitting a {} model ...'.format(clf_name))

                # Check if the model supports sample weights.
                try:
                    argi = inspect.getargspec(clf.fit)
                except:
                    argi = inspect.getfullargspec(clf.fit)

                # Fit the model
                if 'sample_weight' in argi.args:
                    clf.fit(X, y, sample_weight=sample_weight)
                else:
                    clf.fit(X, y)

                if self.verbose > 0:
                    logger.info('  Calibrating a {} model ...'.format(clf_name))

                clf_ = CalibratedClassifierCV(base_estimator=clf,
                                              method='isotonic',
                                              cv='prefit')

                # Calibrate the pre-fitted model
                clf_.fit(X_calibrate, y_calibrate, sample_weight=sample_weight_calibrate)

            # Store the calibrated model
            fitted_estimators.append((clf_name, copy(clf_)))

        return fitted_estimators

    def fit(self,
            data,
            x=None,
            y=None,
            sample_weight=None,
            calibrate=True,
            cv_calibrate=True,
            cv_n_splits=3,
            cv_test_size=0.5,
            cv_train_size=0.5,
            x_calibrate=None,
            y_calibrate=None,
            sample_weight_calibrate=None):

        """
        Fits a model

        Args:
            data (DataFrame or GeoDataFrame)
            x (Optional[str list]): The X variable names.
            y (Optional[str]): The y response name.
            sample_weight (Optional[str]): The sample weight name.
            calibrate (Optional[bool])
            cv_calibrate (Optional[bool])
            cv_n_splits (Optional[int])
            cv_test_size (Optional[int])
            cv_train_size (Optional[int])
            x_calibrate (Optional[str])
            y_calibrate (Optional[str])
            sample_weight_calibrate (Optional[str])
        """

        if not isinstance(x, list):

            if isinstance(x, str):
                x = [x]
            else:
                logger.exception('  The x column(s) must be given.')

        if not isinstance(y, list):

            if isinstance(y, str):
                y = [y]
            else:
                logger.exception('  The y column must be given.')

        estimators = None

        self.x = [xname for xname in x if xname not in ['x', 'y']]
        self.y = y

        # Stratification object for calibrated cross-validation
        skf_cv = StratifiedShuffleSplit(n_splits=cv_n_splits,
                                        test_size=cv_test_size,
                                        train_size=cv_train_size)

        if self.backend.lower() == 'none':
            backend_context = _backend_dummy
        else:
            backend_context = joblib.parallel_backend

        if self.backend.lower() == 'dask':

            cluster = Cluster(n_workers=self.n_jobs,
                              threads_per_worker=1,
                              scheduler_port=0,
                              processes=False)

            cluster.start()

        with backend_context(self.backend, n_jobs=self.n_jobs):

            if calibrate and (len(self.name_list) > 1):

                estimators = self._concat_classifiers(data.loc[:, x].values,
                                                      data.loc[:, y].values.flatten(),
                                                      data.loc[:,
                                                      sample_weight].values.flatten() if sample_weight else None,
                                                      data.loc[:, x_calibrate].values if x_calibrate else None,
                                                      data.loc[:,
                                                      y_calibrate].values.flatten() if y_calibrate else None,
                                                      data.loc[:,
                                                      sample_weight_calibrate].values.flatten() if sample_weight_calibrate else None,
                                                      skf_cv,
                                                      cv_calibrate)

            else:

                model_ = self.clf_dict[self.name]

                model_.fit(data.loc[:, x].values,
                           data.loc[:, y].values.flatten(),
                           sample_weight=data.loc[:, sample_weight].values.flatten() if sample_weight else None)

        if estimators:
            model_ = VotingClassifier(estimators=estimators, y=y)

        self.model = ParallelPostFit(estimator=model_)

        self.model.classes_ = copy(self.classes_)

        if self.backend == 'dask':
            cluster.stop()

    def to_file(self, filename, overwrite=False):

        if overwrite:

            if os.path.isfile(filename):
                os.remove(filename)

        try:

            if self.verbose > 0:
                logger.info('  Saving model to file ...')

            joblib.dump((self.x, self.y, self.model),
                        filename,
                        compress=('zlib', 5),
                        protocol=-1)

        except:
            logger.warning('  Could not dump the model to file.')

    def from_file(self, filename):

        if not os.path.isfile(filename):
            logger.warning('  The model file does not exist.')
        else:

            if self.verbose > 0:
                logger.info('  Loading the model from file ...')

        self.x, self.y, self.model = joblib.load(filename)


class Predict(object):

    @staticmethod
    def _append_xy(data, dim):

        ycoords, xcoords = np.meshgrid(data.y, data.x)

        ycoords = xr.DataArray(da.from_array(ycoords[np.newaxis, :, :],
                                             chunks=(1, data.gw.row_chunks, data.gw.col_chunks)),
                               dims=(dim, 'y', 'x'),
                               coords={dim: ['lat'], 'y': data.y, 'x': data.x})

        xcoords = xr.DataArray(da.from_array(xcoords[np.newaxis, :, :],
                                             chunks=(1, data.gw.row_chunks, data.gw.col_chunks)),
                               dims=(dim, 'y', 'x'),
                               coords={dim: ['lon'], 'y': data.y, 'x': data.x})

        data_concat = xr.concat((data, xcoords, ycoords), dim=dim)
        data_concat.attrs = data.attrs

        return data_concat

    def predict(self,
                data,
                clf,
                outname=None,
                chunksize='same',
                x_chunks=None,
                use_xy=False,
                return_as='array',
                n_jobs=1,
                backend='none',
                verbose=0,
                nodata=None,
                dtype='uint8',
                **kwargs):

        """
        Predicts an image using a pre-fit model

        Args:
            data (DataArray): An ``xarray.DataArray`` to extract data from.
            clf (object): A fitted classifier ``geowombat.model.Model`` instance with a ``predict`` method.
            outname (Optional[str]): An file name for the predictions.
            chunksize (Optional[str or tuple]): The chunk size for I/O. Default is 'same', or use the input chunk size.
            x_chunks (Optional[tuple]): The chunk size for the X predictors (or ``data``).
            overwrite (Optional[bool]): Whether to overwrite an existing file.
            return_as (Optional[str]): Whether to return the predictions as a ``xarray.DataArray`` or ``xarray.Dataset``.
                *Only relevant if ``outname`` is not given.
            nodata (Optional[int or float]): The 'no data' value in the predictors.
            n_jobs (Optional[int]): The number of parallel jobs for the backend (if given). The ``n_workers`` and
                ``n_threads`` should be passed as ``kwargs`` to ``geowombat.to_raster``.
            backend (Optional[str]): The ``joblib`` backend scheduler.
            verbose (Optional[int]): The verbosity level.
            dtype (Optional[str]): The output data type passed to ``geowombat.to_raster``.
            kwargs (Optional[dict]): Additional keyword arguments passed to ``geowombat.to_raster``.

        Returns:
            ``xarray.DataArray``

        Examples:
            >>> from geowombat.models import predict
            >>> from sklearn import ensemble
            >>>
            >>> clf = ensemble.RandomForestClassifier()
            >>> clf.fit(X, y)
            >>>
            >>> with gw.open('image.tif') as ds:
            >>>     pred = predict(ds, clf)
            >>>
            >>> # Write results to file
            >>> with gw.open('image.tif') as ds:
            >>>     predict(ds, clf, outname='predictions.tif', n_workers=4, n_threads=2)
        """

        dim = 'band' if 'band' in data.coords else 'wavelength'

        if isinstance(clf, GeoWombatClassifier):

            # Select the bands that were used to train the model
            if dim == 'band':
                data = data.sel(band=clf.x)
            else:
                data = data.sel(wavelength=clf.x)

            clf = clf.model

        else:

            if not isinstance(clf, ParallelPostFit):
                clf = ParallelPostFit(estimator=clf)

        if not x_chunks:
            x_chunks = (data.gw.row_chunks*data.gw.col_chunks, 1)

        if use_xy:
            data = self._append_xy(data, dim)

        if verbose > 0:
            logger.info('  Predicting and saving to {} ...'.format(outname))

        if isinstance(chunksize, str) and chunksize == 'same':
            chunksize = Chunks().check_chunksize(data.data.chunksize, output='3d')
        else:

            if not isinstance(chunksize, tuple):
                logger.warning('  The chunksize parameter should be a tuple.')

            # TODO: make compatible with multi-layer predictions (e.g., probabilities)
            if len(chunksize) != 2:
                logger.warning('  The chunksize should be two-dimensional.')

        if backend.lower() == 'none':
            backend_context = _backend_dummy
        else:
            backend_context = joblib.parallel_backend

        if backend.lower() == 'dask':

            cluster = Cluster(n_workers=1,
                              threads_per_worker=n_jobs,
                              scheduler_port=0,
                              processes=False)

            cluster.start()

        with backend_context(backend, n_jobs=n_jobs):

            n_dims, n_rows, n_cols = data.shape

            # Reshape the data for fitting and
            #   return a Dask array
            if isinstance(nodata, int) or isinstance(nodata, float):
                X = data.stack(z=('y', 'x')).transpose().chunk(x_chunks).fillna(nodata).data
            else:
                X = data.stack(z=('y', 'x')).transpose().chunk(x_chunks).data

            # Apply the predictions
            predictions = clf.predict(X).reshape(1, n_rows, n_cols).rechunk(chunksize).astype(dtype)

            if return_as == 'dataset':

                # Store the predictions as an xarray.Dataset
                predictions = xr.Dataset({'pred': ([dim, 'y', 'x'], predictions)},
                                         coords={dim: [1],
                                                 'y': ('y', data.y),
                                                 'x': ('x', data.x)},
                                         attrs=data.attrs)

            else:

                # Store the predictions as an xarray.DataArray
                predictions = xr.DataArray(data=predictions,
                                           dims=(dim, 'y', 'x'),
                                           coords={dim: [1],
                                                   'y': ('y', data.y),
                                                   'x': ('x', data.x)},
                                           attrs=data.attrs)

            if isinstance(outname, str):
                predictions.gw.to_raster(outname, **kwargs)

        if backend.lower() == 'dask':
            cluster.stop()

        return predictions
