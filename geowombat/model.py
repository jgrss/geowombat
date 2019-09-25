import os
from copy import copy
import ctypes
import inspect

from .errors import logger

import numpy as np
import joblib
import six

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import ensemble
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedShuffleSplit

import xarray as xr

from dask.distributed import Client, LocalCluster
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

            logger.error('  The length of the weights must match the length of the estimators.')
            raise ArrayShapeError

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


class Model(object):

    """
    A class for model fitting models with Dask

    Args:
        name (str or list): A model or list of models to use.
            Choices are ['lightgbm', 'extra trees', 'random forest', 'bal random forest', 'bag bagging'].
        lightgbm_kwargs (dict): Keyword arguments for LightGBM.
        extra_trees_kwargs (dict): Keyword arguments for Extra Trees.
        random_forest_kwargs (dict): Keyword arguments for Random Forest.
        bal_random_forest_kwargs (dict): Keyword arguments for Balanced Random Forest.
        bal_bagging_kwargs (dict): Keyword arguments for Balanced Bagging.
        n_jobs (Optional[int]): The number of parallel processes.

    Examples:
        >>> ml = Model(['lightgbm', 'random forest'], n_jobs=8)
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
                 use_dask=True,
                 n_jobs=-1):

        if MKL_LIB:
            __ = MKL_LIB.MKL_Set_Num_Threads(n_jobs)

        self.use_dask = use_dask
        self.n_jobs = n_jobs

        self.cluster = LocalCluster(n_workers=1,
                                    threads_per_worker=self.n_jobs,
                                    scheduler_port=0,
                                    processes=False)

        self.client = Client(self.cluster)

        self.name = name
        self.lightgbm_kwargs = lightgbm_kwargs
        self.extra_trees_kwargs = extra_trees_kwargs
        self.random_forest_kwargs = random_forest_kwargs
        self.bal_random_forest_kwargs = bal_random_forest_kwargs
        self.bal_bagging_kwargs = bal_bagging_kwargs
        self.verbose = verbose

        self.clf_dict = dict()
        self.model = None

        self._setup_models()

    @property
    def classes_(self):
        """Returns the class list from the first estimator"""
        try:
            return self.model.estimator.estimators[0][1].classes_
        except:
            return self.model.estimators[0][1].classes_

    @property
    def class_count(self):
        return self.classes_.shape[0]

    def close_client(self):

        self.client.close()
        self.cluster.close()

        self.client = None
        self.cluster = None

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

        model_dict = {'lightgbm': lgb.LGBMClassifier(**self.lightgbm_kwargs),
                      'extra trees': ensemble.ExtraTreesClassifier(**self.extra_trees_kwargs),
                      'random forest': ensemble.RandomForestClassifier(**self.random_forest_kwargs),
                      'bal random forest': imblearn.BalancedRandomForestClassifier(**self.bal_random_forest_kwargs),
                      'bal bagging': imblearn.BalancedBaggingClassifier(**self.bal_bagging_kwargs)}

        if isinstance(self.name, str):
            self.name = [self.name]

        if not self.name:

            self.name = ['lightgbm',
                         'extra trees',
                         'random forest',
                         'bal random forest',
                         'bal bagging']

        for model_name in self.name:
            self.clf_dict[model_name] = model_dict[model_name]

    def _calibrate_classifiers(self,
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
            cv_calibrate=True,
            cv_n_splits=3,
            cv_test_size=0.5,
            cv_train_size=0.5,
            X_calibrate=None,
            y_calibrate=None,
            sample_weight_calibrate=None):

        """
        Fits a model

        Args:
            data (DataFrame or GeoDataFrame)
            x (Optional[str list]): The X variable names.
            y (Optional[str]): The y response name.
            sample_weight (Optional[str]): The sample weight name.
            cv_calibrate (Optional[bool])
            cv_n_splits (Optional[int])
            cv_test_size (Optional[int])
            cv_train_size (Optional[int])
            X_calibrate (Optional[str])
            y_calibrate (Optional[str])
            sample_weight_calibrate (Optional[str])
        """

        # Stratification object for calibrated cross-validation
        skf_cv = StratifiedShuffleSplit(n_splits=cv_n_splits,
                                        test_size=cv_test_size,
                                        train_size=cv_train_size)

        if self.use_dask:

            with joblib.parallel_backend('dask'):

                fitted_estimators = self._calibrate_classifiers(data.loc[:, x].values,
                                                                data.loc[:, y].values.flatten(),
                                                                sample_weight,
                                                                X_calibrate,
                                                                y_calibrate,
                                                                sample_weight_calibrate,
                                                                skf_cv,
                                                                cv_calibrate)

        else:

            fitted_estimators = self._calibrate_classifiers(data.loc[:, x].values,
                                                            data.loc[:, y].values.flatten(),
                                                            sample_weight,
                                                            X_calibrate,
                                                            y_calibrate,
                                                            sample_weight_calibrate,
                                                            skf_cv,
                                                            cv_calibrate)

        if len(fitted_estimators) == 1:
            model_ = fitted_estimators[0]
        else:
            model_ = VotingClassifier(estimators=fitted_estimators, y=y)

        self.model = ParallelPostFit(estimator=model_)

    def to_file(self, filename, overwrite=False):

        if overwrite:

            if os.path.isfile(filename):
                os.remove(filename)

        try:

            if self.verbose > 0:
                logger.info('  Saving model to file ...')

            joblib.dump(self.model,
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

        self.model = joblib.load(filename)
