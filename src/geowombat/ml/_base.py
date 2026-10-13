"""Shared, torch-free base utilities for geowombat classifiers.

These helpers are used by the classifiers that handle their own xarray
``fit()`` / ``predict()`` (deep-learning models in
:mod:`geowombat.ml.dl_classifiers` and the MiniRocket time-series
classifier in :mod:`geowombat.ml.rocket_classifiers`). They are kept
here so that classifiers which do not need PyTorch can import them
without triggering the torch import guard.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np

from .. import polygon_to_array


def _get_nodata_mask(data):
    """Return a 2-D boolean mask (y, x) that is True for nodata pixels.

    Prefers the ``_nodata_mask`` coordinate (warped from the original file)
    when available.  Falls back to value comparison using ``nodatavals`` from
    attributes.
    """
    if '_nodata_mask' in data.coords:
        return data.coords['_nodata_mask']

    nodatavals = data.attrs.get("nodatavals")
    if not nodatavals or len(nodatavals) == 0:
        return None

    src_nd = nodatavals[0]
    if isinstance(src_nd, float) and np.isnan(src_nd):

        def _is_nd(a):
            return a.isnull()

    else:

        def _is_nd(a):
            return a == src_nd

    if data.ndim == 4:
        return _is_nd(data.isel(time=0)).any(dim='band')
    else:
        return _is_nd(data).any(dim='band')


def _rasterize_labels(data, labels, col):
    """Rasterize vector labels onto the data grid.

    Parameters
    ----------
    data : xarray.DataArray
        Reference raster (band, y, x) or (time, band, y, x).
    labels : str, Path, or GeoDataFrame
        Vector labels.
    col : str
        Column name with class values.

    Returns
    -------
    label_array : numpy.ndarray
        2D array (y, x) with integer class labels (0 = nodata).
    n_classes : int
        Number of unique non-zero classes.
    """
    if isinstance(labels, (str, Path)):
        labels = gpd.read_file(labels)

    labels = labels.copy()

    # Encode labels to 1-based integers (0 = nodata)
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(labels[col])
    labels[col] = le.transform(labels[col]) + 1

    # Ensure data is dask-backed (polygon_to_array needs .chunksize)
    if not hasattr(data.data, 'chunksize'):
        data = data.chunk(
            {'y': min(256, data.sizes['y']), 'x': min(256, data.sizes['x'])}
        )

    label_raster = polygon_to_array(labels, col=col, data=data)
    label_arr = label_raster.values.squeeze()

    n_classes = len(le.classes_)

    return label_arr, n_classes


class GeoWombatDLClassifier:
    """Base class for classifiers that handle their own xarray fit/predict.

    Subclasses must implement ``fit()`` and ``predict()``. The
    ``_is_gw_dl_classifier`` attribute is used by ``Classifiers.fit()`` /
    ``predict()`` to delegate to these methods instead of the sklearn_xarray
    pipeline.
    """

    _is_gw_dl_classifier = True

    def fit(self, data, labels=None, col=None, targ_name="targ", **kwargs):
        raise NotImplementedError

    def predict(self, data, **kwargs):
        raise NotImplementedError
