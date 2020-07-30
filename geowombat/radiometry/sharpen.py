import logging

from ..core import ndarray_to_xarray

import numpy as np
import dask
import dask.array as da
import xarray as xr
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_INSTALLED = True
except:
    LIGHTGBM_INSTALLED = False


logger = logging.getLogger(__name__)


@dask.delayed
def _interp(data, bins, cdf1, cdf2):
    return np.interp(np.interp(data.flatten(), bins[:-1], cdf1).flatten(), cdf2, bins[:-1])


def _assign_and_expand(obj, name, **attrs):

    obj = obj.assign_coords(coords={'band': name})
    obj = obj.expand_dims(dim='band')

    return obj.assign_attrs(**attrs)


def predict(datax, datay, bands, model_dict, ordinal, num_workers):

    """
    Applies a pre-trained regressor

    Args:
        datax (DataArray)
        datay (DataArray)
        bands (1d array-like)
        model_dict (dict)
        ordinal (int)
        num_workers (int)

    Returns:
        ``xarray.DataArray``
    """

    predictions = []

    X = datax.squeeze().data.compute(num_workers=num_workers).flatten()

    if isinstance(ordinal, int):

        ordinals = np.array([ordinal] * X.shape[0], dtype='float64')

        # y = datay.sel(band=band).squeeze().data.compute(num_workers=num_workers).flatten()

        X_ = np.c_[X, X ** 2, X ** 3, X ** 0.5, ordinals]

    else:
        X_ = X[:, np.newaxis]

    for band in bands:

        lr = model_dict[band]

        X_[np.isnan(X_) | np.isinf(X_)] = 0

        # Predict on the full array
        yhat = lr.predict(X_).reshape(datay.gw.nrows, datay.gw.ncols)

        # Convert to DataArray
        yhat = ndarray_to_xarray(datay, yhat, [band])

        predictions.append(yhat)

    return xr.concat(predictions, dim='band')


def regress(datax, datay, bands, frac, num_workers, nodata, scale_factor, robust, method, **kwargs):

    """
    Fits and applies a regressor

    Args:
        datax (DataArray)
        datay (DataArray)
        bands (1d array-like)
        frac (float)
        num_workers (int)
        nodata (float | int)
        scale_factor (float)
        robust (bool)
        method (str)
        kwargs (dict)

    Returns:
        ``xarray.DataArray``
    """

    if 'n_jobs' in kwargs:
        del kwargs['n_jobs']

    predictions = []

    X = datax.squeeze().data.compute(num_workers=num_workers).flatten()

    for band in bands:

        # Get the data for the current band
        y = datay.sel(band=band).squeeze().data.compute(num_workers=num_workers).flatten()

        kclu = KMeans(n_clusters=3).fit(y[:, np.newaxis])

        X_ = np.array([], dtype='float64')
        y_ = np.array([], dtype='float64')

        # Sample `frac` from each cluster
        for cluster in np.unique(kclu.labels_).tolist():

            idx0 = np.where((kclu.labels_ == cluster) &
                            (X != nodata*scale_factor) &
                            (y != nodata*scale_factor))[0]

            if idx0.shape[0] > 0:

                # Get a fraction of the samples
                idx1 = np.random.choice(idx0,
                                        size=int(idx0.shape[0] * frac),
                                        replace=False)

                X_ = np.concatenate((X_, X[idx1]))
                y_ = np.concatenate((y_, y[idx1]))

        # Get indices of valid samples
        # idx2 = np.where((X != nodata*scale_factor) & (y != nodata*scale_factor))

        # X_ = X[idx2]
        # y_ = y[idx2]

        if y_.shape[0] > 0:

            # Get a fraction of the samples
            # idx1 = np.random.choice(range(0, y_.shape[0]),
            #                         size=int(y_.shape[0] * frac),
            #                         replace=False)

            # X_ = X_[idx1]

            def prepare_x(xdata, index_y=True):

                # weights = [xdata, xdata ** 2, xdata ** 3, xdata ** 0.5]
                #
                # for other_band in bands:
                #
                #     if index_y:
                #         y0 = datay.sel(band=other_band).squeeze().data.compute(num_workers=num_workers)[idx0].flatten()[idx1]
                #     else:
                #         y0 = datay.sel(band=other_band).squeeze().data.compute(num_workers=num_workers).flatten()
                #
                #     weights.append(xdata / ((y0*0.1 + xdata) / 1.1))

                return np.c_[xdata, xdata ** 2, xdata ** 3, xdata ** 0.5, np.exp(xdata)]

            X_ = prepare_x(X_)

            # X_ = X_[idx][:, np.newaxis]
            # y_ = y_[idx1]

            if method.lower() == 'linear':

                if robust:
                    lr = TheilSenRegressor(n_jobs=num_workers, **kwargs)
                else:
                    lr = LinearRegression(n_jobs=num_workers, **kwargs)

            elif method.lower() == 'gb':

                if not LIGHTGBM_INSTALLED:
                    logger.exception('  LightGBM must be installed to use gradient boosting.')
                    raise ImportError

                if not kwargs:

                    kwargs = dict(boosting_type='dart',
                                  num_leaves=100,
                                  max_depth=50,
                                  n_estimators=200,
                                  subsample=0.75,
                                  subsample_freq=10,
                                  reg_alpha=0.1,
                                  reg_lambda=0.1,
                                  silent=True)

                lr = LGBMRegressor(n_jobs=num_workers, **kwargs)

            elif method.lower() == 'rf':
                lr = RandomForestRegressor(n_jobs=num_workers, **kwargs)

            lr.fit(X_, y_)

            # from sklearn import metrics
            # yhat = lr.predict(X_)
            # logger.info(metrics.mean_squared_error(y_, yhat))
            # logger.info(metrics.r2_score(y_, yhat))

            # Predict on the full array
            yhat = lr.predict(prepare_x(X.flatten(), index_y=False)).reshape(datay.gw.nrows, datay.gw.ncols)

            # Convert to DataArray
            yhat = ndarray_to_xarray(datay, yhat, [band])

            predictions.append(yhat)

        else:
            predictions.append(datay.sel(band=band))

    return xr.concat(predictions, dim='band')


def histogram_matching(data, ref_hist, **hist_kwargs):

    """
    Matches histograms

    Args:
        data (DataArray): The data to adjust.
        ref_hist (1d dask array): The reference histogram.
        hist_kwargs (Optional[dict]): The histogram keyword arguments.

    Returns:
        ``xarray.DataArray``
    """

    nrows = data.gw.nrows
    ncols = data.gw.ncols

    h, b = da.histogram(data.data, **hist_kwargs)

    # Cumulative distribution function.
    cdf1 = h.cumsum(axis=0)
    cdf2 = ref_hist.cumsum(axis=0)

    # Normalize
    cdf1 = (hist_kwargs['range'][1] * cdf1 / cdf1[-1]).astype('float64')
    cdf2 = (hist_kwargs['range'][1] * cdf2 / cdf2[-1]).astype('float64')

    matched = da.from_delayed(_interp(data.data, b, cdf1, cdf2),
                              (nrows * ncols,),
                              dtype='float64').reshape(nrows,
                                                       ncols).rechunk(data.gw.row_chunks,
                                                                      data.gw.col_chunks)

    return xr.DataArray(data=matched,
                        dims=('y', 'x'),
                        coords={'y': data.y,
                                'x': data.x},
                        attrs=data.attrs.copy())


def match_histograms(src_ms, src_sharp, bands, **hist_kwargs):

    """
    Matches histograms for multiple bands

    Args:
        src_ms (DataArray)
        src_sharp (DataArray)
        bands (1d array-like)
        hist_kwargs (Optional[dict]): The histogram keyword arguments.

    Return:
        ``xarray.DataArray``
    """

    matched = list()

    for band in bands:

        h = da.histogram(src_ms.sel(band=band).data, **hist_kwargs)[0]
        m = histogram_matching(src_sharp.clip(0, 1).sel(band=band), h, **hist_kwargs)
        matched.append(_assign_and_expand(m, band, **src_ms.attrs.copy()))

    return xr.concat(matched, dim='band')


def pan_sharpen(data,
                bands=None,
                pan=None,
                method='brovey',
                blue_weight=1.0,
                green_weight=1.0,
                red_weight=1.0,
                scale_factor=1.0,
                frac=0.1,
                num_workers=8,
                nodata=65535,
                robust=False,
                hist_match=False,
                model_dict=None,
                ordinal=None,
                **kwargs):

    """
    Sharpens wavelengths using the panchromatic band

    Args:
        data (DataArray): The band data.
        bands (Optional[list]): The bands to sharpen. If not given, 'blue', 'green', and 'red' are used.
        pan (Optional[DataArray]): The panchromatic ``DataArray``. ``pan`` is only needed if it is not included
            with ``data``.
        method (Optional[str]): The method to use. Choices are ['brovey', 'linear', 'rf', 'gb'].
        blue_weight (Optional[float]): The blue band weight.
        green_weight (Optional[float]): The green band weight.
        red_weight (Optional[float]): The red band weight.
        scale_factor (Optional[float]): A scale factor to apply to the data.
        frac (Optional[float]): The sample fraction.
        num_workers (Optional[int]): The number of parallel workers for ``sklearn.linear_model.LinearRegression``.
        nodata (Optional[int | float]): A 'no data' value to ignore.
        robust (Optional[bool]): Whether to fit a robust regression. Only applies when `method` = 'linear'.
        hist_match (Optional[bool]): Whether to match histograms after sharpening.
        model_dict (Optional[dict]): A dictionary of pre-trained regressors to apply to each band.
        ordinal (Optional[int]): A date ordinal to use for predictions.
        kwargs (Optional[dict]): Keyword arguments for the regressor. Only applies when `method` != 'brovey'.

    Example:
        >>> import geowombat as gw
        >>> from geowombat.radiometry import pan_sharpen
        >>>
        >>> with gw.config.update(sensor='l7', scale_factor=0.0001, ref_res=(15, 15)):
        >>>     with gw.open('image.tif', resampling='cubic') as src:
        >>>         pan_sharpen(src)

    Returns:
        ``xarray.DataArray``
    """

    if method.lower() not in ['brovey', 'linear', 'rf', 'gb']:
        logger.exception('  The method was not understood.')
        raise NameError

    if scale_factor == 1.0:
        scale_factor = data.gw.scale_factor

    if not bands:
        bands = ['blue', 'green', 'red']

    if ','.join(sorted(bands)) != 'blue,green,red':
        if ','.join(sorted(bands)) != 'blue,green,nir,red':
            logger.exception('  The bands must be blue,green,red or blue,green,red,nir')

    attrs = data.attrs.copy()

    data = data * scale_factor

    if isinstance(pan, xr.DataArray):
        pan = pan.sel(band='pan') * scale_factor
    else:
        pan = data.sel(band='pan')

    if model_dict:
        data_sharp = predict(pan, data, bands, model_dict, ordinal, num_workers)
    else:

        if (method.lower() == 'brovey') and (','.join(sorted(bands)) == 'blue,green,red'):

            weights = blue_weight + green_weight + red_weight

            band_avg = (data.sel(band='blue') * blue_weight +
                        data.sel(band='green') * green_weight +
                        data.sel(band='red') * red_weight) / weights

            dnf = pan / band_avg

            data_sharp = data.sel(band=bands) * dnf

        else:
            data_sharp = regress(pan, data, bands, frac, num_workers, nodata, scale_factor, robust, method, **kwargs)

    data_sharp = data_sharp.assign_coords(coords={'band': bands})

    if hist_match:
        data_sharp = match_histograms(data, data_sharp, bands, bins=100, range=(0.01, 1))
    
    data_sharp = (data_sharp / scale_factor).astype(data.dtype)

    return data_sharp.assign_attrs(**attrs)
