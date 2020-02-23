from ..errors import logger
from ..core import ndarray_to_xarray

import numpy as np
import dask
import dask.array as da
import xarray as xr
from sklearn.linear_model import LinearRegression, TheilSenRegressor


@dask.delayed
def _interp(data, bins, cdf1, cdf2):
    return np.interp(np.interp(data.flatten(), bins[:-1], cdf1).flatten(), cdf2, bins[:-1])


def _assign_and_expand(obj, name, **attrs):

    obj = obj.assign_coords(coords={'band': name})
    obj = obj.expand_dims(dim='band')

    return obj.assign_attrs(**attrs)


def regress(datax, datay, bands, frac, num_workers, nodata):

    predictions = []

    X = datax.squeeze().data.compute(num_workers=num_workers)

    for band in bands:

        # Get the data for the current band
        y = datay.sel(band=band).squeeze().data.compute(num_workers=num_workers)

        # Get indices of valid samples
        idx = np.where((X != nodata) & (y != nodata))

        X_ = X[idx].flatten()
        y_ = y[idx].flatten()

        if y_.shape[0] > 0:

            # Get a fraction of the samples
            idx = np.random.choice(range(0, y_.shape[0]),
                                   size=int(y_.shape[0] * frac),
                                   replace=False)

            X_ = X_[idx][:, np.newaxis]
            y_ = y_[idx]

            lr = LinearRegression(n_jobs=num_workers)
            lr.fit(X_, y_)

            # Predict on the full array
            yhat = lr.predict(X.flatten()[:, np.newaxis])

            # Convert to DataArray
            yhat = ndarray_to_xarray(datay, yhat, [band])

            predictions.append(_assign_and_expand(yhat, band, **datay.attrs.copy()))

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
                blue_weight=1.0,
                green_weight=1.0,
                red_weight=1.0,
                nir_weight=1.0,
                scale_factor=1.0,
                hist_match=False):

    """
    Sharpens wavelengths using the panchromatic band

    Args:
        data (DataArray): The band data.
        bands (Optional[list]): The bands to sharpen. If not given, 'blue', 'green', and 'red' are used.
        pan (Optional[DataArray]): The panchromatic ``DataArray``. ``pan`` is only needed if it is not included
            with ``data``.
        blue_weight (Optional[float]): The blue band weight.
        green_weight (Optional[float]): The green band weight.
        red_weight (Optional[float]): The red band weight.
        nir_weight (Optional[float]): The NIR band weight.
        scale_factor (Optional[float]): A scale factor to apply to the data.
        hist_match (Optional[bool]): Whether to match histograms after sharpening.

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

    if ','.join(sorted(bands)) == 'blue,green,red':

        weights = blue_weight + green_weight + red_weight

        band_avg = (data.sel(band='blue') * blue_weight +
                    data.sel(band='green') * green_weight +
                    data.sel(band='red') * red_weight) / weights

        dnf = pan / band_avg

        data_sharp = data.sel(band=bands) * dnf

    else:

        # ESRI Brovey method with NIR
        # dnf = (pan - nir_weight * data.sel(band='nir')) / (data.sel(band='blue') * blue_weight +
        #                                                    data.sel(band='green') * green_weight +
        #                                                    data.sel(band='red') * red_weight)

        # ESRI method with NIR
        # wa = (data.sel(band='blue') * blue_weight +
        #       data.sel(band='green') * green_weight +
        #       data.sel(band='red') * red_weight +
        #       data.sel(band='nir') * nir_weight) / (blue_weight + green_weight + red_weight + nir_weight)
        #
        # adj = pan - wa
        #
        # data_sharp = data.sel(band=bands) + adj

        data_sharp = regress(pan, data, bands, 0.1, 8, 65535)

    data_sharp = data_sharp.assign_coords(coords={'band': bands})

    if hist_match:
        data_sharp = match_histograms(data, data_sharp, bands, bins=100, range=(0.01, 1))
    
    data_sharp = (data_sharp / scale_factor).astype(data.dtype)

    return data_sharp.assign_attrs(**attrs)
