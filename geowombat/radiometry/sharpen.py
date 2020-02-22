from ..errors import logger

import xarray as xr


def pan_sharpen(data,
                bands=None,
                pan=None,
                blue_weight=1.0,
                green_weight=1.0,
                red_weight=1.0,
                nir_weight=1.0,
                scale_factor=1.0):

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

    else:

        # ESRI Brovey method with NIR
        dnf = (pan - nir_weight * data.sel(band='nir')) / (data.sel(band='blue') * blue_weight +
                                                           data.sel(band='green') * green_weight +
                                                           data.sel(band='red') * red_weight)

        # weights = blue_weight + green_weight + red_weight + nir_weight
        #
        # band_avg = (data.sel(band='blue') * blue_weight +
        #             data.sel(band='green') * green_weight +
        #             data.sel(band='red') * red_weight +
        #             data.sel(band='nir') * nir_weight) / weights

    data_sharp = data.sel(band=bands) * dnf
    data_sharp = data_sharp.assign_coords(coords={'band': bands})

    data_sharp = (data_sharp / scale_factor).astype(data.dtype)

    return data_sharp.assign_attrs(**attrs)
