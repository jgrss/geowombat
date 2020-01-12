import xarray as xr


def pan_sharpen(data,
                bands=None,
                blue_weight=0.2,
                green_weight=1.0,
                red_weight=1.0,
                nir_weight=1.0,
                scale_factor=1.0):

    """
    Sharpens wavelengths using the panchromatic band

    Args:
        data (DataArray): The band data.
        bands (Optional[list]): The bands to sharpen. If not given, 'blue', 'green', 'red', and 'nir' are used.
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

    attrs = data.attrs.copy()

    data = data * scale_factor

    dnf = data.sel(band='pan') / ((data.sel(band='blue') * blue_weight +
                                   data.sel(band='green') * green_weight +
                                   data.sel(band='red') * red_weight) / (blue_weight + green_weight + red_weight))

    dnf = dnf.assign_coords(coords={'band': 'pan'})
    dnf = dnf.expand_dims(dim='band')
    dnf = dnf.assign_attrs(**attrs)

    data_sharp = xr.concat([(data.sel(band=bd) * dnf).transpose('band', 'y', 'x') for bd in bands], dim='band')
    data_sharp = data_sharp.assign_coords(coords={'band': bands})
    data_sharp = data_sharp.assign_attrs(**attrs)

    return (data_sharp / scale_factor).astype(data.dtype)
