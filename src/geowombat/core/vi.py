import logging
import typing as T

from ..config import config
from ..handler import add_handler
from .base import PropertyMixin as _PropertyMixin

import numpy as np
import xarray as xr
import dask.array as da


logger = logging.getLogger(__name__)
logger = add_handler(logger)


def _create_nodata_array(data, nodata, band_name, var_name):

    """Creates a 'no data' Xarray.

    Args:
        data (Xarray)
    """

    return xr.DataArray(
        data=da.zeros(
            (1, data.gw.nrows, data.gw.ncols),
            chunks=(1, data.gw.row_chunks, data.gw.col_chunks),
            dtype=data.dtype.name,
        )
        + nodata,
        coords={band_name: [var_name], 'y': data.y.values, 'x': data.x.values},
        dims=(band_name, 'y', 'x'),
        attrs=data.attrs,
    )


class BandMath(object):
    @staticmethod
    def scale_and_assign(
        data: xr.DataArray,
        nodata: T.Union[float, int],
        band_variable: str,
        scale_factor: float,
        names: T.Sequence[T.Any],
        new_names: T.Sequence[T.Any],
    ) -> xr.DataArray:
        attrs = data.attrs.copy()

        if band_variable == 'wavelength':
            band_data = (
                data.astype('float64')
                .where(np.isfinite(data))
                .where(lambda x: x.band != nodata)
                .sel(wavelength=names)
            )

        else:
            band_data = (
                data.astype('float64')
                .where(np.isfinite(data))
                .where(lambda x: x.band != nodata)
                .sel(band=names)
            )

        return (
            (band_data.astype('float64') * scale_factor)
            .assign_coords(coords={band_variable: new_names})
            .assign_attrs(**attrs)
        )

    @staticmethod
    def mask_and_assign(
        data: xr.DataArray,
        result: xr.DataArray,
        band_variable: str,
        band_name: str,
        nodata: T.Union[float, int],
        new_name: str,
        mask: bool,
        clip_min: T.Union[float, int],
        clip_max: T.Union[float, int],
        scale_factor: float,
        sensor: str,
        norm: bool = False,
    ) -> xr.DataArray:
        """Masks an ``xarray.DataArray``.

        Args:
            data (DataArray or Dataset)
            result (DataArray)
            band_variable (str)
            band_name (str)
            nodata (int or float)
            new_name (str)
            mask (bool)
            clip_min (float | int)
            clip_max (float | int)
            scale_factor (float)
            sensor (str)
            norm (bool)

        Returns:
            ``xarray.DataArray``
        """
        if isinstance(nodata, (float, int)):
            if band_variable == 'wavelength':
                result = xr.where(
                    data.sel(wavelength=band_name) == nodata, nodata, result
                ).astype('float64')
            else:
                result = xr.where(
                    data.sel(band=band_name) == nodata, nodata, result
                ).astype('float64')

        if mask:
            if isinstance(data, xr.Dataset):
                result = result.where(data['mask'] < 3)
            else:

                if band_variable == 'wavelength':
                    result = result.where(data.sel(wavelength='mask') < 3)
                else:
                    result = result.where(data.sel(band='mask') < 3)

        new_attrs = data.attrs.copy()

        new_attrs['pre-scaling'] = scale_factor
        new_attrs['sensor'] = sensor
        new_attrs['vi'] = new_name
        new_attrs['drange'] = (clip_min, clip_max)

        result = result.clip(min=clip_min, max=clip_max)

        if norm:
            result = result / clip_max

        if result.gw.has_band_coord and result.gw.has_band_dim:
            result = result.assign_coords(coords={band_variable: [new_name]})
        else:
            result = result.assign_coords(coords={band_variable: new_name})

        if not result.gw.has_band_dim:
            result = result.expand_dims(dim=band_variable)

        # Ensure expected order
        if result.gw.has_time:
            result = result.transpose('time', 'band', 'y', 'x')
        else:
            result = result.transpose('band', 'y', 'x')

        result = result.assign_attrs(**new_attrs).gw.set_nodata(
            nodata, nodata, scale_factor=1.0, offset=0
        )

        return result

    def norm_diff_math(
        self,
        data: xr.DataArray,
        b1: xr.DataArray,
        b2: xr.DataArray,
        name: str,
        sensor: str,
        nodata: T.Union[float, int] = None,
        mask: bool = False,
        scale_factor: float = None,
    ) -> xr.DataArray:
        """Normalized difference index --> (b2 - b1) / (b2 + b1)

        Args:
            data (DataArray)
            b1 (DataArray): Band 1
            b2 (DataArray): Band 2
            name (str)
            sensor (str)
            nodata (Optional[int])
            mask (Optional[bool])
            scale_factor (Optional[float])

        Returns:
            ``xarray.DataArray``
        """
        band_variable = 'wavelength' if 'wavelength' in data.coords else 'band'

        band_data = self.scale_and_assign(
            data, nodata, band_variable, scale_factor, [b1, b2], [b1, b2]
        )

        if band_variable == 'wavelength':
            result = (
                (
                    (
                        band_data.sel(wavelength=b2)
                        - band_data.sel(wavelength=b1)
                    )
                    / (
                        band_data.sel(wavelength=b2)
                        + band_data.sel(wavelength=b1)
                    )
                )
                .fillna(nodata)
                .astype('float64')
            )

        else:

            result = (
                (
                    (band_data.sel(band=b2) - band_data.sel(band=b1))
                    / (band_data.sel(band=b2) + band_data.sel(band=b1))
                )
                .fillna(nodata)
                .astype('float64')
            )

        return self.mask_and_assign(
            band_data,
            result,
            band_variable,
            b2,
            nodata,
            name,
            mask,
            -1,
            1,
            scale_factor,
            sensor,
        )

    def avi_math(
        self,
        data,
        sensor,
        wavelengths,
        nodata=None,
        mask=False,
        scale_factor=None,
    ):

        """Advanced vegetation index.

        Returns:
            ``xarray.DataArray``
        """

        band_variable = 'wavelength' if 'wavelength' in data.coords else 'band'

        if 'nir' in data.coords[band_variable].values.tolist():
            nir = 'nir'
            red = 'red'
        else:
            nir = wavelengths[sensor].nir
            red = wavelengths[sensor].red

        data = self.scale_and_assign(
            data,
            nodata,
            band_variable,
            scale_factor,
            [red, nir],
            ['red', 'nir'],
        )

        if band_variable == 'wavelength':

            result = (
                (
                    (
                        data.sel(wavelength='nir')
                        * (1.0 - data.sel(wavelength='red'))
                        * (
                            data.sel(wavelength='nir')
                            - data.sel(wavelength='red')
                        )
                    )
                    ** 0.3334
                )
                .fillna(nodata)
                .astype('float64')
            )

        else:

            result = (
                (
                    (
                        data.sel(band='nir')
                        * (1.0 - data.sel(band='red'))
                        * (data.sel(band='nir') - data.sel(band='red'))
                    )
                    ** 0.3334
                )
                .fillna(nodata)
                .astype('float64')
            )

        return self.mask_and_assign(
            data,
            result,
            band_variable,
            'nir',
            nodata,
            'avi',
            mask,
            0,
            1,
            scale_factor,
            sensor,
        )

    def evi_math(
        self,
        data,
        sensor,
        wavelengths,
        nodata=None,
        mask=False,
        scale_factor=None,
    ):

        """Enhanced vegetation index.

        Returns:
            ``xarray.DataArray``
        """

        l_factor = 1.0
        c1 = 6.0
        c2 = 7.5
        g_factor = 2.5

        band_variable = 'wavelength' if 'wavelength' in data.coords else 'band'

        if 'nir' in data.coords[band_variable].values.tolist():
            nir = 'nir'
            red = 'red'
            blue = 'blue'
        else:
            nir = wavelengths[sensor].nir
            red = wavelengths[sensor].red
            blue = wavelengths[sensor].blue

        data = self.scale_and_assign(
            data,
            nodata,
            band_variable,
            scale_factor,
            [nir, red, blue],
            ['nir', 'red', 'blue'],
        )

        if band_variable == 'wavelength':

            result = (
                (
                    g_factor
                    * (data.sel(wavelength='nir') - data.sel(wavelength='red'))
                    / (
                        data.sel(wavelength='nir')
                        * c1
                        * data.sel(wavelength='red')
                        - c2 * data.sel(wavelength='blue')
                        + l_factor
                    )
                )
                .fillna(nodata)
                .astype('float64')
            )

        else:

            result = (
                (
                    g_factor
                    * (data.sel(band='nir') - data.sel(band='red'))
                    / (
                        data.sel(band='nir') * c1 * data.sel(band='red')
                        - c2 * data.sel(band='blue')
                        + l_factor
                    )
                )
                .fillna(nodata)
                .astype('float64')
            )

        return self.mask_and_assign(
            data,
            result,
            band_variable,
            'nir',
            nodata,
            'evi',
            mask,
            0,
            1,
            scale_factor,
            sensor,
        )

    def evi2_math(
        self,
        data,
        sensor,
        wavelengths,
        nodata=None,
        mask=False,
        scale_factor=None,
    ):
        """Two-band enhanced vegetation index.

        Returns:
            ``xarray.DataArray``
        """

        band_variable = 'wavelength' if 'wavelength' in data.coords else 'band'

        if 'nir' in data.coords[band_variable].values.tolist():
            nir = 'nir'
            red = 'red'
        else:
            nir = wavelengths[sensor].nir
            red = wavelengths[sensor].red

        data = self.scale_and_assign(
            data,
            nodata,
            band_variable,
            scale_factor,
            [nir, red],
            ['nir', 'red'],
        )

        if band_variable == 'wavelength':

            result = (
                (
                    2.5
                    * (
                        (
                            data.sel(wavelength='nir')
                            - data.sel(wavelength='red')
                        )
                        / (
                            data.sel(wavelength='nir')
                            + 1.0
                            + (2.4 * (data.sel(wavelength='red')))
                        )
                    )
                )
                .fillna(nodata)
                .astype('float64')
            )

        else:

            result = (
                (
                    2.5
                    * (
                        (data.sel(band='nir') - data.sel(band='red'))
                        / (
                            data.sel(band='nir')
                            + 1.0
                            + (2.4 * (data.sel(band='red')))
                        )
                    )
                )
                .fillna(nodata)
                .astype('float64')
            )

        return self.mask_and_assign(
            data,
            result,
            band_variable,
            'nir',
            nodata,
            'evi2',
            mask,
            0,
            1,
            scale_factor,
            sensor,
        )

    def gcvi_math(
        self,
        data,
        sensor,
        wavelengths,
        nodata=None,
        mask=False,
        scale_factor=None,
    ):
        """Green chlorophyll vegetation index.

        Returns:
            ``xarray.DataArray``
        """

        band_variable = 'wavelength' if 'wavelength' in data.coords else 'band'

        if 'nir' in data.coords[band_variable].values.tolist():
            nir = 'nir'
            green = 'green'
        else:
            nir = wavelengths[sensor].nir
            green = wavelengths[sensor].green

        data = self.scale_and_assign(
            data,
            nodata,
            band_variable,
            scale_factor,
            [nir, green],
            ['nir', 'green'],
        )

        if band_variable == 'wavelength':
            result = (
                data.sel(wavelength='nir') / data.sel(wavelength='green')
            ) - 1.0
        else:
            result = (data.sel(band='nir') / data.sel(band='green')) - 1.0

        return self.mask_and_assign(
            data,
            result,
            band_variable,
            'nir',
            nodata,
            'gcvi',
            mask,
            0,
            10,
            scale_factor,
            sensor,
            norm=True,
        )

    def nbr_math(
        self,
        data,
        sensor,
        wavelengths,
        nodata=None,
        mask=False,
        scale_factor=None,
    ):
        """Normalized burn ratio.

        Returns:
            ``xarray.DataArray``
        """

        band_variable = 'wavelength' if 'wavelength' in data.coords else 'band'

        if 'nir' in data.coords[band_variable].values.tolist():
            nir = 'nir'
            swir2 = 'swir2'
        else:
            nir = wavelengths[sensor].nir
            swir2 = wavelengths[sensor].swir2

        return self.norm_diff_math(
            data,
            swir2,
            nir,
            'nbr',
            sensor,
            nodata=nodata,
            mask=mask,
            scale_factor=scale_factor,
        )

    def ndvi_math(
        self,
        data,
        sensor,
        wavelengths,
        nodata=None,
        mask=False,
        scale_factor=None,
    ):
        """Normalized difference vegetation index.

        Returns:
            ``xarray.DataArray``
        """

        band_variable = 'wavelength' if 'wavelength' in data.coords else 'band'

        if 'nir' in data.coords[band_variable].values.tolist():
            nir = 'nir'
            red = 'red'
        else:
            nir = wavelengths[sensor].nir
            red = wavelengths[sensor].red

        return self.norm_diff_math(
            data,
            red,
            nir,
            'ndvi',
            sensor,
            nodata=nodata,
            mask=mask,
            scale_factor=scale_factor,
        )

    def kndvi_math(
        self,
        data,
        sensor,
        wavelengths,
        nodata=None,
        mask=False,
        scale_factor=None,
    ):
        """Kernel normalized difference vegetation index.

        Returns:
            ``xarray.DataArray``
        """
        band_variable = 'wavelength' if 'wavelength' in data.coords else 'band'

        if 'nir' in data.coords[band_variable].values.tolist():
            nir = 'nir'
            red = 'red'
        else:
            nir = wavelengths[sensor].nir
            red = wavelengths[sensor].red

        # Calculate the NDVI
        result = self.norm_diff_math(
            data,
            red,
            nir,
            'kndvi',
            sensor,
            nodata=nodata,
            mask=mask,
            scale_factor=scale_factor,
        )
        # Calculate the kNDVI
        result = np.tanh(result**2)

        # This snippet is the same as above, but allows for different kernels
        # if band_variable == 'wavelength':
        #
        #     sigma = (data.sel(wavelength='red') + data.sel(wavelength='nir')) * 0.5
        #     knr = np.exp(-(data.sel(wavelength='nir') - data.sel(wavelength='red'))**2 / (2.0*sigma**2))
        #     result = (1.0 - knr) / (1.0 + knr)
        #
        # else:
        #     sigma = (data.sel(band='red') + data.sel(band='nir')) * 0.5
        #     knr = np.exp(-(data.sel(band='nir') - data.sel(band='red'))**2 / (2.0*sigma**2))
        #     result = (1.0 - knr) / (1.0 + knr)

        return self.mask_and_assign(
            data,
            result,
            band_variable,
            'nir',
            nodata,
            'kndvi',
            mask,
            -1,
            1,
            scale_factor,
            sensor,
        )

    def wi_math(
        self,
        data,
        sensor,
        wavelengths,
        nodata=None,
        mask=False,
        scale_factor=None,
    ):
        """Woody index.

        Returns:
            ``xarray.DataArray``
        """

        band_variable = 'wavelength' if 'wavelength' in data.coords else 'band'

        if 'swir1' in data.coords[band_variable].values.tolist():
            swir1 = 'swir1'
            red = 'red'
        else:
            swir1 = wavelengths[sensor].swir1
            red = wavelengths[sensor].red

        data = self.scale_and_assign(
            data,
            nodata,
            band_variable,
            scale_factor,
            [swir1, red],
            ['swir1', 'red'],
        )

        if band_variable == 'wavelength':
            result = data.sel(wavelength='swir1') + data.sel(wavelength='red')
        else:
            result = data.sel(band='swir1') + data.sel(band='red')

        result = (
            xr.where(result > 0.5, 0, 1.0 - (result / 0.5))
            .fillna(nodata)
            .astype('float64')
        )

        return self.mask_and_assign(
            data,
            result,
            band_variable,
            'red',
            nodata,
            'wi',
            mask,
            0,
            1,
            scale_factor,
            sensor,
        )


def linear_transform(
    data: xr.DataArray,
    bands: T.Sequence[T.Any],
    scale: T.Sequence[T.Union[float, int]],
    offset: T.Sequence[T.Union[float, int]],
) -> xr.DataArray:
    r"""Linearly scales bands using a scale and an offset

    Args:
        data (DataArray): The ``xarray.DataArray`` to transform.
        bands (1d array-like): The list of bands to transform.
        scale (1d array): The scale coefficients.
        offset (1d array): The offset coefficients.

    Equation:

        .. math::

            y = scale \times band + offset

    Returns:
        ``xarray.DataArray``
    """
    scalexr = xr.DataArray(scale, coords=[bands], dims=['band'])
    offsetxr = xr.DataArray(offset, coords=[bands], dims=['band'])

    return data * scalexr + offsetxr


class TasseledCapLookup(object):
    @staticmethod
    def get_coefficients(wavelengths, sensor):
        lookup_dict = dict(
            aster=np.array(
                [
                    [0.3909, -0.0318, 0.4571],
                    [0.5224, -0.1031, 0.4262],
                    [0.1184, 0.9422, -0.1568],
                    [0.3233, 0.2512, 0.2809],
                    [0.305, -0.0737, -0.2417],
                    [0.3571, -0.069, -0.3269],
                    [0.3347, -0.0957, -0.4077],
                    [0.3169, -0.1195, -0.3731],
                    [0.151, -0.0625, -0.1877],
                ],
                dtype='float64',
            ),
            cbers2=np.array(
                [
                    [0.509, -0.494, 0.581],
                    [0.431, -0.318, -0.07],
                    [0.33, -0.324, -0.811],
                    [0.668, 0.741, 0.003],
                ],
                dtype='float64',
            ),
            ik=np.array(
                [
                    [0.326, -0.311, -0.612],
                    [0.509, -0.256, -0.312],
                    [0.56, -0.325, 0.722],
                    [0.567, 0.819, -0.081],
                ],
                dtype='float64',
            ),
            l4=np.array(
                [
                    [0.433, -0.29, -0.829],
                    [0.632, -0.562, 0.522],
                    [0.586, 0.6, -0.039],
                    [0.264, 0.491, 0.194],
                ],
                dtype='float64',
            ),
            l5=np.array(
                [
                    [0.3037, -0.2848, 0.1509],
                    [0.2793, -0.2435, 0.1793],
                    [0.4343, -0.5436, 0.3299],
                    [0.5585, 0.7243, 0.3406],
                    [0.5082, 0.084, -0.7112],
                    [0.1863, -0.18, -0.4572],
                ],
                dtype='float64',
            ),
            l7=np.array(
                [
                    [0.3561, -0.3344, 0.2626],
                    [0.3972, -0.3544, 0.2141],
                    [0.3904, -0.4556, 0.0926],
                    [0.6966, 0.6966, 0.0656],
                    [0.2286, -0.0242, -0.7629],
                    [0.1596, -0.263, -0.5388],
                ],
                dtype='float64',
            ),
            l8=np.array(
                [
                    [0.3029, -0.2941, 0.1511],
                    [0.2786, -0.243, 0.1973],
                    [0.4733, -0.5424, 0.3283],
                    [0.5599, 0.7276, 0.3407],
                    [0.508, 0.0713, -0.7117],
                    [0.1872, -0.1608, -0.4559],
                ],
                dtype='float64',
            ),
            modis=np.array(
                [
                    [0.4395, -0.4064, 0.1147],
                    [0.5945, 0.5129, 0.2489],
                    [0.2460, -0.2744, 0.2408],
                    [0.3918, -0.2893, 0.3132],
                    [0.3506, 0.4882, -0.3122],
                    [0.2136, -0.0036, -0.6416],
                    [0.2678, -0.4169, -0.5087],
                ],
                dtype='float64',
            ),
            qb=np.array(
                [
                    [0.319, -0.121, 0.652],
                    [0.542, -0.331, 0.375],
                    [0.49, -0.517, -0.639],
                    [0.604, 0.78, -0.163],
                ],
                dtype='float64',
            ),
            rapideye=np.array(
                [
                    [-0.293, -0.406, 0.572],
                    [-0.354, -0.367, 0.402],
                    [-0.372, -0.446, -0.494],
                    [-0.44, -0.128, -0.498],
                    [-0.676, 0.697, 0.138],
                ],
                dtype='float64',
            ),
        )

        return xr.DataArray(
            data=lookup_dict[sensor],
            coords={
                'coeff': ['brightness', 'greenness', 'wetness'],
                'band': list(wavelengths[sensor]._fields),
            },
            dims=('band', 'coeff'),
        )


class TasseledCap(_PropertyMixin, TasseledCapLookup):
    def tasseled_cap(
        self,
        data: xr.DataArray,
        nodata: T.Union[float, int] = None,
        sensor: str = None,
        scale_factor: float = None,
    ) -> xr.DataArray:
        r"""Applies a tasseled cap transformation

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            nodata (Optional[int or float]): A 'no data' value to fill NAs with. If ``None``,
                the 'no data' value is taken from the ``xarray.DataArray`` attributes.
            sensor (Optional[str]): The data's sensor. If ``None``, the band names should
                reflect the index being calculated.
            scale_factor (Optional[float]): A scale factor to apply to the data. If ``None``,
                the scale value is taken from the ``xarray.DataArray`` attributes.

        Examples:
            >>> import geowombat as gw
            >>>
            >>> with gw.config.update(sensor='qb', scale_factor=0.0001):
            >>>     with gw.open('image.tif', band_names=['blue', 'green', 'red', 'nir']) as ds:
            >>>         tcap = gw.tasseled_cap(ds)

        Returns:
            ``xarray.DataArray``

        References:

            ASTER:
                See :cite:`yajuan_danfeng_2005`

            CBERS2:
                See :cite:`sheng_etal_2011`

            IKONOS:
                See :cite:`pereira_2006`

            Landsat ETM+:
                See :cite:`huang_etal_2002`

            Landsat OLI:
                See :cite:`baig_etal_2014`

            MODIS:
                See :cite:`lobser_cohen_2007`

            Quickbird:
                See :cite:`yarbrough_etal_2005`

            RapidEye:
                See :cite:`arnett_etal_2014`
        """
        sensor = self.check_sensor(data, sensor)

        if nodata is None:
            nodata = data.gw.nodataval

        if config['scale_factor'] is not None:
            scale_factor = config['scale_factor']
        else:
            if scale_factor is None:
                scale_factor = data.gw.scaleval

        tc_coefficients = self.get_coefficients(data.gw.wavelengths, sensor)

        tcap = (
            ((data.where(data != nodata) * scale_factor) * tc_coefficients)
            .sum(dim='band')
            .fillna(nodata)
            .transpose('coeff', 'y', 'x')
            .rename({'coeff': 'band'})
        )

        return tcap.astype('float64').assign_attrs(**data.attrs)


class VegetationIndices(_PropertyMixin, BandMath):
    def norm_diff(
        self,
        data,
        b1,
        b2,
        sensor=None,
        nodata=None,
        mask=False,
        scale_factor=None,
    ):

        r"""
        Calculates the normalized difference band ratio

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            b1 (str): The band name of the first band.
            b2 (str): The band name of the second band.
            sensor (Optional[str]): sensor (Optional[str]): The data's sensor. If ``None``, the band names should reflect the index being calculated.
            nodata (Optional[int or float]): A 'no data' value to fill NAs with. If ``None``, the 'no data' value is taken from the ``DataArray`` attributes.
            mask (Optional[bool]): Whether to mask the results.
            scale_factor (Optional[float]): A scale factor to apply to the data. If ``None``, the scale value is taken from the ``DataArray`` attributes.

        Equation:

            .. math::

                {norm}_{diff} = \frac{b2 - b1}{b2 + b1}

        Returns:

            ``xarray.DataArray``:

                Data range: -1 to 1
        """

        sensor = self.check_sensor(data, sensor)

        if nodata is None:
            nodata = data.gw.nodataval

        if config['scale_factor'] is not None:
            scale_factor = config['scale_factor']
        else:
            if scale_factor is None:
                scale_factor = data.gw.scaleval

        return self.norm_diff_math(
            data,
            b1,
            b2,
            'norm-diff',
            sensor,
            nodata=nodata,
            mask=mask,
            scale_factor=scale_factor,
        )

    def avi(
        self, data, nodata=None, mask=False, sensor=None, scale_factor=None
    ):

        r"""
        Calculates the advanced vegetation index

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            nodata (Optional[int or float]): A 'no data' value to fill NAs with. If ``None``, the 'no data' value is taken from the ``DataArray`` attributes.
            mask (Optional[bool]): Whether to mask the results.
            sensor (Optional[str]): The data's sensor. If ``None``, the band names should reflect the index being calculated.
            scale_factor (Optional[float]): A scale factor to apply to the data. If ``None``, the scale value is taken from the ``DataArray`` attributes.

        Equation:

            .. math::

                AVI = {(NIR \times (1.0 - red) \times (NIR - red))}^{0.3334}

        Returns:

            ``xarray.DataArray``:

                Data range: 0 to 1
        """

        sensor = self.check_sensor(data, sensor)

        if nodata is None:
            nodata = data.gw.nodataval

        if config['scale_factor'] is not None:
            scale_factor = config['scale_factor']
        else:
            if scale_factor is None:
                scale_factor = data.gw.scaleval

        return self.avi_math(
            data,
            sensor,
            data.gw.wavelengths,
            nodata=nodata,
            mask=mask,
            scale_factor=scale_factor,
        )

    def evi(
        self, data, nodata=None, mask=False, sensor=None, scale_factor=None
    ):

        r"""
        Calculates the enhanced vegetation index

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            nodata (Optional[int or float]): A 'no data' value to fill NAs with. If ``None``, the 'no data' value is taken from the ``DataArray`` attributes.
            mask (Optional[bool]): Whether to mask the results.
            sensor (Optional[str]): The data's sensor. If ``None``, the band names should reflect the index being calculated.
            scale_factor (Optional[float]): A scale factor to apply to the data. If ``None``, the scale value is taken from the ``DataArray`` attributes.

        Equation:

            .. math::

                EVI = 2.5 \times \frac{NIR - red}{NIR \times 6 \times red - 7.5 \times blue + 1}

        Returns:

            ``xarray.DataArray``:

                Data range: 0 to 1
        """

        sensor = self.check_sensor(data, sensor)

        if nodata is None:
            nodata = data.gw.nodataval

        if config['scale_factor'] is not None:
            scale_factor = config['scale_factor']
        else:
            if scale_factor is None:
                scale_factor = data.gw.scaleval

        return self.evi_math(
            data,
            sensor,
            data.gw.wavelengths,
            nodata=nodata,
            mask=mask,
            scale_factor=scale_factor,
        )

    def evi2(
        self, data, nodata=None, mask=False, sensor=None, scale_factor=None
    ):

        r"""
        Calculates the two-band modified enhanced vegetation index

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            nodata (Optional[int or float]): A 'no data' value to fill NAs with. If ``None``, the 'no data' value is taken from the ``DataArray`` attributes.
            mask (Optional[bool]): Whether to mask the results.
            sensor (Optional[str]): The data's sensor. If ``None``, the band names should reflect the index being calculated.
            scale_factor (Optional[float]): A scale factor to apply to the data. If ``None``, the scale value is taken from the ``DataArray`` attributes.

        Equation:

            .. math::

                EVI2 = 2.5 \times \frac{NIR - red}{NIR + 1 + 2.4 \times red}

        Reference:
            See :cite:`jiang_etal_2008`

        Returns:

            ``xarray.DataArray``:

                Data range: 0 to 1
        """

        sensor = self.check_sensor(data, sensor)

        if nodata is None:
            nodata = data.gw.nodataval

        if config['scale_factor'] is not None:
            scale_factor = config['scale_factor']
        else:
            if scale_factor is None:
                scale_factor = data.gw.scaleval

        return self.evi2_math(
            data,
            sensor,
            data.gw.wavelengths,
            nodata=nodata,
            mask=mask,
            scale_factor=scale_factor,
        )

    def gcvi(
        self, data, nodata=None, mask=False, sensor=None, scale_factor=None
    ):

        r"""
        Calculates the green chlorophyll vegetation index

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            nodata (Optional[int or float]): A 'no data' value to fill NAs with. If ``None``, the 'no data' value is taken from the ``DataArray`` attributes.
            mask (Optional[bool]): Whether to mask the results.
            sensor (Optional[str]): The data's sensor. If ``None``, the band names should reflect the index being calculated.
            scale_factor (Optional[float]): A scale factor to apply to the data. If ``None``, the scale value is taken from the ``DataArray`` attributes.

        Equation:

            .. math::

                GCVI = \frac{NIR}{green} - 1

        Returns:

            ``xarray.DataArray``:

                Data range: 0 to 1
        """

        sensor = self.check_sensor(data, sensor)

        if nodata is None:
            nodata = data.gw.nodataval

        if config['scale_factor'] is not None:
            scale_factor = config['scale_factor']
        else:
            if scale_factor is None:
                scale_factor = data.gw.scaleval

        return self.gcvi_math(
            data,
            sensor,
            data.gw.wavelengths,
            nodata=nodata,
            mask=mask,
            scale_factor=scale_factor,
        )

    def nbr(
        self, data, nodata=None, mask=False, sensor=None, scale_factor=None
    ):

        r"""
        Calculates the normalized burn ratio

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            nodata (Optional[int or float]): A 'no data' value to fill NAs with. If ``None``, the 'no data' value is taken from the ``DataArray`` attributes.
            mask (Optional[bool]): Whether to mask the results.
            sensor (Optional[str]): The data's sensor. If ``None``, the band names should reflect the index being calculated.
            scale_factor (Optional[float]): A scale factor to apply to the data. If ``None``, the scale value is taken from the ``DataArray`` attributes.

        Equation:

            .. math::

                NBR = \frac{NIR - SWIR2}{NIR + SWIR2}

        Returns:

            ``xarray.DataArray``:

                Data range: -1 to 1
        """

        sensor = self.check_sensor(data, sensor)

        if nodata is None:
            nodata = data.gw.nodataval

        if config['scale_factor'] is not None:
            scale_factor = config['scale_factor']
        else:
            if scale_factor is None:
                scale_factor = data.gw.scaleval

        return self.nbr_math(
            data,
            sensor,
            data.gw.wavelengths,
            nodata=nodata,
            mask=mask,
            scale_factor=scale_factor,
        )

    def ndvi(
        self, data, nodata=None, mask=False, sensor=None, scale_factor=None
    ):

        r"""
        Calculates the normalized difference vegetation index

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            nodata (Optional[int or float]): A 'no data' value to fill NAs with. If ``None``, the 'no data' value is taken from the ``DataArray`` attributes.
            mask (Optional[bool]): Whether to mask the results.
            sensor (Optional[str]): The data's sensor. If ``None``, the band names should reflect the index being calculated.
            scale_factor (Optional[float]): A scale factor to apply to the data. If ``None``, the scale value is taken from the ``DataArray`` attributes.

        Equation:

            .. math::

                NDVI = \frac{NIR - red}{NIR + red}

        Returns:

            ``xarray.DataArray``:

                Data range: -1 to 1
        """

        sensor = self.check_sensor(data, sensor)

        if nodata is None:
            nodata = data.gw.nodataval

        if config['scale_factor'] is not None:
            scale_factor = config['scale_factor']
        else:
            if scale_factor is None:
                scale_factor = data.gw.scaleval

        return self.ndvi_math(
            data,
            sensor,
            data.gw.wavelengths,
            nodata=nodata,
            mask=mask,
            scale_factor=scale_factor,
        )

    def kndvi(
        self, data, nodata=None, mask=False, sensor=None, scale_factor=None
    ):

        r"""
        Calculates the kernel normalized difference vegetation index

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            nodata (Optional[int or float]): A 'no data' value to fill NAs with. If ``None``, the 'no data' value is taken from the ``DataArray`` attributes.
            mask (Optional[bool]): Whether to mask the results.
            sensor (Optional[str]): The data's sensor. If ``None``, the band names should reflect the index being calculated.
            scale_factor (Optional[float]): A scale factor to apply to the data. If ``None``, the scale value is taken from the ``DataArray`` attributes.

        Equation:

            .. math::

                kNDVI = tanh({NDVI}^2)

        Reference:
            See :cite:`camps-valls_etal_2021`

        Returns:

            ``xarray.DataArray``:

                Data range: -1 to 1
        """

        sensor = self.check_sensor(data, sensor)

        if nodata is None:
            nodata = data.gw.nodataval

        if config['scale_factor'] is not None:
            scale_factor = config['scale_factor']
        else:
            if scale_factor is None:
                scale_factor = data.gw.scaleval

        return self.kndvi_math(
            data,
            sensor,
            data.gw.wavelengths,
            nodata=nodata,
            mask=mask,
            scale_factor=scale_factor,
        )

    def wi(
        self, data, nodata=None, mask=False, sensor=None, scale_factor=None
    ):
        r"""
        Calculates the woody vegetation index

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            nodata (Optional[int or float]): A 'no data' value to fill NAs with. If ``None``, the 'no data' value is taken from the ``DataArray`` attributes.
            mask (Optional[bool]): Whether to mask the results.
            sensor (Optional[str]): The data's sensor. If ``None``, the band names should reflect the index being calculated.
            scale_factor (Optional[float]): A scale factor to apply to the data. If ``None``, the scale value is taken from the ``DataArray`` attributes.

        Equation:

            .. math::

                WI = \Biggl \lbrace
                {
                0,\text{ if }
                   { red + SWIR1 \ge 0.5 }
                \atop
                1 - \frac{red + SWIR1}{0.5}, \text{ otherwise }
                }

        Reference:
            See :cite:`lehmann_etal_2013`

        Returns:

            ``xarray.DataArray``:

                Data range: 0 to 1
        """
        sensor = self.check_sensor(data, sensor)

        if nodata is None:
            nodata = data.gw.nodataval

        if config['scale_factor'] is not None:
            scale_factor = config['scale_factor']
        else:
            if scale_factor is None:
                scale_factor = data.gw.scaleval

        return self.wi_math(
            data,
            sensor,
            data.gw.wavelengths,
            nodata=nodata,
            mask=mask,
            scale_factor=scale_factor,
        )
