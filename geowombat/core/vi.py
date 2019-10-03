from .conversion import dask_to_datarray

import dask.array as da


class BandMath(object):

    @staticmethod
    def scale_and_assign(data, band_variable, scale_factor, names, new_names):

        attrs = data.attrs

        if band_variable == 'wavelength':
            data = data.sel(wavelength=names) * scale_factor
        else:
            data = data.sel(band=names) * scale_factor

        data = data.assign_coords(coords={band_variable: new_names})
        data = data.assign_attrs(**attrs)

        return data

    @staticmethod
    def mask_and_assign(data, result, band_variable, new_name, mask, clip_min, clip_max, scale_factor, sensor):

        """
        Masks a DataArray

        Args:
            data (DataArray or Dataset)
            result (DataArray)
            band_variable (str)
            new_name (str)
            mask (bool)
            clip_min (int)
            clip_max (int)
            scale_factor (float)
            sensor (str)

        Returns:
            ``xarray.DataArray``
        """

        if mask:

            if isinstance(data, xr.Dataset):
                result = result.where(data['mask'] < 3)
            else:

                if band_variable == 'wavelength':
                    result = result.where(data.sel(wavelength='mask') < 3)
                else:
                    result = result.where(data.sel(band='mask') < 3)

        new_attrs = data.attrs
        new_attrs['pre-scaling'] = scale_factor
        new_attrs['sensor'] = sensor
        new_attrs['drange'] = (clip_min, clip_max)

        result.clip(min=clip_min, max=clip_max)

        if 'time' in result.coords:

            if band_variable == 'wavelength':
                result = result.assign_coords(wavelength=new_name)
            else:
                result = result.assign_coords(band=new_name)

            result = result.expand_dims(dim=band_variable)

        else:
            result = result.assign_coords(coords={band_variable: new_name})

        result = result.assign_attrs(**new_attrs)

        return result

    def norm_diff_math(self, data, b1, b2, name, sensor, nodata=0, mask=False, scale_factor=1.0):

        """
        Normalized difference index --> (b2 - b1) / (b2 + b1)

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

        data = self.scale_and_assign(data, band_variable, scale_factor, [b1, b2], [b1, b2])

        if band_variable == 'wavelength':

            result = ((data.sel(wavelength=b2) - data.sel(wavelength=b1)) /
                      (data.sel(wavelength=b2) + data.sel(wavelength=b1))).fillna(nodata)

        else:

            result = ((data.sel(band=b2) - data.sel(band=b1)) /
                      (data.sel(band=b2) + data.sel(band=b1))).fillna(nodata)

        return self.mask_and_assign(data, result, band_variable, name, mask, -1, 1, scale_factor, sensor)

    def evi_math(self, data, sensor, wavelengths, nodata=0, mask=False, scale_factor=1.0):

        """
        Enhanced vegetation index

        Returns:
            ``xarray.DataArray``
        """

        l = 1.0
        c1 = 6.0
        c2 = 7.5
        g = 2.5

        band_variable = 'wavelength' if 'wavelength' in data.coords else 'band'

        if 'nir' in data.coords[band_variable].values.tolist():
            nir = 'nir'
            red = 'red'
            blue = 'blue'
        else:
            nir = wavelengths[sensor].nir
            red = wavelengths[sensor].red
            blue = wavelengths[sensor].blue

        data = self.scale_and_assign(data, band_variable, scale_factor, [nir, red, blue], ['nir', 'red', 'blue'])

        if band_variable == 'wavelength':

            result = (g * (data.sel(wavelength='nir') - data.sel(wavelength='red')) /
                      (data.sel(wavelength='nir') * c1 * data.sel(wavelength='red') - c2 * data.sel(wavelength='blue') + l)).fillna(nodata)

        else:

            result = (g * (data.sel(band='nir') - data.sel(band='red')) /
                      (data.sel(band='nir') * c1 * data.sel(band='red') - c2 * data.sel(band='blue') + l)).fillna(nodata)

        return self.mask_and_assign(data, result, band_variable, 'evi', mask, 0, 1, scale_factor, sensor)

    def evi2_math(self, data, sensor, wavelengths, nodata=0, mask=False, scale_factor=1.0):

        """
        Two-band enhanced vegetation index

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

        data = self.scale_and_assign(data, band_variable, scale_factor, [nir, red], ['nir', 'red'])

        if band_variable == 'wavelength':

            result = (2.5 * ((data.sel(wavelength='nir') - data.sel(wavelength='red')) /
                             (data.sel(wavelength='nir') + 1.0 + (2.4 * (data.sel(wavelength='red')))))).fillna(nodata)

        else:

            result = (2.5 * ((data.sel(band='nir') - data.sel(band='red')) /
                             (data.sel(band='nir') + 1.0 + (2.4 * (data.sel(band='red')))))).fillna(nodata)

        return self.mask_and_assign(data, result, band_variable, 'evi2', mask, 0, 1, scale_factor, sensor)

    def nbr_math(self, data, sensor, wavelengths, nodata=0, mask=False, scale_factor=1.0):

        """
        Normalized burn ratio

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

        return self.norm_diff_math(data, swir2, nir, 'nbr', sensor, nodata=nodata, mask=mask, scale_factor=scale_factor)

    def ndvi_math(self, data, sensor, wavelengths, nodata=0, mask=False, scale_factor=1.0):

        """
        Normalized difference vegetation index

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

        return self.norm_diff_math(data, red, nir, 'ndvi', sensor, nodata=nodata, mask=mask, scale_factor=scale_factor)

    def wi_math(self, data, sensor, wavelengths, nodata=0, mask=False, scale_factor=1.0):

        """
        Woody index

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

        data = self.scale_and_assign(data, band_variable, scale_factor, [swir1, red], ['swir1', 'red'])

        if band_variable == 'wavelength':
            result = data.sel(wavelength='swir1') + data.sel(wavelength='red')
        else:
            result = data.sel(band='swir1') + data.sel(band='red')

        result = result.where(result > 0.5, 0, 1.0 - (result / 0.5)).fillna(nodata)

        return self.mask_and_assign(data, result, band_variable, 'wi', mask, 0, 1, scale_factor, sensor)


class VegetationIndices(BandMath):

    def norm_diff(self, data, b1, b2, sensor=None, nodata=0, mask=False, scale_factor=1.0):

        r"""
        Calculates the normalized difference band ratio

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            b1 (str): The band name of the first band.
            b2 (str): The band name of the second band.
            sensor (Optional[str]): sensor (Optional[str]): The data's sensor.
            nodata (Optional[int or float]): A 'no data' value to fill NAs with.
            mask (Optional[bool]): Whether to mask the results.
            scale_factor (Optional[float]): A scale factor to apply to the data.

        Equation:

            .. math::
                {norm}_{diff} = \frac{b2 - b1}{b2 + b1}

        Returns:
            ``xarray.DataArray``
        """

        if not sensor:
            sensor = data.gw.sensor

        if scale_factor == 1.0:
            scale_factor = data.gw.scale_factor

        return self.norm_diff_math(data, b1, b2, 'norm-diff', sensor, nodata=nodata, mask=mask, scale_factor=scale_factor)

    def evi(self, data, nodata=0, mask=False, sensor=None, scale_factor=1.0):

        r"""
        Calculates the enhanced vegetation index

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            nodata (Optional[int or float]): A 'no data' value to fill NAs with.
            mask (Optional[bool]): Whether to mask the results.
            sensor (Optional[str]): The data's sensor.
            scale_factor (Optional[float]): A scale factor to apply to the data.

        Equation:

            .. math::

                EVI = 2.5 \times \frac{NIR - red}{NIR \times 6 \times red - 7.5 \times blue + 1}

        Returns:
            ``xarray.DataArray``
        """

        if not sensor:
            sensor = data.gw.sensor

        if scale_factor == 1.0:
            scale_factor = data.gw.scale_factor

        return self.evi_math(data, sensor, data.gw.wavelengths, nodata=nodata, mask=mask, scale_factor=scale_factor)

    def evi2(self, data, nodata=0, mask=False, sensor=None, scale_factor=1.0):

        r"""
        Calculates the two-band modified enhanced vegetation index

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            nodata (Optional[int or float]): A 'no data' value to fill NAs with.
            mask (Optional[bool]): Whether to mask the results.
            sensor (Optional[str]): The data's sensor.
            scale_factor (Optional[float]): A scale factor to apply to the data.

        Equation:

            .. math::

                EVI2 = 2.5 \times \frac{NIR - red}{NIR + 1 + 2.4 \times red}

        Returns:
            ``xarray.DataArray``
        """

        if not sensor:
            sensor = data.gw.sensor

        if scale_factor == 1.0:
            scale_factor = data.gw.scale_factor

        return self.evi2_math(data, sensor, data.gw.wavelengths, nodata=nodata, mask=mask, scale_factor=scale_factor)

    def nbr(self, data, nodata=0, mask=False, sensor=None, scale_factor=1.0):

        r"""
        Calculates the normalized burn ratio

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            nodata (Optional[int or float]): A 'no data' value to fill NAs with.
            mask (Optional[bool]): Whether to mask the results.
            sensor (Optional[str]): The data's sensor.
            scale_factor (Optional[float]): A scale factor to apply to the data.

        Equation:

            .. math::
                NBR = \frac{NIR - SWIR1}{NIR + SWIR1}

        Returns:
            ``xarray.DataArray``
        """

        if not sensor:
            sensor = data.gw.sensor

        if scale_factor == 1.0:
            scale_factor = data.gw.scale_factor

        return self.nbr_math(data, sensor, data.gw.wavelengths, nodata=nodata, mask=mask, scale_factor=scale_factor)

    def ndvi(self, data, nodata=0, mask=False, sensor=None, scale_factor=1.0):

        r"""
        Calculates the normalized difference vegetation index

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            nodata (Optional[int or float]): A 'no data' value to fill NAs with.
            mask (Optional[bool]): Whether to mask the results.
            sensor (Optional[str]): The data's sensor.
            scale_factor (Optional[float]): A scale factor to apply to the data.

        Equation:

            .. math::
                NDVI = \frac{NIR - red}{NIR + red}

        Returns:
            ``xarray.DataArray``
        """

        if not sensor:
            sensor = data.gw.sensor

        if scale_factor == 1.0:
            scale_factor = data.gw.scale_factor

        return self.ndvi_math(data, sensor, data.gw.wavelengths, nodata=nodata, mask=mask, scale_factor=scale_factor)

    def wi(self, data, nodata=0, mask=False, sensor=None, scale_factor=1.0):

        r"""
        Calculates the woody vegetation index

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            nodata (Optional[int or float]): A 'no data' value to fill NAs with.
            mask (Optional[bool]): Whether to mask the results.
            sensor (Optional[str]): The data's sensor.
            scale_factor (Optional[float]): A scale factor to apply to the data.

        Equation:

            .. math::
                WI = SWIR1 + red

        Returns:
            ``xarray.DataArray``
        """

        if not sensor:
            sensor = data.gw.sensor

        if scale_factor == 1.0:
            scale_factor = data.gw.scale_factor

        return self.wi_math(data, sensor, data.gw.wavelengths, nodata=nodata, mask=mask, scale_factor=scale_factor)
