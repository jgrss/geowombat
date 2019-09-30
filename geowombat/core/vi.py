from .conversion import dask_to_datarray

import dask.array as da


class BandMath(object):

    @staticmethod
    def norm_diff_math(data, b1, b2, name, mask=False):

        """
        Normalized difference index --> (b2 - b1) / (b2 + b1)

        Args:
            data (DataArray)
            b1 (DataArray): Band 1
            b2 (DataArray): Band 2
            name (str)
            mask (Optional[bool])
        """

        result = ((b2 - b1) / (b2 + b1)).fillna(0)

        if mask:
            result = result.where(data['mask'] < 3)

        result = da.where(result < -1, -1, result)
        result = da.where(result > 1, 1, result)

        return dask_to_datarray(data, result, name)

    @staticmethod
    def evi_math(data, sensor, wavelengths, mask=False):

        """
        Enhanced vegetation index
        """

        l = 1.0
        c1 = 6.0
        c2 = 7.5
        g = 2.5

        nir = wavelengths[sensor].nir
        red = wavelengths[sensor].red
        blue = wavelengths[sensor].blue

        result = (g * (data.sel(wavelength=nir) - data.sel(wavelength=red)) /
                  (data.sel(wavelength=nir) * c1 * data.sel(wavelength=red) - c2 * data.sel(wavelength=blue) + l)).fillna(0)

        if mask:
            result = result.where(data['mask'] < 3)

        result = da.where(result < 0, 0, result)
        result = da.where(result > 1, 1, result)

        return dask_to_datarray(data, result, 'evi2')

    @staticmethod
    def evi2_math(data, sensor, wavelengths, mask=False):

        """
        Two-band enhanced vegetation index
        """

        nir = wavelengths[sensor].nir
        red = wavelengths[sensor].red

        result = (2.5 * ((data.sel(wavelength=nir) - data.sel(wavelength=red)) /
                         (data.sel(wavelength=nir) + 1.0 + (2.4 * (data.sel(wavelength=red)))))).fillna(0)

        if mask:
            result = result.where(data['mask'] < 3)

        result = da.where(result < 0, 0, result)
        result = da.where(result > 1, 1, result)

        return dask_to_datarray(data, result, 'evi2')

    def nbr_math(self, data, sensor, wavelengths, mask=False):

        """
        Normalized burn ratio
        """

        nir = wavelengths[sensor].nir
        swir2 = wavelengths[sensor].swir2

        return self.norm_diff_math(data, swir2, nir, 'nbr', mask=mask)

    def ndvi_math(self, data, sensor, wavelengths, mask=False):

        """
        Normalized difference vegetation index
        """

        nir = wavelengths[sensor].nir
        red = wavelengths[sensor].red

        return self.norm_diff_math(data, red, nir, 'ndvi', mask=mask)

    @staticmethod
    def wi_math(data, sensor, wavelengths, mask=False):

        """
        Woody index
        """

        swir1 = wavelengths[sensor].swir1
        red = wavelengths[sensor].red

        result = da.where((data.sel(wavelength=swir1) + data.sel(wavelength=red)) > 0.5, 0,
                          1.0 - ((data.sel(wavelength=swir1) + data.sel(wavelength=red)) / 0.5))

        if mask:
            result = result.where(data['mask'] < 3)

        result = da.where(result < 0, 0, result)
        result = da.where(result > 1, 1, result)

        return dask_to_datarray(data, result, 'wi')


class VegetationIndices(BandMath):

    def norm_diff(self, data, b1, b2, mask=False):

        """
        Calculates the normalized difference band ratio

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            b1 (str): The band name of the first band.
            b2 (str): The band name of the second band.
            mask (Optional[bool]): Whether to mask the results.

        Returns:
            ``xarray.DataArray``
        """

        return self.norm_diff_math(data, b1, b2, 'norm-diff', mask=mask)

    def evi(self, data, mask=False):

        """
        Calculates the enhanced vegetation index

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            mask (Optional[bool]): Whether to mask the results.

        Returns:
            ``xarray.DataArray``
        """

        return self.evi_math(data, data.gw.sensor, data.gw.wavelengths, mask=mask)

    def evi2(self, data, mask=False):

        """
        Calculates the two-band modified enhanced vegetation index

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            mask (Optional[bool]): Whether to mask the results.

        Returns:
            ``xarray.DataArray``
        """

        return self.evi2_math(data, data.gw.sensor, data.gw.wavelengths, mask=mask)

    def nbr(self, data, mask=False):

        """
        Calculates the normalized burn ratio

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            mask (Optional[bool]): Whether to mask the results.

        Returns:
            ``xarray.DataArray``
        """

        return self.nbr_math(data, data.gw.sensor, data.gw.wavelengths, mask=mask)

    def ndvi(self, data, mask=False):

        """
        Calculates the normalized difference vegetation index

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            mask (Optional[bool]): Whether to mask the results.

        Returns:
            ``xarray.DataArray``
        """

        return self.ndvi_math(data, data.gw.sensor, data.gw.wavelengths, mask=mask)

    def wi(self, data, mask=False):

        """
        Calculates the woody vegetation index

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            mask (Optional[bool]): Whether to mask the results.

        Returns:
            ``xarray.DataArray``
        """

        return self.wi_math(data, data.gw.sensor, data.gw.wavelengths, mask=mask)
