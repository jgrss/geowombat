import numpy as np
from osgeo import gdal, gdal_array
import dask
import dask.array as da
import xarray as xr
from sklearn.linear_model import LinearRegression


class Topo(object):

    """
    A class for topographic normalization
    """

    @staticmethod
    def calc_slope(elev, **kwargs):

        """
        Calculates slope from elevation

        Args:
            elev (2d array): The elevation data.
            kwargs (Optional[dict]): Keyword arguments passed to ``gdal.DEMProcessingOptions``.

        Returns:
            ``numpy.ndarray``
        """

        ds = gdal_array.OpenArray(elev.astype('float64'))

        slope_options = gdal.DEMProcessingOptions(**kwargs)

        out_ds = gdal.DEMProcessing('', ds, 'slope', options=slope_options)

        dst_array = out_ds.GetRasterBand(1).ReadAsArray()

        ds = None
        out_ds = None

        return np.float64(dst_array)

    @staticmethod
    def calc_aspect(elev, **kwargs):

        """
        Calculates aspect from elevation

        Args:
            elev (2d array): The elevation data.
            kwargs (Optional[dict]): Keyword arguments passed to ``gdal.DEMProcessingOptions``.

        Returns:
            ``numpy.ndarray``
        """

        ds = gdal_array.OpenArray(elev.astype('float64'))

        aspect_options = gdal.DEMProcessingOptions(**kwargs)

        out_ds = gdal.DEMProcessing('', ds, 'aspect', options=aspect_options)

        dst_array = out_ds.GetRasterBand(1).ReadAsArray()

        ds = None
        out_ds = None

        return np.float64(dst_array)

    def _method_c(self, sr, il, cos_z, nodata_samps, n_jobs=1):

        r"""
        Normalizes terrain using the C-correction method

        Args:
            sr (Dask Array): The surface reflectance data.
            il (Dask Array): The solar illumination.
            cos_z (Dask Array): The cosine of the solar zenith angle.
            nodata_samps (Dask Array): Samples where 1='no data' and 0='valid data'.
            n_jobs (Optional[int]): The number of parallel workers for ``LinearRegression.fit``.

        References:

            See :cite:`teillet_etal_1982` for the C-correction method.

        Returns:
            ``dask.array``
        """

        nodata = nodata_samps.compute().flatten()
        idx = np.where(nodata == 0)[0]

        X = il.compute().flatten()[idx][:, np.newaxis]
        y = sr.compute().flatten()[idx]

        model = LinearRegression(n_jobs=n_jobs)

        model.fit(X, y)

        slope_m = model.coef_[0]
        intercept_b = model.intercept_

        c = intercept_b / slope_m

        # Get the A-factor
        a_factor = (cos_z + c) / (il + c)

        a_factor = da.where(da.isnan(a_factor), 1, a_factor)

        sr_a = sr * a_factor

        return da.where((sr_a > 1) | (nodata_samps == 1), sr, sr_a).clip(0, 1)

    @staticmethod
    def norm_topo(data,
                  elev_data,
                  solar_za,
                  solar_az,
                  method='c',
                  angle_scale=0.01,
                  n_jobs=1):

        """
        Applies topographic normalization

        Args:
            data (2d or 3d DataArray): The data to normalize, in the range 0-1.
            solar_za (2d DataArray): The solar zenith angles (degrees).
            solar_az (2d DataArray): The solar azimuth angles (degrees).
            method (Optional[str]): The method to apply. Choices are ['c'].
            angle_scale (Optional[float]): The angle scale factor.
            n_jobs (Optional[int]): The number of parallel workers for ``LinearRegression.fit``.

        References:

            See :cite:`teillet_etal_1982` for the C-correction method.

        Returns:
            ``xarray.DataArray``

        Examples:
            >>> import geowombat as gw
            >>>
            >>> # Example where pixel angles are stored in separate GeoTiff files
            >>> with gw.config.update(sensor='l7', scale_factor=0.0001, nodata=0):
            >>>
            >>>     with gw.open('landsat.tif') as src,
            >>>         gw.open('srtm') as elev,
            >>>             gw.open('solarz.tif') as solarz,
            >>>                 gw.open('solara.tif') as solara:
            >>>
            >>>         src_norm = gw.norm_topo(src, solarz, solara)
        """

        calc_slope_d = dask.delayed(calc_slope)
        calc_aspect_d = dask.delayed(calc_aspect)

        slope_deg = calc_slope_d(elev_data.squeeze().data,
                                 format='MEM',
                                 computeEdges=True,
                                 alg='ZevenbergenThorne',
                                 slopeFormat='degree')

        aspect_deg = calc_aspect_d(elev_data.squeeze().data,
                                   format='MEM',
                                   computeEdges=True,
                                   alg='ZevenbergenThorne',
                                   trigonometric=False,
                                   zeroForFlat=True)

        slope_deg_fd = da.from_delayed(slope_deg, (data.gw.nrows, data.gw.ncols), dtype='float64')
        aspect_deg_fd = da.from_delayed(aspect_deg, (data.gw.nrows, data.gw.ncols), dtype='float64')

        nodata_samps = da.where((elev_data.data == -32768) | (slope_deg_fd < 2), 1, 0)

        # valid_samples = da.where((slopefd != srtm_nodata) & (slopefd > slope_thresh))

        slope_rad = da.deg2rad(slope_deg_fd)
        aspect_rad = da.deg2rad(aspect_deg_fd)

        # Convert degrees to radians
        solar_za = da.deg2rad(solar_za.squeeze().data * angle_scale)
        solar_az = da.deg2rad(solar_az.squeeze().data * angle_scale)

        cos_z = da.cos(solar_za)

        # Calculate the illumination angle
        il = da.cos(slope_rad) * cos_z + da.sin(slope_rad) * da.sin(solar_za) * da.cos(solar_az - aspect_rad)

        sr_adj = list()
        for band in data.band.values.tolist():
            
            # TODO: add other methods
            sr_adj.append(self._method_c(data.sel(band=band).data,
                                         il,
                                         cos_z,
                                         nodata_samps,
                                         n_jobs=n_jobs))

        return xr.DataArray(data=da.concatenate(sr_adj).reshape((data.gw.nbands,
                                                                 data.gw.nrows,
                                                                 data.gw.ncols)),
                            coords={'band': data.band.values.tolist(),
                                    'y': data.y.values,
                                    'x': data.x.values},
                            dims=('band', 'y', 'x'),
                            attrs=data.attrs)
