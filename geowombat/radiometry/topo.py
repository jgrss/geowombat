from ..errors import logger

import numpy as np
from osgeo import gdal, gdal_array
import cv2
import dask
import dask.array as da
import xarray as xr
from sklearn.linear_model import LinearRegression, TheilSenRegressor


def calc_slope(elev, proc_dims=None, w=None, **kwargs):

    """
    Calculates slope from elevation

    Args:
        elev (2d array): The elevation data.
        proc_dims (Optional[tuple]): Dimensions to resize to.
        w (Optional[int]): The smoothing window size when ``proc_dims`` is given.
        kwargs (Optional[dict]): Keyword arguments passed to ``gdal.DEMProcessingOptions``.

    Returns:
        ``numpy.ndarray``
    """

    if proc_dims:

        inrows, incols = elev.shape

        elev = cv2.resize(elev.astype('float32'),
                          proc_dims,
                          interpolation=cv2.INTER_LINEAR)

    ds = gdal_array.OpenArray(elev.astype('float64'))

    slope_options = gdal.DEMProcessingOptions(**kwargs)

    out_ds = gdal.DEMProcessing('', ds, 'slope', options=slope_options)

    dst_array = out_ds.GetRasterBand(1).ReadAsArray()

    ds = None
    out_ds = None

    if proc_dims:

        dst_array = cv2.resize(dst_array.astype('float32'),
                               (incols, inrows),
                               interpolation=cv2.INTER_LINEAR)

        if not w:
            w = 15

        return np.float64(cv2.bilateralFilter(np.float32(dst_array), w, 10, 10))

    else:
        return np.float64(dst_array)


def calc_aspect(elev, proc_dims=None, w=None, **kwargs):

    """
    Calculates aspect from elevation

    Args:
        elev (2d array): The elevation data.
        proc_dims (Optional[tuple]): Dimensions to resize to.
        w (Optional[int]): The smoothing window size when ``proc_dims`` is given.
        kwargs (Optional[dict]): Keyword arguments passed to ``gdal.DEMProcessingOptions``.

    Returns:
        ``numpy.ndarray``
    """

    if proc_dims:

        inrows, incols = elev.shape

        elev = cv2.resize(elev.astype('float32'),
                          proc_dims,
                          interpolation=cv2.INTER_LINEAR)

    ds = gdal_array.OpenArray(elev.astype('float64'))

    aspect_options = gdal.DEMProcessingOptions(**kwargs)

    out_ds = gdal.DEMProcessing('', ds, 'aspect', options=aspect_options)

    dst_array = out_ds.GetRasterBand(1).ReadAsArray()

    ds = None
    out_ds = None

    if proc_dims:

        dst_array = cv2.resize(dst_array.astype('float32'),
                               (incols, inrows),
                               interpolation=cv2.INTER_LINEAR)

        if not w:
            w = 15

        return np.float64(cv2.bilateralFilter(np.float32(dst_array), w, 10, 10))

    else:
        return np.float64(dst_array)


calc_slope_delayed = dask.delayed(calc_slope)
calc_aspect_delayed = dask.delayed(calc_aspect)


class Topo(object):

    """
    A class for topographic normalization
    """

    def _regress_a(self, X, y, robust, n_jobs):

        """
        Calculates the slope and intercept
        """

        if robust:
            model = TheilSenRegressor(n_jobs=n_jobs)
        else:
            model = LinearRegression(n_jobs=n_jobs)

        model.fit(X, y)

        slope_m = model.coef_[0]
        intercept_b = model.intercept_

        return slope_m, intercept_b

    def _method_empirical_rotation(self, sr, il, cos_z, nodata_samps, n_jobs=1, robust=False):

        r"""
        Normalizes terrain using the Empirical Rotation method

        Args:
            sr (Dask Array): The surface reflectance data.
            il (Dask Array): The solar illumination.
            cos_z (Dask Array): The cosine of the solar zenith angle.
            nodata_samps (Dask Array): Samples where 1='no data' and 0='valid data'.
            n_jobs (Optional[int]): The number of parallel workers for ``LinearRegression.fit`` or
                ``TheilSenRegressor.fit``.
            robust (Optional[bool]): Whether to fit a robust regression.

        References:

            See :cite:`tan_etal_2010` for the Empirical Rotation method.

        Returns:
            ``dask.array``
        """

        nodata = nodata_samps.compute().flatten()
        idx = np.where(nodata == 0)[0]

        X = il.compute().flatten()[idx][:, np.newaxis]
        y = sr.compute().flatten()[idx]

        slope_m, intercept_b = self._regress_a(X, y, robust, n_jobs)

        # https://reader.elsevier.com/reader/sd/pii/S0034425713001673?token=6C93FB2E69ABF5729CE9ECBBDFD9C2D985613156753C822A3D160102D46135E01457EE33500DB4648C6AF636F39D8B62
        # Improved forest change detection with terrain illumination corrected Landsat images
        sr_a = sr - slope_m * (il - cos_z)
        # sr_a = sr - (slope_m * il + intercept_b)

        return da.where(nodata_samps == 1, sr, sr_a).clip(0, 1)

    def _method_cos(self, sr, il, cos_z, nodata_samps):

        r"""
        Normalizes terrain using the Cosine method

        Args:
            sr (Dask Array): The surface reflectance data.
            il (Dask Array): The solar illumination.
            cos_z (Dask Array): The cosine of the solar zenith angle.
            nodata_samps (Dask Array): Samples where 1='no data' and 0='valid data'.

        References:

            See :cite:`teillet_etal_1982` for the C-correction method.

        Returns:
            ``dask.array``
        """

        sr_a = sr * (cos_z / il)

        return da.where(nodata_samps == 1, sr, sr_a).clip(0, 1)

    def _method_c(self, sr, il, cos_z, nodata_samps, n_jobs=1, robust=False):

        r"""
        Normalizes terrain using the C-correction method

        Args:
            sr (Dask Array): The surface reflectance data.
            il (Dask Array): The solar illumination.
            cos_z (Dask Array): The cosine of the solar zenith angle.
            nodata_samps (Dask Array): Samples where 1='no data' and 0='valid data'.
            n_jobs (Optional[int]): The number of parallel workers for ``LinearRegression.fit`` or
                ``TheilSenRegressor.fit``.
            robust (Optional[bool]): Whether to fit a robust regression.

        References:

            See :cite:`teillet_etal_1982` for the C-correction method.

        Returns:
            ``dask.array``
        """

        nodata = nodata_samps.compute().flatten()
        idx = np.where(nodata == 0)[0]

        X = il.compute().flatten()[idx][:, np.newaxis]
        y = sr.compute().flatten()[idx]

        slope_m, intercept_b = self._regress_a(X, y, robust, n_jobs)

        c = intercept_b / slope_m

        # Get the A-factor
        a_factor = (cos_z + c) / (il + c)

        a_factor = da.where(da.isnan(a_factor), 1, a_factor)

        sr_a = sr * a_factor

        return da.where((sr_a > 1) | (nodata_samps == 1), sr, sr_a).clip(0, 1)

    def norm_topo(self,
                  data,
                  elev,
                  solar_za,
                  solar_az,
                  slope=None,
                  aspect=None,
                  method='empirical-rotation',
                  slope_thresh=2,
                  nodata=0,
                  elev_nodata=-32768,
                  scale_factor=1,
                  angle_scale=0.01,
                  n_jobs=1,
                  robust=False,
                  slope_kwargs=None,
                  aspect_kwargs=None):

        """
        Applies topographic normalization

        Args:
            data (2d or 3d DataArray): The data to normalize, in the range 0-1.
            elev (2d DataArray): The elevation data.
            solar_za (2d DataArray): The solar zenith angles (degrees).
            solar_az (2d DataArray): The solar azimuth angles (degrees).
            slope (2d DataArray): The slope data. If not given, slope is calculated from ``elev``.
            aspect (2d DataArray): The aspect data. If not given, aspect is calculated from ``elev``.
            method (Optional[str]): The method to apply. Choices are ['c', 'empirical-rotation'].
            slope_thresh (Optional[float or int]): The slope threshold. Any samples with
                values < ``slope_thresh`` are not adjusted.
            nodata (Optional[int or float]): The 'no data' value for ``data``.
            elev_nodata (Optional[float or int]): The 'no data' value for ``elev``.
            scale_factor (Optional[float]): A scale factor to apply to the input data.
            angle_scale (Optional[float]): The angle scale factor.
            n_jobs (Optional[int]): The number of parallel workers for ``LinearRegression.fit``.
            robust (Optional[bool]): Whether to fit a robust regression.
            slope_kwargs (Optional[dict]): Keyword arguments passed to ``gdal.DEMProcessingOptions``
                to calculate the slope.
            aspect_kwargs (Optional[dict]): Keyword arguments passed to ``gdal.DEMProcessingOptions``
                to calculate the aspect.

        References:

            See :cite:`teillet_etal_1982` for the C-correction method.
            See :cite:`tan_etal_2010` for the Empirical Rotation method.

        Returns:
            ``xarray.DataArray``

        Examples:
            >>> import geowombat as gw
            >>> from geowombat.radiometry import Topo
            >>>
            >>> topo = Topo()
            >>>
            >>> # Example where pixel angles are stored in separate GeoTiff files
            >>> with gw.config.update(sensor='l7', scale_factor=0.0001, nodata=0):
            >>>
            >>>     with gw.open('landsat.tif') as src,
            >>>         gw.open('srtm') as elev,
            >>>             gw.open('solarz.tif') as solarz,
            >>>                 gw.open('solara.tif') as solara:
            >>>
            >>>         src_norm = topo.norm_topo(src, elev, solarz, solara, n_jobs=-1)
        """

        method = method.strip().lower()

        if method not in ['c', 'empirical-rotation']:

            logger.exception("  Currently, the only supported methods are 'c' and 'empirical-rotation'.")
            raise NameError

        attrs = data.attrs.copy()

        if not nodata:
            nodata = data.gw.nodata

        if scale_factor == 1.0:
            scale_factor = data.gw.scale_factor

        # Scale the reflectance data
        if scale_factor != 1:
            data = data * scale_factor

        if not slope_kwargs:

            slope_kwargs = dict(format='MEM',
                                computeEdges=True,
                                alg='ZevenbergenThorne',
                                slopeFormat='degree')

        if not aspect_kwargs:

            aspect_kwargs = dict(format='MEM',
                                 computeEdges=True,
                                 alg='ZevenbergenThorne',
                                 trigonometric=False,
                                 zeroForFlat=True)

        slope_kwargs['format'] = 'MEM'
        slope_kwargs['slopeFormat'] = 'degree'
        aspect_kwargs['format'] = 'MEM'

        # Force to SRTM resolution
        proc_dims = (int((data.gw.ncols*data.gw.cellx) / 30.0),
                     int((data.gw.nrows*data.gw.celly) / 30.0))

        w = int((5 * 30.0) / data.gw.celly)

        if w % 2 == 0:
            w += 1

        if isinstance(slope, xr.DataArray):
            slope_deg_fd = slope.squeeze().data
        else:

            slope_deg = calc_slope_delayed(elev.squeeze().data, proc_dims=proc_dims, w=w, **slope_kwargs)
            slope_deg_fd = da.from_delayed(slope_deg, (data.gw.nrows, data.gw.ncols), dtype='float64')

        if isinstance(aspect, xr.DataArray):
            aspect_deg_fd = aspect.squeeze().data
        else:

            aspect_deg = calc_aspect_delayed(elev.squeeze().data, proc_dims=proc_dims, w=w, **aspect_kwargs)
            aspect_deg_fd = da.from_delayed(aspect_deg, (data.gw.nrows, data.gw.ncols), dtype='float64')

        nodata_samps = da.where((elev.data == elev_nodata) |
                                (data.max(dim='band').data == nodata) |
                                (slope_deg_fd < slope_thresh), 1, 0)

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

            if method == 'c':

                sr_adj.append(self._method_c(data.sel(band=band).data,
                                             il,
                                             cos_z,
                                             nodata_samps,
                                             n_jobs=n_jobs,
                                             robust=robust))

            else:

                sr_adj.append(self._method_empirical_rotation(data.sel(band=band).data,
                                                              il,
                                                              cos_z,
                                                              nodata_samps,
                                                              n_jobs=n_jobs,
                                                              robust=robust))

        adj_data = xr.DataArray(data=da.concatenate(sr_adj).reshape((data.gw.nbands,
                                                                     data.gw.nrows,
                                                                     data.gw.ncols)),
                                coords={'band': data.band.values.tolist(),
                                        'y': data.y.values,
                                        'x': data.x.values},
                                dims=('band', 'y', 'x'),
                                attrs=data.attrs)

        attrs['calibration'] = 'Topographic-adjusted'
        attrs['nodata'] = nodata
        attrs['drange'] = (0, 1)

        adj_data.attrs = attrs

        return adj_data
