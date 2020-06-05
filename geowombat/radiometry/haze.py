import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression


class Haze(object):

    @staticmethod
    def get_hot(red, blue, slope_theta):
        """Haze Optimized Transform"""
        return xr.ufuncs.sin(slope_theta) * blue - xr.ufuncs.cos(slope_theta) * red

    @staticmethod
    def remove_haze(data,
                    method='hot',
                    thresh=0.01,
                    n_jobs=-1):

        """
        Removes haze

        Args:
            data (2d or 3d DataArray): The data to normalize, in the range 0-1.
            method (Optional[str])
            thresh (Optional[float])
            n_jobs (Optional[int])

        Returns:
            ``xarray.DataArray``
        """

        lin = LinearRegression(n_jobs=n_jobs)

        last_slope = 1e9
        params = {}

        red = data.sel(band='red').data.compute().flatten()
        blue = data.sel(band='blue').data.compute().flatten()

        red_10p = np.percentile(red, 10)

        for i in range(0, 40):

            lin.fit(red[:, np.newaxis], blue)

            if i == 0:
                params['haze'] = [lin.coef_[0], lin.intercept_]

            if abs(lin.coef_[0] - last_slope) < thresh:
                break

            last_slope = lin.coef_[0]

            yhat = lin.predict(red[:, np.newaxis])

            idx = np.where(((blue - yhat) < 0) | (red < red_10p))[0]

            if idx.shape[0] < 100:
                break

            blue = blue[idx]
            red = red[idx]

            alpha -= 0.05

            if alpha < 0.1:
                alpha = 0.1

        params['clear'] = [lin.coef_[0], lin.intercept_]

        yhat = np.array([0, 1]) * params['haze'][0] + params['haze'][1]

        yhat = np.array([0, 1]) * params['clear'][0] + params['clear'][1]


