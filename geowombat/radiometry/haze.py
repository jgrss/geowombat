import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression


class Haze(object):

    @staticmethod
    def get_hot(red, blue, slope_theta):
        """Haze Optimized Transform"""
        return xr.ufuncs.sin(slope_theta) * blue - xr.ufuncs.cos(slope_theta) * red

    def remove_haze(self,
                    data,
                    method='hot',
                    red_perc=10,
                    thresh1=0.01,
                    thresh2=0.1,
                    min_samples=100,
                    n_iters=40,
                    n_jobs=-1):

        """
        Removes haze

        Args:
            data (2d or 3d DataArray): The data to normalize, in the range 0-1.
            method (Optional[str])
            red_perc (Optional[int])
            thresh1 (Optional[float])
            thresh2 (Optional[float])
            min_samples (Optional[int])
            n_iters (Optional[int])
            n_jobs (Optional[int])

        Returns:
            ``xarray.DataArray``
        """

        yhat, haze_found = self.fit(data, red_perc, thresh1, thresh2, min_samples, n_iters, n_jobs)

        if haze_found:
            return self.transform(data, yhat)
        else:
            return data

    def fit(self, data, red_perc, thresh1, thresh2, min_samples, n_iters, n_jobs):

        lin = LinearRegression(n_jobs=n_jobs)

        last_slope = 1e9
        params = {}

        red = data.sel(band='red').data.compute().flatten()
        blue = data.sel(band='blue').data.compute().flatten()

        red_p = np.percentile(red, red_perc)

        for i in range(0, n_iters):

            lin.fit(red[:, np.newaxis], blue)

            if i == 0:
                params['haze'] = [lin.coef_[0], lin.intercept_]

            if abs(lin.coef_[0] - last_slope) < thresh1:
                break

            last_slope = lin.coef_[0]

            yhat = lin.predict(red[:, np.newaxis])

            idx = np.where(((blue - yhat) < 0) | (red < red_p))[0]

            if idx.shape[0] < min_samples:
                break

            blue = blue[idx]
            red = red[idx]

        params['clear'] = [lin.coef_[0], lin.intercept_]

        if abs(params['haze'][0] - params['clear'][0]) < thresh2:
            haze_found = False
        else:
            haze_found = True

        return np.array([0, 1]) * params['clear'][0] + params['clear'][1], haze_found

    def transform(self, data, yhat):

        attrs = data.attrs.copy()

        # Get the HOT
        src_hot = self.get_hot(data.sel(band='blue'),
                               data.sel(band='red'),
                               np.tan(yhat[1]))

        # Apply the HOT
        src_adj = (data + src_hot).clip(0, 1)

        src_adj.attrs = attrs

        return src_adj
