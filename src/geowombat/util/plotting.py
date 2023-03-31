import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr

from ..handler import add_handler

logger = logging.getLogger(__name__)
logger = add_handler(logger)


class Plotting(object):
    @staticmethod
    def imshow(
        data,
        mask=False,
        nodata=0,
        flip=False,
        text_color='black',
        rot=30,
        **kwargs
    ):

        """Shows an image on a plot.

        Args:
            data (``xarray.DataArray`` or ``xarray.Dataset``): The data to plot.
            mask (Optional[bool]): Whether to mask 'no data' values (given by ``nodata``).
            nodata (Optional[int or float]): The 'no data' value.
            flip (Optional[bool]): Whether to flip an RGB array's band order.
            text_color (Optional[str]): The text color.
            rot (Optional[int]): The degree rotation for the x-axis tick labels.
            kwargs (Optional[dict]): Keyword arguments passed to ``xarray.plot.imshow``.

        Returns:
            ``matplotlib`` axis object

        Examples:
            >>> import geowombat as gw
            >>>
            >>> # Open a 3-band image and plot the first band
            >>> with gw.open('image.tif') as ds:
            >>>     ax = ds.sel(band=1).gw.imshow()
            >>>
            >>> # Open and plot a 3-band image
            >>> with gw.open('image.tif') as ds:
            >>>
            >>>     ax = ds.sel(band=['red', 'green', 'blue']).gw.imshow(mask=True,
            >>>                                                          nodata=0,
            >>>                                                          vmin=0.1,
            >>>                                                          vmax=0.9,
            >>>                                                          robust=True)
        """

        if data.gw.nbands != 1:

            if data.gw.nbands != 3:
                logger.exception(
                    '  Only 1-band or 3-band arrays can be plotted.'
                )

        plt.rcParams['axes.titlesize'] = 5
        plt.rcParams['axes.titlepad'] = 5
        plt.rcParams['text.color'] = text_color
        plt.rcParams['axes.labelcolor'] = text_color
        plt.rcParams['xtick.color'] = text_color
        plt.rcParams['ytick.color'] = text_color
        plt.rcParams['figure.dpi'] = kwargs['dpi'] if 'dpi' in kwargs else 150
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.pad_inches'] = 0.5

        if 'ax' not in kwargs:
            fig, ax = plt.subplots()
            kwargs['ax'] = ax

        ax = kwargs['ax']

        if mask:

            if isinstance(data, xr.Dataset):

                if data.gw.nbands == 1:
                    plot_data = data.where(
                        (data['mask'] < 3) & (data != nodata)
                    )
                else:
                    plot_data = data.where(
                        (data['mask'] < 3) & (data.max(axis=0) != nodata)
                    )

            else:

                if data.gw.nbands == 1:
                    plot_data = data.where(data != nodata)
                else:
                    plot_data = data.where(data.max(axis=0) != nodata)

        else:
            plot_data = data

        if plot_data.gw.nbands == 3:

            plot_data = plot_data.transpose('y', 'x', 'band')

            if flip:
                plot_data = plot_data[..., ::-1]

            plot_data.plot.imshow(rgb='band', **kwargs)

        else:
            plot_data.squeeze().plot.imshow(**kwargs)

        ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

        for tick in ax.get_xticklabels():
            tick.set_rotation(rot)

        return ax
