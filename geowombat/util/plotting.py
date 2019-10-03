import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt


class Plotting(object):

    @staticmethod
    def imshow(data,
               band_names=None,
               mask=False,
               nodata=0,
               flip=False,
               text_color='black',
               rot=30,
               **kwargs):

        """
        Shows an image on a plot

        Args:
            data (``xarray.DataArray`` or ``xarray.Dataset``): The data to plot.
            band_names (Optional[list or str]): The band name or list of band names to plot.
            mask (Optional[bool]): Whether to mask 'no data' values (given by ``nodata``).
            nodata (Optional[int or float]): The 'no data' value.
            flip (Optional[bool]): Whether to flip an RGB array's band order.
            text_color (Optional[str]): The text color.
            rot (Optional[int]): The degree rotation for the x-axis tick labels.
            kwargs (Optional[dict]): Keyword arguments passed to ``xarray.plot.imshow``.

        Returns:
            Matplotlib axis object

        Examples:
            >>> with gw.open('image.tif') as ds:
            >>>     ds.gw.imshow(band_names=['red', 'green', 'red'], mask=True, vmin=0.1, vmax=0.9, robust=True)
        """

        if isinstance(band_names, list):

            if (len(band_names) != 1) and (len(band_names) != 3):
                logger.exception('  Only 1-band or 3-band arrays can be plotted.')

        plt.rcParams['axes.titlesize'] = 5
        plt.rcParams['axes.titlepad'] = 5
        plt.rcParams['text.color'] = text_color
        plt.rcParams['axes.labelcolor'] = text_color
        plt.rcParams['xtick.color'] = text_color
        plt.rcParams['ytick.color'] = text_color
        plt.rcParams['figure.dpi'] = kwargs['dpi'] if 'dpi' in kwargs else 150
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.pad_inches'] = 0.5

        fig = plt.figure()
        ax = fig.add_subplot(111)

        rgb = data.sel(band=band_names)

        if mask:

            if isinstance(data, xr.Dataset):

                if len(band_names) == 1:
                    rgb = rgb.where((data['mask'] < 3) & (rgb != nodata))
                else:
                    rgb = rgb.where((data['mask'] < 3) & (rgb.max(axis=0) != nodata))

            else:

                if len(band_names) == 1:
                    rgb = rgb.where(rgb != nodata)
                else:
                    rgb = rgb.where(rgb.max(axis=0) != nodata)

        if len(band_names) == 3:

            rgb = rgb.transpose('y', 'x', 'band')

            if flip:
                rgb = rgb[..., ::-1]

            rgb.plot.imshow(rgb='band', ax=ax, **kwargs)

        else:
            rgb.plot.imshow(ax=ax, **kwargs)

        ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

        for tick in ax.get_xticklabels():
            tick.set_rotation(rot)

        plt.show()

        return ax
