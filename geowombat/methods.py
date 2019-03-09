from copy import copy

from .geoarray import GeoArray
from .properties import GeoProperties

import numpy as np
from osgeo import gdal, osr
from skimage.exposure import rescale_intensity

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


class GeoMethods(GeoProperties):

    def _get_shape(self):

        self.layers_ = 1
        self.rows_ = 0
        self.cols_ = 0

        if len(self.shape) == 2:
            self.rows_, self.cols_ = self.shape
        else:
            self.layers_, self.rows_, self.cols_ = self.shape

    def set_names(self, names):

        """
        Sets array layer names
        """

        self.layer_names = names

        for i in range(0, len(names)):
            setattr(self, names[i], self[i])

    def set_no_data(self, value):

        """
        Sets array 'no data' value
        """

        self.no_data_ = value

    def extract(self,
                row_start=None,
                rows=None,
                col_start=None,
                cols=None,
                layer_start=None,
                layers=None):

        """
        Slices an array

        Examples:
            >>> # rows 1-10
            >>> # columns 0-20
            >>> print(geoarray.left, geoarray.top)
            >>>
            >>> # Use the `extract` method to maintain geo-coordinates
            >>> new_array = geoarray.extract(row_start=1, rows=10, cols=20)
            >>>
            >>> # The extent is updated
            >>> print(new_array.left, new_array.top)
        """

        if not isinstance(layer_start, int):
            layer_start = 0

        if not isinstance(layers, int):
            layers = self.layers

        if not isinstance(row_start, int):
            row_start = 0

        if not isinstance(rows, int):
            rows = self.rows

        if not isinstance(col_start, int):
            col_start = 0

        if not isinstance(cols, int):
            cols = self.columns

        src = self.src.copy()

        src.left = self.src.left + (self.src.cellY * col_start)
        src.right = self.src.left + (self.src.cellY * (col_start+cols))
        src.top = self.src.top - (self.src.cellY * row_start)
        src.bottom = self.src.top - (self.src.cellY * (row_start+rows))

        # if inplace:
        #
        #     if self.layers == 1:
        #         self = self[row_start:row_start+rows, col_start:col_start+cols]
        #     else:
        #
        #         self = self[layer_start:layer_start+layers,
        #                     row_start:row_start+rows,
        #                     col_start:col_start+cols]
        #
        #     # new_geo = GeoArray(array, src)
        #     # self.__class__ = new_geo.__class__
        #     # self = super(GeoArray, self).__init__(array, src)
        #     # self = self.reinit(array, src)
        #     # self = GeoArray(array, src)
        #
        #     # self.src = src
        #     # self._update_inplace(self)

        if self.layers == 1:
            sub_array = self[row_start:row_start+rows, col_start:col_start+cols]
        else:

            sub_array = self[layer_start:layer_start+layers,
                             row_start:row_start+rows,
                             col_start:col_start+cols]

        sub_array.src = src
        sub_array.set_no_data(self.no_data)

        return sub_array

    def to_crs(self, crs=4326, **kwargs):

        """
        Warps the array projection

        Args:
            crs (Optional[int or str])
            kwargs (Optional[dict])

        Examples:
            >>> print(geoarray.left, geoarray.top, geoarray.projection)
            >>>
            >>> geoarray.to_crs(102033)
            >>>
            >>> print(geoarray.left, geoarray.top, geoarray.projection)
            >>>
            >>> # Create a new array with warped bounds
            >>> geoarray_warp = geoarray.to_crs(102033)
        """

        if isinstance(self.crs, int):
            in_epsg = self.crs
            in_proj = None
        else:
            in_epsg = None
            in_proj = self.crs

        if isinstance(crs, int):
            out_epsg = crs
            out_proj = None
        else:
            out_epsg = None
            out_proj = crs

        n_layers = copy(self.layers)

        if 'outputBounds' not in kwargs:

            ptr = self.src._transform(self.left, self.top, self.crs, crs)
            tleft, ttop = ptr.x_transform, ptr.y_transform

            ptr = self.src._transform(self.right, self.bottom, self.crs, crs)
            tright, tbottom = ptr.x_transform, ptr.y_transform

            kwargs['outputBounds'] = [tleft, tbottom, tright, ttop]

        # Warp the array
        src = self.src._warp(self.src.datasource,
                             'memory.mem',
                             in_epsg=in_epsg,
                             in_proj=in_proj,
                             out_epsg=out_epsg,
                             out_proj=out_proj,
                             return_datasource=True,
                             overwrite=True,
                             srcNodata=self.no_data,
                             **kwargs)

        self_cp = GeoArray(src.read(bands=list(range(1, n_layers+1))), src)
        self_cp.src = src.copy()

        self_cp = self._attach_funcs(self_cp)
        self_cp.set_no_data(self.no_data)

        try:
            gdal.Unlink(src.output_image)
        except:
            pass

        src = None

        return self_cp

    def _attach_funcs(self, self_cp):

        self_cp.src._create = self.src._create
        self_cp.src._warp = self.src._warp
        self_cp.src._transform = self.src._transform
        self_cp.src._nd_to_rgb = self.src._nd_to_rgb
        self_cp.src._get_xy_offsets = self.src._get_xy_offsets

        return self_cp

    def overlap_bounds(self, extent):

        left = max(self.extent[0], extent[0])
        right = min(self.extent[1], extent[1])
        top = min(self.extent[2], extent[2])
        bottom = max(self.extent[3], extent[3])

        return left, right, top, bottom

    def geo_add(self, garray):

        """
        Adds GeoArrays
        """

        # Find the overlap
        left, right, top, bottom = self.overlap_bounds(garray.extent)

        nrows = int((top - bottom) / self.cell_y)
        ncols = int((right - left) / self.cell_y)

        extent_list = [left, top, right, bottom, self.cell_x, self.cell_y]

        l_a, t_a = self.src._get_xy_offsets(image_list=extent_list,
                                            x=self.left,
                                            y=self.top,
                                            check_position=False)[2:]

        l_b, t_b = self.src._get_xy_offsets(image_list=extent_list,
                                            x=garray.left,
                                            y=garray.top,
                                            check_position=False)[2:]

        if self.layers > 1:

            slice_a = self[:, t_a:t_a+nrows, l_a:l_a+ncols]
            slice_b = garray[:, t_b:t_b+nrows, l_b:l_b+ncols]

        else:

            slice_a = self[t_a:t_a+nrows, l_a:l_a+ncols]
            slice_b = garray[t_b:t_b+nrows, l_b:l_b+ncols]

        src = self.src.copy()

        src.left = left
        src.right = right
        src.top = top
        src.bottom = bottom

        result = slice_a + slice_b

        result.src = src
        result.set_no_data(self.no_data)

        return result

    def mask(self, mask_array):

        """
        Masks the GeoArray with a secondary geo-array
        """

        left, right, top, bottom = self.overlap_bounds(mask_array.extent)

        extent_list = [left, top, right, bottom, self.cell_x, self.cell_y]

        x_offset, y_offset = self.src._get_xy_offsets(image_list=extent_list,
                                                      x=self.left,
                                                      y=self.top,
                                                      check_position=False)[2:]

        if self.layers > 1:
            self[:, y_offset:y_offset+mask_array.rows, x_offset:x_offset+mask_array.columns] = self.no_data
        else:
            self[y_offset:y_offset+mask_array.rows, x_offset:x_offset+mask_array.columns] = self.no_data

        return self

    def to_coordinates(self):

        """
        Creates x,y coordinate indices

        Example:
            >>> x, y = geoarray.to_coordinates()
        """

        # Create the longitudes
        x_coordinates = np.arange(self.left, self.right, self.cell_y)
        x_coordinates = np.tile(x_coordinates, self.rows).reshape(self.rows, self.columns)

        # Create latitudes
        y_coordinates = np.arange(self.top, self.bottom, self.cell_x).reshape(self.rows, 1)
        y_coordinates = np.tile(y_coordinates, self.columns)

        return x_coordinates, y_coordinates

    def to_raster(self, file_name, **kwargs):

        """
        Writes an array to a raster file

        Args:
            file_name (str)
            kwargs (dict)

        Examples:
            >>> geoarray.to_raster('image.tif')
        """

        self.src.update_info(bands=self.layers,
                             rows=self.rows,
                             cols=self.columns,
                             left=self.left,
                             top=self.top)

        out_rst = self.src._create(file_name,
                                   self.src,
                                   **kwargs)

        if self.layers == 1:
            out_rst.write_array(self, band=1)
        else:

            for n_band in range(1, self.layers+1):
                out_rst.write_array(self[n_band-1], band=n_band)

        out_rst.close_file()
        out_rst = None

    def _transform_plot_coords(self):

        in_sr = osr.SpatialReference()
        in_sr.ImportFromWkt(self.crs)
        in_proj = in_sr.ExportToProj4()

        out_sr = osr.SpatialReference()
        out_sr.ImportFromEPSG(4326)
        out_proj = out_sr.ExportToProj4()

        ptr = self.src._transform(self.left, self.top, in_proj, out_proj)
        im_left_crs, im_top_crs = ptr.x_transform, ptr.y_transform

        ptr = self.src._transform(self.right, self.bottom, in_proj, out_proj)
        im_right_crs, im_bottom_crs = ptr.x_transform, ptr.y_transform

        return [im_left_crs, im_right_crs, im_top_crs, im_bottom_crs]

    def show(self, bands=None, scale_factor=1.0, cmap=None, percentiles=(5, 95)):

        """
        Plots the array

        Args:
            bands (Optional[int or list])
            scale_factor (Optional[float])
            cmap (Optional[str])
            percentiles (Optional[tuple])
        """

        plt.rcParams['figure.figsize'] = 3, 3
        plt.rcParams['axes.titlesize'] = 5
        plt.rcParams['axes.titlepad'] = 5
        plt.rcParams['axes.grid'] = False
        plt.rcParams['axes.spines.left'] = False
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.bottom'] = False
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.color'] = 'none'
        plt.rcParams['ytick.color'] = 'none'
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.pad_inches'] = 0.5

        extent = self._transform_plot_coords()

        crs = ccrs.PlateCarree()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection=crs)

        gkwargs = dict(color='white',
                       linewidth=0.1,
                       linestyle='--',
                       draw_labels=True)

        crskwargs = dict(origin='upper',
                         extent=extent,
                         transform=crs)

        if isinstance(bands, list):
            rgb = self[np.array(bands, dtype='int64')-1]
        elif isinstance(bands, int):
            rgb = self[bands-1]
        else:
            rgb = self.copy()

        rgb = self._scale_array(rgb, scale_factor, percentiles)

        rgb = np.ma.masked_where(rgb == 0, rgb)

        if (self.layers == 3) or isinstance(bands, list):
            cmap = None

        ax.imshow(rgb,
                  interpolation='nearest',
                  cmap=cmap,
                  **crskwargs)

        ax.outline_patch.set_linewidth(0)
        ax = self._set_grids(ax, **gkwargs)

        plt.tight_layout(pad=0.5)

        plt.show()

    def _scale_array(self, rgb, scale_factor, percentiles):

        """
        Scales and adjusts values for display
        """

        pmin, pmax = percentiles

        rgb = np.float32(rgb)

        rgb[(rgb == self.no_data) | (rgb == 0)] = np.nan

        rgb[~np.isnan(rgb)] *= scale_factor

        if len(rgb.shape) == 2:

            rgb = rescale_intensity(rgb,
                                    in_range=(np.nanpercentile(rgb, pmin),
                                              np.nanpercentile(rgb, pmax)),
                                    out_range=(0, 1))

        else:

            for lidx, layer in enumerate(rgb):

                rgb[lidx] = rescale_intensity(layer,
                                              in_range=(np.nanpercentile(layer, pmin),
                                                        np.nanpercentile(layer, pmax)),
                                              out_range=(0, 1))

            rgb = self.src._nd_to_rgb(rgb)

        rgb[np.isnan(rgb)] = 0

        return rgb

    @staticmethod
    def _set_grids(ax, **gkwargs):

        grids = ax.gridlines(**gkwargs)

        grids.xformatter = LONGITUDE_FORMATTER
        grids.yformatter = LATITUDE_FORMATTER

        grids.ylabel_style = {'size': 3, 'color': 'gray'}
        grids.xlabel_style = {'size': 2, 'color': 'gray', 'rotation': 45}

        grids.ylabels_left = True
        grids.ylabels_right = False
        grids.xlabels_top = False
        grids.xlabels_bottom = True

        return ax
