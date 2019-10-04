from collections import namedtuple

from affine import Affine
from shapely.geometry import Polygon


class DataProperties(object):

    @property
    def wavelengths(self):

        """
        Get a dictionary of sensor wavelengths
        """

        WavelengthsRGB = namedtuple('WavelengthsRGB', 'blue green red')
        WavelengthsRGBN = namedtuple('WavelengthsRGBN', 'blue green red nir')
        WavelengthsL57 = namedtuple('WavelengthsL57', 'blue green red nir swir1 swir2')
        WavelengthsL8 = namedtuple('WavelengthsL8', 'coastal blue green red nir swir1 swir2 cirrus')
        WavelengthsS210 = namedtuple('WavelengthsS210', 'blue green red nir')

        return dict(rgb=WavelengthsRGB(blue=1,
                                       green=2,
                                       red=3),
                    rgbn=WavelengthsRGBN(blue=1,
                                         green=2,
                                         red=3,
                                         nir=4),
                    l5=WavelengthsL57(blue=1,
                                      green=2,
                                      red=3,
                                      nir=4,
                                      swir1=5,
                                      swir2=6),
                    l7=WavelengthsL57(blue=1,
                                      green=2,
                                      red=3,
                                      nir=4,
                                      swir1=5,
                                      swir2=6),
                    l8=WavelengthsL8(coastal=1,
                                     blue=2,
                                     green=3,
                                     red=4,
                                     nir=5,
                                     swir1=6,
                                     swir2=7,
                                     cirrus=8),
                    s210=WavelengthsS210(blue=1,
                                         green=2,
                                         red=3,
                                         nir=4))

    @property
    def ndims(self):
        """Get the number of array dimensions"""
        return len(self._obj.shape)

    @property
    def row_chunks(self):
        """Get the row chunk size"""
        return self._obj.data.chunksize[-2]

    @property
    def col_chunks(self):
        """Get the column chunk size"""
        return self._obj.data.chunksize[-1]

    @property
    def band_chunks(self):

        """
        Get the band chunk size
        """

        if self.ndims > 2:
            return self._obj.data.chunksize[-3]
        else:
            return 1

    @property
    def time_chunks(self):

        """
        Get the time chunk size
        """

        if self.ndims < 3:
            return self._obj.data.chunksize[-4]
        else:
            return 1

    @property
    def ntime(self):

        """
        Get the number of time dimensions
        """

        if self.ndims > 3:
            return self._obj.shape[-4]
        else:
            return 1

    @property
    def nbands(self):

        """
        Get the number of array bands
        """

        if self.ndims > 2:
            return self._obj.shape[-3]
        else:
            return 1

    @property
    def nrows(self):
        """Get the number of array rows"""
        return self._obj.shape[-2]

    @property
    def ncols(self):
        """Get the number of array columns"""
        return self._obj.shape[-1]

    @property
    def left(self):

        """
        Get the array bounding box left coordinate

        Pixel shift reference:
            https://github.com/pydata/xarray/blob/master/xarray/backends/rasterio_.py
            http://web.archive.org/web/20160326194152/http://remotesensing.org/geotiff/spec/geotiff2.5.html#2.5.2
        """

        return float(self._obj.x.min().values) - self.cellxh

    @property
    def right(self):
        """Get the array bounding box right coordinate"""
        return float(self._obj.x.max().values) + self.cellxh

    @property
    def top(self):
        """Get the array bounding box top coordinate"""
        return float(self._obj.y.max().values) + self.cellyh

    @property
    def bottom(self):
        """Get the array bounding box bottom coordinate"""
        return float(self._obj.y.min().values) - self.cellyh

    @property
    def bounds(self):
        """Get the array bounding box (left, bottom, right, top)"""
        return self.left, self.bottom, self.right, self.top

    @property
    def celly(self):
        """Get the cell size in the y direction"""
        return self._obj.res[1]

    @property
    def cellx(self):
        """Get the cell size in the x direction"""
        return self._obj.res[0]

    @property
    def cellyh(self):
        """Get the half width of the cell size in the y direction"""
        return self.celly / 2.0

    @property
    def cellxh(self):
        """Get the half width of the cell size in the x direction"""
        return self.cellx / 2.0

    @property
    def geometry(self):

        """
        Get the polygon geometry of the array bounding box
        """

        return Polygon([(self.left, self.bottom),
                        (self.left, self.top),
                        (self.right, self.top),
                        (self.right, self.bottom),
                        (self.left, self.bottom)])

    @property
    def meta(self):

        """
        Get the array metdata
        """

        Meta = namedtuple('Meta', 'left right top bottom bounds affine geometry')

        return Meta(left=self.left,
                    right=self.right,
                    top=self.top,
                    bottom=self.bottom,
                    bounds=self.bounds,
                    affine=Affine(*self._obj.transform),
                    geometry=self.geometry)
