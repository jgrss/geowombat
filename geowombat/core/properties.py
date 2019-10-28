from collections import namedtuple

from affine import Affine
from shapely.geometry import Polygon


WavelengthsBGR = namedtuple('WavelengthsBGR', 'blue green red')
WavelengthsRGB = namedtuple('WavelengthsRGB', 'red green blue')
WavelengthsBGRN = namedtuple('WavelengthsBGRN', 'blue green red nir')
WavelengthsRGBN = namedtuple('WavelengthsRGBN', 'red green blue nir')
WavelengthsL57 = namedtuple('WavelengthsL57', 'blue green red nir swir1 swir2')
WavelengthsL57Thermal = namedtuple('WavelengthsL57Thermal', 'blue green red nir swir1 thermal swir2')
WavelengthsL8 = namedtuple('WavelengthsL8', 'coastal blue green red nir swir1 swir2 cirrus')
WavelengthsL8Thermal = namedtuple('WavelengthsL8Thermal', 'coastal blue green red nir swir1 swir2 cirrus tirs1 tirs2')
WavelengthsS2 = namedtuple('WavelengthsS2', 'blue green red nir1 nir2 nir3 nir rededge swir1 swir2')


class DataProperties(object):

    @property
    def sensors(self):
        """Get supported sensors"""
        return sorted(list(self.wavelengths.keys()))

    @property
    def central_um(self):

        """
        Get a dictionary of central wavelengths (in micrometers)
        """

        return dict(l5=WavelengthsL57(blue=0.485,
                                      green=0.56,
                                      red=0.66,
                                      nir=0.835,
                                      swir1=1.65,
                                      swir2=2.22),
                    l7=WavelengthsL57(blue=0.485,
                                      green=0.56,
                                      red=0.66,
                                      nir=0.835,
                                      swir1=1.65,
                                      swir2=2.22),
                    l7th=WavelengthsL57Thermal(blue=0.485,
                                               green=0.56,
                                               red=0.66,
                                               nir=0.835,
                                               swir1=1.65,
                                               thermal=11.45,
                                               swir2=2.22),
                    l8=WavelengthsL8(coastal=0.44,
                                     blue=0.48,
                                     green=0.56,
                                     red=0.655,
                                     nir=0.865,
                                     swir1=1.61,
                                     swir2=2.2,
                                     cirrus=1.37),
                    l8l7=WavelengthsL57(blue=0.48,
                                        green=0.56,
                                        red=0.655,
                                        nir=0.865,
                                        swir1=1.61,
                                        swir2=2.2),
                    s2=WavelengthsS2(blue=0.49,
                                     green=0.56,
                                     red=0.665,
                                     nir1=0.705,
                                     nir2=0.74,
                                     nir3=0.783,
                                     nir=0.842,
                                     rededge=0.865,
                                     swir1=1.61,
                                     swir2=2.19),
                    s2l7=WavelengthsL57(blue=0.49,
                                        green=0.56,
                                        red=0.665,
                                        nir=0.842,
                                        swir1=1.61,
                                        swir2=2.19),
                    s210=WavelengthsBGRN(blue=0.49,
                                         green=0.56,
                                         red=0.665,
                                         nir=0.842),
                    planetscope=WavelengthsBGRN(blue=0.485,
                                                green=0.545,
                                                red=0.63,
                                                nir=0.82))

    @property
    def wavelengths(self):

        """
        Get a dictionary of sensor wavelengths
        """

        return dict(rgb=WavelengthsRGB(red=1,
                                       green=2,
                                       blue=3),
                    rgbn=WavelengthsRGBN(red=1,
                                         green=2,
                                         blue=3,
                                         nir=4),
                    bgr=WavelengthsBGR(blue=1,
                                       green=2,
                                       red=3),
                    bgrn=WavelengthsBGRN(blue=1,
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
                    l7th=WavelengthsL57Thermal(blue=1,
                                               green=2,
                                               red=3,
                                               nir=4,
                                               swir1=5,
                                               thermal=6,
                                               swir2=7),
                    l8=WavelengthsL8(coastal=1,
                                     blue=2,
                                     green=3,
                                     red=4,
                                     nir=5,
                                     swir1=6,
                                     swir2=7,
                                     cirrus=8),
                    l8th=WavelengthsL8Thermal(coastal=1,
                                              blue=2,
                                              green=3,
                                              red=4,
                                              nir=5,
                                              swir1=6,
                                              swir2=7,
                                              cirrus=8,
                                              tirs1=9,
                                              tirs2=10),
                    l8l7=WavelengthsL57(blue=2,
                                        green=3,
                                        red=4,
                                        nir=5,
                                        swir1=6,
                                        swir2=7),
                    s2=WavelengthsS2(blue=1,
                                     green=2,
                                     red=3,
                                     nir1=4,
                                     nir2=5,
                                     nir3=6,
                                     nir=7,
                                     rededge=8,
                                     swir1=9,
                                     swir2=10),
                    s2l7=WavelengthsL57(blue=1,
                                        green=2,
                                        red=3,
                                        nir=4,
                                        swir1=5,
                                        swir2=6),
                    s210=WavelengthsBGRN(blue=1,
                                         green=2,
                                         red=3,
                                         nir=4),
                    planetscope=WavelengthsBGRN(blue=1,
                                                green=2,
                                                red=3,
                                                nir=4),
                    quickbird=WavelengthsBGRN(blue=1,
                                              green=2,
                                              red=3,
                                              nir=4),
                    ikonos=WavelengthsBGRN(blue=1,
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

        if self.ndims > 3:
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
