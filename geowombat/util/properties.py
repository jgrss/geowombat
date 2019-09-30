from collections import namedtuple

from affine import Affine
from shapely.geometry import Polygon


class DatasetProperties(object):

    @property
    def wavelengths(self):

        WavelengthsL57 = namedtuple('WavelengthsL57', 'blue green red nir swir1 swir2')
        WavelengthsL8 = namedtuple('WavelengthsL8', 'coastal blue green red nir swir1 swir2')

        return dict(l5=WavelengthsL57(blue='blue',
                                      green='green',
                                      red='red',
                                      nir='nir',
                                      swir1='swir1',
                                      swir2='swir2'),
                    l7=WavelengthsL57(blue='blue',
                                      green='green',
                                      red='red',
                                      nir='nir',
                                      swir1='swir1',
                                      swir2='swir2'),
                    l8=WavelengthsL8(coastal='coastal',
                                     blue='blue',
                                     green='green',
                                     red='red',
                                     nir='nir',
                                     swir1='swir1',
                                     swir2='swir2'))


class DataArrayProperties(object):

    @property
    def ndims(self):
        return len(self._obj.shape)

    @property
    def row_chunks(self):
        return self._obj.data.chunksize[-2]

    @property
    def col_chunks(self):
        return self._obj.data.chunksize[-1]

    @property
    def band_chunks(self):

        if self.ndims > 2:
            return self._obj.data.chunksize[-3]
        else:
            return 1

    @property
    def time_chunks(self):

        if self.ndims < 3:
            return self._obj.data.chunksize[-4]
        else:
            return 1

    @property
    def bands(self):

        if self.ndims > 2:
            return self._obj.shape[-3]
        else:
            return 1

    @property
    def rows(self):
        return self._obj.shape[-2]

    @property
    def cols(self):
        return self._obj.shape[-1]

    @property
    def left(self):
        return float(self._obj.x.min().values)

    @property
    def right(self):
        return float(self._obj.x.max().values)

    @property
    def top(self):
        return float(self._obj.y.max().values)

    @property
    def bottom(self):
        return float(self._obj.y.min().values)

    @property
    def bounds(self):
        return self.left, self.bottom, self.right, self.top

    @property
    def celly(self):
        return self._obj.res[0]

    @property
    def cellx(self):
        return self._obj.res[1]

    @property
    def geometry(self):

        return Polygon([(self.left, self.bottom),
                        (self.left, self.top),
                        (self.right, self.top),
                        (self.right, self.bottom),
                        (self.left, self.bottom)])

    @property
    def meta(self):

        Meta = namedtuple('Meta', 'left right top bottom bounds affine geometry')

        return Meta(left=self.left,
                    right=self.right,
                    top=self.top,
                    bottom=self.bottom,
                    bounds=self.bounds,
                    affine=Affine(*self._obj.transform),
                    geometry=self.geometry)
