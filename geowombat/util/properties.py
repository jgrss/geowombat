from collections import namedtuple


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
