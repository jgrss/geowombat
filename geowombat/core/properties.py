from pathlib import Path
from collections import namedtuple

from .util import n_rows_cols

import pandas as pd
import geopandas as gpd
from rasterio.coords import BoundingBox
from affine import Affine
import shapely
from shapely.geometry import Polygon

try:
    from shapely import speedups
    speedups.enable()
except:
    pass


WavelengthsPan = namedtuple('WavelengthsPan', 'pan')
WavelengthsBGR = namedtuple('WavelengthsBGR', 'blue green red')
WavelengthsRGB = namedtuple('WavelengthsRGB', 'red green blue')
WavelengthsBGRN = namedtuple('WavelengthsBGRN', 'blue green red nir')
WavelengthsRGBN = namedtuple('WavelengthsRGBN', 'red green blue nir')
WavelengthsL57 = namedtuple('WavelengthsL57', 'blue green red nir swir1 swir2')
WavelengthsL57Thermal = namedtuple('WavelengthsL57Thermal', 'blue green red nir swir1 thermal swir2')
WavelengthsL57Pan = namedtuple('WavelengthsL57Pan', 'blue green red nir swir1 swir2 pan')
WavelengthsL8 = namedtuple('WavelengthsL8', 'coastal blue green red nir swir1 swir2 cirrus')
WavelengthsL8Thermal = namedtuple('WavelengthsL8Thermal', 'coastal blue green red nir swir1 swir2 cirrus tirs1 tirs2')
WavelengthsS2 = namedtuple('WavelengthsS2', 'blue green red nir1 nir2 nir3 nir rededge swir1 swir2')
WavelengthsS2Full = namedtuple('WavelengthsS2', 'coastal blue green red nir1 nir2 nir3 nir rededge water cirrus swir1 swir2')
WavelengthsS220 = namedtuple('WavelengthsS220', 'nir1 nir2 nir3 rededge swir1 swir2')
WavelengthsS2Cloudless = namedtuple('WavelengthsS2Cloudless', 'coastal blue red nir1 nir rededge water cirrus swir1 swir2')


class DataProperties(object):

    def __init__(self):
        self._footprint_grid = None

    @property
    def avail_sensors(self):
        """Get supported sensors"""
        return sorted(list(self.wavelengths.keys()))

    @property
    def altitude(self):

        """
        Get satellite altitudes (in km)
        """

        return dict(aster=705,
                    l5=705,
                    l7=705,
                    l8=705,
                    s2=786,
                    ps=475,
                    qb=482,
                    ik=681,
                    wv3=617)

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
                    l7mspan=WavelengthsL57Pan(blue=0.485,
                                              green=0.56,
                                              red=0.66,
                                              nir=0.835,
                                              swir1=1.65,
                                              swir2=2.22,
                                              pan=0.71),
                    l7pan=WavelengthsPan(pan=0.71),
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
                    l8l7mspan=WavelengthsL57Pan(blue=0.48,
                                                green=0.56,
                                                red=0.655,
                                                nir=0.865,
                                                swir1=1.61,
                                                swir2=2.2,
                                                pan=0.59),
                    l8pan=WavelengthsPan(pan=0.59),
                    l5bgrn=WavelengthsBGRN(blue=0.485,
                                           green=0.56,
                                           red=0.66,
                                           nir=0.835),
                    l7bgrn=WavelengthsBGRN(blue=0.485,
                                           green=0.56,
                                           red=0.66,
                                           nir=0.835),
                    l8bgrn=WavelengthsBGRN(blue=0.48,
                                           green=0.56,
                                           red=0.655,
                                           nir=0.865),
                    s2=WavelengthsS2(blue=0.4924,
                                     green=0.5598,
                                     red=0.6646,
                                     nir1=0.7041,
                                     nir2=0.7405,
                                     nir3=0.7828,
                                     nir=0.8328,
                                     rededge=0.8647,
                                     swir1=1.6137,
                                     swir2=2.2024),
                    s2a=WavelengthsS2(blue=0.4924,
                                      green=0.5598,
                                      red=0.6646,
                                      nir1=0.7041,
                                      nir2=0.7405,
                                      nir3=0.7828,
                                      nir=0.8328,
                                      rededge=0.8647,
                                      swir1=1.6137,
                                      swir2=2.2024),
                    s2b=WavelengthsS2(blue=0.4921,
                                      green=0.559,
                                      red=0.665,
                                      nir1=0.7038,
                                      nir2=0.7391,
                                      nir3=0.7797,
                                      nir=0.833,
                                      rededge=0.864,
                                      swir1=1.6104,
                                      swir2=2.1857),
                    s2f=WavelengthsS2Full(coastal=0.4427,
                                          blue=0.4924,
                                          green=0.5598,
                                          red=0.6646,
                                          nir1=0.7041,
                                          nir2=0.7405,
                                          nir3=0.7828,
                                          nir=0.8328,
                                          rededge=0.8647,
                                          water=0.9451,
                                          cirrus=1.3735,
                                          swir1=1.6137,
                                          swir2=2.2024),
                    s2af=WavelengthsS2Full(coastal=0.4427,
                                           blue=0.4924,
                                           green=0.5598,
                                           red=0.6646,
                                           nir1=0.7041,
                                           nir2=0.7405,
                                           nir3=0.7828,
                                           nir=0.8328,
                                           rededge=0.8647,
                                           water=0.9451,
                                           cirrus=1.3735,
                                           swir1=1.6137,
                                           swir2=2.2024),
                    s2bf=WavelengthsS2Full(coastal=0.4423,
                                           blue=0.4921,
                                           green=0.559,
                                           red=0.665,
                                           nir1=0.7038,
                                           nir2=0.7391,
                                           nir3=0.7797,
                                           nir=0.833,
                                           rededge=0.864,
                                           water=0.9432,
                                           cirrus=1.3769,
                                           swir1=1.6104,
                                           swir2=2.1857),
                    s2l7=WavelengthsL57(blue=0.49,
                                        green=0.56,
                                        red=0.665,
                                        nir=0.842,
                                        swir1=1.61,
                                        swir2=2.19),
                    s2al7=WavelengthsL57(blue=0.4924,
                                         green=0.5598,
                                         red=0.6646,
                                         nir=0.8328,
                                         swir1=1.6137,
                                         swir2=2.2024),
                    s2bl7=WavelengthsL57(blue=0.4921,
                                         green=0.559,
                                         red=0.665,
                                         nir=0.833,
                                         swir1=1.6104,
                                         swir2=2.1857),
                    s210=WavelengthsBGRN(blue=0.49,
                                         green=0.56,
                                         red=0.665,
                                         nir=0.842),
                    s2a10=WavelengthsBGRN(blue=0.4924,
                                          green=0.5598,
                                          red=0.6646,
                                          nir=0.8328),
                    s2b10=WavelengthsBGRN(blue=0.4921,
                                          green=0.559,
                                          red=0.665,
                                          nir=0.833),
                    s220=WavelengthsS220(nir1=0.705,
                                         nir2=0.74,
                                         nir3=0.783,
                                         rededge=0.865,
                                         swir1=1.61,
                                         swir2=2.19),
                    s2a20=WavelengthsS220(nir1=0.7041,
                                          nir2=0.7405,
                                          nir3=0.7828,
                                          rededge=0.8647,
                                          swir1=1.6137,
                                          swir2=2.2024),
                    s2b20=WavelengthsS220(nir1=0.7038,
                                          nir2=0.7391,
                                          nir3=0.7797,
                                          rededge=0.864,
                                          swir1=1.6104,
                                          swir2=2.1857),
                    s2cloudless=WavelengthsS2Cloudless(coastal=0.443,
                                                       blue=0.49,
                                                       red=0.665,
                                                       nir1=0.705,
                                                       nir=0.842,
                                                       rededge=0.865,
                                                       water=0.945,
                                                       cirrus=1.375,
                                                       swir1=1.61,
                                                       swir2=2.19),
                    s2acloudless=WavelengthsS2Cloudless(coastal=0.4427,
                                                        blue=0.4924,
                                                        red=0.6646,
                                                        nir1=0.7041,
                                                        nir=0.8328,
                                                        rededge=0.8647,
                                                        water=0.9451,
                                                        cirrus=1.3735,
                                                        swir1=1.6137,
                                                        swir2=2.2024),
                    s2bcloudless=WavelengthsS2Cloudless(coastal=0.4423,
                                                        blue=0.4921,
                                                        red=0.665,
                                                        nir1=0.7038,
                                                        nir=0.833,
                                                        rededge=0.864,
                                                        water=0.9432,
                                                        cirrus=1.3769,
                                                        swir1=1.6104,
                                                        swir2=2.1857),
                    ps=WavelengthsBGRN(blue=0.485,
                                       green=0.545,
                                       red=0.63,
                                       nir=0.82))

    @property
    def sensor_names(self):

        """
        Get sensor full names
        """

        return dict(rgb='red, green, and blue',
                    rgbn='red, green, blue, and NIR',
                    bgr='blue, green, and red',
                    bgrn='blue, green, red, and NIR',
                    l5='Landsat 5 Thematic Mapper (TM)',
                    l7='Landsat 7 Enhanced Thematic Mapper Plus (ETM+) without panchromatic and thermal bands',
                    l7th='Landsat 7 Enhanced Thematic Mapper Plus (ETM+) with thermal band',
                    l7mspan='Landsat 7 Enhanced Thematic Mapper Plus (ETM+) with panchromatic band',
                    l7pan='Landsat 7 panchromatic band',
                    l8='Landsat 8 Operational Land Imager (OLI) and Thermal Infrared Sensor (TIRS) without panchromatic and thermal bands',
                    l8l7='Landsat 8 Operational Land Imager (OLI) and Thermal Infrared Sensor (TIRS) with 6 Landsat 7-like bands',
                    l8l7mspan='Landsat 8 Operational Land Imager (OLI) and panchromatic band with 6 Landsat 7-like bands',
                    l8th='Landsat 8 Operational Land Imager (OLI) and Thermal Infrared Sensor (TIRS) with thermal band',
                    l8pan='Landsat 8 panchromatic band',
                    s2='Sentinel 2 Multi-Spectral Instrument (MSI) without 3 60m bands (coastal, water vapor, cirrus)',
                    s2f='Sentinel 2 Multi-Spectral Instrument (MSI) with 3 60m bands (coastal, water vapor, cirrus)',
                    s2l7='Sentinel 2 Multi-Spectral Instrument (MSI) with 6 Landsat 7-like bands',
                    s210='Sentinel 2 Multi-Spectral Instrument (MSI) with 4 10m (visible + NIR) bands',
                    s220='Sentinel 2 Multi-Spectral Instrument (MSI) with 6 20m bands',
                    s2cloudless='Sentinel 2 Multi-Spectral Instrument (MSI) with 10 bands for s2cloudless',
                    ps='PlanetScope with 4 (visible + NIR) bands',
                    qb='Quickbird with 4 (visible + NIR) bands',
                    ik='IKONOS with 4 (visible + NIR) bands')

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
                    l7mspan=WavelengthsL57Pan(blue=1,
                                              green=2,
                                              red=3,
                                              nir=4,
                                              swir1=5,
                                              swir2=6,
                                              pan=7),
                    l7pan=WavelengthsPan(pan=1),
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
                    l8l7=WavelengthsL57(blue=1,
                                        green=2,
                                        red=3,
                                        nir=4,
                                        swir1=5,
                                        swir2=6),
                    l8l7mspan=WavelengthsL57Pan(blue=1,
                                                green=2,
                                                red=3,
                                                nir=4,
                                                swir1=5,
                                                swir2=6,
                                                pan=7),
                    l8pan=WavelengthsPan(pan=1),
                    l5bgrn=WavelengthsBGRN(blue=1,
                                           green=2,
                                           red=3,
                                           nir=4),
                    l7bgrn=WavelengthsBGRN(blue=1,
                                           green=2,
                                           red=3,
                                           nir=4),
                    l8bgrn=WavelengthsBGRN(blue=1,
                                           green=2,
                                           red=3,
                                           nir=4),
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
                    s2a=WavelengthsS2(blue=1,
                                      green=2,
                                      red=3,
                                      nir1=4,
                                      nir2=5,
                                      nir3=6,
                                      nir=7,
                                      rededge=8,
                                      swir1=9,
                                      swir2=10),
                    s2b=WavelengthsS2(blue=1,
                                      green=2,
                                      red=3,
                                      nir1=4,
                                      nir2=5,
                                      nir3=6,
                                      nir=7,
                                      rededge=8,
                                      swir1=9,
                                      swir2=10),
                    s2f=WavelengthsS2Full(coastal=1,
                                          blue=2,
                                          green=3,
                                          red=4,
                                          nir1=5,
                                          nir2=6,
                                          nir3=7,
                                          nir=8,
                                          rededge=9,
                                          water=10,
                                          cirrus=11,
                                          swir1=12,
                                          swir2=13),
                    s2af=WavelengthsS2Full(coastal=1,
                                           blue=2,
                                           green=3,
                                           red=4,
                                           nir1=5,
                                           nir2=6,
                                           nir3=7,
                                           nir=8,
                                           rededge=9,
                                           water=10,
                                           cirrus=11,
                                           swir1=12,
                                           swir2=13),
                    s2bf=WavelengthsS2Full(coastal=1,
                                           blue=2,
                                           green=3,
                                           red=4,
                                           nir1=5,
                                           nir2=6,
                                           nir3=7,
                                           nir=8,
                                           rededge=9,
                                           water=10,
                                           cirrus=11,
                                           swir1=12,
                                           swir2=13),
                    s2l7=WavelengthsL57(blue=1,
                                        green=2,
                                        red=3,
                                        nir=4,
                                        swir1=5,
                                        swir2=6),
                    s2al7=WavelengthsL57(blue=1,
                                         green=2,
                                         red=3,
                                         nir=4,
                                         swir1=5,
                                         swir2=6),
                    s2bl7=WavelengthsL57(blue=1,
                                         green=2,
                                         red=3,
                                         nir=4,
                                         swir1=5,
                                         swir2=6),
                    s210=WavelengthsBGRN(blue=1,
                                         green=2,
                                         red=3,
                                         nir=4),
                    s2a10=WavelengthsBGRN(blue=1,
                                          green=2,
                                          red=3,
                                          nir=4),
                    s2b10=WavelengthsBGRN(blue=1,
                                          green=2,
                                          red=3,
                                          nir=4),
                    s220=WavelengthsS220(nir1=1,
                                         nir2=2,
                                         nir3=3,
                                         rededge=4,
                                         swir1=5,
                                         swir2=6),
                    s2a20=WavelengthsS220(nir1=1,
                                          nir2=2,
                                          nir3=3,
                                          rededge=4,
                                          swir1=5,
                                          swir2=6),
                    s2b20=WavelengthsS220(nir1=1,
                                          nir2=2,
                                          nir3=3,
                                          rededge=4,
                                          swir1=5,
                                          swir2=6),
                    s2cloudless=dict(coastal=1,
                                     blue=2,
                                     red=3,
                                     nir1=4,
                                     nir=5,
                                     rededge=6,
                                     water=7,
                                     cirrus=8,
                                     swir1=9,
                                     swir2=10),
                    s2acloudless=dict(coastal=1,
                                      blue=2,
                                      red=3,
                                      nir1=4,
                                      nir=5,
                                      rededge=6,
                                      water=7,
                                      cirrus=8,
                                      swir1=9,
                                      swir2=10),
                    s2bcloudless=dict(coastal=1,
                                      blue=2,
                                      red=3,
                                      nir1=4,
                                      nir=5,
                                      rededge=6,
                                      water=7,
                                      cirrus=8,
                                      swir1=9,
                                      swir2=10),
                    ps=WavelengthsBGRN(blue=1,
                                       green=2,
                                       red=3,
                                       nir=4),
                    qb=WavelengthsBGRN(blue=1,
                                       green=2,
                                       red=3,
                                       nir=4),
                    ik=WavelengthsBGRN(blue=1,
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
    def bounds_as_namedtuple(self):
        """Get the array bounding box as a ``rasterio.coords.BoundingBox``"""
        return BoundingBox(left=self.left, bottom=self.bottom, right=self.right, top=self.top)

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
    def pydatetime(self):
        """Get Python datetime objects from the time dimension"""
        return pd.to_datetime(self._obj.time.values).to_pydatetime()

    @property
    def chunk_grid(self):

        """Get the image chunk grid"""

        geometries = []

        top = self.top
        for i in range(0, self.nrows, self.row_chunks):

            i_csize = n_rows_cols(i, self.row_chunks, self.nrows)
            bottom = top - (i_csize * abs(self.celly))
            left = self.left

            for j in range(0, self.ncols, self.col_chunks):

                j_csize = n_rows_cols(j, self.col_chunks, self.ncols)
                right = left + (j_csize * abs(self.cellx))

                geom = Polygon([(left, bottom),
                                (left, top),
                                (right, top),
                                (right, bottom),
                                (left, bottom)])

                geometries.append(geom)

                left += self.col_chunks * abs(self.cellx)
            top -= self.row_chunks * abs(self.celly)

        return gpd.GeoDataFrame(data=list(range(1, len(geometries)+1)),
                                columns=['chunk'],
                                geometry=geometries,
                                crs=self._obj.crs)

    @property
    def __filenames(self):
        return self._filenames

    @__filenames.setter
    def __filenames(self, file_list):
        self._filenames = file_list

    @property
    def __data_are_separate(self):
        return self._data_bool

    @__data_are_separate.setter
    def __data_are_separate(self, data_bool):
        self._data_bool = data_bool

    @property
    def footprint_grid(self):
        """Get the image footprint grid"""
        return self._footprint_grid

    @footprint_grid.setter
    def footprint_grid(self, geometries):

        self._footprint_grid = gpd.GeoDataFrame(data=[fn for fn in self._obj.filename],
                                                columns=['footprint'],
                                                geometry=geometries,
                                                crs=self._obj.crs)

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
    def has_band_coord(self):
        return True if 'band' in self._obj.coords else False

    @property
    def has_band_dim(self):
        return True if self.ndims == 3 else False

    @property
    def has_band(self):
        return True if ('band' in self._obj.coords) and (self.ndims == 3) else False

    @property
    def has_time_coord(self):
        return True if 'time' in self._obj.coords else False

    @property
    def has_time_dim(self):
        return True if self.ndims == 4 else False

    @property
    def has_time(self):
        return True if ('time' in self._obj.coords) and (self.ndims == 4) else False

    @property
    def geodataframe(self):

        """
        Get a ``geopandas.GeoDataFrame`` of the array bounds
        """

        return gpd.GeoDataFrame(data=[Path(self._obj.filename).name if hasattr(self._obj, 'filename') else 1],
                                columns=['grid'],
                                geometry=[self._obj.gw.geometry],
                                crs=self._obj.crs)

    @property
    def unary_union(self):
        """Get a representation of the union of the image bounds"""
        return shapely.ops.unary_union(self.geometry)

    @property
    def meta(self):

        """
        Get the array metadata
        """

        Meta = namedtuple('Meta', 'left right top bottom bounds affine geometry')

        return Meta(left=self.left,
                    right=self.right,
                    top=self.top,
                    bottom=self.bottom,
                    bounds=self.bounds,
                    affine=Affine(*self._obj.transform),
                    geometry=self.geometry)
