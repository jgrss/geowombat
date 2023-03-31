import enum

import dask.array as da
import numpy as np
import xarray as xr


class QABits(enum.Enum):
    """QA bits.

    Reference:
        https://www.usgs.gov/landsat-missions/landsat-project-documents
    """

    landsat_c2_l2 = {
        'fill': 0,
        'dilated_cloud': 1,
        'cirrus': 2,
        'cloud': 3,
        'cloud_shadow': 4,
        'snow': 5,
        'clear': 6,
        'water': 7,
    }


class QAMasker(object):
    """A class for masking bit-packed quality flags.

    Args:
        qa (DataArray): The band quality array.
        sensor (str): The sensor name. Choices are ['ard', 'hls', 'l8-pre', 'l8-c1', 'l-c1', 'modis', 's2a', 's2c'].

            Codes:
                'ard':
                    `USGS Landsat Analysis Ready Data <https://www.usgs.gov/land-resources/nli/landsat/us-landsat-analysis-ready-data?qt-science_support_page_related_con=0#qt`-science_support_page_related_con>`_
                'hls':
                    `NASA Harmonized Landsat Sentinel <https://hls.gsfc.nasa.gov/>`_
                'l-c1':
                    Landsat Collection 1 L4-5 and L7
                'l8-c1':
                    Landsat Collection 1 L8
                's2a':
                    Sentinel 2A (surface reflectance)
                's2c':
                    Sentinel 2C (top of atmosphere)

        mask_items (str list): A list of items to mask.
        modis_qa_position (Optional[int]): The MODIS QA band position. Default is 1.
        modis_quality (Optional[int]): The MODIS quality level. Default is 2.
        confidence_level (Optional[str]): The confidence level. Choices are ['notdet', 'no', 'maybe', 'yes'].

    References:
        Landsat Collection 1:
            https://landsat.usgs.gov/collectionqualityband

    Examples:
        >>> import geowombat as gw
        >>> from geowombat.radiometry import QAMasker
        >>>
        >>> # Get the MODIS cloud mask.
        >>> with gw.open('qa.tif') as qa:
        >>>     mask = QAMasker(qs, 'modis').to_mask()
        >>>
        >>> # NASA HLS
        >>> with gw.open('qa.tif') as qa:
        >>>     mask = QAMasker(qs, 'hls', ['cloud']).to_mask()
    """

    def __init__(
        self,
        qa,
        sensor,
        mask_items=None,
        modis_qa_band=1,
        modis_quality=2,
        confidence_level='yes',
    ):

        self.qa = qa
        self.sensor = sensor
        self.modis_qa_band = modis_qa_band
        self.modis_quality = modis_quality
        self.mask_items = mask_items
        self.confidence_level = confidence_level

        self._set_dicts()

    def to_mask(self):

        """Converts QA bit-packed data to an integer mask.

        Returns:

            ``xarray.DataArray``:

                0: clear,
                1: water,
                2: shadow,
                3: snow or ice,
                4: cloud,
                5: cirrus cloud,
                6: adjacent cloud,
                7: saturated,
                8: dropped,
                9: terrain occluded,
                255: fill
        """

        if self.sensor == 'MODIS':
            mask = self._get_modis_qa_mask()
        else:

            mask = da.zeros(
                (self.qa.gw.nrows, self.qa.gw.ncols),
                chunks=(self.qa.gw.row_chunks, self.qa.gw.col_chunks),
                dtype='uint8',
            )

            for mask_item in self.mask_items:

                if mask_item in self.qa_flags[self.sensor]:

                    if 'conf' in mask_item:

                        # Has high confidence that
                        #   this condition was met.
                        mask_value = self.conf_dict[self.confidence_level]

                    else:
                        mask_value = 1

                    mask = da.where(
                        self._get_qa_mask(mask_item) >= mask_value,
                        self.fmask_dict[mask_item],
                        mask,
                    )

        if self.qa.gw.has_time:
            mask = xr.DataArray(
                mask,
                dims=('time', 'y', 'x'),
                coords={'time': self.qa.time, 'y': self.qa.y, 'x': self.qa.x},
                attrs=self.qa.attrs,
            )
        else:
            mask = xr.DataArray(
                mask,
                dims=('y', 'x'),
                coords={'y': self.qa.y, 'x': self.qa.x},
                attrs=self.qa.attrs,
            )

        mask = mask.expand_dims(dim='band')
        mask = mask.assign_coords(band=['mask'])

        for k, v in self.fmask_dict.items():
            mask.attrs[k] = v

        return mask

    def _set_dicts(self):

        self.fmask_dict = dict(
            clear=0,
            water=1,
            shadow=2,
            shadowconf=2,
            snow=3,
            snowice=3,
            snowiceconf=3,
            cloud=4,
            cloudconf=4,
            cirrus=5,
            cirrusconf=5,
            adjacent=6,
            saturated=7,
            dropped=8,
            terrain=0,
            fill=255,
        )

        self.conf_dict = dict(notdet=0, no=1, maybe=2, yes=3)

        self.qa_flags = {
            'hls': {
                'cirrus': (0, 0),
                'cloud': (1, 1),
                'adjacent': (2, 2),
                'shadow': (3, 3),
                'snowice': (4, 4),
                'water': (5, 5),
            },
            'l8-pre': {
                'cirrus': (13, 12),
                'snowice': (11, 10),
                'water': (5, 4),
                'fill': (0, 0),
                'dropped': (1, 1),
                'terrain': (2, 2),
                'shadow': (7, 6),
                'vegconf': (9, 8),
                'snowiceconf': (11, 10),
                'cirrusconf': (13, 12),
                'cloudconf': (15, 14),
            },
            'l8-c1': {
                'cirrusconf': (12, 11),
                'snowiceconf': (10, 9),
                'shadowconf': (8, 7),
                'cloudconf': (6, 5),
                'cloud': (4, 4),
                'saturated': (3, 2),
                'terrain': (1, 1),
                'fill': (0, 0),
            },
            'l-c1': {
                'fill': (0, 0),
                'dropped': (1, 1),
                'saturated': (3, 2),
                'cloud': (4, 4),
                'cloudconf': (6, 5),
                'shadowconf': (8, 7),
                'snowice': (10, 9),
            },
            'ard': {
                'fill': (0, 0),
                'clear': (1, 1),
                'water': (2, 2),
                'shadow': (3, 3),
                'snow': (4, 4),
                'cloud': (5, 5),
            },
            'modis': {
                'cloud': (0, 0),
                'daynight': (3, 3),
                'sunglint': (4, 4),
                'snowice': (5, 5),
                'landwater': (7, 6),
            },
            's2a': {'cloud': (10, 10), 'cirrus': (11, 11)},
            's2c': {'cloud': (10, 10), 'cirrus': (11, 11)},
        }

        self.modis_bit_shifts = {1: 0, 2: 4, 3: 8, 4: 12, 5: 16, 6: 20, 7: 24}

    def _qa_bits(self, mask_item):

        """
        Args:
            mask_item (str)

        For confidence bits:
            0 = not determined
            1 = no
            2 = maybe
            3 = yes
        """

        bit_location = self.qa_flags[self.sensor][mask_item]

        self.b1 = bit_location[0]
        self.b2 = bit_location[1]

    def _get_modis_qa_mask(self):

        """
        Reference:
            https://github.com/haoliangyu/pymasker/blob/master/pymasker.py
        """

        # `modis_mask`
        #   0: best quality
        #   1: good quality
        #   4: fill value
        #
        # `output`
        #   0: good data = clear
        #   255: bad data = fill
        return np.where(
            np.uint8(self.qa >> self.modis_bit_shifts[self.modis_qa_band] & 4)
            <= self.modis_quality,
            self.fmask_dict['clear'],
            self.fmask_dict['fill'],
        )

    def _get_qa_mask(self, mask_item):

        """
        Args:
            mask_item (str)

        Reference:
            https://github.com/mapbox/landsat8-qa/blob/master/landsat8_qa/qa.py
        """

        self._qa_bits(mask_item)

        width_int = int((self.b1 - self.b2 + 1) * '1', 2)

        return ((self.qa.data.squeeze() >> self.b2) & width_int).astype(
            'uint8'
        )
