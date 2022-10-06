import os
from pathlib import Path
from getpass import getpass
import math
import logging
import zipfile
from retry import retry

import requests
import yaml
from tqdm import tqdm
from cryptography.fernet import Fernet

from ..handler import add_handler


logger = logging.getLogger(__name__)
logger = add_handler(logger)

p = Path(__file__).absolute().parent

rgbn = str(p / 'rgbn.tif')
rgbn_suba = str(p / 'rgbn_suba.tif')
rgbn_subb = str(p / 'rgbn_subb.tif')
rgbn_20160101 = str(p / 'rgbn_20160101.tif')
rgbn_20160401 = str(p / 'rgbn_20160401.tif')
rgbn_20160517 = str(p / 'rgbn_20160517.tif')
rgbn_20170203 = str(p / 'rgbn_20170203.tif')

rgbn_time_list = [rgbn_20160101, rgbn_20160401, rgbn_20160517, rgbn_20170203]

l8_224077_20200518_B2 = str(p / 'LC08_L1TP_224077_20200518_20200518_01_RT_B2.TIF')
l8_224077_20200518_B3 = str(p / 'LC08_L1TP_224077_20200518_20200518_01_RT_B3.TIF')
l8_224077_20200518_B4 = str(p / 'LC08_L1TP_224077_20200518_20200518_01_RT_B4.TIF')
l8_224078_20200518_B2 = str(p / 'LC08_L1TP_224078_20200518_20200518_01_RT_B2.TIF')
l8_224078_20200518_B3 = str(p / 'LC08_L1TP_224078_20200518_20200518_01_RT_B3.TIF')
l8_224078_20200518_B4 = str(p / 'LC08_L1TP_224078_20200518_20200518_01_RT_B4.TIF')
l8_224078_20200518 = str(p / 'LC08_L1TP_224078_20200518_20200518_01_RT.TIF')
l8_224078_20200518_points = str(
    p / 'LC08_L1TP_224078_20200518_20200518_01_RT_points.gpkg'
)
l8_224078_20200518_polygons = str(
    p / 'LC08_L1TP_224078_20200518_20200518_01_RT_polygons.gpkg'
)
l3b_s2b_00390821jxn0l2a_20210319_20220730_c01 = str(
    p / 'L3B_S2B_00390821JXN0L2A_20210319_20220730_C01.nc'
)
l8_224078_20200127_meta = str(
    p
    / 'LC08_L2SP_224078_20200127_02_T1_LC08_L2SP_224078_20200127_20200823_02_T1_MTL.txt'
)
l7_225078_20110306_ang = str(
    p
    / 'LE07_L2SP_225078_20110306_02_T1_LE07_L2SP_225078_20110306_20200910_02_T1_ANG.txt'
)
l7_225078_20110306_SZA = str(p / 'LE07_L2SP_225078_20110306_02_T1_SZA.tif')
l7_225078_20110306_B1 = str(p / 'LE07_L2SP_225078_20110306_02_T1_B1.tif')
srtm30m_bounding_boxes = str(p / 'srtm30m_bounding_boxes.gpkg')
wrs2 = str(p / 'wrs2.tar.gz')


class PassKey(object):
    @staticmethod
    def create_key(key_file):

        key = Fernet.generate_key()

        with open(key_file, mode='w') as pf:
            yaml.dump({'key': key}, pf, default_flow_style=False)

    @staticmethod
    def create_passcode(key_file, passcode_file):

        """
        Args:
            key_file (str)
            passcode_file (str)
        """

        passcode = getpass()

        with open(key_file, mode='r') as pf:
            key = yaml.load(pf, Loader=yaml.FullLoader)

        cipher_suite = Fernet(key['key'])

        ciphered_text = cipher_suite.encrypt(passcode.encode())

        with open(passcode_file, mode='w') as pf:
            yaml.dump({'passcode': ciphered_text}, pf, default_flow_style=False)

    @staticmethod
    def load_passcode(key_file, passcode_file):

        with open(key_file, mode='r') as pf:
            key = yaml.load(pf, Loader=yaml.FullLoader)

        cipher_suite = Fernet(key['key'])

        with open(passcode_file, mode='r') as pf:
            ciphered_text = yaml.load(pf, Loader=yaml.FullLoader)

        return cipher_suite.decrypt(ciphered_text['passcode'])


class BaseDownloader(object):
    def download(self, url, outfile, safe_download=True):

        self.outpath = Path(outfile)

        if self.outpath.is_file():
            return

        if safe_download:
            base64_password = self.load_passcode(self.key_file, self.code_file).decode()

        chunk_size = 256 * 10240

        with requests.Session() as session:

            if safe_download:
                session.auth = (self.username, base64_password)

            # Open
            req = session.request('get', url)

            if safe_download:
                response = session.get(req.url, auth=(self.username, base64_password))
            else:
                response = session.get(req.url)

            if not response.ok:
                logger.exception('  Could not retrieve the page.')
                raise NameError

            if 'Content-Length' in response.headers:

                content_length = float(response.headers['Content-Length'])
                content_iters = int(math.ceil(content_length / chunk_size))
                chunk_size_ = chunk_size * 1

            else:

                content_iters = 1
                chunk_size_ = chunk_size * 1000

            with open(str(outfile), 'wb') as ofn:

                for data in tqdm(
                    response.iter_content(chunk_size=chunk_size_), total=content_iters
                ):
                    ofn.write(data)


class LUTDownloader(BaseDownloader):
    pass


class NASAEarthdataDownloader(PassKey, BaseDownloader):

    """A class to handle NASA Earthdata downloads.

    Args:
        username (str): The NASA Earthdata username.
        key_file (str): The NASA Earthdata secret key file.
        code_file (str): The NASA Earthdata secret code file.

    Account:
        A NASA Earthdata secret key and code file pair are required. First, create a login account at
        https://earthdata.nasa.gov/. Then, to generate a secret key/code pair do:

        >>> from geowombat.data import PassKey
        >>>
        >>> pk = PassKey()
        >>> pk.create_key('my.key')
        >>> pk.create_passcode('my.key', 'my.code')

        Use the key/code pair to download data.

        >>> from geowombat.data import NASAEarthdataDownloader
        >>>
        >>> nedd = NASAEarthdataDownloader('<user name>', 'my.key', 'my.code')
        >>>
        >>> # Download a SRTM elevation grid
        >>> nedd.download_srtm('n00e006', 'NASADEM_HGT_n00e006.zip')
        >>>
        >>> # Download a MODIS aersol HDF file
        >>> nedd.download_aerosol('MOD04_3K.A2020001.0025.061.2020002233531.hdf')
    """

    def __init__(self, username, key_file, code_file):

        self.username = username
        self.key_file = key_file
        self.code_file = code_file

        self.outpath = None

    def download_srtm(self, grid_id, outfile=None):

        if not outfile:
            outfile = f"NASADEM_HGT_{grid_id}.zip"

        @retry(zipfile.BadZipfile, tries=10, delay=5)
        def stream_zipfile():

            try:

                if Path(outfile).is_file():
                    z = zipfile.ZipFile(outfile)

                self.download(
                    f"https://e4ftl01.cr.usgs.gov/MEASURES/NASADEM_HGT.001/2000.02.11/NASADEM_HGT_{grid_id}.zip",
                    outfile,
                )

            except zipfile.BadZipfile:
                Path(outfile).unlink()
                raise zipfile.BadZipfile

        stream_zipfile()

    def download_aerosol(self, year, doy, outfile=None):

        if not outfile:
            outfile = f'{year}{int(doy):03d}'

        self.download(
            f"https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD04_3K/{year}/{doy}/",
            outfile,
        )
