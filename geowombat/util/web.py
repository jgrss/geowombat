import os
import shutil
import fnmatch
import tarfile
import subprocess
from pathlib import Path
from datetime import datetime
from collections import namedtuple
import time
import logging
import concurrent.futures

from ..handler import add_handler
from ..radiometry import BRDF, LinearAdjustments, RadTransforms, landsat_pixel_angles, sentinel_pixel_angles, QAMasker, DOS
from ..radiometry.angles import estimate_cloud_shadows
from ..core.properties import get_sensor_info
from ..core import ndarray_to_xarray
from ..backends.gdal_ import warp

import geowombat as gw

import numpy as np
from osgeo import gdal
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import Polygon
from joblib import Parallel, delayed

try:
    import requests

    REQUESTS_INSTALLED = True
except:
    REQUESTS_INSTALLED = False

try:

    from s2cloudless import S2PixelCloudDetector

    S2CLOUDLESS_INSTALLED = True

except:
    S2CLOUDLESS_INSTALLED = False

logger = logging.getLogger(__name__)
logger = add_handler(logger)


RESAMPLING_DICT = dict(bilinear=gdal.GRA_Bilinear,
                       cubic=gdal.GRA_Cubic,
                       nearest=gdal.GRA_NearestNeighbour)

OrbitDates = namedtuple('OrbitDates', 'start end')
FileInfo = namedtuple('FileInfo', 'name key')
GoogleFileInfo = namedtuple('GoogleFileInfo', 'url url_file meta angles')


def _rmdir(pathdir):

    """
    Removes a directory path
    """

    if pathdir.is_dir():

        for child in pathdir.iterdir():

            if child.is_file():

                try:
                    child.unlink()
                except:
                    pass

        try:
            pathdir.rmdir()
        except:

            try:
                shutil.rmtree(str(pathdir))
            except:
                pass


def _delayed_read(fn):

    attempt = 0
    max_attempts = 10

    while True:

        if Path(fn).is_file():
            break
        else:
            time.sleep(2)

        attempt += 1

        if attempt >= max_attempts:
            break

    with open(str(fn), mode='r') as tx:
        lines = tx.readlines()

    return lines


def _update_status_file(fn, log_name):

    attempt = 0
    max_attempts = 10

    while True:

        wait_on_file = False

        # Check if the file is open by another process
        for proc in psutil.process_iter():

            try:

                for item in proc.open_files():

                    if item.path == str(fn):
                        wait_on_file = True
                        break

            except Exception:
                pass

            if wait_on_file:
                break

        if wait_on_file:
            time.sleep(2)
        else:
            break

        attempt += 1

        if attempt >= max_attempts:
            break

    with open(str(fn), mode='r') as tx:

        lines = tx.readlines()

        if lines:
            lines = list(set(lines))

        if log_name + '\n' not in lines:
            lines.append(log_name + '\n')

    fn.unlink()

    with open(str(fn), mode='w') as tx:
        tx.writelines(lines)


def _clean_and_update(outdir_angles,
                      finfo_dict,
                      meta_name,
                      check_angles=True,
                      check_downloads=True,
                      load_bands_names=None):

    if check_angles:
        _rmdir(outdir_angles)

    if check_downloads:

        for k, v in finfo_dict.items():

            if Path(v.name).is_file():

                try:
                    Path(v.name).unlink()
                except Warning:
                    logger.warning('  Could not delete {}.'.format(v.name))

            else:
                logger.warning('  The {} file does not exist to delete.'.format(v.name))

    # if update_status:
    #     _update_status_file(status, meta_name)

    if load_bands_names:

        for loaded_band in load_bands_names:

            if Path(loaded_band).is_file():

                try:
                    Path(loaded_band).unlink()
                except Warning:
                    logger.warning('  Could not delete {}.'.format(loaded_band))


def _assign_attrs(data, attrs, bands_out):

    if bands_out:
        data = data.sel(band=bands_out)

    data = data.transpose('band', 'y', 'x')
    data.attrs = attrs

    return data


def _parse_google_filename(filename, landsat_parts, sentinel_parts, public_url):

    file_info = GoogleFileInfo(url=None, url_file=None, meta=None, angles=None)

    f_base, f_ext = os.path.splitext(filename)

    fn_parts = f_base.split('_')

    if fn_parts[0].lower() in landsat_parts:

        # Collection 1
        url_ = '{PUBLIC}-landsat/{SENSOR}/01/{PATH}/{ROW}/{FDIR}'.format(PUBLIC=public_url,
                                                                         SENSOR=fn_parts[0],
                                                                         PATH=fn_parts[2][:3],
                                                                         ROW=fn_parts[2][3:],
                                                                         FDIR='_'.join(fn_parts[:-1]))

        url_filename = '{URL}/{FN}'.format(URL=url_, FN=filename)
        url_meta = '{URL}/{FN}_MTL.txt'.format(URL=url_, FN='_'.join(fn_parts[:-1]))
        url_angles = '{URL}/{FN}_ANG.txt'.format(URL=url_, FN='_'.join(fn_parts[:-1]))

        file_info = GoogleFileInfo(url=url_,
                                   url_file=url_filename,
                                   meta=url_meta,
                                   angles=url_angles)

    return file_info


def _download_workers(gcp_str, poutdir, outdir, fname, fn, null_items, verbose):

    # Renaming Sentinel data
    rename = False

    # Full path of GCP local download
    down_file = str(poutdir.joinpath(fname))

    if down_file.endswith('_ANG.txt'):
        fbase = fname.replace('_ANG.txt', '')
        key = 'angle'
    elif down_file.endswith('_MTL.txt'):
        fbase = fname.replace('_MTL.txt', '')
        key = 'meta'
    elif down_file.endswith('MTD_TL.xml'):

        fbase = Path(fn).parent.name
        down_file = str(poutdir.joinpath(fbase + '_MTD_TL.xml'))
        key = 'meta'
        rename = True

    elif down_file.endswith('_BQA.TIF'):
        fbase = fname.replace('_BQA.TIF', '')
        key = 'qa'
    else:

        if fname.endswith('.jp2'):

            fbase = Path(fn).parent.parent.name
            key = Path(fn).name.split('.')[0].split('_')[-1]
            down_file = str(poutdir.joinpath(fbase + '_' + key + '.jp2'))
            rename = True

        else:

            fsplit = fname.split('_')
            fbase = '_'.join(fsplit[:-1])
            key = fsplit[-1].split('.')[0]

        # TODO: QA60

    continue_download = True

    if fbase in null_items:
        continue_download = False

    if continue_download:

        ###################
        # Download the file
        ###################

        if not Path(down_file).is_file():

            if fn.lower().startswith('gs://gcp-public-data'):
                com = 'gsutil cp -r {} {}'.format(fn, outdir)
            else:
                com = 'gsutil cp -r {}/{} {}'.format(gcp_str, fn, outdir)

            if verbose > 0:
                logger.info('  Downloading {} ...'.format(fname))

            subprocess.call(com, shell=True)

            if rename:
                os.rename(str(Path(outdir).joinpath(Path(fn).name)), down_file)

        # Store file information
        return key, FileInfo(name=down_file, key=key)

    else:
        return None, None


class DownloadMixin(object):

    def download_gcp(self,
                     sensor,
                     downloads=None,
                     outdir='.',
                     outdir_brdf=None,
                     search_wildcards=None,
                     search_dict=None,
                     n_jobs=1,
                     verbose=0):

        """
        Downloads a file from Google Cloud platform

        Args:
            sensor (str): The sensor to query. Choices are ['l5', 'l7', 'l8', 's2a', 's2c'].
            downloads (Optional[str or list]): The file or list of keys to download. If not given, keys will be taken
                from ``search_dict`` or ``self.search_dict``.
            outdir (Optional[str | Path]): The output directory.
            outdir_brdf (Optional[Path]): The output directory.
            search_wildcards (Optional[list]): A list of search wildcards.
            search_dict (Optional[dict]): A keyword search dictionary to override ``self.search_dict``.
            n_jobs (Optional[int]): The number of files to download in parallel.
            verbose (Optional[int]): The verbosity level.

        Returns:
            ``dict`` of ``dicts``
                where sub-dictionaries contain a ``namedtuple`` of the downloaded file and tag
        """

        if not search_dict:

            if not self.search_dict:
                logger.exception('  A keyword search dictionary must be provided, either from `self.list_gcp` or the `search_dict` argument.')
            else:
                search_dict = self.search_dict

        poutdir = Path(outdir)

        if outdir != '.':
            poutdir.mkdir(parents=True, exist_ok=True)

        if not downloads:
            downloads = list(search_dict.keys())

        if not isinstance(downloads, list):
            downloads = [downloads]

        if sensor in ['s2', 's2a', 's2b', 's2c']:
            gcp_str = 'gsutil cp -r gs://gcp-public-data-sentinel-2'
        else:
            gcp_str = 'gsutil cp -r gs://gcp-public-data-landsat'

        downloaded = {}
        null_items = []

        for search_key in downloads:

            download_list = self.search_dict[search_key]

            if search_wildcards:

                download_list_ = []

                for swild in search_wildcards:
                    download_list_ += fnmatch.filter(download_list, '*{}'.format(swild))

                download_list = download_list_

            download_list_names = [Path(dfn).name for dfn in download_list]

            logger.info('  The download contains {:d} items: {}'.format(len(download_list_names), ','.join(download_list_names)))

            # Separate each scene
            if sensor.lower() in ['l5', 'l7', 'l8']:

                # list of file ids
                id_list = ['_'.join(fn.split('_')[:-1]) for fn in download_list_names if fn.endswith('_MTL.txt')]

                # list of lists where each sub-list is unique
                download_list_unique = [[fn for fn in download_list if sid in Path(fn).name] for sid in id_list]

            else:

                id_list = list(set(['_'.join(fn.split('_')[:-1]) for fn in download_list_names]))
                download_list_unique = [download_list]

            for scene_id, sub_download_list in zip(id_list, download_list_unique):

                logger.info('  Checking scene {} ...'.format(scene_id))

                downloaded_sub = {}

                # Check if the file has been downloaded
                if sensor.lower() in ['l5', 'l7', 'l8']:

                    if not scene_id.lower().startswith(self.sensor_collections[sensor.lower()]):

                        logger.exception('  The scene id {SCENE_ID} does not match the sensor {SENSOR}.'.format(SCENE_ID=scene_id,
                                                                                                                SENSOR=sensor))
                        raise NameError

                    # Path of BRDF stack
                    out_brdf = outdir_brdf.joinpath(scene_id + '.tif')

                else:

                    fn = sub_download_list[0]
                    fname = Path(fn).name

                    if fname.lower().endswith('.jp2'):

                        fbase = Path(fn).parent.parent.name
                        key = Path(fn).name.split('.')[0].split('_')[-1]
                        down_file = str(poutdir.joinpath(fbase + '_' + key + '.jp2'))

                        brdfp = '_'.join(Path(down_file).name.split('_')[:-1])
                        out_brdf = outdir_brdf.joinpath(brdfp + '_MTD.tif')

                    else:
                        out_brdf = None

                if out_brdf:

                    if out_brdf.is_file() or \
                            Path(str(out_brdf).replace('.tif', '.nc')).is_file() or \
                            Path(str(out_brdf).replace('.tif', '.nodata')).is_file():

                        logger.warning(f'  The output BRDF file, {str(out_brdf)}, already exists.')
                        _clean_and_update(None, None, None, check_angles=False, check_downloads=False)
                        continue

                    else:
                        logger.warning(f'  Continuing with the download for {str(out_brdf)}.')

                # Move the metadata file to the front of the
                # list to avoid unnecessary downloads.
                if sensor.lower() in ['l5', 'l7', 'l8']:
                    meta_index = [i for i in range(0, len(sub_download_list)) if sub_download_list[i].endswith('_MTL.txt')][0]
                    sub_download_list.insert(0, sub_download_list.pop(meta_index))
                else:
                    # The Sentinel 2 metadata files come in their own list
                    pass

                download_list_names = [Path(dfn).name for dfn in sub_download_list]

                results = []

                with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:

                    futures = [executor.submit(_download_workers,
                                               gcp_str,
                                               poutdir,
                                               outdir,
                                               fname,
                                               fn,
                                               null_items,
                                               verbose) for fname, fn in zip(download_list_names, sub_download_list)]

                    for f in concurrent.futures.as_completed(futures):
                        results.append(f.result())

                for key, finfo_ in results:

                    if finfo_:
                        downloaded_sub[key] = finfo_

                if downloaded_sub:

                    if len(downloaded_sub) < len(sub_download_list):

                        downloaded_names = [Path(v.name).name for v in list(downloaded_sub.values())]
                        missing_items = ','.join(list(set(download_list_names).difference(downloaded_names)))

                        logger.warning('  Only {:d} files out of {:d} were downloaded.'.format(len(downloaded_sub), len(sub_download_list)))
                        logger.warning('  {} are missing.'.format(missing_items))

                    downloaded[search_key] = downloaded_sub

        return downloaded

    def download_aws(self,
                     landsat_id,
                     band_list,
                     outdir='.'):

        """
        Downloads Landsat 8 data from Amazon AWS

        Args:
            landsat_id (str): The Landsat id to download.
            band_list (list): The Landsat bands to download.
            outdir (Optional[str]): The output directory.

        Examples:
            >>> from geowombat.util import GeoDownloads
            >>>
            >>> dl = GeoDownloads()
            >>> dl.download_aws('LC08_L1TP_224077_20200518_20200518_01_RT', ['b2', 'b3', 'b4'])
        """

        if not REQUESTS_INSTALLED:
            logger.exception('Requests must be installed.')

        if not isinstance(outdir, Path):
            outdir = Path(outdir)

        parts = landsat_id.split('_')

        path_row = parts[2]
        path = int(path_row[:3])
        row = int(path_row[3:])

        def _download_file(in_file, out_file):

            response = requests.get(in_file)

            with open(out_file, 'wb') as f:
                f.write(response.content)

        mtl_id = '{landsat_id}_MTL.txt'.format(landsat_id=landsat_id)

        url = '{aws_l8_public}/{path:03d}/{row:03d}/{landsat_id}/{mtl_id}'.format(aws_l8_public=self.aws_l8_public,
                                                                                  path=path,
                                                                                  row=row,
                                                                                  landsat_id=landsat_id,
                                                                                  mtl_id=mtl_id)

        mtl_out = outdir / mtl_id

        _download_file(url, str(mtl_out))

        angle_id = '{landsat_id}_ANG.txt'.format(landsat_id=landsat_id)

        url = '{aws_l8_public}/{path:03d}/{row:03d}/{landsat_id}/{angle_id}'.format(aws_l8_public=self.aws_l8_public,
                                                                                    path=path,
                                                                                    row=row,
                                                                                    landsat_id=landsat_id,
                                                                                    angle_id=angle_id)

        angle_out = outdir / angle_id

        _download_file(url, str(angle_out))

        for band in band_list:

            band_id = '{landsat_id}_{band}.TIF'.format(landsat_id=landsat_id,
                                                       band=band.upper())

            url = '{aws_l8_public}/{path:03d}/{row:03d}/{landsat_id}/{band_id}'.format(aws_l8_public=self.aws_l8_public,
                                                                                       path=path,
                                                                                       row=row,
                                                                                       landsat_id=landsat_id,
                                                                                       band_id=band_id)
            band_out = outdir / band_id

            _download_file(url, str(band_out))

    # def download_landsat_range(self, sensors, bands, path_range, row_range, date_range, **kwargs):
    #
    #     """
    #     Downloads Landsat data from iterables
    #
    #     Args:
    #         sensors (str): A list of sensors to download.
    #         bands (str): A list of bands to download.
    #         path_range (iterable): A list of paths.
    #         row_range (iterable): A list of rows.
    #         date_range (iterable): A list of ``datetime`` objects or a list of strings as yyyymmdd.
    #         kwargs (Optional[dict]): Keyword arguments to pass to ``download``.
    #
    #     Examples:
    #         >>> from geowombat.util import GeoDownloads
    #         >>>
    #         >>> dl = GeoDownloads()
    #         >>> dl.download_landsat_range(['lc08'], ['b4'], [42], [34], ['20170616', '20170620'])
    #     """
    #
    #     if (len(date_range) == 2) and not isinstance(date_range[0], datetime):
    #         start_date = date_range[0]
    #         end_date = date_range[1]
    #
    #         sdt = datetime.strptime(start_date, '%Y%m%d')
    #         edt = datetime.strptime(end_date, '%Y%m%d')
    #
    #         date_range = pd.date_range(start=sdt, end=edt).to_pydatetime().tolist()
    #
    #     for sensor in sensors:
    #         for band in bands:
    #             for path in path_range:
    #                 for row in row_range:
    #                     for dt in date_range:
    #                         str_date = '{:d}{:02d}{:02d}'.format(dt.year, dt.month, dt.day)
    #
    #                         # TODO: check if L1TP is used for all sensors
    #                         # TODO: fixed DATE2
    #                         filename = '{SENSOR}_L1TP_{PATH:03d}{ROW:03d}_{DATE}_{DATE2}_01_T1_{BAND}.TIF'.format(
    #                             SENSOR=sensor.upper(),
    #                             PATH=path,
    #                             ROW=row,
    #                             DATE=str_date,
    #                             DATE2=None,
    #                             BAND=band)
    #
    #                         self.download(filename, **kwargs)


class CloudPathMixin(object):

    @staticmethod
    def get_landsat_urls(scene_id, bands=None, cloud='gcp'):

        """
        Gets Google Cloud Platform COG urls for Landsat

        Args:
            scene_id (str): The Landsat scene id.
            bands (Optional[list]): The list of band names.
            cloud (Optional[str]): The cloud strorage to get the URL from. For now, only 'gcp' is supported.

        Returns:
            ``tuple`` of band URLs and metadata URL as strings

        Example:
            >>> import os
            >>> import geowombat as gw
            >>> from geowombat.util import GeoDownloads
            >>>
            >>> os.environ['CURL_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'
            >>>
            >>> gdl = GeoDownloads()
            >>>
            >>> scene_urls, meta_url = gdl.get_landsat_urls('LC08_L1TP_042034_20171225_20180103_01_T1',
            >>>                                             bands=['blue', 'green', 'red'])
            >>>
            >>> with gw.open(urls) as src:
            >>>     print(src)
        """

        gcp_base = 'https://storage.googleapis.com/gcp-public-data-landsat'

        sensor_collection, level, path_row, date_acquire, date_other, collection, tier = scene_id.split('_')
        path = path_row[:3]
        row = path_row[3:]

        if bands:

            sensor = f'{sensor_collection[0].lower()}{sensor_collection[3]}'

            # Landsat 7 has the thermal band
            sensor = 'l7th' if sensor == 'l7' else sensor

            wavelengths = get_sensor_info(key='wavelength', sensor=sensor)
            band_pos = [getattr(wavelengths, b) for b in bands]

        else:
            band_pos = [1]

        lid = f'{sensor_collection}/01/{path}/{row}/{scene_id}'

        scene_urls = [f'{gcp_base}/{lid}/{scene_id}_B{band_pos}.TIF' for band_pos in band_pos]
        meta_url = f'{gcp_base}/{lid}/{scene_id}_MTL.txt'

        return scene_urls, meta_url

    @staticmethod
    def get_sentinel2_urls(safe_id, bands=None, cloud='gcp'):

        """
        Gets Google Cloud Platform COG urls for Sentinel 2

        Args:
            safe_id (str): The Sentinel 2 SAFE id.
            bands (Optional[list]): The list of band names.
            cloud (Optional[str]): The cloud strorage to get the URL from. For now, only 'gcp' is supported.

        Returns:
            ``tuple`` of band URLs and metadata URL as strings

        Example:
            >>> import os
            >>> import geowombat as gw
            >>> from geowombat.util import GeoDownloads
            >>>
            >>> os.environ['CURL_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'
            >>>
            >>> gdl = GeoDownloads()
            >>>
            >>> safe_id = 'S2A_MSIL1C_20180109T135101_N0206_R024_T21HUD_20180109T171608.SAFE/GRANULE/L1C_T21HUD_A013320_20180109T135310'
            >>>
            >>> scene_urls, meta_url = gdl.get_sentinel2_urls(safe_id,
            >>>                                               bands=['blue', 'green', 'red', 'nir'])
            >>>
            >>> with gw.open(urls) as src:
            >>>     print(src)
        """

        gcp_base = 'https://storage.googleapis.com/gcp-public-data-sentinel-2'

        sensor, level, date, __, __, mgrs, __ = safe_id.split('/')[0].split('_')
        utm = mgrs[1:3]
        zone = mgrs[3]
        id_ = mgrs[4:]

        if bands:

            sensor = sensor.lower()
            wavelengths = get_sensor_info(key='wavelength', sensor=sensor)
            band_pos = [getattr(wavelengths, b) for b in bands]

        else:
            band_pos = [1]

        lid = f'{utm}/{zone}/{id_}/{safe_id}/IMG_DATA/{mgrs}_{date}'

        scene_urls = [f'{gcp_base}/tiles/{lid}_B{band_pos:02d}.jp2' for band_pos in band_pos]
        meta_url = f'{utm}/{zone}/{id_}/{safe_id}/MTD_TL.xml'

        return scene_urls, meta_url


class GeoDownloads(CloudPathMixin, DownloadMixin):

    def __init__(self):

        self._gcp_search_dict = None
        self.search_dict = None

        self.gcp_public = 'https://storage.googleapis.com/gcp-public-data'
        self.aws_l8_public = 'https://landsat-pds.s3.amazonaws.com/c1/L8'

        self.landsat_parts = ['lt05', 'le07', 'lc08']
        self.sentinel_parts = ['s2a', 's2b']

        s2_dict = dict(coastal=1,
                       blue=2,
                       green=3,
                       red=4,
                       nir1=5,
                       nir2=6,
                       nir3=7,
                       nir=8,
                       rededge=8,
                       water=9,
                       cirrus=10,
                       swir1=11,
                       swir2=12)

        self.gcp_dict = dict(l5='LT05/01',
                             l7='LE07/01',
                             l8='LC08/01',
                             s2='tiles',
                             s2a='tiles',
                             s2b='tiles',
                             s2c='tiles')

        self.sensor_collections = dict(l5='lt05',
                                       l7='le07',
                                       l8='lc08')

        self.orbit_dates = dict(l5=OrbitDates(start=datetime.strptime('1984-3-1', '%Y-%m-%d'),
                                              end=datetime.strptime('2013-6-5', '%Y-%m-%d')),
                                l7=OrbitDates(start=datetime.strptime('1999-4-15', '%Y-%m-%d'),
                                              end=datetime.strptime('2100-1-1', '%Y-%m-%d')),
                                l8=OrbitDates(start=datetime.strptime('2013-2-11', '%Y-%m-%d'),
                                              end=datetime.strptime('2100-1-1', '%Y-%m-%d')),
                                s2a=OrbitDates(start=datetime.strptime('2015-6-23', '%Y-%m-%d'),
                                               end=datetime.strptime('2100-1-1', '%Y-%m-%d')),
                                s2b=OrbitDates(start=datetime.strptime('2017-3-7', '%Y-%m-%d'),
                                               end=datetime.strptime('2100-1-1', '%Y-%m-%d')))

        self.associations = dict(l5=dict(blue=1,
                                         green=2,
                                         red=3,
                                         nir=4,
                                         swir1=5,
                                         thermal=6,
                                         swir2=7),
                                 l7=dict(blue=1,
                                         green=2,
                                         red=3,
                                         nir=4,
                                         swir1=5,
                                         thermal=6,
                                         swir2=7,
                                         pan=8),
                                 l8=dict(coastal=1,
                                         blue=2,
                                         green=3,
                                         red=4,
                                         nir=5,
                                         swir1=6,
                                         swir2=7,
                                         pan=8,
                                         cirrus=9,
                                         tirs1=10,
                                         tirs2=11),
                                 s2=s2_dict,
                                 s2a=s2_dict,
                                 s2b=s2_dict,
                                 s2c=s2_dict)

    def download_cube(self,
                      sensors,
                      date_range,
                      bounds,
                      bands,
                      bands_out=None,
                      crs=None,
                      out_bounds=None,
                      outdir='.',
                      resampling='bilinear',
                      ref_res=None,
                      l57_angles_path=None,
                      l8_angles_path=None,
                      subsample=1,
                      write_format='gtiff',
                      write_angle_files=False,
                      mask_qa=False,
                      lqa_mask_items=None,
                      chunks=512,
                      cloud_heights=None,
                      sr_method='srem',
                      earthdata_username=None,
                      earthdata_key_file=None,
                      earthdata_code_file=None,
                      srtm_outdir=None,
                      n_jobs=1,
                      num_workers=1,
                      num_threads=1,
                      **kwargs):

        """
        Downloads a cube of Landsat and/or Sentinel 2 imagery

        Args:
            sensors (str or list): The sensors, or sensor, to download.
            date_range (list): The date range, given as [date1, date2], where the date format is yyyy-mm.
            bounds (GeoDataFrame, list, or tuple): The geometry bounds (in WGS84 lat/lon) that define the cube extent
                to download. If given as a ``GeoDataFrame``, only the first ``DataFrame`` record will be used.
                If given as a ``tuple`` or a ``list``, the order should be (left, bottom, right, top).
            bands (str or list): The bands to download.

                E.g.:

                    Sentinel s2cloudless bands:
                        bands = ['coastal', 'blue', 'red', 'nir1', 'nir', 'rededge', 'water', 'cirrus', 'swir1', 'swir2']

            bands_out (Optional[list]): The bands to write to file. This might be useful after downloading all bands to
                mask clouds, but are only interested in subset of those bands.
            crs (Optional[str or object]): The output CRS. If ``bounds`` is a ``GeoDataFrame``, the CRS is taken
                from the object.
            out_bounds (Optional[list or tuple]): The output bounds in ``crs``. If not given, the bounds are
                taken from ``bounds``.
            outdir (Optional[str]): The output directory.
            ref_res (Optional[tuple]): A reference cell resolution.
            resampling (Optional[str]): The resampling method.
            l57_angles_path (str): The path to the Landsat 5 and 7 angles bin.
            l8_angles_path (str): The path to the Landsat 8 angles bin.
            subsample (Optional[int]): The sub-sample factor when calculating the angles.
            write_format (Optional[bool]): The data format to write. Choices are ['gtiff', 'netcdf'].
            write_angle_files (Optional[bool]): Whether to write the angles to file.
            mask_qa (Optional[bool]): Whether to mask data with the QA file.
            lqa_mask_items (Optional[list]): A list of QA mask items for Landsat.
            chunks (Optional[int]): The chunk size to read at.
            cloud_heights (Optional[list]): The cloud heights, in kilometers.
            sr_method (Optional[str]): The surface reflectance correction method. Choices are ['srem', '6s'].
            earthdata_username (Optional[str]): The EarthData username.
            earthdata_key_file (Optional[str]): The EarthData secret key file.
            earthdata_code_file (Optional[str]): The EarthData secret passcode file.
            srtm_outdir (Optional[str]): The output SRTM directory.
            n_jobs (Optional[int]): The number of parallel download workers for ``joblib``.
            num_workers (Optional[int]): The number of parallel workers for ``dask.compute``.
            num_threads (Optional[int]): The number of GDAL warp threads.
            kwargs (Optional[dict]): Keyword arguments passed to ``to_raster``.

        Examples:
            >>> from geowombat.util import GeoDownloads
            >>> gdl = GeoDownloads()
            >>>
            >>> # Download a Landsat 7 panchromatic cube
            >>> gdl.download_cube(['l7'],
            >>>                   ['2010-01-01', '2010-02-01'],
            >>>                   (-91.57, 40.37, -91.46, 40.42),
            >>>                   ['pan'],
            >>>                   crs="+proj=aea +lat_1=-5 +lat_2=-42 +lat_0=-32 +lon_0=-60 +x_0=0 +y_0=0 +ellps=aust_SA +units=m +no_defs")
            >>>
            >>> # Download a Landsat 7, 8 and Sentinel 2 cube of the visible spectrum
            >>> gdl.download_cube(['l7', 'l8', 's2a'],
            >>>                   ['2017-01-01', '2018-01-01'],
            >>>                   (-91.57, 40.37, -91.46, 40.42),
            >>>                   ['blue', 'green', 'red'],
            >>>                   crs={'init': 'epsg:102033'},
            >>>                   readxsize=1024,
            >>>                   readysize=1024,
            >>>                   n_workers=1,
            >>>                   n_threads=8)
        """

        if write_format not in ['gtiff', 'netcdf']:
            logger.warning(f'  Did not recognize {write_format}. Setting the output data format as gtiff.')
            write_format = 'gtiff'

        if not lqa_mask_items:

            lqa_mask_items = ['fill',
                              'saturated',
                              'cloudconf',
                              'shadowconf',
                              'cirrusconf']

        if isinstance(sensors, str):
            sensors = [sensors]

        angle_kwargs = kwargs.copy()
        angle_kwargs['nodata'] = -32768

        nodataval = kwargs['nodata'] if 'nodata' in kwargs else 65535

        angle_infos = {}

        rt = RadTransforms()
        br = BRDF()
        la = LinearAdjustments()
        dos = DOS()

        main_path = Path(outdir)

        outdir_tmp = main_path.joinpath('tmp')
        outdir_brdf = main_path.joinpath('brdf')

        main_path.mkdir(parents=True, exist_ok=True)
        outdir_tmp.mkdir(parents=True, exist_ok=True)
        outdir_brdf.mkdir(parents=True, exist_ok=True)

        # Logging file
        # status = Path(outdir).joinpath('status.txt')
        #
        # if not status.is_file():
        #
        #     with open(str(status), mode='w') as tx:
        #         pass

        # Get bounds from geometry
        if isinstance(bounds, tuple) or isinstance(bounds, list):

            bounds = Polygon([(bounds[0], bounds[3]),  # upper left
                              (bounds[2], bounds[3]),  # upper right
                              (bounds[2], bounds[1]),  # lower right
                              (bounds[0], bounds[1]),  # lower left
                              (bounds[0], bounds[3])])  # upper left

            bounds = gpd.GeoDataFrame([0],
                                      geometry=[bounds],
                                      crs={'init': 'epsg:4326'})

        bounds_object = bounds.geometry.values[0]

        if not out_bounds:

            # Project the bounds
            out_bounds = bounds.to_crs(crs).bounds.values[0].tolist()

        # Get WRS file
        data_bin = os.path.realpath(os.path.dirname(__file__))

        data_dir = Path(data_bin).joinpath('../data')

        shp_dict = {}

        if ('l5' in sensors) or ('l7' in sensors) or ('l8' in sensors):

            path_tar = Path(data_dir).joinpath('wrs2.tar.gz')
            path_shp = Path(data_dir).joinpath('wrs2_descending.shp')
            wrs = os.path.realpath(path_shp.as_posix())

            if not path_shp.is_file():

                with tarfile.open(os.path.realpath(path_tar.as_posix()), mode='r:gz') as tf:
                    tf.extractall(data_dir.as_posix())

            df_wrs = gpd.read_file(wrs)
            df_wrs = df_wrs[df_wrs.geometry.intersects(bounds_object)]

            if df_wrs.empty:
                logger.warning('  The geometry bounds is empty.')
                return

            shp_dict['wrs'] = df_wrs

        if ('s2a' in sensors) or ('s2b' in sensors) or ('s2c' in sensors):

            path_tar = Path(data_dir).joinpath('mgrs.tar.gz')
            path_shp = Path(data_dir).joinpath('sentinel2_grid.shp')
            mgrs = os.path.realpath(path_shp.as_posix())

            if not path_shp.is_file():

                with tarfile.open(os.path.realpath(path_tar.as_posix()), mode='r:gz') as tf:
                    tf.extractall(data_dir.as_posix())

            df_mgrs = gpd.read_file(mgrs)
            df_mgrs = df_mgrs[df_mgrs.geometry.intersects(bounds_object)]

            if df_mgrs.empty:
                logger.warning('  The geometry bounds is empty.')
                return

            shp_dict['mgrs'] = df_mgrs

        dt1 = datetime.strptime(date_range[0], '%Y-%m')
        dt2 = datetime.strptime(date_range[1], '%Y-%m')

        months = list(range(1, 13))
        year_months = {}

        if dt1.month <= dt2.month:
            month_range = months[months.index(dt1.month):months.index(dt2.month) + 1]
        else:
            month_range = months[months.index(dt1.month):] + months[:months.index(dt2.month) + 1]

        if dt1.year == dt2.year:
            year_months[dt1.year] = month_range
        else:

            for y in range(dt1.year, dt2.year + 1):

                if y == dt1.year:
                    year_months[y] = list(range(dt1.month, 13))
                elif y == dt2.year:
                    year_months[y] = list(range(1, dt2.month + 1))
                else:
                    year_months[y] = months

        year = dt1.year

        while True:

            if year > dt2.year:
                break

            for m in year_months[year]:

                yearmonth_query = '{:d}{:02d}'.format(year, m)

                target_date = datetime.strptime(yearmonth_query, '%Y%m')

                for sensor in sensors:

                    # Avoid unnecessary GCP queries
                    if (target_date < self.orbit_dates[sensor.lower()].start) or \
                            (target_date > self.orbit_dates[sensor.lower()].end):

                        continue

                    band_associations = self.associations[sensor]

                    if sensor.lower() in ['s2', 's2a', 's2b', 's2c']:

                        locations = ['{}/{}/{}'.format(dfrow.Name[:2], dfrow.Name[2], dfrow.Name[3:])
                                     for dfi, dfrow in shp_dict['mgrs'].iterrows()]

                    else:

                        locations = ['{:03d}/{:03d}'.format(int(dfrow.PATH), int(dfrow.ROW))
                                     for dfi, dfrow in shp_dict['wrs'].iterrows()]

                    for location in locations:

                        if sensor.lower() in ['s2', 's2a', 's2b', 's2c']:

                            query = '{LOCATION}/{LEVEL}*{YM}*.SAFE/GRANULE/*'.format(LOCATION=location,
                                                                                     LEVEL=sensor.upper(),
                                                                                     YM=yearmonth_query)

                        else:

                            query = '{LOCATION}/*{PATHROW}_{YM}*_T1'.format(LOCATION=location,
                                                                            PATHROW=location.replace('/', ''),
                                                                            YM=yearmonth_query)

                        # Query and list available files on the GCP
                        self.list_gcp(sensor, query)

                        self.search_dict = self.get_gcp_results

                        if not self.search_dict:

                            logger.warning(
                                '  No results found for {SENSOR} at location {LOC}, year {YEAR:d}, month {MONTH:d}.'.format(
                                    SENSOR=sensor,
                                    LOC=location,
                                    YEAR=year,
                                    MONTH=m))

                            continue

                        # Download data
                        if sensor.lower() in ['s2', 's2a', 's2b', 's2c']:

                            load_bands = [f'B{band_associations[bd]:02d}' if bd != 'rededge' else
                                          f'B{band_associations[bd]:01d}A' for bd in bands]

                            search_wildcards = ['MTD_TL.xml'] + [bd + '.jp2' for bd in load_bands]

                            file_info = self.download_gcp(sensor,
                                                          outdir=outdir_tmp,
                                                          outdir_brdf=outdir_brdf,
                                                          search_wildcards=search_wildcards,
                                                          n_jobs=n_jobs,
                                                          verbose=1)

                            # Reorganize the dictionary to combine bands and metadata
                            new_dict_ = {}
                            for finfo_key, finfo_dict in file_info.items():

                                sub_dict_ = {}

                                if 'meta' in finfo_dict:

                                    key = finfo_dict['meta'].name

                                    sub_dict_['meta'] = finfo_dict['meta']

                                    for finfo_key_, finfo_dict_ in file_info.items():

                                        if 'meta' not in finfo_dict_:
                                            for bdkey_, bdinfo_ in finfo_dict_.items():
                                                if '_'.join(bdinfo_.name.split('_')[:-1]) in key:
                                                    sub_dict_[bdkey_] = bdinfo_

                                    new_dict_[finfo_key] = sub_dict_

                            file_info = new_dict_

                        else:

                            del_keys = [k for k, v in self.search_dict.items() if 'gap_mask' in k]

                            for dk in del_keys:
                                del self.search_dict[dk]

                            load_bands = sorted(['B{:d}'.format(band_associations[bd]) for bd in bands])

                            search_wildcards = ['ANG.txt', 'MTL.txt', 'BQA.TIF'] + [bd + '.TIF' for bd in load_bands]

                            file_info = self.download_gcp(sensor,
                                                          outdir=outdir_tmp,
                                                          outdir_brdf=outdir_brdf,
                                                          search_wildcards=search_wildcards,
                                                          n_jobs=n_jobs,
                                                          verbose=1)

                        logger.info('  Finished downloading files for yyyymm query, {}.'.format(yearmonth_query))

                        # Create pixel angle files
                        # TODO: this can be run in parallel
                        for finfo_key, finfo_dict in file_info.items():

                            # Incomplete dictionary because file was checked, existed, and cleaned
                            if 'meta' not in finfo_dict:
                                logger.warning('  The metadata does not exist.')
                                _clean_and_update(None, finfo_dict, None, check_angles=False)
                                continue

                            brdfp = '_'.join(Path(finfo_dict['meta'].name).name.split('_')[:-1])
                            out_brdf = outdir_brdf.joinpath(brdfp + '.tif')
                            out_angles = outdir_brdf.joinpath(brdfp + '_angles.tif')

                            if sensor in ['s2', 's2a', 's2b', 's2c']:
                                outdir_angles = outdir_tmp.joinpath('angles_{}'.format(Path(finfo_dict['meta'].name).name.replace('_MTD_TL.xml', '')))
                            else:
                                outdir_angles = outdir_tmp.joinpath('angles_{}'.format(Path(finfo_dict['meta'].name).name.replace('_MTL.txt', '')))

                            if not Path(finfo_dict['meta'].name).is_file():
                                logger.warning('  The metadata does not exist.')
                                _clean_and_update(outdir_angles, finfo_dict, finfo_dict['meta'].name, check_angles=False)
                                continue

                            if out_brdf.is_file():

                                logger.warning('  The output BRDF file, {}, already exists.'.format(brdfp))
                                _clean_and_update(outdir_angles, finfo_dict, finfo_dict['meta'].name, check_angles=False)
                                continue

                            if load_bands[0] not in finfo_dict:
                                logger.warning('  The download for {} was incomplete.'.format(brdfp))
                                _clean_and_update(outdir_angles, finfo_dict, finfo_dict['meta'].name, check_angles=False)
                                continue

                            outdir_angles.mkdir(parents=True, exist_ok=True)

                            ref_file = finfo_dict[load_bands[0]].name

                            logger.info('  Processing angles for {} ...'.format(brdfp))

                            if sensor.lower() in ['s2', 's2a', 's2b', 's2c']:

                                meta = rt.get_sentinel_coefficients(finfo_dict['meta'].name)

                                angle_info = sentinel_pixel_angles(finfo_dict['meta'].name,
                                                                   ref_file,
                                                                   str(outdir_angles),
                                                                   nodata=-32768,
                                                                   overwrite=False,
                                                                   verbose=1)

                                if ' '.join(bands) == 'coastal blue green red nir1 nir2 nir3 nir rededge water cirrus swir1 swir2':
                                    rad_sensor = 's2af' if angle_info.sensor == 's2a' else 's2bf'
                                elif ' '.join(bands) == 'coastal blue red nir1 nir rededge water cirrus swir1 swir2':
                                    rad_sensor = 's2acloudless' if angle_info.sensor == 's2a' else 's2bcloudless'
                                elif ' '.join(bands) == 'blue green red nir1 nir2 nir3 nir rededge swir1 swir2':
                                    rad_sensor = angle_info.sensor
                                elif ' '.join(bands) == 'blue green red nir swir1 swir2':
                                    rad_sensor = 's2al7' if angle_info.sensor == 's2a' else 's2bl7'
                                elif ' '.join(bands) == 'nir1 nir2 nir3 rededge swir1 swir2':
                                    rad_sensor = 's2a20' if angle_info.sensor == 's2a' else 's2b20'
                                elif ' '.join(bands) == 'blue green red nir':
                                    rad_sensor = 's2a10' if angle_info.sensor == 's2a' else 's2b10'
                                else:
                                    rad_sensor = angle_info.sensor

                                bandpass_sensor = angle_info.sensor

                            else:

                                meta = rt.get_landsat_coefficients(finfo_dict['meta'].name)

                                angle_info = landsat_pixel_angles(finfo_dict['angle'].name,
                                                                  ref_file,
                                                                  str(outdir_angles),
                                                                  meta.sensor,
                                                                  l57_angles_path=l57_angles_path,
                                                                  l8_angles_path=l8_angles_path,
                                                                  subsample=subsample,
                                                                  verbose=1)

                                if (len(bands) == 1) and (bands[0] == 'pan'):
                                    rad_sensor = sensor + bands[0]
                                else:

                                    if (len(bands) == 6) and (meta.sensor == 'l8'):
                                        rad_sensor = 'l8l7'
                                    elif (len(bands) == 7) and (meta.sensor == 'l8') and ('pan' in bands):
                                        rad_sensor = 'l8l7mspan'
                                    elif (len(bands) == 7) and (meta.sensor == 'l7') and ('pan' in bands):
                                        rad_sensor = 'l7mspan'
                                    else:
                                        rad_sensor = meta.sensor

                                bandpass_sensor = sensor

                            if sensor in ['s2', 's2a', 's2b', 's2c']:

                                logger.info(f'  Translating jp2 files to gtiff for {brdfp} ...')

                                load_bands_names = []

                                # Convert to GeoTiffs to avoid CRS issue with jp2 format
                                for bd in load_bands:

                                    # Check if the file exists to avoid duplicate GCP filenames`
                                    if Path(finfo_dict[bd].name).is_file():

                                        warp(finfo_dict[bd].name,
                                             finfo_dict[bd].name.replace('.jp2', '.tif'),
                                             overwrite=True,
                                             delete_input=True,
                                             multithread=True,
                                             warpMemoryLimit=256,
                                             outputBounds=out_bounds,
                                             xRes=ref_res[0],
                                             yRes=ref_res[1],
                                             resampleAlg=RESAMPLING_DICT[resampling],
                                             creationOptions=['TILED=YES',
                                                              'COMPRESS=LZW',
                                                              'BLOCKXSIZE={CHUNKS:d}'.format(CHUNKS=chunks),
                                                              'BLOCKYSIZE={CHUNKS:d}'.format(CHUNKS=chunks)])

                                        load_bands_names.append(finfo_dict[bd].name.replace('.jp2', '.tif'))

                            else:

                                # Get band names from user
                                try:
                                    load_bands_names = [finfo_dict[bd].name for bd in load_bands]
                                except:
                                    logger.exception('  Could not get all band name associations.')
                                    raise NameError

                            logger.info(f'  Applying BRDF and SR correction for {brdfp} ...')

                            with gw.config.update(sensor=rad_sensor,
                                                  ref_bounds=out_bounds,
                                                  ref_crs=crs,
                                                  ref_res=ref_res if ref_res else load_bands_names[-1],
                                                  ignore_warnings=True,
                                                  nasa_earthdata_user=earthdata_username,
                                                  nasa_earthdata_key=earthdata_key_file,
                                                  nasa_earthdata_code=earthdata_code_file):

                                valid_data = True

                                # Ensure there is data
                                with gw.open(load_bands_names[0],
                                             band_names=[1],
                                             chunks=chunks,
                                             num_threads=num_threads) as data:

                                    if data.sel(band=1).min().data.compute(num_workers=num_workers) > 10000:
                                        valid_data = False

                                    if valid_data:

                                        if data.sel(band=1).max().data.compute(num_workers=num_workers) == 0:
                                            valid_data = False

                                if valid_data:

                                    with gw.open(angle_info.sza,
                                                 chunks=chunks,
                                                 resampling='bilinear') as sza, \
                                            gw.open(angle_info.vza,
                                                    chunks=chunks,
                                                    resampling='bilinear') as vza, \
                                            gw.open(angle_info.saa,
                                                    chunks=chunks,
                                                    resampling='bilinear') as saa, \
                                            gw.open(angle_info.vaa,
                                                    chunks=chunks,
                                                    resampling='bilinear') as vaa, \
                                            gw.open(load_bands_names,
                                                    band_names=bands,
                                                    stack_dim='band',
                                                    chunks=chunks,
                                                    resampling=resampling,
                                                    num_threads=num_threads) as data:

                                        attrs = data.attrs.copy()

                                        if mask_qa:

                                            if sensor.lower() in ['s2', 's2a', 's2b', 's2c']:

                                                if S2CLOUDLESS_INSTALLED:

                                                    cloud_detector = S2PixelCloudDetector(threshold=0.4,
                                                                                          average_over=1,
                                                                                          dilation_size=5,
                                                                                          all_bands=False)

                                                    # Get the S2Cloudless bands
                                                    data_cloudless = data.sel(band=['coastal', 'blue', 'red', 'nir1', 'nir', 'rededge', 'water', 'cirrus', 'swir1', 'swir2'])

                                                    # Scale from 0-10000 to 0-1 and reshape
                                                    X = (data_cloudless * 0.0001).clip(0, 1).data\
                                                            .compute(num_workers=num_workers)\
                                                            .transpose(1, 2, 0)[np.newaxis, :, :, :]

                                                    # Predict clouds
                                                    # Potential classes? Currently, only clear and clouds are returned.
                                                    # clear=0, clouds=1, shadow=2, snow=3, cirrus=4, water=5
                                                    mask = ndarray_to_xarray(data,
                                                                             cloud_detector.get_cloud_masks(X),
                                                                             ['mask'])

                                                else:

                                                    if bands_out:

                                                        # If there are extra bands, remove them because they
                                                        # are not supported in the BRDF kernels.
                                                        data = _assign_attrs(data, attrs, bands_out)

                                                    logger.warning('  S2Cloudless is not installed, so skipping Sentinel cloud masking.')

                                        if sr_method == 'srem':

                                            if sensor.lower() in ['s2', 's2a', 's2b', 's2c']:

                                                # The S-2 data are in TOAR (0-10000)
                                                toar_scaled = (data * 0.0001)\
                                                                .astype('float64')\
                                                                .clip(0, 1)\
                                                                .assign_attrs(**attrs)

                                                # Convert TOAR to surface reflectance
                                                sr = rt.toar_to_sr(toar_scaled,
                                                                   sza, saa, vza, vaa,
                                                                   rad_sensor,
                                                                   method='srem',
                                                                   dst_nodata=nodataval)

                                            else:

                                                # Convert DN to surface reflectance
                                                sr = rt.dn_to_sr(data,
                                                                 sza, saa, vza, vaa,
                                                                 method='srem',
                                                                 sensor=rad_sensor,
                                                                 meta=meta,
                                                                 src_nodata=nodataval,
                                                                 dst_nodata=nodataval)

                                        else:

                                            if sensor.lower() in ['s2', 's2a', 's2b', 's2c']:

                                                # The S-2 data are in TOAR (0-10000)
                                                data = (data * 0.0001)\
                                                        .astype('float64')\
                                                        .assign_attrs(**attrs)

                                                data_values = 'toar'

                                            else:
                                                data_values = 'dn'

                                            if isinstance(earthdata_username, str) and \
                                                    isinstance(earthdata_key_file, str) and \
                                                    isinstance(earthdata_code_file, str):

                                                altitude = dos.get_mean_altitude(data,
                                                                                 srtm_outdir,
                                                                                 n_jobs=n_jobs)

                                                altitude *= 0.0001

                                            else:
                                                altitude = 0.0

                                            # Resample to 100m x 100m
                                            data_coarse = data.sel(band=['blue', 'swir2']).gw\
                                                                .transform_crs(dst_res=500.0,
                                                                               resampling='med')

                                            aot = dos.get_aot(data_coarse,
                                                              meta.sza,
                                                              meta,
                                                              data_values=data_values,
                                                              dn_interp=data,
                                                              angle_factor=1.0,
                                                              interp_method='fast',
                                                              aot_fallback=0.3,
                                                              h2o=2.0,
                                                              o3=0.3,  # global average of total ozone in a vertical column (3 cm)
                                                              altitude=altitude,
                                                              w=151,
                                                              n_jobs=n_jobs)

                                            if sensor.lower() in ['s2', 's2a', 's2b', 's2c']:

                                                sr = rt.toar_to_sr(data,
                                                                   meta.sza,
                                                                   None,
                                                                   None,
                                                                   None,
                                                                   meta=meta,
                                                                   src_nodata=nodataval,
                                                                   dst_nodata=nodataval,
                                                                   angle_factor=1.0,
                                                                   method='6s',
                                                                   interp_method='fast',
                                                                   h2o=2.0,
                                                                   o3=0.3,
                                                                   aot=aot,
                                                                   altitude=altitude,
                                                                   n_jobs=n_jobs)

                                            else:

                                                sr = rt.dn_to_sr(data,
                                                                 meta.sza,
                                                                 None,
                                                                 None,
                                                                 None,
                                                                 meta=meta,
                                                                 src_nodata=nodataval,
                                                                 dst_nodata=nodataval,
                                                                 angle_factor=1.0,
                                                                 method='6s',
                                                                 interp_method='fast',
                                                                 h2o=2.0,
                                                                 o3=0.3,
                                                                 aot=aot,
                                                                 altitude=altitude,
                                                                 n_jobs=n_jobs)

                                        # BRDF normalization
                                        sr_brdf = br.norm_brdf(sr,
                                                               sza, saa, vza, vaa,
                                                               sensor=rad_sensor,
                                                               wavelengths=data.band.values.tolist(),
                                                               out_range=10000.0,
                                                               src_nodata=nodataval,
                                                               dst_nodata=nodataval)

                                        if bandpass_sensor.lower() in ['l5', 'l7', 's2', 's2a', 's2b', 's2c']:

                                            # Linearly adjust to Landsat 8
                                            sr_brdf = la.bandpass(sr_brdf,
                                                                  bandpass_sensor.lower(),
                                                                  to='l8',
                                                                  scale_factor=0.0001,
                                                                  src_nodata=nodataval,
                                                                  dst_nodata=nodataval)

                                        if mask_qa:

                                            if sensor.lower() in ['s2', 's2a', 's2b', 's2c']:

                                                if S2CLOUDLESS_INSTALLED:

                                                    wavel_sub = sr_brdf.gw.set_nodata(nodataval,
                                                                                      nodataval,
                                                                                      out_range=(0, 1),
                                                                                      dtype='float64')

                                                    # Estimate the cloud shadows
                                                    mask = estimate_cloud_shadows(wavel_sub,
                                                                                  mask,
                                                                                  sza,
                                                                                  saa,
                                                                                  vza,
                                                                                  vaa,
                                                                                  heights=cloud_heights,
                                                                                  num_workers=num_workers)

                                                    # Update the bands with the mask
                                                    sr_brdf = xr.where((mask.sel(band='mask') == 0) &
                                                                       (sr_brdf != nodataval),
                                                                       sr_brdf.clip(0, 10000),
                                                                       nodataval).astype('uint16')

                                                sr_brdf = _assign_attrs(sr_brdf, attrs, bands_out)

                                                if write_format == 'gtiff':
                                                    sr_brdf.gw.to_raster(str(out_brdf), **kwargs)
                                                else:
                                                    sr_brdf.gw.to_netcdf(str(out_brdf), zlib=True, complevel=5)

                                            else:

                                                with gw.open(finfo_dict['qa'].name,
                                                             band_names=['qa']) as qa:

                                                    if sensor.lower() == 'l8':
                                                        qa_sensor = 'l8-c1'
                                                    else:
                                                        qa_sensor = 'l-c1'

                                                    mask = QAMasker(qa,
                                                                    qa_sensor,
                                                                    mask_items=lqa_mask_items,
                                                                    confidence_level='maybe').to_mask()

                                                    # Mask non-clear pixels
                                                    sr_brdf = xr.where(mask.sel(band='mask') < 2,
                                                                       sr_brdf.clip(0, 10000),
                                                                       nodataval).astype('uint16')

                                                    sr_brdf = _assign_attrs(sr_brdf, attrs, bands_out)

                                                    if write_format == 'gtiff':
                                                        sr_brdf.gw.to_raster(str(out_brdf), **kwargs)
                                                    else:
                                                        sr_brdf.gw.to_netcdf(str(out_brdf), zlib=True, complevel=5)

                                        else:

                                            # Set 'no data' values
                                            sr_brdf = sr_brdf.gw.set_nodata(nodataval,
                                                                            nodataval,
                                                                            out_range=(0, 10000),
                                                                            dtype='uint16')

                                            sr_brdf = _assign_attrs(sr_brdf, attrs, bands_out)

                                            if write_format == 'gtiff':
                                                sr_brdf.gw.to_raster(str(out_brdf), **kwargs)
                                            else:

                                                sr_brdf.gw.to_netcdf(str(out_brdf).replace('.tif', '.nc'),
                                                                     zlib=True,
                                                                     complevel=5)

                                        if write_angle_files:

                                            angle_stack = xr.concat((sza, saa), dim='band')\
                                                                .astype('int16')\
                                                                .assign_coords(band=['sza', 'saa'])\
                                                                .assign_attrs(**sza.attrs.copy())

                                            if write_format == 'gtiff':
                                                angle_stack.gw.to_raster(str(out_angles), **kwargs)
                                            else:

                                                angle_stack.gw.to_netcdf(str(out_angles).replace('.tif', '.nc'),
                                                                         zlib=True,
                                                                         complevel=5)

                                else:

                                    logger.warning('  Not enough data for {} to store on disk.'.format(str(out_brdf)))

                                    # Write an empty file for tracking
                                    with open(str(out_brdf).replace('.tif', '.nodata'), 'w') as tx:
                                        tx.writelines([])

                            angle_infos[finfo_key] = angle_info

                            _clean_and_update(outdir_angles,
                                              finfo_dict,
                                              finfo_dict['meta'].name,
                                              load_bands_names=load_bands_names)

            year += 1

    def list_gcp(self, sensor, query):

        """
        Lists files from Google Cloud Platform

        Args:
            sensor (str): The sensor to query. Choices are ['l5', 'l7', 'l8', 's2a', 's2c'].
            query (str): The query string.

        Examples:
            >>> dl = GeoDownloads()
            >>>
            >>> # Query from a known directory
            >>> dl.list_gcp('landsat', 'LC08/01/042/034/LC08_L1TP_042034_20161104_20170219_01_T1/')
            >>>
            >>> # Query a date for Landsat 5
            >>> dl.list_gcp('l5', '042/034/*2016*')
            >>>
            >>> # Query a date for Landsat 7
            >>> dl.list_gcp('l7', '042/034/*2016*')
            >>>
            >>> # Query a date for Landsat 8
            >>> dl.list_gcp('l8', '042/034/*2016*')
            >>>
            >>> # Query Sentinel-2
            >>> dl.list_gcp('s2a', '21/H/UD/*2019*.SAFE/GRANULE/*')

        Returns:
            ``dict``
        """

        if sensor not in ['l5', 'l7', 'l8', 's2', 's2a', 's2b', 's2c']:
            logger.exception("  The sensor must be 'l5', 'l7', 'l8', 's2', 's2a', 's2b', or 's2c'.")
            raise NameError

        if sensor in ['s2', 's2a', 's2b', 's2c']:
            gcp_str = "gsutil ls -r gs://gcp-public-data-sentinel-2"
        else:
            gcp_str = "gsutil ls -r gs://gcp-public-data-landsat"

        gsutil_str = gcp_str + "/" + self.gcp_dict[sensor] + "/" + query

        try:

            proc = subprocess.run(gsutil_str.split(' '),
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)

        except:
            logger.exception('gsutil must be installed.')

        output = proc.stdout

        gcp_search_list = [outp for outp in output.decode('utf-8').split('\n') if '$folder$' not in outp]
        self._gcp_search_dict = {}

        if gcp_search_list:

            # Check for length-1 lists with empty strings
            if gcp_search_list[0]:

                if sensor in ['s2', 's2a', 's2b', 's2c']:
                    self._gcp_search_dict = self._prepare_gcp_dict(gcp_search_list, 'gs://gcp-public-data-sentinel-2/')
                else:
                    self._gcp_search_dict = self._prepare_gcp_dict(gcp_search_list, 'gs://gcp-public-data-landsat/')

    @property
    def get_gcp_results(self):
        return self._gcp_search_dict.copy()

    @staticmethod
    def _prepare_gcp_dict(search_list, gcp_str):

        """
        Prepares a list of GCP keys into a dictionary

        Args:
            search_list (list)

        Returns:
            ``dict``
        """

        df = pd.DataFrame(data=search_list, columns=['url'])

        df['mask'] = df.url.str.strip().str.endswith('/:')

        mask_idx = np.where(df['mask'].values)[0]
        mask_range = mask_idx.shape[0] - 1 if mask_idx.shape[0] > 1 else 1

        url_dict = {}

        for mi in range(0, mask_range):

            m1 = mask_idx[mi]

            if mask_range > 1:
                m2 = mask_idx[mi + 1] - 1
            else:
                m2 = len(search_list)

            key = search_list[m1].replace(gcp_str, '').replace('/:', '')
            values = search_list[m1:m2]

            values = [value for value in values if value]

            url_dict[key] = [value for value in values if not value.endswith('/:')]

        return url_dict
