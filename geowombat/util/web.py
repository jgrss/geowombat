import os
import shutil
import fnmatch
import tarfile
import subprocess
from pathlib import Path
from datetime import datetime
from collections import namedtuple
import random
import string

from ..errors import logger
from ..radiometry import BRDF, LinearAdjustments, RadTransforms, landsat_pixel_angles, sentinel_pixel_angles, QAMasker

import geowombat as gw

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import shapely
from shapely.geometry import Polygon
# import wget


shapely.speedups.enable()


def _random_id(string_length):

    """
    Generates a random string of letters and digits
    """

    letters_digits = string.ascii_letters + string.digits

    return ''.join(random.choice(letters_digits) for i in range(string_length))


def _parse_google_filename(filename, landsat_parts, sentinel_parts, public_url):

    FileInfo = namedtuple('FileInfo', 'url url_file meta angles')

    file_info = FileInfo(url=None, url_file=None, meta=None, angles=None)

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

        file_info = FileInfo(url=url_,
                             url_file=url_filename,
                             meta=url_meta,
                             angles=url_angles)

    # elif fn_parts[0].lower() in sentinel_parts:
    #
    #     safe_dir = '{SENSOR}_{M}{L}_'.format(SENSOR=fn_parts[0], M=fn_parts[2], L=fn_parts[3])
    #
    #     '/01/K/AB/S2A_MSIL1C_20160519T222025_N0202_R029_T01KAB_20160520T024950.SAFE/GRANULE/S2A_OPER_MSI_L1C_TL_SGS__20160519T234326_A004745_T01KAB_N02.02/IMG_DATA/S2A_OPER_MSI_L1C_TL_SGS__20160519T234326_A004745_T01KAB_B05.jp2'
    #
    #     fn = '{PUBLIC}-sentinel-2/tiles/{UTM}/{LAT}/{GRID}'.format(PUBLIC=public_url,
    #                                                                UTM=,
    #                                                                LAT=,
    #                                                                GRID=)

    return file_info


class GeoDownloads(object):

    def __init__(self):

        self.gcp_public = 'https://storage.googleapis.com/gcp-public-data'

        self.landsat_parts = ['lt05', 'le07', 'lc08']
        self.sentinel_parts = ['s2a']

        self.associations = dict(l7=dict(blue=1,
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
                                 s2=dict(blue=1,
                                         green=2,
                                         red=3,
                                         nir1=4,
                                         nir2=5,
                                         nir3=6,
                                         nir=7,
                                         rededge=8,
                                         swir1=9,
                                         swir2=10))

        self.search_dict = dict()

    def download_cube(self,
                      sensors,
                      date_range,
                      bounds,
                      bands,
                      crs=None,
                      outdir='.',
                      **kwargs):

        """
        Downloads a cube of Landsat and/or Sentinel 2 imagery

        Args:
            sensors (str or list): The sensors, or sensor, to download.
            date_range (list): The date range, given as [date1, date2], where the date format is yyyy-mm-dd.
            bounds (GeoDataFrame, list, or tuple): The geometry bounds (in WGS84 lat/lon) that define the cube extent.
                If given as a ``GeoDataFrame``, only the first ``DataFrame`` record will be used.
                If given as a ``tuple`` or a ``list``, the order should be (left, bottom, right, top).
            bands (str or list): The bands to download.
            crs (Optional[str or object]): The output CRS. If ``bounds`` is a ``GeoDataFrame``, the CRS is taken
                from the object.
            outdir (Optional[str]): The output directory.
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
            >>> gdl.download_cube(['l7', 'l8', 's2'],
            >>>                   ['2017-01-01', '2018-01-01'],
            >>>                   (-91.57, 40.37, -91.46, 40.42),
            >>>                   ['blue', 'green', 'red'],
            >>>                   crs={'init': 'epsg:102033'},
            >>>                   readxsize=1024,
            >>>                   readysize=1024,
            >>>                   n_workers=1,
            >>>                   n_threads=8)
        """

        # TODO: parameterize
        # kwargs = dict(readxsize=1024,
        #               readysize=1024,
        #               verbose=1,
        #               separate=False,
        #               n_workers=1,
        #               n_threads=8,
        #               n_chunks=100,
        #               overwrite=False)

        angle_infos = dict()

        rt = RadTransforms()
        br = BRDF()
        la = LinearAdjustments()

        main_path = Path(outdir)
        outdir_brdf = main_path.joinpath('brdf')

        if not main_path.is_dir():
            main_path.mkdir()

        if not outdir_brdf.is_dir():
            outdir_brdf.mkdir()

        if isinstance(sensors, str):
            sensors = [sensors]

        status = Path(outdir).joinpath('status.txt')

        if not status.is_file():

            with open(status.as_posix(), mode='w') as tx:
                pass

        # Get bounds from geometry
        if isinstance(bounds, tuple) or isinstance(bounds, list):

            bounds = Polygon([(bounds[0], bounds[3]),
                              (bounds[2], bounds[3]),
                              (bounds[2], bounds[1]),
                              (bounds[0], bounds[1])])

            bounds = gpd.GeoDataFrame([0],
                                      geometry=[bounds],
                                      crs={'init': 'epsg:4326'})

        bounds_object = bounds.geometry.values[0]
        bounds_proj = bounds.to_crs(crs)
        bounds_info = bounds_proj.bounds.values[0].tolist()

        # BoundsInfo = namedtuple('BoundsInfo', 'left bottom right top')
        # bounds_info = BoundsInfo(left=left, bottom=bottom, right=right, top=top)

        # TODO: get MGRS file

        # Get WRS file
        data_bin = os.path.realpath(os.path.dirname(__file__))

        wrs_dir = Path(data_bin).joinpath('../data')
        wrs_tar = Path(wrs_dir).joinpath('wrs2.tar.gz')
        wrs_path = Path(wrs_dir).joinpath('wrs2_descending.shp')

        wrs = os.path.realpath(wrs_path.as_posix())

        if not wrs_path.is_file():

            with tarfile.open(os.path.realpath(wrs_tar.as_posix()), mode='r:gz') as tf:
                tf.extractall(wrs_dir)

        df_wrs = gpd.read_file(wrs)
        df_wrs = df_wrs[df_wrs.geometry.intersects(bounds_object)]

        if df_wrs.empty:
            logger.warning('  The geometry bounds is empty.')
            return

        dt1 = datetime.strptime(date_range[0], '%Y-%m-%d')
        dt2 = datetime.strptime(date_range[1], '%Y-%m-%d')

        year = dt1.year
        month = dt1.month

        months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        if month < dt2.month:
            month_range = months[months.index(month):months.index(dt2.month)+1]
        else:
            month_range = months[months.index(month):] + months[:months.index(dt2.month)+1]

        while True:

            if year > dt2.year:
                break

            for m in month_range:

                yearmonth_query = '{:d}{:02d}'.format(year, m)

                for sensor in sensors:

                    band_associations = self.associations[sensor]

                    # TODO: get path/row and MGRS from geometry
                    # location = '21/H/UD' # or '225/083'

                    if sensor.lower() == 's2':
                        # TODO: get MGRS grid name from file
                        locations = list()
                    else:

                        locations = ['{:03d}/{:03d}'.format(int(dfrow.PATH), int(dfrow.ROW))
                                     for dfi, dfrow in df_wrs.iterrows()]

                    for location in locations:

                        if sensor.lower() == 's2':

                            query = '{LOCATION}/*{YM}*.SAFE/GRANULE/*'.format(LOCATION=location,
                                                                              YM=yearmonth_query)

                        else:

                            query = '{LOCATION}/*{PATHROW}_{YM}*_T*'.format(LOCATION=location,
                                                                            PATHROW=location.replace('/', ''),
                                                                            YM=yearmonth_query)

                        # Query and list available files on the GCP
                        self.list_gcp(sensor, query)

                        if not self.search_dict:

                            logger.warning('  No results found for {SENSOR} at location {LOC}, year {YEAR:d}, month {MONTH:d}.'.format(SENSOR=sensor,
                                                                                                                                       LOC=location,
                                                                                                                                       YEAR=year,
                                                                                                                                       MONTH=m))

                            continue

                        # Download data
                        if sensor.lower() == 's2':

                            load_bands = sorted(['B{:02d}'.format(band_associations[bd]) for bd in bands])

                            for k, v in self.search_dict.items():

                                if k.endswith('AUX_DATA') or k.endswith('QI_DATA'):
                                    pass
                                elif k.endswith('IMG_DATA'):

                                    search_wildcards = [bd + '.jp2' for bd in load_bands]

                                    file_info = self.download_gcp(sensor,
                                                                  outdir=outdir,
                                                                  search_wildcards=search_wildcards,
                                                                  verbose=1)

                                else:

                                    search_wildcards = ['MTD_TL.xml']

                                    file_info = self.download_gcp(sensor,
                                                                  outdir=outdir,
                                                                  search_wildcards=search_wildcards,
                                                                  verbose=1)

                                    meta_file = file_info['meta'].name

                        else:

                            del_keys = [k for k, v in self.search_dict.items() if 'gap_mask' in k]

                            for dk in del_keys:
                                del self.search_dict[dk]

                            load_bands = sorted(['B{:d}'.format(band_associations[bd]) for bd in bands])

                            search_wildcards = ['ANG.txt', 'MTL.txt', 'BQA.TIF'] + [bd + '.TIF' for bd in load_bands]

                            file_info = self.download_gcp(sensor,
                                                          outdir=outdir,
                                                          search_wildcards=search_wildcards,
                                                          check_file=status.as_posix(),
                                                          verbose=1)

                        # Create pixel angle files
                        # TODO: this can be run in parallel
                        for finfo_key, finfo_dict in file_info.items():

                            out_brdf = outdir_brdf.joinpath(Path(finfo_key).name + '.tif').as_posix()

                            if os.path.isfile(out_brdf):
                                logger.warning('  The output BRDF file already exists.')
                                continue

                            with open(status.as_posix(), mode='r') as tx:
                                lines = tx.readlines()

                            if finfo_dict['meta'].name + '\n' in lines:
                                logger.warning('  The file has already been checked.')
                                continue

                            outdir_angles = main_path.joinpath('angles_{}'.format(Path(finfo_dict['meta'].name).name.replace('_MTL.txt', '')))

                            if not outdir_angles.is_dir():
                                outdir_angles.mkdir()

                            ref_file = file_info[finfo_key][load_bands[0]].name

                            if sensor.lower() == 's2':

                                angle_info = sentinel_pixel_angles(meta_file,
                                                                   ref_file,
                                                                   outdir_angles.as_posix(),
                                                                   nodata=-32768,
                                                                   overwrite=False,
                                                                   verbose=1)

                                rad_sensor = 's2'
                                
                            else:

                                meta = rt.get_landsat_coefficients(finfo_dict['meta'].name)

                                angle_info = landsat_pixel_angles(finfo_dict['angle'].name,
                                                                  ref_file,
                                                                  outdir_angles.as_posix(),
                                                                  meta.sensor,
                                                                  verbose=1)

                                if (len(bands) == 1) and (bands[0] == 'pan'):
                                    rad_sensor = sensor + bands[0]
                                else:
                                    rad_sensor = meta.sensor

                            # Get band names from user
                            load_bands_names = [finfo_dict[bd].name for bd in load_bands]

                            with gw.config.update(sensor=rad_sensor,
                                                  ref_bounds=bounds_info,
                                                  ref_crs=crs,
                                                  ref_res=load_bands_names[0]):

                                with gw.open(angle_info.sza,
                                             resampling='cubic') as sza, \
                                        gw.open(angle_info.vza,
                                                resampling='cubic') as vza, \
                                        gw.open(angle_info.saa,
                                                resampling='cubic') as saa, \
                                        gw.open(angle_info.vaa,
                                                resampling='cubic') as vaa:

                                    with gw.open(load_bands_names,
                                                 band_names=bands,
                                                 stack_dim='band') as data, \
                                            gw.open(finfo_dict['qa'].name,
                                                    band_names=['qa']) as qa:

                                        # Setup the mask
                                        if sensor.lower() != 's2':

                                            if sensor.lower() == 'l8':
                                                qa_sensor = 'l8-c1'
                                            else:
                                                qa_sensor = 'l-c1'

                                        mask = QAMasker(qa,
                                                        qa_sensor,
                                                        mask_items=['cloud', 'shadow']).to_mask()

                                        if sensor.lower() == 's2':

                                            # The S-2 data are in TOAR (0-10000)
                                            toar_scaled = (data * 0.0001).clip(0, 1).astype('float64')
                                            toar_scaled.attrs = data.attrs.copy()

                                            # Convert TOAR to surface reflectance
                                            sr = rt.toar_to_sr(toar_scaled,
                                                               sza, saa, vza, vaa,
                                                               sensor)

                                        else:

                                            # Convert DN to surface reflectance
                                            sr = rt.dn_to_sr(data,
                                                             sza, saa, vza, vaa,
                                                             sensor=rad_sensor,
                                                             meta=meta)

                                        # BRDF normalization
                                        sr_brdf = br.norm_brdf(sr,
                                                               sza, saa, vza, vaa,
                                                               sensor=rad_sensor,
                                                               wavelengths=data.band.values.tolist(),
                                                               out_range=10000.0,
                                                               nodata=65535)

                                        # TODO: get Sentinel 2 a or b
                                        if sensor.lower() in ['l5', 'l7', 's2']:

                                            # Linear adjust to Landsat 8
                                            sr_brdf = la.bandpass(sr_brdf, sensor, to='l8')

                                        # mask non-clear pixels
                                        attrs = sr_brdf.attrs
                                        sr_brdf = xr.where(mask.sel(band='mask') < 2, sr_brdf, 65535)
                                        sr_brdf = sr_brdf.transpose('band', 'y', 'x')
                                        sr_brdf.attrs = attrs

                                        sr_brdf.gw.to_raster(out_brdf, **kwargs)

                            angle_infos[finfo_key] = angle_info

                            shutil.rmtree(outdir_angles)

                            for k, v in finfo_dict.items():
                                os.remove(v.name)

                            lines.append(finfo_dict['meta'].name + '\n')

                            with open(status.as_posix(), mode='r+') as tx:
                                tx.writelines(lines)

            year += 1

    def list_gcp(self, sensor, query):

        """
        Lists files from Google Cloud Platform

        Args:
            sensor (str): The sensor to query. Choices are ['l5', 'l7', 'l8', 's2'].
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
            >>> dl.list_gcp('s2', '21/H/UD/*2019*.SAFE/GRANULE/*')

        Returns:
            ``dict``
        """

        gcp_dict = dict(l5='LT05/01',
                        l7='LE07/01',
                        l8='LC08/01',
                        s2='tiles')

        if sensor not in ['l5', 'l7', 'l8', 's2']:
            logger.exception("  The sensor must be 'l5', 'l7', 'l8', or 's2'.")

        if sensor == 's2':
            gcp_str = 'gsutil ls -r gs://gcp-public-data-sentinel-2'
        else:
            gcp_str = 'gsutil ls -r gs://gcp-public-data-landsat'

        # 'gsutil ls -r gs://gcp-public-data-landsat/LC08/01/024/032/*024032_201803*_T*'

        gsutil_str = '{GSUTIL}/{COLLECTION}/{QUERY}'.format(GSUTIL=gcp_str,
                                                            COLLECTION=gcp_dict[sensor],
                                                            QUERY=query)

        proc = subprocess.Popen(gsutil_str,
                                stdout=subprocess.PIPE,
                                shell=True)

        output = proc.stdout.read()

        search_list = [outp for outp in output.decode('utf-8').split('\n') if '$folder$' not in outp]

        if search_list:

            # Check for lenth-1 lists with empty strings
            if search_list[0]:

                if sensor == 's2':
                    self.search_dict = self._prepare_gcp_dict(search_list, 'gs://gcp-public-data-sentinel-2/')
                else:
                    self.search_dict = self._prepare_gcp_dict(search_list, 'gs://gcp-public-data-landsat/')

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

        url_dict = dict()

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

    def download_gcp(self,
                     sensor,
                     downloads=None,
                     outdir='.',
                     search_wildcards=None,
                     search_dict=None,
                     check_file=None,
                     verbose=0):

        """
        Downloads a file from Google Cloud platform

        Args:
            sensor (str): The sensor to query. Choices are ['l5', 'l7', 'l8', 's2'].
            downloads (Optional[str or list]): The file or list of keys to download. If not given, keys will be taken
                from ``search_dict`` or ``self.search_dict``.
            outdir (Optional[str]): The output directory.
            search_wildcards (Optional[list]): A list of search wildcards.
            search_dict (Optional[dict]): A keyword search dictionary to override ``self.search_dict``.
            check_file (Optional[str]): A status file to check.
            verbose (Optional[int]): The verbosity level.

        Returns:
            ``namedtuple`` or ``list``
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

        if sensor == 's2':
            gcp_str = 'gsutil cp -r gs://gcp-public-data-sentinel-2'
        else:
            gcp_str = 'gsutil cp -r gs://gcp-public-data-landsat'

        FileInfo = namedtuple('FileInfo', 'name key')

        downloaded = dict()
        null_items = list()

        for search_key in downloads:

            downloaded_sub = dict()

            download_list = self.search_dict[search_key]

            if search_wildcards:

                download_list_ = list()

                for swild in search_wildcards:
                    download_list_ += fnmatch.filter(download_list, '*{}'.format(swild))

                download_list = download_list_

            for fn in download_list:

                fname = Path(fn).name

                down_file = poutdir.joinpath(fname).as_posix()

                if down_file.endswith('_ANG.txt'):
                    fbase = fname.replace('_ANG.txt', '')
                    key = 'angle'
                elif down_file.endswith('_MTL.txt'):
                    fbase = fname.replace('_MTL.txt', '')
                    key = 'meta'
                elif down_file.endswith('MTD_TL.xml'):
                    fbase = fname.replace('MTD_TL.xml', '')
                    key = 'meta'
                elif down_file.endswith('_BQA.TIF'):
                    fbase = fname.replace('_BQA.TIF', '')
                    key = 'qa'
                else:
                    fbase = ''
                    key = down_file.split('_')[-1].split('.')[0]

                continue_download = True

                if fbase in null_items:
                    continue_download = False
                elif check_file and (key == 'meta'):

                    with open(check_file, mode='r') as tx:
                        lines = tx.readlines()

                    if down_file + '\n' in lines:
                        null_items.append(fbase)
                        continue_download = False

                if not continue_download:

                    if downloaded_sub:

                        del_keys = list()

                        for k, v in downloaded_sub.items():

                            if fbase in v:

                                if verbose > 0:
                                    logger.warning('  Removing {} ...'.format(v.name))

                                if Path(v.name).is_file():

                                    os.remove(v.name)
                                    del_keys.append(k)

                        if del_keys:

                            for del_key in del_keys:

                                if del_key in downloaded_sub:
                                    del downloaded_sub[del_key]

                else:

                    if not Path(down_file).is_file():

                        if fn.lower().startswith('gs://gcp-public-data'):
                            com = 'gsutil cp -r {} {}'.format(fn, outdir)
                        else:
                            com = 'gsutil cp -r {}/{} {}'.format(gcp_str, fn, outdir)

                        if verbose > 0:
                            logger.info('  Downloading {} ...'.format(fname))

                        subprocess.call(com, shell=True)

                    downloaded_sub[key] = FileInfo(name=down_file, key=key)

            if downloaded_sub:
                downloaded[search_key] = downloaded_sub

        return downloaded

    def download_landsat_range(self, sensors, bands, path_range, row_range, date_range, **kwargs):

        """
        Downloads Landsat data from iterables

        Args:
            sensors (str): A list of sensors to download.
            bands (str): A list of bands to download.
            path_range (iterable): A list of paths.
            row_range (iterable): A list of rows.
            date_range (iterable): A list of ``datetime`` objects or a list of strings as yyyymmdd.
            kwargs (Optional[dict]): Keyword arguments to pass to ``download``.

        Examples:
            >>> from geowombat.util import download_landsat_range
            >>>
            >>> download_landsat_range(['lc08'], ['b4'], [42], [34], ['20170616', '20170620'])
        """

        if (len(date_range) == 2) and not isinstance(date_range[0], datetime):

            start_date = date_range[0]
            end_date = date_range[1]

            sdt = datetime.strptime(start_date, '%Y%m%d')
            edt = datetime.strptime(end_date, '%Y%m%d')

            date_range = pd.date_range(start=sdt, end=edt).to_pydatetime().tolist()

        for sensor in sensors:
            for band in bands:
                for path in path_range:
                    for row in row_range:
                        for dt in date_range:

                            str_date = '{:d}{:02d}{:02d}'.format(dt.year, dt.month, dt.day)

                            # TODO: check if L1TP is used for all sensors
                            # TODO: fixed DATE2
                            filename = '{SENSOR}_L1TP_{PATH:03d}{ROW:03d}_{DATE}_{DATE2}_01_T1_{BAND}.TIF'.format(SENSOR=sensor.upper(),
                                                                                                                  PATH=path,
                                                                                                                  ROW=row,
                                                                                                                  DATE=str_date,
                                                                                                                  DATE2=None,
                                                                                                                  BAND=band)

                            self.download(filename, **kwargs)

    # def download(self, filename, outdir='.', from_google=True, metadata=True, overwrite=False):
    #
    #     """
    #     Downloads an individual file
    #
    #     Args:
    #         filename (str or list): The file to download.
    #         outdir (Optional[str]): The output directory.
    #         from_google (Optional[bool]): Whether to download from Google Cloud storage
    #         metadata (Optional[bool]): Whether to download metadata files.
    #         overwrite (Optional[bool]): Whether to overwrite an existing file.
    #
    #     https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF
    #
    #     Examples:
    #         >>> from geowombat.util import download
    #         >>>
    #         >>> # Download band 4 from Google Cloud storage to the current directory
    #         >>> download('LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF')
    #
    #     Returns:
    #         None
    #     """
    #
    #     outputs = list()
    #
    #     if not isinstance(filename, list):
    #         filename = [filename]
    #
    #     FileInfo = namedtuple('FileInfo', 'band meta angles')
    #
    #     if outdir != '.':
    #         Path(outdir).mkdir(parents=True, exist_ok=True)
    #
    #     for fn in filename:
    #
    #         if from_google:
    #
    #             file_info = _parse_google_filename(fn,
    #                                                self.landsat_parts,
    #                                                self.sentinel_parts,
    #                                                self.gcp_public)
    #
    #             file_on_disc = Path(outdir).joinpath(fn)
    #             meta_on_disc = Path(outdir).joinpath(Path(file_info.meta).name)
    #             angles_on_disc = Path(outdir).joinpath(Path(file_info.angles).name)
    #
    #             if file_info.url:
    #
    #                 if overwrite:
    #
    #                     if file_on_disc.exists():
    #                         file_on_disc.unlink()
    #
    #                 if file_on_disc.exists():
    #                     logger.warning('  The file already exists.')
    #                 else:
    #
    #                     wget.download(file_info.url_file, out=outdir)
    #
    #                     if metadata:
    #
    #                         wget.download(file_info.meta, out=outdir)
    #                         wget.download(file_info.angles, out=outdir)
    #
    #             outputs.append(FileInfo(band=file_on_disc.as_posix(),
    #                                     meta=meta_on_disc.as_posix(),
    #                                     angles=angles_on_disc.as_posix()))
    #
    #     return outputs
