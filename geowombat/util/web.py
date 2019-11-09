import os
import fnmatch
import subprocess
from pathlib import Path
from datetime import datetime
from collections import namedtuple

from .. import open as gw_open
from .. import config as gw_config
from ..errors import logger
from ..radiometry import BRDF, RadTransforms, landsat_pixel_angles

import numpy as np
import pandas as pd
# import wget


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

        self.landsat_parts = ['le07', 'lt05', 'lc08']
        self.sentinel_parts = ['s2a']

        self.search_dict = dict()

    def download_cube(self, sensors, date_range, geometry, crs, bands, outdir='.'):

        """
        Downloads a cube

        Args:
            sensors (str or list): The sensors, or sensor, to download.
            date_range (list): The date range, given as [date1, date2], where the date format is yyyy-mm-dd.
            geometry (GeoDataFrame or tuple): The geometry that defines the cube extent.
            crs (str or object): The output CRS.
            bands (str or list): The bands to download.
            outdir (Optional[str]): The output directory.
        """

        rt = RadTransforms()
        br = BRDF()

        main_path = Path(outdir)
        outdir_angles = main_path.joinpath('angles')
        outdir_brdf = main_path.joinpath('brdf')

        if not main_path.is_dir():
            main_path.mkdir()

        if not outdir_angles.is_dir():
            outdir_angles.mkdir()

        if not outdir_brdf.is_dir():
            outdir_brdf.mkdir()

        gw_bin = os.path.realpath(os.path.dirname(__file__))
        gw_bin = Path(gw_bin + '../../bin')

        if isinstance(sensors, str):
            sensors = [sensors]

        # TODO: get path/row and MGRS from geometry
        location = '21/H/UD' # or '225/083'

        # TODO: get bounds from geometry
        bounds = (793000.0, 2049000.0, 794000.0, 2050000.0)

        dt1 = datetime.strptime(date_range[0], '%Y-%m-%d')
        dt2 = datetime.strptime(date_range[1], '%Y-%m-%d')

        year = dt1.year
        month = dt1.month

        months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        if month < dt2.month:
            month_range = months[months.index(month):months.index(dt2.month)]
        else:
            month_range = months[months.index(month):] + months[:months.index(dt2.month)]

        while True:

            if year > dt2.year:
                break

            for m in month_range:

                yearmonth_query = '{:d}{:02d}'.format(year, m)

                for sensor in sensors:

                    if sensor.lower() == 's2':
                        query = '{LOCATION}/*{YM}*.SAFE/GRANULE/*'.format(LOCATION=location, YM=yearmonth_query)
                    else:

                        query = '{LOCATION}/*{PATHROW}_{YM}*_T*'.format(LOCATION=location,
                                                                        PATHROW=location.replace('/', ''),
                                                                        YM=yearmonth_query)

                    self.list_gcp(sensor, query)

                    if sensor.lower() != 's2':

                        del_keys = [k for k, v in self.search_dict.items() if 'gap_mask' in k]

                        for dk in del_keys:
                            del self.search_dict[dk]

                    # TODO: Landsat pan band example
                    search_wildcards = ['ANG.txt', 'MTL.txt', 'B8.TIF']

                    file_info = self.download_gcp(sensor,
                                                  outdir=outdir,
                                                  search_wildcards=search_wildcards,
                                                  verbose=1)

                    angle_infos = dict()

                    kwargs = dict(readxsize=1024,
                                  readysize=1024,
                                  verbose=1,
                                  separate=False,
                                  n_workers=1,
                                  n_threads=8,
                                  n_chunks=100,
                                  overwrite=False)

                    # TODO: subset to geometry and DN -> BRDF
                    for finfo_key, finfo_dict in file_info.items():

                        meta = rt.get_landsat_coefficients(finfo_dict['meta'].name)

                        angle_info = landsat_pixel_angles(finfo_dict['angle'].name,
                                                          finfo_dict['B8'].name,
                                                          outdir_angles.as_posix(),
                                                          meta.sensor,
                                                          l57_angles_path=gw_bin.joinpath('/ESPA/landsat_angles').as_posix(),
                                                          l8_angles_path=gw_bin.joinpath('/ESPA/l8_angles').as_posix(),
                                                          verbose=1)

                        out_brdf = outdir_brdf.joinpath(Path(finfo_key).name + '.tif').as_posix()

                        with gw_config.update(sensor=sensor, ref_crs=crs):

                            with gw_open(angle_info.sza, bounds=bounds) as sza, \
                                    gw_open(angle_info.vza, bounds=bounds) as vza, \
                                    gw_open(angle_info.saa, bounds=bounds) as saa, \
                                    gw_open(angle_info.vaa, bounds=bounds) as vaa:

                                # TODO: get bands from user
                                with gw_open([finfo_dict['B2'].name,
                                              finfo_dict['B3'].name,
                                              finfo_dict['B4'].name,
                                              finfo_dict['B5'].name],
                                             bounds=bounds,
                                             stack_dim='band') as dn:

                                    # Convert DN to surface reflectance
                                    dn_sr = rt.dn_to_sr(dn,
                                                        sza, saa, vza, vaa,
                                                        sensor=meta.sensor,
                                                        meta=meta)

                                    # BRDF normalization
                                    dn_sr_brdf = br.norm_brdf(dn_sr,
                                                              sza, saa, vza, vaa,
                                                              sensor=meta.sensor,
                                                              wavelengths=dn.band.values.tolist(),
                                                              nodata=0)

                                    dn_sr_brdf.gw.to_raster(out_brdf, **kwargs)

                        angle_infos[finfo_key] = angle_info

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

        gsutil_str = '{GSUTIL}/{COLLECTION}/{QUERY}'.format(GSUTIL=gcp_str,
                                                            COLLECTION=gcp_dict[sensor],
                                                            QUERY=query)

        proc = subprocess.Popen(gsutil_str,
                                stdout=subprocess.PIPE,
                                shell=True)

        output = proc.stdout.read()

        search_list = [outp for outp in output.decode('utf-8').split('\n') if '$folder$' not in outp]

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

        url_dict = dict()

        for mi in range(0, mask_idx.shape[0] - 1):

            m1 = mask_idx[mi]
            m2 = mask_idx[mi + 1] - 1

            key = search_list[m1].replace(gcp_str, '').replace('/:', '')
            values = search_list[m1:m2]

            values = [value for value in values if value]

            url_dict[key] = [value for value in values if not value.endswith('/:')]

        return url_dict

    def download_gcp(self, sensor, downloads=None, outdir='.', search_wildcards=None, search_dict=None, verbose=0):

        """
        Downloads a file from Google Cloud platform

        Args:
            sensor (str): The sensor to query. Choices are ['l5', 'l7', 'l8', 's2'].
            downloads (Optional[str or list]): The file or list of keys to download. If not given, keys will be taken
                from ``search_dict`` or ``self.search_dict``.
            outdir (Optional[str]): The output directory.
            search_wildcards (Optional[list]): A list of search wildcards.
            search_dict (Optional[dict]): A keyword search dictionary to override ``self.search_dict``.
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

        for search_key in downloads:

            downloaded_sub = dict()

            download_list = self.search_dict[search_key]

            if search_wildcards:

                download_list_ = list()

                for swild in search_wildcards:
                    download_list_ += fnmatch.filter(download_list, '*{}'.format(swild))

                download_list = download_list_

            for fn in download_list:

                down_file = poutdir.joinpath(Path(fn).name).as_posix()

                if down_file.endswith('_ANG.txt'):
                    key = 'angle'
                elif down_file.endswith('_MTL.txt'):
                    key = 'meta'
                elif down_file.endswith('MTD_TL.xml'):
                    key = 'meta'
                else:
                    key = down_file.split('_')[-1].split('.')[0]

                if not Path(down_file).exists():

                    if verbose > 0:
                        logger.info('  Downloading {} ...'.format(Path(fn).name))

                    if fn.lower().startswith('gs://gcp-public-data'):
                        com = 'gsutil cp -r {} {}'.format(fn, outdir)
                    else:
                        com = 'gsutil cp -r {}/{} {}'.format(gcp_str, fn, outdir)

                    subprocess.call(com, shell=True)

                downloaded_sub[key] = FileInfo(name=down_file, key=key)

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
