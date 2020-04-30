import os
from pathlib import Path
import fnmatch
import subprocess
from collections import namedtuple
import tarfile
# import datetime
# from datetime import datetime as dtime

from ..errors import logger

import numpy as np
import cv2
import xarray as xr
# import dask.array as da
import rasterio as rio
from rasterio.warp import reproject
from affine import Affine
import xml.etree.ElementTree as ET
# import numba as nb
# from pysolar.solar import get_altitude_fast, get_azimuth_fast


def scattering_angle(cos_sza, cos_vza, sin_sza, sin_vza, cos_raa):

    """
    Calculates the scattering angle

    Args:
        cos_sza (DataArray): The cosine of the solar zenith angle.
        cos_vza (DataArray): The cosine of the view zenith angle.
        sin_sza (DataArray): The sine of the solar zenith angle.
        sin_vza (DataArray): The sine of the view zenith angle.
        cos_raa (DataArray): The cosine of the relative azimuth angle.

    Equation:

        .. math::

            \Theta = scattering angle

            \theta_0 = solar zenith angle

            \theta_S = sensor zenith angle

            \zeta = relative azimuth angle

            \Theta_s = \arccos{- \cos{\theta_0} \cos{\theta_S} - \sin{\theta_0} \sin{\theta_S} \cos{\zeta}}

    References:
        scattering angle = the angle between the direction of incident and scattered radiation
        Liu, CH and Liu GR (2009) AEROSOL OPTICAL DEPTH RETRIEVAL FOR SPOT HRV IMAGES, Journal of Marine Science and Technology
        http://stcorp.github.io/harp/doc/html/algorithms/derivations/scattering_angle.html

    Returns:
        Scattering angle (in radians) as an ``xarray.DataArray``
    """

    scattering_angle = xr.ufuncs.arccos(-cos_sza * cos_vza - sin_sza * sin_vza * cos_raa)

    return xr.ufuncs.cos(scattering_angle) ** 2


def relative_azimuth(saa, vaa):

    """
    Calculates the relative azimuth angle

    Args:
        saa (DataArray): The solar azimuth angle (in degrees).
        vaa (DataArray): The view azimuth angle (in degrees).

    Reference:
        http://stcorp.github.io/harp/doc/html/algorithms/derivations/relative_azimuth_angle.html

    Returns:
        Relative azimuth (in degrees) as an ``xarray.DataArray``
    """

    # Relative azimuth (in radians)
    raa = xr.ufuncs.deg2rad(saa - vaa)

    # Create masks
    raa_plus = xr.where(raa >= 2.0*np.pi, 1, 0)
    raa_minus = xr.where(raa < 0, 1, 0)

    # raa = xr.where(raa_plus == 1, raa + (2.0*np.pi), raa)
    # raa = xr.where(raa_minus == 1, raa - (2.0*np.pi), raa)

    raa = xr.where(raa_plus == 1, raa - (2.0 * np.pi), raa)
    raa = xr.where(raa_minus == 1, raa + (2.0 * np.pi), raa)

    return xr.ufuncs.fabs(xr.ufuncs.rad2deg(raa))


def get_sentinel_sensor(metadata):

    # Parse the XML file
    tree = ET.parse(metadata)
    root = tree.getroot()

    for child in root:

        if 'general_info' in child.tag[-14:].lower():
            general_info = child

    for ginfo in general_info:

        if ginfo.tag == 'TILE_ID':
            file_name = ginfo.text

    return file_name[:3].lower()


def _parse_sentinel_angles(metadata, proc_angles, nodata):

    """
    Gets the Sentinel-2 solar angles from metadata

    Args:
        metadata (str): The metadata file.
        proc_angles (str): The angles to parse. Choices are ['solar', 'view'].
        nodata (int or float): The 'no data' value.

    Returns:
        zenith and azimuth angles as a ``tuple`` of 2d ``numpy`` arrays
    """

    if proc_angles == 'view':

        zenith_values = np.zeros((13, 23, 23), dtype='float64') + nodata
        azimuth_values = np.zeros((13, 23, 23), dtype='float64') + nodata

    else:

        zenith_values = np.zeros((23, 23), dtype='float64') + nodata
        azimuth_values = np.zeros((23, 23), dtype='float64') + nodata

    view_tag = 'Sun_Angles_Grid' if proc_angles == 'solar' else 'Viewing_Incidence_Angles_Grids'

    # Parse the XML file
    tree = ET.parse(metadata)
    root = tree.getroot()

    # Find the angles
    for child in root:

        if child.tag[-14:] == 'Geometric_Info':
            geoinfo = child

    for segment in geoinfo:

        if segment.tag == 'Tile_Angles':
            angles = segment

    for angle in angles:

        if angle.tag == view_tag:

            if proc_angles == 'view':
                band_id = int(angle.attrib['bandId'])

            for bset in angle:

                if bset.tag == 'Zenith':
                    zenith = bset
                if bset.tag == 'Azimuth':
                    azimuth = bset

            for field in zenith:

                if field.tag == 'Values_List':
                    zvallist = field

            for field in azimuth:

                if field.tag == 'Values_List':
                    avallist = field

            for rindex in range(len(zvallist)):

                zvalrow = zvallist[rindex]
                avalrow = avallist[rindex]
                zvalues = zvalrow.text.split(' ')
                avalues = avalrow.text.split(' ')
                values = list(zip(zvalues, avalues))

                for cindex in range(len(values)):

                    if (values[cindex][0].lower() != 'nan') and (values[cindex][1].lower() != 'nan'):

                        ze = float(values[cindex][0])
                        az = float(values[cindex][1])

                        if proc_angles == 'view':

                            zenith_values[band_id, rindex, cindex] = ze
                            azimuth_values[band_id, rindex, cindex] = az

                        else:

                            zenith_values[rindex, cindex] = ze
                            azimuth_values[rindex, cindex] = az

    return zenith_values, azimuth_values


def sentinel_pixel_angles(metadata,
                          ref_file,
                          outdir='.',
                          nodata=-32768,
                          overwrite=False,
                          verbose=0):

    """
    Generates Sentinel pixel angle files

    Args:
        metadata (str): The metadata file.
        ref_file (str): A reference image to use for geo-information.
        outdir (Optional[str])): The output directory to save the angle files to.
        nodata (Optional[int or float]): The 'no data' value.
        overwrite (Optional[bool]): Whether to overwrite existing angle files.
        verbose (Optional[int]): The verbosity level.

    References:
        https://www.sentinel-hub.com/faq/how-can-i-access-meta-data-information-sentinel-2-l2a
        https://github.com/marujore/sentinel_angle_bands/blob/master/sentinel2_angle_bands.py

    Returns:
        zenith and azimuth angles as a ``namedtuple`` of angle file names
    """

    AngleInfo = namedtuple('AngleInfo', 'vza vaa sza saa sensor')

    sza, saa = _parse_sentinel_angles(metadata, 'solar', nodata)
    vza, vaa = _parse_sentinel_angles(metadata, 'view', nodata)

    sensor_name = get_sentinel_sensor(metadata)

    with rio.open(ref_file) as src:

        profile = src.profile.copy()

        ref_height = src.height
        ref_width = src.width
        ref_extent = src.bounds

        profile.update(transform=Affine(src.res[0], 0.0, ref_extent.left, 0.0, -src.res[1], ref_extent.top),
                       height=ref_height,
                       width=ref_width,
                       nodata=-32768,
                       dtype='int16',
                       count=1,
                       driver='GTiff',
                       tiled=True,
                       compress='lzw')

    ref_base = '_'.join(os.path.basename(ref_file).split('_')[:-1])

    opath = Path(outdir)

    opath.mkdir(parents=True, exist_ok=True)

    # Set output angle file names.
    sensor_azimuth_file = opath.joinpath(ref_base + '_sensor_azimuth.tif').as_posix()
    sensor_zenith_file = opath.joinpath(ref_base + '_sensor_zenith.tif').as_posix()
    solar_azimuth_file = opath.joinpath(ref_base + '_solar_azimuth.tif').as_posix()
    solar_zenith_file = opath.joinpath(ref_base + '_solar_zenith.tif').as_posix()

    for angle_array, angle_file in zip([vaa,
                                        vza,
                                        saa,
                                        sza],
                                       [sensor_azimuth_file,
                                        sensor_zenith_file,
                                        solar_azimuth_file,
                                        solar_zenith_file]):

        pfile = Path(angle_file)

        if overwrite:

            if pfile.is_file():
                pfile.unlink()

        if not pfile.is_file():

            # TODO: write data for each band?
            if len(angle_array.shape) > 2:
                angle_array = angle_array.mean(axis=0)

            with rio.open(angle_file, mode='w', **profile) as dst:

                if verbose > 0:
                    logger.info('  Writing {} to file ...'.format(angle_file))

                # Resample and scale
                angle_array_resamp = np.int16(cv2.resize(angle_array,
                                                         (0, 0),
                                                         fy=ref_height / angle_array.shape[0],
                                                         fx=ref_width / angle_array.shape[1],
                                                         interpolation=cv2.INTER_LINEAR) / 0.01)

                dst.write(angle_array_resamp, indexes=1)

    return AngleInfo(vaa=sensor_azimuth_file,
                     vza=sensor_zenith_file,
                     saa=solar_azimuth_file,
                     sza=solar_zenith_file,
                     sensor=sensor_name)


# Potentially useful for angle creation
# https://github.com/gee-community/gee_tools/blob/master/geetools/algorithms.py

# def slope_between(a, b):
#     return (a[1] - b[1]) / (a[0] - b[0])
#
#
# @nb.jit
# def _calc_sensor_angles(data,
#                         zenith_angles,
#                         azimuth_angles,
#                         yvalues,
#                         xvalues,
#                         celly,
#                         cellx,
#                         satellite_height,
#                         nodata,
#                         acquisition_date):
#
#     """
#     Calculates sensor zenith and azimuth angles
#     """
#
#     slope = slope_between(np.array([data.gw.meta.right + ((data.gw.ncols/2.0)*data.gw.cellx), data.gw.meta.top]),
#                           np.array([data.gw.meta.left, data.gw.meta.top - ((data.gw.nrows / 2.0) * data.gw.celly)]))
#
#     slope_perc = -1.0 / slope
#
#     view_az = (math.pi / 2.0) - math.arctan(slope_perc)
#
#     for i in range(0, yvalues.shape[0]):
#
#         for j in range(0, xvalues.shape[0]):
#
#             if data_band[i, j] != nodata:
#
#                 # TODO: calculate satellite drift angle
#                 dist_from_nadir = None
#
#                 # Calculate the distance from the current location to the satellite
#                 dist_to_satellite = np.hypot(satellite_height, dist_from_nadir)
#
#                 # Calculate the view angle
#
#                 zenith_angles[i, j]
#
#                 # Solar zenith angle = 90 - elevation angle scaled to integer range
#                 zenith_angles[i, j] = (90.0 - get_altitude_fast(xvalues[j], yvalues[i], acquisition_date)) / 0.01
#
#                 # Solar azimuth angle
#                 azimuth_angles[i, j] = float(get_azimuth_fast(xvalues[j], yvalues[i], acquisition_date)) / 0.01
#
#     return zenith_angles, azimuth_angles


# @nb.jit
# def _calc_solar_angles(data_band, zenith_angles, azimuth_angles, yvalues, xvalues, nodata, acquisition_date):
#
#     """
#     Calculates solar zenith and azimuth angles
#     """
#
#     for i in range(0, yvalues):
#
#         for j in range(0, xvalues):
#
#             if data_band[i, j] != nodata:
#
#                 # Solar zenith angle = 90 - elevation angle scaled to integer range
#                 zenith_angles[i, j] = (90.0 - get_altitude_fast(xvalues[j], yvalues[i], acquisition_date)) / 0.01
#
#                 # Solar azimuth angle
#                 azimuth_angles[i, j] = float(get_azimuth_fast(xvalues[j], yvalues[i], acquisition_date)) / 0.01
#
#     return zenith_angles, azimuth_angles


# def pixel_angles(data, band, nodata, meta):
#
#     """
#     Generates pixel zenith and azimuth angles
#
#     Args:
#         data (Xarray): The data with coordinate and transform attributes.
#         band (int or str): The ``data`` band to use for masking.
#         nodata (int or float): The 'no data' value in ``data``.
#         meta (namedtuple): The metadata file. Should have image acquisition year, month, day and hour attributes.
#     """
#
#     acquisition_date = dtime(meta.year, meta.month, meta.day, meta.hour, 0, 0, 0, tzinfo=datetime.timezone.utc)
#
#     yvalues = data.y.values
#     xvalues = data.x.values
#
#     data_band = data.sel(band=band).data.compute()
#     sze = np.zeros((data.gw.nrows, data.gw.ncols), dtype='int16') - 32768
#     saa = np.zeros((data.gw.nrows, data.gw.ncols), dtype='int16') - 32768
#
#     sze, saa = _calc_solar_angles(data_band, sze, saa, yvalues, xvalues, nodata, acquisition_date)
#
#     sze_attrs = data.attrs.copy()
#     saa_attrs = data.attrs.copy()
#
#     sze_attrs['values'] = 'Solar zenith angle'
#     sze_attrs['scale_factor'] = 0.01
#
#     saa_attrs['values'] = 'Solar azimuth angle'
#     sze_attrs['scale_factor'] = 0.01
#
#     szex = xr.DataArray(data=da.from_array(sze[np.newaxis, :, :],
#                                            chunks=(1, data.gw.row_chunks, data.gw.col_chunks)),
#                         coords={'band': 'sze',
#                                 'y': data.y,
#                                 'x': data.x},
#                         dims=('band', 'y', 'x'),
#                         attrs=sze_attrs)
#
#     saax = xr.DataArray(data=da.from_array(saa[np.newaxis, :, :],
#                                            chunks=(1, data.gw.row_chunks, data.gw.col_chunks)),
#                         coords={'band': 'saa',
#                                 'y': data.y,
#                                 'x': data.x},
#                         dims=('band', 'y', 'x'),
#                         attrs=saa_attrs)
#
#     return szex, saax


def landsat_pixel_angles(angles_file,
                         ref_file,
                         outdir,
                         sensor,
                         l57_angles_path=None,
                         l8_angles_path=None,
                         verbose=0):

    """
    Generates Landsat pixel angle files

    Args:
        angles_file (str): The angles file.
        ref_file (str): A reference file.
        outdir (str): The output directory.
        sensor (str): The sensor.
        l57_angles_path (str): The path to the Landsat 5 and 7 angles bin.
        l8_angles_path (str): The path to the Landsat 8 angles bin.
        verbose (Optional[int]): The verbosity level.

    Returns:
        zenith and azimuth angles as a ``namedtuple`` of angle file names
    """

    if not l57_angles_path:

        gw_bin = os.path.realpath(os.path.dirname(__file__))

        gw_out = os.path.realpath(Path(gw_bin).joinpath('../bin').as_posix())
        gw_tar = os.path.realpath(Path(gw_bin).joinpath('../bin/ESPA.tar.gz').as_posix())

        if not Path(gw_bin).joinpath('../bin/ESPA').is_dir():

            with tarfile.open(gw_tar, mode='r:gz') as tf:
                tf.extractall(gw_out)

        l57_angles_path = Path(gw_out).joinpath('ESPA/landsat_angles').as_posix()
        l8_angles_path = Path(gw_out).joinpath('ESPA/l8_angles').as_posix()

    AngleInfo = namedtuple('AngleInfo', 'vza vaa sza saa')

    # Setup the angles name.
    # example file = LE07_L1TP_225098_20160911_20161008_01_T1_sr_band1.tif

    with rio.open(ref_file) as src:

        ref_height = src.height
        ref_width = src.width
        ref_extent = src.bounds

    ref_base = '_'.join(os.path.basename(ref_file).split('_')[:-1])

    opath = Path(outdir)

    opath.mkdir(parents=True, exist_ok=True)

    # Set output angle file names.
    sensor_azimuth_file = opath.joinpath(ref_base + '_sensor_azimuth.tif').as_posix()
    sensor_zenith_file = opath.joinpath(ref_base + '_sensor_zenith.tif').as_posix()
    solar_azimuth_file = opath.joinpath(ref_base + '_solar_azimuth.tif').as_posix()
    solar_zenith_file = opath.joinpath(ref_base + '_solar_zenith.tif').as_posix()

    if not Path(sensor_azimuth_file).is_file():

        # Setup the command.
        if sensor.lower() in ['l5', 'l7']:

            angle_command = '{PATH} {META} -s 1 -b 1'.format(PATH=Path(l57_angles_path).joinpath('landsat_angles').as_posix(),
                                                             META=angles_file)

            # 1=zenith, 2=azimuth
            out_order = dict(azimuth=2, zenith=1)
            # out_order = [2, 1, 2, 1]

        else:

            angle_command = '{PATH} {META} BOTH 1 -f -32768 -b 4'.format(PATH=Path(l8_angles_path).joinpath('l8_angles').as_posix(),
                                                                         META=angles_file)

            # 1=azimuth, 2=zenith
            out_order = dict(azimuth=1, zenith=2)
            # out_order = [1, 2, 1, 2]

        os.chdir(outdir)

        if verbose > 0:
            logger.info('  Generating pixel angles ...')

        # Create the angle files.
        subprocess.call(angle_command, shell=True)

        # Get angle data from 1 band.
        sensor_angles = fnmatch.filter(os.listdir(outdir), '*sensor_B04.img')[0]
        solar_angles = fnmatch.filter(os.listdir(outdir), '*solar_B04.img')[0]

        sensor_angles_fn_in = opath.joinpath(sensor_angles).as_posix()
        solar_angles_fn_in = opath.joinpath(solar_angles).as_posix()

        # Convert the data
        for in_angle, out_angle, band_pos in zip([sensor_angles_fn_in,
                                                  sensor_angles_fn_in,
                                                  solar_angles_fn_in,
                                                  solar_angles_fn_in],
                                                 [sensor_azimuth_file,
                                                  sensor_zenith_file,
                                                  solar_azimuth_file,
                                                  solar_zenith_file],
                                                 ['azimuth',
                                                  'zenith',
                                                  'azimuth',
                                                  'zenith']):

            with rio.open(in_angle) as src:

                profile = src.profile.copy()

                profile.update(transform=Affine(src.res[0], 0.0, ref_extent.left, 0.0, -src.res[1], ref_extent.top),
                               height=ref_height,
                               width=ref_width,
                               nodata=-32768,
                               dtype='int16',
                               count=1,
                               driver='GTiff',
                               tiled=True,
                               compress='lzw')

                src_band = rio.Band(src, out_order[band_pos], 'int16', (src.height, src.width))

                with rio.open(out_angle, mode='w', **profile) as dst:

                    dst_band = rio.Band(dst, 1, 'int16', (dst.height, dst.width))

                    # TODO: num_threads
                    reproject(src_band,
                              destination=dst_band,
                              num_threads=8)

    return AngleInfo(vaa=sensor_azimuth_file,
                     vza=sensor_zenith_file,
                     saa=solar_azimuth_file,
                     sza=solar_zenith_file)
