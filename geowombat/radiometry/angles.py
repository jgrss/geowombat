import os
from pathlib import Path
import fnmatch
import subprocess
from collections import namedtuple

from ..errors import logger

import rasterio as rio
from rasterio.warp import reproject
from affine import Affine


def gen_pixel_angles(angles_file, ref_file, outdir, sensor, l57_angles_path='.', l8_angles_path='.', verbose=0):

    """
    Prepares Landsat pixel angles

    Args:
        angles_file (str): The angles file.
        ref_file (str): A reference file.
        outdir (str): The output directory.
        sensor (str): The sensor.
        l57_angles_path (str)
        l8_angles_path (str)
        verbose (Optional[int]): The verbosity level.

    Returns:
        ``namedtuple``
    """

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

    # Setup the command.
    if sensor.lower() in ['l5', 'l7']:

        angle_command = '{PATH} {META} -s 1 -b 1'.format(PATH=Path(l57_angles_path).joinpath('landsat_angles'),
                                                         META=angles_file)

        # 1=zenith, 2=azimuth
        out_order = dict(azimuth=2, zenith=1)
        # out_order = [2, 1, 2, 1]

    else:

        angle_command = '{PATH} {META} BOTH 1 -f -32768 -b 4'.format(PATH=Path(l8_angles_path).joinpath('l8_angles'),
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
