import os
import fnmatch
import subprocess
from collections import namedtuple

import rasterio as rio


def gen_pixel_angles(angles_file, ref_file, outdir, sensor):

    """
    Prepares Landsat pixel angles

    Args:
        angles_file (str): The angles file.
        ref_file (str): A reference file.
        outdir (str): The output directory.
        sensor (str): The sensor.

    Returns:
        ``namedtuple``
    """

    AngleInfo = namedtuple('AngleInfo', 'senaz senze solaz solze')

    # Setup the angles name.
    # example file = LE07_L1TP_225098_20160911_20161008_01_T1_sr_band1.tif

    with rio.open(ref_file) as src:
        image_extent = src.bounds

    ref_base = '_'.join(os.path.basename(ref_file).split('_')[:-1])

    # Set output angle file names.
    sensor_azimuth_file = os.path.join(outdir, ref_base + '_sensor_azimuth.tif')
    sensor_zenith_file = os.path.join(outdir, ref_base + '_sensor_zenith.tif')
    solar_azimuth_file = os.path.join(outdir, ref_base + '_solar_azimuth.tif')
    solar_zenith_file = os.path.join(outdir, ref_base + '_solar_zenith.tif')

    # Setup the command.
    if sensor.upper() in ['ETM', 'TM', 'MSS']:

        angle_command = 'landsat_angles {META} -s 1 -b 1'.format(META=angles_file)

        # 1=zenith, 2=azimuth
        out_order = dict(azimuth=2, zenith=1)
        # out_order = [2, 1, 2, 1]

    else:

        angle_command = 'l8_angles {META} BOTH 1 -f -32768 -b 4'.format(META=angles_file)

        # 1=azimuth, 2=zenith
        out_order = dict(azimuth=1, zenith=2)
        # out_order = [1, 2, 1, 2]

    # Set the working directory
    if not os.path.isdir(outdir):
        return None

    os.chdir(outdir)

    # Create the angle files.
    subprocess.call(angle_command, shell=True)

    # Get angle data from 1 band.
    sensor_angles = fnmatch.filter(os.listdir(outdir), '*sensor_B04.img')[0]
    solar_angles = fnmatch.filter(os.listdir(outdir), '*solar_B04.img')[0]

    sensor_angles_fn_in = os.path.join(outdir, sensor_angles)
    solar_angles_fn_in = os.path.join(outdir, solar_angles)

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

        # TODO: replace with rasterio
        raster_tools.translate(in_angle,
                               out_angle,
                               noData=-32768,
                               cell_size=30.0,
                               d_type='int16',
                               bandList=[out_order[band_pos]],
                               projWin=[image_extent.left,
                                        image_extent.top,
                                        image_extent.right,
                                        image_extent.bottom],
                               creationOptions=['TILED=YES'])

    return AngleInfo(senaz=sensor_azimuth_file,
                     senze=sensor_zenith_file,
                     solaz=solar_azimuth_file,
                     solze=solar_zenith_file)