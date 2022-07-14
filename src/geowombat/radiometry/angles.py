import os
from pathlib import Path
import fnmatch
import subprocess
from collections import namedtuple
import tarfile
import logging

import geowombat as gw
from ..handler import add_handler
from ..backends import transform_crs

import numpy as np
import dask.array as da
import xarray as xr
import rasterio as rio
from rasterio.crs import CRS
from affine import Affine
import xml.etree.ElementTree as ET

try:
    import cv2
    OPENCV_INSTALLED = True
except:
    OPENCV_INSTALLED = False


logger = logging.getLogger(__name__)
logger = add_handler(logger)


def shift_objects(
    data,
    solar_za,
    solar_az,
    sensor_za,
    sensor_az,
    object_height,
    num_workers
):
    """Shifts objects along x and y dimensions

    Args:
        data (DataArray): The data to shift.
        solar_za (DataArray): The solar zenith angle.
        solar_az (DataArray): The solar azimuth angle.
        sensor_za (DataArray): The sensor, or view, zenith angle.
        sensor_az (DataArray): The sensor, or view, azimuth angle.
        object_height (float): The object height.
        num_workers (Optional[int]): The number of dask workers.

    Returns:
        ``xarray.DataArray``
    """
    # Scale the angles to degrees
    sza = solar_za * 0.01
    sza.coords['band'] = [1]

    saa = solar_az * 0.01
    saa.coords['band'] = [1]

    vza = sensor_za * 0.01
    vza.coords['band'] = [1]

    vaa = sensor_az * 0.01
    vaa.coords['band'] = [1]

    # Convert to radians
    rad_sza = np.deg2rad(sza)
    rad_saa = np.deg2rad(saa)
    rad_vza = np.deg2rad(vza)
    rad_vaa = np.deg2rad(vaa)

    apparent_solar_az = np.pi + np.arctan((np.sin(rad_saa) * np.tan(rad_sza) - np.sin(rad_vaa) * np.tan(rad_vza)) /
                                          (np.cos(rad_saa) * np.tan(rad_sza) - np.cos(rad_vaa) * np.tan(rad_vza)))

    # Maximum horizontal distance
    d = (object_height**2 * ((np.sin(rad_saa) * np.tan(rad_sza) - np.sin(rad_vaa) * np.tan(rad_vza))**2 +
                 (np.cos(rad_saa) * np.tan(rad_sza) - np.cos(rad_vaa) * np.tan(rad_vza))**2))**0.5

    # Convert the polar angle to cartesian offsets
    x = int((np.cos(apparent_solar_az) * d).max(skipna=True).data.compute(num_workers=num_workers))
    y = int((np.sin(apparent_solar_az) * d).max(skipna=True).data.compute(num_workers=num_workers))

    return data.shift(shifts={'x': x, 'y': y}, fill_value=0)


def estimate_cloud_shadows(
    data,
    clouds,
    solar_za,
    solar_az,
    sensor_za,
    sensor_az,
    heights=None,
    num_workers=1
):
    """Estimates shadows from a cloud mask and adds to the existing mask

    Args:
        data (DataArray): The wavelengths, scaled 0-1.
        clouds (DataArray): The cloud mask, where clouds=1 and clear sky=0.
        solar_za (DataArray): The solar zenith angle.
        solar_az (DataArray): The solar azimuth angle.
        sensor_za (DataArray): The sensor, or view, zenith angle.
        sensor_az (DataArray): The sensor, or view, azimuth angle.
        heights (Optional[list]): The cloud heights, in kilometers.
        num_workers (Optional[int]): The number of dask workers.

    Returns:
        ``xarray.DataArray``

    References:

        For the angle offset calculations, see :cite:`fisher_2014`.
        For the shadow test, see :cite:`sun_etal_2018`.
    """
    attrs = data.attrs.copy()

    if not heights:
        heights = list(range(200, 1400, 200))

    shadows = None

    for object_height in heights:
        potential_shadows = shift_objects(
            clouds,
            solar_za,
            solar_az,
            sensor_za,
            sensor_az,
            object_height,
            num_workers
        )

        if not isinstance(shadows, xr.DataArray):
            shadows = xr.where((data.sel(band='nir') < 0.25) & (data.sel(band='swir1') < 0.11) & (potential_shadows.sel(band='mask') == 1), 1, 0)
        else:
            shadows = xr.where(((data.sel(band='nir') < 0.25) & (data.sel(band='swir1') < 0.11) & (potential_shadows.sel(band='mask') == 1)) | shadows.sel(band='mask') == 1, 1, 0)

        shadows = shadows.expand_dims(dim='band')

    # Add the shadows to the cloud mask
    data = xr.where(clouds.sel(band='mask') == 1, 1, xr.where(shadows.sel(band='mask') == 1, 2, 0))
    data = data.expand_dims(dim='band')
    data.attrs = attrs

    return data


def scattering_angle(cos_sza, cos_vza, sin_sza, sin_vza, cos_raa):
    """Calculates the scattering angle

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
    """Calculates the relative azimuth angle

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


def get_sentinel_crs_transform(metadata, resample_res = 10.0):
    tree = ET.parse(metadata)
    root = tree.getroot()

    base_str = './/{https://psd-14.sentinel2.eo.esa.int/PSD/S2_PDI_Level-2A_Tile_Metadata.xsd}Geometric_Info/Tile_Geocoding/'
    crs = base_str + '/HORIZONTAL_CS_CODE'
    left = base_str + '/Geoposition[@resolution="10"]/ULX'
    top = base_str + '/Geoposition[@resolution="10"]/ULY'
    nrows = base_str + '/Size[@resolution="10"]/NROWS'
    ncols = base_str + '/Size[@resolution="10"]/NCOLS'
    crs = root.findall(crs)[0].text
    left = float(root.findall(left)[0].text)
    top = float(root.findall(top)[0].text)
    nrows = int(root.findall(nrows)[0].text)
    ncols = int(root.findall(ncols)[0].text)

    nrows = int(nrows / (resample_res / 10.0))
    ncols = int(ncols / (resample_res / 10.0))
    transform = Affine(resample_res, 0.0, left, 0.0, -resample_res, top)

    return crs, transform, nrows, ncols


def parse_sentinel_angles(metadata, proc_angles, nodata):
    """Gets the Sentinel-2 solar angles from metadata

    Reference:
        https://github.com/hevgyrt/safe_to_netcdf/blob/master/s2_reader_and_NetCDF_converter.py

    Args:
        metadata (str): The metadata file.
        proc_angles (str): The angles to parse. Choices are ['solar', 'view'].
        nodata (int or float): The 'no data' value.

    Returns:
        zenith and azimuth angles as a ``tuple`` of 2d ``numpy`` arrays
    """
    # Parse the XML file
    tree = ET.parse(metadata)
    root = tree.getroot()

    angles_view_list = root.findall('.//Tile_Angles')[0]
    row_step = float(root.findall('.//ROW_STEP')[0].text)
    col_step = float(root.findall('.//COL_STEP')[0].text)
    base_str = './/{https://psd-14.sentinel2.eo.esa.int/PSD/S2_PDI_Level-2A_Tile_Metadata.xsd}Geometric_Info/Tile_Geocoding/Size[@resolution="10"]'
    nrows_str = base_str + '/NROWS'
    ncols_str = base_str + '/NCOLS'
    nrows = int(root.findall(nrows_str)[0].text)
    ncols = int(root.findall(ncols_str)[0].text)
    angle_nrows = int(np.ceil(nrows * 10.0 / row_step)) + 1
    angle_ncols = int(np.ceil(ncols * 10.0 / col_step)) + 1

    if proc_angles == 'view':
        zenith_array = np.zeros((13, angle_nrows, angle_ncols), dtype='float64') + nodata
        azimuth_array = np.zeros((13, angle_nrows, angle_ncols), dtype='float64') + nodata
    else:
        zenith_array = np.zeros((angle_nrows, angle_ncols), dtype='float64') + nodata
        azimuth_array = np.zeros((angle_nrows, angle_ncols), dtype='float64') + nodata

    view_tag = 'Sun_Angles_Grid' if proc_angles == 'solar' else 'Viewing_Incidence_Angles_Grids'

    # Find the angles
    for angle in angles_view_list:
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
                    zenith_values_list = field

            for field in azimuth:
                if field.tag == 'Values_List':
                    azimuth_values_list = field

            for ridx, zenith_values in enumerate(zenith_values_list):
                zenith_values_array = np.array([float(i) for i in zenith_values.text.split()])
                if proc_angles == 'view':
                    zenith_array[band_id, ridx] = zenith_values_array
                else:
                    zenith_array[ridx] = zenith_values_array
            for ridx, azimuth_values in enumerate(azimuth_values_list):
                azimuth_values_array = np.array([float(i) for i in azimuth_values.text.split()])
                if proc_angles == 'view':
                    azimuth_array[band_id, ridx] = azimuth_values_array
                else:
                    azimuth_array[ridx] = azimuth_values_array

    return zenith_array, azimuth_array


def landsat_pixel_angles(
    angles_file,
    ref_file,
    out_dir,
    sensor,
    l57_angles_path=None,
    l8_angles_path=None,
    subsample=1,
    resampling='bilinear',
    num_workers=1,
    verbose=0
):
    """Generates Landsat pixel angle files

    Args:
        angles_file (str): The angles file.
        ref_file (str): A reference file.
        out_dir (str): The output directory.
        sensor (str): The sensor.
        l57_angles_path (str): The path to the Landsat 5 and 7 angles bin.
        l8_angles_path (str): The path to the Landsat 8 angles bin.
        subsample (Optional[int]): The sub-sample factor when calculating the angles.
        resampling (Optional[str]): The resampling method if ``filename`` is a ``list``.
            Choices are ['average', 'bilinear', 'cubic', 'cubic_spline', 'gauss', 'lanczos', 'max', 'med', 'min', 'mode', 'nearest'].
        num_workers (Optional[int]): The maximum number of concurrent workers.
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
        ref_res = src.res

    ref_base = '_'.join(os.path.basename(ref_file).split('_')[:-1])
    opath = Path(out_dir)
    opath.mkdir(parents=True, exist_ok=True)

    # Set output angle file names.
    sensor_azimuth_file = opath.joinpath(ref_base + '_sensor_azimuth.tif').as_posix()
    sensor_zenith_file = opath.joinpath(ref_base + '_sensor_zenith.tif').as_posix()
    solar_azimuth_file = opath.joinpath(ref_base + '_solar_azimuth.tif').as_posix()
    solar_zenith_file = opath.joinpath(ref_base + '_solar_zenith.tif').as_posix()

    if not Path(sensor_azimuth_file).is_file():
        # Setup the command.
        if sensor.lower() in ['l5', 'l7']:
            angle_command = '{PATH} {META} -s {SUBSAMP:d} -b 1'.format(
                PATH=str(Path(l57_angles_path).joinpath('landsat_angles')),
                META=angles_file,
                SUBSAMP=subsample
            )

            # 1=zenith, 2=azimuth
            out_order = dict(azimuth=2, zenith=1)
            # out_order = [2, 1, 2, 1]

        else:
            angle_command = '{PATH} {META} BOTH {SUBSAMP:d} -f -32768 -b 4'.format(
                PATH=str(Path(l8_angles_path).joinpath('l8_angles')),
                META=angles_file,
                SUBSAMP=subsample
            )

            # 1=azimuth, 2=zenith
            out_order = dict(azimuth=1, zenith=2)
            # out_order = [1, 2, 1, 2]

        os.chdir(out_dir)

        if verbose > 0:
            logger.info('  Generating pixel angles ...')

        # Create the angle files.
        subprocess.call(angle_command, shell=True)

        # Get angle data from 1 band.
        sensor_angles = fnmatch.filter(os.listdir(out_dir), '*sensor_B04.img')[0]
        solar_angles = fnmatch.filter(os.listdir(out_dir), '*solar_B04.img')[0]

        sensor_angles_fn_in = opath.joinpath(sensor_angles).as_posix()
        solar_angles_fn_in = opath.joinpath(solar_angles).as_posix()

        # Convert the data
        for in_angle, out_angle, band_pos in zip(
            [
                sensor_angles_fn_in,
                sensor_angles_fn_in,
                solar_angles_fn_in,
                solar_angles_fn_in
            ],
            [
                sensor_azimuth_file,
                sensor_zenith_file,
                solar_azimuth_file,
                solar_zenith_file
            ],
            [
                'azimuth',
                'zenith',
                'azimuth',
                'zenith'
            ]
        ):
            new_res = subsample*ref_res[0]

            # Update the .hdr file
            with open(in_angle + '.hdr', mode='r') as txt:
                lines = txt.readlines()

                for lidx, line in enumerate(lines):
                    if line.startswith('map info'):
                        lines[lidx] = line.replace('30.000, 30.000', f'{new_res:.3f}, {new_res:.3f}')

            Path(in_angle + '.hdr').unlink()

            with open(in_angle + '.hdr', mode='w') as txt:
                txt.writelines(lines)

            with gw.config.update(ref_image=ref_file):
                with gw.open(in_angle, chunks=256, resampling=resampling) as src:
                    src = src.sel(band=out_order[band_pos]).fillna(-32768).astype('int16')
                    src = src.assign_attrs(nodatavals=(-32768,)*src.gw.nbands)
                    src.gw.save(
                        out_angle,
                        overwrite=True,
                        num_workers=num_workers,
                        log_progress=True if verbose > 0 else False
                    )

    os.chdir(os.path.expanduser('~'))

    return AngleInfo(
        vaa=str(sensor_azimuth_file),
        vza=str(sensor_zenith_file),
        saa=str(solar_azimuth_file),
        sza=str(solar_zenith_file)
    )


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


def sentinel_pixel_angles(
    metadata,
    ref_file,
    outdir='.',
    nodata=-32768,
    overwrite=False,
    resampling='bilinear',
    verbose=0,
    num_workers=1,
    resample_res=60.0
):
    """Generates Sentinel pixel angle files

    Args:
        metadata (str): The metadata file.
        ref_file (str): A reference image to use for geo-information.
        outdir (Optional[str])): The output directory to save the angle files to.
        nodata (Optional[int or float]): The 'no data' value.
        overwrite (Optional[bool]): Whether to overwrite existing angle files.
        verbose (Optional[int]): The verbosity level.
        num_workers (Optional[int]): The maximum number of concurrent workers.
        resample_res (Optional[float]): The resolution to resample to.

    References:
        https://www.sentinel-hub.com/faq/how-can-i-access-meta-data-information-sentinel-2-l2a
        https://github.com/marujore/sentinel_angle_bands/blob/master/sentinel2_angle_bands.py

    Returns:
        zenith and azimuth angles as a ``namedtuple`` of angle file names
    """
    if not OPENCV_INSTALLED:
        logger.exception('OpenCV must be installed.')

    AngleInfo = namedtuple('AngleInfo', 'vza vaa sza saa sensor')

    crs, transform, nrows, ncols = get_sentinel_crs_transform(metadata, resample_res=resample_res)
    sza, saa = parse_sentinel_angles(metadata, 'solar', nodata)
    vza, vaa = parse_sentinel_angles(metadata, 'view', nodata)
    sensor_name = get_sentinel_sensor(metadata)

    # Create an array that is a representation of the full S-2 image
    xcoords = (transform * (np.arange(ncols) + 0.5, np.zeros(ncols) + 0.5))[0]
    ycoords = (transform * (np.zeros(nrows) + 0.5, np.arange(nrows) + 0.5))[1]

    ref_base = '_'.join(os.path.basename(ref_file).split('_')[:-1])

    opath = Path(outdir)
    opath.mkdir(parents=True, exist_ok=True)

    # Set output angle file names.
    sensor_azimuth_file = opath.joinpath(ref_base + '_sensor_azimuth.tif').as_posix()
    sensor_zenith_file = opath.joinpath(ref_base + '_sensor_zenith.tif').as_posix()
    solar_azimuth_file = opath.joinpath(ref_base + '_solar_azimuth.tif').as_posix()
    solar_zenith_file = opath.joinpath(ref_base + '_solar_zenith.tif').as_posix()
    full_image_file = opath.joinpath(ref_base + '_full.tif').as_posix()

    for angle_array, angle_file in zip(
            [vaa, vza, saa, sza],
            [
                sensor_azimuth_file,
                sensor_zenith_file,
                solar_azimuth_file,
                solar_zenith_file
            ]
    ):
        pfile = Path(angle_file)
        if overwrite:
            if pfile.is_file():
                pfile.unlink()

        if not pfile.is_file():
            # TODO: write data for each band?
            if len(angle_array.shape) > 2:
                angle_array = angle_array.mean(axis=0)

            # Resample and scale to the full image size
            angle_array_resamp = np.int16(cv2.resize(
                angle_array,
                (0, 0),
                fy=nrows / saa.shape[0],
                fx=ncols / saa.shape[1],
                interpolation=cv2.INTER_LINEAR
            ) / 0.01)
            # Create a DataArray to write to file
            angle_array_resamp = xr.DataArray(
                da.from_array(
                    angle_array_resamp[None],
                    chunks=(1, 256, 256)
                ),
                dims=('band', 'y', 'x'),
                coords={
                    'band': [1],
                    'y': ycoords,
                    'x': xcoords
                },
                attrs={
                    'transform': transform,
                    'crs': crs,
                    'res': (resample_res, resample_res),
                    'nodatavals': (nodata,)
                }
            )
            # Save the full image to file
            angle_array_resamp.gw.save(
                full_image_file,
                num_workers=num_workers,
                overwrite=True,
                log_progress=True if verbose > 0 else False
            )
            # Save the angle image to file using the reference bounds
            with gw.config.update(ref_image=ref_file):
                with gw.open(full_image_file, chunks=256, resample=resampling) as src:
                    src = src.fillna(nodata).astype('int16')
                    src = src.assign_attrs(nodatavals=(nodata,)*src.gw.nbands)
                    src.gw.save(
                        angle_file,
                        overwrite=True,
                        num_workers=num_workers,
                        log_progress=True if verbose > 0 else False
                    )

    return AngleInfo(
        vaa=str(sensor_azimuth_file),
        vza=str(sensor_zenith_file),
        saa=str(solar_azimuth_file),
        sza=str(solar_zenith_file),
        sensor=sensor_name
    )
