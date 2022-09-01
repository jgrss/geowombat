import os
from pathlib import Path
import shutil
import subprocess
from collections import namedtuple
import tarfile
import logging
import typing as T
from dataclasses import dataclass
import tempfile

import geowombat as gw
from geowombat.backends.xarray_rasterio_ import open_rasterio
from ..handler import add_handler
from ..backends import transform_crs

import numpy as np
import dask
from dask.delayed import Delayed, DelayedAttr, DelayedLeaf
import dask.array as da
import xarray as xr
import rasterio as rio
from rasterio.errors import TransformError
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


@dataclass
class AngleInfo:
    vza: T.Union[np.ndarray, xr.DataArray, da.Array, Delayed, DelayedAttr, DelayedLeaf]
    vaa: T.Union[np.ndarray, xr.DataArray, da.Array, Delayed, DelayedAttr, DelayedLeaf]
    sza: T.Union[np.ndarray, xr.DataArray, da.Array, Delayed, DelayedAttr, DelayedLeaf]
    saa: T.Union[np.ndarray, xr.DataArray, da.Array, Delayed, DelayedAttr, DelayedLeaf]


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


def scattering_angle(
    cos_sza: xr.DataArray,
    cos_vza: xr.DataArray,
    sin_sza: xr.DataArray,
    sin_vza: xr.DataArray,
    cos_raa: xr.DataArray
) -> xr.DataArray:
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
    scattering_angle = np.arccos(-cos_sza * cos_vza - sin_sza * sin_vza * cos_raa)

    return np.cos(scattering_angle) ** 2


def relative_azimuth(saa: xr.DataArray, vaa: xr.DataArray) -> xr.DataArray:
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
    raa = np.deg2rad(saa - vaa)

    # Create masks
    raa_plus = xr.where(raa >= 2.0*np.pi, 1, 0)
    raa_minus = xr.where(raa < 0, 1, 0)

    raa = xr.where(raa_plus == 1, raa - (2.0 * np.pi), raa)
    raa = xr.where(raa_minus == 1, raa + (2.0 * np.pi), raa)

    return np.fabs(np.rad2deg(raa))


def get_sentinel_sensor(metadata: T.Union[str, Path]) -> str:
    """Gets the Sentinel sensor from metadata

    Args:
        metadata (str | Path): The Sentinel metadata XML file path.
    """
    tree = ET.parse(str(metadata))
    root = tree.getroot()

    for child in root:
        if 'general_info' in child.tag[-14:].lower():
            general_info = child

    for ginfo in general_info:
        if ginfo.tag == 'TILE_ID':
            file_name = ginfo.text

    return file_name[:3].lower()


def get_sentinel_crs_transform(
    metadata: T.Union[str, Path], resample_res = 10.0
) -> tuple:
    """Gets the Sentinel scene transformation information

    Args:
        metadata (str | Path): The Sentinel metadata XML file path.
        resample_res (float | int): The cell resample resolution. Default is 10.0.
    """
    tree = ET.parse(str(metadata))
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


def get_sentinel_angle_shape(metadata: T.Union[str, Path]) -> tuple:
    """Gets the Sentinel scene angle array shape

    Args:
        metadata (str | Path): The Sentinel metadata XML file path.
    """
    # Parse the XML file
    tree = ET.parse(str(metadata))
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

    return angles_view_list, angle_nrows, angle_ncols


def parse_sentinel_angles(
    metadata: T.Union[str, Path], proc_angles: str, nodata: T.Union[float, int]
):
    """Gets the Sentinel-2 solar angles from metadata

    Reference:
        https://github.com/hevgyrt/safe_to_netcdf/blob/master/s2_reader_and_NetCDF_converter.py

    Args:
        metadata (str | Path): The Sentinel metadata XML file path.
        proc_angles (str): The angles to parse. Choices are ['solar', 'view'].
        nodata (int or float): The 'no data' value.

    Returns:
        zenith and azimuth angles as a ``tuple`` of 2d ``numpy`` arrays
    """
    # Parse the XML file
    tree = ET.parse(str(metadata))
    root = tree.getroot()

    angles_view_list, angle_nrows, angle_ncols = get_sentinel_angle_shape(metadata)

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


def run_espa_command(
    ref_angle_file: T.Union[str, Path],
    angles_file: T.Union[str, Path],
    sensor: str,
    l57_angles_path: T.Union[str, Path],
    l8_angles_path: T.Union[str, Path],
    subsample: T.Union[float, int],
    out_dir: T.Union[str, Path],
    verbose: int
) -> namedtuple:
    AnglePaths = namedtuple('AnglePaths', 'sensor solar out_order')

    angle_paths = AnglePaths(
        sensor=None,
        solar=None,
        out_order=None
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        if not Path(ref_angle_file).is_file():
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

            if verbose > 0:
                logger.info('  Generating pixel angles ...')

            # Create the angle files.
            os.chdir(str(tmp_dir))
            subprocess.call(angle_command, shell=True)
            os.chdir(os.path.expanduser('~'))

            # Get angle data from 1 band.
            sensor_angle_file = list(Path(tmp_dir).glob('*sensor_B04.img'))
            solar_angles_file = list(Path(tmp_dir).glob('*solar_B04.img'))
            if not sensor_angle_file or not solar_angles_file:
                raise ValueError('The angles were not created.')

            # Copy all the associated files
            for file_path in Path(tmp_dir).glob('*B04.img*'):
                shutil.move(str(file_path), str(Path(out_dir) / file_path.name))
            sensor_angle_file = list(Path(out_dir).glob('*sensor_B04.img'))[0]
            solar_angles_file = list(Path(out_dir).glob('*solar_B04.img'))[0]

            # Move the files out of the temporary directory
            sensor_angles_path = Path(out_dir) / sensor_angle_file.name
            solar_angles_path = Path(out_dir) / solar_angles_file.name

            angle_paths = AnglePaths(
                sensor=str(sensor_angles_path),
                solar=str(solar_angles_path),
                out_order=out_order
            )
        for file_ in Path(tmp_dir).iterdir():
            file_.unlink()

    return angle_paths


def open_angle_file(
    in_angle_path: T.Union[str, Path],
    chunks: int,
    band_pos: int,
    nodata: T.Union[float, int],
    subsample: int,
    ref_res: T.Sequence[float]
) -> xr.DataArray:
    new_res = subsample*ref_res[0]
    # Update the .hdr file
    with open(in_angle_path + '.hdr', mode='r') as txt:
        lines = txt.readlines()
        for lidx, line in enumerate(lines):
            if line.startswith('map info'):
                lines[lidx] = line.replace('30.000, 30.000', f'{new_res:.3f}, {new_res:.3f}')

    Path(in_angle_path + '.hdr').unlink()
    with open(in_angle_path + '.hdr', mode='w') as txt:
        txt.writelines(lines)

    with open_rasterio(in_angle_path, chunks=chunks) as data:
        angle_data = (
            data.sel(band=band_pos)
            .expand_dims(dim='band')
            .transpose('band', 'y', 'x')
            .fillna(nodata)
            .astype('int16')
        )

    return angle_data


def postprocess_espa_angles(
    ref_file: T.Union[str, Path],
    angle_paths_in: namedtuple,
    angle_paths_out: namedtuple,
    subsample: int,
    resampling: str,
    num_workers: int,
    chunks: int
) -> AngleInfo:
    nodata = -32768
    angle_array_dict = {
        'vaa': None,
        'vza': None,
        'saa': None,
        'sza': None
    }
    if angle_paths_in.sensor is not None:
        with rio.open(ref_file) as src:
            ref_res = src.res

        # Convert the data
        for in_angle, out_angle, band_pos in zip(
            [
                angle_paths_in.sensor,
                angle_paths_in.sensor,
                angle_paths_in.solar,
                angle_paths_in.solar
            ],
            [
                angle_paths_out.vaa,
                angle_paths_out.vza,
                angle_paths_out.saa,
                angle_paths_out.sza
            ],
            [
                'azimuth',
                'zenith',
                'azimuth',
                'zenith'
            ]
        ):
            if 'sensor_azimuth' in str(out_angle):
                angle_name = 'vaa'
            elif 'sensor_zenith' in str(out_angle):
                angle_name = 'vza'
            elif 'solar_azimuth' in str(out_angle):
                angle_name = 'saa'
            elif 'solar_zenith' in str(out_angle):
                angle_name = 'sza'

            angle_array_resamp_da = open_angle_file(
                in_angle,
                chunks,
                angle_paths_in.out_order[band_pos],
                nodata,
                subsample,
                ref_res
            )
            angle_array_dict = transform_angles(
                ref_file,
                angle_array_resamp_da,
                nodata,
                resampling,
                num_workers,
                angle_array_dict,
                angle_name
            )

    return AngleInfo(
        vaa=angle_array_dict['vaa'],
        vza=angle_array_dict['vza'],
        saa=angle_array_dict['saa'],
        sza=angle_array_dict['sza']
    )


def landsat_pixel_angles(
    angles_file: T.Union[str, Path],
    ref_file: T.Union[str, Path],
    out_dir: T.Union[str, Path],
    sensor: str,
    l57_angles_path: T.Union[str, Path] = None,
    l8_angles_path: T.Union[str, Path] = None,
    subsample: int = 1,
    resampling: str = 'bilinear',
    num_workers: int = 1,
    verbose: int = 0,
    chunks: int = 256
) -> AngleInfo:
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
        chunks (Optional[int]): The file chunk size. Default is 256.

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

    # Setup the angles name.
    # example file = LE07_L1TP_225098_20160911_20161008_01_T1_sr_band1.tif

    ref_base = '_'.join(os.path.basename(ref_file).split('_')[:-1])
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Set output angle file names.
    sensor_azimuth_path = str(out_path.joinpath(ref_base + '_sensor_azimuth.tif'))
    sensor_zenith_path = str(out_path.joinpath(ref_base + '_sensor_zenith.tif'))
    solar_azimuth_path = str(out_path.joinpath(ref_base + '_solar_azimuth.tif'))
    solar_zenith_path = str(out_path.joinpath(ref_base + '_solar_zenith.tif'))

    AnglePathsOut = namedtuple('AnglePathsOut', 'vaa vza saa sza')
    angle_paths_out = AnglePathsOut(
        vaa=sensor_azimuth_path,
        vza=sensor_zenith_path,
        saa=solar_azimuth_path,
        sza=solar_zenith_path
    )
    angle_paths_in = run_espa_command(
        angle_paths_out.vaa,
        angles_file,
        sensor,
        l57_angles_path,
        l8_angles_path,
        subsample,
        out_path,
        verbose
    )
    angle_info = dask.delayed(postprocess_espa_angles)(
        ref_file,
        angle_paths_in,
        angle_paths_out,
        subsample,
        resampling,
        num_workers,
        chunks
    )

    return angle_info


def resample_angles(
    angle_array: np.ndarray,
    nrows: int,
    ncols: int,
    data_shape: T.Tuple[int, int],
    chunksize: T.Tuple[int, int]
) -> da.Array:
    """Resamples an angle array
    """
    data = np.int16(
        cv2.resize(
            angle_array.mean(axis=0) if len(angle_array.shape) > 2 else angle_array,
            (0, 0),
            fy=nrows / data_shape[0],
            fx=ncols / data_shape[1],
            interpolation=cv2.INTER_LINEAR
        ) / 0.01
    )[None]

    return da.from_array(
        data,
        chunks=(1,) + chunksize
    )


def transform_angles(
    ref_file: T.Union[str, Path],
    data: xr.DataArray,
    nodata: T.Union[float, int],
    resampling: str,
    num_workers: int,
    angle_array_dict: dict,
    angle_name: str
) -> dict:
    """Transforms angles to a new CRS
    """
    with open_rasterio(ref_file, chunks=data.gw.row_chunks) as src:
        angle_array_resamp_da_transform = transform_crs(
            data,
            dst_crs=src.gw.crs_to_pyproj,
            dst_width=src.gw.ncols,
            dst_height=src.gw.nrows,
            dst_bounds=src.gw.bounds,
            src_nodata=nodata,
            dst_nodata=nodata,
            resampling=resampling,
            num_threads=num_workers
        )
        # Save the angle image to file
        angle_array_resamp_da_transform = (
            angle_array_resamp_da_transform
            .fillna(nodata)
            .astype('int16')
            .assign_attrs(nodatavals=(nodata,))
        )
        if not angle_array_resamp_da_transform.gw.bounds_overlay(src.gw.bounds):
            raise TransformError('The angle image was not correctly transformed.')
        angle_array_dict[angle_name] = angle_array_resamp_da_transform

    return angle_array_dict


def sentinel_pixel_angles(
    metadata: T.Union[str, Path],
    ref_file: T.Union[str, Path],
    nodata: T.Union[float, int] = -32768,
    resampling: str = 'bilinear',
    num_workers: int = 1,
    resample_res: T.Union[float, int] = 60.0,
    chunksize: T.Tuple[int, int] = None
) -> AngleInfo:
    """Generates Sentinel pixel angle files

    Args:
        metadata (str): The metadata file.
        ref_file (str): A reference image to use for geo-information.
        nodata (Optional[int or float]): The 'no data' value.
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

    if not chunksize:
        chunksize = (256, 256)
    # Get the CRS, transform, and image size of the full S-2 footprint
    crs, transform, nrows, ncols = get_sentinel_crs_transform(
        metadata, resample_res=resample_res
    )
    # Get the angle arrays
    angle_nrows, angle_ncols = get_sentinel_angle_shape(metadata)[1:]
    solar_angles = dask.delayed(parse_sentinel_angles)(metadata, 'solar', nodata)
    view_angles = dask.delayed(parse_sentinel_angles)(metadata, 'view', nodata)
    # Create coordinates that are a representation of the full S-2 image
    xcoords = (transform * (np.arange(ncols) + 0.5, np.zeros(ncols) + 0.5))[0]
    ycoords = (transform * (np.zeros(nrows) + 0.5, np.arange(nrows) + 0.5))[1]

    angle_array_dict = {
        'vaa': None,
        'vza': None,
        'saa': None,
        'sza': None
    }

    for angle_name, angle_array in zip(
        ['vza', 'vaa', 'sza', 'saa'],
        [view_angles[0], view_angles[1], solar_angles[0], solar_angles[1]]
    ):
        # Resample and scale to the full image size
        angle_array_resamp = dask.delayed(resample_angles)(
            angle_array, nrows, ncols, (angle_nrows, angle_ncols), chunksize
        )
        # Create a DataArray to write to file
        angle_array_resamp_da = xr.DataArray(
            da.from_delayed(
                angle_array_resamp,
                shape=(1, nrows, ncols),
                dtype='float32'
            ).rechunk((1,) + chunksize),
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
        angle_array_dict = dask.delayed(transform_angles)(
            ref_file,
            angle_array_resamp_da,
            nodata,
            resampling,
            num_workers,
            angle_array_dict,
            angle_name
        )

    return AngleInfo(
        vaa=angle_array_dict['vaa'],
        vza=angle_array_dict['vza'],
        saa=angle_array_dict['saa'],
        sza=angle_array_dict['sza']
    )
