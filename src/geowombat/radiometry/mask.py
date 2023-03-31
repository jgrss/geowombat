import numpy as np
import xarray as xr

from ..core import ndarray_to_xarray, norm_diff
from ..errors import logger

try:

    from s2cloudless import S2PixelCloudDetector

    S2CLOUDLESS_INSTALLED = True

except ImportError:
    S2CLOUDLESS_INSTALLED = False


def estimate_shadows(
    data,
    cloud_mask,
    solar_zenith,
    solar_azimuth,
    cloud_heights,
    nodata,
    scale_factor,
    num_workers,
):
    """Estimates shadows from masked clouds, solar angle, and solar azimuth.

    Args:
        data (DataArray)
        cloud_mask (DataArray)
        solar_zenith (DataArray)
        solar_azimuth (DataArray)
        cloud_heights (list)
        nodata (int | float)
        scale_factor (float)
        num_workers (int): The number of parallel compute workers.

    Reference:
        https://github.com/samsammurphy/cloud-masking-sentinel2/blob/master/cloud-masking-sentinel2.ipynb

    Returns:

        ``xarray.DataArray``:

            Data range: 0 to 1, where 0=non-shadow; 1=shadow
    """

    potential_shadows = []

    for cloud_height in cloud_heights:

        shadow_vector = np.tan(solar_zenith.sel(band=1)) * cloud_height

        # x and y components of shadow vector length
        # TODO: check if correct
        y = int(
            (
                (np.cos(solar_azimuth.sel(band=1)) * shadow_vector)
                / data.gw.celly
            )
            .round()
            .min()
            .data.compute(num_workers=num_workers)
        )
        x = -int(
            (
                (np.sin(solar_azimuth.sel(band=1)) * shadow_vector)
                / data.gw.celly
            )
            .round()
            .min()
            .data.compute(num_workers=num_workers)
        )

        # affine translation of clouds
        cloud_shift = cloud_mask.shift({'x': x, 'y': y}, fill_value=0)

        potential_shadows.append(cloud_shift)

    potential_shadows = xr.concat(potential_shadows, dim='band')
    potential_shadows = potential_shadows.assign_coords(
        coords={'band': list(range(1, len(cloud_heights) + 1))}
    )
    potential_shadows = potential_shadows.max(dim='band')
    potential_shadows = potential_shadows.expand_dims(dim='band')
    potential_shadows = potential_shadows.assign_coords(coords={'band': [1]})

    dark_pixels = norm_diff(
        data,
        'swir2',
        'green',
        sensor='s2',
        nodata=nodata,
        scale_factor=scale_factor,
    )

    shadows = xr.where(
        (potential_shadows.sel(band=1) >= 1)
        & (cloud_mask.sel(band=1) != 1)
        & (dark_pixels.sel(band='norm-diff') >= 0.1),
        1,
        0,
    )

    shadows = shadows.expand_dims(dim='band')
    shadows = shadows.assign_coords(coords={'band': [1]})

    return shadows


class CloudShadowMasker(object):
    @staticmethod
    def mask_s2(
        data,
        solar_za,
        solar_az,
        cloud_heights=None,
        nodata=None,
        scale_factor=1,
        num_workers=1,
        **kwargs
    ):
        """Masks Sentinel 2 data.

        Args:
            data (DataArray): The Sentinel 2 data to mask.
            solar_za (DataArray): The solar zenith angle.
            solar_az (DataArray): The solar azimuth angle.
            cloud_heights (Optional[list]): A list of potential cloud heights.
            nodata (Optional[int or float]): A 'no data' value to fill NAs with.
            scale_factor (Optional[float]): A scale factor to apply to the data.
            num_workers (Optional[int]): The number of parallel compute workers.
            kwargs (Optional[dict]): Keyword arguments for ``s2cloudless.S2PixelCloudDetector``.

        Returns:

            ``xarray.DataArray``:

                Data range: 0 to 4, where 0=clear; 2=shadow; 4=cloud

        Example:
            >>> import geowombat as gw
            >>> from geowombat.radiometry import mask_s2
            >>>
            >>> with gw.config.update(sensor='s2f', scale_factor=0.0001):
            >>>
            >>>     with gw.open('image.tif') as src, \
            >>>         gw.open('solar_zenith.tif') as sza, \
            >>>             gw.open('solar_azimuth.tif') as saa:
            >>>
            >>>         s2_mask = mask_s2(src, sza, saa)
        """

        # from ..radiometry.mask import CloudShadowMasker
        # mask_s2 = CloudShadowMasker().mask_s2
        #
        # mask = mask_s2(data,
        #                sza,
        #                saa,
        #                scale_factor=0.0001,
        #                nodata=0,
        #                num_workers=num_threads)
        #
        # fnmask = Path(load_bands_names[0]).name.split('.')[0]
        # mask.gw.to_raster(f'/media/jcgr/data/projects/global_fields/data/grids/ms/test/000960/{fnmask}_mask.tif',
        #                   n_workers=1, n_threads=1)
        #
        # if bands_out:
        #     data = _assign_attrs(data, attrs, bands_out)

        new_attrs = data.attrs.copy()

        if not cloud_heights:
            cloud_heights = list(range(500, 2000, 500))

        if not isinstance(nodata, int) and not isinstance(nodata, float):
            nodata = data.gw.nodata

        if scale_factor == 1.0:
            scale_factor = data.gw.scale_factor

        if S2CLOUDLESS_INSTALLED:
            if not kwargs:
                kwargs = dict(
                    threshold=0.4,
                    average_over=4,
                    dilation_size=5,
                    all_bands=False,
                )

            cloud_detector = S2PixelCloudDetector(**kwargs)

            # Get the S2Cloudless bands
            data_cloudless = data.sel(
                band=[
                    'coastal',
                    'blue',
                    'red',
                    'nir1',
                    'nir',
                    'rededge',
                    'water',
                    'cirrus',
                    'swir1',
                    'swir2',
                ]
            )

            # Scale from 0-10000 to 0-1
            if isinstance(nodata, int) or isinstance(nodata, float):
                data_cloudless = (
                    xr.where(
                        data_cloudless != nodata,
                        data_cloudless * scale_factor,
                        nodata,
                    )
                    .clip(0, 1)
                    .astype('float64')
                )
            else:
                data_cloudless = (
                    (data_cloudless * scale_factor)
                    .clip(0, 1)
                    .astype('float64')
                )

            # Reshape for predictions ..
            #   from [bands x rows x columns]
            #   to [images x rows x columns x bands]
            X = data_cloudless.transpose('y', 'x', 'band').data.compute(
                num_workers=num_workers
            )[np.newaxis, :, :, :]

            ################
            # Predict clouds
            ################

            # Convert from NumPy array to DataArray
            # clear=0, clouds=1
            cloud_mask = ndarray_to_xarray(
                data, cloud_detector.get_cloud_masks(X), [1]
            )

            #################
            # Predict shadows
            #################

            # Scale the angles to degrees
            sza = solar_za * 0.01
            sza.coords['band'] = [1]

            saa = solar_az * 0.01
            saa.coords['band'] = [1]

            # Convert to radians
            rad_sza = np.deg2rad(sza)
            rad_saa = np.deg2rad(saa)

            # non-shadow=0, shadows=1
            shadow_mask = estimate_shadows(
                data,
                cloud_mask,
                rad_sza,
                rad_saa,
                cloud_heights,
                nodata,
                scale_factor,
                num_workers,
            )

            # Recode for final output
            mask = (
                xr.where(
                    cloud_mask.sel(band=1) == 1,
                    4,
                    xr.where(
                        shadow_mask.sel(band=1) == 1,
                        2,
                        xr.where(data.max(dim='band') == nodata, 255, 0),
                    ),
                )
                .expand_dims(dim='band')
                .astype('uint8')
            )

            mask = mask.assign_coords(coords={'band': ['mask']})

            new_attrs['nodatavals'] = (255,)
            new_attrs['scales'] = (1.0,)
            new_attrs['offsets'] = (0.0,)
            new_attrs['pre-scaling'] = scale_factor
            new_attrs['sensor'] = 's2'
            new_attrs['clearval'] = (0,)
            new_attrs['shadowval'] = (2,)
            new_attrs['cloudval'] = (4,)
            new_attrs['fillval'] = (255,)

            mask = mask.assign_attrs(**new_attrs)

        else:
            logger.warning('  S2Cloudless is not installed.')
            mask = None

        return mask
