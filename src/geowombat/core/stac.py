import concurrent.futures
import enum
import os
import typing as T
import warnings
from pathlib import Path as _Path

import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from rasterio.enums import Resampling as _Resampling
from tqdm.auto import tqdm as _tqdm

from ..config import config
from ..radiometry import HLSFmaskBits as _HLSFmaskBits
from ..radiometry import QABits as _QABits
from ..radiometry import SCLValues as _SCLValues

try:
    import pystac
    import pystac.errors as pystac_errors
    import stackstac
    import wget
    from pystac.extensions.eo import EOExtension as _EOExtension
    from pystac_client import Client as _Client
    from rich.console import Console as _Console
    from rich.table import Table as _Table
except ImportError as e:
    pystac = None
    pystac_errors = None
    stackstac = None
    wget = None
    _EOExtension = None
    _Client = None
    _Console = None
    _Table = None
    warnings.warn(
        f"Install geowombat with 'pip install .[stac]' to use the STAC API. ({e})"
    )

try:
    from pydantic.errors import PydanticImportError
except ImportError:
    PydanticImportError = ImportError

try:
    import planetary_computer as pc
except (ImportError, PydanticImportError) as e:
    pc = None
    warnings.warn(
        f'The planetary-computer package did not import correctly. Use of the microsoft collection may be limited. ({e})'
    )


class StrEnum(str, enum.Enum):
    """
    Source:
        https://github.com/irgeek/StrEnum/blob/master/strenum/__init__.py
    """

    def __new__(cls, value, *args, **kwargs):
        return super().__new__(cls, value, *args, **kwargs)

    def __str__(self) -> str:
        return self.value


class STACNames(StrEnum):
    """STAC names."""

    ELEMENT84_V0 = 'element84_v0'
    ELEMENT84_V1 = 'element84_v1'
    MICROSOFT_V1 = 'microsoft_v1'
    NASA_LP_CLOUD = 'nasa_lp_cloud'


class STACCollections(StrEnum):
    # Copernicus DEM GLO-30
    COP_DEM_GLO_30 = 'cop_dem_glo_30'
    # All Landsat, Collection 2, Level 1
    LANDSAT_C2_L1 = 'landsat_c2_l1'
    # All Landsat, Collection 2, Level 2 (surface reflectance)
    LANDSAT_C2_L2 = 'landsat_c2_l2'
    # Sentinel-2, Level 2A (surface reflectance missing cirrus band)
    SENTINEL_S2_L2A = 'sentinel_s2_l2a'
    SENTINEL_S2_L2A_COGS = 'sentinel_s2_l2a_cogs'
    # Sentinel-2, Level 1C (top of atmosphere with all 13 bands available)
    SENTINEL_S2_L1C = 'sentinel_s2_l1c'
    # Sentinel-1, Level 1C Ground Range Detected (GRD)
    SENTINEL_S1_L1C = 'sentinel_s1_l1c'
    SENTINEL_3_LST = 'sentinel_3_lst'
    LANDSAT_L8_C2_L2 = 'landsat_l8_c2_l2'
    USDA_CDL = 'usda_cdl'
    IO_LULC = 'io_lulc'
    NAIP = 'naip'
    # Harmonized Landsat Sentinel-2
    HLS = 'hls'
    HLS_L30 = 'hls_l30'
    HLS_S30 = 'hls_s30'
    # ESA WorldCover 10m land cover
    ESA_WORLDCOVER = 'esa_worldcover'


class STACCollectionURLNames(StrEnum):
    # Copernicus DEM GLO-30
    COP_DEM_GLO_30 = STACCollections.COP_DEM_GLO_30.replace('_', '-')
    # All Landsat, Collection 2, Level 1
    LANDSAT_C2_L1 = STACCollections.LANDSAT_C2_L1.replace('_', '-')
    # All Landsat, Collection 2, Level 2 (surface reflectance)
    LANDSAT_C2_L2 = STACCollections.LANDSAT_C2_L2.replace('_', '-')
    # Sentinel-2, Level 2A (surface reflectance missing cirrus band)
    SENTINEL_S2_L2A = 'sentinel-2-l2a'
    SENTINEL_S2_L2A_COGS = STACCollections.SENTINEL_S2_L2A_COGS.replace(
        '_', '-'
    )
    # Sentinel-2, Level 1C (top of atmosphere with all 13 bands available)
    SENTINEL_S2_L1C = 'sentinel-2-l1c'
    # Sentinel-1, Level 1C Ground Range Detected (GRD)
    SENTINEL_S1_L1C = 'sentinel-1-grd'
    SENTINEL_3_LST = 'sentinel-3-slstr-lst-l2-netcdf'
    LANDSAT_L8_C2_L2 = 'landsat-8-c2-l2'
    USDA_CDL = STACCollections.USDA_CDL.replace('_', '-')
    IO_LULC = STACCollections.IO_LULC.replace('_', '-')
    NAIP = STACCollections.NAIP
    HLS = STACCollections.HLS
    HLS_L30 = 'HLSL30.v2.0'
    HLS_S30 = 'HLSS30.v2.0'
    ESA_WORLDCOVER = 'esa-worldcover'


STAC_CATALOGS = {
    STACNames.ELEMENT84_V0: 'https://earth-search.aws.element84.com/v0',
    STACNames.ELEMENT84_V1: 'https://earth-search.aws.element84.com/v1',
    # STACNames.google: 'https://earthengine.openeo.org/v1.0',
    STACNames.MICROSOFT_V1: 'https://planetarycomputer.microsoft.com/api/stac/v1',
    STACNames.NASA_LP_CLOUD: 'https://cmr.earthdata.nasa.gov/stac/LPCLOUD',
}

STAC_SCALING = {
    STACCollections.LANDSAT_C2_L2: {
        # https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2
        STACNames.MICROSOFT_V1: {
            'gain': 0.0000275,
            'offset': -0.2,
            'nodata': 0,
        },
    },
    STACCollections.HLS_L30: {
        # https://lpdaac.usgs.gov/products/hlsl30v002/
        STACNames.NASA_LP_CLOUD: {
            'gain': 0.0001,
            'offset': 0,
            'nodata': -9999,
        },
    },
    STACCollections.HLS_S30: {
        # https://lpdaac.usgs.gov/products/hlss30v002/
        STACNames.NASA_LP_CLOUD: {
            'gain': 0.0001,
            'offset': 0,
            'nodata': -9999,
        },
    },
}

STAC_COLLECTIONS = {
    STACNames.ELEMENT84_V0: (STACCollectionURLNames.SENTINEL_S2_L2A_COGS,),
    STACNames.ELEMENT84_V1: (
        STACCollectionURLNames.COP_DEM_GLO_30,
        STACCollectionURLNames.LANDSAT_C2_L2,
        STACCollectionURLNames.SENTINEL_S2_L2A,
        STACCollectionURLNames.SENTINEL_S2_L1C,
        STACCollectionURLNames.SENTINEL_S1_L1C,
        STACCollectionURLNames.NAIP,
    ),
    STACNames.MICROSOFT_V1: (
        STACCollectionURLNames.COP_DEM_GLO_30,
        STACCollectionURLNames.LANDSAT_C2_L1,
        STACCollectionURLNames.LANDSAT_C2_L2,
        STACCollectionURLNames.SENTINEL_S2_L2A,
        STACCollectionURLNames.SENTINEL_S1_L1C,
        STACCollectionURLNames.SENTINEL_3_LST,
        STACCollectionURLNames.LANDSAT_L8_C2_L2,
        STACCollectionURLNames.USDA_CDL,
        STACCollectionURLNames.IO_LULC,
        STACCollectionURLNames.ESA_WORLDCOVER,
    ),
    STACNames.NASA_LP_CLOUD: (
        STACCollectionURLNames.HLS_L30,
        STACCollectionURLNames.HLS_S30,
    ),
}

_LANDSAT_COLLECTIONS = {
    STACCollections.LANDSAT_C2_L1,
    STACCollections.LANDSAT_C2_L2,
    STACCollections.LANDSAT_L8_C2_L2,
}

_SENTINEL_S2_COLLECTIONS = {
    STACCollections.SENTINEL_S2_L2A,
    STACCollections.SENTINEL_S2_L2A_COGS,
}


def _is_landsat(collection: str) -> bool:
    return STACCollections(collection) in _LANDSAT_COLLECTIONS


def _is_sentinel_s2(collection: str) -> bool:
    return STACCollections(collection) in _SENTINEL_S2_COLLECTIONS


_HLS_COLLECTIONS = {
    STACCollections.HLS_L30,
    STACCollections.HLS_S30,
}


def _is_hls(collection: str) -> bool:
    return STACCollections(collection) in _HLS_COLLECTIONS


# HLS L30 band mapping: friendly name -> STAC asset key
_HLS_L30_BAND_MAP = {
    'coastal': 'B01',
    'blue': 'B02',
    'green': 'B03',
    'red': 'B04',
    'nir': 'B05',
    'swir1': 'B06',
    'swir2': 'B07',
    'cirrus': 'B09',
    'thermal1': 'B10',
    'thermal2': 'B11',
}

# HLS S30 band mapping: friendly name -> STAC asset key
_HLS_S30_BAND_MAP = {
    'coastal': 'B01',
    'blue': 'B02',
    'green': 'B03',
    'red': 'B04',
    'rededge1': 'B05',
    'rededge2': 'B06',
    'rededge3': 'B07',
    'nir_broad': 'B08',
    'nir': 'B8A',
    'water_vapor': 'B09',
    'cirrus': 'B10',
    'swir1': 'B11',
    'swir2': 'B12',
}


def _translate_hls_bands(
    bands: T.Sequence[str],
    collection: str,
) -> T.Tuple[T.List[str], T.Dict[str, str]]:
    """Translate friendly band names to STAC asset keys for HLS.

    Returns:
        Tuple of (translated_bands, reverse_map) where reverse_map
        maps STAC keys back to the original friendly names.
    """
    if STACCollections(collection) == STACCollections.HLS_L30:
        band_map = _HLS_L30_BAND_MAP
    elif STACCollections(collection) == STACCollections.HLS_S30:
        band_map = _HLS_S30_BAND_MAP
    else:
        return list(bands), {}

    translated = []
    reverse = {}
    for b in bands:
        if b in band_map:
            stac_key = band_map[b]
            translated.append(stac_key)
            reverse[stac_key] = b
        else:
            # Assume already a STAC asset key (e.g., 'B02')
            translated.append(b)
    return translated, reverse


def merge_stac(
    data: xr.DataArray, *other: T.Sequence[xr.DataArray]
) -> xr.DataArray:
    """Merges DataArrays by time.

    Args:
        data (DataArray): The ``DataArray`` to merge to.
        other (list of DataArrays): The ``DataArrays`` to merge.

    Returns:
        ``xarray.DataArray``
    """
    name = data.collection
    for darray in other:
        name += f'+{darray.collection}'
    collections = [data.collection] * data.gw.ntime
    for darray in other:
        collections += [darray.collection] * darray.gw.ntime

    stack = xr.DataArray(
        da.concatenate(
            (
                data.transpose('time', 'band', 'y', 'x').data,
                *(
                    darray.transpose('time', 'band', 'y', 'x').data
                    for darray in other
                ),
            ),
            axis=0,
        ),
        dims=('time', 'band', 'y', 'x'),
        coords={
            'time': np.concatenate(
                (data.time.values, *(darray.time.values for darray in other))
            ),
            'band': data.band.values,
            'y': data.y.values,
            'x': data.x.values,
        },
        attrs=data.attrs,
        name=name,
    ).sortby('time')

    return (
        stack.groupby('time')
        .mean(dim='time', skipna=True)
        .assign_attrs(**data.attrs)
        .assign_attrs(collection=None)
    )


def _download_worker(item, extra: str, out_path: _Path) -> dict:
    """Downloads a single STAC item 'extra'."""
    df_dict = {'id': item.id}
    url = item.assets[extra].to_dict()['href']
    out_name = out_path / f"{item.id}_{_Path(url.split('?')[0]).name}"
    df_dict[extra] = str(out_name)
    if not out_name.is_file():
        wget.download(url, out=str(out_name), bar=None)

    return df_dict


def open_stac(
    stac_catalog: str = STACNames.ELEMENT84_V1,
    collection: str = None,
    bounds: T.Union[T.Sequence[float], str, _Path, gpd.GeoDataFrame] = None,
    proj_bounds: T.Sequence[float] = None,
    start_date: str = None,
    end_date: str = None,
    cloud_cover_perc: T.Union[float, int] = None,
    bands: T.Sequence[str] = None,
    chunksize: int = 256,
    mask_items: str = None,
    bounds_query: str = None,
    mask_data: T.Optional[bool] = False,
    epsg: int = None,
    resolution: T.Union[float, int] = None,
    resampling: T.Optional[_Resampling] = _Resampling.nearest,
    nodata_fill: T.Union[float, int] = None,
    view_asset_keys: bool = False,
    extra_assets: T.Optional[T.Sequence[str]] = None,
    out_path: T.Union[_Path, str] = '.',
    max_items: int = 100,
    max_extra_workers: int = 1,
    compute: bool = True,
    num_workers: int = 4,
) -> xr.DataArray:
    """Opens a collection from a spatio-temporal asset catalog (STAC).

    Args:
        stac_catalog (str): Choices are ['element84_v0', 'element84_v1', 'microsoft_v1', 'nasa_lp_cloud'].
        collection (str): The STAC collection to open.
            Catalog options:
                element84_v0:
                    sentinel_s2_l2a_cogs
                element84_v1:
                    cop_dem_glo_30
                    landsat_c2_l2
                    sentinel_s2_l2a
                    sentinel_s2_l1c
                    sentinel_s1_l1c
                    naip
                microsoft_v1:
                    cop_dem_glo_30
                    landsat_c2_l1
                    landsat_c2_l2
                    landsat_l8_c2_l2
                    sentinel_s2_l2a
                    sentinel_s1_l1c
                    sentinel_3_lst
                    io_lulc
                    usda_cdl
                    esa_worldcover
                nasa_lp_cloud:
                    hls_l30 (HLS Landsat 30m)
                    hls_s30 (HLS Sentinel-2 30m)

        bounds (sequence | str | Path | GeoDataFrame): The search bounding box. This can also be given with the
            configuration manager (e.g., ``gw.config.update(ref_bounds=bounds)``). The bounds CRS
            must be given in WGS/84 lat/lon (i.e., EPSG=4326).
        proj_bounds (sequence): The projected bounds to return data. If ``None`` (default), the returned bounds
            are the union of all collection scenes. See ``bounds`` in
            https://github.com/gjoseph92/stackstac/blob/main/stackstac/stack.py for details.
        start_date (str): The start search date (yyyy-mm-dd).
        end_date (str): The end search date (yyyy-mm-dd).
        cloud_cover_perc (float | int): The maximum percentage cloud cover.
        bands (sequence): The bands to open.
        chunksize (int): The dask chunk size.
        mask_items (sequence): The items to mask.
            For Landsat: QA bit names. Defaults to
            ``['fill', 'dilated_cloud', 'cirrus', 'cloud',
            'cloud_shadow', 'snow']``.
            For Sentinel-2: SCL class names. Defaults to
            ``['no_data', 'saturated_defective', 'cloud_shadow',
            'cloud_medium_prob', 'cloud_high_prob', 'thin_cirrus']``.
            For HLS: Fmask bit names. Defaults to
            ``['cirrus', 'cloud', 'adjacent_cloud',
            'cloud_shadow', 'snow_ice']``.
        bounds_query (Optional[str]): A query to select bounds
            from the ``geopandas.GeoDataFrame``.
        mask_data (Optional[bool]): Whether to mask the data.
            When ``True``, the appropriate QA/SCL/Fmask band is
            automatically loaded and used for masking.
        epsg (Optional[int]): An EPSG code to warp to.
        resolution (Optional[float | int]): The cell resolution to resample to.
        resampling (Optional[rasterio.enumsResampling enum]): The resampling method.
        nodata_fill (Optional[float | int]): A fill value to replace 'no data' NaNs.
        view_asset_keys (Optional[bool]): Whether to view asset ids.
        extra_assets (Optional[list]): Extra assets (non-image assets) to download.
        out_path (Optional[str | Path]): The output path to save files to.
        max_items (Optional[int]): The maximum number of items to return from the search, even if there are more
            matching results, passed to ``pystac_client.ItemSearch``.
            See https://pystac-client.readthedocs.io/en/latest/api.html#pystac_client.ItemSearch for details.
        max_extra_workers (Optional[int]): The maximum number of extra assets to download concurrently.
        compute (Optional[bool]): Whether to eagerly load data into memory.
            If ``True`` (default), downloads all remote data with a
            progress bar using parallel threads.
            If ``False``, returns a lazy dask-backed array.
        num_workers (Optional[int]): Number of threads for parallel
            downloads when ``compute=True``. Default is 4. Higher
            values can speed up I/O-bound downloads from cloud storage.

    Returns:
        ``xarray.DataArray``

    Examples:
        >>> from geowombat.core.stac import open_stac, merge_stac
        >>>
        >>> data_l, df_l = open_stac(
        >>>     stac_catalog='microsoft_v1',
        >>>     collection='landsat_c2_l2',
        >>>     start_date='2020-01-01',
        >>>     end_date='2021-01-01',
        >>>     bounds='map.geojson',
        >>>     bands=['red', 'green', 'blue', 'qa_pixel'],
        >>>     mask_data=True,
        >>>     extra_assets=['ang', 'mtl.txt', 'mtl.xml']
        >>> )
        >>>
        >>> from rasterio.enums import Resampling
        >>>
        >>> data_s2, df_s2 = open_stac(
        >>>     stac_catalog='element84_v1',
        >>>     collection='sentinel_s2_l2a',
        >>>     start_date='2020-01-01',
        >>>     end_date='2021-01-01',
        >>>     bounds='map.geojson',
        >>>     bands=['blue', 'green', 'red'],
        >>>     resampling=Resampling.cubic,
        >>>     epsg=int(data_l.epsg.values),
        >>>     extra_assets=['granule_metadata']
        >>> )
        >>>
        >>> # Merge two temporal stacks
        >>> stack = (
        >>>     merge_stac(data_l, data_s2)
        >>>     .sel(band='red')
        >>>     .mean(dim='time')
        >>> )
    """
    if collection is None:
        raise NameError('A collection must be given.')

    df = pd.DataFrame()

    # Date range
    date_range = f"{start_date}/{end_date}"
    # Search bounding box
    if bounds is None:
        bounds = config['ref_bounds']
    assert bounds is not None, 'The bounds must be given in some format.'
    if not isinstance(bounds, (gpd.GeoDataFrame, tuple, list)):
        bounds = gpd.read_file(bounds)
        assert bounds.crs == pyproj.CRS.from_epsg(
            4326
        ), 'The CRS should be WGS84/latlon (EPSG=4326)'
    if (bounds_query is not None) and isinstance(bounds, gpd.GeoDataFrame):
        bounds = bounds.query(bounds_query)
    if isinstance(bounds, gpd.GeoDataFrame):
        bounds = tuple(bounds.total_bounds.flatten().tolist())

    try:
        stac_catalog_url = STAC_CATALOGS[stac_catalog]
        # Open the STAC catalog
        catalog = _Client.open(stac_catalog_url)
    except ValueError as e:
        raise NameError(
            f'The STAC catalog {stac_catalog} is not supported ({e}).'
        )

    # NASA Earthdata auth check
    gdal_env_dict = {}
    if STACNames(stac_catalog) == STACNames.NASA_LP_CLOUD:
        netrc_path = _Path.home() / '.netrc'
        has_netrc = netrc_path.exists()
        if has_netrc:
            netrc_content = netrc_path.read_text()
            has_netrc = 'urs.earthdata.nasa.gov' in netrc_content
        if not has_netrc:
            raise PermissionError(
                "NASA Earthdata authentication is required for "
                "HLS data access but no credentials were found.\n\n"
                "1. Register at: "
                "https://urs.earthdata.nasa.gov/users/new\n\n"
                "2. Create a ~/.netrc file with:\n\n"
                "  machine urs.earthdata.nasa.gov\n"
                "  login <your_username>\n"
                "  password <your_password>\n\n"
                "3. Set file permissions:\n"
                "  Linux/macOS:  chmod 600 ~/.netrc\n"
                "  Windows:      icacls %USERPROFILE%\\.netrc "
                "/inheritance:r /grant:r %USERNAME%:R"
            )
        gdal_env_dict = {
            'GDAL_HTTP_COOKIEFILE': '/tmp/gw_cookies.txt',
            'GDAL_HTTP_COOKIEJAR': '/tmp/gw_cookies.txt',
            'GDAL_HTTP_TIMEOUT': '60',
            'GDAL_HTTP_MAX_RETRY': '3',
            'GDAL_HTTP_RETRY_DELAY': '5',
        }

    if (
        STACCollectionURLNames[STACCollections(collection).name]
        not in STAC_COLLECTIONS[stac_catalog]
    ):
        raise NameError(f'The STAC collection {collection} is not supported.')

    catalog_collections = [
        STACCollectionURLNames[STACCollections(collection).name]
    ]

    query = None
    if cloud_cover_perc is not None:
        query = {"eo:cloud_cover": {"lt": cloud_cover_perc}}

    # Search the STAC
    print(f"Searching {stac_catalog} for {collection}...")
    search = catalog.search(
        collections=catalog_collections,
        bbox=bounds,
        datetime=date_range,
        query=query,
        max_items=max_items,
        limit=max_items,
    )

    if search is None:
        raise ValueError('No items found.')

    if list(search.items()):
        if STACNames(stac_catalog) == STACNames.MICROSOFT_V1:
            items = pc.sign(search)
        else:
            items = pystac.ItemCollection(items=list(search.items()))
        print(f"Found {len(items)} items.")

        if view_asset_keys:
            try:
                selected_item = min(
                    items, key=lambda item: _EOExtension.ext(item).cloud_cover
                )
            except pystac_errors.ExtensionNotImplemented:
                selected_item = items.items[0]

            table = _Table("Asset Key", "Description")
            for asset_key, asset in selected_item.assets.items():
                table.add_row(asset_key, asset.title)
            console = _Console()
            console.print(table)

            return None, None

        # Download metadata and coefficient files
        if extra_assets is not None:
            out_path = _Path(out_path)
            out_path.mkdir(parents=True, exist_ok=True)

            df_dicts: T.List[dict] = []
            with _tqdm(
                desc='Extra assets', total=len(items) * len(extra_assets)
            ) as pbar:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_extra_workers
                ) as executor:
                    futures = {
                        executor.submit(
                            _download_worker, item, extra, out_path
                        ): extra
                        for extra in extra_assets
                        for item in items
                    }
                    for future in concurrent.futures.as_completed(futures):
                        df_dict = future.result()
                        df_dicts.append(df_dict)
                        pbar.update(1)

            for item in items:
                d = {'id': item.id}
                for downloaded_dict in df_dicts:
                    if downloaded_dict['id'] == item.id:
                        d.update(downloaded_dict)
                df = pd.concat((df, pd.DataFrame([d])), ignore_index=True)

        # Auto-inject QA/SCL/Fmask band for pixel-level masking
        stack_bands = list(bands) if bands else []

        # Translate friendly band names for HLS collections
        band_reverse_map = {}
        if _is_hls(collection) and stack_bands:
            stack_bands, band_reverse_map = _translate_hls_bands(
                stack_bands, collection
            )

        if mask_data and bands is not None:
            if _is_landsat(collection):
                if 'qa_pixel' not in stack_bands:
                    stack_bands = stack_bands + ['qa_pixel']
            elif _is_sentinel_s2(collection):
                if 'scl' not in stack_bands:
                    stack_bands = stack_bands + ['scl']
            elif _is_hls(collection):
                if 'Fmask' not in stack_bands:
                    stack_bands = stack_bands + ['Fmask']

        stack_kwargs = dict(
            bounds=proj_bounds,
            bounds_latlon=None if proj_bounds is not None else bounds,
            assets=stack_bands or bands,
            chunksize=chunksize,
            epsg=epsg,
            resolution=resolution,
            resampling=resampling,
            properties=False,
            rescale=False,
        )
        if gdal_env_dict:
            stack_kwargs['gdal_env'] = stackstac.DEFAULT_GDAL_ENV.updated(
                always=gdal_env_dict
            )

        data = stackstac.stack(items, **stack_kwargs)

        # Rename HLS bands back to friendly names
        if band_reverse_map:
            new_band_names = [
                band_reverse_map.get(str(b), str(b))
                for b in data.band.values
            ]
            data = data.assign_coords(band=new_band_names)
        data = data.assign_attrs(
            res=(data.resolution, data.resolution), collection=collection
        )
        attrs = data.attrs.copy()

        if mask_data:
            if _is_landsat(collection):
                if mask_items is None:
                    mask_items = [
                        'fill',
                        'dilated_cloud',
                        'cirrus',
                        'cloud',
                        'cloud_shadow',
                        'snow',
                    ]
                mask_bitfields = [
                    getattr(_QABits, collection).value[mask_item]
                    for mask_item in mask_items
                ]
                bitmask = 0
                for field in mask_bitfields:
                    bitmask |= 1 << field
                qa = data.sel(band='qa_pixel').astype('uint16')
                mask = qa & bitmask
                data = data.sel(
                    band=[
                        b
                        for b in data.band.values
                        if b != 'qa_pixel'
                    ]
                ).where(mask == 0)

            elif _is_sentinel_s2(collection):
                if mask_items is None:
                    mask_items = [
                        'no_data',
                        'saturated_defective',
                        'cloud_shadow',
                        'cloud_medium_prob',
                        'cloud_high_prob',
                        'thin_cirrus',
                    ]
                scl_values = getattr(
                    _SCLValues, 'sentinel_s2_l2a'
                ).value
                bad_values = [
                    scl_values[item] for item in mask_items
                ]
                scl = data.sel(band='scl')
                scl_mask = scl.isin(bad_values)
                data = data.sel(
                    band=[
                        b
                        for b in data.band.values
                        if b != 'scl'
                    ]
                ).where(~scl_mask)

            elif _is_hls(collection):
                if mask_items is None:
                    mask_items = [
                        'cirrus',
                        'cloud',
                        'adjacent_cloud',
                        'cloud_shadow',
                        'snow_ice',
                    ]
                mask_bitfields = [
                    _HLSFmaskBits.hls.value[mask_item]
                    for mask_item in mask_items
                ]
                bitmask = 0
                for field in mask_bitfields:
                    bitmask |= 1 << field
                # Fmask band name (friendly or raw)
                fmask_name = (
                    'fmask'
                    if 'fmask' in data.band.values
                    else 'Fmask'
                )
                qa = data.sel(band=fmask_name).astype('uint8')
                mask = qa & bitmask
                data = data.sel(
                    band=[
                        b
                        for b in data.band.values
                        if b not in ('fmask', 'Fmask')
                    ]
                ).where(mask == 0)

            else:
                warnings.warn(
                    f"mask_data=True is not supported for "
                    f"collection '{collection}'."
                )

        if STACCollections(collection) in STAC_SCALING:
            scaling = STAC_SCALING[STACCollections(collection)][
                STACNames(stac_catalog)
            ]
            if scaling:
                data = xr.where(
                    data == scaling['nodata'],
                    np.nan,
                    (data * scaling['gain'] + scaling['offset']).clip(0, 1),
                ).assign_attrs(**attrs)

        if nodata_fill is not None:
            data = data.fillna(nodata_fill).gw.assign_nodata_attrs(nodata_fill)

        if not df.empty:
            df = df.set_index('id').reindex(data.id.values).reset_index()

        if compute:
            import dask
            from tqdm.dask import TqdmCallback

            try:
                with dask.config.set(
                    scheduler='threads', num_workers=num_workers
                ):
                    with TqdmCallback(
                        desc=f"Downloading {collection}"
                    ):
                        data = data.compute()
            except RuntimeError as e:
                if (
                    STACNames(stac_catalog) == STACNames.NASA_LP_CLOUD
                    and 'not recognized as' in str(e)
                ):
                    raise RuntimeError(
                        "NASA Earthdata authentication failed. "
                        "GDAL received an HTML login page instead "
                        "of raster data.\n\n"
                        "To fix this, create a ~/.netrc file with "
                        "your NASA Earthdata credentials:\n\n"
                        "  machine urs.earthdata.nasa.gov\n"
                        "  login <your_username>\n"
                        "  password <your_password>\n\n"
                        "Then set file permissions:\n"
                        "  Linux/macOS:  chmod 600 ~/.netrc\n"
                        "  Windows:      icacls %USERPROFILE%\\.netrc "
                        "/inheritance:r /grant:r %USERNAME%:R\n\n"
                        "Register at: "
                        "https://urs.earthdata.nasa.gov/users/new"
                    ) from e
                raise

        return data, df

    warnings.warn("No asset items were found.")

    return None, None


def composite_stac(
    stac_catalog: str = STACNames.ELEMENT84_V1,
    collection: str = None,
    bounds: T.Union[
        T.Sequence[float], str, _Path, gpd.GeoDataFrame
    ] = None,
    proj_bounds: T.Sequence[float] = None,
    start_date: str = None,
    end_date: str = None,
    cloud_cover_perc: T.Union[float, int] = None,
    bands: T.Sequence[str] = None,
    chunksize: int = 256,
    mask_items: T.Optional[T.Sequence[str]] = None,
    bounds_query: str = None,
    epsg: int = None,
    resolution: T.Union[float, int] = None,
    resampling: T.Optional[_Resampling] = _Resampling.nearest,
    nodata_fill: T.Union[float, int] = None,
    frequency: str = 'MS',
    max_items: int = 100,
    compute: bool = True,
    num_workers: int = 4,
) -> T.Optional[
    T.Tuple[xr.DataArray, pd.DataFrame]
]:
    """Creates cloud-free temporal composites from STAC data.

    Wraps ``open_stac()`` to produce median composites at a
    specified temporal frequency. Data is cloud-masked using
    pixel-level QA (Landsat ``qa_pixel``), SCL (Sentinel-2),
    or Fmask (HLS) bands, then aggregated using median
    resampling.

    Args:
        stac_catalog (str): The STAC catalog.
            See ``open_stac()`` for options. Ignored when
            ``collection='hls'`` (uses ``nasa_lp_cloud``).
        collection (str): The STAC collection.
            See ``open_stac()`` for options. Use ``'hls'``
            to query both ``hls_l30`` and ``hls_s30``, merge
            observations, then composite. Only bands common
            to both sensors are allowed (``blue``, ``green``,
            ``red``, ``nir``, ``swir1``, ``swir2``,
            ``coastal``, ``cirrus``).
        bounds: The search bounding box.
            See ``open_stac()``.
        proj_bounds: The projected bounds.
            See ``open_stac()``.
        start_date (str): The start search date (yyyy-mm-dd).
        end_date (str): The end search date (yyyy-mm-dd).
        cloud_cover_perc (float | int): Maximum cloud cover
            percentage for scene-level filtering.
        bands (sequence): The bands to open. Do not include
            ``qa_pixel``, ``scl``, or ``Fmask``; these are
            added automatically for masking.
        chunksize (int): The dask chunk size.
        mask_items (sequence): Items to mask.
            See ``open_stac()`` for sensor-specific defaults.
        bounds_query (str): A query for GeoDataFrame bounds.
        epsg (int): An EPSG code to warp to.
        resolution (float | int): Cell resolution.
        resampling: The resampling method.
        nodata_fill (float | int): Fill value for nodata.
        frequency (str): Pandas offset alias for temporal
            grouping. Default ``'MS'`` (month start). Other
            values: ``'W'`` (weekly), ``'QS'`` (quarter),
            ``'YS'`` (yearly).
        max_items (int): Maximum STAC search items.
        compute (bool): Whether to eagerly load data.
        num_workers (int): Number of threads for parallel
            downloads when ``compute=True``. Default is 4.

    Returns:
        tuple of (``xarray.DataArray``, ``pandas.DataFrame``)
        or ``(None, None)`` if no data found.

    Examples:
        >>> from geowombat.core.stac import composite_stac
        >>>
        >>> # Monthly median composite of Sentinel-2
        >>> composite, df = composite_stac(
        ...     collection='sentinel_s2_l2a',
        ...     start_date='2022-01-01',
        ...     end_date='2022-12-31',
        ...     bounds='aoi.geojson',
        ...     bands=['blue', 'green', 'red', 'nir'],
        ...     cloud_cover_perc=50,
        ...     frequency='MS',
        ...     resolution=10.0,
        ... )
        >>>
        >>> # Quarterly composite of Landsat
        >>> composite, df = composite_stac(
        ...     stac_catalog='microsoft_v1',
        ...     collection='landsat_c2_l2',
        ...     start_date='2022-01-01',
        ...     end_date='2022-12-31',
        ...     bounds='aoi.geojson',
        ...     bands=['red', 'green', 'blue'],
        ...     cloud_cover_perc=30,
        ...     frequency='QS',
        ...     resolution=30.0,
        ... )
        >>>
        >>> # Combined HLS (Landsat + Sentinel-2) composite
        >>> composite, df = composite_stac(
        ...     collection='hls',
        ...     start_date='2023-06-01',
        ...     end_date='2023-08-31',
        ...     bounds=(-77.1, 38.85, -76.95, 38.95),
        ...     bands=['blue', 'green', 'red', 'nir'],
        ...     epsg=32618,
        ...     resolution=30.0,
        ...     frequency='MS',
        ... )
    """
    # Combined HLS: query both L30 and S30, merge, then composite
    if collection == STACCollections.HLS:
        _hls_common = set(_HLS_L30_BAND_MAP) & set(
            _HLS_S30_BAND_MAP
        )
        if bands:
            bad = {
                b
                for b in bands
                if b not in _hls_common and not b.startswith('B')
            }
            if bad:
                raise ValueError(
                    f"Bands {bad} are not common to both "
                    f"HLS L30 and S30. Use one of: "
                    f"{sorted(_hls_common)}"
                )

        shared_kwargs = dict(
            stac_catalog=STACNames.NASA_LP_CLOUD,
            bounds=bounds,
            proj_bounds=proj_bounds,
            start_date=start_date,
            end_date=end_date,
            cloud_cover_perc=cloud_cover_perc,
            bands=bands,
            chunksize=chunksize,
            mask_items=mask_items,
            bounds_query=bounds_query,
            mask_data=True,
            epsg=epsg,
            resolution=resolution,
            resampling=resampling,
            nodata_fill=nodata_fill,
            max_items=max_items,
            compute=False,
        )
        r_l30 = open_stac(collection='hls_l30', **shared_kwargs)
        r_s30 = open_stac(collection='hls_s30', **shared_kwargs)

        has_l30 = r_l30 is not None and r_l30[0] is not None
        has_s30 = r_s30 is not None and r_s30[0] is not None

        if not has_l30 and not has_s30:
            return None, None

        parts = []
        dfs = []
        if has_l30:
            parts.append(r_l30[0])
            dfs.append(r_l30[1])
        if has_s30:
            parts.append(r_s30[0])
            dfs.append(r_s30[1])

        # Download each sensor separately so users see progress
        if compute:
            import dask
            from tqdm.dask import TqdmCallback

            _labels = []
            if has_l30:
                _labels.append('HLS L30 (Landsat)')
            if has_s30:
                _labels.append('HLS S30 (Sentinel-2)')
            for i, label in enumerate(_labels):
                with dask.config.set(
                    scheduler='threads',
                    num_workers=num_workers,
                ):
                    with TqdmCallback(desc=label):
                        parts[i] = parts[i].compute()

        if len(parts) == 1:
            data = parts[0]
        else:
            # Simple concatenation along time — skip merge_stac's
            # groupby('time').mean() since L30/S30 never share
            # timestamps and resample().median() handles
            # aggregation.
            data = xr.DataArray(
                da.concatenate(
                    [
                        p.transpose(
                            'time', 'band', 'y', 'x'
                        ).data
                        for p in parts
                    ],
                    axis=0,
                ),
                dims=('time', 'band', 'y', 'x'),
                coords={
                    'time': np.concatenate(
                        [p.time.values for p in parts]
                    ),
                    'band': parts[0].band.values,
                    'y': parts[0].y.values,
                    'x': parts[0].x.values,
                },
                attrs=parts[0].attrs,
            ).sortby('time')

        df = pd.concat(dfs, ignore_index=True)
        attrs = data.attrs.copy()

        # Resample to the requested frequency using median
        print("Computing composite...")
        composite = (
            data.resample(time=frequency)
            .median(dim='time', skipna=True)
            .assign_attrs(**attrs)
        )

        # Drop all-NaN time slices
        valid_times = ~composite.isnull().all(
            dim=['band', 'y', 'x']
        )
        composite = composite.sel(time=valid_times)

        if compute and hasattr(composite, 'compute'):
            composite = composite.compute()

        return composite, df

    else:
        result = open_stac(
            stac_catalog=stac_catalog,
            collection=collection,
            bounds=bounds,
            proj_bounds=proj_bounds,
            start_date=start_date,
            end_date=end_date,
            cloud_cover_perc=cloud_cover_perc,
            bands=bands,
            chunksize=chunksize,
            mask_items=mask_items,
            bounds_query=bounds_query,
            mask_data=True,
            epsg=epsg,
            resolution=resolution,
            resampling=resampling,
            nodata_fill=nodata_fill,
            max_items=max_items,
            compute=False,
        )

        if result is None or result[0] is None:
            return None, None

        data, df = result
        attrs = data.attrs.copy()

    # Resample to the requested frequency using median
    composite = (
        data.resample(time=frequency)
        .median(dim='time', skipna=True)
        .assign_attrs(**attrs)
    )

    # Drop all-NaN time slices (periods with no valid data)
    valid_times = ~composite.isnull().all(
        dim=['band', 'y', 'x']
    )
    composite = composite.sel(time=valid_times)

    if compute:
        import dask
        from tqdm.dask import TqdmCallback

        with dask.config.set(
            scheduler='threads', num_workers=num_workers
        ):
            with TqdmCallback(
                desc=f"Downloading & compositing {collection}"
            ):
                composite = composite.compute()

    return composite, df
