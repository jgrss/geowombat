import concurrent.futures
import enum
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
from ..radiometry import QABits as _QABits

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
    ESA_WORLDCOVER = 'esa-worldcover'


STAC_CATALOGS = {
    STACNames.ELEMENT84_V0: 'https://earth-search.aws.element84.com/v0',
    STACNames.ELEMENT84_V1: 'https://earth-search.aws.element84.com/v1',
    # STACNames.google: 'https://earthengine.openeo.org/v1.0',
    STACNames.MICROSOFT_V1: 'https://planetarycomputer.microsoft.com/api/stac/v1',
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
    STACCollections.HLS: {
        # https://planetarycomputer.microsoft.com/dataset/hls
        STACNames.MICROSOFT_V1: {
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
        STACCollectionURLNames.HLS,
        STACCollectionURLNames.ESA_WORLDCOVER,
    ),
}


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
) -> xr.DataArray:
    """Opens a collection from a spatio-temporal asset catalog (STAC).

    Args:
        stac_catalog (str): Choices are ['element84_v0', 'element84_v1, 'google', 'microsoft_v1'].
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
                    hls
                    esa_worldcover

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
        bounds_query (Optional[str]): A query to select bounds from the ``geopandas.GeoDataFrame``.
        mask_data (Optional[bool]): Whether to mask the data. Only relevant if ``mask_items=True``.
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

        data = stackstac.stack(
            items,
            bounds=proj_bounds,
            bounds_latlon=None if proj_bounds is not None else bounds,
            assets=bands,
            chunksize=chunksize,
            epsg=epsg,
            resolution=resolution,
            resampling=resampling,
        )
        data = data.assign_attrs(
            res=(data.resolution, data.resolution), collection=collection
        )
        attrs = data.attrs.copy()

        if mask_data:
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
            # Source: https://stackstac.readthedocs.io/en/v0.3.0/examples/gif.html
            bitmask = 0
            for field in mask_bitfields:
                bitmask |= 1 << field
            # TODO: get qa_pixel name for different sensors
            qa = data.sel(band='qa_pixel').astype('uint16')
            mask = qa & bitmask
            data = data.sel(
                band=[band for band in bands if band != 'qa_pixel']
            ).where(mask == 0)

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

        return data, df

    warnings.warn("No asset items were found.")

    return None, None
