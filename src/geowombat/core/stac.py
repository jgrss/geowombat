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
    warnings.warn(
        "Install geowombat with 'pip install .[stac]' to use the STAC API."
    )
    warnings.warn(e)

try:
    from pydantic.errors import PydanticImportError
except ImportError:
    PydanticImportError = ImportError

try:
    import planetary_computer as pc
except (ImportError, PydanticImportError) as e:
    warnings.warn(
        'The planetary-computer package did not import correctly. Use of the microsoft collection may be limited.'
    )
    warnings.warn(e)


class STACNames(enum.Enum):
    """STAC names."""

    element84_v0 = 'element84_v0'
    element84_v1 = 'element84_v1'
    microsoft_v1 = 'microsoft_v1'


class STACCollections(enum.Enum):
    cop_dem_glo_30 = 'cop_dem_glo_30'
    landsat_c2_l1 = 'landsat_c2_l1'
    landsat_c2_l2 = 'landsat_c2_l2'
    sentinel_s2_l2a = 'sentinel_s2_l2a'
    sentinel_s2_l2a_cogs = 'sentinel_s2_l2a_cogs'
    sentinel_s2_l1c = 'sentinel_s2_l1c'
    sentinel_s1_l1c = 'sentinel_s1_l1c'
    sentinel_3_lst = 'sentinel_3_lst'
    landsat_l8_c2_l2 = 'landsat_l8_c2_l2'
    usda_cdl = 'usda_cdl'
    io_lulc = 'io_lulc'


STAC_CATALOGS = {
    STACNames.element84_v0: 'https://earth-search.aws.element84.com/v0',
    STACNames.element84_v1: 'https://earth-search.aws.element84.com/v1',
    # STACNames.google: 'https://earthengine.openeo.org/v1.0',
    STACNames.microsoft_v1: 'https://planetarycomputer.microsoft.com/api/stac/v1',
}


STAC_SCALING = {
    STACCollections.landsat_c2_l2: {
        # https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2
        STACNames.microsoft_v1: {
            'gain': 0.0000275,
            'offset': -0.2,
            'nodata': 0,
        },
    }
}

STAC_COLLECTIONS = {
    # Copernicus DEM GLO-30
    STACCollections.cop_dem_glo_30: {
        STACNames.element84_v1: 'cop-dem-glo-30',
        STACNames.microsoft_v1: 'cop-dem-glo-30',
    },
    # All Landsat, Collection 2, Level 1
    STACCollections.landsat_c2_l1: {
        STACNames.microsoft_v1: 'landsat-c2-l1',
    },
    # All Landsat, Collection 2, Level 2 (surface reflectance)
    STACCollections.landsat_c2_l2: {
        STACNames.element84_v1: 'landsat-c2-l2',
        # STACNames.google: [
        #     'LC09/C02/T1_L2',
        #     'LC08/C02/T1_L2',
        #     'LE07/C02/T1_L2',
        #     'LT05/C02/T1_L2',
        # ],
        STACNames.microsoft_v1: 'landsat-c2-l2',
    },
    # Sentinel-2, Level 2A (surface reflectance missing cirrus band)
    STACCollections.sentinel_s2_l2a_cogs: {
        STACNames.element84_v0: 'sentinel-s2-l2a-cogs',
    },
    STACCollections.sentinel_s2_l2a: {
        STACNames.element84_v1: 'sentinel-2-l2a',
        # STACNames.google: 'COPERNICUS/S2_SR',
        STACNames.microsoft_v1: 'sentinel-2-l2a',
    },
    # Sentinel-2, Level 1C (top of atmosphere with all 13 bands available)
    STACCollections.sentinel_s2_l1c: {
        STACNames.element84_v1: 'sentinel-2-l1c'
    },
    # Sentinel-1, Level 1C Ground Range Detected (GRD)
    STACCollections.sentinel_s1_l1c: {
        STACNames.element84_v1: 'sentinel-1-grd',
        STACNames.microsoft_v1: 'sentinel-1-grd',
    },
    STACCollections.sentinel_3_lst: {
        STACNames.microsoft_v1: 'sentinel-3-slstr-lst-l2-netcdf',
    },
    # Landsat 8, Collection 2, Tier 1 (Level 2 (surface reflectance))
    STACCollections.landsat_l8_c2_l2: {
        # STACNames.google: 'LC08_C02_T1_L2',
        STACNames.microsoft_v1: 'landsat-8-c2-l2',
    },
    # USDA CDL
    STACCollections.usda_cdl: {
        STACNames.microsoft_v1: 'usda-cdl',
    },
    # Esri 10 m land cover
    STACCollections.io_lulc: {
        STACNames.microsoft_v1: 'io-lulc',
    },
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
    stac_catalog: str = 'microsoft_v1',
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
        >>>     bands=['B04', 'B03', 'B02'],
        >>>     resampling=Resampling.cubic,
        >>>     epsg=int(data_l.epsg.values),
        >>>     extra_assets=['metadata']
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
        stac_catalog_url = STAC_CATALOGS[STACNames(stac_catalog)]
        # Open the STAC catalog
        catalog = _Client.open(stac_catalog_url)
    except ValueError as e:
        raise NameError(
            f'The STAC catalog {stac_catalog} is not supported ({e}).'
        )

    try:
        collection_dict = STAC_COLLECTIONS[STACCollections(collection)]
    except ValueError as e:
        raise NameError(
            f'The STAC collection {collection} is not supported ({e}).'
        )

    try:
        catalog_collections = [collection_dict[STACNames(stac_catalog)]]
    except KeyError as e:
        raise NameError(
            f'The STAC catalog {stac_catalog} does not have a collection {collection} ({e}).'
        )
    # asset = catalog.get_collection(catalog_collections[0]).assets['geoparquet-items']

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
        if STACNames(stac_catalog) is STACNames.microsoft_v1:
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

    return None, None
