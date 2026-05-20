"""Shared label intake and band-resolution helpers for ML modules.

Centralizes the path-or-GeoDataFrame label intake, CRS coercion, raster-
footprint filtering, and class-name encoding that both classifiers and
detectors need. Also resolves an RGB band-index triplet from the active
``gw.config(sensor=...)`` so callers don't have to repeat
``band_indices=[2, 1, 0]`` everywhere.
"""

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np


_RGB_ALIASES = {
    'red': ('red', 'r'),
    'green': ('green', 'g'),
    'blue': ('blue', 'b'),
}


def prepare_label_gdf(src, labels, class_col, class_names=None):
    """Coerce labels to a GeoDataFrame aligned with the raster.

    - Accepts a path, GeoDataFrame, or other readable vector source.
    - Reprojects to ``src`` CRS if needed.
    - Spatially filters to the raster footprint using
      ``src.gw.geodataframe``.
    - Adds an integer ``_class_id`` column. If ``class_names`` is given,
      labels with unknown classes are dropped (with a warning).

    Parameters
    ----------
    src : xarray.DataArray
        Raster opened with ``gw.open()``.
    labels : geopandas.GeoDataFrame, str, or Path
        Vector labels.
    class_col : str
        Column in ``labels`` holding class name/id.
    class_names : list of str, optional
        Override class ordering. If None, classes are taken from
        ``labels[class_col]`` sorted.

    Returns
    -------
    (geopandas.GeoDataFrame, list[str])
        Labels with ``_class_id`` column, and the ordered class names.
    """
    if isinstance(labels, (str, Path)):
        labels = gpd.read_file(labels)
    if not isinstance(labels, gpd.GeoDataFrame):
        raise TypeError(
            f"labels must be a GeoDataFrame or readable path, "
            f"got {type(labels).__name__}"
        )

    src_crs = src.gw.crs_to_pyproj
    if labels.crs is None:
        raise ValueError("labels GeoDataFrame has no CRS set.")
    if labels.crs.to_epsg() != src_crs.to_epsg():
        labels = labels.to_crs(src_crs)

    # Spatial filter against the raster footprint
    footprint = src.gw.geodataframe
    if not footprint.empty:
        try:
            raster_geom = footprint.union_all()
        except AttributeError:
            raster_geom = footprint.unary_union
        labels = labels[labels.intersects(raster_geom)].copy()
    else:
        labels = labels.copy()
    if labels.empty:
        raise ValueError(
            "No labels intersect the raster footprint. "
            "Check CRS or geometry coverage."
        )

    if class_names is None:
        classes = sorted(labels[class_col].dropna().unique().tolist())
        name_to_id = {name: i for i, name in enumerate(classes)}
        labels['_class_id'] = labels[class_col].map(name_to_id).astype(int)
    else:
        classes = list(class_names)
        name_to_id = {n: i for i, n in enumerate(classes)}
        labels['_class_id'] = labels[class_col].map(name_to_id)
        missing = labels['_class_id'].isna().sum()
        if missing:
            warnings.warn(
                f"{missing} label(s) had class values not in class_names; "
                "they will be dropped."
            )
            labels = labels.dropna(subset=['_class_id'])
        labels['_class_id'] = labels['_class_id'].astype(int)

    return labels, classes


def resolve_band_indices(src, band_indices=None):
    """Pick three (R, G, B) band indices for inference / dataset export.

    Resolution order:

    1. If ``band_indices`` is explicitly provided, use it.
    2. If the DataArray has named bands (e.g. from
       ``gw.config.update(sensor='bgr')``), look up red/green/blue by
       name (case-insensitive).
    3. Fall back to ``[0, 1, 2]`` for 3+ band rasters, or ``[0, 0, 0]``
       to broadcast a single band to grey-RGB.

    Parameters
    ----------
    src : xarray.DataArray
        Raster opened with ``gw.open()``.
    band_indices : list of int, optional
        Explicit override.

    Returns
    -------
    list of int
        Three 0-based band indices in (R, G, B) order.
    """
    if band_indices is not None:
        return list(band_indices)

    band_vals = None
    if 'band' in src.coords:
        try:
            band_vals = [str(b).lower() for b in src.band.values.tolist()]
        except Exception:
            band_vals = None

    if band_vals:
        lookup = {n: i for i, n in enumerate(band_vals)}
        rgb = []
        for channel in ('red', 'green', 'blue'):
            idx = None
            for alias in _RGB_ALIASES[channel]:
                if alias in lookup:
                    idx = lookup[alias]
                    break
            if idx is None:
                rgb = None
                break
            rgb.append(idx)
        if rgb is not None:
            return rgb

    if src.gw.nbands >= 3:
        return [0, 1, 2]
    return [0, 0, 0]
