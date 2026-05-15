"""Training data builders for object detection in geowombat.

Provides utilities to convert GeoDataFrame labels into YOLO-format
training datasets tiled from raster sources opened via ``gw.open()``.

Requires: ``pip install geowombat[detect]``

Example
-------
>>> import geowombat as gw
>>> import geopandas as gpd
>>> from geowombat.ml.detection_data import (
...     boxes_from_polygons, build_yolo_dataset,
... )
>>> labels = gpd.read_file('trees.gpkg')  # polygons
>>> boxes = boxes_from_polygons(labels)
>>> with gw.open('aerial.tif') as src:
...     build_yolo_dataset(
...         src, boxes, class_col='species',
...         out_dir='./yolo_ds', tile_size=640,
...     )
"""

import math
import random
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.affinity import rotate
from shapely.geometry import (
    MultiPolygon,
    Polygon,
    box as shapely_box,
)


def _require_pillow():
    try:
        from PIL import Image
        return Image
    except ImportError as e:
        raise ImportError(
            "Building YOLO datasets requires Pillow. "
            "Install with: pip install geowombat[detect]"
        ) from e


def _require_yaml():
    try:
        import yaml
        return yaml
    except ImportError as e:
        raise ImportError(
            "Writing YOLO data.yaml requires PyYAML. "
            "Install with: pip install geowombat[detect]"
        ) from e


def _min_rotated_rectangle(geom):
    """Return shapely's minimum rotated rectangle for a geometry."""
    if hasattr(geom, 'minimum_rotated_rectangle'):
        return geom.minimum_rotated_rectangle
    return geom.envelope


def boxes_from_polygons(gdf, oriented=False):
    """Convert polygon geometries to bounding-box geometries.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input labels. Geometries may be ``Polygon`` or ``MultiPolygon``.
        Pass-through for any geometry that is already a box.
    oriented : bool
        If True, return minimum rotated rectangles (4-corner polygons).
        If False, return axis-aligned envelopes (default).

    Returns
    -------
    geopandas.GeoDataFrame
        Same columns as input with geometries replaced by boxes. A new
        column ``_box_kind`` is added: ``'aabb'`` or ``'obb'``.
    """
    out = gdf.copy()

    if oriented:
        out['geometry'] = out.geometry.apply(_min_rotated_rectangle)
        out['_box_kind'] = 'obb'
    else:
        out['geometry'] = out.geometry.apply(lambda g: g.envelope)
        out['_box_kind'] = 'aabb'

    return out


def _normalize_class_column(labels, class_col):
    """Build a stable integer class id from a string/int column.

    Returns ``(labels_with_int_col, class_names)`` where ``class_names``
    is an ordered list whose index corresponds to the int id.
    """
    classes = sorted(labels[class_col].dropna().unique().tolist())
    name_to_id = {name: i for i, name in enumerate(classes)}
    labels = labels.copy()
    labels['_class_id'] = labels[class_col].map(name_to_id).astype('Int64')
    return labels, classes


def _tile_grid(src, tile_size, overlap):
    """Generate (row, col, y0, x0, y1, x1) for image tiles.

    Coordinates are in pixel space. The last tile in each direction is
    shifted backwards so it never exceeds the image bounds — i.e. tiles
    are always full size, never padded.
    """
    h = src.gw.nrows
    w = src.gw.ncols
    step = max(1, int(round(tile_size * (1 - overlap))))

    ys = list(range(0, max(1, h - tile_size + 1), step))
    if not ys or ys[-1] + tile_size < h:
        ys.append(max(0, h - tile_size))
    xs = list(range(0, max(1, w - tile_size + 1), step))
    if not xs or xs[-1] + tile_size < w:
        xs.append(max(0, w - tile_size))

    for r, y0 in enumerate(ys):
        for c, x0 in enumerate(xs):
            y1 = min(h, y0 + tile_size)
            x1 = min(w, x0 + tile_size)
            yield r, c, y0, x0, y1, x1


def _tile_bounds_crs(src, y0, x0, y1, x1):
    """Convert pixel-space tile window to CRS bounds (xmin,ymin,xmax,ymax)."""
    affine = src.gw.affine
    xmin, ymax = affine * (x0, y0)
    xmax, ymin = affine * (x1, y1)
    return xmin, ymin, xmax, ymax


def _crs_to_pixel(affine, x, y):
    """Inverse-affine: CRS coords -> pixel coords (col, row) as floats."""
    inv = ~affine
    return inv * (x, y)


def _scale_to_uint8(arr, scale):
    """Scale a (bands, h, w) numeric array to uint8 RGB-compatible.

    Parameters
    ----------
    arr : numpy.ndarray
        Image data with shape ``(bands, h, w)``.
    scale : tuple of (lo, hi) or None
        Min/max range to linearly map to 0..255. If None, use per-tile
        percentile stretch (2-98) as a fallback.
    """
    arr = arr.astype(np.float32)
    if scale is None:
        lo = np.percentile(arr, 2)
        hi = np.percentile(arr, 98)
    else:
        lo, hi = scale
    if hi <= lo:
        hi = lo + 1.0
    out = (arr - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    return (out * 255).astype(np.uint8)


def _prepare_rgb_tile(data_block, band_indices, scale):
    """Return an (H, W, 3) uint8 array from a (bands, H, W) block."""
    if band_indices is None:
        if data_block.shape[0] >= 3:
            band_indices = [0, 1, 2]
        else:
            band_indices = [0] * 3  # grayscale → broadcast to RGB

    selected = data_block[list(band_indices), :, :]

    if selected.dtype == np.uint8 and scale is None:
        rgb = selected
    else:
        rgb = _scale_to_uint8(selected, scale)

    if rgb.shape[0] == 1:
        rgb = np.repeat(rgb, 3, axis=0)
    elif rgb.shape[0] == 2:
        # pad missing channel with zeros to keep RGB-compatible shape
        pad = np.zeros_like(rgb[:1])
        rgb = np.concatenate([rgb, pad], axis=0)

    return np.transpose(rgb, (1, 2, 0))  # (H, W, 3)


def _polygon_to_yolo_aabb(poly, tile_xmin, tile_ymax, cellx, celly,
                          tile_size):
    """Convert a polygon (CRS coords) to YOLO axis-aligned label.

    Returns ``(cx_n, cy_n, w_n, h_n)`` normalized to ``[0, 1]`` against
    ``tile_size``, or None if the polygon is degenerate after clipping.
    """
    minx, miny, maxx, maxy = poly.bounds
    # CRS → pixel (relative to tile origin)
    px_left = (minx - tile_xmin) / cellx
    px_right = (maxx - tile_xmin) / cellx
    py_top = (tile_ymax - maxy) / celly
    py_bot = (tile_ymax - miny) / celly

    px_left = max(0.0, min(px_left, tile_size))
    px_right = max(0.0, min(px_right, tile_size))
    py_top = max(0.0, min(py_top, tile_size))
    py_bot = max(0.0, min(py_bot, tile_size))

    w = px_right - px_left
    h = py_bot - py_top
    if w <= 0 or h <= 0:
        return None

    cx = px_left + w / 2.0
    cy = py_top + h / 2.0

    return (cx / tile_size, cy / tile_size,
            w / tile_size, h / tile_size, w, h)


def _polygon_to_yolo_obb(poly, tile_xmin, tile_ymax, cellx, celly,
                         tile_size):
    """Convert a polygon (CRS coords) to YOLO OBB label.

    YOLO OBB labels are 4 corner pairs normalized to image size. We use
    the minimum rotated rectangle of the polygon, intersected with the
    tile envelope. Returns the 8 normalized coords and the on-tile
    pixel width/height of the bounding envelope, or None if degenerate.
    """
    tile_poly = shapely_box(
        tile_xmin, tile_ymax - tile_size * celly,
        tile_xmin + tile_size * cellx, tile_ymax,
    )
    inter = poly.intersection(tile_poly)
    if inter.is_empty:
        return None
    rect = _min_rotated_rectangle(inter)
    if rect.is_empty or rect.geom_type != 'Polygon':
        return None

    coords = list(rect.exterior.coords)[:4]
    if len(coords) < 4:
        return None

    px_coords = []
    for x, y in coords:
        cx = (x - tile_xmin) / cellx
        cy = (tile_ymax - y) / celly
        px_coords.append((cx, cy))

    minx = min(c[0] for c in px_coords)
    maxx = max(c[0] for c in px_coords)
    miny = min(c[1] for c in px_coords)
    maxy = max(c[1] for c in px_coords)
    w = maxx - minx
    h = maxy - miny
    if w <= 0 or h <= 0:
        return None

    normed = []
    for cx, cy in px_coords:
        normed.append(cx / tile_size)
        normed.append(cy / tile_size)

    return tuple(normed) + (w, h)


def build_yolo_dataset(
    src,
    labels,
    class_col,
    out_dir,
    tile_size=640,
    overlap=0.1,
    val_split=0.2,
    min_box_pixels=8,
    background_ratio=0.0,
    band_indices=None,
    scale=None,
    oriented=False,
    image_format='jpg',
    seed=42,
    class_names=None,
):
    """Write a YOLO-format training dataset from a raster + label GDF.

    Parameters
    ----------
    src : xarray.DataArray
        Raster opened with ``gw.open()``.
    labels : geopandas.GeoDataFrame, str, or Path
        Vector labels. Polygons are converted to bounding boxes; existing
        box geometries are used as-is.
    class_col : str
        Column in ``labels`` holding class name/id.
    out_dir : str or Path
        Output directory. Will be created if missing. The Ultralytics
        layout ``images/{train,val}`` + ``labels/{train,val}`` is written
        plus a ``data.yaml`` at the root.
    tile_size : int
        Square tile edge in pixels. Default 640.
    overlap : float
        Fractional overlap between adjacent tiles (0..0.9). Default 0.1.
    val_split : float
        Fraction of tiles assigned to the validation split. Default 0.2.
    min_box_pixels : int
        Minimum width or height (in pixels) for a box to be kept after
        tile clipping. Default 8.
    background_ratio : float
        Fraction (0..1) of empty tiles to retain. Default 0 (drop all).
    band_indices : list of int, optional
        Three band indices (0-based) for the R, G, B channels. Required
        for non-3-band rasters or non-uint8 data unless the source is
        already 3-band uint8.
    scale : tuple of (lo, hi), optional
        Linear stretch applied before writing. If None and dtype is
        uint8, no stretch is applied; otherwise a per-tile 2-98 pct
        stretch is used.
    oriented : bool
        If True, write OBB labels (8 corner coords). Default False.
    image_format : {'jpg', 'png'}
        Tile image format. Default 'jpg'.
    seed : int
        RNG seed for train/val split. Default 42.
    class_names : list of str, optional
        Override class ordering. If None, classes are taken from
        ``labels[class_col]`` sorted alphabetically.

    Returns
    -------
    dict
        Summary with keys ``out_dir``, ``classes``, ``n_train``,
        ``n_val``, ``n_boxes``.
    """
    Image = _require_pillow()
    yaml = _require_yaml()

    if isinstance(labels, (str, Path)):
        labels = gpd.read_file(labels)
    if not isinstance(labels, gpd.GeoDataFrame):
        raise TypeError(
            f"labels must be a GeoDataFrame, got {type(labels).__name__}"
        )

    src_crs = src.gw.crs_to_pyproj
    if labels.crs is None:
        raise ValueError("labels GeoDataFrame has no CRS set.")
    if labels.crs.to_epsg() != src_crs.to_epsg():
        labels = labels.to_crs(src_crs)

    # Spatial filter: drop labels fully outside the raster footprint
    raster_geom = shapely_box(
        src.gw.left, src.gw.bottom, src.gw.right, src.gw.top,
    )
    labels = labels[labels.intersects(raster_geom)].copy()
    if labels.empty:
        raise ValueError(
            "No labels intersect the raster footprint. "
            "Check CRS or geometry coverage."
        )

    if class_names is None:
        labels, classes = _normalize_class_column(labels, class_col)
    else:
        classes = list(class_names)
        name_to_id = {n: i for i, n in enumerate(classes)}
        labels = labels.copy()
        labels['_class_id'] = labels[class_col].map(name_to_id)
        missing = labels['_class_id'].isna().sum()
        if missing:
            warnings.warn(
                f"{missing} label(s) had class values not in class_names; "
                "they will be dropped."
            )
            labels = labels.dropna(subset=['_class_id'])
        labels['_class_id'] = labels['_class_id'].astype(int)

    if oriented:
        labels = boxes_from_polygons(labels, oriented=True)
    # For axis-aligned, leave geometries as-is — bounds are taken
    # per-tile after intersection.

    out_dir = Path(out_dir)
    for split in ('train', 'val'):
        (out_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    sindex = labels.sindex

    cellx = float(src.gw.cellx)
    celly = float(src.gw.celly)
    affine = src.gw.affine

    n_train = 0
    n_val = 0
    n_boxes = 0
    empty_kept = 0
    empty_skipped = 0

    for r, c, y0, x0, y1, x1 in _tile_grid(src, tile_size, overlap):
        tile_xmin, tile_ymin, tile_xmax, tile_ymax = _tile_bounds_crs(
            src, y0, x0, y1, x1,
        )
        tile_geom = shapely_box(
            tile_xmin, tile_ymin, tile_xmax, tile_ymax,
        )

        cand_idx = list(sindex.intersection(tile_geom.bounds))
        cand = labels.iloc[cand_idx]
        cand = cand[cand.intersects(tile_geom)]

        # Build YOLO label strings for this tile
        lines = []
        for _, row in cand.iterrows():
            cls_id = int(row['_class_id'])
            geom = row.geometry
            if isinstance(geom, MultiPolygon):
                parts = list(geom.geoms)
            else:
                parts = [geom]

            for part in parts:
                clipped = part.intersection(tile_geom)
                if clipped.is_empty:
                    continue
                if not isinstance(clipped, (Polygon, MultiPolygon)):
                    continue
                if isinstance(clipped, MultiPolygon):
                    sub_parts = list(clipped.geoms)
                else:
                    sub_parts = [clipped]

                for sub in sub_parts:
                    if oriented:
                        res = _polygon_to_yolo_obb(
                            sub, tile_xmin, tile_ymax, cellx, celly,
                            tile_size,
                        )
                    else:
                        res = _polygon_to_yolo_aabb(
                            sub, tile_xmin, tile_ymax, cellx, celly,
                            tile_size,
                        )
                    if res is None:
                        continue
                    w_px, h_px = res[-2], res[-1]
                    if w_px < min_box_pixels or h_px < min_box_pixels:
                        continue
                    coords = res[:-2]
                    coord_str = ' '.join(f'{v:.6f}' for v in coords)
                    lines.append(f'{cls_id} {coord_str}')
                    n_boxes += 1

        is_background = not lines
        if is_background:
            if rng.random() > background_ratio:
                empty_skipped += 1
                continue
            empty_kept += 1

        # Read the tile from disk via dask/xarray
        block = src.isel(
            y=slice(y0, y1), x=slice(x0, x1),
        ).values  # (bands, h, w)
        if block.ndim == 4:  # (time, band, y, x)
            block = block[0]

        # Pad if tile is at image edge and short
        h_block, w_block = block.shape[1], block.shape[2]
        if h_block < tile_size or w_block < tile_size:
            padded = np.zeros(
                (block.shape[0], tile_size, tile_size),
                dtype=block.dtype,
            )
            padded[:, :h_block, :w_block] = block
            block = padded

        rgb = _prepare_rgb_tile(block, band_indices, scale)

        split = 'val' if rng.random() < val_split else 'train'
        stem = f'tile_r{r:04d}_c{c:04d}'
        img_path = out_dir / 'images' / split / f'{stem}.{image_format}'
        lbl_path = out_dir / 'labels' / split / f'{stem}.txt'

        Image.fromarray(rgb).save(img_path)
        lbl_path.write_text('\n'.join(lines) + ('\n' if lines else ''))

        if split == 'train':
            n_train += 1
        else:
            n_val += 1

    # data.yaml — Ultralytics format
    data_yaml = {
        'path': str(out_dir.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: n for i, n in enumerate(classes)},
    }
    if oriented:
        data_yaml['task'] = 'obb'
    with open(out_dir / 'data.yaml', 'w') as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    return {
        'out_dir': str(out_dir),
        'classes': classes,
        'n_train': n_train,
        'n_val': n_val,
        'n_boxes': n_boxes,
        'empty_kept': empty_kept,
        'empty_skipped': empty_skipped,
    }
