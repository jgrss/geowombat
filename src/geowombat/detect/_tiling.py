"""Shared tiling helpers for geowombat ML modules.

Wraps ``src.gw.windows()`` with the stride-overlap semantics needed by
object detection (training-data tiling and inference). Detection-time
tiles must overlap so that objects on tile seams can still be matched;
``.gw.windows()`` is non-overlapping by design, so we add the stride
layer here rather than changing the accessor.
"""

import typing as T

from rasterio.windows import Window


def overlapped_windows(
    src,
    tile_size: int,
    overlap: float = 0.0,
) -> T.Generator[T.Tuple[int, int, Window], None, None]:
    """Yield ``rasterio.windows.Window`` tiles with fractional overlap.

    The last tile in each direction is shifted backwards when possible so
    it does not exceed the image bounds. The returned windows may still
    be smaller than ``tile_size`` at image edges, or when the image is
    smaller than ``tile_size`` in either dimension.

    Parameters
    ----------
    src : xarray.DataArray
        Raster opened with ``gw.open()``.
    tile_size : int
        Square tile edge in pixels.
    overlap : float
        Fractional overlap between adjacent tiles in ``[0, 0.9]``.

    Yields
    ------
    (row_idx, col_idx, rasterio.windows.Window)
        Grid indices plus the window itself.
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
            height = min(tile_size, h - y0)
            width = min(tile_size, w - x0)
            yield r, c, Window(
                col_off=x0, row_off=y0,
                width=width, height=height,
            )
