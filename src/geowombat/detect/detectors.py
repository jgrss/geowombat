"""Object detectors for geowombat.

Wraps Ultralytics YOLO and TorchGeo detection models behind a single
geospatial-aware API: tiled windowed inference with cross-tile NMS and
georeferenced ``GeoDataFrame`` outputs.

Requires: ``pip install geowombat[detect]`` (YOLO/TorchGeo)
          ``pip install geowombat[sam]`` (SAMRefiner)

Example
-------
>>> import geowombat as gw
>>> from geowombat.detect import YOLODetector
>>> det = YOLODetector(weights='yolov8n.pt')
>>> with gw.open('aerial.tif') as src:
...     boxes = det.predict(src, tile_size=640, conf=0.25,
...                         band_indices=[2, 1, 0], scale=(0, 3000))
>>> boxes.to_file('detections.gpkg')
"""

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, box as shapely_box

from ..ml._labels import resolve_band_indices
from ._tiling import overlapped_windows
from .data import _prepare_rgb_tile


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pixel_box_to_polygon(affine, tile_x0, tile_y0, x1, y1, x2, y2):
    """Convert pixel-space corners on a tile to a CRS polygon."""
    ul_x, ul_y = affine * (tile_x0 + x1, tile_y0 + y1)
    lr_x, lr_y = affine * (tile_x0 + x2, tile_y0 + y2)
    xmin, xmax = sorted((ul_x, lr_x))
    ymin, ymax = sorted((ul_y, lr_y))
    return shapely_box(xmin, ymin, xmax, ymax)


def _pixel_quad_to_polygon(affine, tile_x0, tile_y0, quad_xy):
    """Convert pixel-space (4,2) corner array on a tile to CRS polygon."""
    coords = []
    for x, y in quad_xy:
        cx, cy = affine * (tile_x0 + x, tile_y0 + y)
        coords.append((cx, cy))
    return Polygon(coords)


def _iou_geom(a, b):
    """IoU between two shapely geometries (assumed in same CRS)."""
    if a.is_empty or b.is_empty:
        return 0.0
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _nms_geodataframe(gdf, iou_threshold=0.5):
    """Greedy NMS across a GeoDataFrame, class-aware.

    Expects columns: ``geometry``, ``score``, ``class_id``.
    """
    if gdf.empty:
        return gdf
    keep_idx = []
    for cls_id, sub in gdf.groupby('class_id', sort=False):
        sub = sub.sort_values('score', ascending=False)
        kept_geoms = []
        for idx, row in sub.iterrows():
            geom = row.geometry
            keep = True
            for kg in kept_geoms:
                if _iou_geom(geom, kg) > iou_threshold:
                    keep = False
                    break
            if keep:
                keep_idx.append(idx)
                kept_geoms.append(geom)
    out = gdf.loc[keep_idx].sort_values(
        'score', ascending=False,
    ).reset_index(drop=True)
    return out


def _resolve_device(device):
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "Detectors require PyTorch. Install with "
            "`pip install geowombat[detect]`."
        ) from e
    if device != 'auto':
        return device
    if torch.cuda.is_available():
        return 'cuda'
    # CPU fallback — but check if there's an NVIDIA GPU on the system
    # that this torch build just can't see. That's almost always a
    # CPU-only torch install; tell the user how to fix it.
    import shutil
    import subprocess
    if shutil.which('nvidia-smi') is not None:
        try:
            r = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=2,
            )
        except (subprocess.SubprocessError, OSError):
            r = None
        if r is not None and r.returncode == 0 and r.stdout.strip():
            gpu_name = r.stdout.strip().splitlines()[0]
            warnings.warn(
                f"geowombat: NVIDIA GPU detected ({gpu_name}) but PyTorch "
                f"can't use it — falling back to CPU. You likely have a "
                f"CPU-only torch build. Install a CUDA build, e.g.:\n"
                f"  pip install torch --index-url "
                f"https://download.pytorch.org/whl/cu124\n"
                f"(see https://pytorch.org/get-started/locally/ for your "
                f"CUDA version).",
                UserWarning, stacklevel=2,
            )
    return 'cpu'


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class GeoWombatDetector:
    """Base class for geowombat object detectors.

    Subclasses must implement ``_detect_tile(rgb_array)`` returning a
    list of ``(x1, y1, x2, y2, score, class_id)`` tuples (axis-aligned)
    or ``(quad_xy_array, score, class_id)`` for oriented boxes. They
    should also set ``class_names`` (list[str]) and ``oriented`` (bool).

    Subclasses can optionally override ``_detect_batch`` to run a list
    of tiles through the underlying model in one call (much faster on
    GPU, modestly faster on CPU). The default falls back to a per-tile
    loop.
    """
    class_names = None
    oriented = False

    def _detect_tile(self, rgb_array):
        raise NotImplementedError

    def _detect_batch(self, rgb_list, conf=0.25, max_det=None):
        """Default: per-tile loop. Override for true batched inference."""
        return [
            self._detect_tile(rgb, conf=conf, max_det=max_det)
            for rgb in rgb_list
        ]

    def predict(
        self,
        src,
        tile_size=640,
        overlap=0.2,
        conf=0.25,
        band_indices=None,
        scale=None,
        nms_iou=0.5,
        max_det=None,
        batch_size=4,
        progress=False,
    ):
        """Run tiled, georeferenced inference over a raster.

        Parameters
        ----------
        src : xarray.DataArray
            Raster opened with ``gw.open()``.
        tile_size : int
            Square tile edge in pixels. Default 640.
        overlap : float
            Fractional overlap between adjacent tiles. Default 0.2.
        conf : float
            Confidence threshold for tile-level detections. Default 0.25.
        band_indices : list of int, optional
            Three band indices for R, G, B. See ``build_yolo_dataset``.
        scale : tuple of (lo, hi), optional
            Linear stretch before inference.
        nms_iou : float
            IoU threshold for cross-tile NMS. Default 0.5.
        max_det : int, optional
            Cap on detections per tile (passed to the underlying model
            when supported). Default unlimited.
        batch_size : int
            Number of tiles sent through the model per inference call.
            Larger batches keep the GPU busy and reduce per-tile Python
            overhead. On CPU, modest batching also helps. Default 4.
        progress : bool
            Show a tqdm progress bar. Default False.

        Returns
        -------
        geopandas.GeoDataFrame
            One row per detection with columns ``geometry, class_id,
            class_name, score, tile_id`` in the source CRS.
        """
        affine = src.gw.affine
        crs = src.gw.crs_to_pyproj
        band_indices = resolve_band_indices(src, band_indices)

        tiles = list(overlapped_windows(src, tile_size, overlap))
        iterator = tiles
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(tiles, desc=self.__class__.__name__)
            except ImportError:
                pass

        records = []
        # Per-tile metadata buffered alongside the RGB array, so a flushed
        # batch knows where each tile's detections belong in the source CRS.
        batch_rgbs = []
        batch_meta = []  # list of (tile_id, x0, y0, w_block, h_block)

        def _flush(rgbs, meta):
            if not rgbs:
                return
            batch_dets = self._detect_batch(
                rgbs, conf=conf, max_det=max_det,
            )
            for (tile_id, x0, y0, w_block, h_block), dets in zip(
                meta, batch_dets,
            ):
                for det in dets:
                    if self.oriented:
                        quad, score, cls_id = det
                        # Reject detections whose center falls in padded
                        # region (outside the true image footprint).
                        cx = float(np.mean(quad[:, 0]))
                        cy = float(np.mean(quad[:, 1]))
                        if cx > w_block or cy > h_block:
                            continue
                        geom = _pixel_quad_to_polygon(
                            affine, x0, y0, quad,
                        )
                    else:
                        bx1, by1, bx2, by2, score, cls_id = det
                        cx = (bx1 + bx2) / 2.0
                        cy = (by1 + by2) / 2.0
                        if cx > w_block or cy > h_block:
                            continue
                        geom = _pixel_box_to_polygon(
                            affine, x0, y0, bx1, by1, bx2, by2,
                        )

                    if self.class_names and cls_id < len(self.class_names):
                        name = self.class_names[int(cls_id)]
                    else:
                        name = str(int(cls_id))

                    records.append({
                        'geometry': geom,
                        'class_id': int(cls_id),
                        'class_name': name,
                        'score': float(score),
                        'tile_id': int(tile_id),
                    })

        for tile_id, (r, c, win) in enumerate(iterator):
            y0 = win.row_off
            x0 = win.col_off
            y1 = y0 + win.height
            x1 = x0 + win.width
            block = src.isel(
                y=slice(y0, y1), x=slice(x0, x1),
            ).values
            if block.ndim == 4:
                block = block[0]

            h_block, w_block = block.shape[1], block.shape[2]
            pad_h = tile_size - h_block
            pad_w = tile_size - w_block
            if pad_h > 0 or pad_w > 0:
                padded = np.zeros(
                    (block.shape[0], tile_size, tile_size),
                    dtype=block.dtype,
                )
                padded[:, :h_block, :w_block] = block
                block = padded

            rgb = _prepare_rgb_tile(block, band_indices, scale)

            batch_rgbs.append(rgb)
            batch_meta.append((tile_id, x0, y0, w_block, h_block))

            if len(batch_rgbs) >= batch_size:
                _flush(batch_rgbs, batch_meta)
                batch_rgbs, batch_meta = [], []

        _flush(batch_rgbs, batch_meta)

        if not records:
            return gpd.GeoDataFrame(
                {
                    'geometry': gpd.GeoSeries([], crs=crs),
                    'class_id': [],
                    'class_name': [],
                    'score': [],
                    'tile_id': [],
                },
                geometry='geometry', crs=crs,
            )
        gdf = gpd.GeoDataFrame(records, geometry='geometry', crs=crs)
        gdf = _nms_geodataframe(gdf, iou_threshold=nms_iou)
        return gdf


# ---------------------------------------------------------------------------
# YOLO (Ultralytics)
# ---------------------------------------------------------------------------

class YOLODetector(GeoWombatDetector):
    """Ultralytics YOLO detector for georeferenced rasters.

    Supports axis-aligned (default) and oriented bounding boxes. The
    underlying model is any path accepted by ``ultralytics.YOLO`` —
    pretrained weights ('yolov8n.pt', 'yolo11n.pt', 'yolov8n-obb.pt')
    or a custom-trained checkpoint.

    NOTE: Ultralytics is licensed AGPL-3.0; ensure your use case is
    compatible before deploying.

    Parameters
    ----------
    weights : str or Path
        YOLO weights file. Default 'yolov8n.pt'.
    classes : list of str, optional
        Override class names. If None, names come from the model.
    oriented : bool
        Set to True when using an OBB weight (file ends in ``-obb.pt``).
        Auto-detected from the filename if not specified.
    device : str
        'cpu', 'cuda', or 'auto'. Default 'auto'.
    imgsz : int
        Inference size passed to YOLO. Default matches ``tile_size`` in
        ``predict()``.

    Example
    -------
    >>> det = YOLODetector(weights='yolov8n-obb.pt', oriented=True)
    >>> with gw.open('aerial.tif') as src:
    ...     gdf = det.predict(src, conf=0.3)
    """

    def __init__(self, weights='yolov8n.pt', classes=None, oriented=None,
                 device='auto', imgsz=None):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "YOLODetector requires ultralytics. "
                "Install with: pip install geowombat[detect]"
            ) from e

        self.weights = str(weights)
        self.device = _resolve_device(device)
        self.imgsz = imgsz
        self._model = YOLO(self.weights)
        try:
            self._model.to(self.device)
        except Exception:
            pass

        if oriented is None:
            stem = Path(self.weights).stem.lower()
            oriented = 'obb' in stem
        self.oriented = bool(oriented)

        model_names = getattr(self._model, 'names', None) or {}
        if classes is not None:
            self.class_names = list(classes)
        elif isinstance(model_names, dict):
            ordered = sorted(model_names.items(), key=lambda kv: kv[0])
            self.class_names = [v for _, v in ordered]
        else:
            self.class_names = list(model_names)

    def fit(self, dataset_yaml, epochs=50, imgsz=640, **kwargs):
        """Fine-tune YOLO on a dataset produced by ``build_yolo_dataset``.

        Thin wrapper around ``ultralytics.YOLO.train``.

        Parameters
        ----------
        dataset_yaml : str or Path
            Path to ``data.yaml`` written by ``build_yolo_dataset``.
        epochs : int
            Training epochs. Default 50.
        imgsz : int
            Training image size. Default 640.
        **kwargs
            Additional kwargs forwarded to ``YOLO.train``.
        """
        results = self._model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            imgsz=imgsz,
            device=self.device,
            **kwargs,
        )
        # Refresh class names if the model was retrained with new ones
        model_names = getattr(self._model, 'names', None) or {}
        if isinstance(model_names, dict):
            ordered = sorted(model_names.items(), key=lambda kv: kv[0])
            self.class_names = [v for _, v in ordered]
        return results

    def _result_to_dets(self, r):
        """Convert a single ultralytics ``Results`` object to our tuple list."""
        out = []
        if self.oriented and getattr(r, 'obb', None) is not None:
            obb = r.obb
            try:
                xyxyxyxy = obb.xyxyxyxy.cpu().numpy()  # (N, 4, 2)
                confs = obb.conf.cpu().numpy()
                clses = obb.cls.cpu().numpy().astype(int)
            except AttributeError:
                return []
            for quad, score, cls_id in zip(xyxyxyxy, confs, clses):
                out.append((np.asarray(quad), float(score), int(cls_id)))
            return out

        boxes = getattr(r, 'boxes', None)
        if boxes is None or boxes.xyxy is None:
            return []
        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clses = boxes.cls.cpu().numpy().astype(int)
        except AttributeError:
            return []
        for (x1, y1, x2, y2), score, cls_id in zip(xyxy, confs, clses):
            out.append(
                (float(x1), float(y1), float(x2), float(y2),
                 float(score), int(cls_id))
            )
        return out

    def _detect_tile(self, rgb_array, conf=0.25, max_det=None):
        # Single-image path for backwards compatibility. The batched
        # path in `predict()` goes through `_detect_batch` instead.
        return self._detect_batch(
            [rgb_array], conf=conf, max_det=max_det,
        )[0]

    def _detect_batch(self, rgb_list, conf=0.25, max_det=None):
        if not rgb_list:
            return []
        imgsz = self.imgsz or max(
            rgb_list[0].shape[0], rgb_list[0].shape[1],
        )
        predict_kwargs = dict(
            source=rgb_list,    # Ultralytics natively batches a list of arrays
            conf=conf,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
        )
        if max_det is not None:
            predict_kwargs['max_det'] = int(max_det)
        results = self._model.predict(**predict_kwargs)
        if not results:
            return [[] for _ in rgb_list]
        # Ultralytics returns one Results per input image when given a list.
        return [self._result_to_dets(r) for r in results]


# ---------------------------------------------------------------------------
# TorchGeo detection wrapper
# ---------------------------------------------------------------------------

class TorchGeoDetector(GeoWombatDetector):
    """Detection wrapper around TorchGeo / torchvision detection models.

    Supports Faster R-CNN and RetinaNet via torchvision with optional
    pretrained weights from TorchGeo (e.g. xView). Axis-aligned only.

    Parameters
    ----------
    model : {'faster-rcnn', 'retinanet'}
        Detection head. Default 'faster-rcnn'.
    weights : str, optional
        TorchGeo weights enum string (e.g. 'FCN_RESNET50_XVIEW') or path
        to a state dict. If None, uses torchvision COCO pretrained.
    num_classes : int, optional
        Number of classes including background. Required when loading
        a custom-trained checkpoint.
    classes : list of str, optional
        Class names corresponding to non-background ids 1..N.
    device : str
        'cpu', 'cuda', or 'auto'. Default 'auto'.

    Notes
    -----
    For aerial / satellite imagery, the most useful TorchGeo weights
    are typically 'FASTERRCNN_RESNET50_FPN_XVIEW' (when available in
    your TorchGeo version). Check ``torchgeo.models`` for the current
    catalogue.
    """

    oriented = False

    def __init__(self, model='faster-rcnn', weights=None, num_classes=None,
                 classes=None, device='auto'):
        try:
            import torch
            import torchvision
            from torchvision.models.detection import (
                fasterrcnn_resnet50_fpn,
                retinanet_resnet50_fpn,
            )
        except ImportError as e:
            raise ImportError(
                "TorchGeoDetector requires torch + torchvision. "
                "Install with: pip install geowombat[dl,detect]"
            ) from e

        self._torch = torch
        self.device = _resolve_device(device)
        self.model_name = model

        torchgeo_weights = None
        state_dict_path = None
        if isinstance(weights, str):
            # Try TorchGeo enum first; fall back to a file path.
            try:
                import torchgeo.models as tgm
                enum_name, _, member = weights.partition('.')
                wenum = getattr(tgm, enum_name, None)
                if wenum is not None and member:
                    torchgeo_weights = getattr(wenum, member, None)
                elif wenum is not None:
                    torchgeo_weights = wenum
            except (ImportError, AttributeError):
                torchgeo_weights = None
            if torchgeo_weights is None and Path(weights).exists():
                state_dict_path = weights

        if model == 'faster-rcnn':
            net = fasterrcnn_resnet50_fpn(
                weights='DEFAULT' if (torchgeo_weights is None
                                      and state_dict_path is None)
                else None,
                num_classes=num_classes,
            )
        elif model == 'retinanet':
            net = retinanet_resnet50_fpn(
                weights='DEFAULT' if (torchgeo_weights is None
                                      and state_dict_path is None)
                else None,
                num_classes=num_classes,
            )
        else:
            raise ValueError(
                f"Unknown model '{model}'. Use 'faster-rcnn' or 'retinanet'."
            )

        if torchgeo_weights is not None:
            try:
                state = torchgeo_weights.get_state_dict(progress=True)
                net.load_state_dict(state, strict=False)
            except Exception as e:
                warnings.warn(
                    f"Could not load TorchGeo weights '{weights}': {e}. "
                    "Falling back to torchvision defaults."
                )
        elif state_dict_path is not None:
            state = torch.load(state_dict_path, map_location='cpu')
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            net.load_state_dict(state, strict=False)

        self._model = net.to(self.device).eval()

        if classes is not None:
            self.class_names = list(classes)
        else:
            # COCO defaults if using torchvision weights
            self.class_names = self._coco_names()

    @staticmethod
    def _coco_names():
        return [
            'background', 'person', 'bicycle', 'car', 'motorcycle',
            'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'street sign', 'stop sign', 'parking meter',
            'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe',
        ]

    def _output_to_dets(self, out, conf=0.25, max_det=None):
        """Convert a single torchvision detection output to tuple list."""
        boxes = out['boxes'].cpu().numpy()
        scores = out['scores'].cpu().numpy()
        labels = out['labels'].cpu().numpy().astype(int)

        keep = scores >= conf
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        if max_det is not None and len(scores) > max_det:
            order = np.argsort(-scores)[:max_det]
            boxes = boxes[order]
            scores = scores[order]
            labels = labels[order]

        return [
            (float(b[0]), float(b[1]), float(b[2]), float(b[3]),
             float(s), int(c))
            for b, s, c in zip(boxes, scores, labels)
        ]

    def _detect_tile(self, rgb_array, conf=0.25, max_det=None):
        return self._detect_batch(
            [rgb_array], conf=conf, max_det=max_det,
        )[0]

    def _detect_batch(self, rgb_list, conf=0.25, max_det=None):
        if not rgb_list:
            return []
        torch = self._torch
        # (H, W, 3) uint8 → (3, H, W) float [0, 1], moved to device
        tensors = [
            (torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0).to(
                self.device,
            )
            for rgb in rgb_list
        ]
        with torch.no_grad():
            outputs = self._model(tensors)
        return [
            self._output_to_dets(o, conf=conf, max_det=max_det)
            for o in outputs
        ]


# ---------------------------------------------------------------------------
# SAM refinement
# ---------------------------------------------------------------------------

class SAMRefiner:
    """Refine bounding-box detections into polygons using SAM.

    Each input box is used as a prompt to the Segment Anything model
    (or SAM2 if installed) to produce a precise polygon mask, then
    polygonized back into vector geometry in the source CRS.

    Requires: ``pip install geowombat[sam]``

    Parameters
    ----------
    checkpoint : str or Path
        Path to a SAM checkpoint (``sam_vit_b.pth``, etc).
    model_type : {'vit_b', 'vit_l', 'vit_h'}
        SAM backbone size. Default 'vit_b'.
    device : str
        'cpu', 'cuda', or 'auto'. Default 'auto'.

    Example
    -------
    >>> ref = SAMRefiner(checkpoint='sam_vit_b.pth')
    >>> polys = ref.refine(src, boxes_gdf, band_indices=[2, 1, 0],
    ...                    scale=(0, 3000))
    """

    def __init__(self, checkpoint, model_type='vit_b', device='auto'):
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError as e:
            raise ImportError(
                "SAMRefiner requires segment-anything. "
                "Install with: pip install geowombat[sam]"
            ) from e

        self.device = _resolve_device(device)
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(self.device)
        self._predictor = SamPredictor(sam)

    def refine(self, src, boxes_gdf, band_indices=None, scale=None,
               pad_pixels=8, simplify_tolerance=0.5):
        """Refine boxes to polygon masks.

        Parameters
        ----------
        src : xarray.DataArray
            Source raster (must match ``boxes_gdf.crs``).
        boxes_gdf : geopandas.GeoDataFrame
            Detector output. Each row's geometry should be the bbox.
        band_indices : list of int, optional
            RGB bands for SAM input.
        scale : tuple of (lo, hi), optional
            Stretch range.
        pad_pixels : int
            Pad around each box when reading the chip. Default 8.
        simplify_tolerance : float
            Polygon simplification tolerance in CRS units. Default 0.5.

        Returns
        -------
        geopandas.GeoDataFrame
            Same columns as input; geometry replaced by polygons.
        """
        import numpy as np
        from shapely.geometry import Polygon as ShPoly

        try:
            from rasterio.features import shapes as rio_shapes
        except ImportError as e:
            raise ImportError(
                "SAMRefiner.refine requires rasterio."
            ) from e

        if boxes_gdf.empty:
            return boxes_gdf.copy()

        if boxes_gdf.crs is None or src.gw.crs_to_pyproj.to_epsg() != \
                boxes_gdf.crs.to_epsg():
            boxes_gdf = boxes_gdf.to_crs(src.gw.crs_to_pyproj)

        band_indices = resolve_band_indices(src, band_indices)
        affine = src.gw.affine
        inv = ~affine

        out_geoms = []
        for _, row in boxes_gdf.iterrows():
            minx, miny, maxx, maxy = row.geometry.bounds
            col_ul, row_ul = inv * (minx, maxy)
            col_lr, row_lr = inv * (maxx, miny)
            x0 = int(np.floor(min(col_ul, col_lr))) - pad_pixels
            y0 = int(np.floor(min(row_ul, row_lr))) - pad_pixels
            x1 = int(np.ceil(max(col_ul, col_lr))) + pad_pixels
            y1 = int(np.ceil(max(row_ul, row_lr))) + pad_pixels

            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(src.gw.ncols, x1)
            y1 = min(src.gw.nrows, y1)
            if x1 <= x0 or y1 <= y0:
                out_geoms.append(row.geometry)
                continue

            chip = src.isel(
                y=slice(y0, y1), x=slice(x0, x1),
            ).values
            if chip.ndim == 4:
                chip = chip[0]

            rgb = _prepare_rgb_tile(chip, band_indices, scale)
            self._predictor.set_image(rgb)

            box_px = np.array([
                min(col_ul, col_lr) - x0,
                min(row_ul, row_lr) - y0,
                max(col_ul, col_lr) - x0,
                max(row_ul, row_lr) - y0,
            ], dtype=np.float32)

            masks, scores, _ = self._predictor.predict(
                box=box_px, multimask_output=False,
            )
            mask = masks[0].astype(np.uint8)

            from affine import Affine
            chip_affine = Affine(
                affine.a, affine.b,
                affine.c + x0 * affine.a + y0 * affine.b,
                affine.d, affine.e,
                affine.f + x0 * affine.d + y0 * affine.e,
            )

            polys = []
            for geom_dict, val in rio_shapes(mask, transform=chip_affine):
                if val != 1:
                    continue
                coords = geom_dict.get('coordinates', [])
                if not coords:
                    continue
                outer = coords[0]
                holes = coords[1:] if len(coords) > 1 else None
                try:
                    p = ShPoly(outer, holes=holes)
                except Exception:
                    continue
                if simplify_tolerance > 0:
                    p = p.simplify(simplify_tolerance)
                if not p.is_empty and p.is_valid:
                    polys.append(p)

            if polys:
                # Keep largest polygon — masks can occasionally produce
                # disconnected components near the prompt box edge.
                polys.sort(key=lambda g: g.area, reverse=True)
                out_geoms.append(polys[0])
            else:
                out_geoms.append(row.geometry)

        result = boxes_gdf.copy()
        result['geometry'] = out_geoms
        return result
