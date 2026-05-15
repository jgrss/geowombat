"""Generate notebooks/object_detection.ipynb.

Run with: ``python notebooks/_build_object_detection_ipynb.py``.

This generator exists because hand-writing nested JSON is error-prone.
The output notebook is checked in alongside the generator so end users
do not need to run it.
"""

import json
from pathlib import Path


def md(*lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _to_source(lines),
    }


def code(*lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _to_source(lines),
    }


def _to_source(lines):
    text = "\n".join(lines)
    parts = text.split("\n")
    # Jupyter stores source as a list where every line except the last
    # has a trailing newline.
    out = []
    for i, p in enumerate(parts):
        if i < len(parts) - 1:
            out.append(p + "\n")
        else:
            out.append(p)
    return out


CELLS = []

CELLS.append(md(
    "# Object Detection with geowombat",
    "",
    "This notebook walks through the geowombat object-detection module:",
    "",
    "1. **Quickstart** — run a pretrained YOLO model on a public satellite",
    "   detection benchmark (NWPU VHR-10 via TorchGeo) and score its accuracy.",
    "2. **Geowombat-native workflow** — fetch a NAIP aerial image from",
    "   Microsoft Planetary Computer plus OpenStreetMap building footprints,",
    "   build a YOLO training dataset, run inference, score it, and export a",
    "   GeoPackage you can review feature-by-feature in QGIS (e.g. with the",
    "   GoToNextFeature3+ plugin).",
    "3. **Optional fine-tuning** — fine-tune YOLO on the dataset built in",
    "   step 2 so detections are actually useful for buildings.",
    "",
    "Required installs:",
    "",
    "```",
    "pip install geowombat[dl,detect]",
    "pip install pystac-client planetary-computer osmnx torchgeo",
    "```",
))

CELLS.append(md(
    "## Setup",
))

CELLS.append(code(
    "import warnings",
    "from pathlib import Path",
    "",
    "import geopandas as gpd",
    "import matplotlib.pyplot as plt",
    "import numpy as np",
    "",
    "import geowombat as gw",
    "from geowombat.ml.detection_data import (",
    "    boxes_from_polygons,",
    "    build_yolo_dataset,",
    ")",
    "from geowombat.ml.detection_metrics import (",
    "    detection_accuracy,",
    "    export_for_review,",
    "    plot_detections,",
    ")",
    "",
    "warnings.filterwarnings('ignore')",
    "WORK_DIR = Path('object_detection_demo')",
    "WORK_DIR.mkdir(exist_ok=True)",
))

# ---------------- Section A ----------------
CELLS.append(md(
    "## 1. Quickstart: pretrained YOLO on NWPU VHR-10",
    "",
    "[NWPU VHR-10](https://gcheng-nwpu.github.io/) is a small (~715 images,",
    "10 classes) public satellite-imagery detection benchmark. TorchGeo",
    "auto-downloads it. We'll run a pretrained Ultralytics YOLO model on a",
    "few images and score the result.",
    "",
    "**Caveat:** YOLO's default weights are trained on COCO, which only",
    "overlaps with VHR-10 on a couple of classes (e.g. *airplane*, *ship*,",
    "*vehicle*). Expect low mAP — this section is about exercising the",
    "API, not about state-of-the-art accuracy. The geowombat-native section",
    "below shows the full train-and-score loop.",
))

CELLS.append(code(
    "from torchgeo.datasets import VHR10",
    "",
    "vhr_dir = WORK_DIR / 'vhr10'",
    "ds = VHR10(root=str(vhr_dir), split='positive', download=True)",
    "print(f'Dataset size: {len(ds)} positive images')",
    "print('Classes:', ds.categories[:5], '...')",
))

CELLS.append(code(
    "# Inspect a single sample. TorchGeo 0.9 keys: image, label, bbox_xyxy, mask",
    "sample = ds[0]",
    "img = sample['image'].permute(1, 2, 0).numpy().astype(np.uint8)",
    "boxes = sample['bbox_xyxy'].numpy()",
    "labels = sample['label'].numpy()",
    "",
    "fig, ax = plt.subplots(figsize=(8, 8))",
    "ax.imshow(img)",
    "for (x1, y1, x2, y2), lbl in zip(boxes, labels):",
    "    ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,",
    "                               fill=False, edgecolor='yellow', lw=2))",
    "    ax.text(x1, y1-3, ds.categories[lbl], color='yellow', fontsize=8)",
    "ax.set_title('NWPU VHR-10 sample with ground-truth boxes')",
    "ax.axis('off')",
    "plt.show()",
))

CELLS.append(code(
    "# Run pretrained YOLO directly on the chip — bypassing gw.open() since",
    "# VHR-10 images are not georeferenced. We'll use the underlying",
    "# ultralytics model directly here for simplicity.",
    "from ultralytics import YOLO",
    "",
    "yolo = YOLO('yolov8n.pt')",
    "result = yolo.predict(source=img, conf=0.1, verbose=False)[0]",
    "",
    "fig, ax = plt.subplots(figsize=(8, 8))",
    "ax.imshow(img)",
    "if result.boxes is not None and len(result.boxes) > 0:",
    "    for b, c, s in zip(result.boxes.xyxy.cpu().numpy(),",
    "                       result.boxes.cls.cpu().numpy().astype(int),",
    "                       result.boxes.conf.cpu().numpy()):",
    "        x1, y1, x2, y2 = b",
    "        ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,",
    "                                    fill=False, edgecolor='red', lw=2))",
    "        ax.text(x1, y1-3, f'{yolo.names[c]} {s:.2f}',",
    "                color='red', fontsize=8)",
    "ax.set_title('YOLO predictions (COCO classes) on VHR-10 chip')",
    "ax.axis('off')",
    "plt.show()",
))

CELLS.append(md(
    "## 2. Geowombat-native workflow: NAIP + OpenStreetMap buildings",
    "",
    "Now the full pipeline using a *georeferenced* raster and *georeferenced*",
    "vector labels:",
    "",
    "1. Pull a NAIP scene from Microsoft Planetary Computer for a small AOI.",
    "2. Pull OSM building footprints for the same AOI with `osmnx`.",
    "3. Build a YOLO-format training dataset with `build_yolo_dataset`.",
    "4. Run inference; observe that COCO-pretrained YOLO won't see buildings.",
    "5. Score with `detection_accuracy` and export a review GeoPackage.",
    "6. Optional: fine-tune YOLO on the dataset from step 3.",
))

CELLS.append(code(
    "# AOI: ~2 km square over Capitol Hill, Washington DC. Row-house",
    "# residential gives the fine-tune cell good signal (dense, regular,",
    "# visually distinctive buildings). Adjust the bounds for any other",
    "# city — wider bounds give more training data but a slower download.",
    "from shapely.geometry import box as shapely_box",
    "",
    "AOI_BOUNDS = (-77.012, 38.882, -76.988, 38.898)  # west, south, east, north",
    "aoi = gpd.GeoDataFrame(",
    "    {'name': ['demo']},",
    "    geometry=[shapely_box(*AOI_BOUNDS)],",
    "    crs='EPSG:4326',",
    ")",
    "aoi.plot()",
    "plt.title('Demo AOI: Capitol Hill, Washington DC')",
    "plt.show()",
))

CELLS.append(code(
    "# Fetch a recent NAIP image via Planetary Computer STAC.",
    "import pystac_client",
    "import planetary_computer",
    "",
    "catalog = pystac_client.Client.open(",
    "    'https://planetarycomputer.microsoft.com/api/stac/v1',",
    "    modifier=planetary_computer.sign_inplace,",
    ")",
    "search = catalog.search(",
    "    collections=['naip'],",
    "    bbox=AOI_BOUNDS,",
    "    datetime='2021/2023',",
    ")",
    "items = list(search.items())",
    "items.sort(key=lambda i: i.datetime, reverse=True)",
    "item = items[0]",
    "naip_url = item.assets['image'].href",
    "print(f'NAIP scene: {item.id}  ({item.datetime.date()})')",
    "print(f'URL: {naip_url[:80]}...')",
))

CELLS.append(code(
    "# Fetch OSM building footprints for the AOI",
    "import osmnx as ox",
    "",
    "tags = {'building': True}",
    "# osmnx 2.x: bbox is a single (left, bottom, right, top) tuple",
    "buildings = ox.features_from_bbox(AOI_BOUNDS, tags=tags)",
    "buildings = buildings[buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()",
    "buildings['class_name'] = 'building'",
    "print(f'Fetched {len(buildings)} building footprints')",
    "buildings.head(3)",
))

CELLS.append(code(
    "# Pre-download the AOI clip to a local TIF. Doing this once up front",
    "# (a) avoids Planetary Computer SAS token expiry mid-notebook and",
    "# (b) sidesteps gw.open()'s aversion to URLs with query strings.",
    "import rasterio",
    "from rasterio.warp import transform_bounds",
    "from rasterio.windows import from_bounds",
    "",
    "naip_local = WORK_DIR / 'naip_aoi.tif'",
    "with rasterio.open(naip_url) as ds_meta:",
    "    naip_crs = ds_meta.crs",
    "    aoi_in_naip_crs = transform_bounds('EPSG:4326', naip_crs, *AOI_BOUNDS)",
    "    window = from_bounds(*aoi_in_naip_crs, transform=ds_meta.transform)",
    "    arr = ds_meta.read(window=window)",
    "    profile = ds_meta.profile.copy()",
    "    profile.update({",
    "        'height': arr.shape[1],",
    "        'width': arr.shape[2],",
    "        'transform': ds_meta.window_transform(window),",
    "        'driver': 'GTiff',",
    "    })",
    "    with rasterio.open(naip_local, 'w', **profile) as dst:",
    "        dst.write(arr)",
    "print(f'Wrote local AOI clip: {naip_local} ({arr.shape})')",
    "",
    "# Read the *actual* bounds from the saved file. NAIP COG tiles often",
    "# don't cover the full requested AOI, so the on-disk raster is",
    "# smaller than what we asked for. Without clipping OSM truth to the",
    "# real extent, features that lie off the raster count as false",
    "# negatives — the detector can't possibly see them.",
    "with rasterio.open(naip_local) as ds_actual:",
    "    actual_naip_bounds = ds_actual.bounds",
    "print(f'Actual NAIP bounds (CRS units): {tuple(round(b, 1) for b in actual_naip_bounds)}')",
    "",
    "buildings_proj = buildings.to_crs(naip_crs)",
    "n_total = len(buildings_proj)",
    "naip_footprint = shapely_box(*actual_naip_bounds)",
    "buildings_proj = buildings_proj[",
    "    buildings_proj.intersects(naip_footprint)",
    "].copy()",
    "print(f'OSM buildings: {n_total} fetched, '",
    "      f'{len(buildings_proj)} inside actual raster footprint')",
    "",
    "with gw.open(naip_local, chunks=512) as src:",
    "    print(f'NAIP scene: {src.gw.nrows} x {src.gw.ncols}, '",
    "          f'{src.gw.nbands} bands, {src.gw.cellx:.2f} m')",
    "    fig, ax = plt.subplots(figsize=(10, 10))",
    "    rgb = src.isel(band=slice(0, 3)).values.transpose(1, 2, 0)",
    "    rgb = rgb.astype(np.uint8)  # NAIP is already uint8",
    "    ax.imshow(rgb, extent=(src.gw.left, src.gw.right,",
    "                           src.gw.bottom, src.gw.top))",
    "    buildings_proj.boundary.plot(ax=ax, color='yellow', lw=1)",
    "    ax.set_title('NAIP + OSM buildings (truth labels)')",
    "    plt.show()",
))

CELLS.append(code(
    "# Build a YOLO-format training dataset from the NAIP raster + buildings.",
    "# Bands 0,1,2 of NAIP are R,G,B and already 8-bit, so no scaling needed.",
    "yolo_dir = WORK_DIR / 'naip_buildings_yolo'",
    "",
    "with gw.open(naip_local, chunks=512) as src:",
    "    info = build_yolo_dataset(",
    "        src,",
    "        labels=buildings_proj,",
    "        class_col='class_name',",
    "        out_dir=yolo_dir,",
    "        tile_size=640,",
    "        overlap=0.1,",
    "        val_split=0.2,",
    "        min_box_pixels=10,",
    "        background_ratio=0.1,",
    "        band_indices=[0, 1, 2],",
    "        scale=None,  # NAIP is already uint8 RGB",
    "    )",
    "print(info)",
))

CELLS.append(md(
    "### Inference with pretrained YOLO",
    "",
    "We run `YOLODetector` directly on the NAIP scene. Pretrained COCO",
    "weights have no 'building' class, so we expect very few or no useful",
    "detections — but this verifies the inference pipeline (tile windowing,",
    "cross-tile NMS, pixel→CRS box conversion) end-to-end.",
))

CELLS.append(code(
    "from geowombat.ml.detectors import YOLODetector",
    "",
    "det = YOLODetector(weights='yolov8n.pt')",
    "",
    "with gw.open(naip_local, chunks=512) as src:",
    "    preds = det.predict(",
    "        src,",
    "        tile_size=640,",
    "        overlap=0.2,",
    "        conf=0.10,        # low threshold since COCO != buildings",
    "        band_indices=[0, 1, 2],",
    "        scale=None,",
    "        nms_iou=0.5,",
    "        progress=True,",
    "    )",
    "print(f'{len(preds)} detections from pretrained YOLO')",
    "preds.head()",
))

CELLS.append(md(
    "### Reading the accuracy output",
    "",
    "Before we compute metrics, a quick glossary — these terms appear in the",
    "tables below and in the per-feature `status` column of the review",
    "GeoPackage.",
    "",
    "**Per-detection / per-truth labels**",
    "",
    "- **TP — True Positive.** A predicted box that overlaps a ground-truth",
    "  box with IoU ≥ the threshold *and* has the correct class. A hit.",
    "- **FP — False Positive.** A predicted box with no matching truth at",
    "  the chosen IoU. A hallucination.",
    "- **FP_class — Wrong-class False Positive.** Predicted box landed in",
    "  about the right place (IoU ≥ threshold against *some* truth) but the",
    "  class label was wrong. Useful to separate 'model can't see it' from",
    "  'model sees it but mislabels it'.",
    "- **FN — False Negative.** A ground-truth box no detection matched.",
    "  A miss.",
    "",
    "**IoU — Intersection over Union.** The area of overlap divided by the",
    "area of union between two boxes. 1.0 = perfect overlap, 0.0 = no",
    "overlap. The IoU threshold (`0.3`, `0.5`, …) controls how strict the",
    "spatial agreement needs to be to count as a match. 0.5 is the standard",
    "PASCAL VOC threshold; 0.3 is more lenient and useful for noisy or very",
    "small targets.",
    "",
    "**Aggregate metrics**",
    "",
    "- **precision = TP / (TP + FP)** — of what the model predicted, what",
    "  fraction was right? Penalizes false alarms.",
    "- **recall = TP / (TP + FN)** — of all the real objects, what fraction",
    "  did the model find? Penalizes misses.",
    "- **F1 = 2 · (precision · recall) / (precision + recall)** — harmonic",
    "  mean of the two; one number that drops if either side is poor.",
    "- **AP — Average Precision.** Sweeps the confidence threshold from",
    "  high to low and integrates the precision-recall curve. A single",
    "  per-class number that captures the precision/recall tradeoff over",
    "  all confidence levels.",
    "- **mAP@0.5** — the mean of per-class AP at IoU ≥ 0.5. The standard",
    "  single-number summary for a detector.",
    "- **mAP@\\[.5:.95]** — average of mAP at IoU thresholds 0.5, 0.55, …,",
    "  0.95. COCO's stricter measure; rewards tightly localized boxes.",
    "- **support** — number of ground-truth boxes for that class (the",
    "  denominator for recall). Small support = noisy estimate.",
    "",
    "Higher is better for everything except the bucket counts (where higher",
    "is better only for TP).",
))

CELLS.append(code(
    "# Accuracy assessment vs. OSM truth. We re-tag predictions as 'building'",
    "# so the comparison is meaningful — pretrained YOLO labels things",
    "# 'car', 'truck', etc., but we just want to see what it *boxed*.",
    "preds_as_building = preds.copy()",
    "preds_as_building['class_name'] = 'building'",
    "",
    "results_pretrained = detection_accuracy(",
    "    predictions=preds_as_building,",
    "    truth=buildings_proj[['class_name', 'geometry']],",
    "    class_col='class_name',",
    "    iou_thresholds=(0.3, 0.5),",
    ")",
    "print('Per-class metrics (pretrained YOLO, no fine-tuning):')",
    "print(results_pretrained['metrics'])",
    "print()",
    "print('Summary:', results_pretrained['summary'])",
    "results = results_pretrained  # alias kept for the cells below",
))

CELLS.append(code(
    "# Visualize TP / FP / FN",
    "with gw.open(naip_local, chunks=512) as src:",
    "    fig, ax = plt.subplots(figsize=(12, 12))",
    "    plot_detections(",
    "        src,",
    "        predictions=results['matched'],",
    "        truth=buildings_proj,",
    "        ax=ax,",
    "        band_indices=[0, 1, 2],",
    "        scale=None,",
    "    )",
    "    plt.show()",
))

CELLS.append(code(
    "# Export the review GeoPackage. Open this in QGIS, use the attribute",
    "# form or GoToNextFeature3+ to step through each detection, and fill in",
    "# `reviewer_label` (TP/FP/FN/unclear). Then call recompute_from_review.",
    "review_path = WORK_DIR / 'review.gpkg'",
    "export_for_review(results['matched'], review_path)",
    "print(f'Review file: {review_path.resolve()}')",
    "print('  In QGIS: Open layer → switch to Form view → use',",
    "      '\"GoToNextFeature3+\" or built-in next-feature shortcut.')",
))

CELLS.append(md(
    "## 3. Fine-tune YOLO on the building dataset",
    "",
    "Pretrained COCO YOLO can't recognize buildings. Fine-tuning teaches",
    "it to. We train `yolov8n` (the smallest variant, ~3M parameters) on",
    "the dataset we just built. Settings chosen for CPU-friendliness:",
    "small `imgsz`, small `batch`, modest `epochs`. On a typical laptop",
    "CPU expect a few minutes total; on GPU, well under one minute.",
    "",
    "We send Ultralytics' `runs/` output into `WORK_DIR` so the demo",
    "leaves no clutter at the repo root.",
))

CELLS.append(code(
    "from ultralytics import YOLO",
    "",
    "yolo_train = YOLO('yolov8n.pt')",
    "_ = yolo_train.train(",
    "    data=str(yolo_dir.resolve() / 'data.yaml'),",
    "    epochs=15,",
    "    imgsz=416,",
    "    batch=4,",
    "    name='gw_buildings',",
    "    exist_ok=True,",
    "    verbose=False,",
    "    plots=False,",
    ")",
    "",
    "# Ultralytics writes runs/ relative to its own cwd regardless of the",
    "# project= kwarg, so we read the actual save_dir from the trainer.",
    "best_weights = Path(yolo_train.trainer.save_dir) / 'weights' / 'best.pt'",
    "print(f'Best weights: {best_weights}')",
    "print(f'Exists: {best_weights.exists()}')",
))

CELLS.append(md(
    "### Re-run inference with the fine-tuned model",
    "",
    "Same `YOLODetector` API, just point it at our new weights. We use a",
    "low confidence threshold (`0.05`) because with only ~17 training tiles",
    "and 15 epochs the model is heavily under-trained and hesitant —",
    "production weights would justify a much higher threshold like `0.25`.",
    "",
    "The point of this section is to show the *fine-tuning pipeline* works",
    "end-to-end. The numbers themselves will improve substantially with",
    "more data (a wider AOI), more epochs (50–100), and a larger backbone",
    "(`yolov8s.pt` or `yolov8m.pt`).",
))

CELLS.append(code(
    "det_ft = YOLODetector(",
    "    weights=str(best_weights),",
    "    classes=['building'],",
    ")",
    "",
    "with gw.open(naip_local, chunks=512) as src:",
    "    preds_ft = det_ft.predict(",
    "        src,",
    "        tile_size=416,           # match the size we trained at",
    "        overlap=0.2,",
    "        conf=0.05,               # under-trained model needs low conf",
    "        band_indices=[0, 1, 2],",
    "        scale=None,",
    "        nms_iou=0.5,",
    "        progress=True,",
    "    )",
    "print(f'{len(preds_ft)} detections from fine-tuned YOLO',",
    "      f'(was {len(preds)} pretrained)')",
))

CELLS.append(code(
    "# Score the fine-tuned predictions",
    "results_ft = detection_accuracy(",
    "    predictions=preds_ft,",
    "    truth=buildings_proj[['class_name', 'geometry']],",
    "    class_col='class_name',",
    "    iou_thresholds=(0.3, 0.5),",
    ")",
    "print('Per-class metrics (fine-tuned YOLO):')",
    "print(results_ft['metrics'])",
    "print()",
    "print('Summary:', results_ft['summary'])",
))

CELLS.append(md(
    "### Before vs. after",
    "",
    "Side-by-side comparison of the pretrained and fine-tuned runs at",
    "IoU ≥ 0.3. Look at TP/FN going up/down for recall and FP for",
    "precision — the F1 column is the single-number summary.",
))

CELLS.append(code(
    "import pandas as pd",
    "",
    "iou = 0.3  # use the more lenient threshold for the comparison",
    "row_pre = results_pretrained['metrics'].loc[(iou, 'building')]",
    "row_ft = results_ft['metrics'].loc[(iou, 'building')]",
    "compare = pd.DataFrame(",
    "    {'pretrained': row_pre, 'fine-tuned': row_ft},",
    ")",
    "compare['delta'] = compare['fine-tuned'] - compare['pretrained']",
    "print(f'Comparison at IoU >= {iou}:')",
    "print(compare.round(3))",
))

CELLS.append(code(
    "# Visualize TP / FP / FN from the fine-tuned run",
    "with gw.open(naip_local, chunks=512) as src:",
    "    fig, ax = plt.subplots(figsize=(12, 12))",
    "    plot_detections(",
    "        src,",
    "        predictions=results_ft['matched'],",
    "        truth=buildings_proj,",
    "        ax=ax,",
    "        band_indices=[0, 1, 2],",
    "        scale=None,",
    "    )",
    "    ax.set_title('Fine-tuned YOLO: TP (lime) / FP (red) / FN (magenta)')",
    "    plt.show()",
))

CELLS.append(md(
    "## Summary",
    "",
    "- `build_yolo_dataset` turns a `gw.open()`'d raster + label GeoDataFrame",
    "  into a YOLO-format training corpus on disk.",
    "- `YOLODetector` and `TorchGeoDetector` produce georeferenced",
    "  `GeoDataFrame` outputs from tiled, windowed inference with cross-tile",
    "  NMS.",
    "- `SAMRefiner` (requires `geowombat[sam]`) refines bounding-box outputs",
    "  to polygon masks using the box as a SAM prompt.",
    "- `detection_accuracy` computes per-class precision/recall/F1/AP at one",
    "  or more IoU thresholds and returns a *review-ready* GeoDataFrame.",
    "- `export_for_review` writes a GeoPackage for QGIS attribute-form",
    "  review; `recompute_from_review` recomputes metrics after a human has",
    "  labeled the review file.",
))

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


if __name__ == "__main__":
    out_path = Path(__file__).parent / "object_detection.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(NOTEBOOK, f, indent=1)
    print(f"Wrote {out_path}")
