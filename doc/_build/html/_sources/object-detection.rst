.. _object-detection:

Object detection
================

GeoWombat ships object detectors that operate on georeferenced rasters
and return ``GeoDataFrame`` outputs in the source CRS. Everything stays
inside the familiar ``with gw.open(...) as src:`` / ``src.gw.<method>``
pattern, with module-level wrappers in ``gw.ml`` that mirror
``fit`` / ``predict`` / ``fit_predict`` for classification.

Three detectors are included:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Detector
     - Backend
     - Notes
   * - ``YOLODetector``
     - Ultralytics YOLO
     - Axis-aligned + oriented boxes. **DOTA-v1 OBB weights recommended
       for aerial / satellite imagery.** License: AGPL-3.0.
   * - ``TorchGeoDetector``
     - TorchVision Faster R-CNN / RetinaNet
     - Optional TorchGeo pretrained weights (e.g. xView).
   * - ``SAMRefiner``
     - Segment Anything
     - Refines bounding boxes to polygon masks.

.. admonition:: Recommended setup for aerial / satellite imagery

   Pair **DOTA-v1 pretrained weights** (Ultralytics ``yolov8*-obb.pt``)
   with **oriented bounding boxes** (``oriented=True``). DOTA-v1 is the
   standard aerial OBB benchmark — 15 classes (planes, ships, vehicles,
   storage tanks, sports fields, ...) — and its rotated boxes capture
   objects at arbitrary heading, which is the norm in overhead imagery.
   See :ref:`yolo-variants` for weight choices and *Convert polygon
   labels to boxes* below for how to generate matching training labels.

   The default COCO weights (``yolov8n.pt``) used in the introductory
   snippets are there to exercise the plumbing on the bundled Landsat
   scene — they are intentionally **not** the right model for real
   aerial work.

Setup
-----

Install the detection extras::

    pip install "geowombat[detect]"

For SAM refinement::

    pip install "geowombat[sam]"

The detector classes load their model state lazily inside ``__init__``,
so importing ``geowombat.detect`` itself stays light — you only pay
for torch / ultralytics when you actually instantiate a detector.

Public API at a glance
----------------------

**``.gw`` accessor (use these from inside ``with gw.open(...) as src:``)**

- ``src.gw.detect(detector, ...)`` — tiled, georeferenced inference.
- ``src.gw.to_yolo_dataset(labels, class_col=..., out_dir=...)`` —
  write a YOLO training corpus on disk.

**Module-level (in ``geowombat.detect``)**

- ``predict(src, detector, **kwargs)`` — functional form of the accessor.
- ``fit(detector, dataset_yaml, **kwargs)`` — fine-tune a detector on
  a YOLO dataset.
- ``fit_predict(src, detector, labels, class_col, out_dir, ...)`` —
  end-to-end: build dataset → fine-tune → predict.
- ``build_dataset(...)`` — function form of ``src.gw.to_yolo_dataset``
  (alias for ``build_yolo_dataset``).
- ``boxes_from_polygons(gdf, oriented=False)`` — polygon labels to
  axis-aligned or oriented bounding boxes.
- ``detection_accuracy(predictions, truth, class_col, iou_thresholds)``
  — per-class precision / recall / F1 / AP plus a review-ready
  GeoDataFrame.
- ``export_for_review`` / ``recompute_from_review`` — QGIS review
  round-trip via GeoPackage.
- ``plot_detections(src, predictions, truth, ax=...)`` — matplotlib
  rendering colored by TP / FP / FN.

Examples below use the bundled Landsat 8 test scene and label polygons
so they are self-contained — no downloads required:

.. code-block:: python

    import warnings
    warnings.filterwarnings('ignore')

    import geopandas as gpd
    import matplotlib.pyplot as plt

    import geowombat as gw
    from geowombat.data import (
        l8_224078_20200518,
        l8_224078_20200518_polygons,
    )
    from geowombat.detect import (
        YOLODetector,
        boxes_from_polygons,
        build_dataset,
        detection_accuracy,
        fit_predict,
        plot_detections,
        predict,
    )

    # The bundled polygon set has a `name` column with land-cover classes.
    # Detection treats it as a generic class column; we rename for clarity.
    labels = gpd.read_file(l8_224078_20200518_polygons)
    labels['class_name'] = labels['name']
    print(sorted(labels['class_name'].unique()))
    # ['crop', 'developed', 'tree', 'water']

Run a pretrained detector
-------------------------

The simplest workflow: open the raster, hand a detector instance to
``src.gw.detect``. The result is a ``GeoDataFrame`` in the raster's CRS.

.. code-block:: python

    # Load the model once — weights file is auto-downloaded on first use.
    det = YOLODetector(weights='yolov8n.pt')

    with gw.config.update(sensor='bgr', ref_res=300):
        with gw.open(l8_224078_20200518, nodata=0) as src:
            preds = src.gw.detect(
                det,
                tile_size=320,    # tile size for inference
                overlap=0.0,      # overlap between tiles (0–0.9)
                conf=0.05,        # keep detections above this confidence
                scale=(0, 10000), # rescale pixel values to 0–255 (skip for 8-bit input)
            )

    print(f'{len(preds)} detections')
    print(preds.columns.tolist())
    # ['geometry', 'class_id', 'class_name', 'score', 'tile_id']

.. note::

   The bundled scene is Landsat 8 surface-reflectance, not aerial imagery,
   and the YOLO weights here are pretrained on COCO. Expect few — or no —
   meaningful detections; this example is exercising the *plumbing*, not
   producing useful labels. The :ref:`fine-tuning <obj-det-finetune>`
   section below shows the realistic flow.

The three call shapes below are equivalent — pick whichever reads best
in your code:

.. code-block:: python

    # 1. Accessor (recommended inside `with gw.open(...) as src:`)
    preds = src.gw.detect(det, conf=0.05, scale=(0, 10000))

    # 2. Module-level function (parallels gw.ml.predict)
    preds = predict(src, det, conf=0.05, scale=(0, 10000))

    # 3. Calling the detector directly
    preds = det.predict(src, conf=0.05, scale=(0, 10000))

Sensor config drives band indices
---------------------------------

``YOLO`` and ``TorchGeo`` detectors consume an RGB image per tile. When
``gw.config.update(sensor=...)`` is active, ``src.gw.detect`` reads
``src.band.values`` and picks the R / G / B triplet automatically — no
need to pass ``band_indices`` per call. Explicit ``band_indices=[...]``
still wins.

.. code-block:: python

    with gw.config.update(sensor='bgr', ref_res=300):
        with gw.open(l8_224078_20200518, nodata=0) as src:
            print(src.band.values.tolist())     # ['blue', 'green', 'red']
            preds = src.gw.detect(det, conf=0.05, scale=(0, 10000))
            # band_indices automatically resolved to [2, 1, 0]

If your raster has unnamed bands, ``src.gw.detect`` falls back to
``[0, 1, 2]`` for 3+-band rasters or broadcasts band 0 across RGB for
single-band rasters.

Convert polygon labels to boxes
-------------------------------

Detection works on bounding boxes. ``boxes_from_polygons`` replaces
polygon geometries with either axis-aligned envelopes (**AABB**) or
minimum rotated rectangles (**OBB**):

- **AABB** (``oriented=False``, default) — sides parallel to the image
  axes. Use for objects that line up with the grid: buildings in nadir
  aerial imagery, parking-lot cars, parcels.
- **OBB** (``oriented=True``) — rotated rectangles. **Recommended for
  most aerial / satellite work**, because real-world objects appear at
  arbitrary heading. Pair OBB labels with DOTA-v1 pretrained weights
  (``yolov8*-obb.pt``) — see :ref:`yolo-variants`.

.. code-block:: python

    aabb = boxes_from_polygons(labels, oriented=False)
    obb  = boxes_from_polygons(labels, oriented=True)
    print(aabb['_box_kind'].unique(), obb['_box_kind'].unique())
    # ['aabb'] ['obb']

You don't have to call this yourself — ``build_dataset`` and
``src.gw.to_yolo_dataset`` do it internally when you pass
``oriented=True``.

.. _digitizing-obb:

Digitizing polygons for high-quality OBB labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``oriented=True`` uses ``shapely.minimum_rotated_rectangle`` under the
hood, which finds the smallest rotated rectangle enclosing the
polygon's extreme points. The OBB is therefore only as good as the
polygon you feed it:

- **Trace the object tightly**, especially along its long axis. A
  rectangle traced around a ship's hull yields a clean OBB; a loose
  blob around the same ship yields a rectangle rotated by the blob's
  noise, not the ship's heading.
- **Prefer 4–8 vertex polygons** that follow the object's outline.
  Extra vertices off the silhouette pull the minimum rotated rectangle
  off-axis.
- **In QGIS**, enable snapping and use the "Add Polygon Feature" tool
  on top of an orthophoto basemap; or digitize a rectangle directly
  with the "Rectangles from Center and a Point" / "Rectangles from
  3 Points" tools when objects are near-rectangular.
- **For large label sets**, generate loose polygons cheaply (manual or
  scripted) and refine them with :class:`~geowombat.detect.SAMRefiner`
  *before* passing to ``boxes_from_polygons(oriented=True)`` — SAM
  masks hug the object outline, so the resulting OBB tracks the true
  heading.

Build a YOLO training dataset
-----------------------------

``src.gw.to_yolo_dataset`` tiles the raster + label GeoDataFrame into
an Ultralytics-layout directory on disk:

.. code-block:: text

    out_dir/
      data.yaml
      images/{train,val}/tile_r####_c####.jpg
      labels/{train,val}/tile_r####_c####.txt

.. code-block:: python

    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / 'yolo_lc'
        with gw.config.update(sensor='bgr', ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                info = src.gw.to_yolo_dataset(
                    labels,                 # vector labels
                    class_col='class_name', # column with the class name
                    out_dir=out_dir,        # where to write the dataset
                    tile_size=128,          # tile size in pixels
                    overlap=0.0,            # overlap between tiles (0–0.9)
                    val_split=0.25,         # fraction of tiles used for validation
                    min_box_pixels=2,       # drop boxes smaller than this
                    scale=(0, 10000),       # rescale pixel values to 0–255
                    background_ratio=0.0,   # keep some empty tiles as negatives (0 = none)
                )
        print(info)
        # {'out_dir': '...', 'classes': ['crop','developed','tree','water'],
        #  'n_train': 3, 'n_val': 1, 'n_boxes': 4, ...}

Equivalent module-level form:

.. code-block:: python

    with gw.open(l8_224078_20200518, nodata=0) as src:
        info = build_dataset(
            src, labels, class_col='class_name',
            out_dir=out_dir, tile_size=128,
            scale=(0, 10000), min_box_pixels=2,
        )

Key parameters:

- ``tile_size``: square tile edge in pixels. Match this to your
  detector's training image size.
- ``overlap``: fractional overlap between adjacent tiles
  (``0..0.9``). Useful for detection because objects on tile seams
  otherwise get cut.
- ``min_box_pixels``: drop boxes smaller than this after tile clipping.
- ``background_ratio``: fraction of empty tiles (0..1) to keep as
  hard negatives. ``0`` drops them all.
- ``scale=(lo, hi)``: linear stretch applied before writing 8-bit
  imagery. Required for non-uint8 rasters (Landsat / Sentinel DN).
- ``oriented=True``: write OBB labels (8 corner coords per box).

.. _obj-det-finetune:

End-to-end: build, fine-tune, predict
-------------------------------------

``fit_predict`` does all three in one call, matching the
``fit_predict`` shape from the classification API. Useful for
notebook-style exploration:

.. code-block:: python

    with tempfile.TemporaryDirectory() as td:
        det = YOLODetector(weights='yolov8n.pt')
        with gw.config.update(sensor='bgr', ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                preds, summary = fit_predict(
                    src,
                    det,
                    labels,
                    class_col='class_name',
                    out_dir=Path(td) / 'ds',  # where the training dataset goes
                    tile_size=128,            # tile size (training + inference)
                    overlap=0.0,              # tile overlap
                    epochs=1,                 # training epochs (1 here for a quick demo)
                    min_box_pixels=2,         # drop boxes smaller than this
                    scale=(0, 10000),         # rescale pixel values to 0–255
                    val_split=0.5,            # bundled set is tiny — keep half for validation
                    seed=42,                  # reproducible split
                    predict_kwargs={'conf': 0.05},  # passed through to inference
                )
        print(summary['classes'], summary['n_boxes'], 'training boxes')
        print(len(preds), 'predictions')

.. note::

   The bundled label set has only 4 polygons — far too few for a real
   training run. Use this snippet to confirm the *pipeline* works, then
   point at a larger dataset. With ~100s of training boxes, ``epochs=50``
   and ``yolov8s.pt``/``yolov8m.pt`` are reasonable starting points.

For finer-grained control — for example, to inspect or save the
fine-tuned weights between training and inference — call the steps
separately. Note ``fit`` writes Ultralytics ``runs/`` output under
the current working directory.

.. code-block:: python

    from geowombat.detect import fit

    with tempfile.TemporaryDirectory() as td:
        ds_dir = Path(td) / 'ds'

        # 1. Build the YOLO dataset
        with gw.config.update(sensor='bgr', ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                summary = src.gw.to_yolo_dataset(
                    labels, class_col='class_name', out_dir=ds_dir,
                    tile_size=128, scale=(0, 10000), min_box_pixels=2,
                    val_split=0.5, seed=42,
                )

        # 2. Fine-tune.
        det = YOLODetector(weights='yolov8n.pt')
        fit(
            det,
            dataset_yaml=ds_dir / 'data.yaml',  # built in step 1
            epochs=1,                           # training epochs
            imgsz=128,                          # training image size
        )

        # 3. Predict — use the same tile size + scaling as in training.
        with gw.config.update(sensor='bgr', ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                preds = src.gw.detect(
                    det, conf=0.05, tile_size=128, scale=(0, 10000),
                )

Accuracy assessment
-------------------

``detection_accuracy`` computes per-class precision / recall / F1 / AP
at one or more IoU thresholds and returns:

- ``metrics`` — a multi-index ``DataFrame`` indexed by
  ``(iou_threshold, class)`` with columns ``ap``, ``precision``,
  ``recall``, ``f1``, ``tp``, ``fp``, ``fn``, ``support``.
- ``summary`` — a dict with ``mAP@{iou}`` keys.
- ``matched`` — a *review-ready* ``GeoDataFrame``: every truth and every
  prediction tagged with status (``TP``, ``FP``, ``FP_class``, ``FN``).

.. code-block:: python

    det = YOLODetector(weights='yolov8n.pt')
    with gw.config.update(sensor='bgr', ref_res=300):
        with gw.open(l8_224078_20200518, nodata=0) as src:
            preds = src.gw.detect(det, tile_size=320, overlap=0.0,
                                  conf=0.05, scale=(0, 10000))

    # COCO classes ≠ our land-cover classes; re-tag to compare spatially.
    preds = preds.copy()
    preds['class_name'] = 'developed'

    results = detection_accuracy(
        predictions=preds,
        truth=labels[['class_name', 'geometry']],
        class_col='class_name',
        iou_thresholds=(0.3, 0.5),
    )
    print(results['metrics'])
    print(results['summary'])    # e.g. {'mAP@0.3': ..., 'mAP@0.5': ...}

**Glossary** for the columns:

- **TP**: prediction overlaps truth with IoU ≥ threshold *and* has the
  right class.
- **FP**: prediction with no matching truth at the IoU threshold.
- **FP_class**: prediction overlaps a truth box but the class is wrong.
- **FN**: truth that no prediction matched.
- **precision** = TP / (TP + FP); **recall** = TP / (TP + FN);
  **F1** is their harmonic mean.
- **AP**: integrated precision-recall curve (per class). The summary's
  ``mAP@{iou}`` is the mean of those AP values.

Visualize TP / FP / FN
----------------------

``plot_detections`` overlays the raster with predictions and truth,
color-coded by status. Pass it the ``matched`` GeoDataFrame from
``detection_accuracy`` to keep colors consistent with the metrics.

.. code-block:: python

    fig, ax = plt.subplots(figsize=(10, 10))
    with gw.config.update(sensor='bgr', ref_res=300):
        with gw.open(l8_224078_20200518, nodata=0) as src:
            plot_detections(
                src,
                predictions=results['matched'],
                truth=labels,
                ax=ax,
                scale=(0, 10000),
            )
    plt.show()

QGIS review round-trip
----------------------

``export_for_review`` writes a GeoPackage you can step through
feature-by-feature in QGIS (e.g. with the ``GoToNextFeature3+`` plugin).
After a human fills in the ``reviewer_label`` field,
``recompute_from_review`` re-derives metrics from that human-corrected
file:

.. code-block:: python

    from geowombat.detect import export_for_review, recompute_from_review

    export_for_review(results['matched'], './review.gpkg')
    # ... open in QGIS, edit reviewer_label, save ...
    final_metrics = recompute_from_review('./review.gpkg')

Refine boxes to polygons with SAM
---------------------------------

``SAMRefiner`` uses each predicted bounding box as a prompt to Segment
Anything and replaces the box with a polygon mask in the same CRS. The
SAM checkpoint must be downloaded once from
`Meta's SAM page <https://github.com/facebookresearch/segment-anything>`_.

.. code-block:: python

    from geowombat.detect import SAMRefiner

    refiner = SAMRefiner(checkpoint='sam_vit_b.pth', model_type='vit_b')
    with gw.open(l8_224078_20200518, nodata=0) as src:
        polygons = refiner.refine(src, preds, scale=(0, 10000))
    polygons.to_file('refined_polygons.gpkg')

The returned GeoDataFrame has the same columns as the detector output
but ``geometry`` is now a polygon rather than a rectangle.

Choosing a backend
------------------

- **YOLO** is the right default for aerial / drone imagery and any case
  where AGPL-3.0 is acceptable. See :ref:`yolo-variants` below for the
  trained-on options — picking the right weights matters far more than
  picking the right model size.
- **TorchGeo** (``TorchGeoDetector``) wraps Faster R-CNN / RetinaNet and
  accepts TorchGeo pretrained weights such as ``FASTERRCNN_RESNET50_FPN_XVIEW``
  (60 aerial-imagery classes). Pick this when xView's class set fits
  better than DOTA's, or when the AGPL license on Ultralytics is a
  non-starter.
- **SAM** (``SAMRefiner``) is not a detector — it polishes detector
  outputs into precise vector polygons. Pair it with either of the
  above.

.. _yolo-variants:

YOLO weight families
~~~~~~~~~~~~~~~~~~~~

Ultralytics ships several pretrained YOLO weight families. The right
choice depends on **what you're detecting**, not just how fast you
need it to run. Default ``yolov8n.pt`` is trained on COCO (person,
car, dog, ...) and is the wrong tool for overhead imagery — most
COCO classes never appear from above.

.. list-table::
   :header-rows: 1
   :widths: 22 18 35 25

   * - Weight family
     - Trained on
     - Classes
     - When to use
   * - ``yolov8{n,s,m,l,x}.pt``
     - COCO
     - 80 everyday classes (person, car, dog, ...)
     - Ground-level photos. Wrong tool for overhead imagery.
   * - ``yolo11{n,s,m,l,x}.pt``
     - COCO
     - Same 80 classes
     - Newer architecture, same training set. Same caveat.
   * - ``yolov8{n,s,m,l,x}-obb.pt``
     - **DOTA-v1** (aerial)
     - 15 aerial classes: *plane, ship, storage tank, baseball diamond,
       tennis court, basketball court, ground track field, harbor, bridge,
       large vehicle, small vehicle, helicopter, roundabout, soccer ball
       field, swimming pool*
     - **Best out-of-the-box choice for satellite/aerial imagery.**
       Produces oriented (rotated) boxes.
   * - ``yolov8{s,m,l,x}-worldv2.pt``
     - LVIS + grounding
     - **Open-vocabulary** — pass text prompts at inference time
     - When DOTA doesn't cover what you need. Use
       ``model.set_classes(['airplane', 'truck', ...])`` before predicting.
   * - Custom-trained checkpoint
     - Your dataset
     - Whatever you trained on
     - After fine-tuning — see :ref:`obj-det-finetune` above.

Size suffix (``n`` < ``s`` < ``m`` < ``l`` < ``x``) trades inference
speed for accuracy. Default to ``n`` or ``s`` for prototyping, ``m``+
for production.

For OBB weights, geowombat auto-detects orientation from the filename
ending in ``-obb.pt`` — you don't need to pass ``oriented=True``
explicitly:

.. code-block:: python

    from geowombat.detect import YOLODetector

    # Predictions come back as rotated polygons instead of axis-aligned
    # boxes. `oriented=True` is inferred from the `-obb.pt` filename.
    det = YOLODetector(weights='yolov8n-obb.pt')

    with gw.open('aerial.tif') as src:
        preds = src.gw.detect(
            det,
            conf=0.25,   # this model is confident — a higher threshold works well
        )
        print(preds.geometry.iloc[0])  # rotated 4-corner Polygon

To fine-tune a DOTA-v1 OBB model on your own data, generate OBB labels
following :ref:`digitizing-obb` and pass ``oriented=True`` to
``build_dataset`` / ``src.gw.to_yolo_dataset``.

See also the notebook
---------------------

``notebooks/object_detection.ipynb`` runs the full real-world flow:
NAIP aerial imagery from Microsoft Planetary Computer, OpenStreetMap
building footprints, dataset construction, fine-tuning, before/after
metric comparison, and the QGIS review export.
