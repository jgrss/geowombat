"""Object detection for geowombat.

Tiled, georeferenced bounding-box detection on top of Ultralytics YOLO
and TorchGeo / torchvision Faster R-CNN / RetinaNet, plus
training-dataset construction, accuracy metrics, and SAM-based polygon
refinement. Mirrors the ``fit / predict / fit_predict`` shape of
``geowombat.ml``.

Quick example
-------------
>>> import geowombat as gw
>>> from geowombat.detect import YOLODetector
>>> det = YOLODetector(weights='yolov8n.pt')
>>> with gw.open('aerial.tif') as src:
...     preds = src.gw.detect(det, conf=0.25)
"""

# Data builders + metrics + SAM-free utilities — no torch dependency.
from .data import boxes_from_polygons, build_yolo_dataset
from .metrics import (
    detection_accuracy,
    export_for_review,
    plot_detections,
    recompute_from_review,
)

# Module-level functional wrappers (require only the data layer).
from .api import (
    build_dataset,
    fit,
    fit_predict,
    predict,
)

__all__ = [
    'boxes_from_polygons',
    'build_dataset',
    'build_yolo_dataset',
    'detection_accuracy',
    'export_for_review',
    'fit',
    'fit_predict',
    'plot_detections',
    'predict',
    'recompute_from_review',
]

# Detector classes pull in torch / ultralytics lazily inside their
# constructors, but their module-level import only needs shapely,
# geopandas, numpy. Guard anyway so a partial install still imports
# the data + metrics surface above.
try:
    from .detectors import (
        SAMRefiner,
        TorchGeoDetector,
        YOLODetector,
    )
    __all__ += [
        'SAMRefiner',
        'TorchGeoDetector',
        'YOLODetector',
    ]
except ImportError:
    pass
