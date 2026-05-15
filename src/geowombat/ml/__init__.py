from .classifiers import Classifiers

_classifier = Classifiers()

fit = _classifier.fit
fit_predict = _classifier.fit_predict
predict = _classifier.predict

__all__ = ['fit', 'fit_predict', 'predict']

try:
    from .dl_classifiers import (
        TabNetClassifier,
        LTAEClassifier,
        TorchGeoClassifier,
    )
    __all__ += [
        'TabNetClassifier',
        'LTAEClassifier',
        'TorchGeoClassifier',
    ]
except ImportError:
    pass

# Object detection: training-data builders + metrics import without
# requiring the heavy [detect]/[sam] extras (they only need shapely,
# geopandas, pandas, numpy, matplotlib). Detector classes themselves
# import lazily inside their constructors.
try:
    from .detection_data import boxes_from_polygons, build_yolo_dataset
    from .detection_metrics import (
        detection_accuracy,
        export_for_review,
        recompute_from_review,
        plot_detections,
    )
    __all__ += [
        'boxes_from_polygons',
        'build_yolo_dataset',
        'detection_accuracy',
        'export_for_review',
        'recompute_from_review',
        'plot_detections',
    ]
except ImportError:
    pass

try:
    from .detectors import (
        YOLODetector,
        TorchGeoDetector,
        SAMRefiner,
    )
    __all__ += [
        'YOLODetector',
        'TorchGeoDetector',
        'SAMRefiner',
    ]
except ImportError:
    pass
