"""Module-level functional wrappers for object detection.

Mirrors the ``gw.ml.fit / predict / fit_predict`` API shape so detection
feels at home next to classification:

>>> import geowombat as gw
>>> from geowombat.detect import YOLODetector, predict
>>> det = YOLODetector(weights='yolov8n.pt')
>>> with gw.open('aerial.tif') as src:
...     preds = predict(src, det, conf=0.25)
"""

from .data import build_yolo_dataset


def predict(src, detector, **kwargs):
    """Run tiled, georeferenced inference over a raster.

    Thin wrapper around ``detector.predict(src, **kwargs)`` so detection
    follows the same module-level call shape as ``gw.ml.predict``.

    Parameters
    ----------
    src : xarray.DataArray
        Raster opened with ``gw.open()``.
    detector : YOLODetector or TorchGeoDetector
        Pre-built detector instance.
    **kwargs
        Forwarded to ``detector.predict`` (``tile_size``, ``overlap``,
        ``conf``, ``band_indices``, ``scale``, ``nms_iou``, ``max_det``,
        ``progress``).

    Returns
    -------
    geopandas.GeoDataFrame
        Detections in the source CRS.
    """
    return detector.predict(src, **kwargs)


def fit(detector, dataset_yaml, **kwargs):
    """Fine-tune a detector on a YOLO-format dataset.

    Parameters
    ----------
    detector : YOLODetector
        Detector to fine-tune (only YOLO supports ``.fit`` today).
    dataset_yaml : str or Path
        Path to ``data.yaml`` written by ``build_dataset``.
    **kwargs
        Forwarded to ``detector.fit`` (``epochs``, ``imgsz``, ...).

    Returns
    -------
    The underlying training results object.
    """
    if not hasattr(detector, 'fit'):
        raise AttributeError(
            f"{type(detector).__name__} does not support .fit(). "
            "Use YOLODetector for fine-tuning."
        )
    return detector.fit(dataset_yaml, **kwargs)


def fit_predict(
    src,
    detector,
    labels,
    class_col,
    out_dir,
    tile_size=640,
    overlap=0.1,
    epochs=50,
    predict_kwargs=None,
    **dataset_kwargs,
):
    """Build a training dataset, fine-tune, and predict in one call.

    Mirrors ``gw.ml.fit_predict`` for classification: end-to-end from
    raster + labels to predictions.

    Parameters
    ----------
    src : xarray.DataArray
        Raster opened with ``gw.open()``.
    detector : YOLODetector
        Detector to fine-tune and run inference with.
    labels : geopandas.GeoDataFrame, str, or Path
        Vector labels.
    class_col : str
        Column in ``labels`` holding class name/id.
    out_dir : str or Path
        Output directory for the generated YOLO dataset.
    tile_size : int
        Tile edge in pixels. Default 640.
    overlap : float
        Tile overlap for both dataset creation and inference. Default 0.1.
    epochs : int
        Fine-tuning epochs. Default 50.
    predict_kwargs : dict, optional
        Extra kwargs passed to ``detector.predict``.
    **dataset_kwargs
        Extra kwargs passed to ``build_dataset`` (e.g. ``val_split``,
        ``min_box_pixels``, ``background_ratio``, ``band_indices``,
        ``scale``, ``oriented``).

    Returns
    -------
    (geopandas.GeoDataFrame, dict)
        Predictions and the dataset-build summary.
    """
    summary = build_yolo_dataset(
        src,
        labels=labels,
        class_col=class_col,
        out_dir=out_dir,
        tile_size=tile_size,
        overlap=overlap,
        **dataset_kwargs,
    )
    fit(
        detector,
        dataset_yaml=f"{summary['out_dir']}/data.yaml",
        epochs=epochs,
        imgsz=tile_size,
    )
    preds = detector.predict(
        src,
        tile_size=tile_size,
        overlap=overlap,
        **(predict_kwargs or {}),
    )
    return preds, summary


# Canonical name for the dataset builder inside this module. The
# underlying implementation (``build_yolo_dataset``) is also exported
# from ``geowombat.detect.data`` for callers who already use that name.
build_dataset = build_yolo_dataset
