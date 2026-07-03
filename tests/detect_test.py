"""Tests for the object detection module.

Covers parts that don't require the heavy [detect]/[sam] extras: the
training-data builder, accuracy metrics, and the QGIS review round-trip.

Detector-class tests (YOLO / TorchGeo / SAM) are guarded by import
flags and skipped when the optional dependencies are missing.
"""

import os
import tempfile
import unittest
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, box as shapely_box

import geowombat as gw
from geowombat.data import (
    l8_224078_20200518,
    l8_224078_20200518_polygons,
)
from geowombat.detect import (
    boxes_from_polygons,
    build_yolo_dataset,
    detection_accuracy,
    export_for_review,
    recompute_from_review,
)
from geowombat.detect._tiling import overlapped_windows
from geowombat.detect.data import (
    _polygon_to_yolo_aabb,
    _polygon_to_yolo_obb,
    _scale_to_uint8,
)
from geowombat.detect.detectors import _nms_geodataframe

try:
    import PIL  # noqa: F401
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import yaml  # noqa: F401
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import ultralytics  # noqa: F401
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    import torchvision  # noqa: F401
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

try:
    import segment_anything  # noqa: F401
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

# Optional SAM checkpoint path for refiner tests. Skipped unless set
# because the constructor needs to load a real ~375 MB weights file.
SAM_CHECKPOINT = os.environ.get('GEOWOMBAT_SAM_CHECKPOINT')

# Smoke tests that load real YOLO / torchvision weights trigger
# automatic network downloads (~5-160 MB) on first run. They're skipped
# by default and opt-in via this env var so the suite stays
# deterministic and CI-safe.
RUN_DETECTOR_DOWNLOADS = bool(
    os.environ.get('GEOWOMBAT_RUN_DETECTOR_DOWNLOADS')
)


# ---------------------------------------------------------------------------
# boxes_from_polygons
# ---------------------------------------------------------------------------

class TestBoxesFromPolygons(unittest.TestCase):

    def setUp(self):
        polys = [
            Polygon([(0, 0), (4, 0), (4, 3), (0, 3)]),
            Polygon([(10, 10), (12, 10), (12, 14), (10, 14)]),
        ]
        self.gdf = gpd.GeoDataFrame(
            {'cls': ['a', 'b']}, geometry=polys, crs='EPSG:32617',
        )

    def test_aabb_returns_envelopes(self):
        out = boxes_from_polygons(self.gdf, oriented=False)
        self.assertEqual(len(out), 2)
        self.assertTrue((out['_box_kind'] == 'aabb').all())
        # Envelope of an axis-aligned poly is itself
        self.assertTrue(out.geometry.iloc[0].equals(
            self.gdf.geometry.iloc[0].envelope,
        ))

    def test_obb_returns_rotated_rectangles(self):
        rot = Polygon([(0, 0), (3, 1), (2, 4), (-1, 3)])
        gdf = gpd.GeoDataFrame(
            {'cls': ['a']}, geometry=[rot], crs='EPSG:32617',
        )
        out = boxes_from_polygons(gdf, oriented=True)
        self.assertEqual(len(out), 1)
        self.assertEqual(out['_box_kind'].iloc[0], 'obb')
        # OBB should be a 4-corner polygon
        self.assertEqual(
            len(list(out.geometry.iloc[0].exterior.coords)), 5,
        )


# ---------------------------------------------------------------------------
# build_yolo_dataset
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    PIL_AVAILABLE and YAML_AVAILABLE,
    "Pillow/PyYAML not installed",
)
class TestBuildYoloDataset(unittest.TestCase):

    def _label_gdf(self, src):
        polys = gpd.read_file(l8_224078_20200518_polygons)
        if polys.crs.to_epsg() != src.gw.crs_to_pyproj.to_epsg():
            polys = polys.to_crs(src.gw.crs_to_pyproj)
        polys['class_name'] = polys['name']
        return polys

    def test_writes_expected_structure(self):
        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                labels = self._label_gdf(src)
                with tempfile.TemporaryDirectory() as td:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        info = build_yolo_dataset(
                            src,
                            labels,
                            class_col='class_name',
                            out_dir=td,
                            tile_size=128,
                            overlap=0.0,
                            val_split=0.25,
                            min_box_pixels=2,
                            band_indices=[2, 1, 0],
                            scale=(0, 10000),
                            background_ratio=0.0,
                        )
                    out = Path(td)
                    self.assertTrue((out / 'data.yaml').exists())
                    self.assertTrue((out / 'images' / 'train').is_dir())
                    self.assertTrue((out / 'labels' / 'train').is_dir())
                    self.assertGreater(info['n_train'] + info['n_val'], 0)
                    self.assertGreater(info['n_boxes'], 0)
                    # Verify class list ordering
                    self.assertEqual(
                        sorted(info['classes']),
                        sorted(labels['class_name'].unique().tolist()),
                    )

    def test_min_box_pixels_filters_small(self):
        """A tiny min_box_pixels keeps boxes, large value drops them."""
        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                labels = self._label_gdf(src)
                with tempfile.TemporaryDirectory() as td:
                    info_loose = build_yolo_dataset(
                        src, labels, class_col='class_name',
                        out_dir=td, tile_size=128, overlap=0.0,
                        band_indices=[2, 1, 0], scale=(0, 10000),
                        min_box_pixels=1,
                    )
                with tempfile.TemporaryDirectory() as td2:
                    info_strict = build_yolo_dataset(
                        src, labels, class_col='class_name',
                        out_dir=td2, tile_size=128, overlap=0.0,
                        band_indices=[2, 1, 0], scale=(0, 10000),
                        min_box_pixels=200,
                    )
                self.assertGreaterEqual(
                    info_loose['n_boxes'], info_strict['n_boxes'],
                )

    def test_oriented_label_format(self):
        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                labels = self._label_gdf(src)
                with tempfile.TemporaryDirectory() as td:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        build_yolo_dataset(
                            src, labels, class_col='class_name',
                            out_dir=td, tile_size=128, overlap=0.0,
                            band_indices=[2, 1, 0], scale=(0, 10000),
                            oriented=True, min_box_pixels=2,
                        )
                    lbls = list((Path(td) / 'labels' / 'train').glob('*.txt'))
                    non_empty = [p for p in lbls if p.read_text().strip()]
                    if non_empty:
                        sample = non_empty[0].read_text().strip().splitlines()[0]
                        parts = sample.split()
                        # class_id + 8 corner coords
                        self.assertEqual(len(parts), 9)

    def test_background_ratio_retains_empty_tiles(self):
        """background_ratio=1.0 keeps every empty tile; 0.0 drops them all."""
        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                labels = self._label_gdf(src)
                kwargs = dict(
                    class_col='class_name', tile_size=64, overlap=0.0,
                    band_indices=[2, 1, 0], scale=(0, 10000),
                    min_box_pixels=2, val_split=0.0, seed=0,
                )
                with tempfile.TemporaryDirectory() as td_drop:
                    info_drop = build_yolo_dataset(
                        src, labels, out_dir=td_drop,
                        background_ratio=0.0, **kwargs,
                    )
                with tempfile.TemporaryDirectory() as td_keep:
                    info_keep = build_yolo_dataset(
                        src, labels, out_dir=td_keep,
                        background_ratio=1.0, **kwargs,
                    )
        # Same labelled-tile count, but background_ratio=1.0 keeps the
        # empty tiles too — strictly more total training tiles.
        total_drop = info_drop['n_train'] + info_drop['n_val']
        total_keep = info_keep['n_train'] + info_keep['n_val']
        self.assertGreater(total_keep, total_drop)

    def test_class_names_override_writes_yaml_order(self):
        """Passing class_names=[...] pins the order in data.yaml."""
        import yaml as yaml_mod
        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                labels = self._label_gdf(src)
                # Force a specific (non-alphabetical) ordering.
                forced = ['water', 'crop', 'tree', 'developed']
                with tempfile.TemporaryDirectory() as td:
                    info = build_yolo_dataset(
                        src, labels, class_col='class_name',
                        out_dir=td, tile_size=128, overlap=0.0,
                        band_indices=[2, 1, 0], scale=(0, 10000),
                        min_box_pixels=2, class_names=forced,
                    )
                    with open(Path(td) / 'data.yaml') as f:
                        cfg = yaml_mod.safe_load(f)
        self.assertEqual(info['classes'], forced)
        # data.yaml's `names` block should reflect the override order.
        names = cfg['names']
        if isinstance(names, dict):
            ordered = [names[i] for i in sorted(names)]
        else:
            ordered = list(names)
        self.assertEqual(ordered, forced)


# ---------------------------------------------------------------------------
# detection_accuracy
# ---------------------------------------------------------------------------

class TestDetectionAccuracy(unittest.TestCase):

    def _make(self, truths, preds_with_score):
        truth = gpd.GeoDataFrame(
            {'class_name': [c for _, c in truths]},
            geometry=[g for g, _ in truths],
            crs='EPSG:32617',
        )
        pred = gpd.GeoDataFrame(
            {
                'class_name': [c for _, c, _ in preds_with_score],
                'score': [s for _, _, s in preds_with_score],
            },
            geometry=[g for g, _, _ in preds_with_score],
            crs='EPSG:32617',
        )
        return truth, pred

    def test_perfect_match_gives_ap_1(self):
        truth, pred = self._make(
            [(shapely_box(0, 0, 10, 10), 'a'),
             (shapely_box(20, 20, 30, 30), 'a')],
            [(shapely_box(0, 0, 10, 10), 'a', 0.9),
             (shapely_box(20, 20, 30, 30), 'a', 0.8)],
        )
        res = detection_accuracy(pred, truth, iou_thresholds=(0.5,))
        ap = res['metrics'].loc[(0.5, 'a'), 'ap']
        self.assertAlmostEqual(ap, 1.0, places=5)
        self.assertEqual(
            res['metrics'].loc[(0.5, 'a'), 'tp'], 2,
        )
        self.assertEqual(
            res['metrics'].loc[(0.5, 'a'), 'fp'], 0,
        )

    def test_false_positive_counted(self):
        truth, pred = self._make(
            [(shapely_box(0, 0, 10, 10), 'a')],
            [(shapely_box(0, 0, 10, 10), 'a', 0.9),
             (shapely_box(50, 50, 60, 60), 'a', 0.8)],
        )
        res = detection_accuracy(pred, truth, iou_thresholds=(0.5,))
        self.assertEqual(res['metrics'].loc[(0.5, 'a'), 'tp'], 1)
        self.assertEqual(res['metrics'].loc[(0.5, 'a'), 'fp'], 1)

    def test_missed_detection_is_fn(self):
        truth, pred = self._make(
            [(shapely_box(0, 0, 10, 10), 'a'),
             (shapely_box(50, 50, 60, 60), 'a')],
            [(shapely_box(0, 0, 10, 10), 'a', 0.9)],
        )
        res = detection_accuracy(pred, truth, iou_thresholds=(0.5,))
        self.assertEqual(res['metrics'].loc[(0.5, 'a'), 'tp'], 1)
        self.assertEqual(res['metrics'].loc[(0.5, 'a'), 'fn'], 1)

    def test_iou_below_threshold_is_fp(self):
        truth, pred = self._make(
            [(shapely_box(0, 0, 10, 10), 'a')],
            [(shapely_box(8, 8, 18, 18), 'a', 0.9)],
        )
        res = detection_accuracy(pred, truth, iou_thresholds=(0.5,))
        self.assertEqual(res['metrics'].loc[(0.5, 'a'), 'tp'], 0)
        self.assertEqual(res['metrics'].loc[(0.5, 'a'), 'fp'], 1)
        self.assertEqual(res['metrics'].loc[(0.5, 'a'), 'fn'], 1)

    def test_matched_geodataframe_has_review_columns(self):
        truth, pred = self._make(
            [(shapely_box(0, 0, 10, 10), 'a')],
            [(shapely_box(0, 0, 10, 10), 'a', 0.9)],
        )
        res = detection_accuracy(pred, truth, iou_thresholds=(0.5,))
        matched = res['matched']
        for col in ['status', 'iou', 'reviewer_label', 'notes']:
            self.assertIn(col, matched.columns)
        self.assertEqual(matched['status'].iloc[0], 'TP')

    def test_class_confusion_detected(self):
        truth, pred = self._make(
            [(shapely_box(0, 0, 10, 10), 'a')],
            [(shapely_box(0, 0, 10, 10), 'b', 0.9)],
        )
        res = detection_accuracy(pred, truth, iou_thresholds=(0.5,))
        # The prediction is a FP for class b and FN for class a.
        # Status FP_class flags the same-location class mismatch.
        statuses = res['matched']['status'].tolist()
        self.assertIn('FP_class', statuses)
        self.assertIn('FN', statuses)

    def test_coco_thresholds_emits_50_95_summary(self):
        """iou_thresholds='coco' expands to 0.5..0.95 and adds mAP@[.5:.95]."""
        truth, pred = self._make(
            [(shapely_box(0, 0, 10, 10), 'a')],
            [(shapely_box(0, 0, 10, 10), 'a', 0.9)],
        )
        res = detection_accuracy(pred, truth, iou_thresholds='coco')
        summary = res['summary']
        self.assertIn('mAP@[.5:.95]', summary)
        # 10 thresholds 0.5, 0.55, ..., 0.95
        per_thr_keys = [k for k in summary
                        if k.startswith('mAP@') and k != 'mAP@[.5:.95]']
        self.assertEqual(len(per_thr_keys), 10)
        # Perfect alignment → 1.0 across the board.
        self.assertAlmostEqual(summary['mAP@[.5:.95]'], 1.0, places=5)

    def test_class_agnostic_ignores_class_labels(self):
        """class_agnostic=True matches any-class pred to any-class truth."""
        truth, pred = self._make(
            [(shapely_box(0, 0, 10, 10), 'a')],
            # Same geometry but a different class label
            [(shapely_box(0, 0, 10, 10), 'b', 0.9)],
        )
        # With classes considered, this is FP_class + FN (see test above).
        res_classed = detection_accuracy(pred, truth, iou_thresholds=(0.5,))
        self.assertIn('FP_class', res_classed['matched']['status'].tolist())
        # With class_agnostic=True it's a clean TP.
        res_agn = detection_accuracy(
            pred, truth, iou_thresholds=(0.5,), class_agnostic=True,
        )
        self.assertEqual(res_agn['metrics'].loc[(0.5, '_all_'), 'tp'], 1)
        self.assertEqual(res_agn['metrics'].loc[(0.5, '_all_'), 'fp'], 0)
        self.assertEqual(res_agn['metrics'].loc[(0.5, '_all_'), 'fn'], 0)

    def test_crs_mismatch_reprojected_before_matching(self):
        """Truth in EPSG:4326 + preds in EPSG:32617 still match correctly."""
        truth, pred = self._make(
            [(shapely_box(500000, 4000000, 500100, 4000100), 'a')],
            [(shapely_box(500000, 4000000, 500100, 4000100), 'a', 0.9)],
        )
        # Reproject only the truth to a different CRS.
        truth_4326 = truth.to_crs('EPSG:4326')
        res = detection_accuracy(pred, truth_4326, iou_thresholds=(0.5,))
        # After reprojection the matched IoU is very close to 1.0 — small
        # imprecision from the 4326 round-trip is acceptable.
        self.assertEqual(res['metrics'].loc[(0.5, 'a'), 'tp'], 1)
        self.assertEqual(res['metrics'].loc[(0.5, 'a'), 'fp'], 0)


# ---------------------------------------------------------------------------
# QGIS review round-trip
# ---------------------------------------------------------------------------

class TestReviewRoundTrip(unittest.TestCase):

    def test_export_and_recompute_from_review(self):
        truth = gpd.GeoDataFrame(
            {'class_name': ['a', 'a', 'a']},
            geometry=[
                shapely_box(0, 0, 10, 10),
                shapely_box(20, 20, 30, 30),
                shapely_box(40, 40, 50, 50),
            ],
            crs='EPSG:32617',
        )
        pred = gpd.GeoDataFrame(
            {'class_name': ['a', 'a', 'a'], 'score': [0.9, 0.8, 0.7]},
            geometry=[
                shapely_box(0, 0, 10, 10),       # TP
                shapely_box(20, 20, 30, 30),     # TP
                shapely_box(100, 100, 110, 110),  # FP
            ],
            crs='EPSG:32617',
        )
        res = detection_accuracy(pred, truth, iou_thresholds=(0.5,))
        matched = res['matched']
        # Override one of the FPs to TP via reviewer_label
        fp_mask = matched['status'] == 'FP'
        if fp_mask.any():
            idx = matched.index[fp_mask][0]
            matched.loc[idx, 'reviewer_label'] = 'TP'

        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / 'review.gpkg'
            export_for_review(matched, out_path)
            self.assertTrue(out_path.exists())

            final = recompute_from_review(out_path)
            overall = final['overall']
            # We flipped one FP→TP, so tp=3, fp=0
            self.assertEqual(overall['tp'], 3)
            self.assertEqual(overall['fp'], 0)


# ---------------------------------------------------------------------------
# Detector smoke tests — only when ultralytics is installed
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    ULTRALYTICS_AVAILABLE,
    "ultralytics not installed (pip install geowombat[detect])",
)
class TestYOLODetectorSmoke(unittest.TestCase):

    @unittest.skipUnless(
        RUN_DETECTOR_DOWNLOADS,
        "set GEOWOMBAT_RUN_DETECTOR_DOWNLOADS=1 to allow yolov8n.pt download",
    )
    def test_predict_returns_geodataframe(self):
        from geowombat.detect import YOLODetector

        # yolov8n.pt is auto-downloaded by ultralytics on first use.
        det = YOLODetector(weights='yolov8n.pt')
        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gdf = det.predict(
                        src,
                        tile_size=320,
                        overlap=0.0,
                        conf=0.05,
                        band_indices=[2, 1, 0],
                        scale=(0, 10000),
                    )
        self.assertIsInstance(gdf, gpd.GeoDataFrame)
        for col in ['geometry', 'class_id', 'class_name', 'score']:
            self.assertIn(col, gdf.columns)

    def test_oriented_auto_detected_from_filename(self):
        """`-obb.pt` filenames flip self.oriented to True without an explicit kwarg."""
        from unittest.mock import patch, MagicMock
        from geowombat.detect import YOLODetector

        fake_model = MagicMock()
        fake_model.names = {0: 'plane'}
        # Don't actually load weights — stub ultralytics.YOLO.
        with patch('ultralytics.YOLO', return_value=fake_model):
            det_aabb = YOLODetector(weights='yolov8n.pt')
            det_obb = YOLODetector(weights='yolov8n-obb.pt')
            det_explicit = YOLODetector(
                weights='yolov8n.pt', oriented=True,
            )
        self.assertFalse(det_aabb.oriented)
        self.assertTrue(det_obb.oriented)
        # Explicit kwarg wins over filename heuristic.
        self.assertTrue(det_explicit.oriented)


# ---------------------------------------------------------------------------
# Band resolver + new gw.ml.* + .gw.* surface
# ---------------------------------------------------------------------------

class TestBandResolver(unittest.TestCase):

    def test_explicit_override_wins(self):
        from geowombat.ml._labels import resolve_band_indices
        with gw.open(l8_224078_20200518, nodata=0) as src:
            self.assertEqual(
                resolve_band_indices(src, [2, 1, 0]), [2, 1, 0],
            )

    def test_sensor_bgr_picks_rgb_indices(self):
        from geowombat.ml._labels import resolve_band_indices
        with gw.config.update(sensor='bgr'):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                self.assertEqual(
                    resolve_band_indices(src), [2, 1, 0],
                )

    def test_unnamed_bands_default_first_three(self):
        from geowombat.ml._labels import resolve_band_indices
        with gw.open(l8_224078_20200518, nodata=0) as src:
            self.assertEqual(resolve_band_indices(src), [0, 1, 2])


@unittest.skipUnless(
    PIL_AVAILABLE and YAML_AVAILABLE,
    "Pillow + PyYAML required",
)
class TestAccessorAndModuleSurface(unittest.TestCase):
    """The .gw.to_yolo_dataset / gw.ml.build_detection_dataset wrappers
    should be drop-in equivalent to build_yolo_dataset(src, ...)."""

    def _label_gdf(self, src):
        polys = gpd.read_file(l8_224078_20200518_polygons)
        if polys.crs.to_epsg() != src.gw.crs_to_pyproj.to_epsg():
            polys = polys.to_crs(src.gw.crs_to_pyproj)
        polys['class_name'] = polys['name']
        return polys

    def test_accessor_to_yolo_dataset_matches_function(self):
        from geowombat.detect import build_dataset

        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                labels = self._label_gdf(src)
                kwargs = dict(
                    class_col='class_name', tile_size=128, overlap=0.0,
                    band_indices=[2, 1, 0], scale=(0, 10000),
                    min_box_pixels=2, background_ratio=0.0,
                )
                with tempfile.TemporaryDirectory() as td1:
                    info_fn = build_dataset(
                        src, labels, out_dir=td1, **kwargs,
                    )
                with tempfile.TemporaryDirectory() as td2:
                    info_acc = src.gw.to_yolo_dataset(
                        labels, out_dir=td2, **kwargs,
                    )

        self.assertEqual(info_fn['classes'], info_acc['classes'])
        self.assertEqual(info_fn['n_boxes'], info_acc['n_boxes'])

    def test_sensor_config_drives_band_indices(self):
        """to_yolo_dataset without band_indices uses sensor config."""
        from geowombat.detect import build_dataset

        with gw.config.update(ref_res=300, sensor='bgr'):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                labels = self._label_gdf(src)
                with tempfile.TemporaryDirectory() as td:
                    info = build_dataset(
                        src, labels, class_col='class_name',
                        out_dir=td, tile_size=128, overlap=0.0,
                        scale=(0, 10000), min_box_pixels=2,
                    )
        self.assertGreater(info['n_boxes'], 0)


@unittest.skipUnless(
    ULTRALYTICS_AVAILABLE,
    "ultralytics not installed (pip install geowombat[detect])",
)
class TestDetectAccessor(unittest.TestCase):

    def test_gw_detect_matches_predict(self):
        from geowombat.detect import YOLODetector, predict

        det = YOLODetector(weights='yolov8n.pt')
        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kwargs = dict(
                        tile_size=320, overlap=0.0, conf=0.05,
                        band_indices=[2, 1, 0], scale=(0, 10000),
                    )
                    a = det.predict(src, **kwargs)
                    b = src.gw.detect(det, **kwargs)
                    c = predict(src, det, **kwargs)
        self.assertEqual(len(a), len(b))
        self.assertEqual(len(a), len(c))


# ---------------------------------------------------------------------------
# Tiling: overlapped_windows
# ---------------------------------------------------------------------------

class TestTilingWindows(unittest.TestCase):

    def test_zero_overlap_covers_whole_image(self):
        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                wins = list(overlapped_windows(src, tile_size=64, overlap=0.0))
        self.assertGreater(len(wins), 0)
        # Every tile should be the full tile_size (last-tile-shifted-back).
        for _, _, w in wins:
            self.assertLessEqual(w.width, 64)
            self.assertLessEqual(w.height, 64)
        # No tile should exceed image bounds.
        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                for _, _, w in wins:
                    self.assertLessEqual(w.col_off + w.width, src.gw.ncols)
                    self.assertLessEqual(w.row_off + w.height, src.gw.nrows)

    def test_overlap_yields_more_tiles(self):
        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                no = list(overlapped_windows(src, tile_size=64, overlap=0.0))
                hi = list(overlapped_windows(src, tile_size=64, overlap=0.5))
        # Higher overlap → strictly more (or equal) tiles.
        self.assertGreaterEqual(len(hi), len(no))

    def test_last_tile_shifted_back_for_non_divisible(self):
        """When image isn't a multiple of tile_size, the last tile shifts
        backwards so it still fits — never overflows."""
        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                # Pick a tile_size that won't divide the image evenly.
                w = src.gw.ncols
                h = src.gw.nrows
                tile = max(1, min(w, h) - 7)
                wins = list(overlapped_windows(src, tile_size=tile, overlap=0.0))
                # The last column's right edge equals image width.
                last_col = max(w_.col_off + w_.width for _, _, w_ in wins)
                last_row = max(w_.row_off + w_.height for _, _, w_ in wins)
                self.assertEqual(last_col, w)
                self.assertEqual(last_row, h)


# ---------------------------------------------------------------------------
# YOLO label math (polygon → normalized label)
# ---------------------------------------------------------------------------

class TestYoloLabelMath(unittest.TestCase):
    """Direct unit tests for the pixel math in _polygon_to_yolo_aabb /
    _polygon_to_yolo_obb. Subtle bugs here would silently corrupt the
    training data on disk, so we test them in isolation rather than only
    through build_yolo_dataset."""

    def test_aabb_inside_tile_normalizes_correctly(self):
        # tile_xmin=0, tile_ymax=10 (UL corner in CRS), 1m/px, tile 10 px.
        poly = shapely_box(0, 0, 5, 5)  # CRS coords
        out = _polygon_to_yolo_aabb(
            poly, tile_xmin=0, tile_ymax=10,
            cellx=1.0, celly=1.0, tile_size=10,
        )
        self.assertIsNotNone(out)
        cx_n, cy_n, w_n, h_n, w_px, h_px = out
        # Box covers pixels (0..5, 5..10) → center (2.5, 7.5) → 0.25, 0.75.
        self.assertAlmostEqual(cx_n, 0.25)
        self.assertAlmostEqual(cy_n, 0.75)
        self.assertAlmostEqual(w_n, 0.5)
        self.assertAlmostEqual(h_n, 0.5)
        self.assertAlmostEqual(w_px, 5.0)
        self.assertAlmostEqual(h_px, 5.0)

    def test_aabb_clipped_at_tile_edge(self):
        """Polygon extending past the right edge is clipped before normalizing."""
        # Polygon spans CRS x = [8, 15]; tile only covers [0, 10].
        poly = shapely_box(8, 0, 15, 5)
        out = _polygon_to_yolo_aabb(
            poly, tile_xmin=0, tile_ymax=10,
            cellx=1.0, celly=1.0, tile_size=10,
        )
        self.assertIsNotNone(out)
        _, _, w_n, _, w_px, _ = out
        # Clipped width is 2 px, not 7 px.
        self.assertAlmostEqual(w_px, 2.0)
        self.assertAlmostEqual(w_n, 0.2)

    def test_aabb_outside_tile_returns_none(self):
        poly = shapely_box(20, 20, 25, 25)  # entirely past the tile
        out = _polygon_to_yolo_aabb(
            poly, tile_xmin=0, tile_ymax=10,
            cellx=1.0, celly=1.0, tile_size=10,
        )
        self.assertIsNone(out)

    def test_obb_inside_tile_returns_eight_normalized_coords(self):
        # Rotated 4-corner polygon
        rot = Polygon([(2, 2), (5, 3), (4, 6), (1, 5)])
        out = _polygon_to_yolo_obb(
            rot, tile_xmin=0, tile_ymax=10,
            cellx=1.0, celly=1.0, tile_size=10,
        )
        self.assertIsNotNone(out)
        # 8 corner coordinates + width + height
        self.assertEqual(len(out), 10)
        coords = out[:8]
        # All normalized coords should fall in [0, 1].
        for c in coords:
            self.assertGreaterEqual(c, 0.0)
            self.assertLessEqual(c, 1.0)

    def test_obb_outside_tile_returns_none(self):
        rot = Polygon([(100, 100), (105, 100), (105, 105), (100, 105)])
        out = _polygon_to_yolo_obb(
            rot, tile_xmin=0, tile_ymax=10,
            cellx=1.0, celly=1.0, tile_size=10,
        )
        self.assertIsNone(out)


# ---------------------------------------------------------------------------
# _scale_to_uint8
# ---------------------------------------------------------------------------

class TestScaleToUint8(unittest.TestCase):

    def test_linear_range_maps_endpoints(self):
        arr = np.array([[[0, 5000, 10000]]], dtype=np.float32)
        out = _scale_to_uint8(arr, scale=(0, 10000))
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out[0, 0, 0], 0)
        self.assertEqual(out[0, 0, 2], 255)
        # Midpoint maps to ~127.
        self.assertIn(out[0, 0, 1], (127, 128))

    def test_percentile_stretch_with_scale_none(self):
        rng = np.random.default_rng(0)
        arr = rng.integers(100, 200, size=(1, 32, 32)).astype(np.float32)
        out = _scale_to_uint8(arr, scale=None)
        self.assertEqual(out.dtype, np.uint8)
        # 2-98 percentile stretch → covers most of [0, 255].
        self.assertLess(out.min(), 20)
        self.assertGreater(out.max(), 235)

    def test_constant_input_does_not_divide_by_zero(self):
        """hi == lo (constant array) is the historically-risky edge case."""
        arr = np.full((1, 4, 4), 42.0, dtype=np.float32)
        out = _scale_to_uint8(arr, scale=(42, 42))
        # Should not raise and should produce a valid uint8 array.
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, (1, 4, 4))


# ---------------------------------------------------------------------------
# Cross-tile NMS
# ---------------------------------------------------------------------------

class TestNMS(unittest.TestCase):

    def _gdf(self, rows):
        return gpd.GeoDataFrame(
            {
                'class_id': [r['class_id'] for r in rows],
                'score': [r['score'] for r in rows],
            },
            geometry=[r['geom'] for r in rows],
            crs='EPSG:32617',
        )

    def test_overlapping_same_class_collapses_to_highest_score(self):
        gdf = self._gdf([
            {'class_id': 0, 'score': 0.9, 'geom': shapely_box(0, 0, 10, 10)},
            # ~80% IoU with the first
            {'class_id': 0, 'score': 0.5, 'geom': shapely_box(1, 1, 11, 11)},
        ])
        out = _nms_geodataframe(gdf, iou_threshold=0.5)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out['score'].iloc[0], 0.9)

    def test_overlapping_different_classes_both_kept(self):
        gdf = self._gdf([
            {'class_id': 0, 'score': 0.9, 'geom': shapely_box(0, 0, 10, 10)},
            {'class_id': 1, 'score': 0.5, 'geom': shapely_box(1, 1, 11, 11)},
        ])
        out = _nms_geodataframe(gdf, iou_threshold=0.5)
        self.assertEqual(len(out), 2)

    def test_disjoint_boxes_all_kept(self):
        gdf = self._gdf([
            {'class_id': 0, 'score': 0.9, 'geom': shapely_box(0, 0, 5, 5)},
            {'class_id': 0, 'score': 0.8, 'geom': shapely_box(20, 20, 25, 25)},
            {'class_id': 0, 'score': 0.7, 'geom': shapely_box(40, 40, 45, 45)},
        ])
        out = _nms_geodataframe(gdf, iou_threshold=0.5)
        self.assertEqual(len(out), 3)

    def test_empty_input_returns_empty(self):
        empty = gpd.GeoDataFrame(
            {'class_id': [], 'score': []}, geometry=[], crs='EPSG:32617',
        )
        out = _nms_geodataframe(empty, iou_threshold=0.5)
        self.assertTrue(out.empty)


# ---------------------------------------------------------------------------
# plot_detections (smoke)
# ---------------------------------------------------------------------------

class TestPlotDetections(unittest.TestCase):

    def test_plot_detections_returns_axes(self):
        # Use the non-interactive Agg backend so this works on a headless box.
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        from geowombat.detect import plot_detections

        with gw.config.update(ref_res=300):
            with gw.open(l8_224078_20200518, nodata=0) as src:
                truth = gpd.read_file(l8_224078_20200518_polygons)
                if truth.crs.to_epsg() != src.gw.crs_to_pyproj.to_epsg():
                    truth = truth.to_crs(src.gw.crs_to_pyproj)
                # Build a synthetic "matched" set: half of the truth as TP preds.
                preds = truth.head(2).copy()
                preds['score'] = 0.9
                preds['status'] = 'TP'
                fig, ax = plt.subplots(figsize=(4, 4))
                returned = plot_detections(
                    src, predictions=preds, truth=truth,
                    ax=ax, band_indices=[2, 1, 0], scale=(0, 10000),
                )
        self.assertIsNotNone(returned)
        self.assertTrue(hasattr(returned, 'plot'))
        plt.close('all')


# ---------------------------------------------------------------------------
# TorchGeoDetector smoke (requires torchvision)
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    TORCHVISION_AVAILABLE and RUN_DETECTOR_DOWNLOADS,
    "torchvision required + GEOWOMBAT_RUN_DETECTOR_DOWNLOADS=1 "
    "to allow torchvision COCO weight download",
)
class TestTorchGeoDetectorSmoke(unittest.TestCase):

    def test_predict_returns_geodataframe(self):
        from geowombat.detect import TorchGeoDetector

        # Default torchvision COCO weights (auto-downloaded on first use).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            det = TorchGeoDetector(model='faster-rcnn')
            with gw.config.update(ref_res=300):
                with gw.open(l8_224078_20200518, nodata=0) as src:
                    gdf = det.predict(
                        src,
                        tile_size=320,
                        overlap=0.0,
                        conf=0.05,
                        band_indices=[2, 1, 0],
                        scale=(0, 10000),
                    )
        self.assertIsInstance(gdf, gpd.GeoDataFrame)
        for col in ['geometry', 'class_id', 'class_name', 'score']:
            self.assertIn(col, gdf.columns)


# ---------------------------------------------------------------------------
# SAMRefiner (skipped unless a checkpoint path is exported)
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    SAM_AVAILABLE and SAM_CHECKPOINT and Path(SAM_CHECKPOINT).exists(),
    "segment-anything + GEOWOMBAT_SAM_CHECKPOINT env var required",
)
class TestSAMRefiner(unittest.TestCase):

    def test_empty_input_short_circuits(self):
        """refine() returns an empty GeoDataFrame without invoking SAM."""
        from geowombat.detect import SAMRefiner
        refiner = SAMRefiner(checkpoint=SAM_CHECKPOINT, model_type='vit_b')
        with gw.open(l8_224078_20200518, nodata=0) as src:
            empty = gpd.GeoDataFrame(
                {'class_name': [], 'score': []},
                geometry=[],
                crs=src.gw.crs_to_pyproj,
            )
            out = refiner.refine(src, empty)
        self.assertTrue(out.empty)


# ---------------------------------------------------------------------------
# fit_predict end-to-end smoke (1 epoch on bundled data)
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    (
        ULTRALYTICS_AVAILABLE
        and PIL_AVAILABLE
        and YAML_AVAILABLE
        and RUN_DETECTOR_DOWNLOADS
    ),
    "ultralytics + Pillow + PyYAML required + "
    "GEOWOMBAT_RUN_DETECTOR_DOWNLOADS=1 to allow yolov8n.pt download",
)
class TestFitPredict(unittest.TestCase):

    def test_fit_predict_runs_end_to_end(self):
        """Build a tiny dataset, fine-tune for 1 epoch, run inference."""
        from geowombat.detect import YOLODetector, fit_predict

        with tempfile.TemporaryDirectory() as td:
            det = YOLODetector(weights='yolov8n.pt')
            with gw.config.update(ref_res=300):
                with gw.open(l8_224078_20200518, nodata=0) as src:
                    polys = gpd.read_file(l8_224078_20200518_polygons)
                    if polys.crs.to_epsg() != src.gw.crs_to_pyproj.to_epsg():
                        polys = polys.to_crs(src.gw.crs_to_pyproj)
                    polys['class_name'] = polys['name']
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        preds, summary = fit_predict(
                            src,
                            det,
                            polys,
                            class_col='class_name',
                            out_dir=Path(td) / 'ds',
                            tile_size=128,
                            overlap=0.0,
                            epochs=1,
                            min_box_pixels=2,
                            band_indices=[2, 1, 0],
                            scale=(0, 10000),
                            val_split=0.5,
                            seed=42,
                            predict_kwargs={'conf': 0.05},
                        )
        # The summary echoes the build_dataset return.
        self.assertIn('classes', summary)
        self.assertIn('n_boxes', summary)
        # preds is a GeoDataFrame (may be empty — that's OK for a 1-epoch run).
        self.assertIsInstance(preds, gpd.GeoDataFrame)


# ---------------------------------------------------------------------------
# prepare_label_gdf — regression tests for null-class handling
# ---------------------------------------------------------------------------

class TestPrepareLabelGdfNullClass(unittest.TestCase):
    """Pin behavior when ``labels[class_col]`` contains null values.

    The pre-fix code did::

        classes = sorted(labels[class_col].dropna().unique().tolist())
        labels['_class_id'] = labels[class_col].map(name_to_id).astype(int)

    which mapped the full (un-dropped) column, producing NaN for null
    rows, then crashed at ``.astype(int)`` with
    ``ValueError: cannot convert float NaN to integer``.
    """

    def _build_labels_in_src_crs(self, src, class_values):
        """Build a GeoDataFrame with one polygon per class value,
        all inside the raster footprint, in the raster's CRS.
        """
        bounds = src.gw.bounds  # (left, bottom, right, top)
        # tile the polygons across a row near the bottom of the raster
        n = len(class_values)
        width = (bounds[2] - bounds[0]) / (n + 2)
        height = (bounds[3] - bounds[1]) / 20
        geoms = [
            shapely_box(
                bounds[0] + (i + 1) * width,
                bounds[1] + height,
                bounds[0] + (i + 1.5) * width,
                bounds[1] + 2 * height,
            )
            for i in range(n)
        ]
        return gpd.GeoDataFrame(
            {'cls': class_values},
            geometry=geoms,
            crs=src.gw.crs_to_pyproj,
        )

    def test_nulls_dropped_with_warning_when_class_names_none(self):
        """Null class_col values are dropped (warning) instead of crashing.

        Pre-fix this would raise ``ValueError: cannot convert float NaN
        to integer`` at the ``.astype(int)`` step.
        """
        from geowombat.ml._labels import prepare_label_gdf

        with gw.open(l8_224078_20200518, nodata=0) as src:
            labels = self._build_labels_in_src_crs(
                src, class_values=['tree', None, 'building', None, 'road'],
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                prepared, classes = prepare_label_gdf(
                    src, labels, class_col='cls',
                )

        # 3 non-null rows survived; 2 nulls dropped with a warning.
        self.assertEqual(len(prepared), 3)
        # _class_id is integer-typed, no NaN survivors.
        self.assertTrue(
            np.issubdtype(prepared['_class_id'].dtype, np.integer),
            f"_class_id dtype={prepared['_class_id'].dtype}",
        )
        self.assertFalse(prepared['_class_id'].isna().any())
        # classes derived from the surviving rows only, sorted.
        self.assertEqual(classes, ['building', 'road', 'tree'])
        # One warning mentioning the null count.
        null_warnings = [
            w for w in caught if 'null' in str(w.message).lower()
        ]
        self.assertEqual(len(null_warnings), 1, [str(w.message) for w in caught])
        self.assertIn('2', str(null_warnings[0].message))

    def test_no_nulls_no_warning(self):
        """Clean input: no null-class warning, full set passes through."""
        from geowombat.ml._labels import prepare_label_gdf

        with gw.open(l8_224078_20200518, nodata=0) as src:
            labels = self._build_labels_in_src_crs(
                src, class_values=['tree', 'building', 'road'],
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                prepared, classes = prepare_label_gdf(
                    src, labels, class_col='cls',
                )

        self.assertEqual(len(prepared), 3)
        self.assertEqual(classes, ['building', 'road', 'tree'])
        null_warnings = [
            w for w in caught if 'null' in str(w.message).lower()
        ]
        self.assertEqual(null_warnings, [])

    def test_null_and_empty_geometries_dropped(self):
        """Rows with None or empty geometry are dropped (with a warning)
        before any spatial op runs — otherwise reprojection / intersects
        would crash later."""
        from geowombat.ml._labels import prepare_label_gdf
        from shapely.geometry import Polygon

        with gw.open(l8_224078_20200518, nodata=0) as src:
            labels = self._build_labels_in_src_crs(
                src, class_values=['tree', 'building', 'road'],
            )
            # Mutate one to None and one to an empty Polygon.
            labels.loc[0, labels.geometry.name] = None
            labels.loc[1, labels.geometry.name] = Polygon()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                prepared, classes = prepare_label_gdf(
                    src, labels, class_col='cls',
                )

        # Only the third (real) row survives.
        self.assertEqual(len(prepared), 1)
        self.assertEqual(classes, ['road'])
        geom_warnings = [
            w for w in caught
            if 'null or empty geometry' in str(w.message)
        ]
        self.assertEqual(len(geom_warnings), 1, [str(w.message) for w in caught])
        # Message reports the count.
        self.assertIn('2', str(geom_warnings[0].message))

    def test_invalid_geometries_repaired_via_make_valid(self):
        """Self-intersecting (bowtie) polygons are repaired in place,
        not silently passed downstream where they'd break .intersects()."""
        from geowombat.ml._labels import prepare_label_gdf
        from shapely.geometry import Polygon

        with gw.open(l8_224078_20200518, nodata=0) as src:
            labels = self._build_labels_in_src_crs(
                src, class_values=['tree', 'building'],
            )
            # Replace the first geometry with a classic self-intersecting
            # bowtie, in the raster's CRS and inside its footprint.
            bounds = src.gw.bounds
            cx = (bounds[0] + bounds[2]) / 2
            cy = (bounds[1] + bounds[3]) / 2
            half = 50.0
            bowtie = Polygon([
                (cx - half, cy - half),
                (cx + half, cy + half),
                (cx + half, cy - half),
                (cx - half, cy + half),
                (cx - half, cy - half),
            ])
            self.assertFalse(bowtie.is_valid)
            labels.loc[0, labels.geometry.name] = bowtie

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                prepared, classes = prepare_label_gdf(
                    src, labels, class_col='cls',
                )

        # Both rows survive — the bowtie was repaired, not dropped.
        self.assertEqual(len(prepared), 2)
        # And every surviving geometry is now valid.
        self.assertTrue(prepared.geometry.is_valid.all())
        invalid_warnings = [
            w for w in caught
            if 'invalid geometry' in str(w.message)
        ]
        self.assertEqual(len(invalid_warnings), 1, [str(w.message) for w in caught])
        # Message records the repair count.
        self.assertIn('repaired', str(invalid_warnings[0].message).lower())
        self.assertIn('1 repaired', str(invalid_warnings[0].message))
        # No 'unrepairable' or 'dropped' wording for a fully-repaired case.
        self.assertNotIn('unrepairable', str(invalid_warnings[0].message).lower())

    def test_unrepairable_geometry_dropped(self):
        """A geometry that make_valid can't salvage is dropped, surviving
        rows continue, and the warning reports both repaired and dropped
        counts.

        shapely 2.x's ``make_valid`` is robust enough that almost any
        real-world input repairs to something usable, so we
        monkey-patch the import inside ``_labels`` to return an empty
        geometry for one specific input — this exercises the
        ``n_lost > 0`` branch we actually want to test.
        """
        from unittest.mock import patch
        from geowombat.ml import _labels as labels_mod
        from geowombat.ml._labels import prepare_label_gdf
        from shapely.geometry import Polygon

        with gw.open(l8_224078_20200518, nodata=0) as src:
            labels = self._build_labels_in_src_crs(
                src, class_values=['tree', 'building', 'road'],
            )
            bounds = src.gw.bounds
            cx = (bounds[0] + bounds[2]) / 2
            cy = (bounds[1] + bounds[3]) / 2
            half = 50.0
            bowtie_a = Polygon([
                (cx - half, cy - half),
                (cx + half, cy + half),
                (cx + half, cy - half),
                (cx - half, cy + half),
                (cx - half, cy - half),
            ])
            bowtie_b = Polygon([
                (cx + 2 * half, cy - half),
                (cx + 4 * half, cy + half),
                (cx + 4 * half, cy - half),
                (cx + 2 * half, cy + half),
                (cx + 2 * half, cy - half),
            ])
            self.assertFalse(bowtie_a.is_valid)
            self.assertFalse(bowtie_b.is_valid)
            # Row 0 and 1 both start invalid. We force ``make_valid`` to
            # repair the first but return an empty Polygon for the second.
            labels.loc[0, labels.geometry.name] = bowtie_a
            labels.loc[1, labels.geometry.name] = bowtie_b

            real_make_valid = labels_mod.make_valid

            def fake_make_valid(geom):
                if geom.equals(bowtie_b):
                    return Polygon()  # unrepairable -> empty
                return real_make_valid(geom)

            with patch.object(labels_mod, 'make_valid', new=fake_make_valid):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    prepared, classes = prepare_label_gdf(
                        src, labels, class_col='cls',
                    )

        # tree (bowtie_a, repaired) + road (clean) survive; building drops.
        self.assertEqual(len(prepared), 2)
        self.assertEqual(sorted(classes), ['road', 'tree'])
        self.assertTrue(prepared.geometry.is_valid.all())
        self.assertFalse(prepared.geometry.is_empty.any())
        invalid_warnings = [
            w for w in caught
            if 'invalid geometry' in str(w.message)
        ]
        self.assertEqual(len(invalid_warnings), 1)
        msg = str(invalid_warnings[0].message)
        # Both counts are surfaced.
        self.assertIn('1 repaired', msg)
        self.assertIn('1 unrepairable', msg)


# ---------------------------------------------------------------------------
# TorchGeoDetector._coco_names — full 91-entry torchvision COCO list
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    TORCHVISION_AVAILABLE,
    "torchvision not installed (pip install geowombat[detect,dl])",
)
class TestCocoNamesFullList(unittest.TestCase):
    """Pin the canonical torchvision COCO label list.

    torchvision's COCO label index uses the original 91-id space
    (with N/A gaps for unused ids: 12, 26, 29, 30, 45, 66, 68, 69,
    71, 83). Pre-fix the list only had 27 entries, so any cls_id
    above 25 fell through to ``str(class_id)`` in
    ``GeoWombatDetector.predict``.
    """

    def test_length_is_91(self):
        from geowombat.detect.detectors import TorchGeoDetector
        self.assertEqual(len(TorchGeoDetector._coco_names()), 91)

    def test_canonical_indices(self):
        """A handful of well-known COCO ids resolve to the right names."""
        from geowombat.detect.detectors import TorchGeoDetector
        names = TorchGeoDetector._coco_names()
        # Background + a sample drawn from across the full range
        # (early, mid, late) so a future truncation regression is caught.
        self.assertEqual(names[0], '__background__')
        self.assertEqual(names[1], 'person')
        self.assertEqual(names[16], 'bird')
        self.assertEqual(names[44], 'bottle')
        self.assertEqual(names[64], 'potted plant')
        self.assertEqual(names[88], 'teddy bear')
        self.assertEqual(names[90], 'toothbrush')

    def test_na_gaps_in_canonical_positions(self):
        """Unused COCO ids are 'N/A' so cls_id indexing stays aligned."""
        from geowombat.detect.detectors import TorchGeoDetector
        names = TorchGeoDetector._coco_names()
        for gap in (12, 26, 29, 30, 45, 66, 68, 69, 71, 83):
            self.assertEqual(
                names[gap], 'N/A',
                f"id {gap} should be 'N/A' (got {names[gap]!r})",
            )


if __name__ == '__main__':
    unittest.main()
