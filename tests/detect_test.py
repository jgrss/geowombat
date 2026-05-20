"""Tests for the object detection module.

Covers parts that don't require the heavy [detect]/[sam] extras: the
training-data builder, accuracy metrics, and the QGIS review round-trip.

Detector-class tests (YOLO / TorchGeo / SAM) are guarded by import
flags and skipped when the optional dependencies are missing.
"""

import tempfile
import unittest
import warnings
from pathlib import Path

import geopandas as gpd
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


if __name__ == '__main__':
    unittest.main()
