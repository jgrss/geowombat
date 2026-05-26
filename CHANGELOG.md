# Changelog

<!-- version list -->

## Unreleased

### Features

* Added `geowombat.detect` submodule for tiled, georeferenced object detection on overhead imagery, mirroring the `fit / predict / fit_predict` shape of `geowombat.ml`:
  * `YOLODetector` (Ultralytics YOLO) supporting axis-aligned (AABB) and DOTA-v1 oriented (OBB) boxes.
  * `TorchGeoDetector` wrapping TorchVision Faster R-CNN / RetinaNet with optional TorchGeo pretrained weights (e.g. `FASTERRCNN_RESNET50_FPN_XVIEW`).
  * `SAMRefiner` for refining bounding-box detections into polygon masks via Meta's Segment Anything Model.
* Added geowombat-native accessors: `src.gw.detect(detector, ...)` (tiled inference with cross-tile NMS and pixel→CRS conversion) and `src.gw.to_yolo_dataset(labels, class_col=..., out_dir=...)` (Ultralytics-layout training-dataset builder).
* Added module-level wrappers: `predict`, `fit`, `fit_predict`, `build_dataset`, `boxes_from_polygons` (AABB ↔ OBB), `detection_accuracy` (per-class precision/recall/F1/AP at one or more IoU thresholds, plus `iou_thresholds='coco'` for mAP@[.5:.95] and `class_agnostic=True`), `plot_detections`, and `export_for_review` / `recompute_from_review` for QGIS GeoPackage round-trip review.
* Sensor-aware band selection: when `gw.config.update(sensor=...)` is active, detection accessors auto-resolve RGB band indices from `src.band.values` — no need to pass `band_indices` per call.
* Added deep-learning classifiers `TabNet`, `L-TAE`, and `TorchGeo` ([#347](https://github.com/jgrss/geowombat/pull/347)).

### Documentation

* New object-detection walkthrough at `doc/source/object-detection.rst` with a "Recommended setup for aerial / satellite imagery" callout (DOTA-v1 OBB weights + `oriented=True`), the AABB vs OBB explainer, and a "Digitizing polygons for high-quality OBB labels" subsection.
* Added a **Notebooks** chapter rendering all 7 Jupyter notebooks (`open_plot`, `mosaic_ndvi_mask`, `moving_windows`, `stac`, `ml_classifiers`, `dl_classifiers`, `object_detection`) inline via `nbsphinx`. Notebooks live in `notebooks/` and are symlinked into the docs tree; outputs are baked in, so the docs build never re-executes.
* Updated install guide with conda extras, GDAL instructions, and the `dl` extra ([#349](https://github.com/jgrss/geowombat/pull/349)).

### Bug Fixes

* Improved block usage and nodata handling ([#348](https://github.com/jgrss/geowombat/pull/348)).
* Restored multi-threaded warp via the existing `num_threads` parameter without GDAL warning spam. Threading is now routed through `rio.Env(GDAL_NUM_THREADS=str(num_threads))` (opened only when `num_threads > 1`) instead of the rasterio `warp_extras` / `multi` `WarpedVRT` kwargs, which produce `Warning 6: warp options does not support option WARP_EXTRAS / MULTI` per `WarpedVRT` construction on rasterio>=1.4 / GDAL>=3.12. Default (`num_threads=1`) leaves the GDAL env untouched, so a user's outer `rio.Env(GDAL_NUM_THREADS=...)` flows through unmodified.
* Fixed NAIP STAC fast path (`_open_stac_multiband_asset`) to always use the `'image'` asset key. The previous logic derived `asset_key` from `bands[0]`, which raised `KeyError` whenever callers passed band names like `bands=['red', 'green', 'blue']` — NAIP exposes a single multi-band COG under `'image'`, not per-band assets. Band selection by name now happens after opening (via `gw.config(sensor='naip')`), not via STAC asset-key lookup.
* Fixed `prepare_label_gdf` (`geowombat.ml._labels`) crash on null `class_col` values. The pre-fix code mapped the full column then cast to `int`, raising `IntCastingNaNError`; now null-class rows are dropped with a warning before integer encoding, matching the symmetric behavior of the `class_names`-supplied branch.
* Hardened `prepare_label_gdf` against malformed input geometries: null and empty geometries are dropped with a warning before any spatial operation, and invalid geometries (e.g. self-intersecting / bowtie polygons) are first repaired via `shapely.validation.make_valid` — only geometries that remain unusable after repair are dropped, and the warning surfaces both the repaired count and the dropped count.
* Fixed `TorchGeoDetector._coco_names()` truncation. The class-name list was only 27 entries, so any `cls_id >= 26` from torchvision DEFAULT weights fell through to `str(class_id)` in `predict()`. Expanded to the full 91-entry torchvision COCO label list (with N/A gaps at unused ids 12, 26, 29, 30, 45, 66, 68, 69, 71, 83) so `cls_id` indexes correctly.

### Build

* Added `nbsphinx`, `ipykernel`, and `myst-parser` to the `[docs]` extras so the docs build can render the notebook chapter and embed `CHANGELOG.md`.
* Added new optional extras: `[detect]` (ultralytics, torchmetrics, pycocotools, pillow, pyyaml, opencv-python) and `[sam]` (segment-anything).

### Testing

* Added 29 new tests for the detect module covering: tiled-inference window math (`overlapped_windows`), polygon → YOLO label math for AABB and OBB, `_scale_to_uint8` edge cases, cross-tile NMS, IoU/COCO/class-agnostic accuracy modes, CRS reprojection in accuracy assessment, QGIS review round-trip, smoke tests for `YOLODetector` / `TorchGeoDetector` / `SAMRefiner` / `plot_detections`, and an end-to-end `fit_predict` smoke test.
* Added `tests/test_threading.py` (4 tests) pinning the `rio.Env(GDAL_NUM_THREADS=...)` wiring: no GDAL warning spam at default or `num_threads=4`, and verified `rio.Env` composition so a user's outer env merges with geowombat's inner env without overriding.
* Added regression tests for the NAIP `asset_key` fix (3, in `tests/test_stac_mocked.py`), `prepare_label_gdf` null-class + null/invalid-geometry handling (5, in `tests/detect_test.py`), and `TorchGeoDetector._coco_names()` length / canonical-indices / N/A-gap layout (3). All new regression tests have been verified to fail under the pre-fix code.
* Gated the YOLO and torchvision smoke tests behind a `GEOWOMBAT_RUN_DETECTOR_DOWNLOADS=1` environment variable so the default suite stays deterministic and offline-safe; opt-in only when local weight downloads (~5–160 MB) are acceptable.

## v2.1.23 (2026-01-12)

### Features

* Migrated to Meson build backend ([#334](https://github.com/jgrss/geowombat/pull/334))
* Added HLS (Harmonized Landsat Sentinel-2) and ESA WorldCover collections for Microsoft STAC catalog ([#333](https://github.com/jgrss/geowombat/pull/333))
* Added mocked STAC tests to reduce flaky CI failures from external API timeouts

### Bug Fixes

* Fixed NaN values in mosaic overlap regions ([#322](https://github.com/jgrss/geowombat/issues/322))
* Fixed mosaic `bounds_by='union'` functionality ([#328](https://github.com/jgrss/geowombat/issues/328))
* Fixed sklearn 1.6+ compatibility for clusterer detection ([#331](https://github.com/jgrss/geowombat/issues/331))
* Fixed non-unique band names when using `gw.open(..., stack_dim='band')` without `band_names` ([#316](https://github.com/jgrss/geowombat/issues/316), [#317](https://github.com/jgrss/geowombat/pull/317))
* Fixed `RasterBlockError` when saving with non-multiple-of-16 chunk sizes ([#237](https://github.com/jgrss/geowombat/issues/237))
* Fixed `is_tiled` deprecation warning from rasterio by using `block_shapes` check
* Pinned `rasterio<1.5.0` to avoid edge case in warp behavior
* Pinned `h5py<3.10.0` to fix netCDF dimension scales issue

### CI/Build

* Updated CI configuration with uv package manager
* Added Python 3.12 support in CI testing

## v2.1.22 (2024-05-01)
* Added Python 3.11 as option ([#311](https://github.com/jgrss/geowombat/pull/311))

## v2.1.21 (2024-05-01)
* Pinned `sklearn` version for 'ml' extra ([#310](https://github.com/jgrss/geowombat/pull/310))

## v2.1.20 (2024-04-30)
* ([#313](https://github.com/jgrss/geowombat/pull/313))
* Added all file geometries to DataArray mosaic
* Fixed import error
* Fixed Sphinx doc builds

## v2.1.19 (2024-04-26)
* ([#306](https://github.com/jgrss/geowombat/pull/306))
* Support for multi-band mosaics
* Fixed bug in `to_raster()`, where nodata values were not being written to file
* Added 'bigtiff' keyword argument to `save()` method

## v2.1.18 (2024-04-20)
* Fixed bug when saving single-band mosaics ([#304](https://github.com/jgrss/geowombat/pull/304))
  
## v2.1.17 (2024-01-13)
* Pinned scikit-learn maximum version ([#301](https://github.com/jgrss/geowombat/pull/301))

## v2.1.16 (2024-01-12)
* Increased minimum `pyproj` version ([#297](https://github.com/jgrss/geowombat/pull/297))

## v2.1.15 (2023-11-30)
* Added BIGTIFF option in series `apply()` ([#295](https://github.com/jgrss/geowombat/pull/295))

## v2.1.14 (2023-11-28)
* Fixed error in EVI equation ([#294](https://github.com/jgrss/geowombat/pull/294))

## v2.1.13 (2023-11-08)
* Fixed multi-threaded timeout raised when downloading STAC extras ([#292](https://github.com/jgrss/geowombat/pull/292))

## v2.1.12 (2023-11-03)
* Updated STAC catalogs ([#289](https://github.com/jgrss/geowombat/pull/289))

## v2.1.11 (2023-10-02)
* Added ``DataArray`` alignment check ([#282](https://github.com/jgrss/geowombat/pull/282))

## v2.1.10 (2023-07-09)
* Added support for Python 3.10 ([#277](https://github.com/jgrss/geowombat/pull/277))

## v2.1.9 (2023-05-03)
* Fixed STAC `DataFrame` concatenation in ([#267](https://github.com/jgrss/geowombat/pull/267))
* Fixed issue with saving small arrays in ([#268](https://github.com/jgrss/geowombat/pull/268))

## v2.1.8 (2023-04-19)
* Removed nested multithreading in `geowombat.moving` ([#259](https://github.com/jgrss/geowombat/pull/259))

## v2.1.7 (2023-04-01)
* [#249](https://github.com/jgrss/geowombat/issues/249) merged in [#250](https://github.com/jgrss/geowombat/pull/250)
* `pyproj>3.4.0` WKT errors addressed in [#254](https://github.com/jgrss/geowombat/pull/254)
* CONTRIBUTING.md guide from [#253](https://github.com/jgrss/geowombat/pull/253)
* New docs theme from [#256](https://github.com/jgrss/geowombat/pull/256)
* Added test coverage in [#255](https://github.com/jgrss/geowombat/pull/255)

## v2.1.6 (2023-03-03)
* Pinned maximum Cython version to <3.0.0 ([#247](https://github.com/jgrss/geowombat/pull/247))

## v2.1.5 (2023-03-02)
* Fixed issue with overwriting existing files on Windows ([#244](https://github.com/jgrss/geowombat/pull/244))

## v2.1.4 (2023-02-28)
* Fixed error in .pxd compilation ([#243](https://github.com/jgrss/geowombat/pull/243))

## v2.1.3 (2023-02-08)
* Fixed typos in temporal aggregation ([#241](https://github.com/jgrss/geowombat/pull/241))

## v2.1.2 (2023-02-06)
* Fixed `zoom` dimensions ([#239](https://github.com/jgrss/geowombat/pull/239))

## v2.1.1 (2023-01-31)
* Pinned required packages to full semantic versions ([#238](https://github.com/jgrss/geowombat/pull/238))

## v2.1.0 (2023-01-04)
* Added weight argument to BRDF module ([#234](https://github.com/jgrss/geowombat/pull/234))

## v2.0.19 (2022-12-21)
* Added support for `sdist` ([#233](https://github.com/jgrss/geowombat/pull/233))
* Merged tarfile safety patch ([#234](https://github.com/jgrss/geowombat/pull/234))

## v2.0.18 (2022-12-19)
* Removed OpenCV dependency ([#232](https://github.com/jgrss/geowombat/pull/232))

## v2.0.17 (2022-10-11)
* Pinned dask version and fixed issue with Landsat angle creation ([#228](https://github.com/jgrss/geowombat/pull/228))

## v2.0.16 (2022-10-05)
* Pinned dependency `threadpoolctl` to a minimum version ([#227](https://github.com/jgrss/geowombat/pull/227))

## v2.0.15 (2022-10-03)
* Fixed issue with polygon point extraction, which caused a one pixel shift north and west of the polygon top left bounds ([#225](https://github.com/jgrss/geowombat/pull/225))

## v2.0.14 (2022-09-29)
* Fixed property errors exposed in online doc builds ([#223](https://github.com/jgrss/geowombat/pull/223))

## v2.0.13 (2022-09-29)
* Changed 'stac' extra requirement to include forked repository of `stackstac` ([#222](https://github.com/jgrss/geowombat/pull/222))

## v2.0.12 (2022-09-28)
* Fixed issue with `scale_factor` and attribute setting ([#221](https://github.com/jgrss/geowombat/pull/221))

## v2.0.11 (2022-09-27)
* Fixed GDAL install issue with 'coreg' extra ([#217](https://github.com/jgrss/geowombat/pull/217))

## v2.0.10 (2022-09-24)
* Improved ``DataArray`` attributes, modified extra dependencies, added STAC and co-reg tests, and improved online documentation ([#205](https://github.com/jgrss/geowombat/pull/205))

## v2.0.9 (2022-09-21)
* Added `nodataval` `DataArray` property to ML workflow ([#209](https://github.com/jgrss/geowombat/pull/209))

## v2.0.8 (2022-09-21)
* Added a `nodataval` `DataArray` property ([#208](https://github.com/jgrss/geowombat/pull/208))

## v2.0.7 (2022-09-20)
* Fixed ML `nodata` and `dtype` ([#207](https://github.com/jgrss/geowombat/pull/207))

## v2.0.6 (2022-09-19)
* Changed behavior of `nodata` values ([#204](https://github.com/jgrss/geowombat/pull/204))

## v2.0.5 (2022-09-16)
* Fixed attribute lookup when co-registration is applied ([#203](https://github.com/jgrss/geowombat/pull/203))

## v2.0.4 (2022-09-15)
* Fixed 'filename' attribute when opening a NetCDF file ([#201](https://github.com/jgrss/geowombat/pull/201))

## v2.0.3 (2022-09-14)
* Pinned Python >= 3.8 in `setup.cfg` ([#200](https://github.com/jgrss/geowombat/pull/200))

## v2.0.2 (2022-09-13)
* Added CRS to WKT transformation for co-registration ([#199](https://github.com/jgrss/geowombat/pull/199))

## v2.0.1 (2022-09-13)
* Fixed ML tests ([#198](https://github.com/jgrss/geowombat/pull/198))

## v2.0.0 (2022-09-01)
* Added `geowombat.save()` method ([#189](https://github.com/jgrss/geowombat/pull/189))
* Warping methods now return `dask.Delayed` objects ([#189](https://github.com/jgrss/geowombat/pull/189))
* Better CRS checks ([#189](https://github.com/jgrss/geowombat/pull/189))

## v1.11.4 (2022-08-31)
* Fixed `to_raster()` ([#187](https://github.com/jgrss/geowombat/pull/187))

## v1.11.3 (2022-07-10)
* Added user proj bounds to return more specific bbox ([#180](https://github.com/jgrss/geowombat/issues/180))

## v1.11.2 (2022-07-10)
* Fixed CRS errors generated from the readthedocs build ([#178](https://github.com/jgrss/geowombat/issues/178))

## v1.11.1 (2022-07-09)
* Fixed CRS translation error of certain EPSG codes ([#177](https://github.com/jgrss/geowombat/issues/177))

## v1.11.0 (2022-07-09)
* Added Landsat 9 to metadata lookup

## v1.10.1 (2022-07-07)
* Removed imports from `geowombat.__init__`

## v1.10.0 (2022-07-07)
* Add STAC API to read Landsat and Sentinel-2 time series

## v1.9.1 (2022-06-18)
* Added support for Landsat 9

## v1.8.6 (2022-06-12)
### Fix
* Added token ([#148](https://github.com/jgrss/geowombat/issues/148)) ([`79b0243`](https://github.com/jgrss/geowombat/commit/79b0243df5765865ef913ab42b911960649ec511))
* Removed semantic version header ([#147](https://github.com/jgrss/geowombat/issues/147)) ([`529b02b`](https://github.com/jgrss/geowombat/commit/529b02bcf128ab31eecf52a7f2067626461cc6b7))
* Test github action release ([#146](https://github.com/jgrss/geowombat/issues/146)) ([`f99f6de`](https://github.com/jgrss/geowombat/commit/f99f6de714dcf355dca3cb82126c7fa4ff65952a))
* Pin min requests version ([#143](https://github.com/jgrss/geowombat/issues/143)) ([`98ad33a`](https://github.com/jgrss/geowombat/commit/98ad33aa15474d88f7396c32f765a33d7265021f))

## v1.8.5 (2022-05-31)
### Fix
* Jgrss/dependencies ([#134](https://github.com/jgrss/geowombat/issues/134)) ([`342bb2b`](https://github.com/jgrss/geowombat/commit/342bb2b518350ac1617dcca3329b9645862c17c9))

## v1.8.4 (2022-05-23)
### Fix
* Changed upload to PyPI to GitHub releases ([#113](https://github.com/jgrss/geowombat/issues/113)) ([`378f8ec`](https://github.com/jgrss/geowombat/commit/378f8ecd6671c6451d87e7d1949967a29f448be0))

## v1.8.3 (2022-05-23)
### Fix
* Added documentation describing how nodata is applied ([#110](https://github.com/jgrss/geowombat/issues/110)) ([`8bd7d3d`](https://github.com/jgrss/geowombat/commit/8bd7d3dc8cd6c1d8a3a3d8dbc391300ad7602a99))

### Documentation
* Replaced version badge ([#109](https://github.com/jgrss/geowombat/issues/109)) ([`537386d`](https://github.com/jgrss/geowombat/commit/537386df4daa4c8cfc567b75db12b555a957d5e8))

## v1.8.2 (2022-05-22)
### Fix
* Added exit ([#108](https://github.com/jgrss/geowombat/issues/108)) ([`000d6fd`](https://github.com/jgrss/geowombat/commit/000d6fd35828ea1625e068b3343a23bd98743987))

## v1.8.1 (2022-05-22)
### Fix
* Added wheel ([#107](https://github.com/jgrss/geowombat/issues/107)) ([`ce86863`](https://github.com/jgrss/geowombat/commit/ce8686389a4a6f94cc441d35c523c8db68057791))

## v1.8.0 (2022-05-22)
### Feature
* Merge pull request #98 from jgrss/semantic-release ([`25aa5f3`](https://github.com/jgrss/geowombat/commit/25aa5f3c0920ae8591578f30998d4aa65010b43a))

### Fix
* Test fingerprint ([#106](https://github.com/jgrss/geowombat/issues/106)) ([`d8919cc`](https://github.com/jgrss/geowombat/commit/d8919cce5e9a4d9cc0a7f13ff600a4c6c79b6f53))
* Comment ([#105](https://github.com/jgrss/geowombat/issues/105)) ([`92e642e`](https://github.com/jgrss/geowombat/commit/92e642e7c5bbc64a6d5cead59bb1237dfddc6d7b))
* Small change ([#104](https://github.com/jgrss/geowombat/issues/104)) ([`5c32d7e`](https://github.com/jgrss/geowombat/commit/5c32d7eeb92b53ff2041a5ce8c8121e835979dcd))
* Small change ([#103](https://github.com/jgrss/geowombat/issues/103)) ([`3512e90`](https://github.com/jgrss/geowombat/commit/3512e901b2dd0f15886651190bd85b9d0ca4e9f6))
* Added git config ([#102](https://github.com/jgrss/geowombat/issues/102)) ([`3e77a4d`](https://github.com/jgrss/geowombat/commit/3e77a4def2b8e3997becdb003ee245bd6b42e8a2))
* Added token ([#101](https://github.com/jgrss/geowombat/issues/101)) ([`67e9bb3`](https://github.com/jgrss/geowombat/commit/67e9bb3263be3d3250fc242461582bc218c605f2))
* Release ([#100](https://github.com/jgrss/geowombat/issues/100)) ([`1e3c0c8`](https://github.com/jgrss/geowombat/commit/1e3c0c862173bd8bc553771b149c966e73f2d3ae))
* Merge pull request #99 from jgrss/semantic2 ([`daf469b`](https://github.com/jgrss/geowombat/commit/daf469ba177c29ec413fa86b76148776c5f415ed))
