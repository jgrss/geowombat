"""Accuracy assessment for object detection.

Computes IoU-matched precision/recall/F1/AP per class and produces a
review-ready GeoDataFrame suitable for stepping through in QGIS (e.g.
with the GoToNextFeature3+ plugin or the built-in Form view) to verify
TP / FP / FN judgments manually before reporting final metrics.

Example
-------
>>> from geowombat.ml.detection_metrics import (
...     detection_accuracy, export_for_review, recompute_from_review,
... )
>>> results = detection_accuracy(pred_gdf, truth_gdf, class_col='species')
>>> results['metrics']               # per-class precision/recall/F1/AP
>>> export_for_review(results['matched'], 'review.gpkg')
>>> # ... edit reviewer_label in QGIS, then ...
>>> final = recompute_from_review('review.gpkg', class_col='species')
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon


# ---------------------------------------------------------------------------
# IoU + matching
# ---------------------------------------------------------------------------

def _iou(a, b):
    if a.is_empty or b.is_empty:
        return 0.0
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _greedy_match(preds, truths, iou_threshold):
    """Greedy IoU matching of predictions to ground truth.

    Predictions are processed in descending score order. Each truth can
    only be matched once. Returns three lists of indices:
    ``(tp_pairs, fp_idx, fn_idx)`` where ``tp_pairs`` are
    ``(pred_idx, truth_idx, iou)``.
    """
    if preds.empty and truths.empty:
        return [], [], []
    if preds.empty:
        return [], [], list(truths.index)
    if truths.empty:
        return [], list(preds.index), []

    pred_order = preds.sort_values('score', ascending=False).index.tolist()
    truth_geoms = {i: g for i, g in truths.geometry.items()}
    matched_truth = set()

    tp = []
    fp = []
    for pi in pred_order:
        pg = preds.geometry.loc[pi]
        best_iou = 0.0
        best_ti = None
        for ti, tg in truth_geoms.items():
            if ti in matched_truth:
                continue
            i = _iou(pg, tg)
            if i > best_iou:
                best_iou = i
                best_ti = ti
        if best_ti is not None and best_iou >= iou_threshold:
            tp.append((pi, best_ti, best_iou))
            matched_truth.add(best_ti)
        else:
            fp.append(pi)

    fn = [ti for ti in truths.index if ti not in matched_truth]
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Per-class AP
# ---------------------------------------------------------------------------

def _voc_ap(recall, precision):
    """11-point interpolated PASCAL VOC AP."""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = precision[recall >= t]
        ap += (p.max() if p.size else 0.0) / 11.0
    return float(ap)


def _ap_for_class(class_preds, class_truths, iou_threshold):
    if class_truths.empty:
        return {
            'ap': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'tp': 0, 'fp': len(class_preds), 'fn': 0,
            'support': 0,
        }
    if class_preds.empty:
        return {
            'ap': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'tp': 0, 'fp': 0, 'fn': len(class_truths),
            'support': len(class_truths),
        }

    order = class_preds.sort_values('score', ascending=False).index.tolist()
    matched = set()
    tps = np.zeros(len(order), dtype=np.int32)
    fps = np.zeros(len(order), dtype=np.int32)

    for i, pi in enumerate(order):
        pg = class_preds.geometry.loc[pi]
        best_iou = 0.0
        best_ti = None
        for ti, tg in class_truths.geometry.items():
            if ti in matched:
                continue
            iou = _iou(pg, tg)
            if iou > best_iou:
                best_iou = iou
                best_ti = ti
        if best_ti is not None and best_iou >= iou_threshold:
            tps[i] = 1
            matched.add(best_ti)
        else:
            fps[i] = 1

    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)
    n_truth = len(class_truths)
    recall = cum_tp / max(n_truth, 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    ap = _voc_ap(recall, precision)

    tp_total = int(cum_tp[-1])
    fp_total = int(cum_fp[-1])
    fn_total = n_truth - tp_total
    p = tp_total / max(tp_total + fp_total, 1)
    r = tp_total / max(n_truth, 1)
    f1 = 2 * p * r / max(p + r, 1e-12)

    return {
        'ap': float(ap),
        'precision': float(p),
        'recall': float(r),
        'f1': float(f1),
        'tp': tp_total,
        'fp': fp_total,
        'fn': fn_total,
        'support': n_truth,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detection_accuracy(
    predictions,
    truth,
    class_col='class_name',
    iou_thresholds=(0.5,),
    score_col='score',
    class_agnostic=False,
):
    """Compute detection accuracy metrics + a review-ready GeoDataFrame.

    Parameters
    ----------
    predictions : geopandas.GeoDataFrame
        Detector output. Must include ``geometry``, ``score`` (or
        ``score_col``), and a class column.
    truth : geopandas.GeoDataFrame
        Ground-truth boxes/polygons in the same CRS.
    class_col : str
        Column with class labels in both inputs.
    iou_thresholds : sequence of float, or 'coco'
        IoU thresholds to evaluate. ``'coco'`` expands to 0.5..0.95
        step 0.05 and returns mAP@[.5:.95].
    score_col : str
        Score column in predictions. Default 'score'.
    class_agnostic : bool
        If True, ignore class labels (treat as a single class).

    Returns
    -------
    dict
        Keys:
          - ``metrics``: DataFrame indexed by class with columns
            ``precision, recall, f1, ap, tp, fp, fn, support`` for each
            IoU threshold.
          - ``summary``: dict of overall mAP per threshold + mAP@[.5:.95].
          - ``matched``: GeoDataFrame of TP/FP/FN rows ready for QGIS
            review (see ``export_for_review``).
          - ``confusion``: DataFrame of class confusion (per-class
            truth vs. matched-prediction class) at the lowest IoU.
    """
    if predictions.crs is None or truth.crs is None:
        raise ValueError("Both predictions and truth must have a CRS set.")
    if predictions.crs.to_epsg() != truth.crs.to_epsg():
        truth = truth.to_crs(predictions.crs)

    # Reset to a simple 0..N integer index — osmnx and other sources can
    # return MultiIndex GeoDataFrames whose keys aren't castable to int.
    preds = predictions.reset_index(drop=True).copy()
    truths = truth.reset_index(drop=True).copy()

    if score_col not in preds.columns:
        raise KeyError(
            f"predictions missing score column '{score_col}'."
        )
    if class_agnostic:
        preds['_cls'] = '_all_'
        truths['_cls'] = '_all_'
        class_col_use = '_cls'
    else:
        if class_col not in preds.columns:
            raise KeyError(
                f"predictions missing class column '{class_col}'."
            )
        if class_col not in truths.columns:
            raise KeyError(
                f"truth missing class column '{class_col}'."
            )
        class_col_use = class_col

    preds = preds.rename(columns={score_col: 'score'})

    if iou_thresholds == 'coco':
        thresholds = list(np.round(np.arange(0.5, 1.0, 0.05), 2))
    else:
        thresholds = list(iou_thresholds)

    classes = sorted(
        set(preds[class_col_use].dropna().unique().tolist())
        | set(truths[class_col_use].dropna().unique().tolist())
    )

    rows = []
    for thr in thresholds:
        for cls in classes:
            cp = preds[preds[class_col_use] == cls]
            ct = truths[truths[class_col_use] == cls]
            stats = _ap_for_class(cp, ct, thr)
            stats.update({'class': cls, 'iou_threshold': thr})
            rows.append(stats)
    metrics_df = pd.DataFrame(rows).set_index(['iou_threshold', 'class'])

    summary = {}
    for thr in thresholds:
        ap_vals = metrics_df.loc[thr, 'ap'].values
        summary[f'mAP@{thr}'] = float(np.mean(ap_vals)) if len(ap_vals) else 0.0
    if iou_thresholds == 'coco':
        summary['mAP@[.5:.95]'] = float(np.mean(list(summary.values())))

    # ------------------------------------------------------------------
    # Build review GeoDataFrame at the lowest IoU threshold — that's the
    # one humans typically audit (most lenient pairing).
    # ------------------------------------------------------------------
    primary_iou = min(thresholds)
    review_rows = []
    confusion_rows = []

    # Per-class greedy matching to fill TP/FP/FN with class context
    for cls in classes:
        cp = preds[preds[class_col_use] == cls]
        ct = truths[truths[class_col_use] == cls]
        tp_pairs, fp_idx, fn_idx = _greedy_match(cp, ct, primary_iou)
        for pi, ti, iou_val in tp_pairs:
            review_rows.append({
                'geometry': preds.geometry.loc[pi],
                'status': 'TP',
                'class_name': cls,
                'truth_class': cls,
                'score': float(preds['score'].loc[pi]),
                'iou': float(iou_val),
                'pred_idx': int(pi),
                'truth_idx': int(ti),
                'reviewer_label': '',
                'notes': '',
            })
            confusion_rows.append((cls, cls))
        for pi in fp_idx:
            # Check whether the FP overlaps a truth of a different class
            pg = preds.geometry.loc[pi]
            best_other = None
            best_iou = 0.0
            for ti, tg in truths.geometry.items():
                if truths[class_col_use].loc[ti] == cls:
                    continue
                v = _iou(pg, tg)
                if v > best_iou:
                    best_iou = v
                    best_other = ti
            if best_other is not None and best_iou >= primary_iou:
                truth_cls = truths[class_col_use].loc[best_other]
                review_rows.append({
                    'geometry': pg,
                    'status': 'FP_class',
                    'class_name': cls,
                    'truth_class': truth_cls,
                    'score': float(preds['score'].loc[pi]),
                    'iou': float(best_iou),
                    'pred_idx': int(pi),
                    'truth_idx': int(best_other),
                    'reviewer_label': '',
                    'notes': '',
                })
                confusion_rows.append((truth_cls, cls))
            else:
                review_rows.append({
                    'geometry': pg,
                    'status': 'FP',
                    'class_name': cls,
                    'truth_class': None,
                    'score': float(preds['score'].loc[pi]),
                    'iou': float(best_iou),
                    'pred_idx': int(pi),
                    'truth_idx': -1,
                    'reviewer_label': '',
                    'notes': '',
                })
        for ti in fn_idx:
            review_rows.append({
                'geometry': truths.geometry.loc[ti],
                'status': 'FN',
                'class_name': None,
                'truth_class': cls,
                'score': np.nan,
                'iou': 0.0,
                'pred_idx': -1,
                'truth_idx': int(ti),
                'reviewer_label': '',
                'notes': '',
            })
            confusion_rows.append((cls, '_missed_'))

    matched_gdf = gpd.GeoDataFrame(
        review_rows, geometry='geometry', crs=preds.crs,
    )

    # Confusion matrix (truth vs. predicted class label among matched)
    if confusion_rows:
        cm = pd.crosstab(
            pd.Series([r[0] for r in confusion_rows], name='truth'),
            pd.Series([r[1] for r in confusion_rows], name='predicted'),
        )
    else:
        cm = pd.DataFrame()

    return {
        'metrics': metrics_df,
        'summary': summary,
        'matched': matched_gdf,
        'confusion': cm,
    }


def export_for_review(matched_gdf, out_path, layer='detections_review'):
    """Write a matched GeoDataFrame to GeoPackage for QGIS review.

    The output file is suitable for the QGIS attribute form workflow and
    feature-stepping plugins (e.g. GoToNextFeature3+). The
    ``reviewer_label`` field is empty for the user to fill in (values
    like ``'TP'``, ``'FP'``, ``'FN'``, ``'unclear'`` are recommended);
    ``notes`` is free text.

    Parameters
    ----------
    matched_gdf : geopandas.GeoDataFrame
        Output from ``detection_accuracy``.
    out_path : str or Path
        Destination .gpkg path.
    layer : str
        GeoPackage layer name.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    matched_gdf.to_file(out_path, layer=layer, driver='GPKG')
    return out_path


def recompute_from_review(review_input, class_col='class_name',
                          layer='detections_review'):
    """Recompute metrics after a human has edited ``reviewer_label``.

    The reviewer is expected to fill ``reviewer_label`` with one of:
    ``'TP'``, ``'FP'``, ``'FN'``, or ``'unclear'`` (case-insensitive).
    Rows where the label is blank are treated as the automatic status.
    ``'unclear'`` rows are excluded from the metrics entirely.

    Parameters
    ----------
    review_input : str, Path, or GeoDataFrame
        Reviewed GeoPackage path or in-memory GeoDataFrame.
    class_col : str
        Column to use for per-class breakdown.
    layer : str
        GeoPackage layer if reading from disk.

    Returns
    -------
    dict
        Per-class and overall counts/precision/recall/F1 after
        incorporating human review.
    """
    if isinstance(review_input, (str, Path)):
        gdf = gpd.read_file(review_input, layer=layer)
    else:
        gdf = review_input.copy()

    label_col = 'reviewer_label'
    if label_col not in gdf.columns:
        raise KeyError(
            f"GeoDataFrame missing '{label_col}' column."
        )

    final = gdf['status'].astype(str).str.upper().copy()
    reviewer = gdf[label_col].astype(str).str.strip().str.upper()
    has_review = reviewer.isin(['TP', 'FP', 'FN', 'UNCLEAR'])
    final[has_review] = reviewer[has_review]
    # Collapse FP_class into FP for top-line counts
    final = final.replace({'FP_CLASS': 'FP'})

    gdf = gdf.assign(_final=final)
    gdf = gdf[gdf['_final'] != 'UNCLEAR']

    # Decide which class column to bucket by for FN rows
    def _row_class(row):
        if row['_final'] == 'FN':
            return row.get('truth_class') or row.get(class_col)
        return row.get(class_col) or row.get('truth_class')

    gdf['_bucket_class'] = gdf.apply(_row_class, axis=1)

    rows = []
    classes = sorted(c for c in gdf['_bucket_class'].dropna().unique())
    for cls in classes:
        sub = gdf[gdf['_bucket_class'] == cls]
        tp = int((sub['_final'] == 'TP').sum())
        fp = int((sub['_final'] == 'FP').sum())
        fn = int((sub['_final'] == 'FN').sum())
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-12)
        rows.append({
            'class': cls, 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': p, 'recall': r, 'f1': f1,
        })

    metrics_df = pd.DataFrame(rows).set_index('class')
    overall = {
        'tp': int(metrics_df['tp'].sum()),
        'fp': int(metrics_df['fp'].sum()),
        'fn': int(metrics_df['fn'].sum()),
    }
    overall['precision'] = overall['tp'] / max(
        overall['tp'] + overall['fp'], 1
    )
    overall['recall'] = overall['tp'] / max(
        overall['tp'] + overall['fn'], 1
    )
    overall['f1'] = 2 * overall['precision'] * overall['recall'] / max(
        overall['precision'] + overall['recall'], 1e-12
    )

    return {'metrics': metrics_df, 'overall': overall}


def plot_detections(src, predictions=None, truth=None, ax=None,
                    band_indices=None, scale=None, max_features=2000):
    """Plot a raster with detections color-coded by TP / FP / FN.

    Parameters
    ----------
    src : xarray.DataArray
        Background raster opened with ``gw.open()``.
    predictions : geopandas.GeoDataFrame, optional
        Detector output. Plotted in solid lines.
    truth : geopandas.GeoDataFrame, optional
        Ground truth. Plotted as dashed lines.
    ax : matplotlib.axes.Axes, optional
        Existing axes. A new figure is created if None.
    band_indices : list of int, optional
        RGB bands for the background.
    scale : tuple of (lo, hi), optional
        Stretch range for background display.
    max_features : int
        Subsample to this many features before plotting to keep the
        figure responsive on large outputs. Default 2000.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from .detection_data import _prepare_rgb_tile

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    block = src.values
    if block.ndim == 4:
        block = block[0]
    rgb = _prepare_rgb_tile(block, band_indices, scale)

    extent = (src.gw.left, src.gw.right, src.gw.bottom, src.gw.top)
    ax.imshow(rgb, extent=extent, origin='upper')

    def _maybe_sample(gdf):
        if gdf is not None and len(gdf) > max_features:
            return gdf.sample(max_features, random_state=0)
        return gdf

    predictions = _maybe_sample(predictions)
    truth = _maybe_sample(truth)

    if truth is not None and not truth.empty:
        truth.boundary.plot(
            ax=ax, color='yellow', linestyle='--',
            linewidth=1.0, label='Truth',
        )

    if predictions is not None and not predictions.empty:
        if 'status' in predictions.columns:
            colors = {
                'TP': 'lime', 'FP': 'red',
                'FP_class': 'orange', 'FN': 'magenta',
            }
            for status, color in colors.items():
                sub = predictions[predictions['status'] == status]
                if sub.empty:
                    continue
                sub.boundary.plot(
                    ax=ax, color=color, linewidth=1.2, label=status,
                )
        else:
            predictions.boundary.plot(
                ax=ax, color='cyan', linewidth=1.0, label='Predictions',
            )

    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    return ax
