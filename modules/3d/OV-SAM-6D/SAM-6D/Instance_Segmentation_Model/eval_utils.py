"""
mAP @ IoU[0.50:0.95] evaluation (COCO-style)
Shared between run_inference_custom.py and infer_ism_ov.py
"""

import os
import cv2
import numpy as np


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union > 0 else 0.0


def _compute_ap_101(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO-style 101-point interpolation AP."""
    if len(recall) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    recall_interp = np.linspace(0, 1, 101)
    precision_interp = np.interp(recall_interp, mrec, mpre)
    return float(np.mean(precision_interp))


def compute_map_iou(pred_masks, pred_scores, gt_masks, iou_thresholds=None):
    """Compute COCO-style mAP @ IoU[0.50:0.95] for one image.

    Parameters
    ----------
    pred_masks : (N, H, W) binary prediction masks
    pred_scores : (N,) confidence scores
    gt_masks : list of (H, W) binary GT masks
    iou_thresholds : IoU thresholds (default 0.50:0.05:0.95)

    Returns
    -------
    dict: map, ap_per_iou, best_iou, n_pred, n_gt
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.50, 0.96, 0.05)
    n_gt = len(gt_masks)
    n_pred = len(pred_masks)
    if n_gt == 0 or n_pred == 0:
        return {"map": 0.0, "ap_per_iou": {}, "best_iou": 0.0,
                "n_pred": n_pred, "n_gt": n_gt}
    order = np.argsort(pred_scores)[::-1]
    pred_masks = pred_masks[order]
    pred_scores = pred_scores[order]
    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float64)
    for i in range(n_pred):
        pm = pred_masks[i].astype(bool)
        for j in range(n_gt):
            iou_matrix[i, j] = _mask_iou(pm, gt_masks[j].astype(bool))
    best_iou = float(iou_matrix[0].max()) if n_pred > 0 else 0.0
    ap_per_iou = {}
    for th in iou_thresholds:
        matched_gt = set()
        tp = np.zeros(n_pred)
        fp = np.zeros(n_pred)
        for i in range(n_pred):
            best_j, best_val = -1, th
            for j in range(n_gt):
                if j in matched_gt:
                    continue
                if iou_matrix[i, j] >= best_val:
                    best_val = iou_matrix[i, j]
                    best_j = j
            if best_j >= 0:
                tp[i] = 1
                matched_gt.add(best_j)
            else:
                fp[i] = 1
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / n_gt
        precision = tp_cum / (tp_cum + fp_cum)
        ap_per_iou[f"{th:.2f}"] = _compute_ap_101(recall, precision)
    mAP = float(np.mean(list(ap_per_iou.values())))
    return {"map": mAP, "ap_per_iou": ap_per_iou, "best_iou": best_iou,
            "n_pred": n_pred, "n_gt": n_gt}


def load_gt_masks(gt_mask_path):
    """Load GT masks from a single PNG or directory of PNGs."""
    masks = []
    if os.path.isdir(gt_mask_path):
        for f in sorted(os.listdir(gt_mask_path)):
            if f.endswith(".png"):
                m = cv2.imread(os.path.join(gt_mask_path, f),
                               cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    masks.append((m > 127).astype(np.uint8))
    elif os.path.isfile(gt_mask_path):
        m = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            masks.append((m > 127).astype(np.uint8))
    return masks


def evaluate_and_print_map(pred_masks_bin, pred_scores, gt_mask_arg,
                           rgb_path, fallback_depth=True):
    """Run mAP evaluation with auto-detect GT and print results.

    Parameters
    ----------
    pred_masks_bin : (N, H, W) uint8 binary masks
    pred_scores : (N,) float scores
    gt_mask_arg : str or None — explicit --gt_mask path
    rgb_path : str — path to the RGB image (for auto-detect)
    fallback_depth : bool — whether to try depth connected-components fallback

    Returns
    -------
    dict or None — metrics dict if GT found, else None
    """
    gt_masks_list = []
    gt_source = None

    if gt_mask_arg:
        gt_masks_list = load_gt_masks(gt_mask_arg)
        gt_source = gt_mask_arg
    else:
        img_dir = os.path.dirname(os.path.abspath(rgb_path))
        mask_visib_dir = os.path.join(img_dir, "mask_visib")
        if os.path.isdir(mask_visib_dir):
            gt_masks_list = load_gt_masks(mask_visib_dir)
            gt_source = mask_visib_dir
        elif fallback_depth:
            depth_fallback = os.path.join(img_dir, "depth.png")
            if os.path.isfile(depth_fallback):
                depth_gt = cv2.imread(depth_fallback, cv2.IMREAD_UNCHANGED)
                if depth_gt is not None:
                    fg = (depth_gt > 0).astype(np.uint8)
                    if fg.ndim == 3:
                        fg = fg[:, :, 0]
                    n_cc, labels = cv2.connectedComponents(fg)
                    for cc_id in range(1, n_cc):
                        cc_mask = (labels == cc_id).astype(np.uint8)
                        if cc_mask.sum() < fg.shape[0] * fg.shape[1] * 0.001:
                            continue
                        gt_masks_list.append(cc_mask)
                    gt_source = (f"{depth_fallback} "
                                 f"({len(gt_masks_list)} objects "
                                 f"via connected components)")

    if gt_masks_list:
        metrics = compute_map_iou(pred_masks_bin, pred_scores, gt_masks_list)
        print(f"\n  {'='*55}")
        print(f"  mAP @ IoU[0.50:0.95] (COCO-style)")
        print(f"  {'='*55}")
        print(f"    GT source   : {gt_source}")
        print(f"    Predictions : {metrics['n_pred']}")
        print(f"    GT masks    : {metrics['n_gt']}")
        print(f"    Best IoU    : {metrics['best_iou']:.4f}")
        for th_key, ap_val in sorted(metrics["ap_per_iou"].items()):
            print(f"    AP@{th_key}    : {ap_val * 100:6.2f}%")
        print(f"    ─────────────────────────────")
        print(f"    mAP@[.50:.95]: {metrics['map'] * 100:6.2f}%")
        return metrics
    else:
        print(f"\n  [mAP] No GT masks found "
              f"(use --gt_mask or place mask_visib/ or depth.png near image)")
        return None


def load_gt_masks_for_object(mask_visib_dir, gt_data, image_id, obj_id):
    """Load GT visible masks for a specific object in a BOP image.

    Parameters
    ----------
    mask_visib_dir : str — path to mask_visib/ directory
    gt_data : dict — parsed scene_gt.json {str(img_id): [{"obj_id":...}, ...]}
    image_id : int
    obj_id : int

    Returns
    -------
    list of (H, W) uint8 binary masks
    """
    gt_entries = gt_data.get(str(image_id), [])
    gt_masks = []
    for gt_idx, entry in enumerate(gt_entries):
        if entry["obj_id"] == obj_id:
            mask_path = os.path.join(mask_visib_dir,
                                     f"{image_id:06d}_{gt_idx:06d}.png")
            if os.path.exists(mask_path):
                m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    gt_masks.append((m > 127).astype(np.uint8))
    return gt_masks


def aggregate_bop_maps(maps_by_obj):
    """Aggregate per-testcase mAP values into per-object and overall metrics.

    Parameters
    ----------
    maps_by_obj : dict of {obj_id: list of float mAP values (0..1 scale)}

    Returns
    -------
    dict with 'per_object', 'overall_mAP' (percent), 'n_total'
    """
    per_object = {}
    all_maps = []
    for oid in sorted(maps_by_obj.keys()):
        obj_maps = maps_by_obj[oid]
        per_object[oid] = {
            "n_testcases": len(obj_maps),
            "mAP": float(np.mean(obj_maps)) * 100 if obj_maps else 0.0,
        }
        all_maps.extend(obj_maps)
    overall = float(np.mean(all_maps)) * 100 if all_maps else 0.0
    return {"per_object": per_object, "overall_mAP": overall,
            "n_total": len(all_maps)}
