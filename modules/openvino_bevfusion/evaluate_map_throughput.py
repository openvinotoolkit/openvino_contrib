#!/usr/bin/env python3
"""
Evaluate mAP and throughput of the BEVFusion OpenVINO pipeline.

This script:
1. Runs inference on validation samples from a dataset info pickle
2. Computes per-class Average Precision (AP) using 3D centre-distance matching
3. Reports mAP (mean AP across classes)
4. Measures throughput in samples per second
5. (Optional) Visualizes GT and OpenVINO detections projected on camera images

Dataset-agnostic evaluation with per-class distance thresholds.
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))


# ============================================================
# Constants
# ============================================================

# Default class names — can be overridden via --class-names CLI argument
CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]

# AP averaged over multiple centre-distance thresholds (metres)
DIST_THRESHOLDS = [0.5, 1.0, 2.0, 4.0]

# Single threshold per class (for reporting)
DISTANCE_THRESHOLDS = {
    'car': 2.0,
    'truck': 2.0,
    'construction_vehicle': 2.0,
    'bus': 2.0,
    'trailer': 2.0,
    'barrier': 2.0,
    'motorcycle': 1.0,
    'bicycle': 1.0,
    'pedestrian': 0.5,
    'traffic_cone': 0.5,
}


# TP metrics: which metrics are NOT applicable per class
# orient_err not for traffic_cone (symmetric)
# vel_err, attr_err not for barrier, traffic_cone (static / no attributes)
TP_METRICS_NOT_APPLICABLE = {
    'orient_err': ['traffic_cone'],
    'vel_err':    ['barrier', 'traffic_cone'],
    'attr_err':   ['barrier', 'traffic_cone'],
}

# TP matching threshold: 2m centre distance
TP_DIST_THRESHOLD = 2.0


# Camera display ordering: front row then back row
CAMERA_ORDER = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT',
]

# Per-class colours (BGR) for drawing — distinct from GT green (0,255,0)
CLASS_COLORS_BGR = {
    'car':                   (255, 165,   0),   # orange
    'truck':                 (255, 200,  50),   # gold
    'construction_vehicle':  (180, 180,   0),   # olive
    'bus':                   (255, 220, 100),   # light orange
    'trailer':               (200, 200, 100),   # khaki
    'barrier':               (180, 130, 180),   # mauve
    'motorcycle':            (  0, 200, 200),   # cyan
    'bicycle':               (  0, 255, 255),   # bright cyan
    'pedestrian':            (255,   0, 255),   # magenta
    'traffic_cone':          (  0, 128, 255),   # blue-orange
}


# ============================================================
# 3D → 2D Projection Utilities
# ============================================================
def get_3d_box_corners(center: np.ndarray, dims: np.ndarray,
                       yaw: float) -> np.ndarray:
    """Return the 8 corners of a 3D box in world (lidar) frame.

    Convention:
        dims = [l, w, h]  (length along x, width along y, height along z)
        yaw  = rotation about the Z-axis

    Returns:
        corners: [8, 3]  — ordered bottom-4 then top-4
    """
    l, w, h = dims
    # half-extents
    dx, dy, dz = l / 2, w / 2, h / 2
    # corners in object frame (bottom then top)
    corners = np.array([
        [ dx,  dy, -dz],
        [ dx, -dy, -dz],
        [-dx, -dy, -dz],
        [-dx,  dy, -dz],
        [ dx,  dy,  dz],
        [ dx, -dy,  dz],
        [-dx, -dy,  dz],
        [-dx,  dy,  dz],
    ], dtype=np.float64)
    # rotate
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    corners = corners @ R.T + np.asarray(center, dtype=np.float64)
    return corners


def project_corners_to_image(
    corners_lidar: np.ndarray,
    cam_intrinsic: np.ndarray,
    sensor2lidar_rot: np.ndarray,
    sensor2lidar_trans: np.ndarray,
    img_w: int,
    img_h: int,
) -> Optional[np.ndarray]:
    """Project 8 lidar-frame corners into a camera image.

    Returns:
        pts_2d: [8, 2] pixel coords  (or None if box is behind camera /
                 entirely outside the image).
    """
    # lidar → camera  =  inverse(sensor2lidar)
    R = np.asarray(sensor2lidar_rot, dtype=np.float64)       # cam→lidar
    t = np.asarray(sensor2lidar_trans, dtype=np.float64)     # cam→lidar
    # Inverse: pts_cam = R^T @ (pts_lidar - t)
    pts_cam = (corners_lidar - t) @ R  # [8,3] @ [3,3] = R^T applied row-wise

    # Discard if all points behind camera (z <= 0)
    if np.all(pts_cam[:, 2] <= 0):
        return None

    # Perspective projection
    K = np.asarray(cam_intrinsic, dtype=np.float64)  # [3,3]
    pts_img = pts_cam @ K.T                          # [8,3]
    valid = pts_img[:, 2] > 0
    if not np.any(valid):
        return None

    pts_2d = pts_img[:, :2] / pts_img[:, 2:3]       # [8,2]

    # Check at least one projected point inside image
    margin = 50  # pixels
    inside = (
        (pts_2d[:, 0] > -margin) & (pts_2d[:, 0] < img_w + margin) &
        (pts_2d[:, 1] > -margin) & (pts_2d[:, 1] < img_h + margin)
    )
    if not np.any(inside & valid):
        return None

    # Mask points behind camera with NaN so caller can skip those edges
    pts_2d[~valid] = np.nan
    return pts_2d.astype(np.float32)


# 12 edges of a cuboid connecting 8 corners  (bottom-4, top-4)
_BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),   # top
    (0, 4), (1, 5), (2, 6), (3, 7),   # pillars
]

# A thicker front-face indicator (edge 0-1 and 4-5 = front of the box)
_FRONT_EDGES = {(0, 1), (4, 5), (0, 4), (1, 5)}


def draw_box_on_image(
    img: np.ndarray,
    pts_2d: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 2,
    label: str = '',
    label_color: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Draw a projected 3D box (12 edges) on an image (in-place)."""
    for i, j in _BOX_EDGES:
        p1, p2 = pts_2d[i], pts_2d[j]
        if np.isnan(p1).any() or np.isnan(p2).any():
            continue
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        x2, y2 = int(round(p2[0])), int(round(p2[1]))
        th = thickness + 1 if (i, j) in _FRONT_EDGES else thickness
        cv2.line(img, (x1, y1), (x2, y2), color, th, cv2.LINE_AA)

    if label:
        # Place label near top-front edge midpoint
        valid = pts_2d[~np.isnan(pts_2d[:, 0])]
        if len(valid) > 0:
            lx = int(valid[:, 0].min())
            ly = int(valid[:, 1].min()) - 4
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th_txt), _ = cv2.getTextSize(label, font, 0.45, 1)
            cv2.rectangle(img, (lx, ly - th_txt - 2), (lx + tw, ly + 2),
                          color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (lx, ly), font, 0.45,
                        label_color, 1, cv2.LINE_AA)


# ============================================================
# Visualisation: multi-camera mosaic
# ============================================================
def visualize_sample(
    info: Dict,
    gt_boxes: List[Dict],
    ov_boxes: List[Dict],
    sample_idx: int,
    output_dir: str,
    score_vis_threshold: float = 0.25,
    max_range: float = 60.0,
    data_root: Optional[Path] = None,
) -> str:
    """Render GT + OpenVINO detections on the 6 camera images and save a
    mosaic (3 columns x 2 rows).

    Args:
        info:  Sample info dict (contains cams, gt_boxes, etc.)
        gt_boxes:  List of GT dicts  {box: {center,dims,yaw}, class_name}
        ov_boxes:  List of det dicts {box: {center,dims,yaw}, score, label_name}
        sample_idx: index for filename
        output_dir: directory to write images
        score_vis_threshold: minimum score for drawing OpenVINO boxes
        max_range: max distance (m) from ego for drawing boxes
        data_root: dataset root directory for resolving relative image paths

    Returns:
        path to saved mosaic image
    """
    cams = info.get('cams', {})
    panels = []

    for cam_name in CAMERA_ORDER:
        cam_info = cams.get(cam_name)
        if cam_info is None:
            # placeholder grey panel
            panels.append(np.full((900, 1600, 3), 40, dtype=np.uint8))
            continue

        # --- load image ------------------------------------------------
        img_path = cam_info['data_path']
        if not os.path.isabs(img_path) and data_root is not None:
            img_path = str(data_root / img_path.lstrip('./'))
        img = cv2.imread(img_path)
        if img is None:
            panels.append(np.full((900, 1600, 3), 40, dtype=np.uint8))
            continue
        img_h, img_w = img.shape[:2]

        # Camera calibration
        K   = np.array(cam_info['cam_intrinsic'], dtype=np.float64)
        R   = np.array(cam_info['sensor2lidar_rotation'], dtype=np.float64)
        t   = np.array(cam_info['sensor2lidar_translation'], dtype=np.float64)

        # --- draw GT boxes (green) -------------------------------------
        for gt in gt_boxes:
            center = np.array(gt['box']['center'])
            if np.linalg.norm(center[:2]) > max_range:
                continue
            corners = get_3d_box_corners(center,
                                         np.array(gt['box']['dims']),
                                         gt['box']['yaw'])
            pts = project_corners_to_image(corners, K, R, t, img_w, img_h)
            if pts is not None:
                cls = gt.get('class_name', '')
                draw_box_on_image(img, pts, color=(0, 255, 0), thickness=2,
                                  label=f"GT:{cls}")

        # --- draw OpenVINO boxes (per-class colour, dashed-ish) --------
        for det in ov_boxes:
            if det['score'] < score_vis_threshold:
                continue
            center = np.array(det['box']['center'])
            if np.linalg.norm(center[:2]) > max_range:
                continue
            corners = get_3d_box_corners(center,
                                         np.array(det['box']['dims']),
                                         det['box']['yaw'])
            pts = project_corners_to_image(corners, K, R, t, img_w, img_h)
            if pts is not None:
                cls = det.get('label_name', '')
                color = CLASS_COLORS_BGR.get(cls, (0, 128, 255))
                draw_box_on_image(img, pts, color=color, thickness=2,
                                  label=f"OV:{cls} {det['score']:.2f}")

        # camera name banner
        cv2.putText(img, cam_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(img)

    # --- assemble mosaic (2 rows x 3 cols) -----------------------------
    # Resize panels to uniform size for tiling
    tile_h, tile_w = 450, 800  # half of 900x1600
    resized = [cv2.resize(p, (tile_w, tile_h)) for p in panels]
    row_top = np.concatenate(resized[:3], axis=1)     # FL, F, FR
    row_bot = np.concatenate(resized[3:6], axis=1)    # BL, B, BR
    mosaic = np.concatenate([row_top, row_bot], axis=0)

    # Legend
    y0 = mosaic.shape[0] - 10
    cv2.putText(mosaic, 'Green = Ground Truth', (10, y0 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(mosaic, 'Coloured = OpenVINO detections', (300, y0 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2, cv2.LINE_AA)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'sample_{sample_idx:04d}.jpg')
    cv2.imwrite(out_path, mosaic, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out_path


# ============================================================
# BEV Visualisation (bird's-eye view)
# ============================================================
def visualize_bev(
    gt_boxes: List[Dict],
    ov_boxes: List[Dict],
    sample_idx: int,
    output_dir: str,
    score_vis_threshold: float = 0.25,
    bev_range: float = 60.0,
    bev_size: int = 800,
) -> str:
    """Render a BEV map showing GT and OpenVINO boxes from above."""
    img = np.full((bev_size, bev_size, 3), 30, dtype=np.uint8)
    scale = bev_size / (2 * bev_range)  # px per metre
    cx, cy = bev_size // 2, bev_size // 2

    def world_to_bev(x, y):
        return int(cx + x * scale), int(cy - y * scale)

    # Grid lines every 20 m
    for d in range(-int(bev_range), int(bev_range) + 1, 20):
        px, py = world_to_bev(d, -bev_range)
        px2, _ = world_to_bev(d, bev_range)
        cv2.line(img, (px, 0), (px, bev_size), (50, 50, 50), 1)
        _, py1 = world_to_bev(-bev_range, d)
        _, py2 = world_to_bev(bev_range, d)
        cv2.line(img, (0, py1), (bev_size, py1), (50, 50, 50), 1)

    # Ego vehicle
    cv2.circle(img, (cx, cy), 5, (255, 255, 255), -1)

    def draw_bev_box(box_dict, color, thickness=2):
        center = np.array(box_dict['center'][:2])
        if np.linalg.norm(center) > bev_range:
            return
        dims = np.array(box_dict['dims'][:2])  # l, w
        yaw = box_dict['yaw']
        corners = get_3d_box_corners(
            np.array([*center, 0]), np.array([dims[0], dims[1], 0.1]), yaw
        )
        pts_bev = np.array([world_to_bev(c[0], c[1]) for c in corners[:4]],
                           dtype=np.int32)
        cv2.polylines(img, [pts_bev], True, color, thickness, cv2.LINE_AA)
        # heading indicator
        mid_front = ((pts_bev[0] + pts_bev[1]) / 2).astype(int)
        cv2.circle(img, tuple(mid_front), 3, color, -1)

    # GT (green)
    for gt in gt_boxes:
        draw_bev_box(gt['box'], (0, 255, 0), 2)

    # OpenVINO (per-class colour)
    for det in ov_boxes:
        if det['score'] < score_vis_threshold:
            continue
        cls = det.get('label_name', '')
        color = CLASS_COLORS_BGR.get(cls, (0, 128, 255))
        draw_bev_box(det['box'], color, 2)

    # Legend
    cv2.putText(img, 'Green=GT  Coloured=OpenVINO', (10, bev_size - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
                cv2.LINE_AA)
    cv2.putText(img, f'Range: {bev_range:.0f}m', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
                cv2.LINE_AA)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'bev_{sample_idx:04d}.jpg')
    cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return out_path


# ============================================================
# 3D IoU Computation
# ============================================================
def rotation_matrix_z(angle: float) -> np.ndarray:
    """Create rotation matrix for Z-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ], dtype=np.float32)


def get_corners_3d(center: np.ndarray, dims: np.ndarray, yaw: float) -> np.ndarray:
    """
    Get 3D bounding box corners.
    
    Args:
        center: [3] (x, y, z)
        dims: [3] (length, width, height) or (w, l, h) - we handle both
        yaw: rotation around Z-axis
    
    Returns:
        corners: [8, 3] corner coordinates
    """
    l, w, h = dims[0], dims[1], dims[2]
    
    # 8 corners in local frame (centered at origin)
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    
    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32).T  # [8, 3]
    
    # Rotate around Z-axis
    R = rotation_matrix_z(yaw)
    corners = corners @ R.T
    
    # Translate to world frame
    corners += center
    
    return corners


def compute_iou_bev(box1_corners: np.ndarray, box2_corners: np.ndarray) -> float:
    """
    Compute BEV (Bird's Eye View) IoU between two boxes using Shapely.
    Falls back to axis-aligned approximation if Shapely isn't available.
    """
    try:
        from shapely.geometry import Polygon
        
        # Take bottom 4 corners (indices 4-7) for BEV
        poly1 = Polygon(box1_corners[4:8, :2])
        poly2 = Polygon(box2_corners[4:8, :2])
        
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0
        
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
        
        if union_area < 1e-6:
            return 0.0
        
        return inter_area / union_area
    except ImportError:
        # Fallback: axis-aligned BEV IoU
        # Get min/max for each box
        x1_min, x1_max = box1_corners[:, 0].min(), box1_corners[:, 0].max()
        y1_min, y1_max = box1_corners[:, 1].min(), box1_corners[:, 1].max()
        x2_min, x2_max = box2_corners[:, 0].min(), box2_corners[:, 0].max()
        y2_min, y2_max = box2_corners[:, 1].min(), box2_corners[:, 1].max()
        
        # Intersection
        xi1, yi1 = max(x1_min, x2_min), max(y1_min, y2_min)
        xi2, yi2 = min(x1_max, x2_max), min(y1_max, y2_max)
        
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        
        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        if union_area < 1e-6:
            return 0.0
        
        return inter_area / union_area


def compute_iou_3d(box1: Dict, box2: Dict) -> float:
    """
    Compute 3D IoU between two boxes.
    Simplified version: BEV IoU + height overlap
    """
    center1 = np.array(box1['center'])
    dims1 = np.array(box1['dims'])
    yaw1 = box1['yaw']
    
    center2 = np.array(box2['center'])
    dims2 = np.array(box2['dims'])
    yaw2 = box2['yaw']
    
    # Get 3D corners
    corners1 = get_corners_3d(center1, dims1, yaw1)
    corners2 = get_corners_3d(center2, dims2, yaw2)
    
    # BEV IoU
    bev_iou = compute_iou_bev(corners1, corners2)
    
    # Height overlap
    z1_min, z1_max = corners1[:, 2].min(), corners1[:, 2].max()
    z2_min, z2_max = corners2[:, 2].min(), corners2[:, 2].max()
    
    z_inter_min = max(z1_min, z2_min)
    z_inter_max = min(z1_max, z2_max)
    z_inter = max(0, z_inter_max - z_inter_min)
    
    z1_height = z1_max - z1_min
    z2_height = z2_max - z2_min
    z_union = z1_height + z2_height - z_inter
    
    if z_union < 1e-6:
        return 0.0
    
    height_iou = z_inter / z_union
    
    # 3D IoU approximation: BEV IoU * height IoU
    # This is an approximation; true 3D IoU requires convex hull intersection
    return bev_iou * height_iou


# ============================================================
# AP Computation
# ============================================================
def compute_center_distance(box1: Dict, box2: Dict) -> float:
    """Compute BEV center distance between two boxes."""
    c1 = np.array(box1['center'][:2])
    c2 = np.array(box2['center'][:2])
    return np.linalg.norm(c1 - c2)


def compute_ap(detections: List[Dict], ground_truths: List[Dict], 
               dist_threshold: float = 2.0) -> float:
    """
    Compute Average Precision for one class using center distance matching.
    
    Args:
        detections: list of {sample_id, box, score}
        ground_truths: list of {sample_id, box}
        dist_threshold: center distance threshold for true positive (meters)
    
    Returns:
        AP value
    """
    if len(ground_truths) == 0:
        return 0.0 if len(detections) > 0 else 1.0
    
    if len(detections) == 0:
        return 0.0
    
    # Sort detections by score (descending)
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    
    # Track which GTs are matched
    gt_matched = {}  # sample_id -> set of matched gt indices
    for gt in ground_truths:
        sid = gt['sample_id']
        if sid not in gt_matched:
            gt_matched[sid] = set()
    
    # Build GT lookup by sample
    gt_by_sample = {}
    for i, gt in enumerate(ground_truths):
        sid = gt['sample_id']
        if sid not in gt_by_sample:
            gt_by_sample[sid] = []
        gt_by_sample[sid].append((i, gt))
    
    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))
    
    for det_idx, det in enumerate(detections):
        sid = det['sample_id']
        
        if sid not in gt_by_sample:
            fp[det_idx] = 1
            continue
        
        best_dist = float('inf')
        best_gt_idx = -1
        
        for gt_idx, gt in gt_by_sample[sid]:
            if gt_idx in gt_matched.get(sid, set()):
                continue
            
            dist = compute_center_distance(det['box'], gt['box'])
            if dist < best_dist:
                best_dist = dist
                best_gt_idx = gt_idx
        
        if best_dist <= dist_threshold and best_gt_idx >= 0:
            tp[det_idx] = 1
            gt_matched[sid].add(best_gt_idx)
        else:
            fp[det_idx] = 1
    
    # Compute precision-recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recall = tp_cumsum / len(ground_truths)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    
    # Append sentinel values
    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[1], precision, [0]])
    
    # Make precision monotonically decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    
    # Compute area under PR curve (PASCAL VOC style)
    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    ap = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])
    
    return float(ap)


# ============================================================
# Main Evaluation
# ============================================================
def load_infos(data_pkl: str):
    """Load validation infos from a dataset info pickle file."""
    pkl_path = Path(data_pkl)
    if not pkl_path.exists():
        # Try common alternatives in the same directory
        parent = pkl_path.parent
        for alt in [
            parent / 'infos_val.pkl',
            parent / 'infos_mini_val.pkl',
        ]:
            if alt.exists():
                pkl_path = alt
                break
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        infos = data.get('infos', data.get('data_list', []))
    else:
        infos = data
    
    return infos


def get_ground_truths(info: Dict, sample_id: int) -> List[Dict]:
    """Extract ground truth boxes from sample info."""
    gts = []
    
    if 'gt_boxes' not in info or 'gt_names' not in info:
        return gts
    
    boxes = np.array(info['gt_boxes'])
    names = info['gt_names']
    gt_velocity = np.array(info['gt_velocity']) if 'gt_velocity' in info else None
    
    for i, (box, name) in enumerate(zip(boxes, names)):
        if name not in CLASS_NAMES:
            continue
        
        vel = [0.0, 0.0]
        if gt_velocity is not None and i < len(gt_velocity):
            vel = gt_velocity[i].tolist()
        
        # box format: [x, y, z, l, w, h, yaw]
        gts.append({
            'sample_id': sample_id,
            'box': {
                'center': box[:3].tolist(),
                'dims': box[3:6].tolist(),  # [l, w, h]
                'yaw': float(box[6]),
                'vel': vel,
            },
            'class_name': name,
            'class_id': CLASS_NAMES.index(name),
        })
    
    return gts


# ============================================================
# TP Metric Computation
# ============================================================
def _angle_diff(x: float, y: float, period: float = 2 * np.pi) -> float:
    """Smallest signed angle difference, result in [-period/2, period/2]."""
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff -= period
    return diff


def _scale_iou(dims1: List[float], dims2: List[float]) -> float:
    """Axis-aligned 3D IoU after aligning centres and orientation (scale error)."""
    d1 = np.array(dims1, dtype=np.float64)  # [l, w, h]
    d2 = np.array(dims2, dtype=np.float64)
    inter = np.prod(np.minimum(d1, d2))
    vol1, vol2 = np.prod(d1), np.prod(d2)
    union = vol1 + vol2 - inter
    return float(inter / union) if union > 1e-10 else 0.0


def compute_tp_metrics(
    detections: Dict[str, List[Dict]],
    ground_truths: Dict[str, List[Dict]],
) -> Dict[str, Dict[str, float]]:
    """Compute true-positive metrics at ``TP_DIST_THRESHOLD`` m.

    For each class, match detections to GTs (greedy, by centre distance),
    then compute ATE / ASE / AOE / AVE / AAE over true-positive pairs.

    Returns:
        per_class dict  {class_name: {ATE, ASE, AOE, AVE, AAE}}.
        Values are ``float('nan')`` for non-applicable metrics and ``1.0``
        when no TPs exist (convention: max error = 1).
    """
    per_class = {}

    for cls_name in CLASS_NAMES:
        dets = sorted(detections.get(cls_name, []),
                      key=lambda x: x['score'], reverse=True)
        gts = ground_truths.get(cls_name, [])

        # ---- greedy matching at ``TP_DIST_THRESHOLD`` m ----------------
        gt_by_sample: Dict[int, List[Tuple[int, Dict]]] = {}
        for i, gt in enumerate(gts):
            gt_by_sample.setdefault(gt['sample_id'], []).append((i, gt))
        matched_gt: set = set()
        tp_pairs: List[Tuple[Dict, Dict]] = []  # (det, gt)

        for det in dets:
            sid = det['sample_id']
            if sid not in gt_by_sample:
                continue
            best_dist, best_gt_idx, best_gt = float('inf'), -1, None
            for gi, gt in gt_by_sample[sid]:
                if gi in matched_gt:
                    continue
                d = np.linalg.norm(
                    np.array(det['box']['center'][:2]) -
                    np.array(gt['box']['center'][:2]))
                if d < best_dist:
                    best_dist, best_gt_idx, best_gt = d, gi, gt
            if best_dist <= TP_DIST_THRESHOLD and best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                tp_pairs.append((det, best_gt))

        # ---- compute per-TP-pair errors --------------------------------
        ates, ases, aoes, aves = [], [], [], []
        for det, gt in tp_pairs:
            dc = np.array(det['box']['center'])
            gc = np.array(gt['box']['center'])
            # ATE: 2D BEV centre distance
            ates.append(float(np.linalg.norm(dc[:2] - gc[:2])))
            # ASE: 1 - scale_iou
            ases.append(1.0 - _scale_iou(det['box']['dims'], gt['box']['dims']))
            # AOE: absolute smallest yaw difference
            aoes.append(abs(_angle_diff(det['box']['yaw'], gt['box']['yaw'])))
            # AVE: L2 velocity error (skip if GT vel is NaN)
            dv = np.array(det['box'].get('vel', [0, 0]), dtype=np.float64)
            gv = np.array(gt['box'].get('vel', [0, 0]), dtype=np.float64)
            if not (np.any(np.isnan(dv)) or np.any(np.isnan(gv))):
                aves.append(float(np.linalg.norm(dv - gv)))

        # ---- aggregate (mean, or 1.0 if no TPs, or nan if N/A) ---------
        def _mean_or_default(vals, cls, metric_key):
            if cls in TP_METRICS_NOT_APPLICABLE.get(metric_key, []):
                return float('nan')
            if len(vals) == 0:
                return 1.0  # convention: max penalty when no TPs
            return float(np.mean(vals))

        per_class[cls_name] = {
            'ATE': _mean_or_default(ates, cls_name, 'trans_err'),
            'ASE': _mean_or_default(ases, cls_name, 'scale_err'),
            'AOE': _mean_or_default(aoes, cls_name, 'orient_err'),
            'AVE': _mean_or_default(aves, cls_name, 'vel_err'),
            'AAE': _mean_or_default([], cls_name, 'attr_err'),  # no attr prediction
            'n_tp': len(tp_pairs),
        }

    return per_class


def print_detail_results(
    per_class_ap: Dict[str, float],
    tp_metrics: Dict[str, Dict[str, float]],
    all_gt: Dict[str, List] = None,
) -> Dict[str, float]:
    """Print and return detailed per-class results table."""
    print("\n" + "=" * 70)
    print("Detailed Per-Class Results")
    print("=" * 70)

    metric_keys = ['ATE', 'ASE', 'AOE', 'AVE', 'AAE']

    # Compute means across applicable classes (skip nan)
    mean_vals = {}
    for mk in metric_keys:
        vals = [tp_metrics[c][mk] for c in CLASS_NAMES
                if not np.isnan(tp_metrics[c][mk])]
        mean_vals['m' + mk] = float(np.mean(vals)) if vals else float('nan')

    # mAP: average AP over classes that have GT
    if all_gt is not None:
        gt_classes = [c for c in CLASS_NAMES if len(all_gt.get(c, [])) > 0]
    else:
        gt_classes = list(CLASS_NAMES)
    valid_aps = [per_class_ap.get(c, 0.0) for c in gt_classes]
    map_val = float(np.mean(valid_aps)) if valid_aps else 0.0

    # Header
    print(f"  mAP: {map_val:.4f}")
    for mk in metric_keys:
        print(f"  m{mk}: {mean_vals['m' + mk]:.4f}")
    print()

    # Per-class table
    hdr = f"  {'Object Class':<24s} {'AP':>6s}"
    for mk in metric_keys:
        hdr += f"  {mk:>6s}"
    print(hdr)

    for cls in CLASS_NAMES:
        ap = per_class_ap.get(cls, 0.0)
        row = f"  {cls:<24s} {ap:6.3f}"
        for mk in metric_keys:
            v = tp_metrics[cls][mk]
            if np.isnan(v):
                row += f"  {'nan':>6s}"
            else:
                row += f"  {v:6.3f}"
        print(row)

    print("=" * 70)

    summary = {'mAP': map_val}
    summary.update(mean_vals)
    return summary


def run_evaluation(num_samples: int = -1, 
                   score_threshold: float = 0.1,
                   warmup: int = 3,
                   verbose: bool = False,
                   visualize: bool = False,
                   vis_dir: str = 'visualization_output',
                   vis_samples: int = 10,
                   vis_score_threshold: float = 0.25,
                   detail_result: bool = False,
                   model_dir: str = None,
                   device: str = 'GPU',
                   data_pkl: str = None,
                   data_root: str = None) -> Dict:
    """
    Run full evaluation.
    
    Args:
        num_samples: Number of samples (-1 for all)
        score_threshold: Detection score threshold
        warmup: Number of warmup iterations
        verbose: Print per-sample details
        visualize: Whether to generate visual output
        vis_dir: Directory for visualisation images
        vis_samples: How many samples to visualise (first N)
        vis_score_threshold: Minimum det score for drawing
        detail_result: Print detailed metrics (ATE/ASE/AOE/AVE/AAE)
        model_dir: Path to OpenVINO model directory (default: openvino_model)
        device: OpenVINO device to run on, e.g. 'GPU', 'CPU' (default: 'GPU')
        data_pkl: Path to dataset info pickle file
        data_root: Dataset root directory for image paths
    
    Returns:
        Dict with mAP, per-class AP, and throughput
    """
    # Import the pipeline
    from run_inference_standalone import BEVFusionInference
    
    # Load dataset infos
    import pickle
    if data_pkl is None:
        raise ValueError("--data-pkl must be provided")
    with open(str(data_pkl), 'rb') as f:
        data = pickle.load(f)
    all_infos = data['infos'] if isinstance(data, dict) else data
    
    root_path = Path(data_root) if data_root else None
    
    # Initialize pipeline
    print("\n" + "=" * 70)
    print("BEVFusion Evaluation: mAP and Throughput")
    print("=" * 70)
    
    effective_model_dir = Path(model_dir) if model_dir else Path('openvino_model')
    ext_dir = 'openvino_extensions'
    print(f"  Model dir: {effective_model_dir}")
    print(f"  Ext dir:   {ext_dir}")
    print(f"  Device:    {device}")
    pipeline = BEVFusionInference(
        model_dir=str(effective_model_dir),
        ext_dir=ext_dir,
        device=device,
    )
    data_root_path = root_path
    
    # Load data
    infos = all_infos
    if num_samples > 0:
        infos = infos[:num_samples]
    
    total_samples = len(infos)
    print(f"\nEvaluating {total_samples} samples (threshold={score_threshold})")
    
    # Storage for detections and ground truths
    all_detections = {cls: [] for cls in CLASS_NAMES}
    all_gt = {cls: [] for cls in CLASS_NAMES}
    
    # Warmup runs
    if warmup > 0 and total_samples > 0:
        print(f"\nWarmup ({warmup} iterations)...")
        for i in range(min(warmup, total_samples)):
            pipeline.run(infos[i], data_root_path, score_threshold=score_threshold)[:2]
        print("Warmup complete.")
    
    # Run evaluation
    print(f"\nRunning inference...")
    total_time = 0.0
    
    for idx, info in enumerate(infos):
        t0 = time.time()
        boxes, timings, *_ = pipeline.run(info, data_root_path, score_threshold=score_threshold)
        inference_time = timings['total']
        total_time += inference_time
        
        # Store detections (keep original dict for vis)
        sample_dets = []
        for det in boxes:
            cls_name = det['label_name']
            if cls_name in CLASS_NAMES:
                det_record = {
                    'sample_id': idx,
                    'box': {
                        'center': det['center'],
                        'dims': det['dims'],
                        'yaw': det['yaw'],
                        'vel': det.get('vel', [0.0, 0.0]),
                    },
                    'score': det['score'],
                    'label_name': cls_name,
                }
                all_detections[cls_name].append(det_record)
                sample_dets.append(det_record)
        
        # Store ground truths
        sample_gts = get_ground_truths(info, idx)
        for gt in sample_gts:
            all_gt[gt['class_name']].append(gt)
        
        if verbose or (idx + 1) % 10 == 0:
            n_det = len(boxes)
            n_gt = len(sample_gts)
            print(f"  [{idx+1:3d}/{total_samples}] det={n_det:3d}, gt={n_gt:3d}, "
                  f"time={inference_time*1000:.0f}ms")
        
        # --- visualise this sample if requested -------------------------
        if visualize and idx < vis_samples:
            cam_path = visualize_sample(
                info, sample_gts, sample_dets, idx, vis_dir,
                score_vis_threshold=vis_score_threshold,
                data_root=root_path)
            bev_path = visualize_bev(
                sample_gts, sample_dets, idx, vis_dir,
                score_vis_threshold=vis_score_threshold)
            if verbose:
                print(f"    Saved {cam_path}")
                print(f"    Saved {bev_path}")
    
    # Compute per-class AP
    print("\n" + "=" * 70)
    print("Computing mAP (center distance matching)...")
    print("=" * 70)
    
    per_class_ap = {}
    for cls_name in CLASS_NAMES:
        dets = all_detections[cls_name]
        gts = all_gt[cls_name]
        
        if len(gts) == 0:
            ap = 0.0
            per_class_ap[cls_name] = ap
            print(f"  {cls_name:22s}: AP={ap:.4f} (no GT, det={len(dets):4d})")
        else:
        # AP averaged over distance thresholds
            aps_per_thresh = []
            for dt in DIST_THRESHOLDS:
                ap_t = compute_ap(dets, gts, dist_threshold=dt)
                aps_per_thresh.append(ap_t)
            ap = np.mean(aps_per_thresh)
            per_class_ap[cls_name] = ap
            thresh_str = ", ".join(f"{dt:.1f}m:{a:.3f}" for dt, a in zip(DIST_THRESHOLDS, aps_per_thresh))
            print(f"  {cls_name:22s}: AP={ap:.4f} ({thresh_str}, det={len(dets):4d}, gt={len(gts):4d})")
    
    # Compute mAP (only classes with GT)
    valid_aps = [ap for cls, ap in per_class_ap.items() 
                 if len(all_gt[cls]) > 0]
    mAP = np.mean(valid_aps) if valid_aps else 0.0
    
    # Throughput
    avg_time = total_time / total_samples if total_samples > 0 else 0
    throughput = 1.0 / avg_time if avg_time > 0 else 0
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  mAP:                  {mAP:.4f}")
    print(f"  Total samples:        {total_samples}")
    print(f"  Total detections:     {sum(len(v) for v in all_detections.values())}")
    print(f"  Total GT boxes:       {sum(len(v) for v in all_gt.values())}")
    print(f"  Avg inference time:   {avg_time*1000:.1f} ms")
    print(f"  Throughput:           {throughput:.2f} samples/sec")
    print("=" * 70)
    
    # Compute diagnostic statistics for detection-GT mismatch
    avg_min_dist = []
    for cls_name in CLASS_NAMES:
        dets = all_detections[cls_name]
        gts = all_gt[cls_name]
        for gt in gts[:10]:  # Sample first 10 GTs per class
            gt_center = np.array(gt['box']['center'][:2])
            min_dist = float('inf')
            for det in dets:
                if det['sample_id'] == gt['sample_id']:
                    det_center = np.array(det['box']['center'][:2])
                    dist = np.linalg.norm(gt_center - det_center)
                    min_dist = min(min_dist, dist)
            if min_dist < float('inf'):
                avg_min_dist.append(min_dist)
    
    if avg_min_dist:
        print("\n" + "=" * 70)
        print("DIAGNOSTIC: Detection-GT Distance (sampled)")
        print("=" * 70)
        print(f"  Mean min distance to GT: {np.mean(avg_min_dist):.1f}m")
        print(f"  Median min distance:     {np.median(avg_min_dist):.1f}m")
        print(f"  Min distance found:      {np.min(avg_min_dist):.1f}m")
        if np.mean(avg_min_dist) > 5:
            print("  [WARN] Detections are far from GT - possible coordinate mismatch")
        print("=" * 70)
    
    results = {
        'mAP': float(mAP),
        'per_class_ap': per_class_ap,
        'total_samples': total_samples,
        'total_detections': sum(len(v) for v in all_detections.values()),
        'total_gt_boxes': sum(len(v) for v in all_gt.values()),
        'avg_inference_time_ms': avg_time * 1000,
        'throughput_samples_per_sec': throughput,
        'score_threshold': score_threshold,
        'avg_min_det_gt_distance': float(np.mean(avg_min_dist)) if avg_min_dist else None,
    }
    
    # --- Detailed TP metrics -----------------------------------
    if detail_result:
        tp_metrics = compute_tp_metrics(all_detections, all_gt)
        detail_summary = print_detail_results(per_class_ap, tp_metrics, all_gt)
        results['detail'] = detail_summary
        results['per_class_tp_metrics'] = {
            cls: {k: (None if np.isnan(v) else v)
                  for k, v in m.items()}
            for cls, m in tp_metrics.items()
        }
    
    if visualize:
        n_vis = min(vis_samples, total_samples)
        print(f"\n  Visualisations saved to {vis_dir}/  ({n_vis} samples)")
        print(f"    Camera mosaics : sample_NNNN.jpg")
        print(f"    BEV plots      : bev_NNNN.jpg")
        results['vis_dir'] = vis_dir
        results['vis_samples'] = n_vis
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate BEVFusion mAP and throughput')
    parser.add_argument('--num-samples', type=int, default=-1,
                        help='Number of samples to evaluate (-1 for all)')
    parser.add_argument('--score-threshold', type=float, default=0.0,
                        help='Detection score threshold (0 = keep all)')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Number of warmup iterations')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='Output JSON file')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-sample output')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visual output (camera + BEV mosaics)')
    parser.add_argument('--vis-dir', type=str, default='visualization_output',
                        help='Directory for visualisation images')
    parser.add_argument('--vis-samples', type=int, default=10,
                        help='Number of samples to visualise')
    parser.add_argument('--vis-score-threshold', type=float, default=0.25,
                        help='Min detection score for drawing')
    parser.add_argument('--detail-result', action='store_true',
                        help='Print detailed metrics (ATE/ASE/AOE/AVE/AAE)')
    parser.add_argument('--model-dir', type=str, default=None,
                        help='Path to OpenVINO model directory (default: openvino_model)')
    parser.add_argument('--device', type=str, default='GPU',
                        help='OpenVINO inference device, e.g. GPU, CPU (default: GPU)')
    parser.add_argument('--data-pkl', type=str, required=True,
                        help='Path to dataset info pickle file')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Dataset root directory (for resolving relative image paths)')
    parser.add_argument('--class-names', type=str, nargs='+', default=None,
                        help='Override class names (space-separated, e.g. car truck pedestrian)')
    args = parser.parse_args()
    
    # Override class names if provided
    if args.class_names:
        global CLASS_NAMES
        CLASS_NAMES = args.class_names
    
    # Set environment variables for optimal performance
    os.environ.setdefault('OMP_NUM_THREADS', '8')
    
    results = run_evaluation(
        num_samples=args.num_samples,
        score_threshold=args.score_threshold,
        warmup=args.warmup,
        verbose=args.verbose,
        visualize=args.visualize,
        vis_dir=args.vis_dir,
        vis_samples=args.vis_samples,
        vis_score_threshold=args.vis_score_threshold,
        detail_result=args.detail_result,
        model_dir=args.model_dir,
        device=args.device,
        data_pkl=args.data_pkl,
        data_root=args.data_root,
    )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    
    # Check acceptable values
    print("\n" + "=" * 70)
    print("ACCEPTANCE CHECK")
    print("=" * 70)
    mAP = results['mAP']
    throughput = results['throughput_samples_per_sec']
    
    # Expected: mAP ~0.5-0.7 for BEVFusion, throughput ~0.5-1.0 samples/sec
    mAP_ok = mAP >= 0.3  # Reasonable threshold for validation
    throughput_ok = throughput >= 0.3  # At least 0.3 samples/sec
    
    if mAP_ok:
        print(f"  [OK] mAP = {mAP:.4f} (>= 0.30)")
    else:
        print(f"  [WARN] mAP = {mAP:.4f} (expected >= 0.30)")
    
    print("=" * 70)


if __name__ == '__main__':
    main()
