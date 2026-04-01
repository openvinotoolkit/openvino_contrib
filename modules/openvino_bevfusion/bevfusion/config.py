"""
BEVFusion configuration constants.

Centralizes all pipeline parameters so any consumer of the bevfusion package
can import a single, consistent configuration.
"""

from pathlib import Path

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / 'openvino_model'
CHECKPOINT_PATH = PROJECT_DIR / 'pretrained_models' / 'bevfusion-det.pth'

# ── Camera ─────────────────────────────────────────────────────────────────────
CAMERA_NAMES = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]

IMAGE_SIZE = (256, 704)  # H, W
RESIZE_LIM = 0.48
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── LiDAR / Voxelization ──────────────────────────────────────────────────────
POINT_CLOUD_RANGE = np.array([-54.0, -54.0, -5.0, 54.0, 54.0, 3.0])
VOXEL_SIZE = np.array([0.075, 0.075, 0.2])
SPARSE_SHAPE = np.array([1440, 1440, 41])  # [X, Y, Z]

# Number of LiDAR sweeps to accumulate (fewer = faster, more = denser)
SWEEPS_NUM = 1

# ── BEV / Depth ───────────────────────────────────────────────────────────────
D = 118   # depth bins
C = 80    # camera feature channels
DBOUND = [1.0, 60.0, 0.5]
XBOUND = [-54.0, 54.0, 0.3]
YBOUND = [-54.0, 54.0, 0.3]
ZBOUND = [-10.0, 10.0, 20.0]

# ── Detection ─────────────────────────────────────────────────────────────────
NUM_CLASSES = 10
CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]
OUT_SIZE_FACTOR = 8
POST_CENTER_RANGE = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
