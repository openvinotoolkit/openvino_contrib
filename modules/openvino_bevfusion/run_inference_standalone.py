#!/usr/bin/env python3
"""
BEVFusion standalone inference using OpenVINO models and custom extensions.

This script is the minimal, self-contained inference pipeline.  It requires:
  - OpenVINO model files  (xml/bin) in a model directory
  - OpenVINO extension libraries (.so) for custom ops
  - Dataset info pickle and raw data files

No other files from the bevfusion package are needed.

Usage:
    python run_inference_standalone.py \\
        --model-dir openvino_model \\
        --ext-dir openvino_extensions \\
        --data-pkl /path/to/dataset_infos_val.pkl \\
        --data-root /path/to/dataset/root \\
        --num-samples 5
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration constants (same as bevfusion/config.py)
# ═══════════════════════════════════════════════════════════════════════════════

CAMERA_NAMES = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]
IMAGE_SIZE = (256, 704)
RESIZE_LIM = 0.48
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

POINT_CLOUD_RANGE = np.array([-54.0, -54.0, -5.0, 54.0, 54.0, 3.0])
VOXEL_SIZE = np.array([0.075, 0.075, 0.2])
SWEEPS_NUM = 1

D = 118
C = 80
DBOUND = [1.0, 60.0, 0.5]
XBOUND = [-54.0, 54.0, 0.3]
YBOUND = [-54.0, 54.0, 0.3]
ZBOUND = [-10.0, 10.0, 20.0]

NX = int((XBOUND[1] - XBOUND[0]) / XBOUND[2])   # 360
NY = int((YBOUND[1] - YBOUND[0]) / YBOUND[2])   # 360

NUM_CLASSES = 10
CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]
OUT_SIZE_FACTOR = 8
POST_CENTER_RANGE = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]

# Padding limits for custom ops
MAX_POINTS = 70000
MAX_VOXELS = 60000

# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_path(path: str, data_root: Path) -> str:
    if os.path.isabs(path):
        return path
    return str(data_root / path.lstrip('./'))


def load_lidar_points(info: dict, data_root: Path) -> np.ndarray:
    lidar_path = info.get('lidar_path', '')
    if not lidar_path:
        lidar_path = info.get('lidar_points', {}).get('lidar_path', '')
    lidar_path = _resolve_path(lidar_path, data_root)

    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    points[:, 4] = 0.0

    sweep_list = [points]
    sweeps = info.get('sweeps', [])
    ts = info.get('timestamp', 0) / 1e6

    if len(sweeps) == 0:
        xy = np.abs(points[:, :2])
        mask = ~((xy[:, 0] < 1.0) & (xy[:, 1] < 1.0))
        padded = points[mask].copy()
        for _ in range(SWEEPS_NUM):
            sweep_list.append(padded.copy())
    else:
        for idx in range(min(len(sweeps), SWEEPS_NUM)):
            sw = sweeps[idx]
            sp = _resolve_path(sw.get('data_path', ''), data_root)
            if not os.path.exists(sp):
                continue
            pts = np.fromfile(sp, dtype=np.float32).reshape(-1, 5)
            xy = np.abs(pts[:, :2])
            pts = pts[~((xy[:, 0] < 1.0) & (xy[:, 1] < 1.0))]
            rot = np.array(sw['sensor2lidar_rotation'], dtype=np.float64)
            trans = np.array(sw['sensor2lidar_translation'], dtype=np.float64)
            pts[:, :3] = (pts[:, :3].astype(np.float64) @ rot.T + trans).astype(np.float32)
            pts[:, 4] = ts - sw['timestamp'] / 1e6
            sweep_list.append(pts)

    return np.concatenate(sweep_list, axis=0)


def load_camera_images(info: dict, data_root: Path):
    images, aug_mats = [], []
    cams = info.get('cams', info.get('images', {}))

    for cam_name in CAMERA_NAMES:
        cam = cams[cam_name]
        img_path = cam.get('data_path', cam.get('img_path', ''))
        img_path = _resolve_path(img_path, data_root)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        nw, nh = int(w * RESIZE_LIM), int(h * RESIZE_LIM)
        img = cv2.resize(img, (nw, nh))
        ch = max(0, nh - IMAGE_SIZE[0])
        cw = max(0, (nw - IMAGE_SIZE[1]) // 2)
        img = img[ch:ch + IMAGE_SIZE[0], cw:cw + IMAGE_SIZE[1]]

        aug = np.eye(4, dtype=np.float32)
        aug[0, 0] = RESIZE_LIM
        aug[1, 1] = RESIZE_LIM
        aug[0, 2] = -cw
        aug[1, 2] = -ch
        aug_mats.append(aug)

        img = ((img.astype(np.float32) / 255.0 - IMG_MEAN) / IMG_STD).transpose(2, 0, 1)
        images.append(img)

    return np.stack(images), np.stack(aug_mats)


def get_camera_transforms(info: dict):
    cams = info.get('cams', info.get('images', {}))
    l2i, c2l, intr = [], [], []
    for cam_name in CAMERA_NAMES:
        cam = cams[cam_name]
        c2l_mat = np.eye(4, dtype=np.float32)
        c2l_mat[:3, :3] = np.array(cam.get('sensor2lidar_rotation', np.eye(3)))
        c2l_mat[:3, 3] = np.array(cam.get('sensor2lidar_translation', [0, 0, 0]))
        c2l.append(c2l_mat)
        ci = np.array(cam.get('cam_intrinsic', np.eye(3)), dtype=np.float32)
        intr.append(ci)
        vp = np.eye(4, dtype=np.float32)
        vp[:3, :3] = ci
        l2i.append(vp @ np.linalg.inv(c2l_mat))
    return np.stack(l2i), np.stack(c2l), np.stack(intr)


# ═══════════════════════════════════════════════════════════════════════════════
# Depth maps
# ═══════════════════════════════════════════════════════════════════════════════

def create_depth_maps(points, lidar2image, img_aug_matrices):
    h, w = IMAGE_SIZE
    depth_maps = np.zeros((6, 1, h, w), dtype=np.float32)
    pts_hom = np.concatenate([points[:, :3], np.ones((len(points), 1), dtype=np.float32)], axis=1)
    proj = np.einsum('cij,nj->cin', lidar2image, pts_hom)
    depth = proj[:, 2, :]
    vd = depth > 0.1
    px = proj[:, 0, :] / np.maximum(depth, 0.1)
    py = proj[:, 1, :] / np.maximum(depth, 0.1)
    sx, sy = img_aug_matrices[:, 0, 0][:, None], img_aug_matrices[:, 1, 1][:, None]
    tx, ty = img_aug_matrices[:, 0, 2][:, None], img_aug_matrices[:, 1, 2][:, None]
    ax, ay = sx * px + tx, sy * py + ty
    valid = vd & (ax >= 0) & (ax < w) & (ay >= 0) & (ay < h) & (depth >= DBOUND[0]) & (depth < DBOUND[1])
    for c in range(6):
        v = valid[c]
        if v.sum() > 0:
            depth_maps[c, 0, ay[c, v].astype(np.int32), ax[c, v].astype(np.int32)] = depth[c, v]
    return depth_maps


# ═══════════════════════════════════════════════════════════════════════════════
# Geometry
# ═══════════════════════════════════════════════════════════════════════════════

def create_frustum():
    d = int((DBOUND[1] - DBOUND[0]) / DBOUND[2])
    fH, fW = IMAGE_SIZE[0] // 8, IMAGE_SIZE[1] // 8
    ds = np.arange(d, dtype=np.float32) * DBOUND[2] + DBOUND[0]
    xs = np.linspace(0, IMAGE_SIZE[1] - 1, fW, dtype=np.float32)
    ys = np.linspace(0, IMAGE_SIZE[0] - 1, fH, dtype=np.float32)
    ds_g, ys_g, xs_g = np.meshgrid(ds, ys, xs, indexing='ij')
    return np.stack([xs_g, ys_g, ds_g], axis=-1)


def get_geometry(frustum, img_aug_matrices, camera2lidar, cam_intrinsics):
    num_cams = 6
    D_bins, fH, fW, _ = frustum.shape
    N = D_bins * fH * fW
    pts = frustum.reshape(-1, 3).astype(np.float32)

    aug_inv = np.linalg.inv(img_aug_matrices[:, :3, :3].astype(np.float64)).astype(np.float32)
    aug_trans = img_aug_matrices[:, :2, 2].astype(np.float32)
    xy = pts[None, :, :2] - aug_trans[:, None, :]
    xy = np.einsum('cij,cnj->cni', aug_inv[:, :2, :2], xy)

    fx = cam_intrinsics[:, 0, 0][:, None]
    fy = cam_intrinsics[:, 1, 1][:, None]
    cx = cam_intrinsics[:, 0, 2][:, None]
    cy = cam_intrinsics[:, 1, 2][:, None]
    depth = pts[None, :, 2]

    cam_x = (xy[:, :, 0] - cx) * depth / fx
    cam_y = (xy[:, :, 1] - cy) * depth / fy
    cam_z = np.broadcast_to(depth, (num_cams, N)).copy()
    cam_pts = np.stack([cam_x, cam_y, cam_z], axis=-1).astype(np.float32)

    R = camera2lidar[:, :3, :3].astype(np.float32)
    t_vec = camera2lidar[:, :3, 3].astype(np.float32)
    pts_3d = np.empty_like(cam_pts)
    for c in range(num_cams):
        np.matmul(cam_pts[c], R[c].T, out=pts_3d[c])
        pts_3d[c] += t_vec[c]
    return pts_3d.reshape(num_cams, D_bins, fH, fW, 3)


# ═══════════════════════════════════════════════════════════════════════════════
# Detection decoding
# ═══════════════════════════════════════════════════════════════════════════════

def decode_detections(detections: dict, score_threshold: float = 0.1):
    center = detections['center'][0]
    height = detections['height'][0]
    dim = detections['dim'][0]
    rot = detections['rot'][0]
    vel = detections['vel'][0]
    heatmap = detections['heatmap'][0]
    top_cls = detections['top_proposals_class'][0].astype(np.int64)

    sig = 1.0 / (1.0 + np.exp(-heatmap))
    if 'query_heatmap_score' in detections:
        qhs = detections['query_heatmap_score'][0]
    else:
        dhm = 1.0 / (1.0 + np.exp(-detections['dense_heatmap'][0]))
        padded = np.pad(dhm, ((0, 0), (1, 1), (1, 1)), mode='constant')
        from numpy.lib.stride_tricks import sliding_window_view
        lm = sliding_window_view(padded, (3, 3), axis=(1, 2)).max(axis=(-2, -1))
        nms = dhm * (dhm == lm)
        flat = nms.reshape(10, -1)
        qhs = flat[:, detections['top_proposals_index'][0].astype(np.int64)]

    one_hot = np.zeros((NUM_CLASSES, center.shape[1]), dtype=np.float32)
    one_hot[top_cls, np.arange(center.shape[1])] = 1.0
    scores = (sig * qhs * one_hot).max(axis=0)
    labels = (sig * qhs * one_hot).argmax(axis=0)

    cx = center[0] * OUT_SIZE_FACTOR * VOXEL_SIZE[0] + POINT_CLOUD_RANGE[0]
    cy = center[1] * OUT_SIZE_FACTOR * VOXEL_SIZE[1] + POINT_CLOUD_RANGE[1]
    dims = np.exp(dim)
    cz = height[0] - dims[2] * 0.5
    yaw = np.arctan2(rot[0], rot[1])

    boxes = []
    for i in range(len(scores)):
        if scores[i] < score_threshold:
            continue
        if cx[i] < POST_CENTER_RANGE[0] or cx[i] > POST_CENTER_RANGE[3] or \
           cy[i] < POST_CENTER_RANGE[1] or cy[i] > POST_CENTER_RANGE[4] or \
           cz[i] < POST_CENTER_RANGE[2] or cz[i] > POST_CENTER_RANGE[5]:
            continue
        boxes.append({
            'center': [float(cx[i]), float(cy[i]), float(cz[i])],
            'dims': [float(dims[0, i]), float(dims[1, i]), float(dims[2, i])],
            'yaw': float(yaw[i]),
            'vel': [float(vel[0, i]), float(vel[1, i])],
            'score': float(scores[i]),
            'label': int(labels[i]),
            'label_name': CLASS_NAMES[int(labels[i])],
        })
    boxes.sort(key=lambda x: x['score'], reverse=True)
    return boxes


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline class
# ═══════════════════════════════════════════════════════════════════════════════

class BEVFusionInference:
    """Standalone BEVFusion inference using OpenVINO xml/bin models."""

    def __init__(self, model_dir: str, ext_dir: str, device: str = 'GPU'):
        self.model_dir = Path(model_dir)
        self.ext_dir = Path(ext_dir)
        self.device = device
        self.gpu_sharing = False  # RemoteTensor GPU buffer sharing

        self.core = ov.Core()

        # Load extension libraries
        for so_name in [
            'bev_pool/build/libopenvino_bevpool_extension.so',
            'voxelize/build/libopenvino_voxelize_extension.so',
            'sparse_encoder/build/libopenvino_sparse_encoder_extension.so',
        ]:
            so_path = self.ext_dir / so_name
            if so_path.exists():
                self.core.add_extension(str(so_path))
                print(f"  Loaded extension: {so_path.name}")

        # GPU custom kernel config
        gpu_cfg = self.ext_dir / 'gpu_custom_layers.xml'
        if gpu_cfg.exists() and device.upper().startswith('GPU'):
            self.core.set_property('GPU', {'CONFIG_FILE': str(gpu_cfg)})
            print(f"  GPU custom kernels configured")

        # Get GPU RemoteContext for shared buffer compilation
        self.gpu_ctx = None
        if device.upper().startswith('GPU'):
            try:
                self.gpu_ctx = self.core.get_default_context('GPU')
                print(f"  GPU RemoteContext acquired for buffer sharing")
            except Exception as e:
                print(f"  Warning: Could not get GPU RemoteContext: {e}")

        # Compile models
        self.models = {}
        gpu_config = {'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_PRECISION_HINT': 'f16'}
        gpu_models = [
            # Individual models (legacy/fallback)
            'camera_backbone_neck', 'camera_backbone_neck_b6',
            'vtransform_dtransform', 'vtransform_dtransform_b6',
            'vtransform_depthnet', 'vtransform_depthnet_b6',
            'vtransform_downsample', 'fuser', 'bev_decoder', 'transfusion_head',
            # Merged models (preferred when available)
            'camera_encoder_b6', 'detection_head',
            # Full detection merge
            'full_detection',
            # Geometry model
            'geometry',
        ]

        # Geometry model compiled on GPU (pure math, fast on GPU)
        cpu_models = []

        for name in gpu_models:
            # Prefer INT8 variant when available
            xml_int8 = self.model_dir / f'{name}_int8.xml'
            xml = xml_int8 if xml_int8.exists() else self.model_dir / f'{name}.xml'
            if xml.exists():
                model = self.core.read_model(str(xml))
                if self.gpu_ctx is not None:
                    self.models[name] = self.core.compile_model(model, self.gpu_ctx, gpu_config)
                else:
                    cfg = gpu_config if device.upper().startswith('GPU') else {}
                    self.models[name] = self.core.compile_model(model, device, cfg)
                variant = 'INT8' if '_int8' in xml.name else 'FP16'
                print(f"  Compiled {name} ({variant}) on {device}")

        for name in cpu_models:
            xml = self.model_dir / f'{name}.xml'
            if xml.exists():
                model = self.core.read_model(str(xml))
                self.models[name] = self.core.compile_model(model, 'CPU')
                print(f"  Compiled {name} on CPU")

        # Custom extension models (compiled on their native devices)
        # Voxelize on GPU
        vox_xml = self.model_dir / 'voxelize.xml'
        if vox_xml.exists():
            m = self.core.read_model(str(vox_xml))
            if self.gpu_ctx is not None:
                self.models['voxelize'] = self.core.compile_model(m, self.gpu_ctx, {'PERFORMANCE_HINT': 'LATENCY'})
            else:
                self.models['voxelize'] = self.core.compile_model(m, 'GPU', {'PERFORMANCE_HINT': 'LATENCY'})
            print(f"  Compiled voxelize on GPU")

        # BEV pool on GPU
        bp_xml = self.model_dir / 'bev_pool.xml'
        if bp_xml.exists():
            m = self.core.read_model(str(bp_xml))
            if self.gpu_ctx is not None:
                self.models['bev_pool'] = self.core.compile_model(m, self.gpu_ctx, {'PERFORMANCE_HINT': 'LATENCY'})
            else:
                self.models['bev_pool'] = self.core.compile_model(m, 'GPU', {'PERFORMANCE_HINT': 'LATENCY'})
            print(f"  Compiled bev_pool on GPU")

        # Sparse encoder on CPU (with internal OpenCL)
        se_xml = self.model_dir / 'sparse_encoder.xml'
        se_bin = self.model_dir / 'sparse_encoder.bin'
        if se_xml.exists() and se_bin.exists():
            m = self.core.read_model(str(se_xml), str(se_bin))
            self.models['sparse_encoder'] = self.core.compile_model(m, 'CPU')
            print(f"  Compiled sparse_encoder on CPU")

        # Create infer requests
        self.requests = {
            name: cm.create_infer_request() for name, cm in self.models.items()
        }

        # Set up RemoteTensor GPU buffer sharing between models
        self._setup_gpu_sharing()

        # Frustum + vtransform params
        frust_path = self.model_dir / 'frustum.npy'
        self.frustum = np.load(str(frust_path)) if frust_path.exists() else create_frustum()
        self.vtransform_dx = np.load(str(self.model_dir / 'vtransform_dx.npy'))
        self.vtransform_bx = np.load(str(self.model_dir / 'vtransform_bx.npy'))
        self.vtransform_nx = np.load(str(self.model_dir / 'vtransform_nx.npy'))

        print(f"\nPipeline ready ({len(self.models)} models compiled)")

    def _setup_gpu_sharing(self):
        """Wire GPU models with shared RemoteTensors to eliminate CPU copies.

        Connections: camera_encoder → bev_pool → full_detection
                     geometry → bev_pool
        Data stays on GPU between these models.
        """
        if self.gpu_ctx is None:
            return

        needed = {'camera_encoder_b6', 'bev_pool', 'full_detection', 'geometry'}
        if not needed.issubset(self.models.keys()):
            return

        try:
            ctx = self.gpu_ctx
            cam_req = self.requests['camera_encoder_b6']
            geo_req = self.requests['geometry']
            bev_req = self.requests['bev_pool']
            det_req = self.requests['full_detection']

            # Shared buffers: camera_encoder outputs → bev_pool inputs
            self._depth_rt = ctx.create_tensor(ov.Type.f32, ov.Shape([6, 118, 32, 88]), {})
            self._context_rt = ctx.create_tensor(ov.Type.f32, ov.Shape([6, 80, 32, 88]), {})
            cam_req.set_tensor('depth_logits', self._depth_rt)
            cam_req.set_tensor('context_features', self._context_rt)
            bev_req.set_tensor('depth_logits', self._depth_rt)
            bev_req.set_tensor('context_feats', self._context_rt)

            # Shared buffer: geometry output → bev_pool input
            self._geom_rt = ctx.create_tensor(ov.Type.f32, ov.Shape([1993728, 3]), {})
            geo_req.set_tensor('geometry_points', self._geom_rt)
            bev_req.set_tensor('geom', self._geom_rt)

            # Shared buffer: bev_pool output → detection input
            self._camera_bev_rt = ctx.create_tensor(ov.Type.f32, ov.Shape([1, 80, 360, 360]), {})
            bev_req.set_tensor('bev_features', self._camera_bev_rt)
            det_req.set_tensor('camera_bev', self._camera_bev_rt)

            self.gpu_sharing = True
            print(f"  GPU buffer sharing enabled (4 RemoteTensors, ~82MB zero-copy)")
        except Exception as e:
            print(f"  Warning: GPU buffer sharing failed: {e}")
            self.gpu_sharing = False

    def _infer(self, name: str, inputs: list[np.ndarray]) -> dict:
        req = self.requests[name]
        cm = self.models[name]
        for i, inp in enumerate(inputs):
            req.set_input_tensor(i, ov.Tensor(np.ascontiguousarray(inp)))
        req.infer()
        result = {}
        for i in range(len(cm.outputs)):
            oname = cm.output(i).get_any_name()
            result[oname] = req.get_output_tensor(i).data.copy()
        return result

    def voxelize(self, points: np.ndarray):
        n = min(points.shape[0], MAX_POINTS)
        padded = np.zeros((MAX_POINTS, 5), dtype=np.float32)
        padded[:n, :min(5, points.shape[1])] = points[:n, :5]
        num_pts = np.array([n], dtype=np.int32)

        req = self.requests['voxelize']
        req.set_input_tensor(0, ov.Tensor(padded))
        req.set_input_tensor(1, ov.Tensor(num_pts))
        req.infer()

        result = req.get_output_tensor(0).data
        num_voxels = min(int(result[0, 0]), MAX_VOXELS)
        if num_voxels <= 0:
            return np.zeros((0, 5), dtype=np.float32), np.zeros((0, 4), dtype=np.int32)

        valid = result[1:num_voxels + 1]
        return valid[:, :5].copy().astype(np.float32), valid[:, 5:].copy().astype(np.int32)

    def sparse_encode(self, voxel_features: np.ndarray, coordinates: np.ndarray):
        N = min(voxel_features.shape[0], MAX_VOXELS)
        feat_pad = np.zeros((MAX_VOXELS, 5), dtype=np.float32)
        coord_pad = np.zeros((MAX_VOXELS, 4), dtype=np.int32)
        feat_pad[:N, :min(5, voxel_features.shape[1])] = voxel_features[:N]
        coord_pad[:N] = coordinates[:N]
        num_vox = np.array([N], dtype=np.int32)

        req = self.requests['sparse_encoder']
        req.set_input_tensor(0, ov.Tensor(feat_pad))
        req.set_input_tensor(1, ov.Tensor(coord_pad))
        req.set_input_tensor(2, ov.Tensor(num_vox))
        req.infer()
        return req.get_output_tensor(0).data.copy()

    def bev_pool(self, depth_logits, context_feats, geom):
        geom_flat = geom.reshape(-1, 3).astype(np.float32)
        req = self.requests['bev_pool']
        req.set_input_tensor(0, ov.Tensor(np.ascontiguousarray(depth_logits)))
        req.set_input_tensor(1, ov.Tensor(np.ascontiguousarray(context_feats)))
        req.set_input_tensor(2, ov.Tensor(np.ascontiguousarray(geom_flat)))
        req.infer()
        return req.get_output_tensor(0).data.copy()

    def run_lidar(self, points):
        t0 = time.time()
        vf, coords = self.voxelize(points)
        t_vox = time.time() - t0

        t0 = time.time()
        bev = self.sparse_encode(vf, coords)
        t_enc = time.time() - t0
        return bev, {'voxelize': t_vox, 'sparse_encode': t_enc}

    def run_camera(self, images, depth_maps, img_aug_matrices, camera2lidar, cam_intrinsics):
        times = {}

        # ── GPU buffer sharing path: data stays on GPU between models ──
        if self.gpu_sharing and 'camera_encoder_b6' in self.models and 'geometry' in self.models:
            # Geometry: compute per-sample, output stays on GPU via RemoteTensor
            t0 = time.time()
            aug_inv = np.linalg.inv(img_aug_matrices[:, :3, :3].astype(np.float64)).astype(np.float32)
            aug_trans = img_aug_matrices[:, :2, 2].astype(np.float32)
            combined = np.zeros((6, 14), dtype=np.float32)
            combined[:, 0] = aug_inv[:, 0, 0]
            combined[:, 1] = aug_inv[:, 0, 1]
            combined[:, 2] = aug_inv[:, 1, 0]
            combined[:, 3] = aug_inv[:, 1, 1]
            combined[:, 4:6] = aug_trans
            combined[:, 6] = cam_intrinsics[:, 0, 0]
            combined[:, 7] = cam_intrinsics[:, 1, 1]
            combined[:, 8] = cam_intrinsics[:, 0, 2]
            combined[:, 9] = cam_intrinsics[:, 1, 2]
            combined[:, 10:13] = camera2lidar[:, :3, 3].astype(np.float32)
            c2l_rot = camera2lidar[:, :3, :3].astype(np.float32)

            geo_req = self.requests['geometry']
            geo_req.set_tensor('combined_matrices', ov.Tensor(np.ascontiguousarray(combined)))
            geo_req.set_tensor('camera2lidar_rot', ov.Tensor(np.ascontiguousarray(c2l_rot)))
            geo_req.start_async()
            geo_req.wait()
            times['geometry'] = time.time() - t0

            # Camera encoder: output stays on GPU via RemoteTensor
            t0 = time.time()
            cam_req = self.requests['camera_encoder_b6']
            cam_req.set_tensor('images', ov.Tensor(np.ascontiguousarray(images)))
            cam_req.set_tensor('depth_maps', ov.Tensor(np.ascontiguousarray(depth_maps)))
            cam_req.start_async()
            cam_req.wait()
            times['camera_encoder'] = time.time() - t0

            # BEV pool: all 3 inputs already on GPU, output stays on GPU
            t0 = time.time()
            bev_req = self.requests['bev_pool']
            bev_req.start_async()
            bev_req.wait()
            times['bev_pool'] = time.time() - t0

            # camera_bev is already wired to full_detection via RemoteTensor
            # Return None to signal GPU sharing path
            if 'detection_head' in self.models or 'full_detection' in self.models:
                return None, times

            # Fallback: read from GPU if no merged detection
            cam_bev = bev_req.get_output_tensor(0).data.copy()
            return cam_bev, times

        # ── Standard path (no GPU sharing) ──
        t0 = time.time()
        if 'geometry' in self.models:
            # GPU-accelerated geometry computation
            aug_inv = np.linalg.inv(img_aug_matrices[:, :3, :3].astype(np.float64)).astype(np.float32)
            aug_trans = img_aug_matrices[:, :2, 2].astype(np.float32)
            # Pack parameters: [aug_inv_00,01,10,11, aug_tx,ty, fx,fy,cx,cy, t_x,t_y,t_z, 0]
            combined = np.zeros((6, 14), dtype=np.float32)
            combined[:, 0] = aug_inv[:, 0, 0]
            combined[:, 1] = aug_inv[:, 0, 1]
            combined[:, 2] = aug_inv[:, 1, 0]
            combined[:, 3] = aug_inv[:, 1, 1]
            combined[:, 4:6] = aug_trans
            combined[:, 6] = cam_intrinsics[:, 0, 0]  # fx
            combined[:, 7] = cam_intrinsics[:, 1, 1]  # fy
            combined[:, 8] = cam_intrinsics[:, 0, 2]  # cx
            combined[:, 9] = cam_intrinsics[:, 1, 2]  # cy
            combined[:, 10:13] = camera2lidar[:, :3, 3].astype(np.float32)
            c2l_rot = camera2lidar[:, :3, :3].astype(np.float32)

            req = self.requests['geometry']
            req.set_input_tensor(0, ov.Tensor(np.ascontiguousarray(combined)))
            req.set_input_tensor(1, ov.Tensor(np.ascontiguousarray(c2l_rot)))
            req.infer()
            D_bins, fH, fW = 118, 32, 88
            geom = req.get_output_tensor(0).data.copy().reshape(6, D_bins, fH, fW, 3)
        else:
            geom = get_geometry(self.frustum, img_aug_matrices, camera2lidar, cam_intrinsics)
        times['geometry'] = time.time() - t0

        # ── Merged camera encoder path (backbone + dtransform + depthnet in one model) ──
        if 'camera_encoder_b6' in self.models:
            t0 = time.time()
            req = self.requests['camera_encoder_b6']
            req.set_input_tensor(0, ov.Tensor(np.ascontiguousarray(images)))
            req.set_input_tensor(1, ov.Tensor(np.ascontiguousarray(depth_maps)))
            req.infer()
            dl = req.get_output_tensor(0).data.copy()
            cf = req.get_output_tensor(1).data.copy()
            times['camera_encoder'] = time.time() - t0
        else:
            # ── Legacy separate models path ──
            t0 = time.time()
            if 'camera_backbone_neck_b6' in self.models:
                res = self._infer('camera_backbone_neck_b6', [images])
                cam_feats = list(res.values())[0]
            else:
                parts = []
                for i in range(6):
                    res = self._infer('camera_backbone_neck', [images[i:i+1]])
                    parts.append(list(res.values())[0][0])
                cam_feats = np.stack(parts)
            times['backbone'] = time.time() - t0

            t0 = time.time()
            has_b6_dt = 'vtransform_dtransform_b6' in self.models
            has_b6_dn = 'vtransform_depthnet_b6' in self.models
            if has_b6_dt and has_b6_dn:
                dt_res = self._infer('vtransform_dtransform_b6', [depth_maps])
                df = list(dt_res.values())[0]
                combined = np.concatenate([cam_feats, df], axis=1)
                dn_res = self._infer('vtransform_depthnet_b6', [combined])
                daf = list(dn_res.values())[0]
                dl, cf = daf[:, :D], daf[:, D:]
            else:
                dl_l, cf_l = [], []
                for i in range(6):
                    dt_res = self._infer('vtransform_dtransform', [depth_maps[i:i+1]])
                    df = list(dt_res.values())[0]
                    combined = np.concatenate([cam_feats[i:i+1], df], axis=1)
                    dn_res = self._infer('vtransform_depthnet', [combined])
                    daf = list(dn_res.values())[0]
                    dl_l.append(daf[0, :D])
                    cf_l.append(daf[0, D:])
                dl, cf = np.stack(dl_l), np.stack(cf_l)
            times['depth'] = time.time() - t0

        t0 = time.time()
        cam_bev = self.bev_pool(dl, cf, geom)
        times['bev_pool'] = time.time() - t0

        # If using merged detection_head or full_detection, they handle downsample internally.
        # Return pre-downsample BEV so detection_head can do it.
        if 'detection_head' in self.models or 'full_detection' in self.models:
            return cam_bev, times

        t0 = time.time()
        ds_res = self._infer('vtransform_downsample', [cam_bev])
        cam_bev_ds = list(ds_res.values())[0]
        times['downsample'] = time.time() - t0

        return cam_bev_ds, times

    def load_data(self, info: dict, data_root: Path):
        """Load all data from disk (CPU/IO bound, ~28ms)."""
        points = load_lidar_points(info, data_root)
        images, aug_mats = load_camera_images(info, data_root)
        l2i, c2l, intr = get_camera_transforms(info)
        return {'points': points, 'images': images, 'aug_mats': aug_mats,
                'l2i': l2i, 'c2l': c2l, 'intr': intr}

    def run(self, info: dict, data_root: Path, score_threshold: float = 0.1,
            preloaded_data: dict = None, prefetch_fn=None):
        """Run inference. If prefetch_fn is provided, it will be launched in a
        background thread to overlap with the GPU-heavy fusion stage."""
        timings = {}

        if preloaded_data is not None:
            points = preloaded_data['points']
            images = preloaded_data['images']
            aug_mats = preloaded_data['aug_mats']
            l2i = preloaded_data['l2i']
            c2l = preloaded_data['c2l']
            intr = preloaded_data['intr']
            timings['data_loading'] = 0.0  # already loaded
        else:
            t0 = time.time()
            points = load_lidar_points(info, data_root)
            images, aug_mats = load_camera_images(info, data_root)
            l2i, c2l, intr = get_camera_transforms(info)
            timings['data_loading'] = time.time() - t0

        # ── Overlap depth_maps (CPU, ~10ms) with lidar_branch start ──
        # depth_maps is pure CPU (numpy) and can overlap with voxelize + sparse precomp
        depth_result = [None, 0.0]  # [depth_maps, elapsed]

        def depth_worker():
            t = time.time()
            depth_result[0] = create_depth_maps(points, l2i, aug_mats)
            depth_result[1] = time.time() - t

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=1) as executor:
            f_depth = executor.submit(depth_worker)
            # Start lidar while depth computes in parallel
            lidar_bev, lidar_times = self.run_lidar(points)
            f_depth.result()  # ensure depth is done
        parallel_wall = time.time() - t0

        timings['lidar_branch'] = lidar_times.get('voxelize', 0) + lidar_times.get('sparse_encode', 0)
        timings.update({f'lidar_{k}': v for k, v in lidar_times.items()})
        timings['depth_maps'] = depth_result[1]

        depth_maps = depth_result[0]

        t0 = time.time()
        cam_bev, cam_times = self.run_camera(images, depth_maps, aug_mats, c2l, intr)
        timings['camera_branch'] = time.time() - t0
        timings.update({f'cam_{k}': v for k, v in cam_times.items()})

        # ── Launch prefetch for next frame right before GPU-heavy fusion ──
        prefetch_future = None
        if prefetch_fn is not None:
            prefetch_future = ThreadPoolExecutor(max_workers=1).submit(prefetch_fn)

        t0 = time.time()
        # ── Fully merged detection (downsample + fuser + decoder + transfusion) ──
        if 'full_detection' in self.models:
            req = self.requests['full_detection']
            # GPU sharing: camera_bev already wired via RemoteTensor, only set lidar_bev
            if self.gpu_sharing and cam_bev is None:
                req.set_tensor('lidar_bev', ov.Tensor(np.ascontiguousarray(lidar_bev)))
                req.start_async()
                req.wait()
            else:
                req.set_input_tensor(0, ov.Tensor(np.ascontiguousarray(cam_bev)))
                req.set_input_tensor(1, ov.Tensor(np.ascontiguousarray(lidar_bev)))
                req.infer()
            head_res = {}
            output_names = ['center', 'height', 'dim', 'rot', 'vel', 'heatmap',
                           'query_heatmap_score', 'dense_heatmap',
                           'top_proposals_index', 'top_proposals_class']
            for i, name in enumerate(output_names):
                head_res[name] = req.get_output_tensor(i).data.copy()
        elif 'detection_head' in self.models:
            # Merged detection pipeline (downsample + fuser + decoder) + separate head
            req = self.requests['detection_head']
            req.set_input_tensor(0, ov.Tensor(np.ascontiguousarray(cam_bev)))
            req.set_input_tensor(1, ov.Tensor(np.ascontiguousarray(lidar_bev)))
            req.infer()
            decoded = req.get_output_tensor(0).data.copy()
            head_res = self._infer('transfusion_head', [decoded])
        else:
            # Legacy separate models path
            fused_input = np.concatenate([cam_bev, lidar_bev], axis=1)
            fuser_res = self._infer('fuser', [fused_input])
            fused_bev = list(fuser_res.values())[0]
            dec_res = self._infer('bev_decoder', [fused_bev])
            decoded = list(dec_res.values())[0]
            head_res = self._infer('transfusion_head', [decoded])
        timings['fusion_detection'] = time.time() - t0

        t0 = time.time()
        boxes = decode_detections(head_res, score_threshold)
        timings['decode'] = time.time() - t0

        timings['total'] = timings['data_loading'] + parallel_wall + \
                           timings['camera_branch'] + \
                           timings['fusion_detection'] + timings['decode']
        return boxes, timings, prefetch_future


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="BEVFusion standalone inference")
    parser.add_argument('--model-dir', default='openvino_model')
    parser.add_argument('--ext-dir', default='openvino_extensions')
    parser.add_argument('--data-pkl', default=None,
                        help='Path to dataset info pickle file')
    parser.add_argument('--data-root', default=None,
                        help='Dataset root directory for resolving data paths')
    parser.add_argument('--device', default='GPU')
    parser.add_argument('--num-samples', type=int, default=5)
    parser.add_argument('--score-threshold', type=float, default=0.1)
    args = parser.parse_args()

    print("=" * 70)
    print("BEVFusion Standalone Inference")
    print("=" * 70)

    pipeline = BEVFusionInference(args.model_dir, args.ext_dir, args.device)

    with open(args.data_pkl, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos'] if isinstance(data, dict) else data
    print(f"\nLoaded {len(infos)} samples")

    data_root = Path(args.data_root) if args.data_root else None
    n = min(args.num_samples, len(infos))

    total_time = 0
    preloaded = None  # first frame has no prefetched data

    for i in range(n):
        # Create prefetch callable for next frame (will be launched during fusion)
        prefetch_fn = None
        if i + 1 < n:
            next_info = infos[i + 1]
            prefetch_fn = lambda info=next_info: pipeline.load_data(info, data_root)

        t_frame = time.time()
        boxes, timings, prefetch_future = pipeline.run(
            infos[i], data_root, args.score_threshold,
            preloaded_data=preloaded, prefetch_fn=prefetch_fn)
        frame_wall = time.time() - t_frame

        # Collect prefetched data (should already be done - fusion took 55ms, data takes 28ms)
        if prefetch_future is not None:
            preloaded = prefetch_future.result()
        else:
            preloaded = None

        print(f"\nSample {i}: {len(boxes)} detections, "
              f"wall={frame_wall*1000:.0f}ms "
              f"(lidar={timings['lidar_branch']*1000:.0f}ms, "
              f"depth={timings['depth_maps']*1000:.0f}ms, "
              f"camera={timings['camera_branch']*1000:.0f}ms, "
              f"fusion={timings['fusion_detection']*1000:.0f}ms)")
        if i > 0:
            total_time += frame_wall
        if boxes:
            print(f"  Top: {boxes[0]['label_name']} score={boxes[0]['score']:.3f}")

    if n > 1:
        avg = total_time / (n - 1)
        print(f"\nAvg (excl warmup): {avg*1000:.0f}ms = {1.0/avg:.2f} FPS")


if __name__ == '__main__':
    main()
