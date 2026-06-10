#!/usr/bin/env python3
"""
FlashOCC PyTorch-vs-OpenVINO Inference Comparison (nuScenes).

Runs both pipelines on the same nuScenes samples and reports metrics in a
format aligned with run_inference_flashocc.py:
  - Module-wise latency & throughput
  - Occupancy output statistics
  - NaN/Inf diagnostics
  - Side-by-side backend comparison

Pipelines:
  1) PyTorch CUDA (original FlashOCC model: config + checkpoint)
  2) OpenVINO split (image_encoder.xml + bev_trunk.xml + numpy bev_pool)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import multiprocessing as mp
import pickle
import sys
import time
import warnings
from collections import deque
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_PROJECTS_DIR = _REPO_ROOT / "projects"
for _p in (_REPO_ROOT, _PROJECTS_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

import cv2
import numpy as np
import openvino as ov
import openvino.opset13 as ov_opset13
from openvino import Type as OVType


def _ov_prec(s: str | None):
    """Convert 'f32'/'f16' string to ov.Type for INFERENCE_PRECISION_HINT.
    Using a plain string is silently ignored by some GPU drivers."""
    if s in ('f32', 'FP32'):
        return OVType.f32
    if s in ('f16', 'FP16'):
        return OVType.f16
    return s  # passthrough for None or 'auto'


_DEFAULT_OV_GPU_CONFIG_XML = (_REPO_ROOT / 'openvino_extensions' / 'gpu_custom_layers.xml').resolve()
_PANTERLAKE_OV_GPU_CONFIG_XML = (_REPO_ROOT / 'openvino_extensions' / 'bev_pool' / 'bev_pool_gpu_panterlake.xml').resolve()
_OV_CACHE_DIR = (_REPO_ROOT / 'debug_output' / 'ov_cache').resolve()


def _ensure_ov_cache_dir() -> str:
    _OV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return str(_OV_CACHE_DIR)


def _apply_common_ov_config(cfg: dict, device: str) -> dict:
    if device.upper().startswith('GPU'):
        cfg.setdefault('PERFORMANCE_HINT', 'LATENCY')
        cfg.setdefault('CACHE_DIR', _ensure_ov_cache_dir())
    return cfg


def _select_bevpool_gpu_config(core: ov.Core, device: str, gpu_config_xml: str | None,
                               verbose: bool = False) -> str | None:
    if not gpu_config_xml or not device.upper().startswith('GPU'):
        return gpu_config_xml

    cfg_path = Path(gpu_config_xml).resolve()
    if cfg_path != _DEFAULT_OV_GPU_CONFIG_XML or not _PANTERLAKE_OV_GPU_CONFIG_XML.exists():
        return str(cfg_path)

    try:
        gpu_device_id = str(core.get_property('GPU', 'GPU_DEVICE_ID')).lower()
        full_name = str(core.get_property('GPU', 'FULL_DEVICE_NAME'))
    except Exception as exc:
        if verbose:
            print(f'  [BEVPoolOVSO] Could not query GPU device properties: {exc}')
        return str(cfg_path)

    if gpu_device_id == '0xb08f':
        if verbose:
            print(f'  [BEVPoolOVSO] Auto-selecting Panther Lake GPU config for {full_name}')
        return str(_PANTERLAKE_OV_GPU_CONFIG_XML)

    return str(cfg_path)

try:
    import onnx
    from onnx import helper as _onnx_helper
    from onnx import TensorProto as _ONNX_TENSOR_PROTO
    _ONNX_AVAILABLE = True
except Exception:
    onnx = None
    _onnx_helper = None
    _ONNX_TENSOR_PROTO = None
    _ONNX_AVAILABLE = False

# Optional GPU BEV pool (pyopencl-based)
try:
    _OV_EXT_DIR = Path(__file__).resolve().parent / 'openvino_extensions'
    import sys as _sys
    if str(_OV_EXT_DIR) not in _sys.path:
        _sys.path.insert(0, str(_OV_EXT_DIR))
    from bev_pool_opencl import BEVPoolOpenCL as _BEVPoolOpenCL
    from bev_pool_opencl import available as _bev_pool_opencl_available
    _BEV_POOL_OPENCL_LOADED = True
except Exception as _e:
    _BEV_POOL_OPENCL_LOADED = False
    _BEVPoolOpenCL = None
    _bev_pool_opencl_available = lambda: False


GRID_CONFIG = {
    'x': [-40.0, 40.0, 0.4],
    'y': [-40.0, 40.0, 0.4],
    'z': [-1.0, 5.4, 6.4],
    'depth': [1.0, 45.0, 1.0],
}

INPUT_H, INPUT_W = 256, 704
SRC_H, SRC_W = 900, 1600
DOWNSAMPLE = 16
FEAT_H = INPUT_H // DOWNSAMPLE
FEAT_W = INPUT_W // DOWNSAMPLE
D = round((GRID_CONFIG['depth'][1] - GRID_CONFIG['depth'][0]) / GRID_CONFIG['depth'][2])
NX = round((GRID_CONFIG['x'][1] - GRID_CONFIG['x'][0]) / GRID_CONFIG['x'][2])
NY = round((GRID_CONFIG['y'][1] - GRID_CONFIG['y'][0]) / GRID_CONFIG['y'][2])
NUM_C = 64
NUM_CLS = 18
FREE_CLASS = 17

CAMERA_NAMES = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
]

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

OCC_CLASSES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free',
]


def _resize_scale() -> float:
    return INPUT_W / SRC_W


def preprocess_image(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = _resize_scale()
    new_w = INPUT_W
    new_h = int(round(SRC_H * scale))
    img_rs = cv2.resize(img_bgr, (new_w, new_h))

    crop_top = new_h - INPUT_H
    img_crop = img_rs[crop_top:, :, :]

    post_rot = np.array([
        [scale, 0.0, 0.0],
        [0.0, scale, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    post_trans = np.array([0.0, -float(crop_top), 0.0], dtype=np.float32)

    img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_norm = (img_rgb - IMG_MEAN) / IMG_STD
    return img_norm.transpose(2, 0, 1), post_rot, post_trans


def _quat_wxyz_to_rot(q: list[float] | np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    w, x, y, z = q
    n = np.linalg.norm(q)
    if n == 0:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = q / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def _se3_from_qt(rotation_wxyz: list[float] | np.ndarray, translation_xyz: list[float] | np.ndarray) -> np.ndarray:
    t = np.asarray(translation_xyz, dtype=np.float32)
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = _quat_wxyz_to_rot(rotation_wxyz)
    m[:3, 3] = t
    return m


def resolve_nuscenes_image_path(data_root: Path, data_path: str) -> Path:
    raw = Path(data_path)
    if raw.is_absolute() and raw.exists():
        return raw

    normalized_parts = [p for p in raw.parts if p not in ('.', '')]
    normalized = Path(*normalized_parts) if normalized_parts else raw

    candidates = [data_root / normalized, data_root.parent / normalized]
    if normalized.parts[:2] == ('data', 'nuscenes'):
        stripped = Path(*normalized.parts[2:]) if len(normalized.parts) > 2 else Path()
        if stripped != Path():
            candidates.insert(0, data_root / stripped)

    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


def resolve_nuscenes_label_path(data_root: Path, label_path: str) -> Path:
    raw = Path(label_path)
    if raw.is_absolute() and raw.exists():
        return raw

    normalized_parts = [p for p in raw.parts if p not in ('.', '')]
    normalized = Path(*normalized_parts) if normalized_parts else raw

    candidates = [data_root / normalized, data_root.parent / normalized]
    if normalized.parts[:2] == ('data', 'nuscenes'):
        stripped = Path(*normalized.parts[2:]) if len(normalized.parts) > 2 else Path()
        if stripped != Path():
            candidates.insert(0, data_root / stripped)

    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


def load_gt_occ_label(
    info: dict,
    data_root: Path,
    gt_key: str | None = None,
    use_mask: str = 'lidar',   # 'lidar', 'camera', 'both', or 'none'
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (semantics, eval_mask) where eval_mask is a boolean array
    selecting the voxels to include in GT evaluation (Occ3D mask_lidar /
    mask_camera). Returns (gt, None) when no mask is available or
    use_mask='none'."""
    key_candidates = [gt_key] if gt_key else []
    key_candidates += ['occ_path', 'occ_gt_path', 'gt_path', 'voxel_path', 'semantics_path', 'labels_path']
    key_candidates = [k for k in key_candidates if k]

    found_key = None
    for key in key_candidates:
        if key in info:
            found_key = key
            break
    if found_key is None:
        raise KeyError(f"No GT path key found in sample info. Tried: {key_candidates}")

    gt_file = resolve_nuscenes_label_path(data_root, str(info[found_key]))
    if not gt_file.exists():
        raise FileNotFoundError(f"GT file not found: {gt_file}")

    eval_mask: np.ndarray | None = None

    if gt_file.suffix == '.npy':
        gt = np.load(gt_file)
    elif gt_file.suffix == '.npz':
        npz_data = np.load(gt_file)
        preferred = ['semantics', 'semantic', 'voxel_label', 'labels', 'occ', 'gt']
        picked = next((name for name in preferred if name in npz_data), None)
        if picked is None:
            keys = list(npz_data.keys())
            if not keys:
                raise ValueError(f"Empty GT npz file: {gt_file}")
            picked = keys[0]
        gt = npz_data[picked]

        # Load Occ3D evaluation masks when available
        if use_mask != 'none':
            mask_lidar = npz_data.get('mask_lidar', None)
            mask_camera = npz_data.get('mask_camera', None)
            if use_mask == 'lidar' and mask_lidar is not None:
                eval_mask = mask_lidar.astype(bool)
            elif use_mask == 'camera' and mask_camera is not None:
                eval_mask = mask_camera.astype(bool)
            elif use_mask == 'both' and mask_lidar is not None and mask_camera is not None:
                eval_mask = mask_lidar.astype(bool) | mask_camera.astype(bool)
            elif use_mask == 'lidar' and mask_lidar is None:
                pass  # fall through silently — .npy GT has no mask
    else:
        raise ValueError(f"Unsupported GT label file type: {gt_file}")

    if gt.ndim != 3:
        raise ValueError(f"Unexpected GT label shape in {gt_file}: {gt.shape}")

    return gt.astype(np.uint8, copy=False), eval_mask


_DEFAULT_SAMPLE_CACHE_DIR = (_REPO_ROOT / 'debug_output' / 'sample_input_cache').resolve()
_SAMPLE_CACHE_VERSION = 'v1'


def _sample_cache_key(info: dict) -> str:
    cams = info['cams']
    key_parts = [
        _SAMPLE_CACHE_VERSION,
        str(INPUT_W),
        str(INPUT_H),
        str(SRC_W),
        str(SRC_H),
    ]
    for cam in CAMERA_NAMES:
        key_parts.append(str(cams[cam]['data_path']))
    return hashlib.sha1('|'.join(key_parts).encode('utf-8')).hexdigest()


def _sample_cache_path(cache_dir: Path | str, info: dict) -> Path:
    cache_dir = Path(cache_dir)
    return cache_dir / f"{_sample_cache_key(info)}.npz"


def load_sample_inputs(info: dict, data_root: Path):
    images, post_rots, post_trans = [], [], []
    sensor2egos, ego2globals, intrinsics = [], [], []

    cams = info['cams']
    for cam in CAMERA_NAMES:
        cd = cams[cam]
        image_path = resolve_nuscenes_image_path(data_root, cd['data_path'])
        raw = cv2.imread(str(image_path))
        if raw is None:
            raise FileNotFoundError(
                f"Failed to read image for {cam}: {image_path}\n"
                f"Original data_path from pkl: {cd['data_path']}"
            )

        img_chw, p_rot, p_tr = preprocess_image(raw)
        images.append(img_chw)
        post_rots.append(p_rot)
        post_trans.append(p_tr)

        sensor2egos.append(_se3_from_qt(cd['sensor2ego_rotation'], cd['sensor2ego_translation']))
        ego2globals.append(_se3_from_qt(cd['ego2global_rotation'], cd['ego2global_translation']))
        intrinsics.append(np.array(cd['cam_intrinsic'], dtype=np.float32))

    return {
        'images': np.stack(images),
        'post_rots': np.stack(post_rots),
        'post_trans': np.stack(post_trans),
        'sensor2egos': np.stack(sensor2egos),
        'ego2globals': np.stack(ego2globals),
        'intrins': np.stack(intrinsics),
    }


def load_sample_inputs_cached(info: dict, data_root: Path, cache_dir: Path | str | None = None):
    if cache_dir is None:
        return load_sample_inputs(info, data_root)

    cache_path = _sample_cache_path(cache_dir, info)
    if cache_path.exists():
        with np.load(cache_path) as cached:
            return {key: cached[key] for key in cached.files}

    sample = load_sample_inputs(info, data_root)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix('.tmp.npz')
    np.savez(tmp_path, **sample)
    tmp_path.replace(cache_path)
    return sample


def _compute_geom_sensor2ego(
        post_rots: np.ndarray,
        post_trans: np.ndarray,
        sensor2egos: np.ndarray,
        intrinsics: np.ndarray,
) -> np.ndarray:
    """Compute 3-D frustum geometry in the EGO frame, matching PyTorch get_ego_coor.

    IMPORTANT: uses sensor2ego ONLY (no ego2global / key_ego_inv).
    The FlashOCC view-transformer (LSSViewTransformerBEVDepth.get_ego_coor) places
    all frustum points in the per-camera sensor2ego frame WITHOUT going through
    ego2global.  Passing ego2global-based geometry to the BEV pool gives wrong
    BEV coordinates and is the root cause of the OV vs PT accuracy gap.

    Returns
    -------
    geom : float32 [6 * D * FEAT_H * FEAT_W, 3]  (x, y, z in ego frame)
    """
    feat_xs = np.linspace(0, INPUT_W - 1, FEAT_W, dtype=np.float32)
    feat_ys = np.linspace(0, INPUT_H - 1, FEAT_H, dtype=np.float32)
    depth_vals = (np.arange(D, dtype=np.float32) * GRID_CONFIG['depth'][2]
                  + GRID_CONFIG['depth'][0])

    d_idx, h_idx, w_idx = np.meshgrid(
        np.arange(D), np.arange(FEAT_H), np.arange(FEAT_W), indexing='ij')
    d_flat = depth_vals[d_idx].ravel()
    px_flat = feat_xs[w_idx].ravel()
    py_flat = feat_ys[h_idx].ravel()
    n_pts = D * FEAT_H * FEAT_W

    geom = np.empty((6 * n_pts, 3), dtype=np.float32)

    for cam in range(6):
        post_rot = post_rots[cam]          # (3, 3)
        post_tr  = post_trans[cam]         # (3,)
        k        = intrinsics[cam]         # (3, 3)

        # 1. Undo image-space augmentation (post_trans / post_rots)
        pts_aug  = np.stack([px_flat, py_flat, np.ones(n_pts, np.float32)], 0)  # (3, n)
        pts      = pts_aug - post_tr[:, None]
        pts_orig = np.linalg.inv(post_rot) @ pts   # back to original pixel space

        # 2. Lift: pixel (u, v) → camera-frame 3-D at given depths
        cx, cy = k[0, 2], k[1, 2]
        fx, fy = k[0, 0], k[1, 1]
        cam_x  = (pts_orig[0] - cx) * d_flat / fx
        cam_y  = (pts_orig[1] - cy) * d_flat / fy
        cam_z  = d_flat
        cam_pts = np.stack([cam_x, cam_y, cam_z], 0)  # (3, n)

        # 3. Camera → ego  (sensor2ego only — NO ego2global, NO key_ego_inv)
        #    Matches: combine = sensor2ego[:3,:3] @ K^{-1};
        #             pts = combine @ pts; pts += sensor2ego[:3, 3]
        R_s2e = sensor2egos[cam][:3, :3]   # (3, 3)
        t_s2e = sensor2egos[cam][:3,  3]   # (3,)
        ego_pts = R_s2e @ cam_pts + t_s2e[:, None]  # (3, n)

        s = cam * n_pts
        geom[s: s + n_pts, :] = ego_pts.T.astype(np.float32, copy=False)

    return geom


def bev_pool_v2_numpy(depth: np.ndarray,
                      tran_feat: np.ndarray,
                      post_rots: np.ndarray,
                      post_trans: np.ndarray,
                      sensor2egos: np.ndarray,
                      ego2globals: np.ndarray,
                      intrinsics: np.ndarray) -> np.ndarray:
    """Original numpy BEV pool using ego2global geometry."""
    bev_feat = np.zeros((NUM_C, NY, NX), dtype=np.float32)

    feat_xs = np.linspace(0, INPUT_W - 1, FEAT_W, dtype=np.float32)
    feat_ys = np.linspace(0, INPUT_H - 1, FEAT_H, dtype=np.float32)
    depth_vals = (np.arange(D, dtype=np.float32) * GRID_CONFIG['depth'][2] + GRID_CONFIG['depth'][0])

    d_idx, h_idx, w_idx = np.meshgrid(np.arange(D), np.arange(FEAT_H), np.arange(FEAT_W), indexing='ij')
    d_val = depth_vals[d_idx]
    px = feat_xs[w_idx]
    py = feat_ys[h_idx]
    n_pts = D * FEAT_H * FEAT_W

    d_flat = d_val.ravel()
    px_flat = px.ravel()
    py_flat = py.ravel()

    key_ego_inv = np.linalg.inv(ego2globals[0])

    for cam in range(6):
        post_rot = post_rots[cam]
        post_tr = post_trans[cam]
        k = intrinsics[cam]

        pts_aug = np.stack([px_flat, py_flat, np.ones(n_pts, np.float32)], 0)
        pts = pts_aug - post_tr[:, None]
        pts_orig = np.linalg.inv(post_rot) @ pts

        cx, cy = k[0, 2], k[1, 2]
        fx, fy = k[0, 0], k[1, 1]
        cam_x = (pts_orig[0] - cx) * d_flat / fx
        cam_y = (pts_orig[1] - cy) * d_flat / fy
        cam_z = d_flat
        pts_cam = np.stack([cam_x, cam_y, cam_z, np.ones_like(cam_x)], 0)

        cam_to_key_ego = key_ego_inv @ ego2globals[cam] @ sensor2egos[cam]
        pts_key_ego = cam_to_key_ego @ pts_cam

        bev_xi = ((pts_key_ego[0] - GRID_CONFIG['x'][0]) / GRID_CONFIG['x'][2]).astype(np.int32)
        bev_yi = ((pts_key_ego[1] - GRID_CONFIG['y'][0]) / GRID_CONFIG['y'][2]).astype(np.int32)

        valid = ((bev_xi >= 0) & (bev_xi < NX) & (bev_yi >= 0) & (bev_yi < NY))
        vi = np.where(valid)[0]
        if vi.size == 0:
            continue

        flat_bev = bev_yi[vi] * NX + bev_xi[vi]
        w = depth[cam].ravel()[vi]
        feat_idx = vi % (FEAT_H * FEAT_W)
        feat_flat = tran_feat[cam].reshape(NUM_C, -1)
        contrib = feat_flat[:, feat_idx] * w[None, :]
        np.add.at(bev_feat.reshape(NUM_C, -1), (slice(None), flat_bev), contrib)

    return bev_feat[None]


class BEVPoolOVSO:
    def __init__(
        self,
        extension_so: str,
        device: str = 'GPU',
        gpu_config_xml: str | None = None,
        gpu_precision: str = 'auto',
        inference_precision: str | None = None,
        verbose: bool = False,
        shared_core: 'ov.Core | None' = None,
    ):
        if not _ONNX_AVAILABLE:
            raise RuntimeError('SO BEV pool backend requires onnx Python package to be installed.')

        if shared_core is not None:
            # Reuse caller's Core so BEV pool and trunk share the same GPU context —
            # eliminates cross-context sync overhead (~17 ms per frame).
            self._core = shared_core
            self._core.add_extension(str(extension_so))
        else:
            self._core = ov.Core()
            self._core.add_extension(str(extension_so))

        self._n_pts = D * FEAT_H * FEAT_W
        feat_xs = np.linspace(0, INPUT_W - 1, FEAT_W, dtype=np.float32)
        feat_ys = np.linspace(0, INPUT_H - 1, FEAT_H, dtype=np.float32)
        depth_vals = (np.arange(D, dtype=np.float32) * GRID_CONFIG['depth'][2] + GRID_CONFIG['depth'][0])
        d_idx, h_idx, w_idx = np.meshgrid(np.arange(D), np.arange(FEAT_H), np.arange(FEAT_W), indexing='ij')
        self._d_flat = depth_vals[d_idx].ravel()
        self._px_flat = feat_xs[w_idx].ravel()
        self._py_flat = feat_ys[h_idx].ravel()
        self._pts_aug = np.stack([self._px_flat, self._py_flat, np.ones(self._n_pts, np.float32)], 0)
        self._geom = np.empty((6 * self._n_pts, 3), dtype=np.float32)

        model = self._core.read_model(self._build_submodel_onnx())

        # Custom OpenVINO extensions are not supported on NPU; use CPU for extension, main models on NPU
        compile_device = device
        if device.upper() == 'NPU':
            compile_device = 'CPU'
            if verbose:
                print(f'  [BEVPoolOVSO] NPU backend does not support custom extensions.')
                print(f'  [BEVPoolOVSO] Running BEV pool extension on CPU (main models will use NPU).')

        gpu_config_xml = _select_bevpool_gpu_config(self._core, compile_device, gpu_config_xml, verbose)

        cfg = {}
        if compile_device.upper().startswith('GPU'):
            if gpu_config_xml:
                cfg['CONFIG_FILE'] = str(gpu_config_xml)
            if gpu_precision != 'auto':
                cfg['INFERENCE_PRECISION_HINT'] = _ov_prec(gpu_precision)
        if inference_precision in ('f32', 'f16', 'FP32', 'FP16'):
            cfg['INFERENCE_PRECISION_HINT'] = _ov_prec(inference_precision)

        _apply_common_ov_config(cfg, compile_device)
        self._compiled = self._core.compile_model(model, compile_device, cfg)
        self._req = self._compiled.create_infer_request()
        self.device = compile_device
        # Keep compile config for shared-core rebinding
        self._ext_so_path = str(extension_so)
        self._compile_device = compile_device
        self._compile_cfg = cfg

        if verbose:
            print(f'  [BEVPoolOVSO] Extension: {extension_so}')
            if gpu_config_xml:
                print(f'  [BEVPoolOVSO] GPU config: {gpu_config_xml}')
            print(f'  [BEVPoolOVSO] Device: {self.device}')

    def _rebind_to_shared_core(self, shared_core: 'ov.Core') -> None:
        """Recompile BEV pool on a shared OV Core to eliminate GPU cross-context sync overhead."""
        shared_core.add_extension(self._ext_so_path)
        onnx_path = self._build_submodel_onnx()
        model = shared_core.read_model(onnx_path)
        self._compiled = shared_core.compile_model(model, self._compile_device, self._compile_cfg)
        self._req = self._compiled.create_infer_request()
        self._core = shared_core

    @staticmethod
    def _build_submodel_onnx() -> str:
        depth = _onnx_helper.make_tensor_value_info('depth', _ONNX_TENSOR_PROTO.FLOAT, [6, 44, 16, 44])
        feat = _onnx_helper.make_tensor_value_info('feat', _ONNX_TENSOR_PROTO.FLOAT, [6, 64, 16, 44])
        geom = _onnx_helper.make_tensor_value_info('geom', _ONNX_TENSOR_PROTO.FLOAT, [185856, 3])
        out = _onnx_helper.make_tensor_value_info('bev', _ONNX_TENSOR_PROTO.FLOAT, [1, 64, 200, 200])

        nodes = [
            _onnx_helper.make_node('BEVPoolBinSort', ['geom'], ['packed'], domain='bevfusion'),
            # bev_pool_v2.cl now writes in Y-major [C,NY,NX] order directly —
            # no Transpose needed; output is C-contiguous, saving ~8ms per frame.
            _onnx_helper.make_node('BEVPoolV2', ['depth', 'feat', 'packed'], ['bev'], domain='bevfusion'),
        ]

        graph = _onnx_helper.make_graph(nodes, 'bevpool_split', [depth, feat, geom], [out])
        model = _onnx_helper.make_model(
            graph,
            opset_imports=[
                _onnx_helper.make_operatorsetid('', 13),
                _onnx_helper.make_operatorsetid('bevfusion', 1),
            ],
            producer_name='flashocc-bevpool-so',
        )
        model.ir_version = 7

        onnx_dir = _REPO_ROOT / 'debug_output' / 'generated_models'
        onnx_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = onnx_dir / 'bevpool_split.onnx'
        onnx.save(model, str(onnx_path))
        return str(onnx_path)

    def _compute_geom(self, sample: dict) -> np.ndarray:
        """Compute geometry using original ego2global-based transformation."""
        key_ego_inv = np.linalg.inv(sample['ego2globals'][0])
        geom = self._geom

        for cam in range(6):
            post_rot = sample['post_rots'][cam]
            post_tr = sample['post_trans'][cam]
            k = sample['intrins'][cam]

            pts = self._pts_aug - post_tr[:, None]
            pts_orig = np.linalg.inv(post_rot) @ pts

            cx, cy = k[0, 2], k[1, 2]
            fx, fy = k[0, 0], k[1, 1]
            cam_x = (pts_orig[0] - cx) * self._d_flat / fx
            cam_y = (pts_orig[1] - cy) * self._d_flat / fy
            cam_z = self._d_flat
            pts_cam = np.stack([cam_x, cam_y, cam_z, np.ones_like(cam_x)], 0)

            cam_to_key_ego = key_ego_inv @ sample['ego2globals'][cam] @ sample['sensor2egos'][cam]
            pts_key_ego = (cam_to_key_ego @ pts_cam)[:3].T.astype(np.float32, copy=False)

            s = cam * self._n_pts
            e = s + self._n_pts
            geom[s:e, :] = pts_key_ego

        return geom

    def compute_geom(self, sample: dict) -> np.ndarray:
        """Public wrapper — compute geometry array for a given sample.

        Callers can invoke this *before* or *in parallel with* other blocking work
        (e.g., encoder GPU inference) so that the result is ready by the time
        :meth:`run_with_geom` is called, eliminating geom from the critical path.
        """
        return self._compute_geom(sample)

    def run_with_geom(self, depth: np.ndarray, tran_feat: np.ndarray,
                      geom: np.ndarray,
                      return_tensor: bool = False) -> np.ndarray | ov.Tensor:
        """Run BEV pool inference with a pre-computed geometry array.

        Separating geometry computation from inference allows the caller to overlap
        CPU geom work with GPU encoder inference (async pipeline).
        """
        self._req.infer({
            'depth': depth if depth.flags.c_contiguous else np.ascontiguousarray(depth),
            'feat': tran_feat if tran_feat.flags.c_contiguous else np.ascontiguousarray(tran_feat),
            'geom': geom,
        })
        bev_tensor = self._req.get_tensor('bev')
        if return_tensor:
            return bev_tensor
        return bev_tensor.data

    def run(self, depth: np.ndarray, tran_feat: np.ndarray, sample: dict,
            debug_geom: bool = False, return_tensor: bool = False) -> np.ndarray | ov.Tensor:
        geom = self._compute_geom(sample)
        if debug_geom:
            _print_geom_debug(geom, sample)
        return self.run_with_geom(depth, tran_feat, geom, return_tensor=return_tensor)


def _compute_geom_ego2global(
        post_rots: np.ndarray,
        post_trans: np.ndarray,
        sensor2egos: np.ndarray,
        ego2globals: np.ndarray,
        intrinsics: np.ndarray,
) -> np.ndarray:
    """Legacy geometry: camera → sensor2ego → ego2global → key_ego_inv.
    Differs from get_ego_coor; kept here for comparison / debugging only."""
    feat_xs   = np.linspace(0, INPUT_W - 1, FEAT_W, dtype=np.float32)
    feat_ys   = np.linspace(0, INPUT_H - 1, FEAT_H, dtype=np.float32)
    depth_vals = (np.arange(D, dtype=np.float32) * GRID_CONFIG['depth'][2]
                  + GRID_CONFIG['depth'][0])
    d_idx, h_idx, w_idx = np.meshgrid(
        np.arange(D), np.arange(FEAT_H), np.arange(FEAT_W), indexing='ij')
    d_flat  = depth_vals[d_idx].ravel()
    px_flat = feat_xs[w_idx].ravel()
    py_flat = feat_ys[h_idx].ravel()
    n_pts   = D * FEAT_H * FEAT_W

    key_ego_inv = np.linalg.inv(ego2globals[0])
    geom = np.empty((6 * n_pts, 3), dtype=np.float32)
    for cam in range(6):
        post_rot = post_rots[cam]
        post_tr  = post_trans[cam]
        k        = intrinsics[cam]
        pts_aug  = np.stack([px_flat, py_flat, np.ones(n_pts, np.float32)], 0)
        pts      = pts_aug - post_tr[:, None]
        pts_orig = np.linalg.inv(post_rot) @ pts
        cx, cy   = k[0, 2], k[1, 2]
        fx, fy   = k[0, 0], k[1, 1]
        cam_x    = (pts_orig[0] - cx) * d_flat / fx
        cam_y    = (pts_orig[1] - cy) * d_flat / fy
        cam_z    = d_flat
        pts_cam  = np.stack([cam_x, cam_y, cam_z, np.ones_like(cam_x)], 0)
        cam_to_key_ego = key_ego_inv @ ego2globals[cam] @ sensor2egos[cam]
        pts_key_ego    = (cam_to_key_ego @ pts_cam)[:3].T.astype(np.float32, copy=False)
        s = cam * n_pts
        geom[s: s + n_pts, :] = pts_key_ego
    return geom


def _print_geom_debug(geom_s2e: np.ndarray, sample: dict) -> None:
    """Print per-axis statistics comparing sensor2ego vs ego2global geometry."""
    geom_eg = _compute_geom_ego2global(
        sample['post_rots'], sample['post_trans'],
        sample['sensor2egos'], sample['ego2globals'], sample['intrins'],
    )
    diff = geom_s2e - geom_eg
    print('  [geom-debug] sensor2ego vs ego2global geometry diff:')
    for ax, name in enumerate(['x', 'y', 'z']):
        print(f'    {name}: mean_abs={np.abs(diff[:, ax]).mean():.4f}  '
              f'max_abs={np.abs(diff[:, ax]).max():.4f}  '
              f'std={diff[:, ax].std():.4f}')
    print(f'  [geom-debug] sensor2ego x range: [{geom_s2e[:, 0].min():.2f}, '
          f'{geom_s2e[:, 0].max():.2f}]')
    print(f'  [geom-debug] ego2global x range: [{geom_eg[:, 0].min():.2f}, '
          f'{geom_eg[:, 0].max():.2f}]')


def tensor_debug_stats(arr: np.ndarray) -> dict:
    return {
        'nan_count': int(np.isnan(arr).sum()),
        'inf_count': int(np.isinf(arr).sum()),
        'finite_ratio': float(np.isfinite(arr).mean()),
    }


def _build_gpu_argmax_model(core: ov.Core, device: str, input_dtype: ov.Type = ov.Type.f32) -> ov.CompiledModel:
    param = ov_opset13.parameter([1, 200, 200, 16, 18], input_dtype, name='occ_pred')
    k = ov_opset13.constant(np.array(1, dtype=np.int32))
    topk = ov_opset13.topk(param, k, axis=4, mode='max', sort='value')
    indices = topk.output(1)
    axes = ov_opset13.constant(np.array([0, 4], dtype=np.int64))
    squeezed = ov_opset13.squeeze(indices, axes)
    cast = ov_opset13.convert(squeezed, ov.Type.i32)
    result = ov_opset13.result(cast, name='occ_label')
    return core.compile_model(ov.Model([result], [param], 'flashocc_argmax'), device)


class OpenVINOSplitPipeline:
    def __init__(self, model_dir: str, device: str = 'GPU', gpu_precision: str = 'auto',
                 inference_precision: str | None = None,
                 enc_inference_precision: str | None = None,
                 trk_inference_precision: str | None = None,
                 bev_pool_cl=None, bev_pool_so=None,
                 profile_image_encoder: bool = False,
                 report_infer_only_timing: bool = True,
                 force_static_reshape: bool = True,
                 enable_io_optimizations: bool = True,
                 use_remote_tensors: bool = True,
                 enable_nan_debug: bool = False):
        core = ov.Core()
        md = Path(model_dir)
        self._device = device
        self._bev_pool_cl = bev_pool_cl  # BEVPoolOpenCL or None
        self._bev_pool_so = bev_pool_so  # BEVPoolOVSO or None
        self._profile_image_encoder = profile_image_encoder
        self._report_infer_only_timing = report_infer_only_timing
        self._force_static_reshape = force_static_reshape
        self._enable_io_optimizations = enable_io_optimizations
        self._use_remote_tensors = use_remote_tensors and device.upper().startswith('GPU')
        self._enable_nan_debug = enable_nan_debug

        self._cross_device_pipeline = False
        if self._bev_pool_so is not None:
            try:
                self._cross_device_pipeline = str(self._bev_pool_so.device).upper() != device.upper()
            except Exception:
                self._cross_device_pipeline = False

        # Mixed-device path (e.g., NPU encoder/trunk + GPU BEV pool) is sensitive to buffer aliasing.
        # Force stable copy-based IO in this mode to avoid NaN outputs.
        if self._cross_device_pipeline and self._enable_io_optimizations:
            self._enable_io_optimizations = False
            print('  [OpenVINOSplitPipeline] Mixed-device pipeline detected; '
                  'disabling no-copy/tensor-reuse IO path for correctness.')
        self._enc_profile_op_type_ms: dict[str, float] = {}
        self._enc_profile_node_ms: dict[str, float] = {}
        self._enc_profile_calls = 0
        self._encoder_split_batch = 3
        self._encoder_frame_batch = 6
        self._encoder_split_enabled = False

        self._enc_input_arr = None
        self._enc_input_tensor = None
        self._enc_split_req = None
        self._trk_input_arr = None
        self._trk_input_tensor = None
        self._argmax_req = None
        self._argmax_wired = False
        self._bev_pool_tensor_handoff = (
            self._enable_io_optimizations
            and self._bev_pool_so is not None
            and not self._cross_device_pipeline
            and device.upper().startswith('GPU')
        )
        self._bev_pool_tensor_handoff_warned = False

        enc_xml = md / 'flashocc-r50-M0.image_encoder.xml'
        trk_xml = md / 'flashocc-r50-M0.bev_trunk.xml'
        if not enc_xml.exists() or not trk_xml.exists():
            enc_candidates = sorted(md.glob('*.image_encoder.xml')) + sorted(md.glob('image_encoder.xml'))
            trk_candidates = sorted(md.glob('*.bev_trunk.xml')) + sorted(md.glob('bev_trunk.xml'))
            assert enc_candidates, f"No image_encoder xml found in {md}"
            assert trk_candidates, f"No bev_trunk xml found in {md}"
            enc_xml = enc_candidates[0]
            trk_xml = trk_candidates[0]

        # Prefer fused bev_trunk+ArgMax model when available on GPU (eliminates one
        # pybind11 infer() call ≈ 3.5 ms per frame).
        self._fused_argmax = False
        if device.upper().startswith('GPU'):
            fused_candidates = (sorted(md.glob('bev_trunk_argmax_only.xml'))
                                + [md / 'bev_trunk_argmax_only.xml'])
            fused_xml = fused_candidates[0] if fused_candidates[0].exists() else None
            if fused_xml is not None and fused_xml.exists():
                trk_xml = fused_xml
                self._fused_argmax = True
                print(f'  [OpenVINOSplitPipeline] Using fused bev_trunk+ArgMax model: {fused_xml.name}')

        enc_cfg = {}
        trk_cfg = {}
        if device.upper().startswith('GPU') and gpu_precision != 'auto':
            enc_cfg['INFERENCE_PRECISION_HINT'] = _ov_prec(gpu_precision)
            trk_cfg['INFERENCE_PRECISION_HINT'] = _ov_prec(gpu_precision)

        if inference_precision in ('f32', 'f16', 'FP32', 'FP16'):
            enc_cfg['INFERENCE_PRECISION_HINT'] = _ov_prec(inference_precision)
            trk_cfg['INFERENCE_PRECISION_HINT'] = _ov_prec(inference_precision)

        if enc_inference_precision in ('f32', 'f16', 'FP32', 'FP16'):
            enc_cfg['INFERENCE_PRECISION_HINT'] = _ov_prec(enc_inference_precision)
        if trk_inference_precision in ('f32', 'f16', 'FP32', 'FP16'):
            trk_cfg['INFERENCE_PRECISION_HINT'] = _ov_prec(trk_inference_precision)

        if self._profile_image_encoder or self._report_infer_only_timing:
            enc_cfg['PERF_COUNT'] = 'YES'
            trk_cfg['PERF_COUNT'] = 'YES'

        _apply_common_ov_config(enc_cfg, device)
        _apply_common_ov_config(trk_cfg, device)

        # Static reshape reduces dynamic-shape overhead and is required for NPU.
        # Applying on GPU too often improves steady-state latency consistency.
        enc_model = core.read_model(str(enc_xml))
        trk_model = core.read_model(str(trk_xml))
        enc_split_model = None

        if self._force_static_reshape:
            if device.upper() == 'NPU':
                print(f"  [NPU] Reshaping models to static shapes (NPU requires fixed input dimensions)...")
            else:
                print(f"  [{device}] Reshaping models to static shapes (latency optimization)...")

            enc_in_shape = enc_model.inputs[0].partial_shape
            if enc_in_shape.is_dynamic:
                enc_model.reshape({enc_model.inputs[0]: [6, 3, 256, 704]})
                print(f"    image_encoder reshaped to {enc_model.inputs[0].shape}")

            trk_in_shape = trk_model.inputs[0].partial_shape
            if trk_in_shape.is_dynamic:
                trk_model.reshape({trk_model.inputs[0]: [1, 64, 200, 200]})
                print(f"    bev_trunk reshaped to {trk_model.inputs[0].shape}")

            print("  [OpenVINOSplitPipeline] Models fixed to batch_img=6, batch_bev=1. "
                  "Ensure your inputs match these shapes.")

        if device.upper().startswith('GPU'):
            try:
                enc_split_model = core.read_model(str(enc_xml))
                enc_split_model.reshape({enc_split_model.inputs[0]: [self._encoder_split_batch, 3, 256, 704]})
                self._encoder_split_enabled = True
                print('  [OpenVINOSplitPipeline] Encoder split enabled: frame batch 6 -> 2x3 GPU submits')
            except Exception as exc:
                print(f'  [OpenVINOSplitPipeline] Encoder split disabled: {exc}')
                enc_split_model = None

        active_enc_model = enc_split_model if self._encoder_split_enabled else enc_model
        self.enc = core.compile_model(active_enc_model, device, enc_cfg)
        self.trk = core.compile_model(trk_model, device, trk_cfg)
        self.enc_req = self.enc.create_infer_request()
        self.trk_req = self.trk.create_infer_request()

        # Extra encoder request used only for parallel 2xB=3 split submission.
        # This keeps the frame-level double-buffer request pair separate.
        if self._encoder_split_enabled:
            self._enc_split_req = self.enc.create_infer_request()

        # Second encoder InferRequest for double-buffered async pipeline.
        # enc_req_B is used by run_pipelined() to submit frame N+1's encoder
        # while frame N's trunk is still executing on the GPU.
        self._enc_req_B = self.enc.create_infer_request()
        self._pipe_use_A = True       # which enc req is "active" for current frame
        self._enc_input_arr_B = None  # separate input buffer for enc_req_B
        self._enc_input_tensor_B = None

        # Async pipeline state (set/cleared by run_pipelined / flush_pipeline)
        self._pipe_pending = False         # True if trunk_N is running async
        self._pipe_pending_t0_enc = 0.0   # when preprocess of pending frame started
        self._pipe_pending_t0_trk = 0.0   # when trunk of pending frame was submitted
        self._pipe_pending_bev_feat = None
        self._pipe_pending_nan_debug = {}
        self._pipe_pending_geom = None
        self._pipe_argmax_wired_B = False  # argmax wiring for trk output (shared)

        if device.upper().startswith('GPU') and not self._fused_argmax:
            # Detect bev_trunk output dtype and build matching argmax model (avoid f16→f32 cast)
            trk_out_type = self.trk.outputs[0].element_type
            self._argmax_req = _build_gpu_argmax_model(core, device, trk_out_type).create_infer_request()
        elif self._fused_argmax:
            self._argmax_req = None  # ArgMax is fused into bev_trunk model

        # Rebind BEV pool to shared core so all models share the same GPU context.
        # This avoids cross-context synchronisation overhead (~17 ms/frame on iGPU).
        if (self._bev_pool_so is not None
                and not self._cross_device_pipeline
                and device.upper().startswith('GPU')):
            try:
                self._bev_pool_so._rebind_to_shared_core(core)
                print('  [OpenVINOSplitPipeline] BEV pool rebound to shared GPU core (zero cross-context sync)')
            except Exception as exc:
                print(f'  [OpenVINOSplitPipeline] BEV pool shared-core rebind skipped: {exc}')

        if self._enable_io_optimizations:
            print('  [OpenVINOSplitPipeline] IO optimizations: enabled (tensor reuse, no output copies)')
        if self._use_remote_tensors:
            print('  [OpenVINOSplitPipeline] Remote-style tensors: enabled (shared host memory for GPU inputs)')
        if device.upper().startswith('GPU'):
            print(f'  [OpenVINOSplitPipeline] GPU compile config: PERFORMANCE_HINT=LATENCY, CACHE_DIR={_ensure_ov_cache_dir()}')
        if self._bev_pool_tensor_handoff:
            print('  [OpenVINOSplitPipeline] Zero-copy BEV tensor handoff: enabled (.so -> trunk)')

        if bev_pool_so is not None:
            print(f'  [OpenVINOSplitPipeline] Using BEV pool via OpenVINO extension (.so) on {bev_pool_so.device}')
        elif bev_pool_cl is not None:
            print(f'  [OpenVINOSplitPipeline] Using GPU BEV pool: {bev_pool_cl.device_name}')
        else:
            print('  [OpenVINOSplitPipeline] Using NumPy BEV pool (CPU)')

    def run(self, sample: dict, debug_geom: bool = False) -> tuple[np.ndarray, np.ndarray, dict, dict]:
        timings = {}
        nan_debug = {}

        t0 = time.time()
        images = np.ascontiguousarray(sample['images'])
        geom_prefetch_fn = None
        if self._bev_pool_so is not None and not debug_geom:
            geom_prefetch_fn = lambda: self._bev_pool_so.compute_geom(sample)
        tran_feat, depth, enc_infer_only_s, geom_prefetch = self._run_encoder(
            self.enc_req, images, is_B=False, geom_prefetch_fn=geom_prefetch_fn)

        if tran_feat.dtype != np.float32:
            tran_feat = tran_feat.astype(np.float32, copy=False)
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32, copy=False)
        nan_debug['enc_tran_feat'] = tensor_debug_stats(tran_feat) if self._enable_nan_debug else {}
        nan_debug['enc_depth'] = tensor_debug_stats(depth) if self._enable_nan_debug else {}
        timings['image_encoder'] = time.time() - t0
        if self._report_infer_only_timing:
            timings['image_encoder_infer_only'] = enc_infer_only_s

        t0 = time.time()
        bev_tensor = None
        if self._bev_pool_so is not None:
            if self._bev_pool_tensor_handoff:
                if geom_prefetch is not None:
                    bev_tensor = self._bev_pool_so.run_with_geom(
                        depth, tran_feat, geom_prefetch, return_tensor=True)
                else:
                    bev_tensor = self._bev_pool_so.run(depth, tran_feat, sample, debug_geom=debug_geom, return_tensor=True)
                bev_feat = bev_tensor.data
            else:
                if geom_prefetch is not None:
                    bev_feat = self._bev_pool_so.run_with_geom(depth, tran_feat, geom_prefetch)
                else:
                    bev_feat = self._bev_pool_so.run(depth, tran_feat, sample, debug_geom=debug_geom)
            pool_key = 'bev_pool_so'
        elif self._bev_pool_cl is not None:
            bev_feat = self._bev_pool_cl.run(depth, tran_feat, sample)
            pool_key = 'bev_pool_opencl'
        else:
            bev_feat = bev_pool_v2_numpy(
                depth, tran_feat,
                sample['post_rots'], sample['post_trans'],
                sample['sensor2egos'], sample['ego2globals'], sample['intrins']
            )
            pool_key = 'bev_pool_numpy'
        nan_debug['bev_feat'] = tensor_debug_stats(bev_feat) if self._enable_nan_debug else {}
        timings[pool_key] = time.time() - t0

        t0 = time.time()
        if bev_tensor is not None:
            try:
                self.trk_req.set_input_tensor(0, bev_tensor)
            except Exception as exc:
                self._bev_pool_tensor_handoff = False
                if not self._bev_pool_tensor_handoff_warned:
                    print(f'  [OpenVINOSplitPipeline] Zero-copy BEV tensor handoff disabled: {exc}')
                    self._bev_pool_tensor_handoff_warned = True
                bev_tensor = None

        if bev_tensor is None:
            bev_feat_contig = bev_feat if bev_feat.flags.c_contiguous else np.ascontiguousarray(bev_feat)
            if self._enable_io_optimizations:
                if self._trk_input_arr is None or self._trk_input_arr.shape != bev_feat_contig.shape or self._trk_input_arr.dtype != bev_feat_contig.dtype:
                    self._trk_input_arr = np.empty_like(bev_feat_contig)
                    if self._use_remote_tensors:
                        try:
                            self._trk_input_tensor = ov.Tensor(self._trk_input_arr, shared_memory=True)
                        except Exception:
                            self._trk_input_tensor = ov.Tensor(self._trk_input_arr)
                    else:
                        self._trk_input_tensor = ov.Tensor(self._trk_input_arr)
                    self.trk_req.set_input_tensor(0, self._trk_input_tensor)
                np.copyto(self._trk_input_arr, bev_feat_contig)
            else:
                self.trk_req.set_input_tensor(0, ov.Tensor(bev_feat_contig))
        self.trk_req.infer()
        timings['bev_trunk'] = time.time() - t0
        if self._report_infer_only_timing:
            timings['bev_trunk_infer_only'] = self._sum_request_profile_ms(self.trk_req) / 1000.0

        t0 = time.time()
        if self._fused_argmax:
            occ_label = self.trk_req.get_tensor(self.trk.outputs[0]).data[0].astype(np.uint8)
            timings['postprocess'] = time.time() - t0
            t0 = time.time()
            occ_pred = np.empty(0, dtype=np.float32)
            timings['occ_pred_host_copy'] = time.time() - t0
            nan_debug['occ_pred'] = {}
        else:
            occ_out_tensor = self.trk_req.get_tensor('occ_pred')
            if self._argmax_req is not None:
                if not self._argmax_wired:
                    self._argmax_req.set_input_tensor(0, occ_out_tensor)
                    self._argmax_wired = True
                # Submit argmax async and copy occ_pred in parallel (concurrent iGPU reads are safe).
                self._argmax_req.start_async()
                t_hcopy = time.time()
                occ_pred = (occ_out_tensor.data[0].copy()
                            if self._enable_io_optimizations
                            else occ_out_tensor.data.copy()[0])
                timings['occ_pred_host_copy'] = time.time() - t_hcopy
                self._argmax_req.wait()
                occ_label = self._argmax_req.get_tensor('occ_label').data.astype(np.uint8)
            else:
                occ_label = occ_out_tensor.data[0].argmax(-1).astype(np.uint8)
                t_hcopy = time.time()
                occ_pred = (occ_out_tensor.data[0].copy()
                            if self._enable_io_optimizations
                            else occ_out_tensor.data.copy()[0])
                timings['occ_pred_host_copy'] = time.time() - t_hcopy
            timings['postprocess'] = time.time() - t0
            nan_debug['occ_pred'] = tensor_debug_stats(occ_pred) if self._enable_nan_debug else {}
        timings['total'] = sum(
            v for k, v in timings.items()
            if not k.endswith('_infer_only') and not k.endswith('_host_copy')
        )
        
        # Return intermediate tensors for debugging
        tensor_state = {
            'tran_feat': tran_feat,
            'depth': depth,
            'bev_feat': bev_feat,
        }
        return occ_label, occ_pred, timings, nan_debug, tensor_state

    # ------------------------------------------------------------------
    # Async double-buffered pipeline helpers
    # ------------------------------------------------------------------

    def _get_active_enc_req(self):
        """Return the encoder InferRequest to use for the current frame."""
        return self.enc_req if self._pipe_use_A else self._enc_req_B

    def _swap_enc_req(self):
        """Alternate between enc_req A and B for double buffering."""
        self._pipe_use_A = not self._pipe_use_A

    def _setup_enc_input(self, enc_req, images: np.ndarray, is_B: bool) -> None:
        """Wire images into enc_req, reusing pre-allocated shared-memory buffers."""
        if self._enable_io_optimizations:
            if is_B:
                if (self._enc_input_arr_B is None
                        or self._enc_input_arr_B.shape != images.shape
                        or self._enc_input_arr_B.dtype != images.dtype):
                    self._enc_input_arr_B = np.empty_like(images)
                    try:
                        self._enc_input_tensor_B = ov.Tensor(self._enc_input_arr_B, shared_memory=True)
                    except Exception:
                        self._enc_input_tensor_B = ov.Tensor(self._enc_input_arr_B)
                enc_req.set_input_tensor(0, self._enc_input_tensor_B)
                np.copyto(self._enc_input_arr_B, images)
            else:
                if (self._enc_input_arr is None
                        or self._enc_input_arr.shape != images.shape
                        or self._enc_input_arr.dtype != images.dtype):
                    self._enc_input_arr = np.empty_like(images)
                    try:
                        self._enc_input_tensor = ov.Tensor(self._enc_input_arr, shared_memory=True)
                    except Exception:
                        self._enc_input_tensor = ov.Tensor(self._enc_input_arr)
                enc_req.set_input_tensor(0, self._enc_input_tensor)
                np.copyto(self._enc_input_arr, images)
        else:
            enc_req.set_input_tensor(0, ov.Tensor(images))

    def _read_encoder_outputs(self, enc_req) -> tuple[np.ndarray, np.ndarray]:
        if self._enable_io_optimizations:
            return enc_req.get_tensor('tran_feat').data, enc_req.get_tensor('depth').data
        return enc_req.get_tensor('tran_feat').data.copy(), enc_req.get_tensor('depth').data.copy()

    def _run_encoder(self, enc_req, images: np.ndarray, is_B: bool,
                     background_work=None, geom_prefetch_fn=None) -> tuple[np.ndarray, np.ndarray, float, object]:
        infer_only_s = 0.0
        geom_prefetch = None
        chunks = [images]
        if self._encoder_split_enabled:
            if images.shape[0] == self._encoder_frame_batch:
                chunks = [
                    np.ascontiguousarray(images[:self._encoder_split_batch]),
                    np.ascontiguousarray(images[self._encoder_split_batch:]),
                ]
            elif images.shape[0] != self._encoder_split_batch:
                raise ValueError(
                    f'Encoder split expects batch {self._encoder_split_batch} or {self._encoder_frame_batch}, '
                    f'got {images.shape[0]}')

        if len(chunks) == 2 and self._enc_split_req is not None:
            enc_req_0 = enc_req
            enc_req_1 = self._enc_split_req
            self._setup_enc_input(enc_req_0, chunks[0], is_B)
            self._setup_enc_input(enc_req_1, chunks[1], not is_B)
            enc_req_0.start_async()
            enc_req_1.start_async()
            if background_work is not None:
                background_work()
            if geom_prefetch_fn is not None:
                geom_prefetch = geom_prefetch_fn()
            enc_req_0.wait()
            enc_req_1.wait()

            if self._profile_image_encoder:
                self._accumulate_image_encoder_profile(enc_req_0)
                self._accumulate_image_encoder_profile(enc_req_1)
            if self._report_infer_only_timing:
                infer_only_s += self._sum_request_profile_ms(enc_req_0) / 1000.0
                infer_only_s += self._sum_request_profile_ms(enc_req_1) / 1000.0

            tran_0, depth_0 = self._read_encoder_outputs(enc_req_0)
            tran_1, depth_1 = self._read_encoder_outputs(enc_req_1)
            tran_feat = np.concatenate((tran_0, tran_1), axis=0)
            depth = np.concatenate((depth_0, depth_1), axis=0)
            return tran_feat, depth, infer_only_s, geom_prefetch

        tran_feat = None
        depth = None
        offset = 0
        for idx, chunk in enumerate(chunks):
            self._setup_enc_input(enc_req, chunk, is_B)
            use_async = (len(chunks) > 1) or background_work is not None or geom_prefetch_fn is not None
            if use_async:
                enc_req.start_async()
                if idx == 0 and background_work is not None:
                    background_work()
                if idx == 0 and geom_prefetch_fn is not None:
                    geom_prefetch = geom_prefetch_fn()
                enc_req.wait()
            else:
                enc_req.infer()

            if self._profile_image_encoder:
                self._accumulate_image_encoder_profile(enc_req)
            if self._report_infer_only_timing:
                infer_only_s += self._sum_request_profile_ms(enc_req) / 1000.0

            tran_chunk, depth_chunk = self._read_encoder_outputs(enc_req)
            if len(chunks) == 1:
                tran_feat = tran_chunk
                depth = depth_chunk
                continue

            if tran_feat is None:
                tran_feat = np.empty((images.shape[0],) + tuple(tran_chunk.shape[1:]), dtype=tran_chunk.dtype)
                depth = np.empty((images.shape[0],) + tuple(depth_chunk.shape[1:]), dtype=depth_chunk.dtype)
            next_offset = offset + chunk.shape[0]
            tran_feat[offset:next_offset] = tran_chunk
            depth[offset:next_offset] = depth_chunk
            offset = next_offset

        return tran_feat, depth, infer_only_s, geom_prefetch

    def _collect_trunk_result(self) -> tuple:
        """Collect argmax + occ_pred from the completed trk_req.

        Assumes trk_req.wait() has already been called.  Runs argmax ASYNC and
        overlaps the occ_pred host-copy with argmax GPU execution (both read-only
        on the same tensor after trunk finishes — safe on iGPU shared memory).

        Returns (occ_label, occ_pred, postprocess_ms, host_copy_ms, nan_occ_pred).
        """
        t0 = time.time()
        if self._fused_argmax:
            occ_label = self.trk_req.get_tensor(self.trk.outputs[0]).data[0].astype(np.uint8)
            postprocess_ms = time.time() - t0
            occ_pred = np.empty(0, dtype=np.float32)
            host_copy_ms = 0.0
            nan_occ = {}
        elif self._argmax_req is not None:
            occ_out_tensor = self.trk_req.get_tensor('occ_pred')
            if not self._argmax_wired:
                self._argmax_req.set_input_tensor(0, occ_out_tensor)
                self._argmax_wired = True
            # Submit argmax async then copy occ_pred in parallel.
            # Both CPU and GPU read from the same (now-idle) trunk output tensor;
            # on iGPU shared LPDDR5 memory this concurrent read is safe.
            self._argmax_req.start_async()
            t_hcopy = time.time()
            occ_pred = (occ_out_tensor.data[0].copy()
                        if self._enable_io_optimizations
                        else occ_out_tensor.data.copy()[0])
            host_copy_ms = time.time() - t_hcopy
            self._argmax_req.wait()
            occ_label = self._argmax_req.get_tensor('occ_label').data.astype(np.uint8)
            postprocess_ms = time.time() - t0
            nan_occ = tensor_debug_stats(occ_pred) if self._enable_nan_debug else {}
        else:
            occ_out_tensor = self.trk_req.get_tensor('occ_pred')
            occ_label = occ_out_tensor.data[0].argmax(-1).astype(np.uint8)
            postprocess_ms = time.time() - t0
            t_hcopy = time.time()
            occ_pred = (occ_out_tensor.data[0].copy()
                        if self._enable_io_optimizations
                        else occ_out_tensor.data.copy()[0])
            host_copy_ms = time.time() - t_hcopy
            nan_occ = tensor_debug_stats(occ_pred) if self._enable_nan_debug else {}
        return occ_label, occ_pred, postprocess_ms, host_copy_ms, nan_occ

    def run_pipelined(self, sample: dict, debug_geom: bool = False):
        """Pipelined inference with two overlapping optimizations:

        1. Argmax-occ_pred overlap: after trunk_N-1 completes, argmax GPU runs while
           occ_pred is copied to CPU (both read-only → safe on iGPU shared memory).
           Also overlaps preprocessing of sample_N with argmax_N-1.
           Saves ~5-7ms per frame.

        2. Geom-encoder overlap: BEV geometry is computed on CPU in parallel with the
           encoder GPU execution (already done in run() too, saves ~1.2ms per frame).

        NOTE: enc_N is intentionally submitted AFTER argmax_N-1 completes to avoid
        GPU contention between the encoder and argmax streams. On Xe3 (96 EU), running
        two heavy GPU models simultaneously degrades both by ~40-50%.

        Returns None on the first call (pipeline priming).
        On subsequent calls returns (occ_label, occ_pred, timings, nan_debug, tensor_state)
        for the PREVIOUS sample.  Call flush_pipeline() after the last sample.
        """
        prev_result = None
        prev_frame_sync = 0.0
        images = None
        finalize_prev_result = None

        # ── Phase 1: Collect previous trunk + overlap argmax with preprocess ──
        if self._pipe_pending:
            t_prev_sync = time.time()
            self.trk_req.wait()
            t_trk_done = time.time()
            prev_frame_sync = t_trk_done - t_prev_sync

            occ_out_tensor = None
            if not self._fused_argmax and self._argmax_req is not None:
                occ_out_tensor = self.trk_req.get_tensor('occ_pred')
                if not self._argmax_wired:
                    self._argmax_req.set_input_tensor(0, occ_out_tensor)
                    self._argmax_wired = True
                self._argmax_req.start_async()

            images = np.ascontiguousarray(sample['images'])
            if occ_out_tensor is not None:
                occ_pred_prev = (occ_out_tensor.data[0].copy()
                                 if self._enable_io_optimizations
                                 else occ_out_tensor.data.copy()[0])
            else:
                occ_pred_prev = np.empty(0, dtype=np.float32)

            if self._argmax_req is not None and not self._fused_argmax:
                self._argmax_req.wait()
                occ_label_prev = self._argmax_req.get_tensor('occ_label').data.astype(np.uint8)
            elif self._fused_argmax:
                occ_label_prev = self.trk_req.get_tensor(self.trk.outputs[0]).data[0].astype(np.uint8)
                occ_pred_prev = np.empty(0, dtype=np.float32)
            else:
                occ_label_prev = occ_pred_prev[0].argmax(-1).astype(np.uint8)

            t_post_done = time.time()
            self._pipe_pending = False
            prev_timings_base = self._pipe_pending_timings.copy()
            prev_nan_base = self._pipe_pending_nan_debug.copy()
            prev_tensor_state = self._pipe_pending_tensor_state.copy()

            def finalize_prev_result():
                nonlocal prev_result
                prev_timings = prev_timings_base.copy()
                prev_timings['bev_trunk'] = t_trk_done - self._pipe_pending_t0_trk
                prev_timings['postprocess'] = t_post_done - t_trk_done
                prev_timings['occ_pred_host_copy'] = 0.0
                prev_timings['total'] = sum(
                    v for k, v in prev_timings.items()
                    if not k.endswith('_infer_only') and not k.endswith('_host_copy')
                )
                prev_nan = prev_nan_base.copy()
                prev_nan['occ_pred'] = (tensor_debug_stats(occ_pred_prev)
                                        if self._enable_nan_debug else {})
                prev_result = (occ_label_prev, occ_pred_prev, prev_timings, prev_nan, prev_tensor_state)
        else:
            images = np.ascontiguousarray(sample['images'])

        # ── Phase 2: Run encoder + overlap geom/pending-result packaging ──────
        enc_req = self._get_active_enc_req()
        is_B = not self._pipe_use_A
        geom_prefetch_fn = None
        if self._bev_pool_so is not None and not debug_geom:
            geom_prefetch_fn = lambda: self._bev_pool_so.compute_geom(sample)
        t_enc_start = time.time()
        tran_feat, depth, enc_infer_only_s, geom_prefetch = self._run_encoder(
            enc_req,
            images,
            is_B,
            background_work=finalize_prev_result,
            geom_prefetch_fn=geom_prefetch_fn,
        )
        t_enc_done = time.time()

        if tran_feat.dtype != np.float32:
            tran_feat = tran_feat.astype(np.float32, copy=False)
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32, copy=False)

        cur_timings: dict = {'prev_frame_sync': prev_frame_sync}
        cur_nan: dict = {}
        cur_nan['enc_tran_feat'] = tensor_debug_stats(tran_feat) if self._enable_nan_debug else {}
        cur_nan['enc_depth'] = tensor_debug_stats(depth) if self._enable_nan_debug else {}
        cur_timings['image_encoder'] = t_enc_done - t_enc_start
        if self._report_infer_only_timing:
            cur_timings['image_encoder_infer_only'] = enc_infer_only_s

        # ── Phase 3: BEV pool ─────────────────────────────────────────────────
        t0 = time.time()
        bev_tensor = None
        if self._bev_pool_so is not None:
            if self._bev_pool_tensor_handoff:
                bev_tensor = (self._bev_pool_so.run_with_geom(depth, tran_feat, geom_prefetch, return_tensor=True)
                              if geom_prefetch is not None
                              else self._bev_pool_so.run(depth, tran_feat, sample, debug_geom=debug_geom, return_tensor=True))
                bev_feat = bev_tensor.data
            else:
                bev_feat = (self._bev_pool_so.run_with_geom(depth, tran_feat, geom_prefetch)
                            if geom_prefetch is not None
                            else self._bev_pool_so.run(depth, tran_feat, sample, debug_geom=debug_geom))
            pool_key = 'bev_pool_so'
        elif self._bev_pool_cl is not None:
            bev_feat = self._bev_pool_cl.run(depth, tran_feat, sample)
            pool_key = 'bev_pool_opencl'
        else:
            bev_feat = bev_pool_v2_numpy(
                depth, tran_feat,
                sample['post_rots'], sample['post_trans'],
                sample['sensor2egos'], sample['ego2globals'], sample['intrins'],
            )
            pool_key = 'bev_pool_numpy'
        cur_nan['bev_feat'] = tensor_debug_stats(bev_feat) if self._enable_nan_debug else {}
        cur_timings[pool_key] = time.time() - t0

        # ── Phase 4: Submit trunk ASYNC, save state, swap enc buffer ──────────
        t0_trk = time.time()
        if bev_tensor is not None:
            try:
                self.trk_req.set_input_tensor(0, bev_tensor)
            except Exception as exc:
                self._bev_pool_tensor_handoff = False
                if not self._bev_pool_tensor_handoff_warned:
                    print(f'  [OpenVINOSplitPipeline] Zero-copy BEV tensor handoff disabled: {exc}')
                    self._bev_pool_tensor_handoff_warned = True
                bev_tensor = None

        if bev_tensor is None:
            bev_feat_contig = bev_feat if bev_feat.flags.c_contiguous else np.ascontiguousarray(bev_feat)
            if self._enable_io_optimizations:
                if (self._trk_input_arr is None
                        or self._trk_input_arr.shape != bev_feat_contig.shape
                        or self._trk_input_arr.dtype != bev_feat_contig.dtype):
                    self._trk_input_arr = np.empty_like(bev_feat_contig)
                    try:
                        self._trk_input_tensor = ov.Tensor(self._trk_input_arr, shared_memory=True)
                    except Exception:
                        self._trk_input_tensor = ov.Tensor(self._trk_input_arr)
                    self.trk_req.set_input_tensor(0, self._trk_input_tensor)
                np.copyto(self._trk_input_arr, bev_feat_contig)
            else:
                self.trk_req.set_input_tensor(0, ov.Tensor(bev_feat_contig))

        self.trk_req.start_async()

        self._pipe_pending = True
        self._pipe_pending_t0_trk = t0_trk
        self._pipe_pending_timings = cur_timings
        self._pipe_pending_nan_debug = cur_nan
        self._pipe_pending_tensor_state = {'tran_feat': tran_feat, 'depth': depth, 'bev_feat': bev_feat}
        self._swap_enc_req()

        if finalize_prev_result is not None and prev_result is None:
            finalize_prev_result()
        return prev_result

    def flush_pipeline(self):
        """Collect the last pending trunk result after the run_pipelined() loop ends.

        Returns (occ_label, occ_pred, timings, nan_debug, tensor_state) or None.
        """
        if not self._pipe_pending:
            return None
        self.trk_req.wait()
        t_trk_done = time.time()
        occ_label, occ_pred, post_ms, hcopy_ms, nan_occ = self._collect_trunk_result()
        self._pipe_pending = False

        timings = self._pipe_pending_timings.copy()
        timings['bev_trunk'] = t_trk_done - self._pipe_pending_t0_trk
        timings['postprocess'] = post_ms
        timings['occ_pred_host_copy'] = hcopy_ms
        timings['total'] = sum(
            v for k, v in timings.items()
            if not k.endswith('_infer_only') and not k.endswith('_host_copy')
        )
        nan_debug = self._pipe_pending_nan_debug.copy()
        nan_debug['occ_pred'] = nan_occ
        return occ_label, occ_pred, timings, nan_debug, self._pipe_pending_tensor_state

    @staticmethod
    def _perf_time_ms(t) -> float:
        if hasattr(t, 'total_seconds'):
            return float(t.total_seconds() * 1000.0)
        try:
            return float(t)
        except Exception:
            return 0.0

    def _accumulate_image_encoder_profile(self, req=None):
        req = self.enc_req if req is None else req
        self._enc_profile_calls += 1
        for pi in req.profiling_info:
            ms = self._perf_time_ms(pi.real_time)
            if ms <= 0.0:
                continue
            op_type = str(pi.node_type)
            node_name = str(pi.node_name)
            self._enc_profile_op_type_ms[op_type] = self._enc_profile_op_type_ms.get(op_type, 0.0) + ms
            self._enc_profile_node_ms[node_name] = self._enc_profile_node_ms.get(node_name, 0.0) + ms

    def _sum_request_profile_ms(self, req) -> float:
        total_ms = 0.0
        try:
            for pi in req.profiling_info:
                ms = self._perf_time_ms(pi.real_time)
                if ms > 0.0:
                    total_ms += ms
        except Exception:
            return 0.0
        return total_ms

    def get_image_encoder_profile_summary(self, topk: int = 20) -> dict:
        calls = max(self._enc_profile_calls, 1)
        op_type_sorted = sorted(self._enc_profile_op_type_ms.items(), key=lambda kv: kv[1], reverse=True)
        node_sorted = sorted(self._enc_profile_node_ms.items(), key=lambda kv: kv[1], reverse=True)
        total_ms = float(sum(v for _, v in op_type_sorted))
        return {
            'calls': self._enc_profile_calls,
            'total_ms': total_ms,
            'op_type_ms_avg': [(k, v / calls) for k, v in op_type_sorted],
            'node_ms_avg_topk': [(k, v / calls) for k, v in node_sorted[:topk]],
        }



class PyTorchCUDApipeline:
    def __init__(self, config_file: str, checkpoint_file: str, device: str = 'cuda:0'):
        import torch
        from mmcv import Config
        from mmdet3d.models import build_model
        import mmdet3d_plugin  # noqa: F401

        self.torch = torch

        cfg = Config.fromfile(config_file)
        cfg.model.pretrained = None
        cfg.model.pop('train_cfg', None)

        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        ckpt = torch.load(checkpoint_file, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        self.device = torch.device(device)
        self.model = model.to(self.device)

    def run(self, sample: dict) -> tuple[np.ndarray, np.ndarray, dict, dict]:
        torch = self.torch
        timings = {}
        nan_debug = {}

        imgs = torch.from_numpy(sample['images'][None]).to(self.device, non_blocking=True)
        sensor2egos = torch.from_numpy(sample['sensor2egos'][None]).to(self.device, non_blocking=True)
        ego2globals = torch.from_numpy(sample['ego2globals'][None]).to(self.device, non_blocking=True)
        intrins = torch.from_numpy(sample['intrins'][None]).to(self.device, non_blocking=True)
        post_rots = torch.from_numpy(sample['post_rots'][None]).to(self.device, non_blocking=True)
        post_trans = torch.from_numpy(sample['post_trans'][None]).to(self.device, non_blocking=True)
        bda = torch.eye(3, dtype=torch.float32, device=self.device).unsqueeze(0)
        img_inputs = [imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda]
        img_metas = [dict(img_shape=(INPUT_H, INPUT_W, 3))]

        torch.cuda.synchronize(self.device)
        t0 = time.time()
        with torch.no_grad():
            img_feats, _, _ = self.model.extract_feat(points=None, img_inputs=img_inputs, img_metas=img_metas)
        torch.cuda.synchronize(self.device)
        timings['extract_feat'] = time.time() - t0

        x = img_feats[0] if isinstance(img_feats, (list, tuple)) else img_feats
        t0 = time.time()
        with torch.no_grad():
            occ_pred_t = self.model.occ_head(x)
        torch.cuda.synchronize(self.device)
        timings['occ_head'] = time.time() - t0

        occ_pred = occ_pred_t.detach().float().cpu().numpy()[0]
        nan_debug['occ_pred'] = tensor_debug_stats(occ_pred)

        t0 = time.time()
        occ_label = occ_pred.argmax(-1).astype(np.uint8)
        timings['postprocess'] = time.time() - t0
        timings['total'] = sum(timings.values())
        return occ_label, occ_pred, timings, nan_debug


def occ_stats(occ_label: np.ndarray, occ_pred_raw: np.ndarray | None) -> dict:
    total_voxels = occ_label.size
    occupied = int((occ_label != FREE_CLASS).sum())
    free = int((occ_label == FREE_CLASS).sum())

    # occ_pred_raw may be None or empty when using a fused bev_trunk+ArgMax model
    # (no raw logits are returned in that case)
    has_logits = occ_pred_raw is not None and occ_pred_raw.size > 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        voxel_max = np.nanmax(occ_pred_raw, axis=-1) if has_logits else np.array([float('nan')])
        max_logit = float(np.nanmax(occ_pred_raw)) if has_logits else float('nan')
        mean_max_logit = float(np.nanmean(voxel_max)) if has_logits else float('nan')

    class_counts = {OCC_CLASSES[c]: int((occ_label == c).sum()) for c in range(NUM_CLS)}
    return {
        'total_voxels': total_voxels,
        'occupied_voxels': occupied,
        'free_voxels': free,
        'occ_fraction': round(occupied / total_voxels, 4),
        'max_logit': round(max_logit, 4) if not np.isnan(max_logit) else float('nan'),
        'mean_max_logit': round(mean_max_logit, 4) if not np.isnan(mean_max_logit) else float('nan'),
        'pred_nan_count': int(np.isnan(occ_pred_raw).sum()) if has_logits else 0,
        'pred_inf_count': int(np.isinf(occ_pred_raw).sum()) if has_logits else 0,
        'class_counts': class_counts,
    }


def compare_pt_ov_labels(
    pt_label: np.ndarray,
    ov_label: np.ndarray,
    eval_mask: np.ndarray | None = None,
) -> dict:
    """Compare two occupancy label grids.  When *eval_mask* is provided
    (boolean array, same shape) only the True voxels contribute to all
    metrics — this is required for correct Occ3D GT evaluation."""
    if pt_label.shape != ov_label.shape:
        raise ValueError(f"Label shape mismatch: {pt_label.shape} vs {ov_label.shape}")

    pt_flat = pt_label.reshape(-1).astype(np.int64)
    ov_flat = ov_label.reshape(-1).astype(np.int64)

    if eval_mask is not None:
        mask_flat = eval_mask.reshape(-1)
        pt_flat = pt_flat[mask_flat]
        ov_flat = ov_flat[mask_flat]

    voxel_agreement = float((pt_flat == ov_flat).mean())

    cm = np.bincount(
        pt_flat * NUM_CLS + ov_flat,
        minlength=NUM_CLS * NUM_CLS,
    ).reshape(NUM_CLS, NUM_CLS)

    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    denom = tp + fp + fn
    valid = denom > 0

    per_class_iou = np.divide(tp, denom, out=np.full_like(tp, np.nan), where=valid)

    valid_no_free = valid.copy()
    valid_no_free[FREE_CLASS] = False
    miou_no_free = float(np.nanmean(per_class_iou[valid_no_free])) if np.any(valid_no_free) else float('nan')

    support = cm.sum(axis=1).astype(np.float64)
    weights = np.where(valid_no_free, support, 0.0)
    total_w = weights.sum()
    miou_no_free_weighted = (
        float(np.nansum(per_class_iou * weights) / total_w) if total_w > 0 else float('nan')
    )

    pt_occ = pt_flat != FREE_CLASS
    ov_occ = ov_flat != FREE_CLASS
    occ_inter = float(np.logical_and(pt_occ, ov_occ).sum())
    occ_union = float(np.logical_or(pt_occ, ov_occ).sum())
    occupied_iou = occ_inter / occ_union if occ_union > 0 else float('nan')

    return {
        'voxel_agreement': voxel_agreement,
        'occupied_iou': occupied_iou,
        'miou_no_free': miou_no_free,
        'miou_no_free_weighted': miou_no_free_weighted,
        'per_class_iou': per_class_iou,
        'valid_mask': valid,
        'confusion_matrix': cm,
    }


def aggregate_stats(per_frame: list[dict]) -> dict:
    def _safe_nanmean(values: list[float]) -> float:
        arr = np.asarray(values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        return float(finite.mean()) if finite.size else float('nan')

    occ_fracs = [f['occ_fraction'] for f in per_frame]
    max_lgts = [f['max_logit'] for f in per_frame]
    mmx_lgts = [f['mean_max_logit'] for f in per_frame]
    combined_cls = {c: sum(f['class_counts'][c] for f in per_frame) for c in OCC_CLASSES}
    return {
        'n_frames': len(per_frame),
        'mean_occ_frac': float(np.mean(occ_fracs)),
        'std_occ_frac': float(np.std(occ_fracs)),
        'mean_max_logit': _safe_nanmean(max_lgts),
        'mean_mean_max_logit': _safe_nanmean(mmx_lgts),
        'total_pred_nan': int(sum(f.get('pred_nan_count', 0) for f in per_frame)),
        'total_pred_inf': int(sum(f.get('pred_inf_count', 0) for f in per_frame)),
        'class_totals': combined_cls,
    }


def _split_cold_warm_timings(timings_list: list[dict], warmup_frames: int) -> tuple[list[dict], list[dict]]:
    warmup_frames = max(0, min(int(warmup_frames), len(timings_list)))
    return timings_list[:warmup_frames], timings_list[warmup_frames:]


def _print_latency_block(timings_list: list[dict], label: str, suffix: str = ''):
    if not timings_list:
        print(f"\n  [{label}] {suffix.strip() or 'No samples available'}")
        return

    infer_only_keys = {'image_encoder_infer_only', 'bev_trunk_infer_only'}
    stages = [k for k in timings_list[0].keys() if k not in (infer_only_keys | {'total'})] + ['total']
    print(f"\n  [{label}] Module-wise latency & E2E throughput {suffix}")
    print(f"  {'Stage':<22}  {'Mean (ms)':>10}  {'Std (ms)':>10}  {'Share':>8}")
    print("  " + "-" * 56)
    total_means = np.mean([t['total'] * 1000 for t in timings_list])
    for s in stages:
        vals = np.array([t[s] * 1000 for t in timings_list])
        share = f"{vals.mean() / total_means * 100:.1f}%" if s != 'total' else ''
        print(f"  {s:<22}  {vals.mean():>10.1f}  {vals.std():>10.1f}  {share:>8}")
    fps = 1000.0 / total_means
    print(f"  {'E2E throughput':<22}  {fps:>10.2f} fps")

    # Infer-only (OV PERF_COUNT) summary for apples-to-apples comparison with benchmark_app.
    available = [k for k in ('image_encoder_infer_only', 'bev_trunk_infer_only') if k in timings_list[0]]
    if available:
        print(f"\n  [{label}] OV infer-only timing (PERF_COUNT) {suffix}")
        print(f"  {'Model':<22}  {'Mean (ms)':>10}  {'Std (ms)':>10}")
        print("  " + "-" * 46)
        pretty = {
            'image_encoder_infer_only': 'image_encoder',
            'bev_trunk_infer_only': 'bev_trunk',
        }
        for k in available:
            vals = np.array([t[k] * 1000 for t in timings_list if k in t])
            if vals.size == 0:
                continue
            print(f"  {pretty.get(k, k):<22}  {vals.mean():>10.1f}  {vals.std():>10.1f}")


def print_latency_throughput(timings_list: list[dict], label: str, warmup_frames: int = 0):
    cold_timings, warm_timings = _split_cold_warm_timings(timings_list, warmup_frames)
    if warmup_frames > 0:
        print(f"\n  [{label}] Warmup exclusion      : first {len(cold_timings)} frame(s)")
        _print_latency_block(cold_timings, label, suffix='(cold start)')
        _print_latency_block(warm_timings, label, suffix='(steady state, warm frames only)')
    else:
        _print_latency_block(timings_list, label)


def print_ov_image_encoder_profile(profile: dict):
    if not profile or profile.get('calls', 0) <= 0:
        return

    print("\n  [OpenVINO split] image_encoder internal profile (from OV PERF_COUNT)")
    print(f"  Profiled runs          : {profile['calls']}")
    print(f"  Mean profiled sum      : {profile['total_ms'] / profile['calls']:.1f} ms")
    print(f"  {'Op type':<22}  {'Mean (ms)':>10}  {'Share':>8}")
    print("  " + "-" * 46)

    op_items = profile['op_type_ms_avg']
    op_total = float(sum(v for _, v in op_items))
    for op, ms in op_items[:12]:
        share = (100.0 * ms / op_total) if op_total > 0 else 0.0
        print(f"  {op:<22}  {ms:>10.2f}  {share:>7.1f}%")

    print(f"\n  {'Top kernel/node':<34}  {'Mean (ms)':>10}")
    print("  " + "-" * 50)
    for node, ms in profile['node_ms_avg_topk']:
        print(f"  {node[:34]:<34}  {ms:>10.2f}")


def print_accuracy(agg: dict, label: str):
    print(f"\n  [{label}] E2E Accuracy (occupancy output)")
    print(f"  Frames evaluated        : {agg['n_frames']}")
    print(f"  Mean occupied fraction  : {agg['mean_occ_frac']:.4f}  (std={agg['std_occ_frac']:.4f})")
    print(f"  Mean max logit          : {agg['mean_max_logit']:.4f}")
    print(f"  Mean avg-voxel logit    : {agg['mean_mean_max_logit']:.4f}")
    print(f"  Total NaN logits        : {agg['total_pred_nan']}")
    print(f"  Total Inf logits        : {agg['total_pred_inf']}")


def print_backend_comparison(name_a: str, agg_a: dict, name_b: str, agg_b: dict):
    print("\n" + "=" * 72)
    print(f"  BACKEND COMPARISON: {name_a} vs {name_b}")
    print("=" * 72)
    print(f"  {'Metric':<32}  {name_a:>14}  {name_b:>14}")
    print("  " + "-" * 64)
    rows = [
        ('Frames run', agg_a['n_frames'], agg_b['n_frames']),
        ('Mean occupied fraction', agg_a['mean_occ_frac'], agg_b['mean_occ_frac']),
        ('StdDev occupied frac', agg_a['std_occ_frac'], agg_b['std_occ_frac']),
        ('Mean max logit', agg_a['mean_max_logit'], agg_b['mean_max_logit']),
        ('Mean avg-voxel logit', agg_a['mean_mean_max_logit'], agg_b['mean_mean_max_logit']),
        ('Total NaN logits', agg_a['total_pred_nan'], agg_b['total_pred_nan']),
        ('Total Inf logits', agg_a['total_pred_inf'], agg_b['total_pred_inf']),
    ]
    for label, va, vb in rows:
        if isinstance(va, float):
            print(f"  {label:<32}  {va:>14.4f}  {vb:>14.4f}")
        else:
            print(f"  {label:<32}  {str(va):>14}  {str(vb):>14}")


def aggregate_consistency(consistency_list: list[dict]) -> dict:
    if not consistency_list:
        return {}

    voxel_agreement = float(np.mean([c['voxel_agreement'] for c in consistency_list]))
    occupied_iou = float(np.mean([c['occupied_iou'] for c in consistency_list]))
    miou_no_free = float(np.nanmean([c['miou_no_free'] for c in consistency_list]))
    miou_no_free_weighted_per_frame = float(np.nanmean([c['miou_no_free_weighted'] for c in consistency_list]))

    confusion_matrix = np.sum(
        np.stack([c['confusion_matrix'] for c in consistency_list], axis=0),
        axis=0,
    ).astype(np.int64)

    tp = np.diag(confusion_matrix).astype(np.float64)
    fp = confusion_matrix.sum(axis=0).astype(np.float64) - tp
    fn = confusion_matrix.sum(axis=1).astype(np.float64) - tp
    support = confusion_matrix.sum(axis=1).astype(np.int64)

    iou_denom = tp + fp + fn
    prec_denom = tp + fp
    rec_denom = tp + fn

    per_class_iou = np.divide(tp, iou_denom, out=np.full_like(tp, np.nan), where=iou_denom > 0)

    valid_no_free_agg = (iou_denom > 0).copy()
    valid_no_free_agg[FREE_CLASS] = False
    agg_weights = np.where(valid_no_free_agg, support.astype(np.float64), 0.0)
    agg_total_w = agg_weights.sum()
    miou_no_free_weighted_agg = (
        float(np.nansum(per_class_iou * agg_weights) / agg_total_w)
        if agg_total_w > 0 else float('nan')
    )

    per_class_precision = np.divide(tp, prec_denom, out=np.full_like(tp, np.nan), where=prec_denom > 0)
    per_class_recall = np.divide(tp, rec_denom, out=np.full_like(tp, np.nan), where=rec_denom > 0)
    f1_denom = per_class_precision + per_class_recall
    per_class_f1 = np.divide(
        2 * per_class_precision * per_class_recall,
        f1_denom,
        out=np.full_like(tp, np.nan),
        where=np.isfinite(f1_denom) & (f1_denom > 0),
    )

    return {
        'n_frames': len(consistency_list),
        'voxel_agreement': voxel_agreement,
        'occupied_iou': occupied_iou,
        'miou_no_free': miou_no_free,
        'miou_no_free_weighted_per_frame': miou_no_free_weighted_per_frame,
        'miou_no_free_weighted_agg': miou_no_free_weighted_agg,
        'confusion_matrix': confusion_matrix,
        'per_class_iou': per_class_iou,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'per_class_support': support,
    }


def print_consistency(consistency_agg: dict, title: str = 'PT vs OV CONSISTENCY', matrix_label: str = 'PT vs OV confusion matrix'):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)
    print(f"  Frames compared         : {consistency_agg['n_frames']}")
    print(f"  Exact voxel agreement   : {consistency_agg['voxel_agreement'] * 100:.2f}%")
    print(f"  Occupied-only IoU       : {consistency_agg['occupied_iou']:.4f}")
    print(f"  mIoU macro (excl. free) : {consistency_agg['miou_no_free']:.4f}  (unweighted, each class equal)")
    print(f"  mIoU weighted (excl free): {consistency_agg['miou_no_free_weighted_agg']:.4f}  (support-weighted, per aggregated CM)")
    print(f"  mIoU weighted (per-frame): {consistency_agg['miou_no_free_weighted_per_frame']:.4f}  (support-weighted, avg of per-frame)")

    print(f"\n  Per-class metrics ({matrix_label}):")
    print(f"  {'Class':<25}  {'IoU':>8}  {'Prec':>8}  {'Recall':>8}  {'F1':>8}  {'Support':>8}")
    print("  " + "-" * 78)
    for idx, cls_name in enumerate(OCC_CLASSES):
        cls_iou = consistency_agg['per_class_iou'][idx]
        cls_prec = consistency_agg['per_class_precision'][idx]
        cls_rec = consistency_agg['per_class_recall'][idx]
        cls_f1 = consistency_agg['per_class_f1'][idx]
        cls_sup = int(consistency_agg['per_class_support'][idx])

        iou_str = "n/a" if np.isnan(cls_iou) else f"{cls_iou:.4f}"
        prec_str = "n/a" if np.isnan(cls_prec) else f"{cls_prec:.4f}"
        rec_str = "n/a" if np.isnan(cls_rec) else f"{cls_rec:.4f}"
        f1_str = "n/a" if np.isnan(cls_f1) else f"{cls_f1:.4f}"
        marker = '  ← free' if cls_name == 'free' else ''
        print(f"  {cls_name:<25}  {iou_str:>8}  {prec_str:>8}  {rec_str:>8}  {f1_str:>8}  {cls_sup:>8}{marker}")


def save_consistency_csv(consistency_agg: dict, output_dir: Path, prefix: str = 'pt_ov', matrix_header: str = 'pt\\ov'):
    output_dir.mkdir(parents=True, exist_ok=True)

    per_class_csv = output_dir / f'{prefix}_per_class_metrics.csv'
    with per_class_csv.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class_id', 'class_name', 'iou', 'precision', 'recall', 'f1', 'support'])
        for idx, cls_name in enumerate(OCC_CLASSES):
            writer.writerow([
                idx,
                cls_name,
                consistency_agg['per_class_iou'][idx],
                consistency_agg['per_class_precision'][idx],
                consistency_agg['per_class_recall'][idx],
                consistency_agg['per_class_f1'][idx],
                int(consistency_agg['per_class_support'][idx]),
            ])

    cm_csv = output_dir / f'{prefix}_confusion_matrix.csv'
    with cm_csv.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([matrix_header] + OCC_CLASSES)
        cm = consistency_agg['confusion_matrix']
        for i, cls_name in enumerate(OCC_CLASSES):
            writer.writerow([cls_name] + cm[i].tolist())

    summary_csv = output_dir / f'{prefix}_consistency_summary.csv'
    with summary_csv.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n_frames', 'voxel_agreement', 'occupied_iou', 'miou_no_free',
                         'miou_no_free_weighted_agg', 'miou_no_free_weighted_per_frame'])
        writer.writerow([
            consistency_agg['n_frames'],
            consistency_agg['voxel_agreement'],
            consistency_agg['occupied_iou'],
            consistency_agg['miou_no_free'],
            consistency_agg['miou_no_free_weighted_agg'],
            consistency_agg['miou_no_free_weighted_per_frame'],
        ])

    print(f"\n  Saved {prefix} consistency CSVs to: {output_dir}")


def _resolve_label_dir(path_str: str | None, backend_name: str) -> Path | None:
    if not path_str:
        return None
    base = Path(path_str)
    nested = base / backend_name
    if nested.exists() and nested.is_dir():
        return nested
    return base


def save_occ_label(occ_label: np.ndarray, label_dir: Path, sample_idx: int):
    label_dir.mkdir(parents=True, exist_ok=True)
    out_file = label_dir / f"sample_{sample_idx:05d}.npy"
    np.save(out_file, occ_label.astype(np.uint8), allow_pickle=False)


def load_occ_label(label_dir: Path, sample_idx: int) -> np.ndarray:
    in_file = label_dir / f"sample_{sample_idx:05d}.npy"
    if not in_file.exists():
        raise FileNotFoundError(f"Missing saved label file: {in_file}")
    arr = np.load(in_file)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected label shape in {in_file}: {arr.shape}")
    return arr.astype(np.uint8, copy=False)


def _collect_saved_label_indices(label_dir: Path) -> list[int]:
    indices = []
    for p in sorted(label_dir.glob('sample_*.npy')):
        suffix = p.stem.split('_')[-1]
        if suffix.isdigit():
            indices.append(int(suffix))
    return sorted(indices)


def compare_saved_labels(pt_label_dir: Path, ov_label_dir: Path, num_samples: int) -> tuple[dict, int]:
    pt_indices = set(_collect_saved_label_indices(pt_label_dir))
    ov_indices = set(_collect_saved_label_indices(ov_label_dir))
    common = sorted(pt_indices & ov_indices)
    if not common:
        raise RuntimeError(
            f"No overlapping saved label indices between {pt_label_dir} and {ov_label_dir}."
        )

    if num_samples > 0:
        common = common[:num_samples]

    print("\n" + "─" * 72)
    print("  Comparing pre-saved labels")
    print("─" * 72)
    print(f"  PT labels dir      : {pt_label_dir}")
    print(f"  OV labels dir      : {ov_label_dir}")
    print(f"  Matched samples    : {len(common)}")

    consistency_stats = []
    for i in common:
        pt_occ_label = load_occ_label(pt_label_dir, i)
        ov_occ_label = load_occ_label(ov_label_dir, i)
        consistency = compare_pt_ov_labels(pt_occ_label, ov_occ_label)
        consistency_stats.append(consistency)
        print(
            f"  CMP Sample {i:3d}: agree={consistency['voxel_agreement'] * 100:.2f}% "
            f"occ_iou={consistency['occupied_iou']:.4f} "
            f"miou_no_free={consistency['miou_no_free']:.4f}"
        )

    return aggregate_consistency(consistency_stats), len(common)


def main():
    parser = argparse.ArgumentParser(
        description='Compare FlashOCC PyTorch CUDA vs OpenVINO split on identical nuScenes samples'
    )
    parser.add_argument('--config', default='projects/configs/flashocc/flashocc-r50-M0.py')
    parser.add_argument('--checkpoint', default='checkpoints/flashocc-r50.pth')
    parser.add_argument('--model-dir',
                        help='OpenVINO split model dir (contains *.image_encoder.xml and *.bev_trunk.xml)')
    parser.add_argument('--data-pkl')
    parser.add_argument('--data-root')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--run-duration', type=float, default=0.0,
                        help='Run inference for at least this many seconds, re-cycling the sample '
                             'batch as needed. 0 (default) = run exactly --num-samples frames once.')
    parser.add_argument('--run', choices=['both', 'pytorch', 'openvino', 'compare', 'pytorch_twice'], default='both')
    parser.add_argument('--pt-device', default='cuda:0')
    parser.add_argument('--ov-device', default='GPU')
    parser.add_argument('--ov-bevpool-device', default=None,
                        help='Device for BEV pool .so backend only (e.g., GPU/CPU/NPU). '
                             'If not set, defaults to --ov-device.')
    parser.add_argument('--ov-gpu-precision', choices=['auto', 'f16', 'f32'], default='f32')
    parser.add_argument('--ov-inference-precision', choices=['f32', 'f16'], default=None,
                        help='Force OV INFERENCE_PRECISION_HINT (f32/f16). Overrides --ov-gpu-precision hint. '
                             'Use f32 to disable implicit FP16 on GPU.')
    parser.add_argument('--ov-enc-inference-precision', choices=['f32', 'f16'], default=None,
                        help='Per-model OV precision hint for image_encoder only.')
    parser.add_argument('--ov-trk-inference-precision', choices=['f32', 'f16'], default=None,
                        help='Per-model OV precision hint for bev_trunk only.')
    parser.add_argument('--ov-disable-static-reshape', action='store_true',
                        help='Disable static reshape optimization for OV models (enabled by default).')
    parser.add_argument('--ov-disable-io-optimizations', action='store_true',
                        help='Disable OV tensor reuse and no-copy output path (enabled by default).')
    parser.add_argument('--ov-disable-remote-tensors', action='store_true',
                        help='Disable shared-memory remote-style tensors for GPU inputs (enabled by default).')
    parser.add_argument('--ov-disable-infer-only-timing', action='store_true',
                        help='Disable OV PERF_COUNT infer-only timing summary for encoder/trunk.')
    parser.add_argument('--ov-enable-nan-debug', action='store_true',
                        help='Enable per-frame NaN/Inf tensor statistics (adds ~3ms overhead; disabled by default).')
    parser.add_argument('--ov-async-pipeline', action='store_true', default=True,
                        help='Enable inter-frame async pipeline: overlaps frame N+1 encoder submission '
                             'with frame N trunk execution (default: enabled).')
    parser.add_argument('--no-ov-async-pipeline', dest='ov_async_pipeline', action='store_false',
                        help='Disable inter-frame async pipeline (use synchronous single-frame inference).')
    parser.add_argument('--prefetch-data', action='store_true', default=True,
                        help='Prefetch next sample in background during GPU inference to hide I/O latency (default: True)')
    parser.add_argument('--no-prefetch-data', dest='prefetch_data', action='store_false',
                        help='Disable async data prefetching (use sequential load+infer)')
    parser.add_argument('--prefetch-backend', choices=['thread', 'process'], default='thread',
                        help='Backend used for sample prefetching (default: thread; process is experimental).')
    parser.add_argument('--prefetch-depth', type=int, default=2,
                        help='How many future samples to keep queued for async data prefetch (default: 2).')
    parser.add_argument('--cache-sample-inputs', action='store_true',
                        help='Cache preprocessed per-sample inputs on disk to remove repeated cv2.imread + resize cost.')
    parser.add_argument('--sample-cache-dir', default=str(_DEFAULT_SAMPLE_CACHE_DIR),
                        help='Directory used by --cache-sample-inputs (default: debug_output/sample_input_cache).')
    parser.add_argument('--precache-sample-inputs', action='store_true',
                        help='Populate the sample input cache for the selected samples before running inference.')
    parser.add_argument('--latency-warmup-frames', type=int, default=5,
                        help='Exclude the first N frames from steady-state latency/throughput reporting. '
                             'Cold-start latency is still reported separately.')
    parser.add_argument('--ov-profile-image-encoder', action='store_true',
                        help='Enable OV PERF_COUNT for image_encoder and print internal op/node latency breakdown.')
    parser.add_argument('--ov-profile-topk', type=int, default=20,
                        help='Top-K image_encoder OV nodes to print when --ov-profile-image-encoder is enabled.')
    parser.add_argument('--ov-bevpool-inference-precision', choices=['f32', 'f16'], default=None,
                        help='Per-model OV precision hint for BEV pool .so backend only.')
    parser.add_argument('--ov-backend', choices=['split'], default='split',
                        help='OpenVINO backend: split (encoder+bev_pool+trunk)')
    parser.add_argument('--save-consistency-csv', action='store_true',
                        help='Save PT-vs-OV consistency outputs (summary, per-class metrics, confusion matrix) as CSV')
    parser.add_argument('--consistency-output-dir', default='benchmark/pt_ov_consistency',
                        help='Output directory for consistency CSV files')
    parser.add_argument('--use-opencl-bev-pool', action='store_true',
                        help='Replace bev_pool_numpy with GPU OpenCL kernel (requires pyopencl)')
    parser.add_argument('--opencl-prefer-gpu', action='store_true', default=True,
                        help='Prefer GPU device for OpenCL BEV pool (default: True)')
    parser.add_argument('--use-so-bev-pool', action='store_true',
                        help='Force use of OV extension (.so) BEV pool (auto-used if .so exists)')
    parser.add_argument('--no-so-bev-pool', action='store_true',
                        help='Disable OV extension (.so) BEV pool; fall back to NumPy')
    parser.add_argument('--ov-extension-so',
                        default=str((_REPO_ROOT / 'openvino_extensions' / 'bev_pool' / 'build' / 'libopenvino_bevpool_extension.so').resolve()),
                        help='Path to BEV pool extension .so used by --use-so-bev-pool')
    parser.add_argument('--ov-gpu-config-xml',
                        default=str((_REPO_ROOT / 'openvino_extensions' / 'gpu_custom_layers.xml').resolve()),
                        help='GPU custom layers xml for BEV pool extension backend')
    parser.add_argument('--debug-geom', action='store_true',
                        help='Print geometry debug stats (sensor2ego vs ego2global diff) for just the first sample')
    parser.add_argument('--dump-tensors', action='store_true',
                        help='Save intermediate OV tensors (encoder output, BEV feat) for the first 3 samples')
    parser.add_argument('--tensor-dump-dir', default='tensor_dumps',
                        help='Directory to save dumped tensors')
    parser.add_argument('--save-label-dir', default=None,
                        help='Save predicted labels under this directory as <dir>/<backend>/sample_XXXXX.npy')
    parser.add_argument('--compare-pt-label-dir', default=None,
                        help='Directory of pre-saved PyTorch labels for consistency comparison')
    parser.add_argument('--compare-ov-label-dir', default=None,
                        help='Directory of pre-saved OpenVINO labels for consistency comparison')
    parser.add_argument('--compare-with-gt', action='store_true',
                        help='Also compare PT/OV predictions with original GT occupancy labels')
    parser.add_argument('--gt-dir', default=None,
                        help='Root directory of GT occupancy labels (gts/). '
                             'Structure: <gt-dir>/<scene>/<token>/labels.npz. '
                             'Builds a token->path map automatically when occ_path is missing from pkl.')
    parser.add_argument('--gt-key', default=None,
                        help='Optional key in info dict that points to GT occupancy label path')
    parser.add_argument('--no-gt-mask', action='store_true',
                        help='Disable mask_lidar filtering for GT comparison (evaluates all voxels). '
                             'By default only lidar-observed voxels (mask_lidar=True) are scored, '
                             'matching the official Occ3D evaluation protocol.')
    args = parser.parse_args()

    if args.run in ('both', 'openvino') and args.ov_backend == 'split' and not args.model_dir:
        parser.error('--model-dir is required when running OpenVINO split inference')
    if args.run in ('both', 'pytorch', 'openvino', 'pytorch_twice') and (not args.data_pkl or not args.data_root):
        parser.error('--data-pkl and --data-root are required when running model inference')
    if args.use_opencl_bev_pool and args.use_so_bev_pool:
        parser.error('Choose only one BEV pool acceleration backend: --use-opencl-bev-pool or --use-so-bev-pool')
    if args.run == 'compare' and args.compare_with_gt:
        parser.error('--compare-with-gt is supported only when running model inference (not --run compare).')

    # ── Build GT token→path map from --gt-dir if provided ────────────────────
    gt_token_map: dict[str, str] = {}
    if args.compare_with_gt and args.gt_dir:
        import glob as _glob
        _gt_dir = Path(args.gt_dir)
        for npz_path in _glob.glob(str(_gt_dir / '**' / 'labels.npz'), recursive=True):
            token = Path(npz_path).parent.name  # .../scene-XXXX/<TOKEN>/labels.npz
            gt_token_map[token] = npz_path
        print(f"[GT] Loaded {len(gt_token_map)} GT token→path mappings from {_gt_dir}")

    pt_saved_label_dir = _resolve_label_dir(args.compare_pt_label_dir, 'pytorch')
    ov_saved_label_dir = _resolve_label_dir(args.compare_ov_label_dir, 'openvino')

    if args.run == 'compare':
        if pt_saved_label_dir is None or ov_saved_label_dir is None:
            parser.error('--run compare requires --compare-pt-label-dir and --compare-ov-label-dir')

        print("=" * 72)
        print("  FlashOCC PT-vs-OV Consistency from Saved Labels")
        print("=" * 72)
        consistency_agg, _ = compare_saved_labels(pt_saved_label_dir, ov_saved_label_dir, args.num_samples)
        print_consistency(consistency_agg)
        if args.save_consistency_csv:
            save_consistency_csv(consistency_agg, Path(args.consistency_output_dir))
        return

    with open(args.data_pkl, 'rb') as f:
        d = pickle.load(f)
    infos = d['infos'] if isinstance(d, dict) else d

    # ── Patch GT occ_path into infos from --gt-dir map (if built earlier) ────
    if gt_token_map:
        for _info in infos:
            tok = _info.get('token', '')
            if tok in gt_token_map and 'occ_path' not in _info:
                _info['occ_path'] = gt_token_map[tok]

    n = min(args.num_samples, len(infos))
    data_root = Path(args.data_root)
    sample_cache_dir = Path(args.sample_cache_dir).resolve() if getattr(args, 'cache_sample_inputs', False) else None

    if args.precache_sample_inputs and sample_cache_dir is None:
        parser.error('--precache-sample-inputs requires --cache-sample-inputs')
    if args.precache_sample_inputs:
        t_precache = time.perf_counter()
        for info in infos[:n]:
            load_sample_inputs_cached(info, data_root, sample_cache_dir)
        precache_ms = (time.perf_counter() - t_precache) * 1000.0
        print(f"  [SampleCache] Pre-cached {n} samples in {precache_ms:.1f} ms -> {sample_cache_dir}")

    print("=" * 72)
    print("  FlashOCC PyTorch vs OpenVINO Comparison")
    print("=" * 72)
    print(f"  Running samples : {n}/{len(infos)}")
    print(f"  Run mode        : {args.run}")
    if args.run in ('both', 'openvino'):
        bevpool_device = args.ov_bevpool_device or args.ov_device
        print(f"  OV models device : {args.ov_device}")
        print(f"  BEV pool device  : {bevpool_device}")
        print(f"  OV backend      : {args.ov_backend}")

    pt_pipe = None
    ov_pipe = None
    if args.run in ('both', 'pytorch', 'pytorch_twice'):
        pt_pipe = PyTorchCUDApipeline(args.config, args.checkpoint, args.pt_device)
    if args.run in ('both', 'openvino'):
        bev_pool_cl = None
        bev_pool_so = None
        # Auto-detect .so extension: use it unless --no-so-bev-pool is set
        ext_so = Path(args.ov_extension_so)
        want_so = (not getattr(args, 'no_so_bev_pool', False)
                   and not getattr(args, 'use_opencl_bev_pool', False))
        if want_so and ext_so.exists():
            if bevpool_device.upper().startswith('GPU') and not Path(args.ov_gpu_config_xml).exists():
                print(f'  [WARNING] OV extension (.so) found but CONFIG_FILE not found: {args.ov_gpu_config_xml}')
                print('  [WARNING] Falling back to NumPy BEV pool.')
            else:
                print('  Initializing BEV pool via OpenVINO extension (.so) ...')
                bev_pool_so = BEVPoolOVSO(
                    extension_so=str(ext_so),
                    device=bevpool_device,
                    gpu_config_xml=args.ov_gpu_config_xml,
                    gpu_precision=args.ov_gpu_precision,
                    inference_precision=(args.ov_bevpool_inference_precision or args.ov_inference_precision),
                    verbose=True,
                )
        elif want_so and not ext_so.exists():
            print(f'  [INFO] OV extension .so not found at {ext_so}  → using NumPy BEV pool.')
            print('  [INFO] Build with: cd openvino_extensions/bev_pool && mkdir build && cd build && cmake .. && make -j')

        if getattr(args, 'use_opencl_bev_pool', False) and bev_pool_so is None:
            if not _BEV_POOL_OPENCL_LOADED:
                print('  [WARNING] --use-opencl-bev-pool requested but bev_pool_opencl module '
                      'could not be loaded. Falling back to NumPy.')
            elif not _bev_pool_opencl_available():
                print('  [WARNING] --use-opencl-bev-pool: pyopencl found but no OpenCL '
                      'devices available. Falling back to NumPy.')
            else:
                prefer_gpu = getattr(args, 'opencl_prefer_gpu', True)
                print(f'  Initializing OpenCL BEV pool (prefer_gpu={prefer_gpu}) ...')
                bev_pool_cl = _BEVPoolOpenCL(prefer_gpu=prefer_gpu, verbose=True)

        ov_pipe = OpenVINOSplitPipeline(
            args.model_dir,
            args.ov_device,
            args.ov_gpu_precision,
            inference_precision=args.ov_inference_precision,
            enc_inference_precision=args.ov_enc_inference_precision,
            trk_inference_precision=args.ov_trk_inference_precision,
            bev_pool_cl=bev_pool_cl,
            bev_pool_so=bev_pool_so,
            profile_image_encoder=args.ov_profile_image_encoder,
            report_infer_only_timing=(not args.ov_disable_infer_only_timing),
            force_static_reshape=(not args.ov_disable_static_reshape),
            enable_io_optimizations=(not args.ov_disable_io_optimizations),
            use_remote_tensors=(not args.ov_disable_remote_tensors),
            enable_nan_debug=getattr(args, 'ov_enable_nan_debug', False),
        )

    pt_save_label_dir = None
    ov_save_label_dir = None
    dump_dir = None
    if getattr(args, 'dump_tensors', False):
        dump_dir = Path(args.tensor_dump_dir)
        dump_dir.mkdir(exist_ok=True, parents=True)
        print(f"  Saving intermediate OV tensors to: {dump_dir}")
    if args.save_label_dir:
        save_base = Path(args.save_label_dir)
        if args.run in ('both', 'pytorch', 'pytorch_twice'):
            pt_save_label_dir = save_base / 'pytorch'
            print(f"  Saving PT labels to : {pt_save_label_dir}")
        if args.run in ('both', 'openvino'):
            ov_save_label_dir = save_base / 'openvino'
            print(f"  Saving OV labels to : {ov_save_label_dir}")

    pt_stats, pt2_stats, ov_stats = [], [], []
    pt_timings, pt2_timings, ov_timings = [], [], []
    consistency_stats = []
    pt_gt_consistency_stats = []
    ov_gt_consistency_stats = []

    print("\n" + "─" * 72)
    print("  Real nuScenes data")
    print("─" * 72)

    # Helper: process one completed OV result (save labels, stats, print, consistency check)
    def _record_ov_sample(idx: int, ov_result: tuple, pt_lbl=None, pt_saved_lbl_dir=None):
        """Record OV inference result for sample *idx* into shared stats lists."""
        occ_label, occ_raw, timings, nan_debug, tensor_state = ov_result
        if ov_save_label_dir is not None:
            save_occ_label(occ_label, ov_save_label_dir, idx)
        if dump_dir is not None and idx < 3:
            np.save(dump_dir / f'ov_sample{idx:02d}_enc_tran_feat.npy', tensor_state['tran_feat'])
            np.save(dump_dir / f'ov_sample{idx:02d}_enc_depth.npy', tensor_state['depth'])
            np.save(dump_dir / f'ov_sample{idx:02d}_bev_feat.npy', tensor_state['bev_feat'])
        stats = occ_stats(occ_label, occ_raw)
        ov_stats.append(stats)
        ov_timings.append(timings)
        top3 = sorted(stats['class_counts'].items(), key=lambda x: x[1], reverse=True)[:3]
        nan_info = (
            f"nan[enc={nan_debug['enc_tran_feat'].get('nan_count','?')},"
            f"bev={nan_debug['bev_feat'].get('nan_count','?')},"
            f"trk={nan_debug['occ_pred'].get('nan_count','?')}]"
            if nan_debug.get('enc_tran_feat') else ""
        )
        # print(f"  OV  Sample {idx:3d}: occ={stats['occ_fraction']:.3f} max_logit={stats['max_logit']:.2f} "
        #       f"total={timings['total']*1000:.0f}ms top3={[f'{k}={v}' for k,v in top3]} "
        #       f"{nan_info}")
        # Consistency vs PT (if available)
        ov_lbl = occ_label
        if ov_saved_label_dir is not None and ov_lbl is None:
            ov_lbl = load_occ_label(ov_saved_label_dir, idx)
        cur_pt_lbl = pt_lbl
        if cur_pt_lbl is None and pt_saved_label_dir is not None:
            cur_pt_lbl = load_occ_label(pt_saved_label_dir, idx)
        if cur_pt_lbl is not None and ov_lbl is not None and args.run not in ('pytorch_twice',):
            consistency = compare_pt_ov_labels(cur_pt_lbl, ov_lbl)
            consistency_stats.append(consistency)
            print(f"  CMP Sample {idx:3d}: agree={consistency['voxel_agreement'] * 100:.2f}% "
                  f"occ_iou={consistency['occupied_iou']:.4f} "
                  f"miou_no_free={consistency['miou_no_free']:.4f}")
        if args.compare_with_gt:
            gt_use_mask = 'none' if args.no_gt_mask else 'lidar'
            gt_occ_label, gt_eval_mask = load_gt_occ_label(
                infos[idx], data_root, gt_key=args.gt_key, use_mask=gt_use_mask
            )
            mask_info = '' if gt_eval_mask is None else f' mask={gt_eval_mask.sum()}/{gt_eval_mask.size}'
            if ov_lbl is not None:
                gt_ov = compare_pt_ov_labels(gt_occ_label, ov_lbl, eval_mask=gt_eval_mask)
                ov_gt_consistency_stats.append(gt_ov)
                print(f"  GT/OV Sample {idx:3d}: agree={gt_ov['voxel_agreement'] * 100:.2f}% "
                      f"occ_iou={gt_ov['occupied_iou']:.4f} "
                      f"miou_no_free={gt_ov['miou_no_free']:.4f}{mask_info}")
        return occ_label

    # Per-sample PT labels stored for pipelined consistency comparison (1-frame offset).
    _pt_labels: dict[int, np.ndarray] = {}

    # Whether to use the double-buffered async pipeline for OV inference.
    _use_ov_async = (ov_pipe is not None
                     and getattr(args, 'ov_async_pipeline', True)
                     and args.run in ('openvino', 'both'))
    if _use_ov_async:
        print(f"  [OpenVINOSplitPipeline] Async inter-frame pipeline: ENABLED "
              f"(enc_N queued while trunk_N-1 runs on GPU)")

    _ov_wall_t0: float | None = None  # wall-clock start for OV (first run_pipelined or run call)
    _ov_wall_t1: float | None = None  # wall-clock end (after flush or last run)
    _ov_wall_frames: int = 0          # frames that count toward wall-clock FPS

    # ── Async data prefetch setup ─────────────────────────────────────────────
    # Prefetch is only useful when OV inference is running (not pure PT).
    _use_prefetch = (getattr(args, 'prefetch_data', True)
                     and args.run in ('openvino', 'both')
                     and n > 0)
    _prefetch_executor: concurrent.futures.Executor | None = None
    _prefetch_futures: deque[concurrent.futures.Future] = deque()
    _prefetch_wait_ms_list: list[float] = []
    _prefetched_sample_0 = None
    _prefetch_submit_idx = 1
    _prefetch_depth = max(1, int(getattr(args, 'prefetch_depth', 2)))
    _prefetch_backend = getattr(args, 'prefetch_backend', 'thread')
    _load_sample_fn = load_sample_inputs_cached

    def _submit_prefetch(idx: int):
        assert _prefetch_executor is not None
        return _prefetch_executor.submit(_load_sample_fn, infos[idx], data_root, sample_cache_dir)

    if _use_prefetch:
        if _prefetch_backend == 'process':
            mp_method = 'fork' if sys.platform != 'win32' else 'spawn'
            _prefetch_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=1,
                mp_context=mp.get_context(mp_method),
            )
        else:
            _prefetch_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix='prefetch')
        cache_msg = f", cache={sample_cache_dir}" if sample_cache_dir is not None else ""
        print(f"  [Prefetch] Async data prefetching: ENABLED "
              f"(backend={_prefetch_backend}, depth={_prefetch_depth}{cache_msg})")

    _run_deadline: float | None = time.time() + args.run_duration if args.run_duration > 0 else None
    _pass_idx = 0
    while True:
        # ── Per-pass prefetch bootstrap ───────────────────────────────────────
        if _use_prefetch:
            _prefetch_futures.clear()
            _prefetch_submit_idx = 1
            _prefetched_sample_0 = _load_sample_fn(infos[0], data_root, sample_cache_dir)
            while _prefetch_submit_idx < n and len(_prefetch_futures) < _prefetch_depth:
                _prefetch_futures.append(_submit_prefetch(_prefetch_submit_idx))
                _prefetch_submit_idx += 1

        for i in range(n):
            # ── Sample loading (with optional async prefetch) ─────────────────────
            if _use_prefetch:
                if i == 0:
                    sample = _prefetched_sample_0
                else:
                    # Retrieve the oldest prefetched sample (should already be ready).
                    t_wait = time.perf_counter()
                    sample = _prefetch_futures.popleft().result()
                    wait_ms = (time.perf_counter() - t_wait) * 1000
                    _prefetch_wait_ms_list.append(wait_ms)
                # Keep a small look-ahead queue full so later frames get more than
                # one frame-time of slack for disk I/O + CPU preprocessing.
                while _prefetch_submit_idx < n and len(_prefetch_futures) < _prefetch_depth:
                    _prefetch_futures.append(_submit_prefetch(_prefetch_submit_idx))
                    _prefetch_submit_idx += 1
            else:
                sample = _load_sample_fn(infos[i], data_root, sample_cache_dir)
            pt_occ_label = None
            pt2_occ_label = None
            ov_occ_label = None
    
            if pt_pipe is not None:
                occ_label, occ_raw, timings, nan_debug = pt_pipe.run(sample)
                pt_occ_label = occ_label
                _pt_labels[i] = occ_label
                if pt_save_label_dir is not None:
                    save_occ_label(pt_occ_label, pt_save_label_dir, i)
                stats = occ_stats(occ_label, occ_raw)
                pt_stats.append(stats)
                pt_timings.append(timings)
                top3 = sorted(stats['class_counts'].items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"  PT  Sample {i:3d}: occ={stats['occ_fraction']:.3f} max_logit={stats['max_logit']:.2f} "
                      f"total={timings['total']*1000:.0f}ms top3={[f'{k}={v}' for k,v in top3]} "
                      f"nan[trk={nan_debug['occ_pred'].get('nan_count','?')}]")
    
                if args.run == 'pytorch_twice':
                    occ_label2, occ_raw2, timings2, nan_debug2 = pt_pipe.run(sample)
                    pt2_occ_label = occ_label2
                    stats2 = occ_stats(occ_label2, occ_raw2)
                    pt2_stats.append(stats2)
                    pt2_timings.append(timings2)
                    top3_2 = sorted(stats2['class_counts'].items(), key=lambda x: x[1], reverse=True)[:3]
                    print(f"  PT2 Sample {i:3d}: occ={stats2['occ_fraction']:.3f} max_logit={stats2['max_logit']:.2f} "
                        f"total={timings2['total']*1000:.0f}ms top3={[f'{k}={v}' for k,v in top3_2]} "
                        f"nan[trk={nan_debug2['occ_pred']['nan_count']}]")
    
            if ov_pipe is not None:
                debug_geom_this = getattr(args, 'debug_geom', False) and (i == 0)
                if _use_ov_async:
                    # Wall-clock: start just before first GPU work (warmup frames included so
                    # we measure from the very first submission; warmup FPS excluded from stage
                    # timings but wall-clock gives a single holistic number).
                    if _ov_wall_t0 is None and i >= args.latency_warmup_frames:
                        _ov_wall_t0 = time.time()
                    # Pipelined path: submit enc_i + collect result for i-1.
                    prev_result = ov_pipe.run_pipelined(sample, debug_geom=debug_geom_this)
                    if prev_result is not None:
                        # Result is for sample i-1; process it now.
                        prev_pt_lbl = _pt_labels.get(i - 1)
                        ov_occ_label_prev = _record_ov_sample(i - 1, prev_result, prev_pt_lbl)
                        if _ov_wall_t0 is not None:
                            _ov_wall_frames += 1
                        # Consistency for i-1 vs PT is handled inside _record_ov_sample.
                    # Current sample's OV result will arrive in the next iteration.
                    ov_occ_label = None  # not yet available
                else:
                    if _ov_wall_t0 is None and i >= args.latency_warmup_frames:
                        _ov_wall_t0 = time.time()
                    # Standard synchronous path (unchanged behaviour).
                    occ_label, occ_raw, timings, nan_debug, tensor_state = ov_pipe.run(sample, debug_geom=debug_geom_this)
                    ov_occ_label = _record_ov_sample(i, (occ_label, occ_raw, timings, nan_debug, tensor_state), pt_occ_label)
                    if _ov_wall_t0 is not None:
                        _ov_wall_frames += 1
                        _ov_wall_t1 = time.time()
    
            if not _use_ov_async:
                # Non-pipelined consistency checks (OV available in-loop).
                if pt_occ_label is None and pt_saved_label_dir is not None:
                    pt_occ_label = load_occ_label(pt_saved_label_dir, i)
                if ov_occ_label is None and ov_saved_label_dir is not None:
                    ov_occ_label = load_occ_label(ov_saved_label_dir, i)
    
                if args.run == 'pytorch_twice' and pt_occ_label is not None and pt2_occ_label is not None:
                    consistency = compare_pt_ov_labels(pt_occ_label, pt2_occ_label)
                    consistency_stats.append(consistency)
                    print(f"  PT/PT Sample {i:3d}: agree={consistency['voxel_agreement'] * 100:.2f}% "
                          f"occ_iou={consistency['occupied_iou']:.4f} "
                          f"miou_no_free={consistency['miou_no_free']:.4f}")
    
                if args.compare_with_gt:
                    gt_use_mask = 'none' if args.no_gt_mask else 'lidar'
                    gt_occ_label, gt_eval_mask = load_gt_occ_label(
                        infos[i], data_root, gt_key=args.gt_key, use_mask=gt_use_mask
                    )
                    mask_info = '' if gt_eval_mask is None else f' mask={gt_eval_mask.sum()}/{gt_eval_mask.size}'
    
                    if pt_occ_label is not None:
                        gt_pt = compare_pt_ov_labels(gt_occ_label, pt_occ_label, eval_mask=gt_eval_mask)
                        pt_gt_consistency_stats.append(gt_pt)
                        print(f"  GT/PT Sample {i:3d}: agree={gt_pt['voxel_agreement'] * 100:.2f}% "
                              f"occ_iou={gt_pt['occupied_iou']:.4f} "
                              f"miou_no_free={gt_pt['miou_no_free']:.4f}{mask_info}")
    
                    if ov_occ_label is not None:
                        gt_ov = compare_pt_ov_labels(gt_occ_label, ov_occ_label, eval_mask=gt_eval_mask)
                        ov_gt_consistency_stats.append(gt_ov)
                    print(f"  GT/OV Sample {i:3d}: agree={gt_ov['voxel_agreement'] * 100:.2f}% "
                          f"occ_iou={gt_ov['occupied_iou']:.4f} "
                          f"miou_no_free={gt_ov['miou_no_free']:.4f}{mask_info}")
    
        # ── End of pass: flush async pipeline ─────────────────────────────────
        if _use_ov_async and ov_pipe is not None:
            last_result = ov_pipe.flush_pipeline()
            if last_result is not None:
                last_pt_lbl = _pt_labels.get(n - 1)
                _record_ov_sample(n - 1, last_result, last_pt_lbl)
                if _ov_wall_t0 is not None:
                    _ov_wall_frames += 1
            _ov_wall_t1 = time.time()

        _pass_idx += 1
        if _run_deadline is None or time.time() >= _run_deadline:
            break  # end while True (timed run loop)

    # ── Prefetch executor shutdown + stats ────────────────────────────────────
    if _prefetch_executor is not None:
        _prefetch_executor.shutdown(wait=True, cancel_futures=True)
    if _prefetch_wait_ms_list:
        avg_wait = sum(_prefetch_wait_ms_list) / len(_prefetch_wait_ms_list)
        max_wait = max(_prefetch_wait_ms_list)
        print(f"\n  [Prefetch] Stats: avg_wait={avg_wait:.1f}ms  max_wait={max_wait:.1f}ms  "
              f"frames_prefetched={len(_prefetch_wait_ms_list)}")
        if avg_wait < 5.0:
            print(f"  [Prefetch] ✓ Data arrived before GPU was ready — full overlap achieved")
        else:
            print(f"  [Prefetch] ⚠ avg wait > 5ms — prefetch may not fully hide I/O latency")

    print("\n" + "=" * 72)
    print("  LATENCY & THROUGHPUT")
    print("=" * 72)
    if pt_timings:
        print_latency_throughput(pt_timings, 'PyTorch CUDA', warmup_frames=args.latency_warmup_frames)
    if pt2_timings:
        print_latency_throughput(pt2_timings, 'PyTorch CUDA (pass2)', warmup_frames=args.latency_warmup_frames)
    if ov_timings:
        print_latency_throughput(ov_timings, 'OpenVINO split', warmup_frames=args.latency_warmup_frames)
        if _use_ov_async:
            print(f"  NOTE: In async-pipeline mode 'bev_trunk' wall-time includes inter-frame data-loading gap.")
            print(f"        Use Wall-clock FPS below for true E2E throughput comparison.")
        if _ov_wall_t0 is not None and _ov_wall_t1 is not None and _ov_wall_frames > 0:
            wall_fps = _ov_wall_frames / (_ov_wall_t1 - _ov_wall_t0)
            wall_ms = 1000.0 / wall_fps
            pipeline_label = "async pipeline" if _use_ov_async else "sync"
            print(f"\n  [OpenVINO split] Wall-clock E2E ({pipeline_label}, includes data loading):")
            print(f"  {'Frames measured':<22}: {_ov_wall_frames}")
            print(f"  {'Wall-clock FPS':<22}: {wall_fps:>8.2f} fps  ({wall_ms:.1f} ms/frame)")
        if ov_pipe is not None and args.ov_profile_image_encoder:
            print_ov_image_encoder_profile(ov_pipe.get_image_encoder_profile_summary(topk=args.ov_profile_topk))

    print("\n" + "=" * 72)
    print("  ACCURACY")
    print("=" * 72)
    pt_agg = aggregate_stats(pt_stats) if pt_stats else None
    pt2_agg = aggregate_stats(pt2_stats) if pt2_stats else None
    ov_agg = aggregate_stats(ov_stats) if ov_stats else None
    if pt_agg:
        print_accuracy(pt_agg, 'PyTorch CUDA')
    if pt2_agg:
        print_accuracy(pt2_agg, 'PyTorch CUDA (pass2)')
    if ov_agg:
        print_accuracy(ov_agg, 'OpenVINO split')
    if pt_agg and ov_agg:
        print_backend_comparison('PyTorch', pt_agg, 'OpenVINO', ov_agg)
    if pt_agg and pt2_agg:
        print_backend_comparison('PyTorch pass1', pt_agg, 'PyTorch pass2', pt2_agg)
    if consistency_stats:
        consistency_agg = aggregate_consistency(consistency_stats)
        if args.run == 'pytorch_twice':
            print_consistency(consistency_agg, 'PT vs PT CONSISTENCY', 'PT1 vs PT2 confusion matrix')
        else:
            print_consistency(consistency_agg, 'PT vs OV CONSISTENCY', 'PT vs OV confusion matrix')
        if args.save_consistency_csv:
            if args.run == 'pytorch_twice':
                save_consistency_csv(consistency_agg, Path(args.consistency_output_dir), prefix='pt_pt', matrix_header='pt1\\pt2')
            else:
                save_consistency_csv(consistency_agg, Path(args.consistency_output_dir), prefix='pt_ov', matrix_header='pt\\ov')

    if pt_gt_consistency_stats:
        pt_gt_agg = aggregate_consistency(pt_gt_consistency_stats)
        print_consistency(pt_gt_agg, 'GT vs PT CONSISTENCY', 'GT vs PT confusion matrix')
        if args.save_consistency_csv:
            save_consistency_csv(pt_gt_agg, Path(args.consistency_output_dir), prefix='gt_pt', matrix_header='gt\\pt')

    if ov_gt_consistency_stats:
        ov_gt_agg = aggregate_consistency(ov_gt_consistency_stats)
        print_consistency(ov_gt_agg, 'GT vs OV CONSISTENCY', 'GT vs OV confusion matrix')
        if args.save_consistency_csv:
            save_consistency_csv(ov_gt_agg, Path(args.consistency_output_dir), prefix='gt_ov', matrix_header='gt\\ov')

if __name__ == '__main__':
    main()
