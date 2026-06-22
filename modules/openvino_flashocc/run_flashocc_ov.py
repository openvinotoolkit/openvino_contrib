# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


#!/usr/bin/env python3
"""FlashOCC OpenVINO end-to-end inference runner.

Runs the optimized OpenVINO deployment pipeline using a pluggable sample
provider, with the BEV pool OpenVINO extension as the required backend:
    - async inter-frame pipeline
    - encoder split submission
    - geometry/data overlap
    - fused bev_trunk+ArgMax when available
    - tensor reuse / remote-style input tensors
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
from collections import deque
from pathlib import Path
from typing import Protocol, runtime_checkable

_REPO_ROOT = Path(__file__).resolve().parent
_PROJECTS_DIR = _REPO_ROOT / "projects"
for _p in (_REPO_ROOT, _PROJECTS_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

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

# OpenVINO extension is the only supported BEV pool backend
_OV_EXT_DIR = Path(__file__).resolve().parent / 'openvino_extensions'
if str(_OV_EXT_DIR) not in sys.path:
    sys.path.insert(0, str(_OV_EXT_DIR))


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
D = int((GRID_CONFIG['depth'][1] - GRID_CONFIG['depth'][0]) / GRID_CONFIG['depth'][2])


def _compute_geom_ego2global(
        post_rots: np.ndarray,
        post_trans: np.ndarray,
        sensor2egos: np.ndarray,
        ego2globals: np.ndarray,
        intrinsics: np.ndarray,
) -> np.ndarray:
    """Legacy geometry path for debug comparison against runtime geometry."""
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

    key_ego_inv = np.linalg.inv(ego2globals[0])
    geom = np.empty((6 * n_pts, 3), dtype=np.float32)
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
        pts_key_ego = (cam_to_key_ego @ pts_cam)[:3].T.astype(np.float32, copy=False)
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


def _prepare_encoder_images(images: np.ndarray) -> np.ndarray:
    """Normalize provider image layout to encoder expected NCHW float32."""
    arr = np.asarray(images)
    if arr.ndim != 4:
        raise ValueError(f'Expected images with 4 dims [N,H,W,C] or [N,C,H,W], got shape={arr.shape}')
    if arr.shape[-1] == 3:
        arr = np.transpose(arr, (0, 3, 1, 2))
    elif arr.shape[1] != 3:
        raise ValueError(f'Expected channel dimension of 3, got shape={arr.shape}')
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return np.ascontiguousarray(arr)


def make_random_sample(rng: np.random.Generator | None = None) -> dict:
    """Generate a random sample with the correct shapes for FlashOCC inference.

    Use this to verify the pipeline works without real data, or as a template
    showing the exact format your own data must match.

    Input format expected by the pipeline
    --------------------------------------
    images      : float32  (6, 256, 704, 3)   — 6 camera images, BGR, 0-255 range
                                                order: FRONT, FRONT_RIGHT, BACK_RIGHT,
                                                       BACK,  BACK_LEFT,  FRONT_LEFT
    post_rots   : float32  (6, 3, 3)           — per-camera image-space augmentation
                                                rotation (identity = no augmentation)
    post_trans  : float32  (6, 3)              — per-camera image-space augmentation
                                                translation (zeros = no augmentation)
    sensor2egos : float32  (6, 4, 4)           — camera-to-ego-vehicle SE3 transform
    ego2globals : float32  (6, 4, 4)           — ego-vehicle-to-global SE3 transform
    intrins     : float32  (6, 3, 3)           — camera intrinsic matrix (K)

    Replace the values below with real calibration and image data from your sensor.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    N_CAMS = 6

    # Placeholder images: uniform noise in [0, 255]
    images = rng.uniform(0, 255, (N_CAMS, 256, 704, 3)).astype(np.float32)

    # Identity rotations (no image-space augmentation)
    post_rots = np.tile(np.eye(3, dtype=np.float32), (N_CAMS, 1, 1))

    # Zero translations (no image-space augmentation)
    post_trans = np.zeros((N_CAMS, 3), dtype=np.float32)

    # Identity SE3 transforms: cameras at the ego-vehicle origin facing forward
    sensor2egos = np.tile(np.eye(4, dtype=np.float32), (N_CAMS, 1, 1))
    ego2globals = np.tile(np.eye(4, dtype=np.float32), (N_CAMS, 1, 1))

    # Typical pinhole intrinsics for a 704x256 image (fx=fy≈550, cx=352, cy=128)
    K = np.array([[550.0, 0.0, 352.0],
                  [0.0, 550.0, 128.0],
                  [0.0, 0.0,   1.0]], dtype=np.float32)
    intrins = np.tile(K, (N_CAMS, 1, 1))

    return {
        'images':      images,
        'post_rots':   post_rots,
        'post_trans':  post_trans,
        'sensor2egos': sensor2egos,
        'ego2globals': ego2globals,
        'intrins':     intrins,
    }


@runtime_checkable
class SampleProvider(Protocol):
    """Interface for feeding samples into the inference pipeline.

    Implement this to connect any data source (nuScenes, ROS bag, live cameras,
    custom datasets, etc.).  The pipeline calls provider(index) for each frame;
    the returned dict must match the format described in make_random_sample().

    Example
    -------
    class MyCameraProvider:
        def __init__(self, image_dir, calibration):
            self._frames = sorted(Path(image_dir).glob('*.jpg'))
            self._calib = calibration          # your own loader

        def __len__(self) -> int:
            return len(self._frames)

        def __call__(self, index: int) -> dict:
            images = load_and_resize_6_cameras(self._frames[index])  # (6,256,704,3)
            calib  = self._calib[index]
            return {
                'images':      images,
                'post_rots':   calib['post_rots'],    # (6,3,3)
                'post_trans':  calib['post_trans'],   # (6,3)
                'sensor2egos': calib['sensor2egos'],  # (6,4,4)
                'ego2globals': calib['ego2globals'],  # (6,4,4)
                'intrins':     calib['intrins'],      # (6,3,3)
            }
    """
    def __call__(self, index: int) -> dict: ...
    def __len__(self) -> int: ...


class RandomSampleProvider:
    """Built-in SampleProvider that returns synthetic random samples.

    This is the default provider used when running without a real dataset.
    It exercises the full pipeline and is the reference for the input format
    every custom SampleProvider must return.

    To use real data, implement SampleProvider (see its docstring) and replace
    the ``provider = RandomSampleProvider(...)`` line in main() with your own.
    """

    def __init__(self, n_samples: int):
        self._n = n_samples
        self._rng = np.random.default_rng(42)

    def __len__(self) -> int:
        return self._n

    def __call__(self, index: int) -> dict:
        return make_random_sample(self._rng)


class CustomSampleProvider:
    """Template SampleProvider for plugging in real customer data.

    Update this class to load your own multi-camera frames and calibration data.
    The returned dict must follow the exact format documented in
    make_random_sample().
    """

    def __init__(self, n_samples: int):
        self._n = n_samples

    def __len__(self) -> int:
        return self._n

    def __call__(self, index: int) -> dict:
        # TODO: Replace this with your real data loading logic.
        # Keep this fallback so users can still run immediately while integrating.
        return make_random_sample()


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


class BEVPoolOVSO:
    def __init__(
        self,
        extension_so: str,
        device: str = 'GPU',
        gpu_config_xml: str | None = None,
        gpu_precision: str = 'auto',
        inference_precision: str | None = None,
        verbose: bool = False,
        shared_core: ov.Core | None = None,
    ):
        if not _ONNX_AVAILABLE:
            raise RuntimeError('SO BEV pool backend requires onnx Python package to be installed.')

        if shared_core is not None:
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

        compile_device = device
        if device.upper() == 'NPU':
            compile_device = 'CPU'
            if verbose:
                print('  [BEVPoolOVSO] NPU backend does not support custom extensions.')
                print('  [BEVPoolOVSO] Running BEV pool extension on CPU (main models will use NPU).')

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
        self._ext_so_path = str(extension_so)
        self._compile_device = compile_device
        self._compile_cfg = cfg

        if verbose:
            print(f'  [BEVPoolOVSO] Extension: {extension_so}')
            if gpu_config_xml:
                print(f'  [BEVPoolOVSO] GPU config: {gpu_config_xml}')
            print(f'  [BEVPoolOVSO] Device: {self.device}')

    def _rebind_to_shared_core(self, shared_core: ov.Core) -> None:
        """Recompile on a shared OV Core to avoid cross-context GPU sync overhead."""
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
        """Public wrapper to precompute geometry outside the critical path."""
        return self._compute_geom(sample)

    def run_with_geom(self, depth: np.ndarray, tran_feat: np.ndarray,
                      geom: np.ndarray,
                      return_tensor: bool = False) -> np.ndarray | ov.Tensor:
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


class OpenVINOSplitPipeline:
    def __init__(self, model_dir: str, device: str = 'GPU', gpu_precision: str = 'auto',
                 inference_precision: str | None = None,
                 enc_inference_precision: str | None = None,
                 trk_inference_precision: str | None = None,
                 bev_pool_so=None,
                 profile_image_encoder: bool = False,
                 report_infer_only_timing: bool = True,
                 force_static_reshape: bool = True,
                 enable_io_optimizations: bool = True,
                 use_remote_tensors: bool = True,
                 enable_nan_debug: bool = False):
        core = ov.Core()
        md = Path(model_dir)
        self._device = device
        self._bev_pool_so = bev_pool_so  # BEVPoolOVSO required
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

        print(f'  [OpenVINOSplitPipeline] Using BEV pool via OpenVINO extension (.so) on {bev_pool_so.device}')

    def run(self, sample: dict, debug_geom: bool = False) -> tuple[np.ndarray, np.ndarray, dict, dict]:
        timings = {}
        nan_debug = {}

        t0 = time.time()
        images = _prepare_encoder_images(sample['images'])
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

            images = _prepare_encoder_images(sample['images'])
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
            images = _prepare_encoder_images(sample['images'])

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





def main():
    parser = argparse.ArgumentParser(
        description='Run the optimized FlashOCC OpenVINO split pipeline'
    )
    # ── Model & device ────────────────────────────────────────────────────────
    parser.add_argument('--model-dir', required=True,
                        help='OpenVINO split model dir (contains *.image_encoder.xml and *.bev_trunk.xml)')
    parser.add_argument('--ov-device', default='GPU',
                        help='OpenVINO device for encoder and trunk models (default: GPU)')
    parser.add_argument('--ov-bevpool-device', default=None,
                        help='Device for BEV pool extension only; defaults to --ov-device if not set.')
    parser.add_argument('--ov-gpu-precision', choices=['auto', 'f16', 'f32'], default='f32',
                        help='GPU precision for models (auto/f16/f32, default: f32)')
    parser.add_argument('--ov-inference-precision', choices=['f32', 'f16'], default=None,
                        help='Force OV INFERENCE_PRECISION_HINT for all models (overrides --ov-gpu-precision)')
    parser.add_argument('--ov-enc-inference-precision', choices=['f32', 'f16'], default=None,
                        help='Per-model OV precision hint for image_encoder only')
    parser.add_argument('--ov-trk-inference-precision', choices=['f32', 'f16'], default=None,
                        help='Per-model OV precision hint for bev_trunk only')
    parser.add_argument('--ov-bevpool-inference-precision', choices=['f32', 'f16'], default=None,
                        help='Per-model OV precision hint for BEV pool extension only')
    parser.add_argument('--ov-extension-so',
                        default=str((_REPO_ROOT / 'openvino_extensions' / 'bev_pool' / 'build' / 'libopenvino_bevpool_extension.so').resolve()),
                        help='Path to BEV pool extension .so')
    parser.add_argument('--ov-gpu-config-xml',
                        default=str((_REPO_ROOT / 'openvino_extensions' / 'gpu_custom_layers.xml').resolve()),
                        help='GPU custom layers xml for BEV pool extension')
    # ── Run control ──────────────────────────────────────────────────────────
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of frames to run inference on')
    parser.add_argument('--run-duration', type=float, default=0.0,
                        help='Run inference for at least this many seconds, re-cycling samples. '
                             '0 (default) = run exactly --num-samples frames once.')
    parser.add_argument('--warmup-frames', type=int, default=5,
                        help='Exclude first N frames from latency/throughput reporting (cold-start stabilization)')
    parser.add_argument('--sample-provider', choices=['random', 'custom'], default='random',
                        help='Input source: random synthetic samples (for FPS testing) or custom provider template')
    args = parser.parse_args()

    n = args.num_samples
    bevpool_device = args.ov_bevpool_device or args.ov_device

    # ── Data provider ────────────────────────────────────────────────────────
    if args.sample_provider == 'custom':
        provider: SampleProvider = CustomSampleProvider(n_samples=n)
        print('  Data provider   : custom (template)')
    else:
        provider = RandomSampleProvider(n_samples=n)
        print('  Data provider   : random synthetic samples')

    # To run on real data, implement CustomSampleProvider.__call__ and return
    # the sample dict format defined by make_random_sample().
    # ──────────────────────────────────────────────────────────────────────────
    n = min(n, len(provider))

    print("=" * 72)
    print("  FlashOCC OpenVINO Inference")
    print("=" * 72)
    print(f"  Frames          : {n}")
    print(f"  OV models device: {args.ov_device}")
    print(f"  BEV pool device : {bevpool_device}")

    ext_so = Path(args.ov_extension_so)
    if not ext_so.exists():
        raise FileNotFoundError(
            f'Required BEV pool extension .so not found at {ext_so}. '
            'Build it before running deployment inference.'
        )
    if bevpool_device.upper().startswith('GPU') and not Path(args.ov_gpu_config_xml).exists():
        raise FileNotFoundError(
            f'Required GPU config XML for the BEV pool extension not found: {args.ov_gpu_config_xml}'
        )
    print('  Initializing required BEV pool OpenVINO extension (.so) ...')
    bev_pool_so = BEVPoolOVSO(
        extension_so=str(ext_so),
        device=bevpool_device,
        gpu_config_xml=args.ov_gpu_config_xml,
        gpu_precision=args.ov_gpu_precision,
        inference_precision=(args.ov_bevpool_inference_precision or args.ov_inference_precision),
        verbose=True,
    )

    ov_pipe = OpenVINOSplitPipeline(
        args.model_dir,
        args.ov_device,
        args.ov_gpu_precision,
        inference_precision=args.ov_inference_precision,
        enc_inference_precision=args.ov_enc_inference_precision,
        trk_inference_precision=args.ov_trk_inference_precision,
        bev_pool_so=bev_pool_so,
        profile_image_encoder=False,
        report_infer_only_timing=True,
        force_static_reshape=True,
        enable_io_optimizations=True,
        use_remote_tensors=True,
        enable_nan_debug=False,
    )

    ov_timings = []
    print('  [OpenVINOSplitPipeline] Async inter-frame pipeline: ENABLED (optimized for production latency)')

    ov_wall_t0: float | None = None
    ov_wall_t1: float | None = None
    ov_wall_frames = 0

    def _record_ov_sample(idx: int, ov_result: tuple):
        occ_label, _occ_raw, timings, _nan_debug, _tensor_state = ov_result
        ov_timings.append(timings)

    prefetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix='prefetch')
    prefetch_futures: deque[concurrent.futures.Future] = deque()
    prefetch_submit_idx = 1
    prefetch_depth = 2
    print('  [Prefetch] Async data prefetching: ENABLED (thread pool, depth=2)')

    def _submit_prefetch(idx: int):
        return prefetch_executor.submit(provider, idx)

    run_deadline: float | None = time.time() + args.run_duration if args.run_duration > 0 else None
    warmup_frames = args.warmup_frames
    
    while True:
        prefetch_futures.clear()
        prefetch_submit_idx = 1
        prefetched_sample_0 = provider(0)
        while prefetch_submit_idx < n and len(prefetch_futures) < prefetch_depth:
            prefetch_futures.append(_submit_prefetch(prefetch_submit_idx))
            prefetch_submit_idx += 1

        for i in range(n):
            if i == 0:
                sample = prefetched_sample_0
            else:
                sample = prefetch_futures.popleft().result()
            while prefetch_submit_idx < n and len(prefetch_futures) < prefetch_depth:
                prefetch_futures.append(_submit_prefetch(prefetch_submit_idx))
                prefetch_submit_idx += 1

            if ov_wall_t0 is None and i >= warmup_frames:
                ov_wall_t0 = time.time()
            
            prev_result = ov_pipe.run_pipelined(sample, debug_geom=False)
            if prev_result is not None:
                _record_ov_sample(i - 1, prev_result)
                if ov_wall_t0 is not None:
                    ov_wall_frames += 1

        last_result = ov_pipe.flush_pipeline()
        if last_result is not None:
            _record_ov_sample(n - 1, last_result)
            if ov_wall_t0 is not None:
                ov_wall_frames += 1
        ov_wall_t1 = time.time()

        if run_deadline is None or time.time() >= run_deadline:
            break

    prefetch_executor.shutdown(wait=True, cancel_futures=True)

    print("\n" + "=" * 72)
    print("  LATENCY & THROUGHPUT")
    print("=" * 72)
    if ov_timings:
        print_latency_throughput(ov_timings, 'OpenVINO split', warmup_frames=warmup_frames)
        print("  NOTE: In async-pipeline mode 'bev_trunk' wall-time includes inter-frame data-loading gap.")
        print('        Wall-clock FPS below reflects true E2E throughput including data loading.')
        if ov_wall_t0 is not None and ov_wall_t1 is not None and ov_wall_frames > 0:
            wall_fps = ov_wall_frames / (ov_wall_t1 - ov_wall_t0)
            wall_ms = 1000.0 / wall_fps
            print(f"\n  [OpenVINO split] Wall-clock E2E (async pipeline, includes data loading):")
            print(f"  {'Frames measured':<22}: {ov_wall_frames}")
            print(f"  {'Wall-clock FPS':<22}: {wall_fps:>8.2f} fps  ({wall_ms:.1f} ms/frame)")

if __name__ == '__main__':
    main()
