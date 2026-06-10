"""
BEVPool OpenCL GPU implementation for FlashOCC.

Drop-in replacement for bev_pool_v2_numpy using the GPU kernels in this folder.
Uses pyopencl to run bev_pool_binsort.cl (geometry sort) and bev_pool_v2.cl (scatter).

FlashOCC-specific constants used:
  NX = NY = 200      (x/y: -40..40, step 0.4)
  CHANNELS = 64
  FEAT_H = 16, FEAT_W = 44  (INPUT 256x704 / DOWNSAMPLE 16)
  FEAT_HW = 704
  D = 44             (depth bins 1..45, step 1.0)
  DEPTH_HW = 30976   (D * FEAT_HW)
  TOTAL_PTS = 185856 (6 cams * DEPTH_HW)
  X_MIN = Y_MIN = -40.0
  X_STEP = Y_STEP = 0.4
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# FlashOCC constants (must match GRID_CONFIG and INPUT dims in main script)
# ---------------------------------------------------------------------------
_NX = 200
_NY = 200
_CHANNELS = 64
_FEAT_H = 16
_FEAT_W = 44
_FEAT_HW = _FEAT_H * _FEAT_W          # 704
_D = 44
_DEPTH_HW = _D * _FEAT_HW             # 30976
_TOTAL_PTS = 6 * _DEPTH_HW            # 185856
_PACKED_SIZE = _TOTAL_PTS * 2 + _NX * _NY * 2  # 451712  int32 elements

_X_MIN = -40.0
_Y_MIN = -40.0
_X_STEP = 0.4
_Y_STEP = 0.4

_INPUT_W = 704
_INPUT_H = 256
_DEPTH_START = 1.0
_DEPTH_STEP = 1.0

_KERNEL_DIR = Path(__file__).resolve().parent


def _patch_binsort_src(src: str) -> str:
    """Replace BEVFusion grid constants with FlashOCC values."""
    src = src.replace('#define X_MIN  -54.0f', f'#define X_MIN  {_X_MIN}f')
    src = src.replace('#define Y_MIN  -54.0f', f'#define Y_MIN  {_Y_MIN}f')
    src = src.replace('#define X_STEP   0.3f', f'#define X_STEP  {_X_STEP}f')
    src = src.replace('#define Y_STEP   0.3f', f'#define Y_STEP  {_Y_STEP}f')
    return src


def _v2_build_options() -> str:
    """OpenCL build defines for bev_pool_v2.cl (FlashOCC values)."""
    return (
        f'-DNX={_NX} -DNY={_NY} -DCHANNELS={_CHANNELS} '
        f'-DFEAT_HW={_FEAT_HW} -DDEPTH_HW={_DEPTH_HW} '
        f'-DTOTAL_PTS={_TOTAL_PTS} '
        f'-DINPUT0_TYPE=float -DINPUT1_TYPE=float '
        f'-DINPUT2_TYPE=int -DOUTPUT0_TYPE=float'
    )


def _binsort_build_options() -> str:
    """OpenCL build defines for bev_pool_binsort.cl (FlashOCC values)."""
    return (
        f'-DNX={_NX} -DNY={_NY} -DTOTAL_PTS={_TOTAL_PTS} '
        f'-DINPUT0_TYPE=float -DOUTPUT0_TYPE=int'
    )


def _select_cl_device(prefer_gpu: bool = True):
    """
    Select an OpenCL device.
    Returns (platform, device) or raises RuntimeError if none found.
    """
    try:
        import pyopencl as cl
    except ImportError:
        raise ImportError(
            'pyopencl is required for GPU BEV pooling. '
            'Install with: pip install pyopencl'
        )

    preferred_type = cl.device_type.GPU if prefer_gpu else cl.device_type.CPU

    for platform in cl.get_platforms():
        devices = platform.get_devices(device_type=preferred_type)
        if devices:
            return platform, devices[0]

    # Fall back to any device
    for platform in cl.get_platforms():
        devices = platform.get_devices()
        if devices:
            return platform, devices[0]

    raise RuntimeError('No OpenCL devices found.')


class BEVPoolOpenCL:
    """
    GPU-accelerated BEV pooling via OpenCL.

    Replaces bev_pool_v2_numpy with a two-kernel pipeline:
      1. bev_pool_binsort: geometry sort (CPU => GPU geom upload, single WG sort)
      2. bev_pool_v2:      feature scatter (per-frame, fully parallel on GPU)

    The geometry array [TOTAL_PTS, 3] is computed from camera calibration in
    NumPy (identical math to bev_pool_v2_numpy projection step) and uploaded
    each sample. The expensive np.add.at scatter is fully replaced by GPU.

    Usage:
        bev_cl = BEVPoolOpenCL()
        # per frame:
        bev_feat = bev_cl.run(depth, tran_feat, sample)
    """

    def __init__(self, prefer_gpu: bool = True, verbose: bool = False):
        import pyopencl as cl

        platform, device = _select_cl_device(prefer_gpu)
        self._device_name = device.name.strip()
        if verbose:
            print(f'  [BEVPoolOpenCL] Using device: {self._device_name}')

        self._ctx = cl.Context([device])
        self._queue = cl.CommandQueue(
            self._ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE,
        )
        self._cl = cl

        # ---- Load and compile bev_pool_binsort ----
        binsort_src_path = _KERNEL_DIR / 'bev_pool' / 'bev_pool_binsort.cl'
        if not binsort_src_path.exists():
            binsort_src_path = _KERNEL_DIR / 'bev_pool_binsort.cl'
        binsort_src = _patch_binsort_src(binsort_src_path.read_text())
        self._binsort_prog = cl.Program(self._ctx, binsort_src).build(
            _binsort_build_options()
        )
        self._binsort_k = self._binsort_prog.bevpool_binsort

        # ---- Load and compile bev_pool_v2 ----
        v2_src_path = _KERNEL_DIR / 'bev_pool' / 'bev_pool_v2.cl'
        if not v2_src_path.exists():
            v2_src_path = _KERNEL_DIR / 'bev_pool_v2.cl'
        v2_src = v2_src_path.read_text()
        self._v2_prog = cl.Program(self._ctx, v2_src).build(
            _v2_build_options()
        )
        self._v2_k = self._v2_prog.bevpool_v2

        # ---- Pre-allocate persistent GPU buffers ----
        mf = cl.mem_flags
        # packed sort buffer: [sorted_ranks | cell_scratch | starts | lengths]
        self._packed_buf = cl.Buffer(
            self._ctx, mf.READ_WRITE,
            size=_PACKED_SIZE * 4  # int32
        )
        # output bev feature buffer
        self._bev_buf = cl.Buffer(
            self._ctx, mf.WRITE_ONLY,
            size=_CHANNELS * _NY * _NX * 4  # float32
        )

        # Pre-compute camera-space partial geometry (before ego transforms).
        # pts_orig_per_cam[cam] = [3, DEPTH_HW] float32 in camera frame.
        # This part is FIXED per session (depends only on intrinsics +
        # post_rots + post_trans which are constant for a given camera rig).
        # Recomputed only when _intrinsic_key changes.
        self._intrinsic_key: tuple | None = None
        self._cam_pts_cache: list[np.ndarray] | None = None  # [cam][3, DEPTH_HW]

        if verbose:
            print(f'  [BEVPoolOpenCL] Kernels compiled. Device: {self._device_name}')



    # ------------------------------------------------------------------
    # Internal: compute geometry [TOTAL_PTS, 3] for a sample
    # ------------------------------------------------------------------

    @staticmethod
    def _make_intrinsic_key(sample: dict) -> tuple:
        """Cheap hash of camera intrinsics + post transforms for cache invalidation."""
        return (
            tuple(sample['intrins'].ravel()[:12].tolist()),
            tuple(sample['post_rots'].ravel()[:9].tolist()),
            tuple(sample['post_trans'].ravel()[:3].tolist()),
        )

    def _precompute_cam_pts(self, sample: dict):
        """
        Compute and cache camera-space 3D points (before ego transforms).
        Shape per camera: [3, DEPTH_HW]  (x, y, z in camera frame, z = depth_val)
        """
        if self._intrinsic_key == self._make_intrinsic_key(sample):
            return  # already cached

        feat_xs = np.linspace(0, _INPUT_W - 1, _FEAT_W, dtype=np.float32)
        feat_ys = np.linspace(0, _INPUT_H - 1, _FEAT_H, dtype=np.float32)
        depth_vals = (
            np.arange(_D, dtype=np.float32) * _DEPTH_STEP + _DEPTH_START
        )

        d_idx, h_idx, w_idx = np.meshgrid(
            np.arange(_D), np.arange(_FEAT_H), np.arange(_FEAT_W),
            indexing='ij'
        )
        d_flat = depth_vals[d_idx.ravel()]
        px_flat = feat_xs[w_idx.ravel()]
        py_flat = feat_ys[h_idx.ravel()]

        cam_pts = []
        for cam in range(6):
            post_rot = sample['post_rots'][cam]
            post_tr = sample['post_trans'][cam]
            k = sample['intrins'][cam]

            pts_aug = np.stack(
                [px_flat, py_flat, np.ones(_DEPTH_HW, np.float32)], axis=0
            )
            pts = pts_aug - post_tr[:, None]
            pts_orig = np.linalg.inv(post_rot) @ pts

            cx, cy = k[0, 2], k[1, 2]
            fx, fy = k[0, 0], k[1, 1]
            cam_x = (pts_orig[0] - cx) * d_flat / fx
            cam_y = (pts_orig[1] - cy) * d_flat / fy
            cam_z = d_flat  # z = depth value
            cam_pts.append(
                np.stack([cam_x, cam_y, cam_z], axis=0).astype(np.float32)
            )

        self._cam_pts_cache = cam_pts
        self._intrinsic_key = self._make_intrinsic_key(sample)

    def _compute_geom(self, sample: dict) -> np.ndarray:
        """
        Compute world-space geometry [TOTAL_PTS, 3] for binsort.
        Each row is the (x, y, z) in key-ego (vehicle) frame using ego2global transform.
        """
        self._precompute_cam_pts(sample)

        key_ego_inv = np.linalg.inv(sample['ego2globals'][0])
        geom = np.empty((_TOTAL_PTS, 3), dtype=np.float32)

        for cam in range(6):
            cam_pts_3 = self._cam_pts_cache[cam]   # [3, DEPTH_HW]
            cam_pts_4 = np.vstack([
                cam_pts_3,
                np.ones((1, _DEPTH_HW), dtype=np.float32)
            ])  # [4, DEPTH_HW]

            cam_to_key_ego = (
                key_ego_inv
                @ sample['ego2globals'][cam]
                @ sample['sensor2egos'][cam]
            )
            pts_key_ego = cam_to_key_ego @ cam_pts_4  # [4, DEPTH_HW]

            base = cam * _DEPTH_HW
            geom[base: base + _DEPTH_HW, 0] = pts_key_ego[0]
            geom[base: base + _DEPTH_HW, 1] = pts_key_ego[1]
            geom[base: base + _DEPTH_HW, 2] = pts_key_ego[2]

        return geom

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def device_name(self) -> str:
        return self._device_name

    def run(
        self,
        depth: np.ndarray,
        tran_feat: np.ndarray,
        sample: dict,
    ) -> np.ndarray:
        """
        GPU BEV pooling.

        Args:
            depth:     [6, D, FEAT_H, FEAT_W] float32 - depth softmax from encoder
            tran_feat: [6, CHANNELS, FEAT_H, FEAT_W] float32 - features from encoder
            sample:    dict with 'intrins', 'post_rots', 'post_trans',
                       'sensor2egos', 'ego2globals'

        Returns:
            bev_feat: [1, CHANNELS, NY, NX] float32
        """
        cl = self._cl
        mf = cl.mem_flags

        # 1. Compute geometry on CPU (fast: matrix math only, no scatter)
        geom = self._compute_geom(sample)

        # 2. Upload geometry to GPU and run binsort (geometry sort)
        #    geom: [TOTAL_PTS, 3] float32
        geom_c = np.ascontiguousarray(geom)
        geom_buf = cl.Buffer(
            self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=geom_c
        )
        self._binsort_k(
            self._queue,
            (1024,), (1024,),
            geom_buf, self._packed_buf,
        )

        # 3. Upload depth + features and run bev_pool_v2 (feature scatter)
        #    depth: [6, D, FEAT_H, FEAT_W] -> [1, 6*D, FEAT_H, FEAT_W] (flat)
        #    tran_feat: [6, C, FEAT_H, FEAT_W] (flat)
        depth_c = np.ascontiguousarray(depth.reshape(-1))
        feat_c = np.ascontiguousarray(tran_feat.reshape(-1))

        depth_buf = cl.Buffer(
            self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=depth_c
        )
        feat_buf = cl.Buffer(
            self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=feat_c
        )

        # Zero output buffer before scatter
        cl.enqueue_fill_buffer(
            self._queue, self._bev_buf,
            np.float32(0.0), 0, _CHANNELS * _NY * _NX * 4
        )

        self._v2_k(
            self._queue,
            (_NX * _NY * _CHANNELS,), (_CHANNELS,),
            depth_buf, feat_buf, self._packed_buf, self._bev_buf,
        )

        # 4. Read back result
        bev_flat = np.empty(_CHANNELS * _NY * _NX, dtype=np.float32)
        cl.enqueue_copy(self._queue, bev_flat, self._bev_buf)
        self._queue.finish()

        # Kernel binsort uses cell = ix*NY + iy (x-major), so the output
        # buffer layout is [C, NX, NY].  The bev_trunk (and numpy version)
        # expect [1, C, NY, NX] (y-major), so we transpose the last two dims.
        bev = bev_flat.reshape(1, _CHANNELS, _NX, _NY).transpose(0, 1, 3, 2)
        return np.ascontiguousarray(bev)   # [1, CHANNELS, NY, NX]

    def warmup(self, sample: dict):
        """
        Run one dummy pass to trigger JIT compilation.
        Call once before the benchmark loop using a real sample.
        """
        dummy_depth = np.zeros((6, _D, _FEAT_H, _FEAT_W), dtype=np.float32)
        dummy_feat = np.zeros((6, _CHANNELS, _FEAT_H, _FEAT_W), dtype=np.float32)
        self.run(dummy_depth, dummy_feat, sample)


def available() -> bool:
    """Return True if pyopencl is installed and at least one device is found."""
    try:
        import pyopencl as cl
        return len(cl.get_platforms()) > 0
    except Exception:
        return False
