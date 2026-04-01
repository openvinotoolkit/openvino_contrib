"""
Voxelization using OpenVINO custom extension.

Two-layer GPU pipeline replaces the torch/OpenCL voxel_layer:
  1. VoxelizeScatter — GPU hash-table insert + int32 atomic scatter
  2. VoxelizeMean    — int32 fixed-point → float32 mean features

Produces the same output format as the existing voxelization module:
  voxel_features: [M, 5] float32 — mean features per voxel
  coordinates:    [M, 4] int32   — [batch, x, y, z]
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import openvino as ov

from .config import POINT_CLOUD_RANGE, SPARSE_SHAPE, VOXEL_SIZE

# ── Paths ──────────────────────────────────────────────────────────────────────
_EXT_DIR = Path(__file__).resolve().parent.parent / 'openvino_extensions' / 'voxelize'
_EXT_LIB = _EXT_DIR / 'build' / 'libopenvino_voxelize_extension.so'
_GPU_CFG = _EXT_DIR / 'voxelize_gpu.xml'

# ── Constants (must match GPU kernels and C++ ops) ─────────────────────────────
MAX_POINTS   = 70000
MAX_VOXELS   = 60000
NUM_FEATURES = 5

HAS_VOXELIZE_EXTENSION = _EXT_LIB.exists()


def _make_voxelize_ir() -> str:
    """Generate OpenVINO IR XML for the two-layer voxelization pipeline.

    Layers:
      1. VoxelizeScatter (custom) — GPU hash + atomic scatter → workspace
      2. VoxelizeMean    (custom) — workspace → mean features + coords

    Inputs:
      0: points      [70000, 5]  FP32  (padded LiDAR points)
      1: num_points   [1]        I32   (actual point count)

    Output:
      0: result       [60001, 9]  FP32  (row 0 = metadata, rest = features+coords)
    """
    ws_size = 1441793  # 1 + 131072*11

    return f"""\
<?xml version="1.0"?>
<net name="voxelize" version="11">
  <layers>
    <!-- ─── Parameters ─── -->
    <layer id="0" name="points" type="Parameter" version="opset1">
      <data shape="{MAX_POINTS},{NUM_FEATURES}" element_type="f32"/>
      <output><port id="0" precision="FP32" names="points">
        <dim>{MAX_POINTS}</dim><dim>{NUM_FEATURES}</dim>
      </port></output>
    </layer>
    <layer id="1" name="num_points" type="Parameter" version="opset1">
      <data shape="1" element_type="i32"/>
      <output><port id="0" precision="I32" names="num_points">
        <dim>1</dim>
      </port></output>
    </layer>

    <!-- ─── Layer 1: Hash scatter (int32 atomics) ─── -->
    <layer id="2" name="scatter" type="VoxelizeScatter" version="bevfusion">
      <input>
        <port id="0"><dim>{MAX_POINTS}</dim><dim>{NUM_FEATURES}</dim></port>
        <port id="1"><dim>1</dim></port>
      </input>
      <output><port id="2" precision="I32">
        <dim>{ws_size}</dim>
      </port></output>
    </layer>

    <!-- ─── Layer 2: Mean conversion (int32 → float32) ─── -->
    <layer id="3" name="mean" type="VoxelizeMean" version="bevfusion">
      <input><port id="0">
        <dim>{ws_size}</dim>
      </port></input>
      <output><port id="1" precision="FP32" names="result">
        <dim>{MAX_VOXELS + 1}</dim><dim>9</dim>
      </port></output>
    </layer>

    <!-- ─── Result ─── -->
    <layer id="4" name="output" type="Result" version="opset1">
      <input><port id="0">
        <dim>{MAX_VOXELS + 1}</dim><dim>9</dim>
      </port></input>
    </layer>
  </layers>
  <edges>
    <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
    <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
    <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
    <edge from-layer="3" from-port="1" to-layer="4" to-port="0"/>
  </edges>
</net>
"""


class VoxelizeExtension:
    """Compile-once wrapper around the Voxelization OpenVINO custom ops.

    Uses GPU backend with single work-group barrier approach:
    each kernel uses GWS=LWS=1024 so barrier(CLK_GLOBAL_MEM_FENCE) can
    synchronize a zeroing phase before the hash/compact phase, avoiding
    the stale-buffer issue with OV GPU memory reuse.
    """

    def __init__(self, core: ov.Core, device: str = 'GPU'):
        self._core = core
        self._device = device
        self._compiled = None
        self._request = None

    def export_model(self, output_dir: str | Path | None = None):
        """Export the Voxelize IR model (xml only, no weights) to disk."""
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent / 'openvino_model'
        else:
            output_dir = Path(output_dir)

        xml_path = output_dir / 'voxelize.xml'
        xml_str = _make_voxelize_ir()
        with open(xml_path, 'w') as f:
            f.write(xml_str)
        print(f"[voxelize_extension] Exported: {xml_path}")
        return str(xml_path)

    def _ensure_compiled(self):
        if self._compiled is not None:
            return

        # Try pre-exported model first
        model_dir = Path(__file__).resolve().parent.parent / 'openvino_model'
        pre_xml = model_dir / 'voxelize.xml'

        if pre_xml.exists():
            print(f"[voxelize_extension] Loading pre-exported model: {pre_xml.name}")
            model = self._core.read_model(str(pre_xml))
        else:
            xml_str = _make_voxelize_ir()
            with tempfile.NamedTemporaryFile(suffix='.xml', mode='w', delete=False) as f:
                f.write(xml_str)
                xml_path = f.name
            model = self._core.read_model(xml_path)

        gpu_config = {}
        if self._device.upper() == 'GPU':
            gpu_config['PERFORMANCE_HINT'] = 'LATENCY'

        self._compiled = self._core.compile_model(model, self._device, gpu_config)
        self._request = self._compiled.create_infer_request()
        print(f"[voxelize_extension] Compiled Voxelize model on {self._device}")

    def run(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Voxelize a LiDAR point cloud.

        Args:
            points: [N, 5] float32 — raw LiDAR points (x, y, z, intensity, time)

        Returns:
            voxel_features: [M, 5] float32 — mean features per voxel
            coordinates:    [M, 4] int32   — [batch, x, y, z]
        """
        self._ensure_compiled()

        n = points.shape[0]
        nf = min(points.shape[1], NUM_FEATURES)

        # Pad / truncate to MAX_POINTS
        if n > MAX_POINTS:
            points = points[:MAX_POINTS]
            n = MAX_POINTS

        padded = np.zeros((MAX_POINTS, NUM_FEATURES), dtype=np.float32)
        padded[:n, :nf] = points[:n, :nf]

        num_pts = np.array([n], dtype=np.int32)

        self._request.set_input_tensor(0, ov.Tensor(padded))
        self._request.set_input_tensor(1, ov.Tensor(num_pts))

        self._request.infer()

        result = self._request.get_output_tensor(0).data

        # Extract metadata and valid voxels
        num_voxels = int(result[0, 0])
        if num_voxels <= 0:
            return (np.zeros((0, NUM_FEATURES), dtype=np.float32),
                    np.zeros((0, 4), dtype=np.int32))

        num_voxels = min(num_voxels, MAX_VOXELS)
        valid = result[1:num_voxels + 1]

        voxel_features = valid[:, :NUM_FEATURES].copy().astype(np.float32)
        coordinates = valid[:, NUM_FEATURES:].copy().astype(np.int32)  # [batch, x, y, z]

        return voxel_features, coordinates
