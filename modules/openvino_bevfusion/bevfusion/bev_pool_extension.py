"""
BEV pool using OpenVINO custom extension (replaces standalone OpenCL kernel).

Creates a tiny OpenVINO IR model containing a single BEVPool custom op,
compiles it to GPU once, and runs inference for each frame — keeping the
data on the GPU and avoiding the torch/pybind11/OpenCL host-side overhead
of the standalone kernel.

The GPU kernel uses the same optimizations as our fused OpenCL BEV pool:
  - per-point parallelism (work-groups over N*D*H*W, 16 WIs on C)
  - int32 atomic_add with 2^13 fixed-point scaling (no CAS loop)
  - geometry-based grid lookup (no pre-sorting / interval tables)
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import openvino as ov

from .config import C, D, XBOUND, YBOUND, ZBOUND

# ── Paths ──────────────────────────────────────────────────────────────────────
_EXT_DIR = Path(__file__).resolve().parent.parent / 'openvino_extensions' / 'bev_pool'
_EXT_LIB = _EXT_DIR / 'build' / 'libopenvino_bevpool_extension.so'
_GPU_CFG = _EXT_DIR / 'bev_pool_gpu.xml'

# ── Grid constants ─────────────────────────────────────────────────────────────
NX = int((XBOUND[1] - XBOUND[0]) / XBOUND[2])  # 360
NY = int((YBOUND[1] - YBOUND[0]) / YBOUND[2])  # 360
NZ = int((ZBOUND[1] - ZBOUND[0]) / ZBOUND[2])  # 1

HAS_BEVPOOL_EXTENSION = _EXT_LIB.exists()


def _make_bevpool_ir() -> str:
    """Generate OpenVINO IR XML for the three-layer BEV pool pipeline.

    Three-layer design matching our fast OpenCL kernel:
      1. Softmax (opset8 built-in) — depth softmax over axis=1
      2. BEVPoolScatter (custom) — int32 native atomic scatter
      3. BEVPoolConvert (custom) — int32 fixed-point → float32

    Inputs:
      0: depth_logits  [6, 118, 32, 88]  FP32 (pre-softmax)
      1: context_feats [6,  80, 32, 88]  FP32
      2: geom          [1993728, 3]       FP32

    Output:
      0: bev           [1, 80, 360, 360]  FP32
    """
    num_cams = 6
    depth_bins = D       # 118
    channels = C         # 80
    feat_h = 32
    feat_w = 88
    total_pts = num_cams * depth_bins * feat_h * feat_w  # 1_993_728
    out_c = channels * NZ

    return f"""\
<?xml version="1.0"?>
<net name="bevpool" version="11">
  <layers>
    <!-- ─── Parameters ─── -->
    <layer id="0" name="depth_logits" type="Parameter" version="opset1">
      <data shape="{num_cams},{depth_bins},{feat_h},{feat_w}" element_type="f32"/>
      <output><port id="0" precision="FP32" names="depth_logits">
        <dim>{num_cams}</dim><dim>{depth_bins}</dim><dim>{feat_h}</dim><dim>{feat_w}</dim>
      </port></output>
    </layer>
    <layer id="1" name="context_feats" type="Parameter" version="opset1">
      <data shape="{num_cams},{channels},{feat_h},{feat_w}" element_type="f32"/>
      <output><port id="0" precision="FP32" names="context_feats">
        <dim>{num_cams}</dim><dim>{channels}</dim><dim>{feat_h}</dim><dim>{feat_w}</dim>
      </port></output>
    </layer>
    <layer id="2" name="geom" type="Parameter" version="opset1">
      <data shape="{total_pts},3" element_type="f32"/>
      <output><port id="0" precision="FP32" names="geom">
        <dim>{total_pts}</dim><dim>3</dim>
      </port></output>
    </layer>

    <!-- ─── Layer 1: Built-in Softmax on depth axis ─── -->
    <layer id="3" name="depth_softmax" type="Softmax" version="opset8">
      <data axis="1"/>
      <input><port id="0">
        <dim>{num_cams}</dim><dim>{depth_bins}</dim><dim>{feat_h}</dim><dim>{feat_w}</dim>
      </port></input>
      <output><port id="1" precision="FP32">
        <dim>{num_cams}</dim><dim>{depth_bins}</dim><dim>{feat_h}</dim><dim>{feat_w}</dim>
      </port></output>
    </layer>

    <!-- ─── Layer 2: Int32 atomic scatter ─── -->
    <layer id="4" name="bev_scatter" type="BEVPoolScatter" version="bevfusion"
           nx="{NX}" ny="{NY}" nz="{NZ}"
           num_cams="{num_cams}" depth_bins="{depth_bins}"
           channels="{channels}" feat_h="{feat_h}" feat_w="{feat_w}"
           x_min="{XBOUND[0]}" y_min="{YBOUND[0]}" z_min="{ZBOUND[0]}"
           x_step="{XBOUND[2]}" y_step="{YBOUND[2]}" z_step="{ZBOUND[2]}">
      <input>
        <port id="0"><dim>{num_cams}</dim><dim>{depth_bins}</dim><dim>{feat_h}</dim><dim>{feat_w}</dim></port>
        <port id="1"><dim>{num_cams}</dim><dim>{channels}</dim><dim>{feat_h}</dim><dim>{feat_w}</dim></port>
        <port id="2"><dim>{total_pts}</dim><dim>3</dim></port>
      </input>
      <output><port id="3" precision="I32">
        <dim>1</dim><dim>{out_c}</dim><dim>{NX}</dim><dim>{NY}</dim>
      </port></output>
    </layer>

    <!-- ─── Layer 3: Int32 → Float32 conversion ─── -->
    <layer id="5" name="bev_convert" type="BEVPoolConvert" version="bevfusion"
           scale="8192">
      <input><port id="0">
        <dim>1</dim><dim>{out_c}</dim><dim>{NX}</dim><dim>{NY}</dim>
      </port></input>
      <output><port id="1" precision="FP32" names="bev_features">
        <dim>1</dim><dim>{out_c}</dim><dim>{NX}</dim><dim>{NY}</dim>
      </port></output>
    </layer>

    <!-- ─── Result ─── -->
    <layer id="6" name="bev_output" type="Result" version="opset1">
      <input><port id="0">
        <dim>1</dim><dim>{out_c}</dim><dim>{NX}</dim><dim>{NY}</dim>
      </port></input>
    </layer>
  </layers>
  <edges>
    <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    <edge from-layer="3" from-port="1" to-layer="4" to-port="0"/>
    <edge from-layer="1" from-port="0" to-layer="4" to-port="1"/>
    <edge from-layer="2" from-port="0" to-layer="4" to-port="2"/>
    <edge from-layer="4" from-port="3" to-layer="5" to-port="0"/>
    <edge from-layer="5" from-port="1" to-layer="6" to-port="0"/>
  </edges>
</net>
"""


class BEVPoolExtension:
    """Compile-once wrapper around the BEVPool OpenVINO custom op."""

    def __init__(self, core: ov.Core, device: str = 'GPU'):
        self._core = core
        self._device = device
        self._compiled = None
        self._request = None
        self._total_pts = 6 * D * 32 * 88  # 1_993_728

    def export_model(self, output_dir: str | Path | None = None):
        """Export the BEV pool IR model (xml only, no weights) to disk."""
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent / 'openvino_model'
        else:
            output_dir = Path(output_dir)

        xml_path = output_dir / 'bev_pool.xml'
        xml_str = _make_bevpool_ir()
        with open(xml_path, 'w') as f:
            f.write(xml_str)
        print(f"[bev_pool_extension] Exported: {xml_path}")
        return str(xml_path)

    def _ensure_compiled(self):
        if self._compiled is not None:
            return

        # Try pre-exported model first
        model_dir = Path(__file__).resolve().parent.parent / 'openvino_model'
        pre_xml = model_dir / 'bev_pool.xml'

        if pre_xml.exists():
            print(f"[bev_pool_extension] Loading pre-exported model: {pre_xml.name}")
            model = self._core.read_model(str(pre_xml))
        else:
            xml_str = _make_bevpool_ir()
            with tempfile.NamedTemporaryFile(suffix='.xml', mode='w', delete=False) as f:
                f.write(xml_str)
                xml_path = f.name
            model = self._core.read_model(xml_path)

        gpu_config = {}
        if self._device.upper() == 'GPU':
            gpu_config['PERFORMANCE_HINT'] = 'LATENCY'

        self._compiled = self._core.compile_model(model, self._device, gpu_config)
        self._request = self._compiled.create_infer_request()
        print(f"[bev_pool_extension] Compiled BEVPool model on {self._device}")

    def run(self,
            depth_logits: np.ndarray,
            context_feats: np.ndarray,
            geom: np.ndarray) -> np.ndarray:
        """Run BEV pool.

        Args:
            depth_logits:  [6, D, fH, fW]  float32 (pre-softmax logits)
            context_feats: [6, C, fH, fW]  float32
            geom:          [6, D, fH, fW, 3] float32  (world coords)

        Returns:
            bev: [1, C, NX, NY] float32
        """
        self._ensure_compiled()

        # Flatten geom to [N*D*H*W, 3]
        geom_flat = np.ascontiguousarray(geom.reshape(-1, 3), dtype=np.float32)
        dl = np.ascontiguousarray(depth_logits, dtype=np.float32)
        cf = np.ascontiguousarray(context_feats, dtype=np.float32)

        self._request.set_input_tensor(0, ov.Tensor(dl))
        self._request.set_input_tensor(1, ov.Tensor(cf))
        self._request.set_input_tensor(2, ov.Tensor(geom_flat))

        self._request.infer()

        return self._request.get_output_tensor().data.copy()


# ── Module-level convenience function ──────────────────────────────────────────
_singleton: BEVPoolExtension | None = None


def bev_pool_extension(
    depth_logits: np.ndarray,
    context_feats: np.ndarray,
    geom: np.ndarray,
    *,
    core: ov.Core | None = None,
    device: str = 'GPU',
) -> np.ndarray:
    """BEV pool via OpenVINO extension (module-level helper).

    Falls back to the fused OpenCL kernel if the extension is unavailable.
    """
    global _singleton

    if not HAS_BEVPOOL_EXTENSION:
        from .bev_pool import bev_pool_fused
        return bev_pool_fused(depth_logits, context_feats, geom)

    if _singleton is None:
        if core is None:
            core = ov.Core()
            # Standalone usage: load extension + GPU config ourselves
            if _EXT_LIB.exists():
                core.add_extension(str(_EXT_LIB))
            if _GPU_CFG.exists() and device.upper() == 'GPU':
                core.set_property('GPU', {'CONFIG_FILE': str(_GPU_CFG)})
        _singleton = BEVPoolExtension(core, device)

    return _singleton.run(depth_logits, context_feats, geom)
