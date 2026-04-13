"""
Sparse Encoder using OpenVINO custom extension.

Single monolithic GPU op that runs the full sparse encoder pipeline:
  - CPU neighbor-map construction (hash table, OpenMP)
  - GPU SubM conv (FP16), residual blocks, strided conv (atomic scatter)
  - Sparse-to-dense, permute, reshape → BEV [1, 256, 180, 180]

Replaces SparseEncoderFused (torch + OpenCL) with pure OV + OpenCL.
"""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import openvino as ov

# ── Paths ──────────────────────────────────────────────────────────────────────
_EXT_DIR = Path(__file__).resolve().parent.parent / 'openvino_extensions' / 'sparse_encoder'
_EXT_LIB = _EXT_DIR / 'build' / 'libopenvino_sparse_encoder_extension.so'

# ── Architecture constants ─────────────────────────────────────────────────────
MAX_N         = 60000
NUM_FEATURES  = 5
BEV_C, BEV_H, BEV_W = 256, 180, 180

# Layer definitions: (type, C_in, C_out, num_k)
#   type: 0=subm, 1=residual, 2=downsample, 3=conv_out
LAYER_DEFS = [
    (0,  5,  16,  27),  # conv_input
    (1,  16, 16,  27),  # res0_0_c1
    (1,  16, 16,  27),  # res0_0_c2
    (1,  16, 16,  27),  # res0_1_c1
    (1,  16, 16,  27),  # res0_1_c2
    (2,  16, 32,  27),  # ds0
    (1,  32, 32,  27),  # res1_0_c1
    (1,  32, 32,  27),  # res1_0_c2
    (1,  32, 32,  27),  # res1_1_c1
    (1,  32, 32,  27),  # res1_1_c2
    (2,  32, 64,  27),  # ds1
    (1,  64, 64,  27),  # res2_0_c1
    (1,  64, 64,  27),  # res2_0_c2
    (1,  64, 64,  27),  # res2_1_c1
    (1,  64, 64,  27),  # res2_1_c2
    (2,  64, 128, 27),  # ds2
    (1, 128, 128, 27),  # res3_0_c1
    (1, 128, 128, 27),  # res3_0_c2
    (1, 128, 128, 27),  # res3_1_c1
    (1, 128, 128, 27),  # res3_1_c2
    (3, 128, 128,  3),  # conv_out (1×1×3)
]

TOTAL_WEIGHTS = sum(d[2] * d[3] * d[1] for d in LAYER_DEFS)  # 2691696
TOTAL_SCALES  = sum(d[2] for d in LAYER_DEFS)                  # 1328
TOTAL_BIASES  = TOTAL_SCALES                                    # 1328
TOTAL_PARAMS  = TOTAL_WEIGHTS + TOTAL_SCALES + TOTAL_BIASES    # 2694352

HAS_SPARSE_ENCODER_EXTENSION = _EXT_LIB.exists()


def _load_weight(weights_dir: Path, name: str) -> Optional[np.ndarray]:
    path = weights_dir / f'{name}.npy'
    if path.exists():
        return np.load(path).astype(np.float32)
    return None


def _fuse_bn(weights_dir: Path, w_name: str, b_name: str,
             m_name: str, v_name: str, eps: float = 1e-3):
    w = _load_weight(weights_dir, w_name)
    b = _load_weight(weights_dir, b_name)
    m = _load_weight(weights_dir, m_name)
    v = _load_weight(weights_dir, v_name)
    if any(x is None for x in (w, b, m, v)):
        return None, None
    std = np.sqrt(v + eps)
    scale = w / std
    bias = b - m * scale
    return scale.astype(np.float32), bias.astype(np.float32)


def _reformat_weight(weight: np.ndarray) -> np.ndarray:
    """Reshape [k0, k1, k2, C_in, C_out] → flat [num_k * C_in * C_out]."""
    return weight.reshape(-1).astype(np.float32)


def pack_sparse_encoder_weights(weights_dir: str) -> np.ndarray:
    """Load all sparse encoder weights and pack into a single flat FP32 array.

    Layout: [all_weights | all_scales | all_biases]
    Order:  conv_input, res0_0_c1, res0_0_c2, res0_1_c1, res0_1_c2, ds0,
            res1_*, ds1, res2_*, ds2, res3_*, conv_out

    Returns: np.ndarray of shape [TOTAL_PARAMS] = [2694352], dtype=float32
    """
    wd = Path(weights_dir)
    export_style = (wd / 'conv_input_0_weight.npy').exists()

    all_w, all_s, all_b = [], [], []

    # ── conv_input ──
    if export_style:
        ci_w = _load_weight(wd, 'conv_input_0_weight')
        ci_s, ci_b = _fuse_bn(wd, 'conv_input_1_weight', 'conv_input_1_bias',
                               'conv_input_1_running_mean', 'conv_input_1_running_var')
    else:
        ci_w = _load_weight(wd, 'conv_input_weight')
        ci_s, ci_b = _fuse_bn(wd, 'bn_input_weight', 'bn_input_bias',
                               'bn_input_running_mean', 'bn_input_running_var')
    assert ci_w is not None, "Failed to load conv_input weight"
    all_w.append(_reformat_weight(ci_w))
    all_s.append(ci_s)
    all_b.append(ci_b)

    # ── Encoder layers 0-3 ──
    for layer_idx in range(4):
        file_li = layer_idx + 1 if export_style else layer_idx

        # 2 residual blocks
        for block_idx in [0, 1]:
            prefix = (f'encoder_layers_encoder_layer{file_li}_{block_idx}'
                      if export_style else f'encoder_layers_{file_li}_{block_idx}')

            c1w = _load_weight(wd, f'{prefix}_conv1_weight')
            c1s, c1b = _fuse_bn(wd, f'{prefix}_bn1_weight', f'{prefix}_bn1_bias',
                                f'{prefix}_bn1_running_mean', f'{prefix}_bn1_running_var')

            c2w = _load_weight(wd, f'{prefix}_conv2_weight')
            c2s, c2b = _fuse_bn(wd, f'{prefix}_bn2_weight', f'{prefix}_bn2_bias',
                                f'{prefix}_bn2_running_mean', f'{prefix}_bn2_running_var')

            assert c1w is not None, f"Missing weights for {prefix}_conv1"
            assert c2w is not None, f"Missing weights for {prefix}_conv2"

            all_w.append(_reformat_weight(c1w))
            all_s.append(c1s)
            all_b.append(c1b)

            all_w.append(_reformat_weight(c2w))
            all_s.append(c2s)
            all_b.append(c2b)

        # Downsample (layers 0-2 only)
        if layer_idx < 3:
            ds_prefix = (f'encoder_layers_encoder_layer{file_li}_2'
                         if export_style else f'encoder_layers_{file_li}_2')

            ds_w = _load_weight(wd, f'{ds_prefix}_0_weight')
            ds_s, ds_b = _fuse_bn(wd, f'{ds_prefix}_1_weight', f'{ds_prefix}_1_bias',
                                  f'{ds_prefix}_1_running_mean', f'{ds_prefix}_1_running_var')
            assert ds_w is not None, f"Missing weights for {ds_prefix}"

            all_w.append(_reformat_weight(ds_w))
            all_s.append(ds_s)
            all_b.append(ds_b)

    # ── conv_out ──
    if export_style:
        co_w = _load_weight(wd, 'conv_out_0_weight')
        co_s, co_b = _fuse_bn(wd, 'conv_out_1_weight', 'conv_out_1_bias',
                               'conv_out_1_running_mean', 'conv_out_1_running_var')
    else:
        co_w = _load_weight(wd, 'conv_out_weight')
        co_s, co_b = _fuse_bn(wd, 'bn_out_weight', 'bn_out_bias',
                               'bn_out_running_mean', 'bn_out_running_var')
    assert co_w is not None, "Failed to load conv_out weight"
    all_w.append(_reformat_weight(co_w))
    all_s.append(co_s)
    all_b.append(co_b)

    # ── Pack: [weights | scales | biases] ──
    packed = np.concatenate(all_w + all_s + all_b).astype(np.float32)
    assert packed.shape[0] == TOTAL_PARAMS, \
        f"Param count mismatch: got {packed.shape[0]}, expected {TOTAL_PARAMS}"

    print(f"[sparse_encoder_ext] Packed {len(all_w)} layers, "
          f"{TOTAL_PARAMS} params ({packed.nbytes / 1e6:.1f} MB)")
    return packed


def _make_sparse_encoder_xml(n_params: int) -> str:
    """Generate the IR XML string for the sparse encoder op."""
    return f"""\
<?xml version="1.0"?>
<net name="sparse_encoder" version="11">
  <layers>
    <!-- Parameters -->
    <layer id="0" name="features" type="Parameter" version="opset1">
      <data shape="{MAX_N},{NUM_FEATURES}" element_type="f32"/>
      <output><port id="0" precision="FP32" names="features">
        <dim>{MAX_N}</dim><dim>{NUM_FEATURES}</dim>
      </port></output>
    </layer>
    <layer id="1" name="coords" type="Parameter" version="opset1">
      <data shape="{MAX_N},4" element_type="i32"/>
      <output><port id="0" precision="I32" names="coords">
        <dim>{MAX_N}</dim><dim>4</dim>
      </port></output>
    </layer>
    <layer id="2" name="num_voxels" type="Parameter" version="opset1">
      <data shape="1" element_type="i32"/>
      <output><port id="0" precision="I32" names="num_voxels">
        <dim>1</dim>
      </port></output>
    </layer>

    <!-- Constant weights -->
    <layer id="3" name="params" type="Const" version="opset1">
      <data element_type="f32" shape="{n_params}" offset="0" size="{n_params * 4}"/>
      <output><port id="0" precision="FP32">
        <dim>{n_params}</dim>
      </port></output>
    </layer>

    <!-- Sparse encoder op -->
    <layer id="4" name="sparse_encoder" type="SparseEncoder" version="bevfusion">
      <input>
        <port id="0"><dim>{MAX_N}</dim><dim>{NUM_FEATURES}</dim></port>
        <port id="1"><dim>{MAX_N}</dim><dim>4</dim></port>
        <port id="2"><dim>1</dim></port>
        <port id="3"><dim>{n_params}</dim></port>
      </input>
      <output><port id="4" precision="FP32" names="bev_features">
        <dim>1</dim><dim>{BEV_C}</dim><dim>{BEV_H}</dim><dim>{BEV_W}</dim>
      </port></output>
    </layer>

    <!-- Result -->
    <layer id="5" name="output" type="Result" version="opset1">
      <input><port id="0">
        <dim>1</dim><dim>{BEV_C}</dim><dim>{BEV_H}</dim><dim>{BEV_W}</dim>
      </port></input>
    </layer>
  </layers>
  <edges>
    <edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
    <edge from-layer="1" from-port="0" to-layer="4" to-port="1"/>
    <edge from-layer="2" from-port="0" to-layer="4" to-port="2"/>
    <edge from-layer="3" from-port="0" to-layer="4" to-port="3"/>
    <edge from-layer="4" from-port="4" to-layer="5" to-port="0"/>
  </edges>
</net>
"""


def _make_sparse_encoder_ir(params: np.ndarray) -> Tuple[str, str]:
    """Generate IR XML and .bin for the single-op sparse encoder.

    The params array becomes an opset1::Constant embedded in the graph.
    Returns (xml_path, bin_path).
    """
    n_params = params.shape[0]

    # Write binary weights file
    bin_tmp = tempfile.NamedTemporaryFile(suffix='.bin', delete=False)
    bin_tmp.write(params.tobytes())
    bin_tmp.close()
    bin_path = bin_tmp.name

    xml = _make_sparse_encoder_xml(n_params)

    xml_tmp = tempfile.NamedTemporaryFile(suffix='.xml', mode='w', delete=False)
    xml_tmp.write(xml)
    xml_tmp.close()

    return xml_tmp.name, bin_path


class SparseEncoderExtension:
    """Compile-once wrapper for the Sparse Encoder OpenVINO custom op.

    The op runs entirely on CPU (evaluate callback) with internal OpenCL
    for GPU kernel dispatch — same approach as the fused torch extension.

    The IR model (xml+bin) can be pre-exported to the model directory,
    or generated on-the-fly from weights.
    """

    def __init__(self, core: ov.Core, weights_dir: str):
        self._core = core
        self._weights_dir = weights_dir
        self._compiled = None
        self._request = None
        self._packed = None

        # Check for pre-exported model files
        wd = Path(weights_dir)
        self._model_xml = wd.parent / 'sparse_encoder.xml'
        self._model_bin = wd.parent / 'sparse_encoder.bin'

    def export_model(self, output_dir: str | Path | None = None):
        """Export the sparse encoder model (xml + bin) to disk.

        Saves sparse_encoder.xml and sparse_encoder.bin alongside
        the other OpenVINO model files.
        """
        if output_dir is None:
            output_dir = Path(self._weights_dir).parent
        else:
            output_dir = Path(output_dir)

        packed = pack_sparse_encoder_weights(self._weights_dir)

        xml_path = output_dir / 'sparse_encoder.xml'
        bin_path = output_dir / 'sparse_encoder.bin'

        # Write binary weights
        with open(bin_path, 'wb') as f:
            f.write(packed.tobytes())

        # Write XML (referencing the .bin)
        xml_str = _make_sparse_encoder_xml(packed.shape[0])
        with open(xml_path, 'w') as f:
            f.write(xml_str)

        print(f"[sparse_encoder_ext] Exported: {xml_path} + {bin_path}")
        return str(xml_path), str(bin_path)

    def _ensure_compiled(self):
        if self._compiled is not None:
            return

        # Option 1: Load pre-exported model
        if self._model_xml.exists() and self._model_bin.exists():
            print(f"[sparse_encoder_ext] Loading pre-exported model: {self._model_xml.name}")
            model = self._core.read_model(str(self._model_xml), str(self._model_bin))
        else:
            # Option 2: Generate on-the-fly
            print("[sparse_encoder_ext] Generating model on-the-fly from weights…")
            self._packed = pack_sparse_encoder_weights(self._weights_dir)
            xml_path, bin_path = _make_sparse_encoder_ir(self._packed)
            model = self._core.read_model(xml_path, bin_path)

        # Compile on CPU (evaluate() runs internally, OpenCL dispatched inside)
        self._compiled = self._core.compile_model(model, 'CPU')
        self._request = self._compiled.create_infer_request()
        print(f"[sparse_encoder_ext] Compiled sparse encoder model on CPU")

    def run(self, voxel_features: np.ndarray,
            coordinates: np.ndarray) -> np.ndarray:
        """Run sparse encoder.

        Args:
            voxel_features: [N, 5] float32
            coordinates: [N, 4] int32 — [batch, x, y, z]

        Returns:
            bev_features: [1, 256, 180, 180] float32
        """
        self._ensure_compiled()

        N = voxel_features.shape[0]

        # Pad to MAX_N
        feat_pad = np.zeros((MAX_N, NUM_FEATURES), dtype=np.float32)
        coord_pad = np.zeros((MAX_N, 4), dtype=np.int32)

        n = min(N, MAX_N)
        nf = min(voxel_features.shape[1], NUM_FEATURES)
        feat_pad[:n, :nf] = voxel_features[:n, :nf]
        coord_pad[:n] = coordinates[:n]

        num_vox = np.array([n], dtype=np.int32)

        self._request.set_input_tensor(0, ov.Tensor(feat_pad))
        self._request.set_input_tensor(1, ov.Tensor(coord_pad))
        self._request.set_input_tensor(2, ov.Tensor(num_vox))

        self._request.infer()

        result = self._request.get_output_tensor(0).data
        return result.copy()

    def encode(self, voxel_features: np.ndarray,
               coordinates: np.ndarray,
               verbose: bool = False) -> np.ndarray:
        """Alias matching the SparseEncoder API used by the pipeline."""
        return self.run(voxel_features, coordinates)
