# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Export HGGD AnchorNet and LocalNet to OpenVINO IR for Intel GPU.

Loads PyTorch checkpoint, exports via ONNX, converts to OpenVINO IR (FP16),
and validates on GPU.

Usage:
    conda activate hggd_xpu
    cd path_to_openvino_hggd
    python export_models.py \
        --checkpoint-path /path/to/HGGD_realsense_checkpoint \
        --output-dir openvino_models
"""
import argparse
import logging
import sys
import os
import types
import functools
from pathlib import Path

# ── CUDA → CPU patches for model loading (checkpoint has CUDA tensors) ──
import torch
import torch.cuda


def _tensor_cuda_noop(self, *args, **kwargs):
    return self


torch.Tensor.cuda = _tensor_cuda_noop

_orig_to = torch.Tensor.to


def _patched_to(self, *args, **kwargs):
    new_args = list(args)
    for i, a in enumerate(new_args):
        if isinstance(a, str) and 'cuda' in a:
            new_args[i] = 'cpu'
        elif isinstance(a, torch.device) and a.type == 'cuda':
            new_args[i] = torch.device('cpu')
    if 'device' in kwargs:
        dev = kwargs['device']
        if isinstance(dev, str) and 'cuda' in dev:
            kwargs['device'] = 'cpu'
        elif isinstance(dev, torch.device) and getattr(dev, 'type', '') == 'cuda':
            kwargs['device'] = torch.device('cpu')
    return _orig_to(self, *tuple(new_args), **kwargs)


torch.Tensor.to = _patched_to


def _patch_device_arg(orig_fn):
    @functools.wraps(orig_fn)
    def wrapper(*args, **kwargs):
        if 'device' in kwargs:
            dev = kwargs['device']
            if isinstance(dev, str) and 'cuda' in dev:
                kwargs['device'] = 'cpu'
            elif isinstance(dev, torch.device) and getattr(dev, 'type', '') == 'cuda':
                kwargs['device'] = torch.device('cpu')
        return orig_fn(*args, **kwargs)
    return wrapper


for _fn_name in ['zeros', 'ones', 'empty', 'randn', 'randint', 'full',
                  'tensor', 'linspace', 'arange', 'zeros_like', 'ones_like',
                  'empty_like', 'randn_like', 'rand']:
    if hasattr(torch, _fn_name):
        setattr(torch, _fn_name, _patch_device_arg(getattr(torch, _fn_name)))

torch.cuda.synchronize = lambda *a, **k: None
torch.cuda.is_available = lambda: False
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.Tensor.is_cuda = property(lambda self: False)

# Stub cupoch (imported by some HGGD modules)
_cs = types.ModuleType("cupoch")
_cs.geometry = types.ModuleType("cupoch.geometry")
_cs.visualization = types.ModuleType("cupoch.visualization")
_cs.visualization.draw_geometries = lambda *a, **k: None
sys.modules["cupoch"] = _cs
sys.modules["cupoch.geometry"] = _cs.geometry
sys.modules["cupoch.visualization"] = _cs.visualization

# ── End patches ──────────────────────────────────────────────────────

import numpy as np
import openvino as ov

# Add this package to path so model imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Install OV shim before importing models (localgraspnet imports pytorch3d at module level)
from ov_gpu_extensions.ov_shim_gpu import install_ov_shim
install_ov_shim(verbose=False)

from models.anchornet import AnchorGraspNet
from models.localgraspnet import PointMultiGraspNet

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)


class AnchorNetWrapper(torch.nn.Module):
    """Flattens AnchorGraspNet's nested tuple output for ONNX export."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        (loc_map, cls_mask, theta_offset, depth_offset, width_offset), features = self.model(x)
        return loc_map, cls_mask, theta_offset, depth_offset, width_offset, features


class LocalNetWrapper(torch.nn.Module):
    """Drops intermediate features output for ONNX export."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, points: torch.Tensor, info: torch.Tensor):
        _features, pred, offset = self.model(points, info)
        return pred, offset


def export_anchornet(
    model: torch.nn.Module,
    output_dir: Path,
    ov_device: str,
) -> Path:
    """Export AnchorGraspNet: PyTorch → ONNX → OpenVINO IR (FP16), validate on GPU."""
    log.info("Exporting AnchorGraspNet...")
    wrapper = AnchorNetWrapper(model)
    wrapper.eval()

    dummy_input = torch.randn(1, 4, 640, 360)
    onnx_path = output_dir / "anchornet.onnx"

    with torch.no_grad():
        pt_outputs = wrapper(dummy_input)
        log.info(f"  PyTorch output shapes: {[o.shape for o in pt_outputs]}")

    torch.onnx.export(
        wrapper, dummy_input, str(onnx_path),
        opset_version=11,
        input_names=["rgbd_input"],
        output_names=["loc_map", "cls_mask", "theta_offset",
                       "depth_offset", "width_offset", "features"],
    )
    log.info(f"  ONNX exported: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    ov_model = ov.convert_model(str(onnx_path))
    xml_path = output_dir / "anchornet_fp16.xml"
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=True)
    log.info(f"  OpenVINO IR saved: {xml_path}")

    # Validate on GPU
    core = ov.Core()
    compiled = core.compile_model(str(xml_path), ov_device)
    ov_result = compiled(dummy_input.numpy())
    for i, (name, pt_out) in enumerate(zip(
        ["loc_map", "cls_mask", "theta_offset", "depth_offset", "width_offset", "features"],
        pt_outputs,
    )):
        max_diff = np.abs(pt_out.numpy() - ov_result[i]).max()
        log.info(f"  {name}: shape={pt_out.shape}, max_diff(PT vs OV@{ov_device})={max_diff:.6f}")

    return xml_path


def export_localnet(
    model: torch.nn.Module,
    output_dir: Path,
    ov_device: str,
) -> Path:
    """Export PointMultiGraspNet: PyTorch → ONNX → OpenVINO IR (FP16), validate on GPU."""
    log.info("Exporting PointMultiGraspNet...")
    wrapper = LocalNetWrapper(model)
    wrapper.eval()

    n_centers = 48
    dummy_points = torch.randn(n_centers, 512, 35)
    dummy_info = torch.randn(n_centers, 3)
    onnx_path = output_dir / "localnet.onnx"

    with torch.no_grad():
        pt_outputs = wrapper(dummy_points, dummy_info)
        log.info(f"  PyTorch output shapes: pred={pt_outputs[0].shape}, offset={pt_outputs[1].shape}")

    torch.onnx.export(
        wrapper, (dummy_points, dummy_info), str(onnx_path),
        opset_version=11,
        input_names=["points", "info"],
        output_names=["pred", "offset"],
        dynamic_axes={
            "points": {0: "num_centers"},
            "info": {0: "num_centers"},
            "pred": {0: "num_centers"},
            "offset": {0: "num_centers"},
        },
    )
    log.info(f"  ONNX exported: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    ov_model = ov.convert_model(str(onnx_path))
    xml_path = output_dir / "localnet_fp16.xml"
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=True)
    log.info(f"  OpenVINO IR saved: {xml_path}")

    # Validate on GPU
    core = ov.Core()
    compiled = core.compile_model(str(xml_path), ov_device)
    ov_result = compiled({"points": dummy_points.numpy(), "info": dummy_info.numpy()})
    for name, pt_out in zip(["pred", "offset"], pt_outputs):
        max_diff = np.abs(pt_out.numpy() - ov_result[name]).max()
        log.info(f"  {name}: shape={pt_out.shape}, max_diff(PT vs OV@{ov_device})={max_diff:.6f}")

    return xml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HGGD models to OpenVINO IR (GPU)")
    parser.add_argument('--checkpoint-path', required=True,
                        help='Path to HGGD_realsense_checkpoint')
    parser.add_argument('--output-dir', default='openvino_models',
                        help='Directory for exported OV models (default: openvino_models)')
    parser.add_argument('--ov-device', default='GPU',
                        help='OpenVINO device for validation (default: GPU)')
    parser.add_argument('--anchor-k', type=int, default=6)
    parser.add_argument('--anchor-num', type=int, default=7)
    parser.add_argument('--ratio', type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
    log.info(f"Checkpoint loaded: {args.checkpoint_path}")

    # Build and load AnchorNet
    anchornet = AnchorGraspNet(in_dim=4, ratio=args.ratio, anchor_k=args.anchor_k)
    anchornet.load_state_dict(checkpoint['anchor'])
    anchornet.eval()
    log.info(f"AnchorGraspNet params: {sum(p.numel() for p in anchornet.parameters()):,}")

    # Build and load LocalNet
    localnet = PointMultiGraspNet(info_size=3, k_cls=args.anchor_num ** 2)
    localnet.load_state_dict(checkpoint['local'])
    localnet.eval()
    log.info(f"PointMultiGraspNet params: {sum(p.numel() for p in localnet.parameters()):,}")

    # Export both
    anchor_xml = export_anchornet(anchornet, output_dir, args.ov_device)
    local_xml = export_localnet(localnet, output_dir, args.ov_device)

    log.info("")
    log.info("=== Export Summary ===")
    log.info(f"AnchorNet IR:  {anchor_xml}")
    log.info(f"LocalNet IR:   {local_xml}")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            log.info(f"  {f.name}: {f.stat().st_size / 1e6:.2f} MB")


if __name__ == '__main__':
    main()
