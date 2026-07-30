#!/usr/bin/env python
#
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""
Export CDPN PyTorch model directly to OpenVINO IR (.xml/.bin).

Uses ov.convert_model() which traces the PyTorch model via torch.export
internally and converts to OV IR in one pass.

Produces:
  - cdpn_stage3.xml/.bin        (FP32 weights)      - NN body only
  - cdpn_stage3_extnn.xml/.bin  (with --extnn flag) - Extended-NN model
                                includes custom ops without PnP
  - cdpn_stage3_e2e.xml/.bin    (with --e2e flag)   - End-to-End model
                                includes PnP custom ops

Usage (inside cdpn_intel container):
  cd /workspace/tools

  # NN body only:
  python ../ov_export.py \
      --cfg exps_cfg/config_rot_trans.yaml \
      --load_model /workspace/checkpoints/stage3.checkpoint \
      --output_dir /workspace/checkpoints

  # Extended-NN (with pre/post processing ops from ov_plugins/):
  python ../ov_export.py \
      --cfg exps_cfg/config_rot_trans.yaml \
      --load_model /workspace/checkpoints/stage3.checkpoint \
      --output_dir /workspace/checkpoints \
      --extnn

  # End-to-End (with all custom ops including PnP):
  python ../ov_export.py \
      --cfg exps_cfg/config_rot_trans.yaml \
      --load_model /workspace/checkpoints/stage3.checkpoint \
      --output_dir /workspace/checkpoints \
      --e2e \
      --extension ov_plugins/build/cdpn_extension.so

Verify:
  python ../ov_export.py \
      --cfg exps_cfg/config_rot_trans.yaml \
      --load_model /workspace/checkpoints/stage3.checkpoint \
      --output_dir /workspace/checkpoints \
      --verify
"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys

import numpy as np
import torch

import openvino as ov
from openvino import opset13 as opset
from openvino import Type as OVType
from openvino.op import Result as OVResult



def build_cdpn_model(cfg_path, checkpoint_path):
    """Build and load the CDPN model from config + checkpoint."""
    # Ensure lib/ is on the path (works from repo root or tools/)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    tools_dir = os.path.join(repo_root, 'tools')
    lib_dir = os.path.join(repo_root, 'lib')
    for d in [tools_dir, lib_dir]:
        if d not in sys.path:
            sys.path.insert(0, d)

    # Resolve cfg_path: try CWD first, then tools/ dir
    if not os.path.isabs(cfg_path):
        if os.path.isfile(cfg_path):
            cfg_path = os.path.abspath(cfg_path)
        else:
            cfg_path = os.path.join(tools_dir, cfg_path)

    # config.parse() reads sys.argv
    sys.argv = ['ov_export.py',
                '--cfg', cfg_path,
                '--load_model', checkpoint_path]

    from config import config
    from model import build_model

    cfg = config().parse()
    cfg.pytorch.load_model = checkpoint_path
    model, _ = build_model(cfg)
    model.eval()

    return model, cfg


def _slice_last_dim(opset, tensor, start_idx, end_idx):
    axis_3 = opset.constant(np.array([3], dtype=np.int64))
    one_step = opset.constant(np.array([1], dtype=np.int64))

    return opset.slice(tensor,
                       opset.constant(np.array([start_idx], dtype=np.int64)),
                       opset.constant(np.array([end_idx], dtype=np.int64)),
                       one_step, axis_3)


def _build_coord_denorm_subgraph(opset, coord_maps, obj_extents):
    split_axis = opset.constant(np.int64(1))
    split_lengths = opset.constant(np.array([3, 1], dtype=np.int64))
    coord_split = opset.variadic_split(coord_maps, split_axis, split_lengths)
    raw_xyz = coord_split.output(0)
    raw_conf = coord_split.output(1)

    ext_reshaped = opset.reshape(
        obj_extents,
        opset.constant(np.array([0, 3, 1, 1], dtype=np.int64)), True)
    denorm_coords = opset.multiply(raw_xyz, ext_reshaped)

    reduce_axes = opset.constant(np.array([2, 3], dtype=np.int64))
    conf_min = opset.reduce_min(raw_conf, reduce_axes, keep_dims=True)
    conf_max = opset.reduce_max(raw_conf, reduce_axes, keep_dims=True)
    conf_range = opset.subtract(conf_max, conf_min)
    conf_range_safe = opset.maximum(conf_range, opset.constant(np.float32(1e-8)))
    confidence = opset.divide(opset.subtract(raw_conf, conf_min), conf_range_safe)

    return denorm_coords, confidence


def _build_trans_decode_subgraph(opset, raw_trans_4d, bbox_wh, crop_meta,
                                 cam_K, out_res=64.0):
    ratio_dcx = _slice_last_dim(opset, raw_trans_4d, 0, 1)
    ratio_dcy = _slice_last_dim(opset, raw_trans_4d, 1, 2)
    ratio_depth = _slice_last_dim(opset, raw_trans_4d, 2, 3)

    bw = _slice_last_dim(opset, bbox_wh, 0, 1)
    bh = _slice_last_dim(opset, bbox_wh, 1, 2)

    c_w = _slice_last_dim(opset, crop_meta, 0, 1)
    c_h = _slice_last_dim(opset, crop_meta, 1, 2)
    s_val = _slice_last_dim(opset, crop_meta, 2, 3)

    fx = _slice_last_dim(opset, cam_K, 0, 1)
    fy = _slice_last_dim(opset, cam_K, 1, 2)
    cx = _slice_last_dim(opset, cam_K, 2, 3)
    cy = _slice_last_dim(opset, cam_K, 3, 4)

    out_res_const = opset.constant(np.float32(out_res))
    pred_depth = opset.multiply(ratio_depth, opset.divide(out_res_const, s_val))
    pred_cx = opset.add(opset.multiply(ratio_dcx, bw), c_w)
    pred_cy = opset.add(opset.multiply(ratio_dcy, bh), c_h)

    tx = opset.divide(opset.multiply(opset.subtract(pred_cx, cx), pred_depth), fx)
    ty = opset.divide(opset.multiply(opset.subtract(pred_cy, cy), pred_depth), fy)

    return opset.concat([tx, ty, pred_depth], axis=3)


def _make_CdpnPreprocess(ov, OVType):
    """Factory: return the CdpnPreprocess graph-topology wrapper class.

    Defined as a factory because ov/OVType are lazy-imported inside each
    export function.  The returned class is structurally identical whether
    used in EXTNN or E2E graphs.

    The class defines graph topology ONLY (zero Python computation).
    At inference time:
      CPU → cdpn_extension.so provides evaluate() in C++
      GPU → cdpn_custom_gpu_kernels.xml + .cl kernels (SimpleGPU)
    """
    class CdpnPreprocess(ov.Op):
        """image[N,H,W,3] u8 + bbox[N,1,1,4] → tensor[N,3,inp_res,inp_res] + crop_meta[N,1,1,5]."""

        def __init__(self, image, bbox,
                     inp_res=256, pad_ratio=1.5, im_w=640, im_h=480):
            super().__init__(self)
            self._inp_res = int(inp_res)
            self._pad_ratio = float(pad_ratio)
            self._im_w = int(im_w)
            self._im_h = int(im_h)
            self.set_arguments([image, bbox])
            self.constructor_validate_and_infer_types()

        @classmethod
        def get_type_info_static(cls):
            return ov.DiscreteTypeInfo("CdpnPreprocess", "extension")

        def validate_and_infer_types(self):
            batch = self.get_input_partial_shape(0)[0]
            self.set_output_type(
                0, OVType.f32,
                ov.PartialShape([batch, 3, self._inp_res, self._inp_res]))
            # Padded to 4D BFYX
            self.set_output_type(1, OVType.f32, ov.PartialShape([batch, 1, 1, 5]))

        def visit_attributes(self, visitor):
            visitor.on_attributes({
                "inp_res": self._inp_res,
                "pad_ratio": self._pad_ratio,
                "im_w": self._im_w,
                "im_h": self._im_h,
            })
            return True

        def has_evaluate(self):
            return False

        def clone_with_new_inputs(self, new_args):
            return CdpnPreprocess(
                new_args[0], new_args[1],
                self._inp_res, self._pad_ratio, self._im_w, self._im_h)

    return CdpnPreprocess


def _build_composite_input_params(opset, ov, OVType):
    """Create the 5 shared 4D-BFYX input parameters used by EXTNN and E2E graphs."""
    param_image = opset.parameter(
        ov.PartialShape([-1, -1, -1, 3]), OVType.u8, name='image')
    param_bbox = opset.parameter(
        ov.PartialShape([-1, 1, 1, 4]), OVType.f32, name='bbox')
    param_extents = opset.parameter(
        ov.PartialShape([-1, 1, 1, 3]), OVType.f32, name='obj_extents')
    param_bbox_wh = opset.parameter(
        ov.PartialShape([-1, 1, 1, 2]), OVType.f32, name='bbox_wh')
    param_cam_K = opset.parameter(
        ov.PartialShape([-1, 1, 1, 4]), OVType.f32, name='cam_K')

    return param_image, param_bbox, param_extents, param_bbox_wh, param_cam_K


def _rewire_nn_body(opset, nn_model, tensor_out):
    """Rewire nn_model's input to tensor_out; return (coord_maps_node, raw_trans_4d).

    Replaces the NN Parameter input with the output of a preceding custom op
    (e.g. CdpnPreprocess), then extracts the two NN Result outputs and
    reshapes raw_trans from [N,3] → [N,1,1,3] (4D BFYX) ready for downstream ops.
    special_zero=True on reshape: dim 0 value '0' copies batch size from input.
    """
    nn_params = nn_model.get_parameters()
    nn_results = nn_model.get_results()
    nn_input_param = nn_params[0]
    for target_input in nn_input_param.output(0).get_target_inputs():
        target_input.replace_source_output(tensor_out)
    coord_maps_node = nn_results[0].input(0).get_source_output()  # [N,4,64,64]
    raw_trans_node = nn_results[1].input(0).get_source_output()   # [N,3]
    raw_trans_4d = opset.reshape(
        raw_trans_node,
        opset.constant(np.array([0, 1, 1, 3], dtype=np.int64)), True)

    return coord_maps_node, raw_trans_4d


def _make_verify_inputs():
    """Return a dict of canonical single-sample test inputs for composite models."""
    np.random.seed(42)

    return {
        'image':       np.random.randint(0, 255, (1, 480, 640, 3), dtype=np.uint8),
        'bbox':        np.array([[[[100.0, 80.0, 120.0, 150.0]]]], dtype=np.float32),
        'obj_extents': np.array([[[[0.05, 0.04, 0.06]]]], dtype=np.float32),
        'bbox_wh':     np.array([[[[120.0, 150.0]]]], dtype=np.float32),
        'cam_K':       np.array([[[[572.41, 573.57, 325.26, 242.05]]]], dtype=np.float32),
    }


def _save_ir(ov, model, output_dir, basename, tag):
    """Save OV model as FP32 IR; print file size; return (xml_path, bin_path, size_mb)."""
    xml_path = os.path.join(output_dir, basename + '.xml')
    bin_path = os.path.join(output_dir, basename + '.bin')

    ov.save_model(model, xml_path, compress_to_fp16=False)

    size_mb = os.path.getsize(bin_path) / 1e6
    print('[{}] IR: {} ({:.1f} MB)'.format(tag, xml_path, size_mb))

    return xml_path, bin_path, size_mb


def _print_model_io(model, tag):
    """Print a model's input and output names, shapes, and element types."""
    print('[{}]   Inputs:'.format(tag))
    for i, inp in enumerate(model.inputs):
        print('[{}]     {}: names={}, shape={}, type={}'.format(
            tag, i, inp.get_names(), inp.get_partial_shape(),
            inp.get_element_type()))
    print('[{}]   Outputs:'.format(tag))
    for i, out in enumerate(model.outputs):
        print('[{}]     {}: names={}, shape={}'.format(
            tag, i, out.get_names(), out.get_partial_shape()))


def export_to_ov_ir(model, output_dir, basename='cdpn_stage3',
                    verify=False, batch_size=1):
    """
    Export PyTorch model to OpenVINO IR (FP32).

    Parameters
    ----------
    model : torch.nn.Module
        The CDPN model in eval mode.
    output_dir : str
        Directory to write .xml/.bin files.
    basename : str
        Base filename (without extension).
    verify : bool
        If True, run numerical verification against PyTorch.
    batch_size : int
        Example batch size for tracing (dynamic batch is enabled).

    Returns
    -------
    dict with paths to exported files and verification results.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Trace with example input
    dummy = torch.randn(batch_size, 3, 256, 256)
    print('[ov_export] OpenVINO version: {}'.format(ov.__version__))
    print('[ov_export] Tracing model with input shape {} ...'.format(
        list(dummy.shape)))

    ov_model = ov.convert_model(
        model,
        example_input=dummy,
        input=[ov.PartialShape([-1, 3, 256, 256])]
    )

    # Name the outputs for easier downstream usage
    ov_model.outputs[0].tensor.set_names({'coord_maps'})
    ov_model.outputs[1].tensor.set_names({'translation'})

    # Confirm output shapes
    _print_model_io(ov_model, 'ov_export:nn')

    # Save FP32
    xml_path, bin_path, bin_size = _save_ir(
        ov, ov_model, output_dir, basename, 'ov_export:nn')
    results = {'xml': xml_path, 'bin': bin_path, 'size_mb': bin_size, 'ov_model': ov_model}

    # Verification
    if verify:
        print()
        print('[ov_export] === Numerical Verification ===')
        core = ov.Core()

        torch.manual_seed(42)
        test_input = torch.randn(1, 3, 256, 256)

        # PyTorch reference
        with torch.no_grad():
            pt_rot, pt_trans = model(test_input)
        pt_rot_np = pt_rot.numpy()
        pt_trans_np = pt_trans.numpy()

        compiled = core.compile_model(xml_path, 'CPU')
        ov_out = compiled(test_input.numpy())

        diff_rot = np.abs(pt_rot_np - ov_out[0])
        diff_trans = np.abs(pt_trans_np - ov_out[1])

        print('[ov_export] rot  - max: {:.2e}, mean: {:.2e}, '
              'P50: {:.2e}, P99: {:.2e}'.format(
                  np.max(diff_rot), np.mean(diff_rot),
                  np.percentile(diff_rot, 50),
                  np.percentile(diff_rot, 99)))
        print('[ov_export] trans - max: {:.2e}, mean: {:.2e}'.format(
            np.max(diff_trans), np.mean(diff_trans)))

        results['max_diff_rot'] = float(np.max(diff_rot))
        results['mean_diff_rot'] = float(np.mean(diff_rot))

    print()
    print('[ov_export] Done.')
    return results


def export_extnn_model(output_dir, basename='cdpn_stage3_extnn',
                      extension_path=None, verify=False, nn_ov_model=None):
    """
    Export the EXTNN CDPN model - single XML/BIN with preprocessing,
    NN body, and postprocessing (everything except PnP).

    Graph (N = dynamic batch):
      image[N,H,W,3] u8 + bbox[N,1,1,4] -> CdpnPreprocess
        -> tensor[N,3,256,256] + crop_meta[N,1,1,5]
      tensor -> NN body -> coord_maps[N,4,64,64] + raw_trans[N,3]
      coord_maps + obj_extents[N,1,1,3] -> CdpnCoordDenorm -> combined[N,4,64,64]
      combined -> VariadicSplit -> denorm_coords[N,3,64,64] + confidence[N,1,64,64]
      raw_trans + bbox_wh[N,1,1,2] + crop_meta + cam_K[N,1,1,4]
        -> CdpnTransDecode -> translation[N,1,1,3]

    Outputs:
      denorm_coords [N,3,64,64], confidence [N,1,64,64],
      translation [N,1,1,3], crop_meta [N,1,1,5]

    PnP stays in host code (ov_infer.py).
    Requires cdpn_extension.so at runtime (core.add_extension).

    All shapes are 4D.
    """
    os.makedirs(output_dir, exist_ok=True)

    # CdpnPreprocess: graph-topology wrapper (export-time only).
    # At inference time:
    #   CPU → cdpn_extension.so provides evaluate() in C++
    #   GPU → cdpn_custom_gpu_kernels.xml + .cl kernels (SimpleGPU)
    CdpnPreprocess = _make_CdpnPreprocess(ov, OVType)

    # Step 1: Get NN body
    if nn_ov_model is not None:
        print('[ov_export:extnn] Step 1: Reusing provided OV model ...')
        nn_model = nn_ov_model
    else:
        print('[ov_export:extnn] Step 1: NN body NOT converted, check!!...')
        exit(1)

    # Step 2: Build EXTNN graph with custom ops
    print('[ov_export:extnn] Step 2: Building EXTNN graph ...')

    # Input parameters (all 4D)
    param_image, param_bbox, param_extents, param_bbox_wh, param_cam_K = \
        _build_composite_input_params(opset, ov, OVType)

    # CdpnPreprocess: image + bbox -> tensor + crop_meta
    preprocess = CdpnPreprocess(
        param_image.output(0), param_bbox.output(0))
    tensor_out = preprocess.output(0)      # [N, 3, 256, 256]
    crop_meta_out = preprocess.output(1)   # [N, 1, 1, 5]

    # Rewire NN body input; extract coord_maps and raw_trans_4d
    coord_maps_node, raw_trans_4d = _rewire_nn_body(opset, nn_model, tensor_out)

    denorm_coords_4d, confidence_4d = _build_coord_denorm_subgraph(
        opset, coord_maps_node, param_extents.output(0))

    eps = opset.constant(np.float32(1e-30))
    crop_meta_for_trans = opset.add(crop_meta_out, eps)

    translation_4d = _build_trans_decode_subgraph(
        opset, raw_trans_4d, param_bbox_wh.output(0),
        crop_meta_for_trans, param_cam_K.output(0))


    # Step 3: Assemble EXTNN model
    print('[ov_export:extnn] Step 3: Creating EXTNN model ...')

    extnn_model = ov.Model(
        results=[
            OVResult(denorm_coords_4d.output(0)),
            OVResult(confidence_4d.output(0)),
            OVResult(translation_4d.output(0)),
            OVResult(crop_meta_out),
        ],
        parameters=[
            param_image, param_bbox, param_extents,
            param_bbox_wh, param_cam_K,
        ],
        name='cdpn_stage3_extnn',
    )

    extnn_model.outputs[0].tensor.set_names({'denorm_coords'})
    extnn_model.outputs[1].tensor.set_names({'confidence'})
    extnn_model.outputs[2].tensor.set_names({'translation'})
    extnn_model.outputs[3].tensor.set_names({'crop_meta'})

    extnn_model.inputs[0].tensor.set_names({'image'})
    extnn_model.inputs[1].tensor.set_names({'bbox'})
    extnn_model.inputs[2].tensor.set_names({'obj_extents'})
    extnn_model.inputs[3].tensor.set_names({'bbox_wh'})
    extnn_model.inputs[4].tensor.set_names({'cam_K'})

    _print_model_io(extnn_model, 'ov_export:extnn')

    # Step 4: Save EXTNN model
    xml_path, bin_path, bin_size = _save_ir(
        ov, extnn_model, output_dir, basename, 'ov_export:extnn')
    results = {'xml': xml_path, 'bin': bin_path, 'size_mb': bin_size}

    # Step 5: Verification
    if verify:
        if not extension_path or not os.path.isfile(extension_path):
            print('[ov_export:e2e] WARNING: Skipping verify - '
                  'provide --extension /path/to/cdpn_extension.so')
        else:
            print()
            print('[ov_export:extnn] === Verification ===')
            core = ov.Core()
            core.add_extension(extension_path)
            compiled = core.compile_model(xml_path, 'CPU')

            ov_out = compiled(_make_verify_inputs())

            for idx, name in enumerate(
                    ['denorm_coords', 'confidence', 'translation', 'crop_meta']):
                print('[ov_export:extnn]   {} shape: {}'.format(
                    name, ov_out[compiled.output(idx)].shape))

    print()
    print('[ov_export:extnn] Done.')
    return results


def export_e2e_model(output_dir, basename='cdpn_stage3_e2e',
                     extension_path=None, verify=False, nn_ov_model=None):
    """
    Export the End-to-End CDPN model with custom extension ops.

    The E2E model includes preprocessing, NN body, postprocessing,
    and PnP solve all within the OV graph via custom ops from ov_plugins/.

    Graph:
      image[H,W,3] u8 + bbox[4] -> CdpnPreprocess -> tensor[1,3,256,256] + crop_meta[5]
      tensor -> NN body -> coord_maps[1,4,64,64] + raw_trans[1,3]
      coord_maps + obj_extents -> CdpnCoordDenorm -> denorm_coords[3,64,64] + confidence[64,64]
      raw_trans + bbox_wh + crop_meta + cam_K -> CdpnTransDecode -> translation[3]
      denorm_coords + confidence + obj_ext + crop_meta + cam_K -> CdpnPnpSolve -> R[3,3] + T_pnp[3]
      Pose composition (standard opset): pose_rot = [R|T_pnp], pose_trans = [R|T_trans]

    Outputs: pose_rot[3,4], pose_trans[3,4], denorm_coords, confidence,
             num_corres, pnp_success

    Requires the C++ extension .so from ov_plugins/build/ at runtime.

    Parameters
    ----------
    model : torch.nn.Module
        CDPN model in eval mode.
    output_dir : str
        Directory to write .xml/.bin files.
    basename : str
        Base filename (default: cdpn_stage3_e2e).
    extension_path : str or None
        Path to cdpn_extension.so (needed for verification only).
    verify : bool
        If True, run verification (requires extension .so).
    nn_ov_model : ov.Model or None
        Pre-converted NN body (reuse to avoid re-conversion).
    """
    os.makedirs(output_dir, exist_ok=True)

    CdpnPreprocess = _make_CdpnPreprocess(ov, OVType)

    class CdpnPnpSolve(ov.Op):
        """denorm[3,64,64] + conf[64,64] + obj_ext[3] + crop_meta[5] + cam_K[4]
            -> R[3,3] + T_pnp[3] + num_corres[1] + pnp_success[1]."""

        def __init__(self, denorm_coords, confidence, obj_extents,
                        crop_meta, cam_K,
                        mask_threshold=0.5, out_res=64,
                        max_iterations=100, reproj_threshold=8.0):
            super().__init__(self)
            self._mask_threshold = float(mask_threshold)
            self._out_res = int(out_res)
            self._max_iterations = int(max_iterations)
            self._reproj_threshold = float(reproj_threshold)
            self.set_arguments([denorm_coords, confidence,
                                obj_extents, crop_meta, cam_K])
            self.constructor_validate_and_infer_types()

        @classmethod
        def get_type_info_static(cls):
            return ov.DiscreteTypeInfo("CdpnPnpSolve", "extension")

        def validate_and_infer_types(self):
            # All outputs padded to 4D
            batch = self.get_input_partial_shape(0)[0]
            self.set_output_type(0, OVType.f32, ov.PartialShape([batch, 1, 3, 3]))
            self.set_output_type(1, OVType.f32, ov.PartialShape([batch, 1, 1, 3]))
            self.set_output_type(2, OVType.f32, ov.PartialShape([batch, 1, 1, 1]))
            self.set_output_type(3, OVType.f32, ov.PartialShape([batch, 1, 1, 1]))

        def visit_attributes(self, visitor):
            visitor.on_attributes({
                "mask_threshold": self._mask_threshold,
                "out_res": self._out_res,
                "max_iterations": self._max_iterations,
                "reproj_threshold": self._reproj_threshold,
            })
            return True

        def has_evaluate(self):
            return False

        def clone_with_new_inputs(self, new_args):
            return CdpnPnpSolve(
                new_args[0], new_args[1], new_args[2],
                new_args[3], new_args[4],
                self._mask_threshold, self._out_res,
                self._max_iterations, self._reproj_threshold)

    # Step 1: Get NN body
    if nn_ov_model is not None:
        print('[ov_export:e2e] Step 1: Reusing provided OV model ...')
        nn_model = nn_ov_model
    else:
        print('[ov_export:e2e] Step 1: NN body NOT converted, check!!...')
        exit(1)

    # Step 2: Build E2E graph
    print('[ov_export:e2e] Step 2: Building E2E graph with custom ops ...')

    # Input parameters
    param_image, param_bbox, param_extents, param_bbox_wh, param_cam_K = \
        _build_composite_input_params(opset, ov, OVType)

    # CdpnPreprocess: image + bbox -> tensor + crop_meta
    preprocess = CdpnPreprocess(
        param_image.output(0), param_bbox.output(0))
    tensor_out = preprocess.output(0)     # [N, 3, 256, 256]
    crop_meta_out = preprocess.output(1)  # [N, 1, 1, 5]  (4D)

    # Rewire NN body input; extract coord_maps and raw_trans_4d to receive CdpnPreprocess output
    coord_maps_node, raw_trans_4d = _rewire_nn_body(opset, nn_model, tensor_out)

    denorm_coords_4d, confidence_4d = _build_coord_denorm_subgraph(
        opset, coord_maps_node, param_extents.output(0))

    # CdpnTransDecode: replace with standard opset (same as EXTNN export)
    eps = opset.constant(np.float32(1e-30))
    crop_meta_for_trans = opset.add(crop_meta_out, eps)
    crop_meta_for_pnp = opset.add(crop_meta_out, eps)

    translation_4d = _build_trans_decode_subgraph(
        opset, raw_trans_4d, param_bbox_wh.output(0),
        crop_meta_for_trans, param_cam_K.output(0))

    # CdpnPnpSolve: denorm_coords + confidence + obj_ext + crop_meta + cam_K
    denorm_for_pnp = opset.add(denorm_coords_4d, eps)
    conf_for_pnp = opset.add(confidence_4d, eps)

    pnp_solve = CdpnPnpSolve(
        denorm_for_pnp.output(0), conf_for_pnp.output(0),
        param_extents.output(0), crop_meta_for_pnp.output(0),
        param_cam_K.output(0))
    R_4d = pnp_solve.output(0)             # [N, 1, 3, 3]
    T_pnp_4d = pnp_solve.output(1)         # [N, 1, 1, 3]
    num_corres_4d = pnp_solve.output(2)    # [N, 1, 1, 1]
    pnp_success_4d = pnp_solve.output(3)   # [N, 1, 1, 1]

    # Squeeze 4D custom op outputs to original shapes.
    axis_1 = opset.constant(np.int64(1))
    axes_12 = opset.constant(np.array([1, 2], dtype=np.int64))

    R_3d = opset.squeeze(R_4d, axis_1)                            # [N, 3, 3]
    denorm_coords_out = denorm_coords_4d                          # [N, 3, 64, 64]
    confidence_out = opset.squeeze(confidence_4d, axis_1)         # [N, 64, 64]
    num_corres_out = opset.squeeze(num_corres_4d, axes_12)        # [N, 1]
    pnp_success_out = opset.squeeze(pnp_success_4d, axes_12)      # [N, 1]

    # Pose composition using standard opset: [N,3,3] concat [N,3,1] → [N,3,4]
    shape_N31 = opset.constant(np.array([0, 3, 1], dtype=np.int64))
    T_pnp_col = opset.reshape(T_pnp_4d, shape_N31, True)          # [N,3,1]
    T_trans_col = opset.reshape(translation_4d, shape_N31, True)  # [N,3,1]
    pose_rot_out = opset.concat([R_3d, T_pnp_col], axis=2)        # [N,3,4]
    pose_trans_out = opset.concat([R_3d, T_trans_col], axis=2)    # [N,3,4]

    # Step 3: Assemble E2E model
    print('[ov_export:e2e] Step 3: Creating E2E model ...')
    e2e_model = ov.Model(
        results=[
            OVResult(pose_rot_out.output(0)),
            OVResult(pose_trans_out.output(0)),
            OVResult(denorm_coords_out.output(0)),
            OVResult(confidence_out.output(0)),
            OVResult(num_corres_out.output(0)),
            OVResult(pnp_success_out.output(0)),
        ],
        parameters=[
            param_image, param_bbox, param_extents,
            param_bbox_wh, param_cam_K,
        ],
        name='cdpn_stage3_e2e',
    )

    e2e_model.outputs[0].tensor.set_names({'pose_rot'})
    e2e_model.outputs[1].tensor.set_names({'pose_trans'})
    e2e_model.outputs[2].tensor.set_names({'denorm_coords'})
    e2e_model.outputs[3].tensor.set_names({'confidence'})
    e2e_model.outputs[4].tensor.set_names({'num_corres'})
    e2e_model.outputs[5].tensor.set_names({'pnp_success'})

    _print_model_io(e2e_model, 'ov_export:e2e')

    # Step 4: Save
    xml_path, bin_path, bin_size = _save_ir(
        ov, e2e_model, output_dir, basename, 'ov_export:e2e')
    results = {'xml': xml_path, 'bin': bin_path, 'size_mb': bin_size}

    # Step 5: Verification (requires extension .so)
    if verify:
        if not extension_path or not os.path.isfile(extension_path):
            print('[ov_export:e2e] WARNING: Skipping verify -- '
                  'provide --extension /path/to/cdpn_extension.so')
        else:
            print()
            print('[ov_export:e2e] === Verification ===')
            core = ov.Core()
            core.add_extension(extension_path)
            compiled = core.compile_model(xml_path, 'CPU')

            ov_out = compiled(_make_verify_inputs())

            for idx, name in enumerate(
                    ['pose_rot', 'pose_trans', 'denorm_coords',
                     'confidence', 'num_corres', 'pnp_success']):
                print('[ov_export:e2e]   {} shape: {}'.format(
                    name, ov_out[compiled.output(idx)].shape))

    print()
    print('[ov_export:e2e] Done.')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export CDPN to OpenVINO IR (direct path)')
    parser.add_argument('--cfg', required=True,
                        help='Config YAML (relative to tools/ or absolute)')
    parser.add_argument('--load_model', required=True,
                        help='.checkpoint file')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for .xml/.bin')
    parser.add_argument('--basename', default='cdpn_stage3',
                        help='Base filename (default: cdpn_stage3)')
    parser.add_argument('--verify', action='store_true',
                        help='Numerical verification after export')
    parser.add_argument('--extnn', action='store_true',
                        help='Also export EXTNN model with pre/post processing')
    parser.add_argument('--e2e', action='store_true',
                        help='Also export E2E model with custom extension ops')
    parser.add_argument('--extension', type=str, default=None,
                        help='Path to cdpn_extension.so')
    args = parser.parse_args()

    if args.extnn and args.e2e:
        parser.error('Export --extnn and --e2e in separate invocations; '
                     'each composite export rewires the converted NN body.')

    model, cfg = build_cdpn_model(args.cfg, args.load_model)

    # Always export the NN body first
    nn_results = export_to_ov_ir(model, args.output_dir, args.basename, args.verify)

    # Optionally export the EXTNN model
    if args.extnn:
        print()
        print('=' * 70)
        print('[ov_export] Building EXTNN model with pre/post processing ops ...')
        print('=' * 70)
        export_extnn_model(args.output_dir,
                          basename=args.basename + '_extnn',
                          extension_path=args.extension,
                          verify=args.verify,
                          nn_ov_model=nn_results.get('ov_model'))

    # Optionally export the E2E model (requires ov_plugins extension at runtime)
    if args.e2e:
        print()
        print('=' * 70)
        print('[ov_export] Building E2E model with custom extension ops ...')
        print('=' * 70)
        export_e2e_model(args.output_dir,
                         basename=args.basename + '_e2e',
                         extension_path=args.extension,
                         verify=args.verify,
                         nn_ov_model=nn_results.get('ov_model'))
