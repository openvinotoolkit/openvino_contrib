#!/usr/bin/env python
#
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""
Export CDPN PyTorch model to OpenVINO IR (.xml/.bin).

Produces:
  - cdpn_stage3.xml/.bin        (FP32 weights)      - NN body only
  - cdpn_stage3_extnn.xml/.bin  (with --extnn flag) - Extended-NN model
                                includes custom ops without PnP
  - cdpn_stage3_e2e.xml/.bin    (with --e2e flag)   - End-to-End model
                                includes PnP custom ops

Usage:
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
import xml.etree.ElementTree as ET

import numpy as np
import torch

import openvino as ov
from openvino import opset13 as opset
from openvino import Type as OVType
from openvino.op import Result as OVResult
import openvino.passes as ov_passes



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


def _slice_last_dim(tensor, start_idx, end_idx):
    axis_3 = opset.constant(np.array([3], dtype=np.int64))
    one_step = opset.constant(np.array([1], dtype=np.int64))

    return opset.slice(tensor,
                       opset.constant(np.array([start_idx], dtype=np.int64)),
                       opset.constant(np.array([end_idx], dtype=np.int64)),
                       one_step, axis_3)


def _build_coord_denorm_subgraph(coord_maps, obj_extents):
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


def _build_trans_decode_subgraph(raw_trans_4d, bbox_wh, crop_meta,
                                 cam_K, out_res=64.0):
    ratio_dcx = _slice_last_dim(raw_trans_4d, 0, 1)
    ratio_dcy = _slice_last_dim(raw_trans_4d, 1, 2)
    ratio_depth = _slice_last_dim(raw_trans_4d, 2, 3)

    bw = _slice_last_dim(bbox_wh, 0, 1)
    bh = _slice_last_dim(bbox_wh, 1, 2)

    c_w = _slice_last_dim(crop_meta, 0, 1)
    c_h = _slice_last_dim(crop_meta, 1, 2)
    s_val = _slice_last_dim(crop_meta, 2, 3)

    fx = _slice_last_dim(cam_K, 0, 1)
    fy = _slice_last_dim(cam_K, 1, 2)
    cx = _slice_last_dim(cam_K, 2, 3)
    cy = _slice_last_dim(cam_K, 3, 4)

    out_res_const = opset.constant(np.float32(out_res))
    pred_depth = opset.multiply(ratio_depth, opset.divide(out_res_const, s_val))
    pred_cx = opset.add(opset.multiply(ratio_dcx, bw), c_w)
    pred_cy = opset.add(opset.multiply(ratio_dcy, bh), c_h)

    tx = opset.divide(opset.multiply(opset.subtract(pred_cx, cx), pred_depth), fx)
    ty = opset.divide(opset.multiply(opset.subtract(pred_cy, cy), pred_depth), fy)

    return opset.concat([tx, ty, pred_depth], axis=3)


def _make_CdpnPreprocess():
    """Return the CdpnPreprocess graph-topology wrapper class.

    The class defines graph topology.
    At inference time:
      CPU → cdpn_extension.so provides evaluate() in C++
      GPU → cdpn_custom_gpu_kernels.xml + .cl kernels
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


def _build_composite_input_params():
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


def _rewire_nn_body(nn_model, tensor_out):
    """Connect an external input to the NN model and extract its outputs.

    Replaces the model's original parameter with tensor_out, then extracts
    the two output nodes: the coordinate maps and the translation vector
    (reshaped to 4D for downstream processing).
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


def _save_ir(model, output_dir, basename, tag):
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
        ov_model, output_dir, basename, 'ov_export:nn')
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
    #   GPU → cdpn_custom_gpu_kernels.xml + .cl kernels
    CdpnPreprocess = _make_CdpnPreprocess()

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
        _build_composite_input_params()

    # CdpnPreprocess: image + bbox -> tensor + crop_meta
    preprocess = CdpnPreprocess(
        param_image.output(0), param_bbox.output(0))
    tensor_out = preprocess.output(0)      # [N, 3, 256, 256]
    crop_meta_out = preprocess.output(1)   # [N, 1, 1, 5]

    # Rewire NN body input; extract coord_maps and raw_trans_4d
    coord_maps_node, raw_trans_4d = _rewire_nn_body(nn_model, tensor_out)

    denorm_coords_4d, confidence_4d = _build_coord_denorm_subgraph(
        coord_maps_node, param_extents.output(0))

    eps = opset.constant(np.float32(1e-30))
    crop_meta_for_trans = opset.add(crop_meta_out, eps)

    translation_4d = _build_trans_decode_subgraph(
        raw_trans_4d, param_bbox_wh.output(0),
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
        extnn_model, output_dir, basename, 'ov_export:extnn')
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

    CdpnPreprocess = _make_CdpnPreprocess()

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
        _build_composite_input_params()

    # CdpnPreprocess: image + bbox -> tensor + crop_meta
    preprocess = CdpnPreprocess(
        param_image.output(0), param_bbox.output(0))
    tensor_out = preprocess.output(0)     # [N, 3, 256, 256]
    crop_meta_out = preprocess.output(1)  # [N, 1, 1, 5]  (4D)

    # Rewire NN body input; extract coord_maps and raw_trans_4d to receive CdpnPreprocess output
    coord_maps_node, raw_trans_4d = _rewire_nn_body(nn_model, tensor_out)

    denorm_coords_4d, confidence_4d = _build_coord_denorm_subgraph(
        coord_maps_node, param_extents.output(0))

    # CdpnTransDecode: replace with standard opset (same as EXTNN export)
    eps = opset.constant(np.float32(1e-30))
    crop_meta_for_trans = opset.add(crop_meta_out, eps)
    crop_meta_for_pnp = opset.add(crop_meta_out, eps)

    translation_4d = _build_trans_decode_subgraph(
        raw_trans_4d, param_bbox_wh.output(0),
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
        e2e_model, output_dir, basename, 'ov_export:e2e')
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


def _find_nn_head_entry_map(model):
    """Return the entry Convolution of each NN head, keyed by head tag.

    Maps each head ('trans', 'rot') to the Convolution whose output feeds
    that head's first feature block, and raises if either is missing.
    """
    entry_names = {}
    tags = (
        ('trans', 'trans_head_net.features.1/'),
        ('rot', 'rot_head_net.features.1/'),
    )
    for op in model.get_ordered_ops():
        if op.get_type_name() not in ('Convolution', 'ConvolutionBackpropData'):
            continue
        consumers = []
        for output in op.outputs():
            consumers.extend(
                target.get_node().get_friendly_name()
                for target in output.get_target_inputs())
        for tag, marker in tags:
            if any(marker in name for name in consumers):
                entry_names[tag] = op.get_friendly_name()

    missing = [tag for tag, _ in tags if tag not in entry_names]
    if missing:
        raise RuntimeError('Expected NN head entry ops for {}, found {}'.format(
            ', '.join(missing), entry_names))

    return entry_names


def _find_rot_final_conv_name(model):
    """Return the friendly name of the final Convolution in the rotation head.

    Scans every Convolution under rot_head_net.features and returns the one
    that appears last; raises if the rotation head contains no convolutions.
    """
    rot_convs = []
    for op in model.get_ordered_ops():
        if op.get_type_name() != 'Convolution':
            continue
        name = op.get_friendly_name()
        if 'rot_head_net.features.' not in name:
            continue
        rot_convs.append(name)

    if not rot_convs:
        raise RuntimeError('No RotHead feature convolutions found in the model')

    return rot_convs[-1]


def _constant_input_names(model, op_names):
    names = set()
    for op in model.get_ordered_ops():
        if op.get_friendly_name() not in op_names:
            continue
        for inp in op.inputs():
            source = inp.get_source_output().get_node()
            if source.get_type_name() == 'Constant':
                names.add(source.get_friendly_name())

    return names


def _downstream_op_names(model, entry_names):
    stack = []
    for op in model.get_ordered_ops():
        if op.get_friendly_name() in entry_names:
            stack.append(op)
    if not stack:
        raise RuntimeError('Entry ops not found: {}'.format(
            ', '.join(entry_names)))

    names = set()
    while stack:
        op = stack.pop()
        name = op.get_friendly_name()
        if name in names:
            continue
        names.add(name)
        for output in op.outputs():
            for target_input in output.get_target_inputs():
                target = target_input.get_node()
                if target.get_type_name() != 'Result':
                    stack.append(target)

    for op in model.get_ordered_ops():
        if op.get_friendly_name() not in names:
            continue
        for inp in op.inputs():
            source = inp.get_source_output().get_node()
            if source.get_type_name() == 'Constant':
                names.add(source.get_friendly_name())

    return names


def _is_float_output(output):
    # Reports whether the output carries a floating-point element type.
    return output.get_element_type() in (ov.Type.f16, ov.Type.f32, ov.Type.bf16)


def _wrap_fp16_island_ops(model, op_names):
    """Insert precision-conversion boundaries around the given island ops.

    Converts each island op's FP32 inputs down to FP16 and its FP16 outputs
    back up to FP32 for any consumer outside the island, reusing a single
    Convert node per output port across all of its consumers.
    """
    input_converts = 0
    output_converts = 0
    for op in list(model.get_ordered_ops()):
        name = op.get_friendly_name()
        if name not in op_names:
            continue
        for idx, inp in enumerate(op.inputs()):
            source = inp.get_source_output()
            source_node = source.get_node()
            if source_node.get_type_name() == 'Constant':
                continue
            if not _is_float_output(source):
                continue
            to_f16 = opset.convert(source, ov.Type.f16)
            to_f16.set_friendly_name(name + '/input{}_fp16'.format(idx))
            inp.replace_source_output(to_f16.output(0))
            input_converts += 1

        for port, output in enumerate(op.outputs()):
            if not _is_float_output(output):
                continue
            to_f32 = None
            for target_input in list(output.get_target_inputs()):
                target = target_input.get_node()
                if (target.get_type_name() != 'Result'
                        and target.get_friendly_name() in op_names):
                    continue
                if to_f32 is None:
                    to_f32 = opset.convert(output, ov.Type.f32)
                    to_f32.set_friendly_name(name + '/output{}_fp32'.format(port))
                target_input.replace_source_output(to_f32.output(0))
                output_converts += 1

    return input_converts, output_converts


def _inject_rt_attribute(xml_path, layer_names, attr_name):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    layers = root.find('layers')
    if layers is None:
        raise RuntimeError('No <layers> found in {}'.format(xml_path))

    marked = 0
    for layer in layers.findall('layer'):
        if layer.get('name') not in layer_names:
            continue
        rt_info = layer.find('rt_info')
        if rt_info is None:
            rt_info = ET.Element('rt_info')
            output = layer.find('output')
            children = list(layer)
            insert_at = children.index(output) + 1 if output in children else 0
            layer.insert(insert_at, rt_info)
        if any(child.get('name') == attr_name for child in rt_info):
            continue
        ET.SubElement(rt_info, 'attribute', {
            'name': attr_name,
            'version': '0',
        })
        marked += 1

    tree.write(xml_path, encoding='UTF-8', xml_declaration=True)

    return marked


def _find_single_consumer_name(model, op_name, consumer_type=None):
    source = None
    for op in model.get_ordered_ops():
        if op.get_friendly_name() == op_name:
            source = op
            break
    if source is None:
        raise RuntimeError('Op not found: {}'.format(op_name))

    matches = []
    for output in source.outputs():
        for target_input in output.get_target_inputs():
            target = target_input.get_node()
            if consumer_type is None or target.get_type_name() == consumer_type:
                matches.append(target.get_friendly_name())

    if len(matches) != 1:
        raise RuntimeError('Expected one consumer for {}, found {}'.format(
            op_name, matches))

    return matches[0]


def _ordered_region_op_names(model, start_name, end_name):
    ordered = model.get_ordered_ops()
    start_idx = None
    end_idx = None

    for idx, op in enumerate(ordered):
        name = op.get_friendly_name()
        if name == start_name:
            start_idx = idx
        if name == end_name:
            end_idx = idx

    if start_idx is None or end_idx is None or end_idx < start_idx:
        raise RuntimeError('Invalid ordered region: {} -> {}'.format(
            start_name, end_name))

    names = set(op.get_friendly_name() for op in ordered[start_idx:end_idx + 1])

    for op in ordered[start_idx:end_idx + 1]:
        for inp in op.inputs():
            source = inp.get_source_output().get_node()
            if source.get_type_name() == 'Constant':
                names.add(source.get_friendly_name())

    return names


def _prepare_composite_fp16_inputs(model):
    inserted = 0
    for op in list(model.get_ordered_ops()):
        if op.get_type_name() != 'CdpnPreprocess':
            continue
        tensor_out = op.output(0)
        if _is_float_output(tensor_out):
            to_f16 = opset.convert(tensor_out, ov.Type.f16)
            to_f16.set_friendly_name(op.get_friendly_name() + '/tensor_fp16')
            for target_input in list(tensor_out.get_target_inputs()):
                if target_input.get_node() is to_f16:
                    continue
                target_input.replace_source_output(to_f16.output(0))
                inserted += 1

        if len(op.outputs()) < 2:
            continue
        crop_meta_out = op.output(1)
        if not _is_float_output(crop_meta_out):
            continue
        to_f16 = opset.convert(crop_meta_out, ov.Type.f16)
        to_f16.set_friendly_name(op.get_friendly_name() + '/crop_meta_fp16')
        for target_input in list(crop_meta_out.get_target_inputs()):
            target = target_input.get_node()
            if target is to_f16 or target.get_type_name() == 'Result':
                continue
            target_input.replace_source_output(to_f16.output(0))
            inserted += 1

    return inserted


def _convert_precise_region_inputs_to_f32(model, precise_names,
                                          skip_source_names=None):
    skip_source_names = skip_source_names or set()
    inserted = 0
    for op in list(model.get_ordered_ops()):
        if op.get_friendly_name() not in precise_names:
            continue
        if op.get_type_name() in ('Constant', 'Parameter', 'Result'):
            continue
        for idx, inp in enumerate(op.inputs()):
            source = inp.get_source_output()
            source_node = source.get_node()
            source_name = source_node.get_friendly_name()
            if source_name in precise_names or source_name in skip_source_names:
                continue
            if not _is_float_output(source):
                continue
            to_f32 = opset.convert(source, ov.Type.f32)
            to_f32.set_friendly_name(
                op.get_friendly_name() + '/input{}_fp32'.format(idx))
            inp.replace_source_output(to_f32.output(0))
            inserted += 1

    return inserted


def _convert_region_outputs_to_f16(model, source_names, target_region_names):
    inserted = 0
    for op in list(model.get_ordered_ops()):
        if op.get_friendly_name() not in source_names:
            continue
        for port, output in enumerate(op.outputs()):
            if not _is_float_output(output):
                continue
            to_f16 = None
            for target_input in list(output.get_target_inputs()):
                target = target_input.get_node()
                if target.get_type_name() == 'Result':
                    continue
                if target.get_friendly_name() in target_region_names:
                    continue
                if to_f16 is None:
                    to_f16 = opset.convert(output, ov.Type.f16)
                    to_f16.set_friendly_name(
                        op.get_friendly_name() + '/output{}_fp16'.format(port))
                target_input.replace_source_output(to_f16.output(0))
                inserted += 1

    return inserted


def _prepare_e2e_pose_concat_inputs(model):
    inserted = 0

    for op in list(model.get_ordered_ops()):
        if op.get_type_name() != 'Concat':
            continue

        shape = op.output(0).get_partial_shape()
        if shape.rank.is_dynamic or shape.rank.get_length() != 3:
            continue

        dims = list(shape)
        if dims[1].is_dynamic or dims[2].is_dynamic:
            continue

        if dims[1].get_length() != 3 or dims[2].get_length() != 4:
            continue

        for idx, inp in enumerate(op.inputs()):
            source = inp.get_source_output()
            if not _is_float_output(source):
                continue

            to_f32 = opset.convert(source, ov.Type.f32)
            to_f32.set_friendly_name(
                op.get_friendly_name() + '/input{}_fp32'.format(idx))
            inp.replace_source_output(to_f32.output(0))

            inserted += 1

    return inserted


def _export_mixed_fp16_ir(model_path, output_dir, basename, tag, ir_label,
                          *, load_extension=False, extension_path=None,
                          require_outputs=(), require_pnp=False,
                          use_ordered_region=False, include_custom_precise=False,
                          input_hooks=(), pose_concat_hook=None,
                          do_boundary_converts=False):
    """Export a mixed-precision FP16 IR from an FP32 model.

    Keeps the RotHead core in FP32, converts the remaining graph to FP16,
    and joins the two precision regions with Convert nodes. The EXTNN and E2E
    variants pass extra steps through the ``input_hooks`` and
    ``pose_concat_hook`` parameters to cast custom-op inputs and the pose
    Concat inputs.
    """
    os.makedirs(output_dir, exist_ok=True)

    core = ov.Core()
    if load_extension:
        if extension_path is None:
            extension_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'ov_plugins', 'build', 'cdpn_extension.so')
        if extension_path and os.path.isfile(extension_path):
            core.add_extension(extension_path)

    original = core.read_model(model_path)

    if require_outputs or require_pnp:
        output_names = set()
        for output in original.outputs:
            output_names.update(output.get_names())
        for required in require_outputs:
            if required not in output_names:
                raise RuntimeError('Expected {} model with {} output'.format(
                    ir_label, required))
        if require_pnp and not any(op.get_type_name() == 'CdpnPnpSolve'
                                   for op in original.get_ordered_ops()):
            raise RuntimeError('Expected {} model with CdpnPnpSolve op'.format(
                ir_label))

    head_entry_map = _find_nn_head_entry_map(original)
    rot_entry_name = head_entry_map['rot']

    rot_final_conv_name = _find_rot_final_conv_name(original)
    fp16_island_names = set([rot_final_conv_name])
    fp16_island_constants = _constant_input_names(original, fp16_island_names)

    # Plain NN follows the rot head to the model outputs; the composite
    # variants stop at the final Add so the custom ops stay in FP32.
    if use_ordered_region:
        rot_final_add_name = _find_single_consumer_name(
            original, rot_final_conv_name, consumer_type='Add')
        rot_head_names = _ordered_region_op_names(
            original, rot_entry_name, rot_final_add_name)
    else:
        rot_final_add_name = None
        rot_head_names = _downstream_op_names(original, [rot_entry_name])

    precise_names = rot_head_names
    if include_custom_precise:
        custom_precise_names = set(
            op.get_friendly_name() for op in original.get_ordered_ops()
            if op.get_type_name() in ('CdpnPreprocess', 'CdpnPnpSolve'))
        precise_names = precise_names | custom_precise_names
    precise_names = precise_names - fp16_island_names - fp16_island_constants

    # Snapshot the FP32 constants feeding the precise region so it can be
    # restored after the blanket FP16 conversion below.
    fp32_constants = {}
    for op in original.get_ordered_ops():
        if (op.get_friendly_name() in precise_names
                and op.get_type_name() == 'Constant'
                and _is_float_output(op.output(0))):
            fp32_constants[op.get_friendly_name()] = op.get_data().astype(np.float32)

    mixed_model = core.read_model(model_path)

    preprocess_converts = 0
    for hook in input_hooks:
        preprocess_converts += hook(mixed_model)
    pose_concat_converts = pose_concat_hook(mixed_model) if pose_concat_hook else 0

    manager = ov_passes.Manager()
    manager.register_pass(ov_passes.ConvertFP32ToFP16())
    manager.run_passes(mixed_model)

    for op in list(mixed_model.get_ordered_ops()):
        name = op.get_friendly_name()
        if name not in fp32_constants:
            continue
        restored = opset.constant(fp32_constants[name])
        restored.set_friendly_name(name + '/fp32')
        for target_input in list(op.output(0).get_target_inputs()):
            target_input.replace_source_output(restored.output(0))

    for op in mixed_model.get_ordered_ops():
        if op.get_friendly_name() != rot_entry_name:
            continue
        source = op.input(0).get_source_output()
        to_f32 = opset.convert(source, ov.Type.f32)
        to_f32.set_friendly_name(op.get_friendly_name() + '/input_fp32')
        op.input(0).replace_source_output(to_f32.output(0))

    fp16_inputs, fp16_outputs = _wrap_fp16_island_ops(
        mixed_model, fp16_island_names)

    precise_boundary_converts = 0
    precise_output_converts = 0
    if do_boundary_converts:
        precise_boundary_converts = _convert_precise_region_inputs_to_f32(
            mixed_model, precise_names, skip_source_names=fp16_island_names)
        precise_output_converts = _convert_region_outputs_to_f16(
            mixed_model, [rot_final_add_name], precise_names)

    mixed_model.validate_nodes_and_infer_types()

    xml_path = os.path.join(output_dir, basename + '.xml')
    ov.save_model(mixed_model, xml_path, compress_to_fp16=False)

    marked = _inject_rt_attribute(xml_path, precise_names, 'precise')

    bin_path = os.path.join(output_dir, basename + '.bin')
    size_mb = os.path.getsize(bin_path) / 1e6

    print('[{}] Mixed {} IR: {} ({:.1f} MB)'.format(
        tag, ir_label, xml_path, size_mb))
    print('[{}] RotHead core kept FP32 from: {}'.format(tag, rot_entry_name))
    print('[{}] FP16 island ops: {}'.format(
        tag, ', '.join(sorted(fp16_island_names))))
    print('[{}] Restored {} FP32 constants; marked {} precise layers'.format(
        tag, len(fp32_constants), marked))
    print('[{}] Inserted FP16 island converts: {} input, {} output'.format(
        tag, fp16_inputs, fp16_outputs))
    if do_boundary_converts:
        print('[{}] Inserted precise-boundary FP32 converts: {}'.format(
            tag, precise_boundary_converts))
        print('[{}] Inserted precise-output FP16 converts: {}'.format(
            tag, precise_output_converts))
        print('[{}] Inserted CdpnPreprocess FP16 tensor converts: {}'.format(
            tag, preprocess_converts))
    if pose_concat_hook is not None:
        print('[{}] Inserted E2E pose concat FP32 converts: {}'.format(
            tag, pose_concat_converts))

    return {'xml': xml_path, 'bin': bin_path, 'size_mb': size_mb}


def export_nn_mixed_fp16_ir(model_path, output_dir,
                            basename='cdpn_stage3_fp16'):
    """Export the NN IR with an FP16 body and an FP32 RotHead core."""
    return _export_mixed_fp16_ir(
        model_path, output_dir, basename,
        tag='ov_export:fp16', ir_label='NN')


def export_extnn_mixed_fp16_ir(model_path, output_dir,
                               basename='cdpn_stage3_extnn_fp16',
                               extension_path=None):
    """Export the EXTNN IR with an FP16 body/postproc and FP32 custom-op boundaries."""
    return _export_mixed_fp16_ir(
        model_path, output_dir, basename,
        tag='ov_export:fp16-extnn', ir_label='EXTNN',
        load_extension=True, extension_path=extension_path,
        require_outputs=('crop_meta',),
        use_ordered_region=True, include_custom_precise=True,
        input_hooks=(_prepare_composite_fp16_inputs,),
        do_boundary_converts=True)


def export_e2e_mixed_fp16_ir(model_path, output_dir,
                             basename='cdpn_stage3_e2e_fp16',
                             extension_path=None):
    """Export the E2E IR with an FP16 body/postproc (incl. PnP) and FP32 custom-op boundaries."""
    return _export_mixed_fp16_ir(
        model_path, output_dir, basename,
        tag='ov_export:fp16-e2e', ir_label='E2E',
        load_extension=True, extension_path=extension_path,
        require_outputs=('pose_rot',), require_pnp=True,
        use_ordered_region=True, include_custom_precise=True,
        input_hooks=(_prepare_composite_fp16_inputs,),
        pose_concat_hook=_prepare_e2e_pose_concat_inputs,
        do_boundary_converts=True)


_NN_INT8_HEAD_GROUPS = (
    'trans_conv0', 'trans_conv1', 'trans_mlp0',
    'rot_conv1', 'rot_conv7',
)

_NN_FP16_HEAD_GROUPS = (
    'trans_conv2', 'trans_mlp1', 'trans_mlp2',
    'rot_conv2', 'rot_conv5', 'rot_conv8', 'rot_final',
)


def _find_head_layer_op_name(model, marker):
    matches = []
    for op in model.get_ordered_ops():
        name = op.get_friendly_name()
        if marker not in name or '/fq_' in name:
            continue
        matches.append(name)

    if len(matches) != 1:
        raise RuntimeError('[ov_export:find_head_layer_op_name] Expected one op containing {}, found {}'.format(
            marker, matches))

    return matches[0]


def _nn_head_feature_group_names(model, prefix, bn_idx):
    bn_name = _find_head_layer_op_name(
        model, '{}.features.{}/'.format(prefix, bn_idx))
    relu_name = _find_head_layer_op_name(
        model, '{}.features.{}/'.format(prefix, bn_idx + 1))

    bn_op = None
    for op in model.get_ordered_ops():
        if op.get_friendly_name() == bn_name:
            bn_op = op
            break

    if bn_op is None:
        raise RuntimeError('[ov_export:_nn_head_feature_group_names] BatchNorm op not found: {}'.format(bn_name))

    conv_name = bn_op.input(0).get_source_output().get_node().get_friendly_name()

    return set([conv_name, bn_name, relu_name])


def _nn_trans_linear_group_names(model, linear_idx, relu_idx=None):
    names = set([
        _find_head_layer_op_name(
            model,
            'trans_head_net.linears.{}/aten::linear/MatMul'.format(linear_idx)),
        _find_head_layer_op_name(
            model,
            'trans_head_net.linears.{}/aten::linear/Add'.format(linear_idx)),
    ])

    if relu_idx is not None:
        names.add(_find_head_layer_op_name(
            model,
            'trans_head_net.linears.{}/aten::relu_/Relu'.format(relu_idx)))

    return names


def _nn_head_feature_bn_indices(model, prefix):
    """Return the BatchNorm block indices for a head's feature Sequential.

    Scans the exported IR for ops under '{prefix}.features.{N}/' and identifies
    each conv→bn→relu block by locating an 'Add' (the folded BatchNorm
    scale-shift) immediately followed by a 'Relu'. Returns the index of the
    'Add' for every such block, sorted in ascending order.
    """
    type_by_idx = {}
    marker = '{}.features.'.format(prefix)
    for op in model.get_ordered_ops():
        name = op.get_friendly_name()
        if marker not in name:
            continue
        try:
            idx = int(name.split(marker)[1].split('/')[0])
        except (IndexError, ValueError):
            continue
        type_by_idx.setdefault(idx, op.get_type_name())

    bn_indices = [idx for idx, tname in type_by_idx.items()
                  if tname == 'Add' and type_by_idx.get(idx + 1) == 'Relu']
    bn_indices.sort()

    return bn_indices


def _nn_trans_linear_indices(model):
    """Return the TransHead MLP linear indices and their trailing ReLU indices.

    Scans the exported IR for ops under 'trans_head_net.linears.{N}/' and
    treats each 'MatMul' as a linear layer (index N). When a 'Relu' follows at
    index N+1, that index is paired with the linear; the final linear has no
    trailing ReLU and is paired with None. Returns a list of
    (linear_idx, relu_idx) pairs in ascending linear-index order.
    """
    type_by_idx = {}
    marker = 'trans_head_net.linears.'
    for op in model.get_ordered_ops():
        name = op.get_friendly_name()
        if marker not in name:
            continue
        try:
            idx = int(name.split(marker)[1].split('/')[0])
        except (IndexError, ValueError):
            continue
        # Prefer the MatMul/Relu op type for each index.
        tname = op.get_type_name()
        if idx not in type_by_idx or tname in ('MatMul', 'Relu'):
            type_by_idx[idx] = tname

    linear_indices = sorted(idx for idx, tname in type_by_idx.items()
                            if tname == 'MatMul')
    pairs = []
    for linear_idx in linear_indices:
        relu_idx = linear_idx + 1 if type_by_idx.get(linear_idx + 1) == 'Relu' else None
        pairs.append((linear_idx, relu_idx))

    return pairs


def _nn_head_layer_groups(model):
    """Return the NN head layer groups keyed by logical unit name.

    Builds one group per logical unit: a trans_conv/rot_conv group for each
    conv→bn→relu block in the trans and rot feature Sequentials, a trans_mlp
    group for each TransHead linear layer, and a rot_final group for the final
    rotation convolution and its consumer Add. Each group is a set of op
    friendly names that make up that unit.
    """
    groups = {}

    for group_idx, bn_idx in enumerate(
            _nn_head_feature_bn_indices(model, 'trans_head_net')):
        groups['trans_conv{}'.format(group_idx)] = _nn_head_feature_group_names(
            model, 'trans_head_net', bn_idx)

    for mlp_idx, (linear_idx, relu_idx) in enumerate(
            _nn_trans_linear_indices(model)):
        groups['trans_mlp{}'.format(mlp_idx)] = _nn_trans_linear_group_names(
            model, linear_idx, relu_idx)

    for group_idx, bn_idx in enumerate(
            _nn_head_feature_bn_indices(model, 'rot_head_net')):
        groups['rot_conv{}'.format(group_idx)] = _nn_head_feature_group_names(
            model, 'rot_head_net', bn_idx)

    rot_final_conv_name = _find_rot_final_conv_name(model)
    rot_final_add_name = _find_single_consumer_name(
        model, rot_final_conv_name, 'Add')
    groups['rot_final'] = set([rot_final_conv_name, rot_final_add_name])

    return groups


def _nn_head_group_region_names(model, group_names):
    available = _nn_head_layer_groups(model)

    unknown = [name for name in group_names if name not in available]
    if unknown:
        raise RuntimeError('[ov_export:nn_head_group_region_names] Unknown NN head groups: {}'.format(
            ', '.join(unknown)))

    op_names = set()
    for group_name in group_names:
        op_names.update(available[group_name])

    constant_names = _constant_input_names(model, op_names)

    return op_names, constant_names, op_names | constant_names


def _nn_int8_precise_scope(model):
    head_entry_map = _find_nn_head_entry_map(model)

    trans_entry_name = head_entry_map['trans']
    rot_entry_name = head_entry_map['rot']

    rot_final_conv_name = _find_rot_final_conv_name(model)
    trans_head_names = _downstream_op_names(model, [trans_entry_name])
    rot_head_names = _downstream_op_names(model, [rot_entry_name])
    head_names = rot_head_names | trans_head_names

    int8_op_names, int8_constant_names, int8_region_names = (
        _nn_head_group_region_names(model, _NN_INT8_HEAD_GROUPS))
    fp16_op_names, fp16_constant_names, fp16_region_names = (
        _nn_head_group_region_names(model, _NN_FP16_HEAD_GROUPS))

    precise_names = head_names - int8_region_names - fp16_region_names
    ignored_for_nncf = head_names - int8_region_names

    ignored_names = set()
    for op in model.get_ordered_ops():
        if op.get_friendly_name() not in ignored_for_nncf:
            continue
        if op.get_type_name() in ('Constant', 'Parameter', 'Result'):
            continue
        ignored_names.add(op.get_friendly_name())

    return {
        'trans_entry_name': trans_entry_name,
        'rot_entry_name': rot_entry_name,
        'rot_final_conv_name': rot_final_conv_name,
        'head_names': head_names,
        'int8_group_names': _NN_INT8_HEAD_GROUPS,
        'fp16_group_names': _NN_FP16_HEAD_GROUPS,
        'int8_op_names': int8_op_names,
        'int8_constant_names': int8_constant_names,
        'int8_region_names': int8_region_names,
        'fp16_op_names': fp16_op_names,
        'fp16_constant_names': fp16_constant_names,
        'fp16_region_names': fp16_region_names,
        'precise_names': precise_names,
        'ignored_names': ignored_names,
    }


def _balanced_calibration_indices(dataset, limit):
    total = len(dataset)
    if limit <= 0 or limit >= total:
        return list(range(total))

    annot = getattr(dataset, 'annot', None)
    if not annot:
        return list(range(min(limit, total)))

    by_obj = {}
    for idx, item in enumerate(annot):
        by_obj.setdefault(item.get('obj', ''), []).append(idx)

    obj_names = sorted(by_obj)
    offsets = dict((obj, 0) for obj in obj_names)
    selected = []
    while len(selected) < limit:
        progressed = False
        for obj in obj_names:
            offset = offsets[obj]
            if offset >= len(by_obj[obj]):
                continue
            selected.append(by_obj[obj][offset])
            offsets[obj] = offset + 1
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break

    return selected


class _NNInt8CalibrationData(object):
    def __init__(self, dataset, indices):
        self._dataset = dataset
        self._indices = indices
        self.batch_size = 1

    def __len__(self):
        return len(self._indices)

    def __iter__(self):
        for idx in self._indices:
            sample = self._dataset[idx]
            inp = sample[2]
            yield np.expand_dims(inp.astype(np.float32, copy=False), axis=0)


def _make_nn_int8_calibration_dataset(cfg, subset_size, dataset_dir=None):
    import nncf
    from datasets.lm import LM

    if dataset_dir:
        import ref
        ref.lm_dir = dataset_dir
        ref.lm_test_dir = os.path.join(dataset_dir, 'real_test')
        ref.lm_train_real_dir = os.path.join(dataset_dir, 'real_train')
        ref.lm_train_imgn_dir = os.path.join(dataset_dir, 'imgn')
        ref.lm_model_info_pth = os.path.join(dataset_dir, 'models', 'models_info.txt')
        ref.cache_dir = os.path.join(os.path.dirname(dataset_dir), 'dataset_cache')

    dataset = LM(cfg, 'test')
    if len(dataset) == 0:
        raise RuntimeError('[ov_export:make_nn_int8_calibration_dataset] No calibration samples found in dataset')

    limit = len(dataset) if subset_size <= 0 else min(subset_size, len(dataset))
    indices = _balanced_calibration_indices(dataset, limit)
    data_source = _NNInt8CalibrationData(dataset, indices)

    return nncf.Dataset(data_source), len(indices)


def _nncf_target_device(nncf, target_device):
    key = target_device.upper()
    try:
        return getattr(nncf.TargetDevice, key)
    except AttributeError:
        raise ValueError('[ov_export:nncf_target_device] Unsupported NNCF target device: {}'.format(
            target_device))


def _nncf_quantization_preset(nncf, preset):
    key = preset.upper()
    try:
        return getattr(nncf.QuantizationPreset, key)
    except AttributeError:
        raise ValueError('[ov_export:nncf_quantization_preset] Unsupported NNCF quantization preset: {}'.format(
            preset))


def _convert_region_constants_to_f16(model, region_names):
    converted = 0
    rewired = 0
    for op in list(model.get_ordered_ops()):
        name = op.get_friendly_name()
        if name not in region_names or op.get_type_name() != 'Constant':
            continue
        if not _is_float_output(op.output(0)):
            continue

        targets = [target_input for target_input in list(
            op.output(0).get_target_inputs())
            if target_input.get_node().get_friendly_name() in region_names]
        if not targets:
            continue

        replacement = opset.constant(op.get_data().astype(np.float16))
        replacement.set_friendly_name(name + '/fp16')
        for target_input in targets:
            target_input.replace_source_output(replacement.output(0))
            rewired += 1
        converted += 1

    return converted, rewired


def _nn_int8_quantization_point_summary(model, head_names):
    total = 0
    named_inside_head = 0
    directly_feeding_head = 0
    for op in model.get_ordered_ops():
        if op.get_type_name() != 'FakeQuantize':
            continue
        total += 1

        if op.get_friendly_name() in head_names:
            named_inside_head += 1

        consumers = []
        for output in op.outputs():
            for target_input in output.get_target_inputs():
                consumers.append(target_input.get_node().get_friendly_name())

        if any(consumer in head_names for consumer in consumers):
            directly_feeding_head += 1

    return total, named_inside_head, directly_feeding_head


def export_nn_int8_ir(model_path, output_dir, cfg,
                      basename='cdpn_stage3_int8', subset_size=300,
                      target_device='GPU', preset='PERFORMANCE',
                      dataset_dir=None):
    """Export NN IR with NNCF INT8 PTQ and a mixed-precision head policy.
    """
    import nncf

    os.makedirs(output_dir, exist_ok=True)

    core = ov.Core()
    ov_model = core.read_model(model_path)

    scope = _nn_int8_precise_scope(ov_model)
    if not scope['ignored_names']:
        print('[ov_export:int8-nn] No FP32 precise NN layers found for INT8 export')

    calibration_dataset, calibration_size = _make_nn_int8_calibration_dataset(
        cfg, subset_size, dataset_dir=dataset_dir)
    ignored_scope = nncf.IgnoredScope(
        names=sorted(scope['ignored_names']), validate=True)

    print('[ov_export:int8-nn] Calibration samples: {}'.format(
        calibration_size))
    print('[ov_export:int8-nn] Target device: {}, preset: {}'.format(
        target_device.upper(), preset.upper()))
    print('[ov_export:int8-nn] INT8 head groups: {}'.format(
        ', '.join(scope['int8_group_names'])))
    print('[ov_export:int8-nn] FP16 head groups: {}'.format(
        ', '.join(scope['fp16_group_names'])))
    print('[ov_export:int8-nn] FP32 precise head layers: {}'.format(
        len(scope['precise_names'])))
    print('[ov_export:int8-nn] Ignored NNCF layers: {}'.format(
        len(scope['ignored_names'])))

    quantized_model = nncf.quantize(
        ov_model,
        calibration_dataset,
        subset_size=calibration_size,
        target_device=_nncf_target_device(nncf, target_device),
        preset=_nncf_quantization_preset(nncf, preset),
        ignored_scope=ignored_scope)

    fp16_op_names, _, fp16_region_names = (
        _nn_head_group_region_names(quantized_model, scope['fp16_group_names']))
    constants_converted, constant_edges = _convert_region_constants_to_f16(
        quantized_model, fp16_region_names)
    fp16_inputs, fp16_outputs = _wrap_fp16_island_ops(
        quantized_model, fp16_op_names)
    quantized_model.validate_nodes_and_infer_types()

    xml_path, bin_path, size_mb = _save_ir(
        quantized_model, output_dir, basename, 'ov_export:int8-nn')
    marked = _inject_rt_attribute(xml_path, scope['precise_names'], 'precise')

    int8_points, head_points, head_feed_points = (
        _nn_int8_quantization_point_summary(
            quantized_model, scope['head_names']))

    print('[ov_export:int8-nn] NNCF INT8 quantization points: {}'.format(
        int8_points))
    print('[ov_export:int8-nn] Head quantization points: named={}, directly_feeding={}'.format(
        head_points, head_feed_points))
    print('[ov_export:int8-nn] FP16 constants converted: {} constants, {} edges'.format(
        constants_converted, constant_edges))
    print('[ov_export:int8-nn] FP16 head converts: {} input, {} output'.format(
        fp16_inputs, fp16_outputs))
    print('[ov_export:int8-nn] Marked {} precise FP32 layers'.format(marked))
    return {'xml': xml_path, 'bin': bin_path, 'size_mb': size_mb,
            'ov_model': quantized_model,
            'precise_names': scope['precise_names'],
            'int8_quantization_points': int8_points}


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
                        help='Also export EXTNN model with pre/post processing without PnP')
    parser.add_argument('--e2e', action='store_true',
                        help='Also export E2E model with custom extension ops with PnP')
    parser.add_argument('--fp16_nn', action='store_true',
                        help='Also export NN mixed FP16 IR')
    parser.add_argument('--fp16_extnn', action='store_true',
                        help='Also export EXTNN mixed FP16 IR (auto-enables the EXTNN export)')
    parser.add_argument('--fp16_e2e', action='store_true',
                        help='Also export E2E mixed FP16 IR (auto-enables the E2E export)')
    parser.add_argument('--int8_nn', action='store_true',
                        help='Also export NN INT8 IR via NNCF PTQ')
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Dataset dir used by config and NNCF calibration')
    parser.add_argument('--int8_subset_size', type=int, default=300,
                        help='Calibration samples for NNCF PTQ (0 = all)')
    parser.add_argument('--int8_target_device', type=str, default='GPU',
                        choices=('CPU','GPU', 'NPU'),
                        help='NNCF PTQ target device (default: GPU)')
    parser.add_argument('--int8_preset', type=str, default='PERFORMANCE',
                        choices=('PERFORMANCE', 'MIXED'),
                        help='NNCF quantization preset (default: PERFORMANCE)')
    parser.add_argument('--extension', type=str, default=None,
                        help='Path to cdpn_extension.so')
    args = parser.parse_args()

    if args.extnn and args.e2e:
        parser.error('Export --extnn and --e2e in separate invocations; '
                     'each composite export rewires the converted NN body.')
    if args.int8_nn and not args.dataset_dir:
        parser.error('--dataset_dir is required for INT8 calibration')

    model, cfg = build_cdpn_model(args.cfg, args.load_model)

    # Always export the NN body first
    nn_results = export_to_ov_ir(model, args.output_dir, args.basename, args.verify)

    if args.fp16_nn:
        export_nn_mixed_fp16_ir(
            nn_results['xml'], args.output_dir,
            basename=args.basename + '_fp16')

    if args.int8_nn:
        export_nn_int8_ir(
            nn_results['xml'], args.output_dir, cfg,
            basename=args.basename + '_int8',
            subset_size=args.int8_subset_size,
            target_device=args.int8_target_device,
            preset=args.int8_preset,
            dataset_dir=args.dataset_dir)

    # Optionally export the EXTNN model
    if args.extnn or args.fp16_extnn:
        print()
        print('=' * 70)
        print('[ov_export] Building EXTNN model with pre/post processing ops without PnP ...')
        print('=' * 70)
        extnn_results = export_extnn_model(args.output_dir,
                          basename=args.basename + '_extnn',
                          extension_path=args.extension,
                          verify=args.verify,
                          nn_ov_model=nn_results.get('ov_model'))

        if args.fp16_extnn:
            export_extnn_mixed_fp16_ir(
                extnn_results['xml'], args.output_dir,
                basename=args.basename + '_extnn_fp16',
                extension_path=args.extension)

    # Optionally export the E2E model (requires ov_plugins extension at runtime)
    if args.e2e or args.fp16_e2e:
        print()
        print('=' * 70)
        print('[ov_export] Building E2E model with custom extension ops with PnP ...')
        print('=' * 70)
        e2e_results = export_e2e_model(args.output_dir,
                         basename=args.basename + '_e2e',
                         extension_path=args.extension,
                         verify=args.verify,
                         nn_ov_model=nn_results.get('ov_model'))

        if args.fp16_e2e:
            export_e2e_mixed_fp16_ir(
                e2e_results['xml'], args.output_dir,
                basename=args.basename + '_e2e_fp16',
                extension_path=args.extension)
