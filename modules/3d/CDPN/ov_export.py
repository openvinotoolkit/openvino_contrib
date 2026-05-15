#!/usr/bin/env python
"""
Export CDPN PyTorch model directly to OpenVINO IR (.xml/.bin).

Uses ov.convert_model() which traces the PyTorch model via torch.export
internally and converts to OV IR in one pass.

Produces:
  - cdpn_stage3.xml/.bin      (FP32 weights)  - NN body only
  - cdpn_stage3_extnn.xml/.bin  (with --extnn flag)      - Extended-NN model
                              includes pre/post processing custom ops
  - cdpn_stage3_e2e.xml/.bin    (with --e2e flag)        - End-to-End model
                              includes pre/post processing + PnP custom ops

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

# Python wrappers define the graph topology (op names, shapes, attributes)
# for the OpenVINO IR file.  They contain ZERO computation.
# has_evaluate() returns False -> no Python execution at inference.
#
# At inference time:
#   CPU -> cdpn_extension.so provides evaluate() in C++
#   GPU -> cdpn_custom_gpu_kernels.xml + .cl kernels (SimpleGPU)

class CdpnPreprocess(ov.Op):
    """image[H,W,3] u8 + bbox[4] -> tensor[1,3,256,256] + crop_meta[5]."""

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
        self.set_output_type(
            0, OVType.f32,
            ov.PartialShape([1, 3, self._inp_res, self._inp_res]))
         # Padded to 4D
        self.set_output_type(1, OVType.f32, ov.PartialShape([1, 1, 1, 5]))

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

class CdpnCoordDenorm(ov.Op):
    """coord_maps[4,64,64] + obj_extents[3] -> combined[4,64,64].
    Channels 0-2: denorm coords, channel 3: confidence (min-max normalised)."""

    def __init__(self, coord_maps, obj_extents):
        super().__init__(self)
        self.set_arguments([coord_maps, obj_extents])
        self.constructor_validate_and_infer_types()

    @classmethod
    def get_type_info_static(cls):
        return ov.DiscreteTypeInfo("CdpnCoordDenorm", "extension")

    def validate_and_infer_types(self):
        # Single output: [1, 4, 64, 64] - 4D
        self.set_output_type(0, OVType.f32, ov.PartialShape([1, 4, 64, 64]))

    def has_evaluate(self):
        return False

    def clone_with_new_inputs(self, new_args):
        return CdpnCoordDenorm(new_args[0], new_args[1])

class CdpnTransDecode(ov.Op):
    """pred_trans[3] + bbox_wh[2] + crop_meta[5] + cam_K[4] -> translation[3]."""

    def __init__(self, pred_trans, bbox_wh, crop_meta, cam_K, out_res=64):
        super().__init__(self)
        self._out_res = int(out_res)
        self.set_arguments([pred_trans, bbox_wh, crop_meta, cam_K])
        self.constructor_validate_and_infer_types()

    @classmethod
    def get_type_info_static(cls):
        return ov.DiscreteTypeInfo("CdpnTransDecode", "extension")

    def validate_and_infer_types(self):
        # Padded to 4D
        self.set_output_type(0, OVType.f32, ov.PartialShape([1, 1, 1, 3]))

    def visit_attributes(self, visitor):
        visitor.on_attributes({"out_res": self._out_res})
        return True

    def has_evaluate(self):
        return False

    def clone_with_new_inputs(self, new_args):
        return CdpnTransDecode(
            new_args[0], new_args[1], new_args[2], new_args[3],
            self._out_res)

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
        self.set_output_type(0, OVType.f32, ov.PartialShape([1, 1, 3, 3]))
        self.set_output_type(1, OVType.f32, ov.PartialShape([1, 1, 1, 3]))
        self.set_output_type(2, OVType.f32, ov.PartialShape([1, 1, 1, 1]))
        self.set_output_type(3, OVType.f32, ov.PartialShape([1, 1, 1, 1]))

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
                '--test',
                '--load_model', checkpoint_path,
                '--gpu', '-1']

    from config import config
    from model import build_model

    cfg = config().parse()
    cfg.pytorch.load_model = checkpoint_path
    model, _ = build_model(cfg)
    model.eval()
    return model, cfg


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
    for i, out in enumerate(ov_model.outputs):
        print('[ov_export] Output {}: names={}, shape={}'.format(
            i, out.get_names(), out.get_partial_shape()))

    results = {}
    results['ov_model'] = ov_model

    # Save FP32
    xml_path = os.path.join(output_dir, basename + '.xml')
    bin_path = os.path.join(output_dir, basename + '.bin')
    ov.save_model(ov_model, xml_path, compress_to_fp16=False)
    bin_size = os.path.getsize(bin_path) / 1e6
    print('[ov_export] FP32 IR: {} ({:.1f} MB)'.format(xml_path, bin_size))
    results['xml'] = xml_path
    results['bin'] = bin_path
    results['size_mb'] = bin_size

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
                      verify=False, nn_ov_model=None):
    """
    Export the EXTNN CDPN model - single XML/BIN with preprocessing,
    NN body, and postprocessing (everything except PnP).

    Graph:
      image[1,H,W,3] u8 + bbox[1,1,1,4] -> CdpnPreprocess
        -> tensor[1,3,256,256] + crop_meta[1,1,1,5]
      tensor -> NN body -> coord_maps[1,4,64,64] + raw_trans[1,3]
      coord_maps + obj_extents[1,1,1,3] -> CdpnCoordDenorm -> combined[1,4,64,64]
      combined -> VariadicSplit -> denorm_coords[1,3,64,64] + confidence[1,1,64,64]
      raw_trans + bbox_wh[1,1,1,2] + crop_meta + cam_K[1,1,1,4]
        -> CdpnTransDecode -> translation[1,1,1,3]

    Outputs:
      denorm_coords [1,3,64,64], confidence [1,1,64,64],
      translation [1,1,1,3], crop_meta [1,1,1,5]

    PnP stays in host code (ov_infer.py).
    Requires cdpn_extension.so at runtime (core.add_extension).

    All shapes are 4D.
    """
    os.makedirs(output_dir, exist_ok=True)

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
    param_image = opset.parameter(
        ov.PartialShape([1, -1, -1, 3]), OVType.u8, name='image')
    param_bbox = opset.parameter(
        ov.PartialShape([1, 1, 1, 4]), OVType.f32, name='bbox')
    param_extents = opset.parameter(
        ov.PartialShape([1, 1, 1, 3]), OVType.f32, name='obj_extents')
    param_bbox_wh = opset.parameter(
        ov.PartialShape([1, 1, 1, 2]), OVType.f32, name='bbox_wh')
    param_cam_K = opset.parameter(
        ov.PartialShape([1, 1, 1, 4]), OVType.f32, name='cam_K')

    # CdpnPreprocess: image + bbox -> tensor + crop_meta
    preprocess = CdpnPreprocess(
        param_image.output(0), param_bbox.output(0))
    tensor_out = preprocess.output(0)      # [1, 3, 256, 256]
    crop_meta_out = preprocess.output(1)   # [1, 1, 1, 5]

    # Rewire NN body input to receive CdpnPreprocess output
    nn_params = nn_model.get_parameters()
    nn_results = nn_model.get_results()
    nn_input_param = nn_params[0]
    for target_input in nn_input_param.output(0).get_target_inputs():
        target_input.replace_source_output(tensor_out)

    # NN body outputs
    coord_maps_node = nn_results[0].input(0).get_source_output()  # [1,4,64,64]
    raw_trans_node = nn_results[1].input(0).get_source_output()   # [1,3]

    # Reshape raw_trans [1,3] -> [1,1,1,3] for CdpnTransDecode
    raw_trans_4d = opset.reshape(
        raw_trans_node,
        opset.constant(np.array([1, 1, 1, 3], dtype=np.int64)), False)

    # CdpnCoordDenorm: coord_maps[1,4,64,64] + obj_extents[1,1,1,3]
    #                  -> combined[1,4,64,64]
    coord_denorm = CdpnCoordDenorm(
        coord_maps_node, param_extents.output(0))
    combined_4d = coord_denorm.output(0)  # [1, 4, 64, 64]

    # Split combined -> denorm_coords[1,3,64,64] + confidence[1,1,64,64]
    split_axis = opset.constant(np.int64(1))
    split_lengths = opset.constant(np.array([3, 1], dtype=np.int64))
    coord_split = opset.variadic_split(combined_4d, split_axis, split_lengths)
    denorm_coords_4d = coord_split.output(0)  # [1, 3, 64, 64]
    confidence_4d = coord_split.output(1)      # [1, 1, 64, 64]

    # CdpnTransDecode: raw_trans + bbox_wh + crop_meta + cam_K -> translation
    eps = opset.constant(np.float32(1e-30))
    crop_meta_for_trans = opset.add(crop_meta_out, eps)

    trans_decode = CdpnTransDecode(
        raw_trans_4d.output(0), param_bbox_wh.output(0),
        crop_meta_for_trans.output(0), param_cam_K.output(0))
    translation_4d = trans_decode.output(0)  # [1, 1, 1, 3]

    # Step 3: Assemble EXTNN model
    print('[ov_export:extnn] Step 3: Creating EXTNN model ...')

    extnn_model = ov.Model(
        results=[
            OVResult(denorm_coords_4d),
            OVResult(confidence_4d),
            OVResult(translation_4d),
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

    print('[ov_export:extnn]   Inputs:')
    for i, inp in enumerate(extnn_model.inputs):
        print('[ov_export:extnn]     {}: names={}, shape={}, type={}'.format(
            i, inp.get_names(), inp.get_partial_shape(),
            inp.get_element_type()))
    print('[ov_export:extnn]   Outputs:')
    for i, out in enumerate(extnn_model.outputs):
        print('[ov_export:extnn]     {}: names={}, shape={}'.format(
            i, out.get_names(), out.get_partial_shape()))

    # Step 4: Save EXTNN model
    xml_path = os.path.join(output_dir, basename + '.xml')
    bin_path = os.path.join(output_dir, basename + '.bin')
    ov.save_model(extnn_model, xml_path, compress_to_fp16=False)
    bin_size = os.path.getsize(bin_path) / 1e6
    print('[ov_export:extnn] EXTNN IR: {} ({:.1f} MB)'.format(
        xml_path, bin_size))

    results = {'xml': xml_path, 'bin': bin_path, 'size_mb': bin_size}

    # Step 5: Verification
    if verify:
        print()
        print('[ov_export:extnn] === Verification ===')
        ext_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'ov_plugins', 'build', 'cdpn_extension.so')
        if not os.path.isfile(ext_path):
            print('[ov_export:extnn] WARNING: Skipping verify - '
                  'cdpn_extension.so not found at {}'.format(ext_path))
        else:
            core = ov.Core()
            core.add_extension(ext_path)
            compiled = core.compile_model(xml_path, 'CPU')

            np.random.seed(42)
            test_image = np.random.randint(
                0, 255, (1, 480, 640, 3), dtype=np.uint8)
            test_bbox = np.array(
                [[[[100.0, 80.0, 120.0, 150.0]]]], dtype=np.float32)
            test_extents = np.array(
                [[[[0.05, 0.04, 0.06]]]], dtype=np.float32)
            test_bbox_wh = np.array(
                [[[[120.0, 150.0]]]], dtype=np.float32)
            test_cam_K = np.array(
                [[[[572.41, 573.57, 325.26, 242.05]]]], dtype=np.float32)

            ov_out = compiled({
                'image': test_image,
                'bbox': test_bbox,
                'obj_extents': test_extents,
                'bbox_wh': test_bbox_wh,
                'cam_K': test_cam_K,
            })

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
    output_dir : str
        Directory to write .xml/.bin files.
    basename : str
        Base filename (default: cdpn_stage3_e2e).
    extension_path : str or None
        Path to cdpn_cpu_extension.so (needed for verification only).
    verify : bool
        If True, run verification (requires extension .so).
    nn_ov_model : ov.Model or None
        Pre-converted NN body (reuse to avoid re-conversion).
    """
    os.makedirs(output_dir, exist_ok=True)

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
    param_image = opset.parameter(
        ov.PartialShape([-1, -1, 3]), OVType.u8, name='image')
    param_bbox = opset.parameter(
        ov.PartialShape([4]), OVType.f32, name='bbox')
    param_extents = opset.parameter(
        ov.PartialShape([3]), OVType.f32, name='obj_extents')
    param_bbox_wh = opset.parameter(
        ov.PartialShape([2]), OVType.f32, name='bbox_wh')
    param_cam_K = opset.parameter(
        ov.PartialShape([4]), OVType.f32, name='cam_K')

    # CdpnPreprocess: image + bbox -> tensor + crop_meta
    preprocess = CdpnPreprocess(
        param_image.output(0), param_bbox.output(0))
    tensor_out = preprocess.output(0)     # [1, 3, 256, 256]
    crop_meta_out = preprocess.output(1)  # [1, 1, 1, 5]  (4D)

    # Rewire NN body input to receive CdpnPreprocess output
    nn_params = nn_model.get_parameters()
    nn_results = nn_model.get_results()
    nn_input_param = nn_params[0]
    for target_input in nn_input_param.output(0).get_target_inputs():
        target_input.replace_source_output(tensor_out)

    # NN body outputs (follow Result->source pattern from EXTNN export)
    coord_maps_node = nn_results[0].input(0).get_source_output()
    raw_trans_node = nn_results[1].input(0).get_source_output()

    # Squeeze batch dimension: [1,3]->[3]
    axis_0 = opset.constant(np.int64(0))
    raw_trans_sq = opset.squeeze(raw_trans_node, axis_0)

    # CdpnCoordDenorm: coord_maps + obj_extents -> combined [1,4,64,64]
    # Keep coord_maps as 4D [1,4,64,64] - C++ validate_and_infer_types
    # propagates input shape, and downstream VariadicSplit needs axis=1
    # to be the channel dimension (size 4).
    coord_denorm = CdpnCoordDenorm(
        coord_maps_node, param_extents.output(0))
    combined_4d = coord_denorm.output(0)  # [1, 4, 64, 64]

    # Split combined output into denorm [1,3,64,64] and confidence [1,1,64,64]
    # Using standard VariadicSplit.
    split_axis = opset.constant(np.int64(1))
    split_lengths = opset.constant(np.array([3, 1], dtype=np.int64))
    coord_split = opset.variadic_split(combined_4d, split_axis, split_lengths)
    denorm_coords_4d = coord_split.output(0)  # [1, 3, 64, 64]
    confidence_4d = coord_split.output(1)      # [1, 1, 64, 64]

    # CdpnTransDecode: raw_trans + bbox_wh + crop_meta + cam_K -> translation
    eps = opset.constant(np.float32(1e-30))
    crop_meta_for_trans = opset.add(crop_meta_out, eps)
    crop_meta_for_pnp = opset.add(crop_meta_out, eps)

    trans_decode = CdpnTransDecode(
        raw_trans_sq.output(0), param_bbox_wh.output(0),
        crop_meta_for_trans.output(0), param_cam_K.output(0))
    translation_4d = trans_decode.output(0)  # [1, 1, 1, 3]

    # CdpnPnpSolve: denorm_coords + confidence + obj_ext + crop_meta + cam_K
    #               -> R[1,1,3,3] + T_pnp[1,1,1,3] + num_corres[1,1,1,1]
    #                 + pnp_success[1,1,1,1]
    denorm_for_pnp = opset.add(denorm_coords_4d, eps)
    conf_for_pnp = opset.add(confidence_4d, eps)

    pnp_solve = CdpnPnpSolve(
        denorm_for_pnp.output(0), conf_for_pnp.output(0),
        param_extents.output(0), crop_meta_for_pnp.output(0),
        param_cam_K.output(0))
    R_4d = pnp_solve.output(0)             # [1, 1, 3, 3]
    T_pnp_4d = pnp_solve.output(1)        # [1, 1, 1, 3]
    num_corres_4d = pnp_solve.output(2)    # [1, 1, 1, 1]
    pnp_success_4d = pnp_solve.output(3)   # [1, 1, 1, 1]

    # Squeeze 4D custom op outputs to original shapes.
    axes_012 = opset.constant(np.array([0, 1, 2], dtype=np.int64))
    axes_01 = opset.constant(np.array([0, 1], dtype=np.int64))

    R_out = opset.squeeze(R_4d, axes_01)                          # [3, 3]
    denorm_coords_out = opset.squeeze(denorm_coords_4d, axis_0)  # [3, 64, 64]
    confidence_out = opset.squeeze(confidence_4d, axes_01)        # [64, 64]
    num_corres_out = opset.squeeze(num_corres_4d, axes_012)       # [1]
    pnp_success_out = opset.squeeze(pnp_success_4d, axes_012)    # [1]

    # Pose composition using standard opset
    shape_3x1 = opset.constant(np.array([3, 1], dtype=np.int64))
    T_pnp_col = opset.reshape(T_pnp_4d, shape_3x1, False)         # [3,1]
    T_trans_col = opset.reshape(translation_4d, shape_3x1, False)  # [3,1]
    pose_rot_out = opset.concat([R_out, T_pnp_col], axis=1)    # [3,4]
    pose_trans_out = opset.concat([R_out, T_trans_col], axis=1)  # [3,4]

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

    print('[ov_export:e2e]   Inputs:')
    for i, inp in enumerate(e2e_model.inputs):
        print('[ov_export:e2e]     {}: names={}, shape={}, type={}'.format(
            i, inp.get_names(), inp.get_partial_shape(),
            inp.get_element_type()))
    print('[ov_export:e2e]   Outputs:')
    for i, out in enumerate(e2e_model.outputs):
        print('[ov_export:e2e]     {}: names={}, shape={}'.format(
            i, out.get_names(), out.get_partial_shape()))

    # Step 4: Save
    xml_path = os.path.join(output_dir, basename + '.xml')
    bin_path = os.path.join(output_dir, basename + '.bin')
    ov.save_model(e2e_model, xml_path, compress_to_fp16=False)
    bin_size = os.path.getsize(bin_path) / 1e6
    print('[ov_export:e2e] E2E IR: {} ({:.1f} MB)'.format(xml_path, bin_size))

    results = {'xml': xml_path, 'bin': bin_path, 'size_mb': bin_size}

    # Step 5: Verification (requires extension .so)
    if verify:
        if not extension_path or not os.path.isfile(extension_path):
            print('[ov_export:e2e] WARNING: Skipping verify -- '
                  'provide --extension /path/to/cdpn_cpu_extension.so')
        else:
            print()
            print('[ov_export:e2e] === Verification ===')
            core = ov.Core()
            core.add_extension(extension_path)
            compiled = core.compile_model(xml_path, 'CPU')

            np.random.seed(42)
            test_image = np.random.randint(
                0, 255, (480, 640, 3), dtype=np.uint8)
            test_bbox = np.array(
                [100.0, 80.0, 120.0, 150.0], dtype=np.float32)
            test_extents = np.array([0.05, 0.04, 0.06], dtype=np.float32)
            test_bbox_wh = np.array([120.0, 150.0], dtype=np.float32)
            test_cam_K = np.array(
                [572.41, 573.57, 325.26, 242.05], dtype=np.float32)

            ov_out = compiled({
                'image': test_image,
                'bbox': test_bbox,
                'obj_extents': test_extents,
                'bbox_wh': test_bbox_wh,
                'cam_K': test_cam_K,
            })

            for idx, name in enumerate(
                    ['pose_rot', 'pose_trans', 'denorm_coords',
                     'confidence', 'num_corres', 'pnp_success']):
                print('[ov_export:e2e]   {} shape: {}'.format(
                    name, ov_out[compiled.output(idx)].shape))
            print('[ov_export:e2e]   pnp_success: {}'.format(
                ov_out[compiled.output(5)]))

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
                        help='Path to cdpn_cpu_extension.so (for verification)')
    args = parser.parse_args()

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
