#!/usr/bin/env python
#
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""
ov_infer.py - Reusable CDPN OpenVINO Inference Class

End-to-end 6D pose estimation pipeline using OpenVINO

Supports three model types (auto-detected by input count):
    NN-only (1 input)  - preprocessed_tensor only
    EXTNN   (2 inputs) - preprocessed_tensor + obj_extents
    E2E     (5 inputs) - image + bbox + obj_extents + bbox_wh + cam_K

Supports both CPU and GPU execution:
  --cpu : Entire OV model runs on CPU
  --gpu : Entire OV model runs on GPU

The EXTNN model (cdpn_stage3_extnn.xml/bin) contains:
  - CDPN neural network body (ResNet34 backbone + RotHead + TransHead)
  - Coordinate map denormalisation (xyz * obj_extents)
  - Confidence min-max normalisation

The E2E model (cdpn_stage3_e2e.xml/bin) additionally contains:
  - CdpnPreprocess   - image crop + resize + normalise (fused custom op)
  - CdpnPnpSolve     - DLT PnP + RANSAC (custom op)
  - Pose composition - [R|T] concat via standard opset
  Full CPU or full GPU execution - zero host-side compute in the E2E path.

Usage:
    # As a library:
    from ov_infer import CdpnOVInference
    infer = CdpnOVInference(model_path, obj_info, device='GPU')
    results = infer(rgb_images, boxes, obj_names)

    # Standalone test (EXTNN model):
    python ov_infer.py --model checkpoints/cdpn_stage3_extnn.xml \
        --dataset_dir dataset/lm_full --gpu

    # Standalone test (E2E model with extension):
    python ov_infer.py --model checkpoints/cdpn_stage3_e2e.xml \
        --extension ov_plugins/build/cdpn_extension.so \
        --dataset_dir dataset/lm_full --gpu
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import argparse

import cv2
import numpy as np
import openvino as ov

# make lib/ importable
_cur = os.path.dirname(os.path.abspath(__file__))
_lib = os.path.join(_cur, 'lib')
if _lib not in sys.path:
    sys.path.insert(0, _lib)

import ref
from utils.img import zoom_in, xyxy_to_xywh


class CdpnOVInference:
    """
    Reusable CDPN 6-DoF Pose Inference using OpenVINO.

    Parameters
    ----------
    model_path : str
        Path to cdpn_stage3.xml.
    obj_info : dict
        Object 3D model metadata, keyed by integer object ID.
        Each value: {'diameter': float, 'min_x': float, 'min_y': float, 'min_z': float}.
    device : str
        OV device: 'CPU' or 'GPU'.
    camera_matrix : np.ndarray or None
        3x3 camera intrinsics. Defaults to LINEMOD K.
    extension_path : str or None
        Path to the CDPN extension .so (e.g. cdpn_extension.so).
        Required for E2E models.
    inference_precision : str
        GPU inference precision hint: 'f32', 'f16', or 'none'. Ignored on CPU.
    """

    def __init__(self, model_path, obj_info,
                 device='CPU', camera_matrix=None,
                 extension_path=None,
                 inference_precision='f32'):

        self.device = device
        self.obj_info = obj_info
        self.camera_matrix = camera_matrix if camera_matrix is not None else ref.K.copy()
        self.inp_res = 256
        self.out_res = 64
        self.pad_ratio = 1.5
        self.mask_threshold = 0.5
        self.uses_cdpn_custom_ops = self._model_uses_cdpn_custom_ops(model_path)

        # Compile model
        self.core = ov.Core()

        # Load CPU extension .so (for custom op evaluate())
        if self.uses_cdpn_custom_ops:
            ext_auto = os.path.join(_cur, 'ov_plugins', 'build', 'cdpn_extension.so')
            ext_to_load = extension_path if extension_path and os.path.isfile(extension_path) else ext_auto
            if os.path.isfile(ext_to_load):
                self.core.add_extension(ext_to_load)
                if extension_path and ext_to_load != extension_path:
                    print('[CdpnOVInference] Resolved extension: {} -> {}'.format(
                        extension_path, ext_to_load))
                else:
                    print('[CdpnOVInference] Loaded extension: {}'.format(ext_to_load))
            elif extension_path:
                print('[CdpnOVInference] WARNING: extension not found: {}'.format(
                    extension_path))


        config = {}

        if device == 'GPU' and self.uses_cdpn_custom_ops:
            gpu_config = os.path.join(
                _cur, 'ov_plugins', 'cdpn_gpu',
                'cdpn_custom_gpu_kernels.xml')
            if os.path.isfile(gpu_config):
                self.core.set_property('GPU', {'CONFIG_FILE': gpu_config})
                print('[CdpnOVInference] Loaded GPU custom kernels: {}'.format(
                    gpu_config))

        if device == 'GPU':
            precision_key = inference_precision.lower()
            if precision_key in ('f32', 'fp32'):
                config[ov.properties.hint.inference_precision()] = ov.Type.f32
                print('[CdpnOVInference] GPU: using FP32 inference precision')
            elif precision_key in ('f16', 'fp16'):
                config[ov.properties.hint.inference_precision()] = ov.Type.f16
                print('[CdpnOVInference] GPU: using FP16 inference precision')
            elif precision_key in ('none', 'auto'):
                print('[CdpnOVInference] GPU: using plugin default inference precision')
            else:
                raise ValueError('Unsupported GPU inference precision: {}'.format(
                    inference_precision))

        print('[CdpnOVInference] OpenVINO {} - device: {}'.format(
            ov.__version__, device))

        self.model = self.core.read_model(model_path)

        self.int8_quantization_points = sum(1 for op in self.model.get_ordered_ops()
            if op.get_type_name() == 'FakeQuantize')
        if self.int8_quantization_points:
            print('[CdpnOVInference] Model contains {} INT8 quantization points'.format(
                self.int8_quantization_points))

        self.compiled = self.core.compile_model(self.model, device, config)

        # One reusable request per instance; not thread-safe, use one instance per thread.
        self._infer_request = self.compiled.create_infer_request()

        # Detect model type by checking output names
        output_names = set()
        for out in self.compiled.outputs:
            output_names.update(out.get_names())

        n_inputs = len(self.compiled.inputs)
        if 'pose_rot' in output_names:
            print('[CdpnOVInference] Detected E2E model')
            self.model_type = 'e2e'
        elif 'crop_meta' in output_names and n_inputs >= 5:
            print('[CdpnOVInference] Detected EXTNN model')
            self.model_type = 'extnn'
        else:
            print('[CdpnOVInference] Detected NN-only model')
            self.model_type = 'nn_only'

        print('[CdpnOVInference] Model type: {} ({} inputs, {} outputs)'.format(
            self.model_type.upper(), n_inputs, len(self.compiled.outputs)))

        # Cache obj_extents per object ID
        self._obj_ext_cache = {}
        for obj_id, info in obj_info.items():
            self._obj_ext_cache[obj_id] = np.array([
                abs(info['min_x']),
                abs(info['min_y']),
                abs(info['min_z']),
            ], dtype=np.float32)

    # public API
    def __call__(self, rgb_list, box_list, obj_list):
        """
        Run end-to-end inference on one or more images.

        Parameters
        ----------
        rgb_list : list[np.ndarray]
            BGR images, each (H, W, 3) uint8 (OpenCV convention).
        box_list : list[tuple/np.ndarray]
            Bounding boxes in **xywh** format: (x, y, w, h).
        obj_list : list[str]
            Object class names, e.g. ['ape', 'can'].

        Returns
        -------
        results : list[dict] - one dict per input, with keys:
            'pose_rot'       : (3, 4) pose [R_pnp | T_pnp]
            'pose_trans'     : (3, 4) pose [R_pnp | T_trans]
            'R'              : (3, 3) rotation from PnP
            'T_pnp'          : (3,)   translation from PnP
            'T_trans'        : (3,)   translation from trans head
            'pred_coor'      : (3, 64, 64) denormalised coordinate maps
            'pred_conf'      : (64, 64) confidence map
            'num_corres'     : int, #correspondences sent to PnP
            'pnp_success'    : bool
        """
        assert len(rgb_list) == len(box_list) == len(obj_list)

        # E2E path: entire pipeline in the graph including PnP
        if self.model_type == 'e2e':
            return self._run_e2e(rgb_list, box_list, obj_list)

        # EXTNN path: preprocess + NN + postprocess in graph, PnP on host
        if self.model_type == 'extnn':
            return self._run_extnn(rgb_list, box_list, obj_list)

        # NN-only: batched forward through OV, host pre/post ---
        if self.model_type == 'nn_only':
            return self._run_nn_batched(rgb_list, box_list, obj_list)



    # internal helpers

    @staticmethod
    def _model_uses_cdpn_custom_ops(model_path):
        import xml.etree.ElementTree as ET
        custom_ops = {
            'CdpnPreprocess',
            'CdpnPnpSolve',
        }
        try:
            for _event, elem in ET.iterparse(model_path, events=('start',)):
                if elem.tag == 'layer' and elem.get('type') in custom_ops:
                    return True
        except (OSError, ET.ParseError):
            return False
        return False


    @staticmethod
    def _xywh_to_center_scale(box, pad_ratio, scale_max):
        """Convert xywh bounding box to (centre, scale) for cropping."""
        x, y, w, h = box
        crop_center = np.array((x + 0.5 * w, y + 0.5 * h))
        crop_scale = max(w, h) * pad_ratio
        crop_scale = min(crop_scale, scale_max)
        return crop_center, crop_scale


    def _preprocess_single(self, rgb, box):
        """
        Preprocess one image: crop + resize + normalise.

        Returns (inp, crop_center_out, crop_scale_, box)
        """
        box = np.asarray(box, dtype=np.float64)
        crop_center, crop_scale = self._xywh_to_center_scale(
            box, self.pad_ratio, scale_max=max(ref.im_w, ref.im_h))
        rgb_crop, c_h_, c_w_, crop_scale_ = zoom_in(
            rgb, crop_center, crop_scale, self.inp_res)
        inp = rgb_crop.transpose(2, 0, 1).astype(np.float32) / 255.0
        crop_center_out = np.array([c_w_, c_h_])
        return inp, crop_center_out, crop_scale_, box


    def _preprocess_batch(self, rgb_list, box_list):
        """Preprocess a batch."""
        inps, centers, scales, boxes = [], [], [], []
        for rgb, box in zip(rgb_list, box_list):
            inp, crop_center, crop_scale, box_arr = self._preprocess_single(rgb, box)
            inps.append(inp)
            centers.append(crop_center)
            scales.append(crop_scale)
            boxes.append(box_arr)
        inp_batch = np.stack(inps, axis=0)
        return inp_batch, centers, scales, boxes


    def _run_e2e(self, rgb_list, box_list, obj_list):
        """
        End-to-end inference path for E2E models (5 inputs).

        Preprocessing, NN body, coord denorm, conf norm, trans decode,
        and PnP solve are all inside the OV graph.
        """
        num_samples = len(rgb_list)

        if num_samples == 0:
            return []

        cam_K_arr = np.array([
            self.camera_matrix[0, 0], self.camera_matrix[1, 1],
            self.camera_matrix[0, 2], self.camera_matrix[1, 2],
        ], dtype=np.float32)

        # Pre-compute per-sample metadata
        boxes = []
        extents = []
        bbox_whs = []
        for i in range(num_samples):
            obj_id = ref.obj2idx(obj_list[i])
            box = np.asarray(box_list[i], dtype=np.float32)
            boxes.append(box)
            extents.append(self._obj_ext_cache[obj_id])
            bbox_whs.append(np.array([box[2], box[3]], dtype=np.float32))

        all_results = [None] * num_samples
        h, w = rgb_list[0].shape[:2]
        img_batch  = np.empty((num_samples, h, w, 3), dtype=np.uint8)
        bbox_batch = np.empty((num_samples, 1, 1, 4), dtype=np.float32)
        ext_batch  = np.empty((num_samples, 1, 1, 3), dtype=np.float32)
        bwh_batch  = np.empty((num_samples, 1, 1, 2), dtype=np.float32)
        camK_batch = np.empty((num_samples, 1, 1, 4), dtype=np.float32)

        for idx in range(num_samples):
            img_batch[idx]        = rgb_list[idx]
            bbox_batch[idx, 0, 0] = boxes[idx]
            ext_batch[idx, 0, 0]  = extents[idx]
            bwh_batch[idx, 0, 0]  = bbox_whs[idx]
            camK_batch[idx, 0, 0] = cam_K_arr

        ov_out = self._infer_request.infer({
            'image': img_batch,
            'bbox': bbox_batch,
            'obj_extents': ext_batch,
            'bbox_wh': bwh_batch,
            'cam_K': camK_batch,
        })

        pose_rot_all      = ov_out[self.compiled.output(0)]
        pose_trans_all    = ov_out[self.compiled.output(1)]
        denorm_coords_all = ov_out[self.compiled.output(2)]
        confidence_all    = ov_out[self.compiled.output(3)]
        num_corres_all    = ov_out[self.compiled.output(4)]
        pnp_success_all   = ov_out[self.compiled.output(5)]

        for idx in range(num_samples):
            pose_rot = pose_rot_all[idx]
            pose_trans = pose_trans_all[idx]
            all_results[idx] = {
                'pose_rot': pose_rot,
                'pose_trans': pose_trans,
                'R': pose_rot[:, :3],
                'T_pnp': pose_rot[:, 3],
                'T_trans': pose_trans[:, 3],
                'pred_coor': denorm_coords_all[idx],
                'pred_conf': confidence_all[idx],
                'num_corres': int(num_corres_all[idx, 0]),
                'pnp_success': bool(pnp_success_all[idx, 0] > 0.5),
            }

        return all_results


    def _run_extnn(self, rgb_list, box_list, obj_list):
        """
        EXTNN inference path: preprocess + NN + postprocess in graph,
        PnP on host.

        Graph inputs:
          image [N,H,W,3] u8, bbox [N,1,1,4] f32, obj_extents [N,1,1,3] f32,
          bbox_wh [N,1,1,2] f32, cam_K [N,1,1,4] f32

        Graph outputs:
          denorm_coords [N,3,64,64], confidence [N,1,64,64],
          translation [N,1,1,3], crop_meta [N,1,1,5]

        Host-side: PnP RANSAC using denorm_coords + confidence + crop_meta.
        """
        num_samples = len(rgb_list)
        if num_samples == 0:
            return []

        cam_K_arr = np.array([
            self.camera_matrix[0, 0], self.camera_matrix[1, 1],
            self.camera_matrix[0, 2], self.camera_matrix[1, 2],
        ], dtype=np.float32)

        # Pre-compute per-sample metadata
        boxes = []
        extents = []
        bbox_whs = []
        obj_ext_list = []
        for i in range(num_samples):
            obj_id = ref.obj2idx(obj_list[i])
            box = np.asarray(box_list[i], dtype=np.float32)
            boxes.append(box)
            extents.append(self._obj_ext_cache[obj_id])
            bbox_whs.append(np.array([box[2], box[3]], dtype=np.float32))
            obj_ext_list.append(self._obj_ext_cache[obj_id])

        all_results = [None] * num_samples
        h, w = rgb_list[0].shape[:2]
        img_batch  = np.empty((num_samples, h, w, 3), dtype=np.uint8)
        bbox_batch = np.empty((num_samples, 1, 1, 4), dtype=np.float32)
        ext_batch  = np.empty((num_samples, 1, 1, 3), dtype=np.float32)
        bwh_batch  = np.empty((num_samples, 1, 1, 2), dtype=np.float32)
        camK_batch = np.empty((num_samples, 1, 1, 4), dtype=np.float32)
        for idx in range(num_samples):
            img_batch[idx] = rgb_list[idx]
            bbox_batch[idx, 0, 0] = boxes[idx]
            ext_batch[idx, 0, 0] = extents[idx]
            bwh_batch[idx, 0, 0] = bbox_whs[idx]
            camK_batch[idx, 0, 0] = cam_K_arr

        ov_out = self._infer_request.infer({
            'image': img_batch,
            'bbox': bbox_batch,
            'obj_extents': ext_batch,
            'bbox_wh': bwh_batch,
            'cam_K': camK_batch,
        })
        denorm_coords_all = ov_out[self.compiled.output(0)]
        confidence_all    = ov_out[self.compiled.output(1)]
        translation_all   = ov_out[self.compiled.output(2)]
        crop_meta_all     = ov_out[self.compiled.output(3)]

        # Per-sample PnP + pose composition (host)
        for idx in range(num_samples):
            denorm_coords   = denorm_coords_all[idx]       # [3, 64, 64]
            confidence      = confidence_all[idx, 0]       # [64, 64]
            T_trans         = translation_all[idx, 0, 0]   # [3]
            crop_meta       = crop_meta_all[idx, 0, 0]     # [5]

            c_w = crop_meta[0]
            c_h = crop_meta[1]
            s = crop_meta[2]
            w_begin = crop_meta[3]
            h_begin = crop_meta[4]

            R_pnp, T_pnp, num_corres, pnp_ok = self._pnp_solve(
                denorm_coords, confidence, obj_ext_list[idx],
                np.array([c_w, c_h]), float(s),
                w_begin=float(w_begin), h_begin=float(h_begin))

            pose_rot = np.concatenate(
                [R_pnp, T_pnp.reshape(3, 1)], axis=1)
            pose_trans = np.concatenate(
                [R_pnp, T_trans.reshape(3, 1)], axis=1)

            all_results[idx] = {
                'pose_rot': pose_rot,
                'pose_trans': pose_trans,
                'R': R_pnp,
                'T_pnp': T_pnp,
                'T_trans': T_trans,
                'pred_coor': denorm_coords,
                'pred_conf': confidence,
                'num_corres': num_corres,
                'pnp_success': pnp_ok,
            }

        return all_results


    def _run_nn_batched(self, rgb_list, box_list, obj_list):
        """
        NN-only batched inference: host preprocess -> single batched OV
        forward -> host postprocess (coord denorm, trans decode, PnP).

        The NN model has a single input [N, 3, 256, 256] and two outputs:
          coord_maps [N, 4, 64, 64]  and  translation [N, 3].
        Dynamic batch is supported by the exported IR.
        """
        n = len(rgb_list)

        # Step 1: Preprocess
        inp_batch, c_batch, s_batch, box_batch = self._preprocess_batch(
            rgb_list, box_list)

        # Step 2: Single batched OV forward
        ov_res = self._infer_request.infer(inp_batch)

        raw_rot_all = ov_res[0]
        raw_trans_all = ov_res[1]

        # Step 3: Per-sample postprocess (host CPU)
        results = []
        for i in range(n):
            obj_id = ref.obj2idx(obj_list[i])
            obj_ext = self._obj_ext_cache[obj_id]

            raw_rot = raw_rot_all[i]       # [4, 64, 64]
            raw_trans = raw_trans_all[i]   # [3]

            # Coord denorm: channels 0-2 × obj extents
            denorm_coords = raw_rot[:3] * obj_ext[:, np.newaxis, np.newaxis]

            # Confidence: min-max normalise channel 3
            conf = raw_rot[3]
            cmin = conf.min()
            crange = conf.max() - cmin
            if crange < 1e-8:
                crange = 1.0
            confidence = (conf - cmin) / crange

            # Translation decode
            T_trans = self._trans_decode(
                raw_trans, box_batch[i], c_batch[i], s_batch[i])

            # PnP solve (cv2 C++ backed)
            R_pnp, T_pnp, num_corres, pnp_ok = self._pnp_solve(
                denorm_coords, confidence, obj_ext,
                c_batch[i], s_batch[i])

            # Compose pose matrices
            pose_rot = np.concatenate(
                [R_pnp, T_pnp.reshape(3, 1)], axis=1)
            pose_trans = np.concatenate(
                [R_pnp, T_trans.reshape(3, 1)], axis=1)

            results.append({
                'pose_rot': pose_rot,
                'pose_trans': pose_trans,
                'R': R_pnp,
                'T_pnp': T_pnp,
                'T_trans': T_trans,
                'pred_coor': denorm_coords,
                'pred_conf': confidence,
                'num_corres': num_corres,
                'pnp_success': pnp_ok,
            })

        return results


    def _trans_decode(self, raw_trans, box, crop_center, crop_scale):
        """
        Decode raw translation head output to absolute translation.

        Parameters
        ----------
        raw_trans    : (3,) - [ratio_delta_cx, ratio_delta_cy, ratio_depth]
        box          : (4,) - xywh bounding box
        crop_center  : (2,) - [c_w, c_h] crop centre from zoom_in
        crop_scale   : float - crop scale from zoom_in (int-truncated)

        Returns
        -------
        T_trans : (3,) - [tx, ty, tz] in metres
        """
        ratio_delta_c = raw_trans[:2]
        ratio_depth = raw_trans[2]
        pred_depth = ratio_depth * (self.out_res / crop_scale)
        pred_c = ratio_delta_c * box[2:] + crop_center
        pred_x = (pred_c[0] - self.camera_matrix[0, 2]) * pred_depth / self.camera_matrix[0, 0]
        pred_y = (pred_c[1] - self.camera_matrix[1, 2]) * pred_depth / self.camera_matrix[1, 1]
        return np.array([pred_x, pred_y, pred_depth])

    def _pnp_solve(self, denorm_coords, confidence, obj_ext,
                   crop_center, crop_scale,
                   w_begin=None, h_begin=None):
        """
        Build 2D-3D correspondences and solve PnP.
        Uses numpy vectorised operations and cv2.solvePnPRansac.

        Parameters
        ----------
        denorm_coords : (3, 64, 64)  - denormalised 3D coordinates
        confidence    : (64, 64)     - normalised confidence map
        obj_ext       : (3,)         - |min_x|, |min_y|, |min_z|
        crop_center   : (2,)         - crop centre [c_w, c_h]
        crop_scale    : float        - crop scale
        w_begin       : float or None - crop left edge (E2E: from crop_meta)
        h_begin       : float or None - crop top edge  (E2E: from crop_meta)

        Returns
        -------
        R     : (3, 3) rotation matrix
        T     : (3,)   translation vector
        n_pts : int    number of correspondences
        ok    : bool   PnP success
        """
        R = np.eye(3)
        T = np.zeros(3)

        # Compute crop-region edges
        if w_begin is None:
            # int-truncate
            c_w = int(crop_center[0])
            c_h = int(crop_center[1])
            w_begin = c_w - crop_scale / 2.0
            h_begin = c_h - crop_scale / 2.0
        # w_begin/h_begin
        w_unit = crop_scale / self.out_res
        h_unit = crop_scale / self.out_res

        # Near-zero thresholds (0.1% of object extent)
        min_thresh = 0.001 * obj_ext   # [3]

        # Vectorised correspondence building
        # Confidence mask
        conf_mask = confidence >= self.mask_threshold   # [64, 64]

        # Transpose coords for spatial indexing: [64, 64, 3]
        coords_hwc = denorm_coords.transpose(1, 2, 0)  # [64, 64, 3]

        # Near-zero mask: all 3 channels must exceed threshold
        near_zero = (np.abs(coords_hwc) < min_thresh[np.newaxis, np.newaxis, :]).all(axis=2)
        valid_mask = conf_mask & ~near_zero              # [64, 64]

        # Get valid indices
        iy, ix = np.where(valid_mask)
        n_pts = len(iy)

        if n_pts < 4:
            return R, T, n_pts, False

        # 3D points from denormalised coords
        pts_3d = coords_hwc[iy, ix]   # [N, 3]

        # 2D points from grid position + crop metadata
        pts_2d = np.column_stack([
            w_begin + ix.astype(np.float64) * w_unit,
            h_begin + iy.astype(np.float64) * h_unit,
        ])   # [N, 2]

        # PnP RANSAC
        try:
            dist_coeffs = np.zeros((4, 1))
            _, R_vec, T_vec, _ = cv2.solvePnPRansac(
                pts_3d.astype(np.float32),
                pts_2d.astype(np.float32),
                self.camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP)
            R = cv2.Rodrigues(R_vec, jacobian=0)[0]
            T = np.asarray(T_vec).reshape(3)
            return R, T, n_pts, True
        except Exception:
            return R, T, n_pts, False


# Standalone test
def _collect_test_samples(dataset_dir, obj_names, max_samples=0):
    """
    Collect test sample metadata from a CDPN-format dataset directory.
    Images are NOT loaded here (lazy).
    """
    from glob import glob
    samples = []
    for obj in obj_names:
        test_dir = os.path.join(dataset_dir, 'real_test', obj)
        if not os.path.isdir(test_dir):
            print('WARNING: test dir not found: {}'.format(test_dir))
            continue
        rgb_paths = sorted(glob(os.path.join(test_dir, '*-color.png')))
        count = 0
        for rgb_pth in rgb_paths:
            if 0 < max_samples <= count:
                break
            box_xyxy = np.loadtxt(
                rgb_pth.replace('-color.png', '-box_fasterrcnn.txt'))
            samples.append({
                'box_xywh': np.array(xyxy_to_xywh(box_xyxy)),
                'pose_gt': np.loadtxt(
                    rgb_pth.replace('-color.png', '-pose.txt')),
                'obj_name': obj,
                'rgb_path': rgb_pth,
            })
            count += 1
    return samples


def _load_batch(samples):
    """Read images for a batch of samples (lazy loading)."""
    return [cv2.imread(sample['rgb_path']) for sample in samples]


def main():
    """
    Batch-driven test harness for CdpnOVInference.
    """
    from datasets.lm import LM
    from utils.eval import calc_rt_dist_m

    parser = argparse.ArgumentParser(description='CDPN OV Inference Test')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to xml')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Dataset dir with models/ and real_test/')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--cpu', action='store_true',
                        help='Run on CPU')
    parser.add_argument('--gpu', action='store_true',
                        help='Run on GPU (Intel Arc)')
    parser.add_argument('--obj_name', type=str, default='all',
                        help='Object name(s), comma-separated (default: all)')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Max samples per object (0 = all)')
    parser.add_argument('--extension', type=str, default=None,
                        help='Path to CDPN extension .so')
    parser.add_argument('--infer_precision', type=str, default='f32',
                        choices=('f32', 'f16', 'none'),
                        help='GPU inference precision hint (default: f32)')
    parser.add_argument('--int8_nn', action='store_true', default=False,
                        help='Require an INT8 NN-only model')
    args = parser.parse_args()

    if args.gpu:
        device = 'GPU'
    elif args.cpu:
        device = 'CPU'
    else:
        device = 'CPU'
        print('WARNING: No device specified, defaulting to CPU')

    print('=' * 70)
    print('CdpnOVInference - Batched Test Harness')
    print('=' * 70)

    # 1. Load obj_info from dataset
    info_path = os.path.join(args.dataset_dir, 'models', 'models_info.txt')
    obj_info = LM.load_lm_model_info(info_path)

    # 2. Initialise inference class
    infer = CdpnOVInference(
        model_path=args.model,
        obj_info=obj_info,
        device=device,
        extension_path=args.extension,
        inference_precision=args.infer_precision,
    )

    if args.int8_nn:
        if infer.model_type != 'nn_only':
            raise RuntimeError('--int8_nn expects a NN model, detected {}'.format(
                infer.model_type))

        # An INT8 NN-only export carries a fixed number of INT8 quantization
        # points. The model matches only when its count equals that number.
        EXPECTED_INT8_QUANTIZATION_POINTS = 57
        if infer.int8_quantization_points != EXPECTED_INT8_QUANTIZATION_POINTS:
            raise RuntimeError(
                '--int8_nn expects an INT8 model with {} quantization points, '
                'found {}. The model does not match the expected INT8 export.'.format(
                    EXPECTED_INT8_QUANTIZATION_POINTS, infer.int8_quantization_points))

        print('INT8 NN accuracy check: {} INT8 quantization points'.format(
            infer.int8_quantization_points))

    # 3. Determine which objects to test
    if args.obj_name and args.obj_name.lower() != 'all':
        obj_names = [o.strip() for o in args.obj_name.split(',')]
    else:
        obj_names = list(ref.lm_obj)
    print('Objects: {}'.format(obj_names))
    print('Batch size: {}'.format(args.batch_size))

    # 4. Collect test samples
    print('\nCollecting test samples from: {}'.format(args.dataset_dir))
    samples = _collect_test_samples(
        args.dataset_dir, obj_names, max_samples=args.max_samples)
    print('Total samples: {}'.format(len(samples)))

    # 5. Run inference in batches
    all_results = []
    num_samples = len(samples)
    batch_size = args.batch_size

    for start in range(0, num_samples, batch_size):
        batch = samples[start:start + batch_size]
        rgb_list = _load_batch(batch)
        box_list = [sample['box_xywh'] for sample in batch]
        obj_list = [sample['obj_name'] for sample in batch]

        results = infer(rgb_list, box_list, obj_list)

        for sample, res in zip(batch, results):
            res['pose_gt'] = sample['pose_gt']
            res['obj_name'] = sample['obj_name']
            res['rgb_path'] = sample['rgb_path']
            all_results.append(res)

        done = min(start + batch_size, num_samples)
        if done % (batch_size * 50) == 0 or done == num_samples:
            print('  [{}/{}]'.format(done, num_samples))

    # 6. Load PLY models for ADD / Proj. 2D metrics
    from utils.io import load_ply_vtx
    obj_vtx = {}
    for obj in obj_names:
        ply_path = os.path.join(
            args.dataset_dir, 'models', obj, '{}.ply'.format(obj))
        if os.path.isfile(ply_path):
            obj_vtx[obj] = load_ply_vtx(ply_path)
    camera_matrix = infer.camera_matrix
    sym_objs = {'eggbox', 'glue'}

    # 7. Compute per-object metrics
    print('\n' + '=' * 78)
    print('Results  (device={}, batch_size={}, {} samples)'.format(
        device, batch_size, num_samples))
    print('-' * 78)

    # Collect per-sample metrics
    obj_metrics = {}
    for res in all_results:
        obj = res['obj_name']
        obj_metrics.setdefault(obj, [])
        pose_est = res['pose_trans']
        pose_gt = res['pose_gt']
        R_est, T_est = pose_est[:, :3], pose_est[:, 3]
        R_gt, T_gt = pose_gt[:, :3], pose_gt[:, 3]

        r_err, t_err = calc_rt_dist_m(pose_est, pose_gt)

        add_dist = np.inf
        if obj in obj_vtx:
            vtx = obj_vtx[obj]
            vtx_est = (R_est @ vtx.T + T_est.reshape(3, 1)).T
            vtx_gt = (R_gt @ vtx.T + T_gt.reshape(3, 1)).T
            if obj in sym_objs:
                from scipy.spatial import cKDTree
                tree = cKDTree(vtx_gt)
                dists, _ = tree.query(vtx_est, k=1)
                add_dist = np.mean(dists)
            else:
                add_dist = np.mean(np.linalg.norm(vtx_est - vtx_gt, axis=1))

        proj2d_dist = np.inf
        if obj in obj_vtx:
            vtx = obj_vtx[obj]
            proj_est = camera_matrix @ (R_est @ vtx.T + T_est.reshape(3, 1))
            proj_est = proj_est[:2] / proj_est[2:3]
            proj_gt = camera_matrix @ (R_gt @ vtx.T + T_gt.reshape(3, 1))
            proj_gt = proj_gt[:2] / proj_gt[2:3]
            proj2d_dist = np.mean(np.linalg.norm(
                proj_est - proj_gt, axis=0))

        obj_metrics[obj].append({
            'r_err': r_err, 't_err': t_err,
            'add': add_dist, 'proj2d': proj2d_dist,
            'pnp_ok': res['pnp_success'],
        })

    # 8. Print summary table
    obj_diam = {}
    for obj in obj_names:
        oid = ref.obj2idx(obj)
        if oid in obj_info:
            obj_diam[obj] = obj_info[oid]['diameter']

    print('\n{:<14s} {:>6s} {:>6s} {:>8s} {:>8s} {:>10s} {:>8s} {:>8s}'.format(
        'Object', 'Count', 'PnP%', '5°5cm', 'ADD', 'Proj. 2D',
        'MedR(°)', 'MedT(m)'))
    print('-' * 78)

    sum_5d5cm, sum_add, sum_proj = 0.0, 0.0, 0.0
    n_obj = 0
    for obj in obj_names:
        if obj not in obj_metrics:
            continue
        metrics = obj_metrics[obj]
        count = len(metrics)
        pnp_pct = 100.0 * sum(1 for m in metrics if m['pnp_ok']) / count
        acc_5_5 = 100.0 * sum(
            1 for m in metrics if m['r_err'] < 5 and m['t_err'] < 0.05) / count
        diam = obj_diam.get(obj, 1.0)
        thresh_add = 0.10 * diam
        acc_add = 100.0 * sum(
            1 for m in metrics if m['add'] < thresh_add) / count
        acc_proj = 100.0 * sum(
            1 for m in metrics if m['proj2d'] < 5.0) / count
        med_r = np.median([m['r_err'] for m in metrics])
        med_t = np.median([m['t_err'] for m in metrics])

        print('{:<14s} {:>6d} {:>5.1f}% {:>7.2f}% {:>7.2f}% {:>9.2f}% {:>7.2f} {:>8.4f}'.format(
            obj, count, pnp_pct, acc_5_5, acc_add, acc_proj, med_r, med_t))
        sum_5d5cm += acc_5_5
        sum_add += acc_add
        sum_proj += acc_proj
        n_obj += 1

    if n_obj > 0:
        total_count = sum(
            len(obj_metrics[o]) for o in obj_names if o in obj_metrics)
        all_metrics = [m for o in obj_names if o in obj_metrics
                  for m in obj_metrics[o]]
        pnp_pct_all = 100.0 * sum(1 for m in all_metrics if m['pnp_ok']) / len(all_metrics)
        med_r_all = np.median([m['r_err'] for m in all_metrics])
        med_t_all = np.median([m['t_err'] for m in all_metrics])
        print('-' * 78)
        print('{:<14s} {:>6d} {:>5.1f}% {:>7.2f}% {:>7.2f}% {:>9.2f}% {:>7.2f} {:>8.4f}'.format(
            '**Average**', total_count, pnp_pct_all,
            sum_5d5cm / n_obj, sum_add / n_obj, sum_proj / n_obj,
            med_r_all, med_t_all))

    print('=' * 70)


if __name__ == '__main__':
    main()
