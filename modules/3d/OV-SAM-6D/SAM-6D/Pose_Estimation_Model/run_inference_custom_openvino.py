# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys
import time
import json
import random

import numpy as np
import cv2

import torch
from openvino import Core
import openvino as ov

import trimesh
from common_infer_utils import (
    _get_template as shared_get_template,
    get_templates as shared_get_templates,
    get_test_data as shared_get_test_data,
    load_yaml_config,
    visualize as shared_visualize,
)

from eval_utils import evaluate_and_print_ar, load_symmetries

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..', 'Pose_Estimation_Model')
sys.path.append(os.path.join(ROOT_DIR, 'provider'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'model', 'pointnet2'))


# ============================================================================
# FP16 Precision Strategy
# ============================================================================


def _collect_pem_feature_cut_outputs(model):
    cut_names = {
        "/feature_extraction/Add",
        "/feature_extraction/Div",
        "/feature_extraction/GatherElements",
        "/Transpose_1",
        "/GatherOperation_1",
        "/feature_extraction/Div_1",
        "/GatherOperation_3",
        "/Transpose_4",
    }
    return [
        op.output(0)
        for op in model.get_ordered_ops()
        if op.get_friendly_name() in cut_names
    ]


def _build_pem_feature_split(core, ov_pem_model_path):
    import openvino.opset13 as opset13

    stage1_model = core.read_model(ov_pem_model_path)
    cut_outputs = _collect_pem_feature_cut_outputs(stage1_model)
    if not cut_outputs:
        raise RuntimeError("No PEM feature-extraction cut outputs found")

    stage1_model = ov.Model(
        cut_outputs,
        stage1_model.get_parameters(),
        "pem_feature_extraction_fp16",
    )

    stage2_model = core.read_model(ov_pem_model_path)
    cut_names = {
        output.get_node().get_friendly_name()
        for output in _collect_pem_feature_cut_outputs(stage2_model)
    }
    cut_params = []
    cut_param_names = []
    for op in stage2_model.get_ordered_ops():
        if op.get_friendly_name() not in cut_names:
            continue
        output = op.output(0)
        targets = list(output.get_target_inputs())
        if not targets:
            continue

        param_name = f"pem_feature_cut_{len(cut_param_names)}"
        param = opset13.parameter(
            output.get_partial_shape(),
            output.get_element_type(),
            name=param_name,
        )
        param.set_friendly_name(param_name)
        cut_params.append(param)
        cut_param_names.append(param_name)
        for target in targets:
            target.replace_source_output(param.output(0))

    stage2_model = ov.Model(
        stage2_model.outputs,
        list(stage2_model.get_parameters()) + cut_params,
        "pem_matching_fp32",
    )
    return stage1_model, stage2_model, cut_param_names


class _SplitPemCompiled:
    def __init__(self, feature_compiled, matching_compiled, cut_param_names):
        self.feature_compiled = feature_compiled
        self.matching_compiled = matching_compiled
        self.cut_param_names = cut_param_names
        self.outputs = matching_compiled.outputs
        self.matching_input_names = {
            input_port.get_any_name() for input_port in matching_compiled.inputs
        }

    def __call__(self, inputs):
        feature_results = self.feature_compiled(inputs)
        feature_outputs = [
            feature_results[output_port]
            for output_port in self.feature_compiled.outputs
        ]
        if len(feature_outputs) != len(self.cut_param_names):
            raise RuntimeError(
                f"PEM split output mismatch: got {len(feature_outputs)} feature outputs "
                f"for {len(self.cut_param_names)} cut inputs"
            )

        matching_inputs = dict(inputs)
        for name, value in zip(self.cut_param_names, feature_outputs):
            if name in self.matching_input_names:
                matching_inputs[name] = np.asarray(value, dtype=np.float32)
        return self.matching_compiled(matching_inputs)


def _squeeze_bfyx(arr):
    """Remove a trailing dim-1 added by the GPU BFYX format for 3-D custom-op outputs.

    The GatherOperation GPU kernel allocates its output in BFYX (4-D) format.
    For a logically 3-D tensor (B, C, N) the physical buffer is (B, C, N, 1).
    After in-graph Transpose the shape becomes (B, N, C, 1).  This helper
    strips that spurious trailing 1 so downstream code always sees 3-D arrays.
    CPU inference is unaffected because it never adds the extra dimension.
    """
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    return arr


def _ordered_outputs(results, compiled_model):
    return [results[output_port] for output_port in compiled_model.outputs]


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Shared inference logic (same as run_inference_custom_openvino.py)
# ============================================================================

def get_parser():
    parser = argparse.ArgumentParser(
        description="[OpenVINO] Pose Estimation with multiple precision strategies")
    parser.add_argument("--device", type=str, default="GPU", help="device (CPU/GPU)")
    parser.add_argument("--precision", type=str, default="fp16",
                        choices=["fp32", "fp16"],
                        help="Precision strategy: "
                             "fp32=baseline, "
                             "fp16=FE and PEM feature extraction in FP16, "
                             "PEM matching/SVD in FP32")
    parser.add_argument("--model", type=str, default="pose_estimation_model")
    parser.add_argument("--config", type=str, default="config/base.yaml")
    parser.add_argument("--iter", type=int, default=600000)
    parser.add_argument("--exp_id", type=int, default=0)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    default_cad_path = os.path.join(project_root, 'SAM-6D/Data/Example/obj_000005.ply')
    default_rgb_path = os.path.join(project_root, 'SAM-6D/Data/Example/rgb.png')
    default_depth_path = os.path.join(project_root, 'SAM-6D/Data/Example/depth.png')
    default_camera_path = os.path.join(project_root, 'SAM-6D/Data/Example/camera.json')
    default_output_dir = os.path.join(project_root, 'SAM-6D/Data/Example/outputs')
    default_seg_path = os.path.join(default_output_dir, 'sam6d_results/detection_ism.json')
    default_gt_path = os.path.join(project_root, 'SAM-6D/Data/Example/gt_pose.json')
    default_models_info_path = os.path.join(project_root, 'SAM-6D/Data/Example/models_info.json')

    parser.add_argument("--output_dir", nargs="?", default=default_output_dir)
    parser.add_argument("--cad_path", nargs="?", default=default_cad_path)
    parser.add_argument("--rgb_path", nargs="?", default=default_rgb_path)
    parser.add_argument("--depth_path", nargs="?", default=default_depth_path)
    parser.add_argument("--cam_path", nargs="?", default=default_camera_path)
    parser.add_argument("--seg_path", nargs="?", default=default_seg_path)
    parser.add_argument("--det_score_thresh", type=float, default=0.2)
    parser.add_argument("--max_batch_size", type=int, default=3)
    parser.add_argument("--topk_ism_score", type=int, default=3)
    parser.add_argument("--gt_path", type=str, default=default_gt_path)
    parser.add_argument("--models_info_path", type=str, default=default_models_info_path)
    parser.add_argument("--seed", type=int, default=1,
                        help="RNG seed for deterministic template/model/observed point sampling "
                             "(default: rd_seed from config)")
    parser.add_argument("--skip_vsd", action="store_true")
    return parser.parse_args()


def init():
    args = get_parser()
    cfg = load_yaml_config(args.config)
    cfg.device = args.device
    cfg.precision = args.precision
    cfg.output_dir = args.output_dir
    cfg.cad_path = args.cad_path
    cfg.rgb_path = args.rgb_path
    cfg.depth_path = args.depth_path
    cfg.cam_path = args.cam_path
    cfg.seg_path = args.seg_path
    cfg.det_score_thresh = args.det_score_thresh
    cfg.max_batch_size = args.max_batch_size
    cfg.topk_ism_score = args.topk_ism_score
    cfg.gt_path = args.gt_path if args.gt_path else None
    cfg.models_info_path = args.models_info_path
    cfg.seed = args.seed if args.seed is not None else getattr(cfg, "rd_seed", 0)
    cfg.skip_vsd = args.skip_vsd
    return cfg


def visualize(rgb, pred_rot, pred_trans, model_points, K, save_path):
    return shared_visualize(rgb, pred_rot, pred_trans, model_points, K, save_path)


def _get_template_np(path, cfg, tem_index=1):
    return shared_get_template(path, cfg, device=None, tem_index=tem_index, backend="numpy")


def get_templates_np(path, cfg):
    return shared_get_templates(path, cfg, backend="numpy")


def get_test_data_np(rgb_path, depth_path, cam_path, cad_path, seg_path, det_score_thresh, cfg, topk):
    return shared_get_test_data(
        rgb_path,
        depth_path,
        cam_path,
        cad_path,
        seg_path,
        det_score_thresh,
        cfg,
        backend="numpy",
        topk=topk,
        observed_index_mode="linspace",
    )


# ============================================================================
# Main
# ============================================================================

def main():
    cfg = init()
    _seed_everything(cfg.seed)
    precision_mode = cfg.precision
    print(f"[OpenVINO] Device: {cfg.device}, Precision: {precision_mode}, Seed: {cfg.seed}")

    # Paths
    ov_pem_model_path = "model_save/ov_pem_model_cpu.xml"
    ov_fe_model_path = "model_save/ov_fe_model_cpu.xml"
    ov_gpu_kernel_path = "./model/ov_pointnet2_op/pem_gpu_ops.xml"
    ov_extension_lib_path = "./model/ov_pointnet2_op/build/libopenvino_operation_extension.so"

    # Initialize OpenVINO
    core = Core()
    core.add_extension(ov_extension_lib_path)

    # -----------------------------------------------------------------------
    # Configure precision strategy
    # -----------------------------------------------------------------------
    if precision_mode == "fp32":
        # Baseline: everything in FP32
        if "GPU" in cfg.device:
            core.set_property("GPU", {"INFERENCE_PRECISION_HINT": "f32"})
            core.set_property("GPU", {"CONFIG_FILE": ov_gpu_kernel_path})
        ov_fe_compiled = core.compile_model(core.read_model(ov_fe_model_path), cfg.device)
        ov_pem_compiled = core.compile_model(core.read_model(ov_pem_model_path), cfg.device)
        print("[FP32] Baseline: all operations in FP32")

    elif precision_mode == "fp16":
        # FE and PEM feature extraction run in FP16; PEM matching/SVD stays FP32.
        # This is a real precision boundary: GPU's model-wide FP16 hint does not
        # preserve FP32 islands for Softmax/SVD, so split the graph explicitly.
        fe_model = core.read_model(ov_fe_model_path)
        pem_feature_model, pem_matching_model, cut_param_names = _build_pem_feature_split(
            core, ov_pem_model_path
        )

        if "GPU" in cfg.device:
            fe_config = {"INFERENCE_PRECISION_HINT": "f16",
                         "CONFIG_FILE": ov_gpu_kernel_path}
            pem_feature_config = {"INFERENCE_PRECISION_HINT": "f16",
                                  "CONFIG_FILE": ov_gpu_kernel_path}
            pem_matching_config = {"INFERENCE_PRECISION_HINT": "f32",
                                   "CONFIG_FILE": ov_gpu_kernel_path}
            ov_fe_compiled = core.compile_model(fe_model, cfg.device, fe_config)
            pem_feature_compiled = core.compile_model(
                pem_feature_model, cfg.device, pem_feature_config)
            pem_matching_compiled = core.compile_model(
                pem_matching_model, cfg.device, pem_matching_config)
        else:
            ov_fe_compiled = core.compile_model(fe_model, cfg.device)
            pem_feature_compiled = core.compile_model(pem_feature_model, cfg.device)
            pem_matching_compiled = core.compile_model(pem_matching_model, cfg.device)

        ov_pem_compiled = _SplitPemCompiled(
            pem_feature_compiled,
            pem_matching_compiled,
            cut_param_names,
        )
        print(f"[FP16] FE=FP16, PEM feature=FP16, PEM matching/SVD=FP32 "
              f"({len(cut_param_names)} cut tensors)")

    else:
        raise ValueError(f"Unknown precision mode: {precision_mode}")

    # -----------------------------------------------------------------------
    # Feature Extraction
    # -----------------------------------------------------------------------
    print(f"[OpenVINO] Extracting templates ({precision_mode})...")
    tem_path = os.path.join(cfg.output_dir, 'templates')
    all_tem, all_tem_pts, all_tem_choose = get_templates_np(tem_path, cfg.test_dataset)

    # rgb_input:    (42, 3, H, W)  — views as a true batch, no channel concat
    # choose_input: (42, n_sample)  — per-view pixel choose indices
    # pts_input:    (1, 42*n_sample, 3)
    tem_rgb_concat    = np.stack(all_tem, axis=0).astype(np.float32)      # (42, 3, H, W)
    tem_pts_concat    = np.concatenate(all_tem_pts, axis=1)               # (1, 210000, 3)
    tem_choose_concat = np.stack(all_tem_choose, axis=0).astype(np.int64) # (42, 5000)

    feature_inputs = {
        "rgb_input": tem_rgb_concat,
        "pts_input": tem_pts_concat,
        "choose_input": tem_choose_concat
    }

    # Warm up — use an isolated InferRequest so its GPU BFYX-padded output
    # tensors ([B,N,C,1]) are not cached in the compiled model's default
    # request.  Reusing a request whose output tensor has shape [1,2048,3,1]
    # against a port that expects [1,2048,3] raises a shape-mismatch error.
    _warmup_req = ov_fe_compiled.create_infer_request()
    _warmup_req.infer(feature_inputs)
    del _warmup_req

    time_start = time.time()
    feature_results = ov_fe_compiled(feature_inputs)
    results_list = _ordered_outputs(feature_results, ov_fe_compiled)
    fe_time = time.time() - time_start
    print(f"[OpenVINO] FE inference time: {fe_time * 1000:.2f} ms ({precision_mode})")

    # GPU BFYX format pads 3-D gather outputs to 4-D [B,N,C,1]; squeeze it off.
    all_tem_pts  = _squeeze_bfyx(results_list[0])   # (1, 2048, 3)
    all_tem_feat = _squeeze_bfyx(results_list[1])   # (1, 2048, 256)

    # -----------------------------------------------------------------------
    # Pose Estimation
    # -----------------------------------------------------------------------
    print(f"[OpenVINO] Loading PEM input data...")
    input_data, img, whole_pts, model_points, detections = get_test_data_np(
        cfg.rgb_path, cfg.depth_path, cfg.cam_path, cfg.cad_path, cfg.seg_path,
        cfg.det_score_thresh, cfg.test_dataset, cfg.topk_ism_score
    )
    ninstance = input_data['pts'].shape[0]
    input_data['dense_po'] = np.repeat(all_tem_pts, ninstance, axis=0)
    input_data['dense_fo'] = np.repeat(all_tem_feat, ninstance, axis=0)

    # Warm up
    warmup_end = min(cfg.max_batch_size, ninstance)
    warmup_inputs = {k: input_data[k][0:warmup_end] for k in
                     ['pts', 'rgb', 'rgb_choose', 'model', 'dense_po', 'dense_fo']}
    _ = ov_pem_compiled(warmup_inputs)

    # Batched inference
    batch_size = cfg.max_batch_size
    all_R, all_t, all_score = [], [], []
    total_pem_time = 0

    for start in range(0, ninstance, batch_size):
        end = min(start + batch_size, ninstance)
        batch_inputs = {k: input_data[k][start:end] for k in
                        ['pts', 'rgb', 'rgb_choose', 'model', 'dense_po', 'dense_fo']}

        pem_time_start = time.time()
        results = ov_pem_compiled(batch_inputs)
        pem_time = time.time() - pem_time_start
        total_pem_time += pem_time
        print(f"  Batch {start}:{end} PEM time: {pem_time * 1000:.2f} ms")

        results_output = _ordered_outputs(results, ov_pem_compiled)
        all_R.append(results_output[0])
        all_t.append(results_output[1])
        all_score.append(results_output[2])

    ov_pred_R = np.concatenate(all_R, axis=0)
    ov_pred_t = np.concatenate(all_t, axis=0)
    ov_pred_pose_score = np.concatenate(all_score, axis=0)

    pose_scores = ov_pred_pose_score * input_data['score']
    pred_rot = ov_pred_R
    pred_trans = ov_pred_t * 1000

    print(f"\n[Summary] Precision={precision_mode}, Device={cfg.device}")
    print(f"  FE time:    {fe_time * 1000:.2f} ms")
    print(f"  PEM time:   {total_pem_time * 1000:.2f} ms")
    print(f"  Total time: {(fe_time + total_pem_time) * 1000:.2f} ms")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    os.makedirs(f"{cfg.output_dir}/sam6d_results", exist_ok=True)
    for idx in range(len(detections)):
        detections[idx]['score'] = float(pose_scores[idx])
        detections[idx]['R'] = list(pred_rot[idx].tolist())
        detections[idx]['t'] = list(pred_trans[idx].tolist())

    result_filename = f'detection_pem_ov_{cfg.device}_{precision_mode}.json'
    with open(os.path.join(f"{cfg.output_dir}/sam6d_results", result_filename), "w") as f:
        json.dump(detections, f)

    # Visualize
    save_path = os.path.join(f"{cfg.output_dir}/sam6d_results",
                             f'vis_pem_ov_{cfg.device}_{precision_mode}.png')
    valid_masks = pose_scores == pose_scores.max()
    K = input_data['K'][valid_masks]
    vis_img = visualize(img, pred_rot[valid_masks], pred_trans[valid_masks],
                        model_points * 1000, K, save_path)
    vis_img.save(save_path)
    print(f"[Done] Results saved to {result_filename}")

    # -----------------------------------------------------------------------
    # AR Evaluation
    # -----------------------------------------------------------------------
    if cfg.gt_path and os.path.exists(cfg.gt_path):
        print(f"\n[OpenVINO] Running AR evaluation ({precision_mode})...")

        with open(cfg.gt_path) as f:
            gt_targets = json.load(f)

        mesh_eval = trimesh.load_mesh(cfg.cad_path)
        pts_mm = np.array(mesh_eval.vertices, dtype=np.float64)
        center_mm = pts_mm.mean(axis=0)
        diameter_mm = float(2.0 * np.max(np.linalg.norm(pts_mm - center_mm, axis=1)))

        models_info = None
        if cfg.models_info_path and os.path.exists(cfg.models_info_path):
            with open(cfg.models_info_path) as f:
                models_info = json.load(f)

        unique_obj_ids = list({gt["obj_id"] for gt in gt_targets})

        mesh_pyr = None
        if not cfg.skip_vsd:
            try:
                import pyrender
                tm = trimesh.load_mesh(cfg.cad_path, process=False)
                mesh_pyr = pyrender.Mesh.from_trimesh(tm)
            except ImportError:
                print("  [AR] pyrender not available, skipping VSD")
                cfg.skip_vsd = True

        model_data = {}
        for oid in unique_obj_ids:
            if models_info and str(oid) in models_info:
                bop_diameter = float(models_info[str(oid)]["diameter"])
                syms = load_symmetries(models_info, oid)
            else:
                bop_diameter = diameter_mm
                syms = [{"R": np.eye(3), "t": np.zeros(3)}]

            model_data[oid] = {"pts": pts_mm, "diameter": bop_diameter, "syms": syms}
            if mesh_pyr is not None:
                model_data[oid]["mesh_pyrender"] = mesh_pyr

        cam_info_raw = json.load(open(cfg.cam_path))
        cam_data = {}
        for gt in gt_targets:
            img_id_str = str(gt["image_id"])
            if img_id_str not in cam_data:
                cam_data[img_id_str] = {
                    "cam_K": cam_info_raw["cam_K"],
                    "depth_scale": cam_info_raw.get("depth_scale", 1.0),
                }

        scene_dir = os.path.dirname(cfg.depth_path)
        det_pem_path = os.path.join(cfg.output_dir, "sam6d_results", result_filename)
        evaluate_and_print_ar(
            det_pem_path, gt_targets, model_data, cam_data,
            scene_dir, skip_vsd=cfg.skip_vsd,
        )


if __name__ == "__main__":
    main()
