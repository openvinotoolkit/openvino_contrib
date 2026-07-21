# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib
import json
import logging
import os
import os.path as osp
import random
import sys
import time
import glob
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from eval_utils import compute_mssd, compute_mspd, load_symmetries

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "provider"))
sys.path.append(os.path.join(BASE_DIR, "utils"))
sys.path.append(os.path.join(BASE_DIR, "model"))
sys.path.append(os.path.join(BASE_DIR, "model", "pointnet2"))


DETECTION_PATHS = {
    "ycbv": "../Instance_Segmentation_Model/log/sam_ov/result_ycbv.json",
    "tudl": "../Instance_Segmentation_Model/log/sam_ov/result_tudl.json",
    "tless": "../Instance_Segmentation_Model/log/sam_ov/result_tless.json",
    "lmo": "../Instance_Segmentation_Model/log/sam_ov/result_lmo.json",
    "itodd": "../Instance_Segmentation_Model/log/sam_ov/result_itodd.json",
    "icbin": "../Instance_Segmentation_Model/log/sam_ov/result_icbin.json",
    "hb": "../Instance_Segmentation_Model/log/sam_ov/result_hb.json",
}


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluate PEM(OpenVINO) on a subset of BOP samples")
    parser.add_argument("--device", type=str, default="GPU", help="OpenVINO device: CPU/GPU")
    parser.add_argument("--gpus", type=str, default="0", help="Kept for compatibility (unused by OV)")
    parser.add_argument("--model", type=str, default="pose_estimation_model", help="Kept for compatibility")
    parser.add_argument("--config", type=str, default="config/base.yaml", help="Path to config file")
    parser.add_argument("--dataset", type=str, default="lmo", help="Dataset name")
    parser.add_argument("--checkpoint_path", type=str, default="none", help="Kept for compatibility")
    parser.add_argument("--iter", type=int, default=0, help="Experiment iter id used in output naming")
    parser.add_argument("--view", type=int, default=-1, help="Number of template views")
    parser.add_argument("--exp_id", type=int, default=0, help="Experiment id used in output naming")

    parser.add_argument("--max_samples", type=int, default=10, help="Maximum number of image samples to evaluate")
    parser.add_argument("--detection_path", type=str, default="", help="Override detection json path")
    parser.add_argument("--output_name", type=str, default="subset_eval_ov", help="Output folder suffix")

    parser.add_argument("--ov_fe_model_path", type=str, default="", help="Path to ov_fe_model xml")
    parser.add_argument("--ov_pem_model_path", type=str, default="", help="Path to ov_pem_model xml")
    parser.add_argument("--ov_extension_path", type=str, default="", help="Path to OpenVINO custom op extension .so")
    parser.add_argument("--ov_gpu_kernel_path", type=str, default="", help="Path to OpenVINO GPU kernel xml")

    # AR thresholds
    parser.add_argument(
        "--mssd_thresholds",
        type=str,
        default="0.05,0.10,0.20,0.30,0.40,0.50",
        help="Comma-separated MSSD thresholds as diameter ratio",
    )
    parser.add_argument(
        "--mspd_thresholds",
        type=str,
        default="5,10,20,30,40,50",
        help="Comma-separated MSPD thresholds in pixels",
    )
    parser.add_argument(
        "--vsd_thresholds",
        type=str,
        default="0.05,0.10,0.20,0.30,0.40,0.50",
        help="Comma-separated VSD thresholds as diameter ratio (for VSD approximation)",
    )
    return parser


def parse_float_list(v: str) -> List[float]:
    return [float(x.strip()) for x in v.split(",") if x.strip()]


class _Cfg:
    """Minimal nested-attribute config object (replaces gorilla.Config)."""
    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, _Cfg(value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        return json.dumps(self._to_dict(), indent=2)

    def _to_dict(self):
        d = {}
        for k, v in self.__dict__.items():
            d[k] = v._to_dict() if isinstance(v, _Cfg) else v
        return d


def init_cfg(args):
    import yaml  # pylint: disable=import-outside-toplevel

    with open(args.config, "r") as f:
        raw = yaml.safe_load(f)
    cfg = _Cfg(raw)

    exp_name = args.model + "_" + osp.splitext(args.config.split("/")[-1])[0] + "_id" + str(args.exp_id)
    log_dir = osp.join("log", exp_name)
    os.makedirs(log_dir, exist_ok=True)

    cfg.exp_name = exp_name
    cfg.gpus = args.gpus
    cfg.model_name = args.model
    cfg.log_dir = log_dir
    cfg.checkpoint_path = args.checkpoint_path
    cfg.test_iter = args.iter
    cfg.dataset = args.dataset

    if args.view != -1:
        cfg.test_dataset.n_template_view = args.view

    return cfg


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_vsd_surrogate(
    R_pred: np.ndarray,
    t_pred_mm: np.ndarray,
    R_gt: np.ndarray,
    t_gt_mm: np.ndarray,
    pts_mm: np.ndarray,
) -> float:
    pred_pts = (R_pred @ pts_mm.T).T + t_pred_mm.reshape(1, 3)
    gt_pts = (R_gt @ pts_mm.T).T + t_gt_mm.reshape(1, 3)
    return float(np.mean(np.abs(pred_pts[:, 2] - gt_pts[:, 2])))


def ar_from_thresholds(value: float, thresholds: List[float]) -> float:
    if len(thresholds) == 0:
        return 0.0
    hits = [1.0 if value <= thr else 0.0 for thr in thresholds]
    return float(np.mean(hits))


def choose_detection_path(dataset_name: str, override_path: str) -> str:
    if override_path:
        return override_path

    default_path = DETECTION_PATHS.get(dataset_name)
    if default_path and osp.isfile(default_path):
        return default_path

    # Fallback: pick a subset-export file if present (e.g. result_lmo_10imgs.json)
    fallback_glob = osp.join(
        "..", "Instance_Segmentation_Model", "log", "sam_ov", f"result_{dataset_name}_*imgs.json"
    )
    candidates = sorted(glob.glob(fallback_glob))
    if candidates:
        return candidates[-1]

    if default_path:
        return default_path
    raise ValueError(f"No default detection path for dataset={dataset_name}")


def build_gt_cache(test_root: str) -> Dict[str, Dict[str, List[dict]]]:
    cache = {}
    test_dir = osp.join(test_root, "test")
    if not osp.isdir(test_dir):
        return cache
    for scene_id in sorted(os.listdir(test_dir)):
        scene_path = osp.join(test_dir, scene_id)
        if not osp.isdir(scene_path):
            continue
        gt_path = osp.join(scene_path, "scene_gt.json")
        cam_path = osp.join(scene_path, "scene_camera.json")
        if not osp.isfile(gt_path) or not osp.isfile(cam_path):
            continue
        with open(gt_path, "r") as f:
            scene_gt = json.load(f)
        with open(cam_path, "r") as f:
            scene_cam = json.load(f)
        cache[scene_id] = {"gt": scene_gt, "cam": scene_cam}
    return cache


def load_models_info(test_root: str) -> dict | None:
    candidates = [
        osp.join(test_root, "models", "models_info.json"),
        osp.join(test_root, "models_eval", "models_info.json"),
        osp.join(test_root, "models_cad", "models_info.json"),
    ]
    for path in candidates:
        if osp.isfile(path):
            with open(path, "r") as f:
                return json.load(f)
    return None


def build_symmetry_cache(obj_ids: List[int], models_info: dict | None) -> Dict[int, List[dict]]:
    identity = [{"R": np.eye(3), "t": np.zeros(3)}]
    if models_info is None:
        return {int(obj_id): identity for obj_id in obj_ids}

    cache: Dict[int, List[dict]] = {}
    for obj_id in obj_ids:
        obj_id_int = int(obj_id)
        if str(obj_id_int) in models_info:
            cache[obj_id_int] = load_symmetries(models_info, obj_id_int)
        else:
            cache[obj_id_int] = identity
    return cache


def _name_of_output(k):
    if hasattr(k, "get_any_name"):
        try:
            return k.get_any_name()
        except Exception:
            return str(k)
    return str(k)


def _pick_outputs(infer_result, prefer_names: List[str], fallback_index: int):
    entries = [(_name_of_output(k), v) for k, v in infer_result.items()]
    for prefer in prefer_names:
        for name, arr in entries:
            if prefer in name:
                return arr
    values = list(infer_result.values())
    if fallback_index < len(values):
        return values[fallback_index]
    raise RuntimeError(f"Unable to fetch output index={fallback_index} from OpenVINO result")


def _squeeze_bfyx(arr):
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    return arr


def init_openvino_models(args):
    try:
        from openvino import Core
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenVINO is not installed in this environment. Install runtime first."
        ) from exc

    core = Core()

    ov_extension_lib_path = args.ov_extension_path or osp.join(
        BASE_DIR, "model/ov_pointnet2_op/build/libopenvino_operation_extension.so"
    )
    ov_gpu_kernel_path = args.ov_gpu_kernel_path or osp.join(
        BASE_DIR, "model/ov_pointnet2_op/pem_gpu_ops.xml"
    )
    ov_fe_model_path = args.ov_fe_model_path or osp.join(BASE_DIR, "model_save/ov_fe_model_cpu.xml")
    ov_pem_model_path = args.ov_pem_model_path or osp.join(BASE_DIR, "model_save/ov_pem_model_cpu.xml")

    if not osp.isfile(ov_extension_lib_path):
        raise FileNotFoundError(f"OpenVINO extension not found: {ov_extension_lib_path}")
    if not osp.isfile(ov_fe_model_path):
        raise FileNotFoundError(f"OpenVINO FE model not found: {ov_fe_model_path}")
    if not osp.isfile(ov_pem_model_path):
        raise FileNotFoundError(f"OpenVINO PEM model not found: {ov_pem_model_path}")

    core.add_extension(ov_extension_lib_path)
    if args.device.upper().startswith("GPU"):
        if osp.isfile(ov_gpu_kernel_path):
            core.set_property("GPU", {"CONFIG_FILE": ov_gpu_kernel_path})
        core.set_property("GPU", {"INFERENCE_PRECISION_HINT": "f32"})
        ov_fe_compiled = core.compile_model(core.read_model(ov_fe_model_path), args.device)
        ov_pem_compiled = core.compile_model(core.read_model(ov_pem_model_path), args.device)
    else:
        ov_fe_compiled = core.compile_model(core.read_model(ov_fe_model_path), args.device)
        ov_pem_compiled = core.compile_model(core.read_model(ov_pem_model_path), args.device)

    return ov_fe_compiled, ov_pem_compiled


def compute_template_features_ov(dataset, ov_fe_compiled):
    obj_count = len(dataset.objects)
    n_template_view = int(dataset.n_template_view)

    dense_po_all = []
    dense_fo_all = []
    for obj in dataset.objects:
        all_tem = []
        all_tem_choose = []
        all_tem_pts = []
        for i in range(n_template_view):
            tem_rgb, tem_choose, tem_pts = dataset._get_template(obj, i)
            all_tem.append(np.expand_dims(to_numpy(tem_rgb), axis=0).astype(np.float32))
            all_tem_choose.append(to_numpy(tem_choose).astype(np.int64))
            all_tem_pts.append(np.expand_dims(to_numpy(tem_pts), axis=0).astype(np.float32))

        tem_rgb = np.concatenate(all_tem, axis=0)
        tem_pts = np.concatenate(all_tem_pts, axis=1)
        tem_choose = np.stack(all_tem_choose, axis=0)

        feature_inputs = {
            "rgb_input": tem_rgb,
            "pts_input": tem_pts,
            "choose_input": tem_choose,
        }

        # Use an isolated InferRequest to avoid default-request tensor shape
        # mismatch on GPU BFYX-padded custom-op outputs.
        fe_req = ov_fe_compiled.create_infer_request()
        fe_results = fe_req.infer(feature_inputs)
        dense_po = _squeeze_bfyx(_pick_outputs(fe_results, ["dense_po", "pts", "po"], 0))
        dense_fo = _squeeze_bfyx(_pick_outputs(fe_results, ["dense_fo", "feat", "fo"], 1))
        dense_po_all.append(np.asarray(dense_po, dtype=np.float32))
        dense_fo_all.append(np.asarray(dense_fo, dtype=np.float32))

    if len(dense_po_all) != obj_count or len(dense_fo_all) != obj_count:
        raise RuntimeError("Template feature extraction did not cover all objects")

    dense_po_all = np.concatenate(dense_po_all, axis=0)
    dense_fo_all = np.concatenate(dense_fo_all, axis=0)
    return dense_po_all, dense_fo_all


def main():
    args = get_parser().parse_args()
    cfg = init_cfg(args)

    random.seed(cfg.rd_seed)
    np.random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)

    mssd_thresh_ratio = parse_float_list(args.mssd_thresholds)
    mspd_thresh_px = parse_float_list(args.mspd_thresholds)
    vsd_thresh_ratio = parse_float_list(args.vsd_thresholds)

    detection_path = choose_detection_path(args.dataset, args.detection_path)
    if not osp.isfile(detection_path):
        raise FileNotFoundError(f"Detection file not found: {detection_path}")

    print("************************ Start Logging ************************")
    print(cfg)
    print(f"ov_device: {args.device}")
    print(f"detection_path: {detection_path}")

    print("initializing OpenVINO models ...")
    t0 = time.time()
    ov_fe_compiled, ov_pem_compiled = init_openvino_models(args)
    print(f"OpenVINO init time: {time.time() - t0:.2f}s")

    dataset_module = importlib.import_module(cfg.test_dataset.name)
    dataset = dataset_module.BOPTestset(cfg.test_dataset, args.dataset, detection_path)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
    )

    print("extracting template features with OpenVINO ...")
    t0 = time.time()
    dense_po_all, dense_fo_all = compute_template_features_ov(dataset, ov_fe_compiled)
    print(f"Template FE time: {time.time() - t0:.2f}s")

    test_root = osp.join(cfg.test_dataset.data_dir, args.dataset)
    gt_cache = build_gt_cache(test_root)
    models_info = load_models_info(test_root)
    sym_cache = build_symmetry_cache(list(dataset.obj_idxs.keys()), models_info)
    has_gt = len(gt_cache) > 0
    print(f"GT available: {has_gt}")

    out_dir = osp.join(cfg.log_dir, f"{args.dataset}_{args.output_name}_iter{str(cfg.test_iter).zfill(6)}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = osp.join(out_dir, f"result_{args.dataset}.csv")
    per_sample_path = osp.join(out_dir, "per_sample_metrics.json")
    summary_path = osp.join(out_dir, "summary.json")

    lines = []
    per_sample = []

    total_images = min(args.max_samples, len(dataloader))
    total_instances = 0
    total_eval_instances = 0

    bs = 1  # process one instance at a time to avoid GPU OOM
    with tqdm(total=total_images, desc="Subset PEM eval (OV)") as pbar:
        for image_idx, data in enumerate(dataloader):
            if image_idx >= total_images:
                break

            start = time.time()

            n_instance = data["pts"].size(1)
            total_instances += int(n_instance)

            n_batch = int(np.ceil(n_instance / bs))
            pred_Rs = []
            pred_Ts = []
            pred_scores = []

            for j in range(n_batch):
                start_idx = j * bs
                end_idx = n_instance if j == n_batch - 1 else (j + 1) * bs

                obj_idx_np = to_numpy(data["obj"][0][start_idx:end_idx]).reshape(-1).astype(np.int64)

                batch_inputs = {
                    "pts": to_numpy(data["pts"][0][start_idx:end_idx]).astype(np.float32),
                    "rgb": to_numpy(data["rgb"][0][start_idx:end_idx]).astype(np.float32),
                    "rgb_choose": to_numpy(data["rgb_choose"][0][start_idx:end_idx]).astype(np.int32),
                    "model": to_numpy(data["model"][0][start_idx:end_idx]).astype(np.float32),
                    "dense_po": dense_po_all[obj_idx_np].astype(np.float32),
                    "dense_fo": dense_fo_all[obj_idx_np].astype(np.float32),
                }

                infer_out = ov_pem_compiled(batch_inputs)
                pred_R = _pick_outputs(infer_out, ["pred_R", "rot"], 0)
                pred_t = _pick_outputs(infer_out, ["pred_t", "trans"], 1)
                pred_score = _pick_outputs(infer_out, ["pred_pose_score", "score"], 2)

                pred_Rs.append(np.asarray(pred_R, dtype=np.float32))
                pred_Ts.append(np.asarray(pred_t, dtype=np.float32))
                pred_scores.append(np.asarray(pred_score, dtype=np.float32))

            pred_Rs = np.concatenate(pred_Rs, axis=0).reshape(-1, 9)
            pred_Ts = np.concatenate(pred_Ts, axis=0) * 1000.0
            pred_scores = np.concatenate(pred_scores, axis=0)
            det_scores = to_numpy(data["score"][0, :, 0]).astype(np.float32)
            pred_scores = pred_scores * det_scores

            scene_id = int(data["scene_id"].item())
            img_id = int(data["img_id"].item())
            image_time = time.time() - start + float(data["seg_time"].item())

            # Save PEM predictions in the same csv format as test_bop.py
            for k in range(n_instance):
                line = ",".join(
                    (
                        str(scene_id),
                        str(img_id),
                        str(data["obj_id"][0][k].item()),
                        str(float(pred_scores[k])),
                        " ".join((str(v) for v in pred_Rs[k])),
                        " ".join((str(v) for v in pred_Ts[k])),
                        f"{image_time}\n",
                    )
                )
                lines.append(line)

            # Evaluate against GT when available.
            scene_key = f"{scene_id:06d}"
            img_key = str(img_id)
            if has_gt and scene_key in gt_cache and img_key in gt_cache[scene_key]["gt"]:
                gt_list = gt_cache[scene_key]["gt"][img_key]
                K = np.array(gt_cache[scene_key]["cam"][img_key]["cam_K"]).reshape(3, 3)

                # Greedy one-to-one matching by object and minimum MSSD approximation.
                used_gt = set()
                for k in range(n_instance):
                    obj_id = int(data["obj_id"][0][k].item())
                    if obj_id not in dataset.obj_idxs:
                        continue

                    obj_idx = dataset.obj_idxs[obj_id]
                    diameter_mm = float(dataset.objects[obj_idx].diameter * 1000.0)
                    model_pts_mm = dataset.objects[obj_idx].model_points * 1000.0

                    syms = sym_cache.get(obj_id, [{"R": np.eye(3), "t": np.zeros(3)}])
                    R_pred = pred_Rs[k].reshape(3, 3).astype(np.float64)
                    t_pred = pred_Ts[k].reshape(3).astype(np.float64)

                    cand = []
                    for gi, gt in enumerate(gt_list):
                        if gi in used_gt:
                            continue
                        if int(gt["obj_id"]) != obj_id:
                            continue
                        R_gt = np.array(gt["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
                        t_gt = np.array(gt["cam_t_m2c"], dtype=np.float64).reshape(3)
                        mssd_mm = compute_mssd(R_pred, t_pred, R_gt, t_gt, model_pts_mm, syms)
                        mspd_px = compute_mspd(R_pred, t_pred, R_gt, t_gt, model_pts_mm, K, syms)
                        vsd_mm = compute_vsd_surrogate(R_pred, t_pred, R_gt, t_gt, model_pts_mm)
                        cand.append((mssd_mm, gi, mspd_px, vsd_mm))

                    if len(cand) == 0:
                        continue

                    cand.sort(key=lambda x: x[0])
                    best = cand[0]
                    used_gt.add(best[1])
                    mssd_mm = float(best[0])
                    mspd_px = float(best[2])
                    vsd_mm = float(best[3])

                    mssd_abs_thresh = [r * diameter_mm for r in mssd_thresh_ratio]
                    vsd_abs_thresh = [r * diameter_mm for r in vsd_thresh_ratio]

                    ar_mssd = ar_from_thresholds(mssd_mm, mssd_abs_thresh)
                    ar_mspd = ar_from_thresholds(mspd_px, mspd_thresh_px)
                    ar_vsd = ar_from_thresholds(vsd_mm, vsd_abs_thresh)
                    ar_mean = float(np.mean([ar_mssd, ar_mspd, ar_vsd]))

                    rec = {
                        "sample_idx": len(per_sample),
                        "image_sample_idx": image_idx,
                        "scene_id": scene_id,
                        "img_id": img_id,
                        "obj_id": obj_id,
                        "score": float(pred_scores[k]),
                        "diameter_mm": diameter_mm,
                        "mssd_mm": mssd_mm,
                        "mspd_px": mspd_px,
                        "vsd_mm_approx": vsd_mm,
                        "AR_MSSD": ar_mssd,
                        "AR_MSPD": ar_mspd,
                        "AR_VSD": ar_vsd,
                        "AR_mean": ar_mean,
                    }
                    per_sample.append(rec)
                    total_eval_instances += 1

                    print(
                        f"[sample {rec['sample_idx']:04d}] scene={scene_id:06d} img={img_id:06d} obj={obj_id:02d} "
                        f"MSSD={mssd_mm:.2f}mm MSPD={mspd_px:.2f}px VSD~={vsd_mm:.2f}mm "
                        f"AR(MSSD/MSPD/VSD/mean)=({ar_mssd:.3f}/{ar_mspd:.3f}/{ar_vsd:.3f}/{ar_mean:.3f})"
                    )

            pbar.update(1)

    with open(csv_path, "w+") as f:
        f.writelines(lines)

    # Aggregate summary.
    if total_eval_instances > 0:
        mean_ar_mssd = float(np.mean([x["AR_MSSD"] for x in per_sample]))
        mean_ar_mspd = float(np.mean([x["AR_MSPD"] for x in per_sample]))
        mean_ar_vsd = float(np.mean([x["AR_VSD"] for x in per_sample]))
        mean_ar = float(np.mean([x["AR_mean"] for x in per_sample]))
    else:
        mean_ar_mssd = None
        mean_ar_mspd = None
        mean_ar_vsd = None
        mean_ar = None

    summary = {
        "dataset": args.dataset,
        "ov_device": args.device,
        "max_samples": int(args.max_samples),
        "n_images_tested": int(total_images),
        "n_instances_total": int(total_instances),
        "n_instances_evaluated": int(total_eval_instances),
        "detection_path": detection_path,
        "gt_available": has_gt,
        "metric_note": {
            "MSSD": "Symmetry-aware MSSD via eval_utils.compute_mssd",
            "MSPD": "Symmetry-aware MSPD via eval_utils.compute_mspd",
            "VSD": "Approximate VSD surrogate from mean depth disagreement",
        },
        "thresholds": {
            "mssd_ratio": mssd_thresh_ratio,
            "mspd_px": mspd_thresh_px,
            "vsd_ratio": vsd_thresh_ratio,
        },
        "mean_AR_MSSD": mean_ar_mssd,
        "mean_AR_MSPD": mean_ar_mspd,
        "mean_AR_VSD": mean_ar_vsd,
        "mean_AR_overall": mean_ar,
        "csv_path": csv_path,
        "per_sample_path": per_sample_path,
    }

    with open(per_sample_path, "w") as f:
        json.dump(per_sample, f, indent=2)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ SUBSET EVAL SUMMARY (OV) ================")
    print(json.dumps(summary, indent=2))
    print(f"Saved predictions csv: {csv_path}")
    print(f"Saved per-sample metrics: {per_sample_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
