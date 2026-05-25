import logging
import os
import os.path as osp
import json
import time
import types
from collections import defaultdict

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Subset

from utils.inout import load_json, save_json_bop23
from utils.bbox_utils import force_binary_mask
from model.utils import mask_to_rle


# ── mAP helpers ──────────────────────────────────────────────────────────────
def compute_mask_iou_matrix(pred_masks, gt_masks):
    """Compute pairwise IoU between predicted and GT binary masks.
    pred_masks: (N, H, W)  gt_masks: (M, H, W)  → returns (N, M)
    """
    N = pred_masks.shape[0]
    M = gt_masks.shape[0]
    pred_flat = pred_masks.reshape(N, -1).astype(bool)
    gt_flat = gt_masks.reshape(M, -1).astype(bool)
    intersection = (pred_flat[:, None, :] & gt_flat[None, :, :]).sum(axis=2)
    union = (pred_flat[:, None, :] | gt_flat[None, :, :]).sum(axis=2)
    iou = np.where(union > 0, intersection / union, 0.0)
    return iou


def compute_ap(scores, matched, n_gt):
    """Compute Average Precision for one class given ranked matches."""
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    matched = np.array(matched, dtype=bool)[order]
    tp = np.cumsum(matched)
    fp = np.cumsum(~matched)
    recall = tp / n_gt
    precision = tp / (tp + fp)
    # Append sentinel values
    mrec = np.concatenate(([0.0], recall, [recall[-1]]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    # Compute area under PR curve (all-point interpolation)
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def evaluate_image(pred_masks, pred_scores, pred_obj_ids,
                   gt_masks, gt_obj_ids, iou_thresh=0.5):
    """Evaluate detections for a single image.
    Returns per-class AP contributions (tp/fp flags + scores + gt counts).
    """
    per_class = {}  # obj_id -> {"scores": [], "matched": [], "n_gt": int}
    all_gt_ids = set(gt_obj_ids)
    all_pred_ids = set(pred_obj_ids)
    all_ids = all_gt_ids | all_pred_ids

    for obj_id in all_ids:
        gt_idx = [i for i, g in enumerate(gt_obj_ids) if g == obj_id]
        pred_idx = [i for i, p in enumerate(pred_obj_ids) if p == obj_id]
        n_gt = len(gt_idx)
        if len(pred_idx) == 0:
            per_class[obj_id] = {"scores": [], "matched": [], "n_gt": n_gt}
            continue
        p_masks = pred_masks[pred_idx]
        p_scores = pred_scores[pred_idx]
        matched_flags = []
        gt_matched = set()
        # Sort by score descending
        order = np.argsort(-p_scores)
        for o in order:
            best_iou = 0.0
            best_gt = -1
            for gi in gt_idx:
                if gi in gt_matched:
                    continue
                iou_val = compute_mask_iou_matrix(
                    p_masks[o:o+1], gt_masks[gi:gi+1]
                )[0, 0]
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt = gi
            if best_iou >= iou_thresh and best_gt >= 0:
                matched_flags.append(True)
                gt_matched.add(best_gt)
            else:
                matched_flags.append(False)
        per_class[obj_id] = {
            "scores": p_scores[order].tolist(),
            "matched": matched_flags,
            "n_gt": n_gt,
        }
    return per_class


def load_gt_masks_for_frame(scene_path, frame_id, gt_obj_ids):
    """Load visible masks for each GT object in a frame."""
    masks = []
    mask_dir = osp.join(scene_path, "mask_visib")
    for gt_idx in range(len(gt_obj_ids)):
        mask_path = osp.join(mask_dir, f"{int(frame_id):06d}_{gt_idx:06d}.png")
        if osp.exists(mask_path):
            from PIL import Image
            m = np.array(Image.open(mask_path))
            masks.append((m > 0).astype(np.uint8))
        else:
            masks.append(None)
    return masks


# ── main ─────────────────────────────────────────────────────────────────────
@hydra.main(version_base=None, config_path="configs", config_name="run_inference")
def run_inference(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_path = hydra_cfg["runtime"]["output_dir"]
    print(f"[run_inference_10] Output path: {output_path}")
    print("[run_inference_10] Initializing logger, callbacks and trainer")

    if cfg.machine.name == "slurm":
        num_gpus = int(os.environ["SLURM_GPUS_ON_NODE"])
        num_nodes = int(os.environ["SLURM_NNODES"])
        cfg.machine.trainer.devices = num_gpus
        cfg.machine.trainer.num_nodes = num_nodes
        print(f"[run_inference_10] Slurm config: {num_gpus} gpus, {num_nodes} nodes")
    trainer = instantiate(cfg.machine.trainer)

    default_ref_dataloader_config = cfg.data.reference_dataloader
    default_query_dataloader_config = cfg.data.query_dataloader

    query_dataloader_config = default_query_dataloader_config.copy()
    ref_dataloader_config = default_ref_dataloader_config.copy()

    if cfg.dataset_name in ["hb", "tless"]:
        query_dataloader_config.split = "test_primesense"
    else:
        query_dataloader_config.split = "test"
    query_dataloader_config.root_dir += f"{cfg.dataset_name}"
    query_dataset = instantiate(query_dataloader_config)

    # Limit to first 10 images
    max_images = 10
    total_images = len(query_dataset)
    num_images = min(max_images, total_images)
    query_dataset = Subset(query_dataset, list(range(num_images)))
    print(f"[run_inference_10] Limited dataset to {num_images} images (out of {total_images} total)")

    print("[run_inference_10] Initializing model")
    model = instantiate(cfg.model)

    model.ref_obj_names = cfg.data.datasets[cfg.dataset_name].obj_names
    model.dataset_name = cfg.dataset_name

    query_dataloader = DataLoader(
        query_dataset,
        batch_size=1,
        num_workers=cfg.machine.num_workers,
        shuffle=False,
    )
    if cfg.model.onboarding_config.rendering_type == "pyrender":
        ref_dataloader_config.template_dir += f"templates_pyrender/{cfg.dataset_name}"
        ref_dataset = instantiate(ref_dataloader_config)
    elif cfg.model.onboarding_config.rendering_type == "pbr":
        print("[run_inference_10] Using BlenderProc for reference images")
        ref_dataloader_config._target_ = "provider.bop_pbr.BOPTemplatePBR"
        ref_dataloader_config.root_dir = f"{query_dataloader_config.root_dir}"
        ref_dataloader_config.template_dir += f"templates_pyrender/{cfg.dataset_name}"
        if not os.path.exists(ref_dataloader_config.template_dir):
            os.makedirs(ref_dataloader_config.template_dir)
        ref_dataset = instantiate(ref_dataloader_config)
        ref_dataset.load_processed_metaData(reset_metaData=True)
    else:
        raise NotImplementedError
    model.ref_dataset = ref_dataset

    model.name_prediction_file = f"result_{cfg.dataset_name}"

    # ── Load ground-truth for the scenes we will evaluate ────────────────
    test_root = query_dataloader_config.root_dir  # e.g. .../BOP/lmo
    meta_path = osp.join(test_root, "test_metaData.json")
    meta = load_json(meta_path)
    scene_ids_list = meta["scene_id"][:num_images]
    frame_ids_list = meta["frame_id"][:num_images]

    # Pre-load scene_gt once per unique scene
    scene_gt_cache = {}
    for sid in set(scene_ids_list):
        scene_dir = osp.join(test_root, "test", str(sid))
        scene_gt_cache[sid] = load_json(osp.join(scene_dir, "scene_gt.json"))

    # ── Accumulator for mAP across images ────────────────────────────────
    IOU_THRESHOLDS = [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]
    # global_per_class[thresh][obj_id] -> {"scores", "matched", "n_gt"}
    global_per_class = {
        t: defaultdict(lambda: {"scores": [], "matched": [], "n_gt": 0})
        for t in IOU_THRESHOLDS
    }
    image_results = []
    formatted_detections = []

    # ── Monkey-patch test_step for progress + per-image evaluation ───────
    original_test_step = model.test_step.__func__

    def verbose_test_step(self, batch, idx):
        scene_id = batch["scene_id"][0]
        frame_id = batch["frame_id"][0]
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{num_images}] Processing scene={scene_id} frame={frame_id}")
        print(f"{'='*60}")

        result = original_test_step(self, batch, idx)

        # ── Read back predictions ────────────────────────────────────
        det_path = osp.join(
            self.log_dir,
            f"predictions/{self.dataset_name}/{self.name_prediction_file}/scene{scene_id}_frame{frame_id}.npz",
        )
        rt_path = det_path.replace(".npz", "_runtime.npz")

        if not os.path.exists(det_path):
            print(f"  [WARN] No detections file found at {det_path}")
            return result

        data = np.load(det_path, allow_pickle=True)
        pred_scores = data["score"]
        pred_obj_ids = data["category_id"]
        pred_masks = data["segmentation"]
        pred_bboxes = data["bbox"]
        num_dets = len(pred_scores)

        runtime_total = 0.0

        # Print runtime
        if os.path.exists(rt_path):
            rt = np.load(rt_path)
            print(f"  Runtime: proposal={float(rt['proposal_stage']):.3f}s  matching={float(rt['matching_stage']):.3f}s")
            runtime_total = float(rt["proposal_stage"]) + float(rt["matching_stage"])

        # Save PEM-compatible flat detections while iterating each sample.
        for i in range(num_dets):
            formatted_detections.append({
                "scene_id": int(scene_id),
                "image_id": int(frame_id),
                "category_id": int(pred_obj_ids[i]),
                "bbox": pred_bboxes[i].tolist(),
                "score": float(pred_scores[i]),
                "time": float(runtime_total),
                "segmentation": mask_to_rle(force_binary_mask(pred_masks[i])),
            })

        print(f"  Detections: {num_dets}")
        for i in range(num_dets):
            print(f"    det {i}: obj_id={pred_obj_ids[i]:>2d}  score={pred_scores[i]:.4f}  bbox={pred_bboxes[i].tolist()}")

        # ── Load ground truth for this frame ─────────────────────────
        sid_str = str(scene_id)
        fid_str = str(int(frame_id))
        gt_entries = scene_gt_cache.get(sid_str, {}).get(fid_str, [])
        gt_obj_ids = [g["obj_id"] for g in gt_entries]
        scene_dir = osp.join(test_root, "test", sid_str)

        gt_raw_masks = load_gt_masks_for_frame(scene_dir, frame_id, gt_obj_ids)
        # Filter out GTs whose masks could not be loaded
        valid = [(oid, m) for oid, m in zip(gt_obj_ids, gt_raw_masks) if m is not None]
        if valid:
            gt_obj_ids_arr = np.array([v[0] for v in valid])
            gt_masks_arr = np.stack([v[1] for v in valid])
        else:
            gt_obj_ids_arr = np.array([], dtype=int)
            gt_masks_arr = np.zeros((0, 1, 1), dtype=np.uint8)

        print(f"  Ground truth objects ({len(gt_obj_ids_arr)}): {gt_obj_ids_arr.tolist()}")

        # ── Binarize prediction masks ────────────────────────────────
        pred_masks_bin = np.array([force_binary_mask(m) for m in pred_masks]) if num_dets > 0 else np.zeros((0, 1, 1), dtype=np.uint8)

        # ── Per-image AP at multiple IoU thresholds ─────────────────
        img_aps_per_thresh = {}  # thresh -> {obj_id: ap}
        for t in IOU_THRESHOLDS:
            if num_dets > 0 and len(gt_obj_ids_arr) > 0:
                img_eval = evaluate_image(
                    pred_masks_bin, pred_scores, pred_obj_ids,
                    gt_masks_arr, gt_obj_ids_arr, iou_thresh=t,
                )
            else:
                img_eval = {}
                for oid in gt_obj_ids_arr:
                    img_eval[oid] = {"scores": [], "matched": [], "n_gt": 1}

            aps_at_t = {}
            for obj_id, info in img_eval.items():
                ap = compute_ap(
                    np.array(info["scores"]) if info["scores"] else np.array([]),
                    info["matched"] if info["matched"] else [],
                    info["n_gt"],
                )
                aps_at_t[obj_id] = ap
                # Accumulate globally
                global_per_class[t][obj_id]["scores"].extend(info["scores"])
                global_per_class[t][obj_id]["matched"].extend(info["matched"])
                global_per_class[t][obj_id]["n_gt"] += info["n_gt"]
            img_aps_per_thresh[t] = aps_at_t

        # Compute per-image mAP@0.5 and mAP@[0.5:0.95]
        img_aps_50 = img_aps_per_thresh.get(0.5, {})
        img_map_50 = np.mean(list(img_aps_50.values())) if img_aps_50 else 0.0
        all_thresh_maps = []
        for t in IOU_THRESHOLDS:
            aps = img_aps_per_thresh.get(t, {})
            all_thresh_maps.append(np.mean(list(aps.values())) if aps else 0.0)
        img_map_5095 = np.mean(all_thresh_maps) if all_thresh_maps else 0.0

        print(f"  Per-class AP@0.5: { {k: f'{v:.4f}' for k, v in sorted(img_aps_50.items())} }")
        print(f"  Image mAP@0.5:        {img_map_50:.4f}")
        print(f"  Image mAP@[.5:.95]:   {img_map_5095:.4f}")

        image_results.append({
            "scene_id": str(scene_id),
            "frame_id": int(frame_id),
            "num_detections": num_dets,
            "num_gt": int(len(gt_obj_ids_arr)),
            "per_class_ap_0.5": {int(k): round(v, 4) for k, v in img_aps_50.items()},
            "mAP_0.5": round(img_map_50, 4),
            "mAP_0.5_0.95": round(img_map_5095, 4),
            "scores": pred_scores.tolist(),
            "pred_obj_ids": pred_obj_ids.tolist(),
        })

        print(f"[{idx+1}/{num_images}] Done\n")
        return result

    model.test_step = types.MethodType(verbose_test_step, model)

    # ── Monkey-patch test_epoch_end to skip slow npz→json conversion ─────
    def fast_test_epoch_end(self, outputs):
        print("\n[run_inference_10] Skipping full npz→json conversion (only 10 images).")

    model.test_epoch_end = types.MethodType(fast_test_epoch_end, model)

    # ── Run inference ────────────────────────────────────────────────────
    print(f"\n[run_inference_10] Starting inference on {num_images} images for dataset={cfg.dataset_name}")
    run_start = time.time()
    trainer.test(model, dataloaders=query_dataloader)
    run_elapsed = time.time() - run_start

    # ── Compute overall mAP ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  OVERALL RESULTS  ({num_images} images, {run_elapsed:.1f}s total)")
    print(f"{'='*60}")

    # Collect all obj_ids seen across all thresholds
    all_obj_ids = set()
    for t in IOU_THRESHOLDS:
        all_obj_ids.update(global_per_class[t].keys())

    # Per-class AP at each threshold
    class_aps_per_thresh = {}  # thresh -> {obj_id: ap}
    for t in IOU_THRESHOLDS:
        class_aps_per_thresh[t] = {}
        for obj_id in sorted(all_obj_ids):
            info = global_per_class[t][obj_id]
            ap = compute_ap(
                np.array(info["scores"]) if info["scores"] else np.array([]),
                info["matched"] if info["matched"] else [],
                info["n_gt"],
            )
            class_aps_per_thresh[t][obj_id] = ap

    # Print per-class results
    print(f"\n  {'obj':>5s}  {'AP@.5':>8s}  {'AP@.75':>8s}  {'AP@[.5:.95]':>12s}  {'GT':>4s}  {'dets':>5s}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*4}  {'-'*5}")
    class_ap_5095 = {}
    for obj_id in sorted(all_obj_ids):
        ap50 = class_aps_per_thresh[0.5].get(obj_id, 0.0)
        ap75 = class_aps_per_thresh[0.75].get(obj_id, 0.0)
        ap_avg = np.mean([class_aps_per_thresh[t].get(obj_id, 0.0) for t in IOU_THRESHOLDS])
        class_ap_5095[obj_id] = ap_avg
        n_gt = global_per_class[0.5][obj_id]["n_gt"]
        n_det = len(global_per_class[0.5][obj_id]["scores"])
        print(f"  {obj_id:>5d}  {ap50:>8.4f}  {ap75:>8.4f}  {ap_avg:>12.4f}  {n_gt:>4d}  {n_det:>5d}")

    overall_map_50 = np.mean(list(class_aps_per_thresh[0.5].values())) if class_aps_per_thresh[0.5] else 0.0
    overall_map_75 = np.mean(list(class_aps_per_thresh[0.75].values())) if class_aps_per_thresh[0.75] else 0.0
    overall_map_5095 = np.mean(list(class_ap_5095.values())) if class_ap_5095 else 0.0

    print(f"\n  ** mAP@0.5       = {overall_map_50:.4f} **")
    print(f"  ** mAP@0.75      = {overall_map_75:.4f} **")
    print(f"  ** mAP@[.5:.95]  = {overall_map_5095:.4f} **")
    print(f"{'='*60}\n")

    # ── Save flat BOP-format detections for PEM/test_bop loaders ───────
    detections_path = osp.join(model.log_dir, f"{model.name_prediction_file}_{num_images}imgs.json")
    save_json_bop23(detections_path, formatted_detections)
    print(f"[run_inference_10] BOP detections saved to {detections_path}")

    # ── Save results to JSON ─────────────────────────────────────────────
    results_summary = {
        "dataset": cfg.dataset_name,
        "num_images": num_images,
        "total_runtime_s": round(run_elapsed, 2),
        "overall_mAP_0.5": round(overall_map_50, 4),
        "overall_mAP_0.75": round(overall_map_75, 4),
        "overall_mAP_0.5_0.95": round(overall_map_5095, 4),
        "per_class_AP_0.5": {int(k): round(v, 4) for k, v in class_aps_per_thresh[0.5].items()},
        "per_class_AP_0.5_0.95": {int(k): round(v, 4) for k, v in class_ap_5095.items()},
        "per_image": image_results,
    }

    results_dir = osp.join(model.log_dir, "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = osp.join(results_dir, f"results_{cfg.dataset_name}_{num_images}imgs.json")
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"[run_inference_10] Results saved to {results_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_inference()
