"""
OpenVINO version of run_inference_10.py.

Runs the ISM pipeline using OpenVINO backends (SAM + DINOv2) on the first
10 images of the BOP LM-O test set, computes per-image and overall mAP,
and produces output matching the reference format.

Usage:
    python3 run_inference_ov_10.py dataset_name=lmo
"""

import logging
import os
import os.path as osp
import json
import time
import glob
import gc
from collections import defaultdict

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from utils.inout import load_json, save_json_bop23
from utils.bbox_utils import CropResizePad, force_binary_mask
from model.utils import BatchedData, Detections, convert_npz_to_json, mask_to_rle


# ── mAP helpers (same as run_inference_10.py) ────────────────────────────────
def compute_mask_iou_matrix(pred_masks, gt_masks):
    N = pred_masks.shape[0]
    M = gt_masks.shape[0]
    pred_flat = pred_masks.reshape(N, -1).astype(bool)
    gt_flat = gt_masks.reshape(M, -1).astype(bool)
    intersection = (pred_flat[:, None, :] & gt_flat[None, :, :]).sum(axis=2)
    union = (pred_flat[:, None, :] | gt_flat[None, :, :]).sum(axis=2)
    iou = np.where(union > 0, intersection / union, 0.0)
    return iou


def compute_ap(scores, matched, n_gt):
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    matched = np.array(matched, dtype=bool)[order]
    tp = np.cumsum(matched)
    fp = np.cumsum(~matched)
    recall = tp / n_gt
    precision = tp / (tp + fp)
    mrec = np.concatenate(([0.0], recall, [recall[-1]]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def evaluate_image(pred_masks, pred_scores, pred_obj_ids,
                   gt_masks, gt_obj_ids, iou_thresh=0.5):
    per_class = {}
    all_ids = set(gt_obj_ids) | set(pred_obj_ids)
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
    import torchvision.transforms as T

    OmegaConf.set_struct(cfg, False)

    print("[run_inference_ov_10] Initializing OpenVINO ISM pipeline")

    # ── Init OV model ────────────────────────────────────────────────────
    from infer_ism_ov import init_model_ov

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ov_model_dir = osp.join(BASE_DIR, "checkpoints", "ov_models")
    ov_device = os.environ.get("OV_DEVICE", "GPU")
    sam_device = os.environ.get("OV_SAM_DEVICE", ov_device)

    t_init = time.time()
    model, _, torch_device, selected_poses, proposal_processor = init_model_ov(
        ov_model_dir=ov_model_dir,
        ov_device=sam_device,
        segmentor_model_name="sam",
        stability_score_thresh=0.85,
        points_per_side=32,
    )
    # Re-compile DINOv2 on the requested device if different from SAM
    if ov_device != sam_device:
        import openvino as _ov
        _core = _ov.Core()
        dinov2_xml = osp.join(ov_model_dir, "dinov2_vitl14.xml")
        _config = ({"PERFORMANCE_HINT": "LATENCY",
                    "NUM_STREAMS": "1",
                    "INFERENCE_PRECISION_HINT": "f16"}
                   if ov_device.upper().startswith("GPU") else {})
        dinov2_compiled = _core.compile_model(_core.read_model(dinov2_xml),
                                              ov_device, _config)
        model.descriptor_model.dinov2_compiled = dinov2_compiled
        print(f"  DINOv2 re-compiled on {ov_device}")
    init_time = time.time() - t_init
    print(f"  OV model init: {init_time:.1f}s (SAM={sam_device}, DINOv2={ov_device})")

    # ── Load query dataset (same as run_inference_10.py) ─────────────────
    default_query_dataloader_config = cfg.data.query_dataloader.copy()
    default_ref_dataloader_config = cfg.data.reference_dataloader.copy()

    if cfg.dataset_name in ["hb", "tless"]:
        default_query_dataloader_config.split = "test_primesense"
    else:
        default_query_dataloader_config.split = "test"
    default_query_dataloader_config.root_dir += f"{cfg.dataset_name}"
    query_dataset = instantiate(default_query_dataloader_config)

    # Determine max_images from env, default 10, 0 means all images
    max_images_env = os.environ.get("MAX_IMAGES")
    try:
        max_images = int(max_images_env) if max_images_env is not None else 10
    except ValueError:
        max_images = 10
    start_image = int(os.environ.get("START_IMAGE", "0"))
    total_images = len(query_dataset)
    if max_images == 0:
        num_images = total_images - start_image
    else:
        num_images = min(max_images, total_images - start_image)
    query_dataset = Subset(query_dataset, list(range(start_image, start_image + num_images)))
    print(f"[run_inference_ov_10] Dataset: {cfg.dataset_name}, images {start_image}..{start_image+num_images-1} of {total_images} (max_images={max_images})")

    query_dataloader = DataLoader(
        query_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
    )

    # ── Load reference dataset (templates) ───────────────────────────────
    ref_dataloader_config = default_ref_dataloader_config.copy()
    if cfg.model.onboarding_config.rendering_type == "pyrender":
        ref_dataloader_config.template_dir += f"templates_pyrender/{cfg.dataset_name}"
    elif cfg.model.onboarding_config.rendering_type == "pbr":
        ref_dataloader_config._target_ = "provider.bop_pbr.BOPTemplatePBR"
        ref_dataloader_config.root_dir = f"{default_query_dataloader_config.root_dir}"
        ref_dataloader_config.template_dir += f"templates_pyrender/{cfg.dataset_name}"
    ref_dataset = instantiate(ref_dataloader_config)
    if cfg.model.onboarding_config.rendering_type == "pbr":
        ref_dataset.load_processed_metaData(reset_metaData=True)

    model.ref_dataset = ref_dataset
    model.dataset_name = cfg.dataset_name
    model.ref_obj_names = cfg.data.datasets[cfg.dataset_name].obj_names

    # ── Compute reference descriptors ────────────────────────────────────
    print("[run_inference_ov_10] Computing reference descriptors...")
    t0 = time.time()

    # Check for cached descriptors
    cache_dir = ref_dataset.template_dir
    os.makedirs(cache_dir, exist_ok=True)
    descriptors_path = osp.join(cache_dir, "descriptors.pth")
    appe_descriptors_path = osp.join(cache_dir, "descriptors_appe.pth")
    if cfg.model.onboarding_config.rendering_type == "pbr":
        descriptors_path = descriptors_path.replace(".pth", "_pbr.pth")
        appe_descriptors_path = appe_descriptors_path.replace(".pth", "_pbr.pth")

    model.ref_data = {}

    if os.path.exists(descriptors_path):
        model.ref_data["descriptors"] = torch.load(descriptors_path, map_location="cpu")
        print(f"  Loaded cached descriptors: {model.ref_data['descriptors'].shape}")
    else:
        desc_batched = BatchedData(None)
        for idx in tqdm(range(len(ref_dataset)), desc="  Computing descriptors"):
            ref_imgs = ref_dataset[idx]["templates"]
            ref_feats = model.descriptor_model.compute_features(
                ref_imgs, token_name="x_norm_clstoken"
            )
            desc_batched.append(ref_feats)
        desc_batched.stack()
        model.ref_data["descriptors"] = desc_batched.data
        torch.save(model.ref_data["descriptors"], descriptors_path)
        print(f"  Computed + saved descriptors: {model.ref_data['descriptors'].shape}")

    if os.path.exists(appe_descriptors_path):
        model.ref_data["appe_descriptors"] = torch.load(appe_descriptors_path, map_location="cpu")
        print(f"  Loaded cached appe_descriptors: {model.ref_data['appe_descriptors'].shape}")
    else:
        appe_batched = BatchedData(None)
        for idx in tqdm(range(len(ref_dataset)), desc="  Computing appe descriptors"):
            ref_imgs = ref_dataset[idx]["templates"]
            ref_masks = ref_dataset[idx]["template_masks"]
            ref_feats = model.descriptor_model.compute_masked_patch_feature(
                ref_imgs, ref_masks
            )
            appe_batched.append(ref_feats)
        appe_batched.stack()
        model.ref_data["appe_descriptors"] = appe_batched.data
        torch.save(model.ref_data["appe_descriptors"], appe_descriptors_path)
        print(f"  Computed + saved appe_descriptors: {model.ref_data['appe_descriptors'].shape}")

    # ── Compute reference pointclouds ────────────────────────────────────
    pointcloud_path = osp.join(ref_dataset.template_dir, "pointcloud.pth")
    obj_pose_path = osp.join(ref_dataset.template_dir, "template_poses.npy")

    from utils.poses.pose_utils import (
        get_obj_poses_from_template_level,
        load_index_level_in_level2,
    )

    if os.path.exists(obj_pose_path):
        poses = torch.tensor(np.load(obj_pose_path)).to(torch.float32)
    else:
        template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
        template_poses[:, :3, 3] *= 0.4
        poses = torch.tensor(template_poses).to(torch.float32)
        np.save(obj_pose_path, template_poses)

    model.ref_data["poses"] = poses[ref_dataset.index_templates, :, :]

    if os.path.exists(pointcloud_path):
        model.ref_data["pointcloud"] = torch.load(pointcloud_path, map_location="cpu")
        print(f"  Loaded cached pointcloud: {model.ref_data['pointcloud'].shape}")
    else:
        import trimesh
        mesh_path = osp.join(default_query_dataloader_config.root_dir, "models")
        pc_batched = BatchedData(None)
        all_pc_idx = [1, 5, 6, 8, 9, 10, 11, 12] if cfg.dataset_name == "lmo" else None
        for idx in tqdm(range(len(ref_dataset)), desc="  Computing pointclouds"):
            pc_id = all_pc_idx[idx] if all_pc_idx else idx + 1
            mesh = trimesh.load_mesh(osp.join(mesh_path, f'obj_{pc_id:06d}.ply'))
            pts = mesh.sample(model.pointcloud_sample_num).astype(np.float32) / 1000.0
            pc_batched.append(torch.tensor(pts))
        pc_batched.stack()
        model.ref_data["pointcloud"] = pc_batched.data
        torch.save(model.ref_data["pointcloud"], pointcloud_path)
        print(f"  Computed + saved pointcloud: {model.ref_data['pointcloud'].shape}")

    ref_time = time.time() - t0
    print(f"  Reference data ready ({ref_time:.1f}s)")

    # ── Load ground-truth metadata ───────────────────────────────────────
    test_root = default_query_dataloader_config.root_dir
    meta_path = osp.join(test_root, "test_metaData.json")
    meta = load_json(meta_path)
    scene_ids_list = meta["scene_id"][:num_images]
    frame_ids_list = meta["frame_id"][:num_images]

    scene_gt_cache = {}
    for sid in set(scene_ids_list):
        scene_dir = osp.join(test_root, "test", str(sid))
        scene_gt_cache[sid] = load_json(osp.join(scene_dir, "scene_gt.json"))

    # ── Inverse RGB transform (same as detector.py) ──────────────────────
    inv_rgb_transform = T.Compose([
        T.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        ),
    ])

    # ── mAP accumulators ─────────────────────────────────────────────────
    IOU_THRESHOLDS = [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]
    global_per_class = {
        t: defaultdict(lambda: {"scores": [], "matched": [], "n_gt": 0})
        for t in IOU_THRESHOLDS
    }
    image_results = []
    formatted_detections = []
    npz_file_paths = []  # Collect npz paths for later conversion to JSON

    # ── Create output directory ──────────────────────────────────────────
    log_dir = "./log/sam_ov"
    pred_dir = osp.join(log_dir, f"predictions/{cfg.dataset_name}/result_{cfg.dataset_name}")
    os.makedirs(pred_dir, exist_ok=True)

    # ── Main inference loop ──────────────────────────────────────────────
    print(f"\n[run_inference_ov_10] Starting inference on {num_images} images")
    run_start = time.time()

    # GPU memory monitoring
    import openvino as ov
    ov_core = ov.Core()
    def gpu_mem_mb():
        try:
            stats = ov_core.get_property("GPU", "GPU_MEMORY_STATISTICS")
            return {k: v / 1e6 for k, v in stats.items()}
        except Exception:
            return {}

    for idx, batch in enumerate(tqdm(query_dataloader, desc="Testing DataLoader 0")):
        scene_id = batch["scene_id"][0]
        frame_id = batch["frame_id"][0]

        mem = gpu_mem_mb()
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{num_images}] Processing scene={scene_id} frame={frame_id}")
        if mem:
            print(f"  GPU mem: {mem}")
        print(f"{'='*60}")

        # ── Convert image to numpy ───────────────────────────────────
        image_np = inv_rgb_transform(batch["image"][0]).cpu().numpy().transpose(1, 2, 0)
        image_np = np.uint8(image_np.clip(0, 1) * 255)

        # ── Proposal stage ───────────────────────────────────────────
        proposal_start = time.time()
        proposals = model.segmentor_model.generate_masks(image_np)
        detections = Detections(proposals)
        detections.remove_very_small_detections(
            config=model.post_processing_config.mask_post_processing
        )
        query_descriptors, query_appe_descriptors = model.descriptor_model(
            image_np, detections
        )
        proposal_time = time.time() - proposal_start

        # ── Matching stage ───────────────────────────────────────────
        matching_start = time.time()
        (idx_selected, pred_idx_objects, semantic_score, best_template) = \
            model.compute_semantic_score(query_descriptors)

        detections.filter(idx_selected)
        query_appe_descriptors = query_appe_descriptors[idx_selected, :]

        appe_scores, ref_aux_descriptor = model.compute_appearance_score(
            best_template, pred_idx_objects, query_appe_descriptors
        )

        # Geometric score
        batch_geo = {
            "depth": batch["depth"].to(torch_device),
            "cam_intrinsic": batch["cam_intrinsic"].to(torch_device),
            "depth_scale": batch["depth_scale"],
        }
        image_uv = model.project_template_to_image(
            best_template, pred_idx_objects, batch_geo, detections.masks
        )
        geometric_score, visible_ratio = model.compute_geometric_score(
            image_uv, detections, query_appe_descriptors, ref_aux_descriptor,
            visible_thred=model.visible_thred,
        )

        # Final score
        final_score = (semantic_score + appe_scores + geometric_score * visible_ratio) / \
                      (1 + 1 + visible_ratio)

        detections.add_attribute("scores", final_score)
        detections.add_attribute("object_ids", pred_idx_objects)
        detections.apply_nms_per_object_id(
            nms_thresh=model.post_processing_config.nms_thresh
        )
        matching_time = time.time() - matching_start

        # ── Convert to numpy and save ────────────────────────────────
        detections.to_numpy()

        file_path = osp.join(pred_dir, f"scene{scene_id}_frame{frame_id}")
        detections.save_to_file(
            scene_id=int(scene_id),
            frame_id=int(frame_id),
            runtime=proposal_time + matching_time,
            file_path=file_path,
            dataset_name=cfg.dataset_name,
            return_results=True,
        )
        npz_file_paths.append(file_path + ".npz")  # Collect for JSON conversion
        np.savez(
            file_path + "_runtime",
            proposal_stage=proposal_time,
            matching_stage=matching_time,
        )

        # ── Print detections ─────────────────────────────────────────
        from model.utils import lmo_object_ids
        pred_scores = detections.scores
        pred_obj_ids_raw = detections.object_ids
        # Map 0-indexed predictions to BOP object IDs
        if cfg.dataset_name == "lmo":
            pred_obj_ids = lmo_object_ids[pred_obj_ids_raw]
        else:
            pred_obj_ids = pred_obj_ids_raw + 1
        pred_masks = detections.masks
        pred_bboxes = detections.boxes
        num_dets = len(pred_scores) if pred_scores is not None else 0
        runtime_total = float(proposal_time + matching_time)

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

        print(f"  Runtime: proposal={proposal_time:.3f}s  matching={matching_time:.3f}s")
        print(f"  Detections: {num_dets}")
        for i in range(num_dets):
            print(f"    det {i}: obj_id={pred_obj_ids[i]:>2d}  score={pred_scores[i]:.4f}  bbox={pred_bboxes[i].tolist()}")

        # ── Load ground truth ────────────────────────────────────────
        sid_str = str(scene_id)
        fid_str = str(int(frame_id))
        gt_entries = scene_gt_cache.get(sid_str, {}).get(fid_str, [])
        gt_obj_ids = [g["obj_id"] for g in gt_entries]
        scene_dir = osp.join(test_root, "test", sid_str)

        gt_raw_masks = load_gt_masks_for_frame(scene_dir, frame_id, gt_obj_ids)
        valid = [(oid, m) for oid, m in zip(gt_obj_ids, gt_raw_masks) if m is not None]
        if valid:
            gt_obj_ids_arr = np.array([v[0] for v in valid])
            gt_masks_arr = np.stack([v[1] for v in valid])
        else:
            gt_obj_ids_arr = np.array([], dtype=int)
            gt_masks_arr = np.zeros((0, 1, 1), dtype=np.uint8)

        print(f"  Ground truth objects ({len(gt_obj_ids_arr)}): {gt_obj_ids_arr.tolist()}")

        # ── Binarize prediction masks ────────────────────────────────
        pred_masks_bin = np.array([force_binary_mask(m) for m in pred_masks]) \
            if num_dets > 0 else np.zeros((0, 1, 1), dtype=np.uint8)

        # ── Per-image mAP ────────────────────────────────────────────
        img_aps_per_thresh = {}
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
                global_per_class[t][obj_id]["scores"].extend(info["scores"])
                global_per_class[t][obj_id]["matched"].extend(info["matched"])
                global_per_class[t][obj_id]["n_gt"] += info["n_gt"]
            img_aps_per_thresh[t] = aps_at_t

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
        })
        print(f"[{idx+1}/{num_images}] Done\n")

        # Free per-image tensors to avoid GPU memory accumulation
        del detections, proposals, query_descriptors, query_appe_descriptors
        del pred_masks, pred_masks_bin, pred_scores, pred_obj_ids, pred_bboxes
        del final_score, semantic_score, appe_scores, geometric_score, visible_ratio
        del idx_selected, pred_idx_objects, best_template, ref_aux_descriptor
        del image_uv, batch_geo, image_np
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    run_elapsed = time.time() - run_start

    # ── Overall results ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  OVERALL RESULTS  ({num_images} images, {run_elapsed:.1f}s total)")
    print(f"{'='*60}")

    all_obj_ids = set()
    for t in IOU_THRESHOLDS:
        all_obj_ids.update(global_per_class[t].keys())

    class_aps_per_thresh = {}
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
    detections_path = osp.join(log_dir, f"result_{cfg.dataset_name}_{num_images}imgs.json")
    save_json_bop23(detections_path, formatted_detections)
    print(f"[run_inference_ov_10] BOP detections saved to {detections_path}")

    # ── Save results ─────────────────────────────────────────────────────
    results_dir = osp.join(log_dir, "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    results_summary = {
        "dataset": cfg.dataset_name,
        "num_images": num_images,
        "total_runtime_s": round(run_elapsed, 2),
        "init_time_s": round(init_time, 2),
        "ov_device": ov_device,
        "overall_mAP_0.5": round(overall_map_50, 4),
        "overall_mAP_0.75": round(overall_map_75, 4),
        "overall_mAP_0.5_0.95": round(overall_map_5095, 4),
        "per_class_AP_0.5": {int(k): round(v, 4) for k, v in class_aps_per_thresh[0.5].items()},
        "per_class_AP_0.5_0.95": {int(k): round(v, 4) for k, v in class_ap_5095.items()},
        "per_image": image_results,
    }
    results_path = osp.join(results_dir, f"results_{cfg.dataset_name}_{num_images}imgs.json")
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"[run_inference_ov_10] Results saved to {results_path}")

    # ── Convert npz detections to JSON ──────────────────────────────────
    print(f"\n[run_inference_ov_10] Converting {len(npz_file_paths)} npz files to JSON format")
    all_detections = []
    for idx, npz_path in enumerate(npz_file_paths):
        try:
            detections_for_image = convert_npz_to_json(idx, npz_file_paths)
            all_detections.extend(detections_for_image)
        except Exception as e:
            print(f"  Warning: Failed to convert {npz_path}: {e}")
    
    detections_json_path = osp.join(results_dir, f"detections_{cfg.dataset_name}_{num_images}imgs.json")
    with open(detections_json_path, "w") as f:
        json.dump(all_detections, f, indent=2)
    print(f"[run_inference_ov_10] Detections saved to {detections_json_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_inference()
