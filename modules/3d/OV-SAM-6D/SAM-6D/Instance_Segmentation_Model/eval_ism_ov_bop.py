#!/usr/bin/env python3
"""
BOP dataset evaluation for OpenVINO ISM pipeline.

Runs the full ISM inference (segmentation → DINOv2 → scoring) on the BOP
LM-O test set, computes mAP @ IoU[0.50:0.95] per testcase, and aggregates
per-object and overall metrics.

Example:
    python3 eval_ism_ov_bop.py \
        --bop_dir /workspace/SAM-6D/Data/BOP/lmo \
        --ov_device GPU --segmentor_model fastsam \
        --max_images 3 --obj_ids 5 6 9 12 --batch_size 4
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from collections import defaultdict

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

import cv2

# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from infer_ism_ov import init_model_ov, init_model_ov_ext
from model.utils import Detections, convert_npz_to_json
from utils.inout import save_json_bop23
from segment_anything.utils.amg import rle_to_mask
from eval_utils import compute_map_iou, load_gt_masks_for_object, aggregate_bop_maps


# ===================================================================
#  Helper functions (mesh sampling, template loading, scoring)
# ===================================================================

def _sample_mesh_surface(verts, faces, n_pts):
    """Sample *n_pts* points uniformly on a triangle mesh surface."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    probs = areas / areas.sum()
    tri_idx = np.random.choice(len(faces), size=n_pts, p=probs)
    r1 = np.sqrt(np.random.rand(n_pts, 1))
    r2 = np.random.rand(n_pts, 1)
    return ((1 - r1) * verts[faces[tri_idx, 0]]
            + r1 * (1 - r2) * verts[faces[tri_idx, 1]]
            + r1 * r2 * verts[faces[tri_idx, 2]])


def load_object_pointcloud(model, cad_path, selected_poses, device):
    """Load CAD pointcloud and poses into *model.ref_data*."""
    model.ref_data["poses"] = selected_poses
    ply = PlyData.read(cad_path)
    verts = np.column_stack(
        [ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"]]
    ).astype(np.float64)
    faces = np.vstack(ply["face"]["vertex_indices"])
    pts = _sample_mesh_surface(verts, faces, 2048).astype(np.float32) / 1000.0
    model.ref_data["pointcloud"] = (
        torch.tensor(pts).unsqueeze(0).data.to(device)
    )


def load_template_descriptors(model, template_dir, proposal_processor, device):
    """Load pre-rendered templates and compute DINOv2 descriptors.

    Populates model.ref_data["descriptors"] and
    model.ref_data["appe_descriptors"].

    Returns the number of templates loaded (0 if template_dir is empty).
    """
    n_tpl = len(glob.glob(os.path.join(template_dir, "*.npy")))
    if n_tpl == 0:
        return 0

    tpl_boxes, tpl_masks_list, tpl_images = [], [], []
    for idx in range(n_tpl):
        im = Image.open(os.path.join(template_dir, f"rgb_{idx}.png"))
        mk = Image.open(os.path.join(template_dir, f"mask_{idx}.png"))
        tpl_boxes.append(mk.getbbox())
        im_t = torch.from_numpy(
            np.array(im.convert("RGB")) / 255
        ).float()
        mk_t = torch.from_numpy(
            np.array(mk.convert("L")) / 255
        ).float()
        tpl_images.append(im_t * mk_t[:, :, None])
        tpl_masks_list.append(mk_t.unsqueeze(-1))

    tpl_imgs = torch.stack(tpl_images).permute(0, 3, 1, 2)
    tpl_msks = torch.stack(tpl_masks_list).permute(0, 3, 1, 2)
    tpl_bxs = torch.tensor(np.array(tpl_boxes))
    tpl_imgs = proposal_processor(images=tpl_imgs, boxes=tpl_bxs).to(device)
    tpl_msks_crop = proposal_processor(
        images=tpl_msks, boxes=tpl_bxs
    ).to(device)

    model.ref_data = {}
    model.ref_data["descriptors"] = (
        model.descriptor_model.compute_features(
            tpl_imgs, token_name="x_norm_clstoken"
        ).unsqueeze(0).data
    )
    model.ref_data["appe_descriptors"] = (
        model.descriptor_model.compute_masked_patch_feature(
            tpl_imgs, tpl_msks_crop[:, 0, :, :]
        ).unsqueeze(0).data
    )
    return n_tpl


def run_scoring_pipeline(model, detections, query_cls, query_patch,
                         depth_path, cam_info, device):
    """Run the full ISM scoring pipeline on proposals.

    Returns
    -------
    (final_score, det_score, idx_sel, score_label) or None if no proposals
    survive the semantic filter.
    """
    # Clone detections + descriptors so originals stay intact
    det_score = Detections({
        key: getattr(detections, key).clone()
        if torch.is_tensor(getattr(detections, key))
        else getattr(detections, key)
        for key in detections.keys
    })
    qcls = query_cls.clone()
    qpatch = query_patch.clone()

    # -- Semantic score + proposal filter ---------------------------
    (idx_sel, pred_idx_obj, sem_score, best_tpl) = \
        model.compute_semantic_score(qcls)
    if len(idx_sel) == 0:
        return None
    det_score.filter(idx_sel)
    qpatch_sel = qpatch[idx_sel, :]

    # -- Appearance score -------------------------------------------
    appe_score, ref_aux = model.compute_appearance_score(
        best_tpl, pred_idx_obj, qpatch_sel
    )

    # -- Geometric score (requires depth) ---------------------------
    has_depth = (depth_path and os.path.isfile(depth_path)
                 and cam_info is not None)
    if has_depth:
        depth = np.array(imageio.imread(depth_path)).astype(np.int32)
        cam_K = np.array(cam_info["cam_K"]).reshape((3, 3))
        depth_scale = np.array(cam_info["depth_scale"])
        batch = {
            "depth": torch.from_numpy(depth).unsqueeze(0).to(device),
            "cam_intrinsic": torch.from_numpy(cam_K).unsqueeze(0).to(device),
            "depth_scale": torch.from_numpy(depth_scale).unsqueeze(0).to(device),
        }
        image_uv = model.project_template_to_image(
            best_tpl, pred_idx_obj, batch, det_score.masks
        )
        geo_score, vis_ratio = model.compute_geometric_score(
            image_uv, det_score, qpatch_sel, ref_aux,
            visible_thred=model.visible_thred,
        )
        final_score = (sem_score + appe_score + geo_score * vis_ratio) / \
                       (1 + 1 + vis_ratio)
        score_label = "sem+appe+geo"
    else:
        final_score = (sem_score + appe_score) / 2.0
        score_label = "sem+appe"

    return final_score, det_score, idx_sel, score_label


# ===================================================================
#  CLI
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="BOP dataset evaluation — OpenVINO ISM pipeline"
    )
    # --- args shared with infer_ism_ov.py --------------------------
    p.add_argument("--ov_model_dir",
                   default=os.path.join(BASE_DIR, "checkpoints/ov_models"),
                   help="Directory containing OV IR files")
    p.add_argument("--ov_device", default="GPU",
                   help="OpenVINO device: CPU or GPU")
    p.add_argument("--ext", action="store_true",
                   help="Use extended models (baked pre/post-processing)")
    p.add_argument("--segmentor_model", default="fastsam",
                   choices=["sam", "fastsam"],
                   help="Segmentor: sam or fastsam")
    p.add_argument("--points_per_side", type=int, default=32,
                   help="SAM grid size NxN (default: 32)")
    p.add_argument("--stability_score_thresh", type=float, default=0.85,
                   help="SAM stability_score_thresh (default: 0.85)")
    p.add_argument("--points_per_batch", type=int, default=64,
                   help="SAM decoder points per call (default: 64)")
    p.add_argument("--chunk_size", type=int, default=16,
                   help="DINOv2 proposals per call (default: 16)")

    # --- BOP-specific args -----------------------------------------
    p.add_argument("--bop_dir", required=True,
                   help="BOP dataset root (e.g. /workspace/SAM-6D/Data/BOP/lmo)")
    p.add_argument("--max_images", type=int, default=None,
                   help="Limit to first N unique images")
    p.add_argument("--obj_ids", type=int, nargs="+", default=None,
                   help="Restrict to these object IDs")
    p.add_argument("--max_objects", type=int, default=None,
                   help="Limit to first N object IDs")
    p.add_argument("--skip_existing", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Skip testcases whose result JSON already exists")
    p.add_argument("--batch_size", type=int, default=4,
                   help="(Informational, not used for ISM)")
    p.add_argument("--output_dir", default=None,
                   help="Output directory (default: <bop_dir>/bop/ism_ov_<dev>_<seg>_results)")
    p.add_argument("--templates_root", default=None,
                   help="Root dir for pre-rendered templates "
                        "(default: <bop_dir>/eval_output)")
    p.add_argument("--vis_ism", action="store_true", default=False,
                   help="Save vis_ism.png visualisation for each testcase")
    return p.parse_args()


# ===================================================================
#  Main evaluation loop
# ===================================================================

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # -- random seeds for reproducibility -----------------------
    np.random.seed(42)
    torch.manual_seed(42)

    # -- Resolve paths ----------------------------------------------
    scene_dir = os.path.join(args.bop_dir, "test", "000002")
    scene_gt_path = os.path.join(scene_dir, "scene_gt.json")
    scene_cam_path = os.path.join(scene_dir, "scene_camera.json")
    mask_visib_dir = os.path.join(scene_dir, "mask_visib")
    rgb_dir = os.path.join(scene_dir, "rgb")
    depth_dir = os.path.join(scene_dir, "depth")
    models_dir = os.path.join(args.bop_dir, "models")
    templates_root = args.templates_root or os.path.join(
        args.bop_dir, "eval_output"
    )

    device_tag = args.ov_device.lower()
    seg_tag = args.segmentor_model
    output_dir = args.output_dir or os.path.join(
        args.bop_dir, "bop",
        f"ism_ov_{device_tag}_{seg_tag}_results",
    )
    os.makedirs(output_dir, exist_ok=True)

    # -- Load BOP annotations ---------------------------------------
    with open(scene_gt_path) as f:
        gt_data = json.load(f)
    with open(scene_cam_path) as f:
        cam_data = json.load(f)

    # -- Determine target objects -----------------------------------
    all_obj_ids = sorted({
        e["obj_id"]
        for entries in gt_data.values()
        for e in entries
    })
    if args.obj_ids:
        target_obj_ids = [o for o in args.obj_ids if o in all_obj_ids]
    else:
        target_obj_ids = list(all_obj_ids)
    if args.max_objects:
        target_obj_ids = target_obj_ids[:args.max_objects]
    target_set = set(target_obj_ids)

    # -- Build testcases (image_id, obj_id) -------------------------
    # Ordered by image_id to maximise segmentation cache reuse.
    testcases = []
    for img_id_str in sorted(gt_data.keys(), key=int):
        img_id = int(img_id_str)
        for entry in gt_data[img_id_str]:
            if entry["obj_id"] in target_set:
                testcases.append({
                    "image_id": img_id,
                    "obj_id": entry["obj_id"],
                })

    if args.max_images is not None:
        unique_imgs = sorted({tc["image_id"] for tc in testcases})
        keep_imgs = set(unique_imgs[:args.max_images])
        testcases = [tc for tc in testcases if tc["image_id"] in keep_imgs]

    n_total = len(testcases)

    # -- Header -----------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  BOP ISM Evaluation — OpenVINO Pipeline")
    print(f"{'='*65}")
    print(f"  BOP dir       : {args.bop_dir}")
    print(f"  Device        : {args.ov_device} {'(ext)' if args.ext else ''}")
    print(f"  Segmentor     : {args.segmentor_model}")
    print(f"  Objects       : {target_obj_ids}")
    print(f"  Testcases     : {n_total}")
    print(f"  Max images    : {args.max_images or 'all'}")
    print(f"  Output        : {output_dir}")
    print(f"  Skip existing : {args.skip_existing}")
    print(f"  Templates root: {templates_root}")
    print()

    if n_total == 0:
        print("No testcases to evaluate.")
        return

    # -- Init OV ISM pipeline ---------------------------------------
    print("Initializing OV ISM pipeline ...")
    t_init_start = time.time()
    init_fn = init_model_ov_ext if args.ext else init_model_ov
    model, cfg, device, selected_poses, proposal_processor = init_fn(
        ov_model_dir=args.ov_model_dir,
        ov_device=args.ov_device,
        segmentor_model_name=args.segmentor_model,
        stability_score_thresh=args.stability_score_thresh,
        points_per_side=args.points_per_side,
    )
    init_time = time.time() - t_init_start
    print(f"  Model init: {init_time:.1f}s\n")

    # -- Caches -----------------------------------------------------
    # Image-level: segmentation + DINOv2 query descriptors
    img_cache = {}          # image_id -> (detections, query_cls, query_patch)
    # Object-level: template + pointcloud ref_data
    obj_ref_cache = {}      # obj_id -> dict (cloned model.ref_data)

    # -- Tracking ---------------------------------------------------
    maps_by_obj = defaultdict(list)
    timings = []
    n_completed = 0
    n_skipped = 0
    t_eval_start = time.time()

    for tc_idx, tc in enumerate(testcases):
        img_id = tc["image_id"]
        obj_id = tc["obj_id"]
        result_name = f"img{img_id:06d}_obj{obj_id:06d}.json"
        result_path = os.path.join(output_dir, result_name)

        # -- skip_existing ------------------------------------------
        if args.skip_existing and os.path.exists(result_path):
            try:
                with open(result_path) as f:
                    existing = json.load(f)
                maps_by_obj[obj_id].append(existing.get("map", 0.0))
                timings.append(existing.get("timing_total", 0.0))
            except Exception:
                pass
            n_completed += 1
            n_skipped += 1
            if n_completed % 500 == 0:
                _print_progress(n_completed, n_total, maps_by_obj, timings)
            continue

        t_tc_start = time.time()
        timing = {}

        # ---- Step 1: Segmentation (cached per image) --------------
        if img_id not in img_cache:
            t0 = time.time()
            rgb_path = os.path.join(rgb_dir, f"{img_id:06d}.png")
            rgb = np.array(Image.open(rgb_path).convert("RGB"))
            detections = model.segmentor_model.generate_masks(rgb)
            detections = Detections(detections)
            timing["segmentation"] = time.time() - t0

            t0 = time.time()
            query_cls, query_patch = model.descriptor_model(rgb, detections)
            timing["descriptor"] = time.time() - t0

            img_cache[img_id] = (detections, query_cls, query_patch)
            # Evict old entries to save memory (keep only current image)
            for k in [k for k in img_cache if k != img_id]:
                del img_cache[k]
        else:
            detections, query_cls, query_patch = img_cache[img_id]
            timing["segmentation"] = 0.0
            timing["descriptor"] = 0.0

        n_proposals = len(detections.masks)

        # ---- Step 2: Load template descriptors (cached per obj) ---
        t0 = time.time()
        template_dir = os.path.join(
            templates_root, f"obj_{obj_id:06d}", "templates"
        )
        cad_path = os.path.join(models_dir, f"obj_{obj_id:06d}.ply")

        if not os.path.isdir(template_dir):
            print(f"  [SKIP] img={img_id} obj={obj_id}: "
                  f"no templates at {template_dir}")
            continue
        if not os.path.isfile(cad_path):
            print(f"  [SKIP] img={img_id} obj={obj_id}: "
                  f"no CAD at {cad_path}")
            continue

        if obj_id in obj_ref_cache:
            # Restore cached ref_data
            model.ref_data = {
                k: v.clone() if torch.is_tensor(v) else v
                for k, v in obj_ref_cache[obj_id].items()
            }
        else:
            load_template_descriptors(
                model, template_dir, proposal_processor, device
            )
            load_object_pointcloud(model, cad_path, selected_poses, device)
            # Cache for reuse
            obj_ref_cache[obj_id] = {
                k: v.clone() if torch.is_tensor(v) else v
                for k, v in model.ref_data.items()
            }
        timing["template_load"] = time.time() - t0

        # ---- Step 3: Scoring pipeline -----------------------------
        t0 = time.time()
        depth_path = os.path.join(depth_dir, f"{img_id:06d}.png")
        cam_info = cam_data.get(str(img_id))

        result = run_scoring_pipeline(
            model, detections, query_cls, query_patch,
            depth_path, cam_info, device,
        )
        timing["scoring"] = time.time() - t0

        # ---- Step 4: mAP evaluation ------------------------------
        t0 = time.time()
        gt_masks = load_gt_masks_for_object(
            mask_visib_dir, gt_data, img_id, obj_id
        )

        if result is None:
            metrics = {"map": 0.0, "best_iou": 0.0,
                       "n_pred": 0, "n_gt": len(gt_masks)}
            n_after_sem = 0
            n_detections = 0
            score_label = "n/a"
        else:
            final_score, det_score, idx_sel, score_label = result

            # Build numpy detection masks for mAP
            obj_ids_tensor = torch.full(
                (len(final_score),), obj_id - 1, dtype=torch.long
            )
            det_score.add_attribute("scores", final_score)
            det_score.add_attribute("object_ids", obj_ids_tensor)
            det_np = Detections({
                key: getattr(det_score, key).clone()
                if torch.is_tensor(getattr(det_score, key))
                else getattr(det_score, key)
                for key in det_score.keys
            })
            det_np.to_numpy()
            n_after_sem = len(final_score)
            n_detections = len(det_np.masks) if det_np.masks is not None else 0

            # ---- Save detection_ism.npz + detection_ism.json ------
            det_save_path = os.path.join(
                output_dir,
                f"img{img_id:06d}_obj{obj_id:06d}_detection_ism",
            )
            det_np.save_to_file(
                0, img_id, 0, det_save_path, "Custom",
                return_results=False,
            )
            det_list = convert_npz_to_json(
                idx=0, list_npz_paths=[det_save_path + ".npz"]
            )
            save_json_bop23(det_save_path + ".json", det_list)

            # ---- Optional vis_ism.png -----------------------------
            if args.vis_ism and det_list:
                rgb_path_vis = os.path.join(rgb_dir, f"{img_id:06d}.png")
                rgb_vis = np.array(Image.open(rgb_path_vis).convert("RGB"))
                best_det = max(det_list, key=lambda d: d["score"])
                gray = cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2GRAY)
                overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(
                    np.float32)
                bmask = rle_to_mask(best_det["segmentation"])
                alpha_v = 0.33
                overlay[bmask, 0] = np.clip(
                    alpha_v * 255 + (1 - alpha_v) * overlay[bmask, 0], 0, 255)
                overlay[bmask, 1] = np.clip(
                    alpha_v * 105 + (1 - alpha_v) * overlay[bmask, 1], 0, 255)
                overlay[bmask, 2] = np.clip(
                    alpha_v * 0 + (1 - alpha_v) * overlay[bmask, 2], 0, 255)
                pred_pil = Image.fromarray(overlay.astype(np.uint8))
                rgb_pil = Image.fromarray(rgb_vis)
                h, w = rgb_vis.shape[:2]
                concat = Image.new("RGB", (w * 2, h))
                concat.paste(rgb_pil, (0, 0))
                concat.paste(pred_pil, (w, 0))
                vis_path = os.path.join(
                    output_dir,
                    f"img{img_id:06d}_obj{obj_id:06d}_vis_ism.png",
                )
                concat.save(vis_path)

            if gt_masks and n_detections > 0:
                nms_masks = det_np.masks
                nms_scores = det_np.scores
                nms_masks_bin = (nms_masks > 0.5).astype(np.uint8)
                if nms_masks_bin.ndim == 4:
                    nms_masks_bin = nms_masks_bin.squeeze(1)
                metrics = compute_map_iou(nms_masks_bin, nms_scores, gt_masks)
            else:
                metrics = {"map": 0.0, "best_iou": 0.0,
                           "n_pred": n_detections, "n_gt": len(gt_masks)}

        timing["eval"] = time.time() - t0
        total_time = time.time() - t_tc_start

        # ---- Save result JSON -------------------------------------
        tc_result = {
            "image_id": img_id,
            "obj_id": obj_id,
            "n_proposals": n_proposals,
            "n_after_semantic": n_after_sem,
            "n_detections": n_detections,
            "n_gt": metrics.get("n_gt", len(gt_masks)),
            "map": metrics["map"],
            "best_iou": metrics["best_iou"],
            "ap_per_iou": metrics.get("ap_per_iou", {}),
            "score_type": score_label,
            "timing_total": total_time,
            "timing": timing,
            "config": {
                "ov_device": args.ov_device,
                "ext": args.ext,
                "segmentor_model": args.segmentor_model,
                "points_per_side": args.points_per_side,
                "stability_score_thresh": args.stability_score_thresh,
            },
        }
        with open(result_path, "w") as f:
            json.dump(tc_result, f, indent=2, default=str)

        maps_by_obj[obj_id].append(metrics["map"])
        timings.append(total_time)
        n_completed += 1

        # ---- Progress every 500 iterations ------------------------
        if n_completed % 500 == 0:
            _print_progress(n_completed, n_total, maps_by_obj, timings)

    # -- Final summary ----------------------------------------------
    total_eval_time = time.time() - t_eval_start
    _print_final_summary(
        n_completed, n_skipped, n_total,
        maps_by_obj, timings, init_time, total_eval_time, output_dir,
    )

    # Save aggregate summary JSON
    agg = aggregate_bop_maps(maps_by_obj)
    agg["init_time_s"] = init_time
    agg["total_eval_time_s"] = total_eval_time
    agg["avg_inference_s"] = float(np.mean(timings)) if timings else 0.0
    agg["config"] = {
        "ov_device": args.ov_device,
        "ext": args.ext,
        "segmentor_model": args.segmentor_model,
        "points_per_side": args.points_per_side,
        "stability_score_thresh": args.stability_score_thresh,
        "obj_ids": target_obj_ids,
        "max_images": args.max_images,
        "n_testcases": n_total,
        "n_completed": n_completed,
        "n_skipped": n_skipped,
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"\n  Summary saved to: {summary_path}")


# ===================================================================
#  Progress / summary printing
# ===================================================================

def _print_progress(n_done, n_total, maps_by_obj, timings):
    all_maps = [m for maps in maps_by_obj.values() for m in maps]
    avg_map = np.mean(all_maps) * 100 if all_maps else 0.0
    avg_time = np.mean(timings) if timings else 0.0
    elapsed = sum(timings)
    print(
        f"\n  --- Progress: {n_done}/{n_total} "
        f"({100 * n_done / n_total:.1f}%) | "
        f"Avg mAP: {avg_map:.2f}% | "
        f"Avg time/tc: {avg_time:.2f}s | "
        f"Elapsed: {elapsed:.0f}s ---"
    )


def _print_final_summary(n_completed, n_skipped, n_total,
                          maps_by_obj, timings, init_time,
                          total_eval_time, output_dir):
    print(f"\n{'='*65}")
    print(f"  BOP ISM Evaluation — FINAL RESULTS")
    print(f"{'='*65}")
    print(f"  Completed      : {n_completed}/{n_total} "
          f"(skipped existing: {n_skipped})")
    print(f"  Model init     : {init_time:.1f}s")
    if timings:
        print(f"  Avg time/tc    : {np.mean(timings):.2f}s")
        print(f"  Total eval time: {total_eval_time:.1f}s")

    print(f"\n  Per-Object mAP @ IoU[0.50:0.95]:")
    all_maps = []
    for oid in sorted(maps_by_obj.keys()):
        obj_maps = maps_by_obj[oid]
        obj_map = np.mean(obj_maps) * 100 if obj_maps else 0.0
        all_maps.extend(obj_maps)
        print(f"    Object {oid:>2} ({len(obj_maps):>4} tc):  "
              f"mAP = {obj_map:6.2f}%")

    if all_maps:
        overall_map = np.mean(all_maps) * 100
        print(f"\n  {'='*55}")
        print(f"  OVERALL mAP @ IoU[0.50:0.95]: {overall_map:.2f}%")
        print(f"\n  Paper reference (SAM, LM-O, all scores): 46.0% mAP")
        diff = overall_map - 46.0
        print(f"  Your result vs paper: {diff:+.1f}% "
              f"({'above' if diff >= 0 else 'below'})")
    else:
        print("\n  No testcases evaluated.")

    print(f"\n  Results dir: {output_dir}")


if __name__ == "__main__":
    main()
