#!/usr/bin/env python3
"""
BOP dataset evaluation for OpenVINO ISM pipeline.

Example:
    python3 eval_ism_ov_bop.py \
        --bop_dir /workspace/SAM-6D/Data/BOP/lmo \
        --ov_device GPU --segmentor_model fastsam \
        --ext --max_images 10 --batch_size 4
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import sys
import time
from collections import defaultdict

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from plyfile import PlyData

import openvino as ov

# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from infer_ism_ov import init_model_ov, init_model_ov_ext
from model.utils import Detections
from utils.inout import save_npz
from utils.bbox_utils import force_binary_mask, xyxy_to_xywh
from eval_utils import (
    evaluate_image_multiclass,
    compute_global_ap_voc,
)

_RGB_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
_INV_RGB_TRANSFORM = T.Compose([
    T.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    ),
])


# ===================================================================
#  Helper functions
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


def _compute_template_descriptors(model, template_dir,
                                  proposal_processor, device):
    """Compute DINOv2 cls + appearance descriptors for one object.

    Returns
    -------
    (cls_desc, appe_desc) - (N_tpl, D) and (N_tpl, N_patch, D)
    """
    n_tpl = len(glob.glob(os.path.join(template_dir, "*.npy")))

    if n_tpl == 0:
        return None, None

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

    cls_desc = model.descriptor_model.compute_features(
        tpl_imgs, token_name="x_norm_clstoken"
    ).data
    appe_desc = model.descriptor_model.compute_masked_patch_feature(
        tpl_imgs, tpl_msks_crop[:, 0, :, :]
    ).data

    return cls_desc, appe_desc


def _load_pointcloud(cad_path, n_pts=2048):
    """Load mesh and sample *n_pts* surface points -> (n_pts, 3) tensor."""
    ply = PlyData.read(cad_path)
    verts = np.column_stack(
        [ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"]]
    ).astype(np.float64)
    faces = np.vstack(ply["face"]["vertex_indices"])
    pts = _sample_mesh_surface(verts, faces, n_pts).astype(np.float32) / 1000.0

    return torch.tensor(pts)


def load_gt_for_frame(mask_visib_dir, gt_data, img_id):
    """Load all GT visible masks + object IDs for a BOP frame.

    Returns (gt_masks, gt_obj_ids) - lists of (H,W) uint8 and int.
    """
    gt_entries = gt_data.get(str(img_id), [])
    gt_masks, gt_obj_ids = [], []
    for gt_idx, entry in enumerate(gt_entries):
        mask_path = os.path.join(
            mask_visib_dir, f"{img_id:06d}_{gt_idx:06d}.png"
        )
        if os.path.exists(mask_path):
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                gt_masks.append((m > 0).astype(np.uint8))
                gt_obj_ids.append(entry["obj_id"])

    return gt_masks, gt_obj_ids


def _load_rgb(rgb_path):
    """dataloader inverse-normalize path."""
    image_np = _INV_RGB_TRANSFORM(
        _RGB_TRANSFORM(Image.open(rgb_path).convert("RGB"))
    ).cpu().numpy().transpose(1, 2, 0)

    return np.uint8(image_np.clip(0, 1) * 255)


def _dataset_name_from_bop_dir(bop_dir):
    return os.path.basename(os.path.normpath(bop_dir))


def _load_bop_test_records(bop_dir):
    """Load BOP test records in the deterministic order."""
    split_dir = os.path.join(bop_dir, "test")
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Missing BOP test split: {split_dir}")

    scene_dirs = sorted(
        os.path.join(split_dir, scene)
        for scene in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, scene))
    )

    records = []
    scene_gt_cache = {}
    scene_cam_cache = {}

    for scene_dir in scene_dirs:
        scene_id = os.path.basename(scene_dir)
        gt_path = os.path.join(scene_dir, "scene_gt.json")
        cam_path = os.path.join(scene_dir, "scene_camera.json")
        if not os.path.isfile(gt_path) or not os.path.isfile(cam_path):
            continue

        with open(gt_path) as f:
            scene_gt_cache[scene_id] = json.load(f)
        with open(cam_path) as f:
            scene_cam_cache[scene_id] = json.load(f)

        rgb_dir = os.path.join(scene_dir, "rgb")
        gray_dir = os.path.join(scene_dir, "gray")
        if os.path.isdir(rgb_dir):
            image_paths = sorted(
                glob.glob(os.path.join(rgb_dir, "*.[pj][pn][g]"))
            )
        else:
            image_paths = sorted(glob.glob(os.path.join(gray_dir, "*.tif")))

        for rgb_path in image_paths:
            img_id = int(os.path.splitext(os.path.basename(rgb_path))[0])
            depth_path = os.path.join(scene_dir, "depth", f"{img_id:06d}.png")
            if not os.path.isfile(depth_path):
                depth_path = None
            records.append({
                "scene_id": scene_id,
                "image_id": img_id,
                "rgb_path": rgb_path,
                "depth_path": depth_path,
                "mask_visib_dir": os.path.join(scene_dir, "mask_visib"),
            })

    if not records:
        raise RuntimeError(f"No BOP test images found under {split_dir}")

    records_df = pd.DataFrame.from_records(records)

    records_df = records_df.sample(frac=1, random_state=2021).reset_index(drop=True)
    # Match BaseBOPTest: BaseBOP.load_metaData shuffles once, then BaseBOPTest
    # shuffles the resulting metadata again before DataLoader iteration.
    records_df = records_df.sample(frac=1, random_state=2021).reset_index(drop=True)

    return records_df.to_dict("records"), scene_gt_cache, scene_cam_cache


def _shape_of(value):
    if value is None:
        return "-"
    if torch.is_tensor(value):
        return list(value.shape)
    if isinstance(value, np.ndarray):
        return list(value.shape)
    if hasattr(value, "shape"):
        return list(value.shape)
    return str(value)


def _print_perf(enabled, index, step, elapsed_s,
                inputs=(), outputs=()):
    if not enabled:
        return
    print(f"    [perf {index}] {step}: {elapsed_s * 1000.0:.1f} ms")
    for name, value in inputs:
        print(f"      input  {name}: {_shape_of(value)}")
    for name, value in outputs:
        print(f"      output {name}: {_shape_of(value)}")


# ===================================================================
#  CLI
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="BOP evaluation - OpenVINO ISM (multi-class, VOC mAP)"
    )
    p.add_argument("--ov_model_dir",
                   default=os.path.join(BASE_DIR, "checkpoints/ov_models"))
    p.add_argument("--ov_device", default="GPU")
    p.add_argument(
        "--ext", action=argparse.BooleanOptionalAction, default=True,
        help="Use extended OV IRs. Default: True."
    )
    p.add_argument("--segmentor_model", default="fastsam",
                   choices=["sam", "fastsam"])
    p.add_argument("--points_per_side", type=int, default=32)
    p.add_argument("--stability_score_thresh", type=float, default=0.85)
    p.add_argument("--points_per_batch", type=int, default=64)
    p.add_argument("--chunk_size", type=int, default=16,
                   help="DINOv2 proposals per call (default: 16)")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Batch size (default: 4)")
    p.add_argument("--bop_dir", default=None,
                   help="BOP dataset root")
    p.add_argument("--profile_example", action="store_true",
                   help="Run one Data/Example image with t3-t9 timing and shapes, then exit.")
    p.add_argument("--example_dir",
                   default=os.path.abspath(os.path.join(BASE_DIR, "..", "Data", "Example")),
                   help="Example directory used with --profile_example")
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--obj_ids", type=int, nargs="+", default=None)
    p.add_argument("--max_objects", type=int, default=None)
    p.add_argument("--skip_existing", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Skip per-image files already present in output_dir. "
                        "Default is false because skipped images cannot "
                        "contribute to global VOC mAP accumulation.")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--templates_root", default=None)
    p.add_argument("--ref_cache_dir", default=None,
                   help="Dir with cached descriptors.pth / pointcloud.pth "
                        "(e.g. templates_pyrender/lmo). Skips recompute.")
    p.add_argument("--precision", default=os.environ.get("OV_PRECISION", "fp16"),
                   choices=["fp32", "fp16"],
                   help="Inference precision: fp32 or fp16. "
                        "Default: OV_PRECISION env var or fp16.")

    return p.parse_args()


def recompile_gpu_with_cache(args, model):
    if not args.ov_device.upper().startswith("GPU"):
        return

    _core = ov.Core()

    _ism_ext_so = os.path.join(BASE_DIR, "ov_plugins", "build", "ism_extension.so")
    if os.path.isfile(_ism_ext_so):
        _core.add_extension(_ism_ext_so)

    _cache_dir = os.path.join(args.ov_model_dir, ".ov_gpu_cache")

    os.makedirs(_cache_dir, exist_ok=True)

    _ism_gpu_xml = os.path.join(BASE_DIR, "ov_plugins", "ism_gpu",
                                "ism_custom_gpu_kernels.xml")

    _prec_hint = "f32" if args.precision == "fp32" else "f16"
    _gpu_config = {
        "PERFORMANCE_HINT": "LATENCY",
        "NUM_STREAMS": "1",
        "INFERENCE_PRECISION_HINT": _prec_hint,
        "CACHE_DIR": _cache_dir,
    }
    if args.precision == "fp32":
        _gpu_config["EXECUTION_MODE_HINT"] = "ACCURACY"
    if os.path.isfile(_ism_gpu_xml):
        _gpu_config["CONFIG_FILE"] = _ism_gpu_xml

    if args.segmentor_model == "fastsam":
        _xml = os.path.join(args.ov_model_dir, "fastsam_x_dynamic.xml")
        _fastsam_config = dict(_gpu_config)
        _fastsam_config["INFERENCE_PRECISION_HINT"] = "f32"
        model.segmentor_model.compiled = _core.compile_model(
            _core.read_model(_xml), args.ov_device, _fastsam_config)
        print(f"  FastSAM re-compiled with CACHE_DIR (precision={args.precision})", flush=True)

    _dinov2_name = "dinov2_vitl14_ext.xml" if args.ext else "dinov2_vitl14.xml"
    _dinov2_xml = os.path.join(args.ov_model_dir, _dinov2_name)
    model.descriptor_model.dinov2_compiled = _core.compile_model(
        _core.read_model(_dinov2_xml), args.ov_device, _gpu_config)
    if hasattr(model.descriptor_model, "_dinov2_request"):
        model.descriptor_model._dinov2_request = (
            model.descriptor_model.dinov2_compiled.create_infer_request()
        )
    print(f"  DINOv2 re-compiled with CACHE_DIR (precision={args.precision})", flush=True)


def _batch_example_input_data(example_dir, device):
    depth_path = os.path.join(example_dir, "depth.png")
    cam_path = os.path.join(example_dir, "camera.json")
    with open(cam_path) as f:
        cam_info = json.load(f)
    depth = np.array(imageio.imread(depth_path)).astype(np.int32)
    cam_K = np.array(cam_info["cam_K"]).reshape((3, 3))
    depth_scale = np.array(cam_info["depth_scale"])

    return {
        "depth": torch.from_numpy(depth).unsqueeze(0).to(device),
        "cam_intrinsic": torch.from_numpy(cam_K).unsqueeze(0).to(device),
        "depth_scale": torch.from_numpy(depth_scale).unsqueeze(0).to(device),
    }


def _run_example_profile(args):
    example_dir = args.example_dir

    rgb_path = os.path.join(example_dir, "rgb.png")
    cad_path = os.path.join(example_dir, "obj_000005.ply")
    template_dir = os.path.join(example_dir, "outputs", "templates")

    for path, label in ((rgb_path, "RGB image"), (cad_path, "CAD model"),
                        (template_dir, "templates dir")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing Example {label}: {path}")

    print(f"\n{'='*65}")
    print("  Example ISM Profile - OpenVINO (t3-t9)")
    print(f"{'='*65}")
    print(f"  Example dir   : {example_dir}")
    print(f"  Device        : {args.ov_device} (ext={args.ext})")
    print(f"  Segmentor     : {args.segmentor_model}")
    print(f"  Chunk size    : {args.chunk_size}")
    print(f"  Templates dir : {template_dir}")
    print(flush=True)

    init_fn = init_model_ov_ext if args.ext else init_model_ov
    model, _cfg, device, selected_poses, proposal_processor = init_fn(
        ov_model_dir=args.ov_model_dir,
        ov_device=args.ov_device,
        segmentor_model_name=args.segmentor_model,
        stability_score_thresh=args.stability_score_thresh,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        chunk_size=args.chunk_size,
    )

    if hasattr(model.segmentor_model, "points_per_batch"):
        model.segmentor_model.points_per_batch = args.points_per_batch
    if hasattr(model.descriptor_model, "chunk_size"):
        model.descriptor_model.chunk_size = args.chunk_size

    recompile_gpu_with_cache(args, model)

    print("Pre-loading one-object reference data ...", flush=True)
    cls_desc, appe_desc = _compute_template_descriptors(
        model, template_dir, proposal_processor, device
    )

    pointcloud = _load_pointcloud(cad_path, n_pts=2048)
    model.ref_data = {
        "descriptors": cls_desc.unsqueeze(0),
        "appe_descriptors": appe_desc.unsqueeze(0),
        "pointcloud": pointcloud.unsqueeze(0).to(device),
        "poses": selected_poses,
    }

    print(f"    descriptors      : {model.ref_data['descriptors'].shape}")
    print(f"    appe_descriptors : {model.ref_data['appe_descriptors'].shape}")
    print(f"    pointcloud       : {model.ref_data['pointcloud'].shape}")

    rgb = _load_rgb(rgb_path)
    print("  GPU warmup (outside t3-t9 timing)...", flush=True)
    _ = model.segmentor_model.generate_masks(rgb)
    print("  GPU warmup done\n", flush=True)

    timings = []

    t0 = time.time()
    proposals = model.segmentor_model.generate_masks(rgb)
    detections = Detections(proposals)
    detections.remove_very_small_detections(
        config=model.post_processing_config.mask_post_processing
    )
    seg_time = time.time() - t0
    timings.append(seg_time)

    _print_perf(
        True, "t3", "generate_masks", seg_time,
        inputs=(("np.array(rgb).shape", rgb),),
        outputs=(("detections['masks'].size()", detections.masks),
                 ("detections['boxes'].size()", detections.boxes)),
    )

    t0 = time.time()
    query_cls, query_appe = model.descriptor_model(rgb, detections)
    desc_time = time.time() - t0
    timings.append(desc_time)

    _print_perf(
        True, "t4", "descriptor_model.forward", desc_time,
        inputs=(("np.array(rgb).shape", rgb),
                ("detections.masks.size()", detections.masks),
                ("detections.boxes.size()", detections.boxes)),
        outputs=(("query_descriptors.size()", query_cls),
                 ("query_appe_descriptors.size()", query_appe),
                 ("detections.masks.size()", detections.masks)),
    )

    t0 = time.time()
    idx_selected, pred_idx_objects, semantic_score, best_template = \
        model.compute_semantic_score(query_cls)
    semantic_time = time.time() - t0
    timings.append(semantic_time)

    _print_perf(
        True, "t5", "compute_semantic_score", semantic_time,
        inputs=(("query_descriptors.size()", query_cls),),
        outputs=(("idx_selected_proposals.size()", idx_selected),
                 ("pred_idx_objects.size()", pred_idx_objects),
                 ("semantic_score.size()", semantic_score),
                 ("best_template.size()", best_template)),
    )

    selected_count = len(idx_selected)
    if selected_count == 0:
        print("\n  no proposals survived semantic filter")
        print("  mAP@0.50: 0.0000")
        print("  mAP@0.50:0.95: 0.0000")
        print(f"{'='*65}")
        return

    detections.filter(idx_selected)
    query_appe_sel = query_appe[idx_selected, :]

    t0 = time.time()
    appe_scores, ref_aux_descriptor = model.compute_appearance_score(
        best_template, pred_idx_objects, query_appe_sel
    )
    appearance_time = time.time() - t0
    timings.append(appearance_time)

    _print_perf(
        True, "t6", "compute_appearance_score", appearance_time,
        inputs=(("best_template.size()", best_template),
                ("pred_idx_objects.size()", pred_idx_objects),
                ("query_appe_descriptors.size()", query_appe_sel)),
        outputs=(("appe_scores.size()", appe_scores),
                 ("ref_aux_descriptor.size()", ref_aux_descriptor)),
    )

    batch_geo = _batch_example_input_data(example_dir, device)
    t0 = time.time()
    image_uv = model.project_template_to_image(
        best_template, pred_idx_objects, batch_geo, detections.masks
    )
    project_time = time.time() - t0
    timings.append(project_time)

    _print_perf(
        True, "t7", "project_template_to_image", project_time,
        inputs=(("best_template.size()", best_template),
                ("pred_idx_objects.size()", pred_idx_objects),
                ("detections.masks.size()", detections.masks),
                ("batch['depth'].size()", batch_geo["depth"]),
                ("batch['cam_intrinsic'].size()", batch_geo["cam_intrinsic"]),
                ("batch['depth_scale'].size()", batch_geo["depth_scale"])),
        outputs=(("image_uv.size()", image_uv),),
    )

    t0 = time.time()
    geometric_score, visible_ratio = model.compute_geometric_score(
        image_uv, detections, query_appe_sel, ref_aux_descriptor,
        visible_thred=model.visible_thred,
    )
    geometric_time = time.time() - t0
    timings.append(geometric_time)

    _print_perf(
        True, "t8", "compute_geometric_score", geometric_time,
        inputs=(("image_uv.size()", image_uv),
                ("detections.masks.size()", detections.masks),
                ("detections.boxes.size()", detections.boxes),
                ("query_appe_descriptors.size()", query_appe_sel),
                ("ref_aux_descriptor.size()", ref_aux_descriptor)),
        outputs=(("geometric_score.size()", geometric_score),
                 ("visible_ratio.size()", visible_ratio)),
    )

    t0 = time.time()
    final_score = (
        semantic_score + appe_scores + geometric_score * visible_ratio
    ) / (1 + 1 + visible_ratio)
    final_time = time.time() - t0
    timings.append(final_time)

    _print_perf(
        True, "t9", "final score", final_time,
        inputs=(("semantic_score.size()", semantic_score),
                ("appe_scores.size()", appe_scores),
                ("geometric_score.size()", geometric_score),
                ("visible_ratio.size()", visible_ratio)),
        outputs=(("final_score.size()", final_score),),
    )

    scores_cpu = final_score.detach().cpu().numpy()
    semantic_cpu = semantic_score.detach().cpu().numpy()
    appe_cpu = appe_scores.detach().cpu().numpy()
    geometric_cpu = geometric_score.detach().cpu().numpy()
    template_cpu = best_template.detach().cpu().numpy()
    score_order = np.argsort(scores_cpu)[::-1]

    detections.add_attribute("scores", final_score)
    detections.add_attribute("object_ids", pred_idx_objects)
    detections.apply_nms_per_object_id(
        nms_thresh=model.post_processing_config.nms_thresh
    )
    n_dets = len(detections)
    detections.to_numpy()
    pred_masks = detections.masks
    pred_scores = detections.scores
    pred_bop_ids = np.full(n_dets, 5, dtype=np.int64)
    pred_masks_bin = (
        np.array([force_binary_mask(m) for m in pred_masks])
        if n_dets > 0
        else np.zeros((0, 1, 1), dtype=np.uint8)
    )

    mask_paths = sorted(glob.glob(os.path.join(example_dir, "mask_visib", "*.png")))
    gt_masks = [
        (np.array(imageio.imread(mask_path)) > 0).astype(np.uint8)
        for mask_path in mask_paths
    ]
    gt_masks_arr = (
        np.stack(gt_masks)
        if gt_masks
        else np.zeros((0, 1, 1), dtype=np.uint8)
    )
    gt_obj_ids_arr = np.full(len(gt_masks), 5, dtype=np.int64)

    iou_thresholds = [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]
    map_values = []
    map_50 = 0.0
    for iou_thresh in iou_thresholds:
        if n_dets > 0 and len(gt_masks_arr) > 0:
            img_eval = evaluate_image_multiclass(
                pred_masks_bin, pred_scores, pred_bop_ids,
                gt_masks_arr, gt_obj_ids_arr, iou_thresh=iou_thresh,
            )
        else:
            img_eval = {
                int(obj_id): {"scores": [], "matched": [], "n_gt": 1}
                for obj_id in gt_obj_ids_arr
            }
        aps = []
        for info in img_eval.values():
            aps.append(compute_global_ap_voc(
                np.array(info["scores"]) if info["scores"] else np.array([]),
                info["matched"] if info["matched"] else [],
                info["n_gt"],
            ))
        mean_ap = float(np.mean(aps)) if aps else 0.0
        if iou_thresh == 0.5:
            map_50 = mean_ap
        map_values.append(mean_ap)
    map_5095 = float(np.mean(map_values)) if map_values else 0.0

    print(f"\n  t3-t9 total: {sum(timings) * 1000.0:.1f} ms")
    print(f"  selected proposals: {selected_count} / {len(query_cls)}")
    print(f"  detections after NMS: {n_dets}")
    print(f"  GT masks: {len(gt_masks)} (obj_id=5)")
    print(f"  mAP@0.50: {map_50:.4f}")
    print(f"  mAP@0.50:0.95: {map_5095:.4f}")
    if len(score_order) > 0:
        best_idx = int(score_order[0])
        best_score = float(scores_cpu[best_idx])
        confidence = "CONFIDENT" if best_score >= 0.5 else "LOW"
        print(f"  Best match score: {best_score:.4f} ({confidence})")
        print("  Top-5 proposal scores:")
        print("    Rank    Final   Semantic   Appearance   Geometric   Template")
        for rank, idx in enumerate(score_order[:5], 1):
            print(
                f"    {rank:>4}  {scores_cpu[idx]:>7.4f}"
                f"  {semantic_cpu[idx]:>9.4f}"
                f"  {appe_cpu[idx]:>11.4f}"
                f"  {geometric_cpu[idx]:>10.4f}"
                f"  template_{int(template_cpu[idx]):>3d}"
            )
    print("  scoring components: semantic + appearance + geometric")
    print(f"{'='*65}")


# ===================================================================
#  Main
# ===================================================================

def main():
    args = parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    if args.profile_example:
        _run_example_profile(args)
        return

    if not args.bop_dir:
        raise ValueError("--bop_dir is required unless --profile_example is used")

    # -- Resolve paths ------------------------------------------------
    dataset_name = _dataset_name_from_bop_dir(args.bop_dir)
    image_records, scene_gt_cache, scene_cam_cache = _load_bop_test_records(
        args.bop_dir
    )
    if args.max_images is not None:
        image_records = image_records[:args.max_images]
    n_images = len(image_records)

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

    # -- Target objects -----------------------------------------------
    all_obj_ids = sorted({
        e["obj_id"]
        for gt_data in scene_gt_cache.values()
        for entries in gt_data.values()
        for e in entries
    })
    if args.obj_ids:
        target_obj_ids = [o for o in args.obj_ids if o in all_obj_ids]
    else:
        target_obj_ids = list(all_obj_ids)
    if args.max_objects:
        target_obj_ids = target_obj_ids[:args.max_objects]

    profile_enabled = args.max_images is not None and args.max_images <= 5

    # -- Header -------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  BOP ISM Evaluation - OpenVINO")
    print(f"{'='*65}")
    print(f"  BOP dir       : {args.bop_dir}")
    print(f"  Device        : {args.ov_device} ({args.ext})")
    print(f"  Segmentor     : {args.segmentor_model}")
    print(f"  Objects       : {target_obj_ids}")
    print(f"  Images        : {n_images}")
    print(f"  Batch size    : {args.batch_size}")
    print(f"  Chunk size    : {args.chunk_size}")
    print(f"  Output        : {output_dir}")
    print(f"  Skip existing : {args.skip_existing}")
    print(f"  Templates root: {templates_root}")
    print(f"  Ref cache dir : {args.ref_cache_dir}")
    if profile_enabled:
        print(f"  Perf profile  : enabled (max_images <= 5)")
    print(flush=True)

    if n_images == 0:
        print("No images to evaluate.")
        return

    # =================================================================
    #  1. Init OV ISM pipeline
    # =================================================================
    print("Initializing OV ISM pipeline ...", flush=True)
    t_init_start = time.time() if profile_enabled else None
    init_fn = init_model_ov_ext if args.ext else init_model_ov
    model, cfg, device, selected_poses, proposal_processor = init_fn(
        ov_model_dir=args.ov_model_dir,
        ov_device=args.ov_device,
        segmentor_model_name=args.segmentor_model,
        stability_score_thresh=args.stability_score_thresh,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        chunk_size=args.chunk_size,
    )
    if hasattr(model.segmentor_model, "points_per_batch"):
        model.segmentor_model.points_per_batch = args.points_per_batch
    if hasattr(model.descriptor_model, "chunk_size"):
        model.descriptor_model.chunk_size = args.chunk_size

    # -- Re-compile with CACHE_DIR for GPU ----------------------------
    if args.ov_device.upper().startswith("GPU"):
        _core = ov.Core()

        _cache_dir = os.path.join(args.ov_model_dir, ".ov_gpu_cache")
        os.makedirs(_cache_dir, exist_ok=True)

        _prec_hint = "f32" if args.precision == "fp32" else "f16"
        _gpu_config = {
            "PERFORMANCE_HINT": "LATENCY",
            "NUM_STREAMS": "1",
            "INFERENCE_PRECISION_HINT": _prec_hint,
            "CACHE_DIR": _cache_dir,
        }
        if args.precision == "fp32":
            _gpu_config["EXECUTION_MODE_HINT"] = "ACCURACY"

        if args.segmentor_model == "fastsam":
            _xml = os.path.join(args.ov_model_dir, "fastsam_x_dynamic.xml")
            _fastsam_config = dict(_gpu_config)
            _fastsam_config["INFERENCE_PRECISION_HINT"] = "f32"
            model.segmentor_model.compiled = _core.compile_model(
                _core.read_model(_xml), args.ov_device, _fastsam_config)

            print(f"  FastSAM re-compiled with CACHE_DIR (precision={args.precision})", flush=True)

        _dinov2_name = "dinov2_vitl14_ext.xml" if args.ext else "dinov2_vitl14.xml"
        _dinov2_xml = os.path.join(args.ov_model_dir, _dinov2_name)
        model.descriptor_model.dinov2_compiled = _core.compile_model(
            _core.read_model(_dinov2_xml), args.ov_device, _gpu_config)
        if hasattr(model.descriptor_model, "_dinov2_request"):
            model.descriptor_model._dinov2_request = (
                model.descriptor_model.dinov2_compiled.create_infer_request()
            )
        print(f"  DINOv2 re-compiled with CACHE_DIR (precision={args.precision})", flush=True)

    # -- GPU warmup ---------------------------------------------------
    warmup_rgb_path = image_records[0]["rgb_path"]
    if os.path.isfile(warmup_rgb_path):
        print("  GPU warmup (compiling kernels)...", flush=True)
        t_warmup = time.time() if profile_enabled else None
        warmup_rgb = _load_rgb(warmup_rgb_path)
        _ = model.segmentor_model.generate_masks(warmup_rgb)
        if profile_enabled:
            print(f"  GPU warmup done: {time.time() - t_warmup:.1f}s", flush=True)
        else:
            print(f"  GPU warmup done", flush=True)

    init_time = time.time() - t_init_start if profile_enabled else None
    if profile_enabled:
        print(f"  Model init: {init_time:.1f}s", flush=True)

    # =================================================================
    #  2. Pre-load reference data (multi-class)
    # =================================================================
    print("\nPre-loading reference data for all objects ...", flush=True)
    t_ref_start = time.time() if profile_enabled else None
    bop_obj_ids = np.array(target_obj_ids)   # ref index -> BOP ID mapping

    # -- Check for cached descriptors --
    ref_cache_dir = args.ref_cache_dir
    desc_path = os.path.join(ref_cache_dir, "descriptors.pth") if ref_cache_dir else ""
    appe_path = os.path.join(ref_cache_dir, "descriptors_appe.pth") if ref_cache_dir else ""
    pc_path = os.path.join(ref_cache_dir, "pointcloud.pth") if ref_cache_dir else ""

    if ref_cache_dir and os.path.exists(desc_path) and os.path.exists(appe_path) and os.path.exists(pc_path):
        print(f"  Loading cached ref data from: {ref_cache_dir}")
        all_descriptors = torch.load(desc_path, map_location="cpu")
        all_appe_descriptors = torch.load(appe_path, map_location="cpu")
        all_pointclouds = torch.load(pc_path, map_location="cpu")
        print(f"    descriptors      : {all_descriptors.shape}")
        print(f"    appe_descriptors : {all_appe_descriptors.shape}")
        print(f"    pointcloud       : {all_pointclouds.shape}")
    else:
        # -- Compute from template images --
        all_cls_descs = []
        all_appe_descs = []
        all_pc_list = []

        for obj_id in target_obj_ids:
            template_dir = os.path.join(
                templates_root, f"obj_{obj_id:06d}", "templates"
            )
            cad_path = os.path.join(models_dir, f"obj_{obj_id:06d}.ply")

            if not os.path.isdir(template_dir):
                print(f"  [ERROR] No templates at {template_dir}")
                return
            if not os.path.isfile(cad_path):
                print(f"  [ERROR] No CAD at {cad_path}")
                return

            print(f"  obj {obj_id:>2d}: templates ...", end="", flush=True)
            cls_desc, appe_desc = _compute_template_descriptors(
                model, template_dir, proposal_processor, device
            )
            pc = _load_pointcloud(cad_path, n_pts=2048)
            all_cls_descs.append(cls_desc)
            all_appe_descs.append(appe_desc)
            all_pc_list.append(pc)
            print(f" done ({cls_desc.shape[0]} templates)", flush=True)

        all_descriptors = torch.stack(all_cls_descs)
        all_appe_descriptors = torch.stack(all_appe_descs)
        all_pointclouds = torch.stack(all_pc_list)

    model.ref_data = {
        "descriptors": all_descriptors,
        "appe_descriptors": all_appe_descriptors,
        "pointcloud": all_pointclouds.to(device),
        "poses": selected_poses,
    }
    ref_time = time.time() - t_ref_start if profile_enabled else None
    if profile_enabled:
        print(f"  Reference data ready ({ref_time:.1f}s)")
    else:
        print(f"  Reference data ready")
    print(f"    descriptors      : {model.ref_data['descriptors'].shape}")
    print(f"    appe_descriptors : {model.ref_data['appe_descriptors'].shape}")
    print(f"    pointcloud       : {model.ref_data['pointcloud'].shape}")
    total_setup = time.time() - t_init_start if profile_enabled else None
    if profile_enabled:
        print(f"  Total init + ref: {total_setup:.1f}s\n", flush=True)
    else:
        print(flush=True)

    # =================================================================
    #  3. mAP accumulators (VOC-style, global across images)
    # =================================================================
    IOU_THRESHOLDS = [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]
    global_per_class = {
        t: defaultdict(lambda: {"scores": [], "matched": [], "n_gt": 0})
        for t in IOU_THRESHOLDS
    }
    image_results = []
    timings = []

    # =================================================================
    #  4. Main inference loop
    # =================================================================
    print(f"Starting inference on {n_images} images\n", flush=True)
    t_eval_start = time.time()
    evaluated_images = 0

    for img_idx, record in enumerate(image_records):
        scene_id = record["scene_id"]
        img_id = int(record["image_id"])
        t_img_start = time.time() if profile_enabled else None
        print(
            f"  [{img_idx + 1}/{n_images}] scene={scene_id} img={img_id}",
            flush=True,
        )

        # -- skip_existing (per image) --------------------------------
        result_path = os.path.join(
            output_dir, f"scene{scene_id}_img{img_id:06d}.json"
        )
        if args.skip_existing and os.path.exists(result_path):
            print(f"    (skipped - existing)", flush=True)
            continue
        evaluated_images += 1

        # ---- Step 1: Segmentation -----------------------------------
        rgb_path = record["rgb_path"]
        rgb = _load_rgb(rgb_path)
        t0 = time.time() if profile_enabled else None
        proposals = model.segmentor_model.generate_masks(rgb)
        detections = Detections(proposals)
        detections.remove_very_small_detections(
            config=model.post_processing_config.mask_post_processing
        )
        seg_time = time.time() - t0 if profile_enabled else None
        if profile_enabled:
            print(f"    segmentation ... {seg_time:.1f}s ({len(detections)} proposals)", flush=True)
            _print_perf(
                profile_enabled, "t3", "generate_masks", seg_time,
                inputs=(("np.array(rgb).shape", rgb),),
                outputs=(("detections['masks'].size()", detections.masks),
                         ("detections['boxes'].size()", detections.boxes)),
            )
        else:
            print(f"    segmentation ({len(detections)} proposals)", flush=True)

        # ---- Step 2: DINOv2 query descriptors -----------------------
        t0 = time.time() if profile_enabled else None
        query_cls, query_appe = model.descriptor_model(rgb, detections)
        desc_time = time.time() - t0 if profile_enabled else None
        if profile_enabled:
            print(f"    descriptors ... {desc_time:.1f}s", flush=True)
            _print_perf(
                profile_enabled, "t4", "descriptor_model.forward", desc_time,
                inputs=(("np.array(rgb).shape", rgb),
                        ("detections.masks.size()", detections.masks),
                        ("detections.boxes.size()", detections.boxes)),
                outputs=(("query_descriptors.size()", query_cls),
                         ("query_appe_descriptors.size()", query_appe),
                         ("detections.masks.size()", detections.masks)),
            )
        else:
            print(f"    descriptors", flush=True)

        # ---- Step 3: Semantic matching (multi-class) ----------------
        t_match_start = time.time() if profile_enabled else None
        t0 = time.time() if profile_enabled else None
        (idx_selected, pred_idx_objects, semantic_score, best_template) = \
            model.compute_semantic_score(query_cls)
        semantic_time = time.time() - t0 if profile_enabled else None
        _print_perf(
            profile_enabled, "t5", "compute_semantic_score", semantic_time,
            inputs=(("query_descriptors.size()", query_cls),),
            outputs=(("idx_selected_proposals.size()", idx_selected),
                     ("pred_idx_objects.size()", pred_idx_objects),
                     ("semantic_score.size()", semantic_score),
                     ("best_template.size()", best_template)),
        )

        # Load GT early (need even if no detections, for n_gt counts)
        gt_data = scene_gt_cache[scene_id]
        gt_masks, gt_obj_ids = load_gt_for_frame(
            record["mask_visib_dir"], gt_data, img_id
        )
        if gt_masks:
            gt_masks_arr = np.stack(gt_masks)
            gt_obj_ids_arr = np.array(gt_obj_ids)
        else:
            gt_masks_arr = np.zeros((0, 1, 1), dtype=np.uint8)
            gt_obj_ids_arr = np.array([], dtype=int)

        if len(idx_selected) == 0:
            print(f"    no proposals survived semantic filter", flush=True)
            # Accumulate GT counts (no predictions)
            for t in IOU_THRESHOLDS:
                for oid in gt_obj_ids:
                    global_per_class[t][oid]["n_gt"] += 1
            img_time = time.time() - t_img_start if profile_enabled else None
            if profile_enabled:
                timings.append(img_time)
            image_results.append({
                "scene_id": scene_id,
                "image_id": img_id, "num_detections": 0,
                "num_gt": len(gt_masks),
                "mAP_0.5": 0.0, "mAP_0.5_0.95": 0.0,
            })
            if profile_enabled:
                image_results[-1]["time_s"] = round(img_time, 2)
            with open(result_path, "w") as f:
                json.dump(image_results[-1], f, indent=2, default=str)
            continue

        detections.filter(idx_selected)
        query_appe_sel = query_appe[idx_selected, :]

        # ---- Step 4: Appearance score -------------------------------
        t0 = time.time() if profile_enabled else None
        appe_scores, ref_aux_descriptor = model.compute_appearance_score(
            best_template, pred_idx_objects, query_appe_sel
        )
        appearance_time = time.time() - t0 if profile_enabled else None
        _print_perf(
            profile_enabled, "t6", "compute_appearance_score", appearance_time,
            inputs=(("best_template.size()", best_template),
                    ("pred_idx_objects.size()", pred_idx_objects),
                    ("query_appe_descriptors.size()", query_appe_sel)),
            outputs=(("appe_scores.size()", appe_scores),
                     ("ref_aux_descriptor.size()", ref_aux_descriptor)),
        )

        # ---- Step 5: Geometric score --------------------------------
        depth_path = record["depth_path"]
        cam_data = scene_cam_cache[scene_id]
        cam_info = cam_data.get(str(img_id))
        has_depth = (
            depth_path is not None
            and os.path.isfile(depth_path)
            and cam_info is not None
        )

        if has_depth:
            depth = np.array(imageio.imread(depth_path)).astype(np.int32)
            cam_K = np.array(cam_info["cam_K"]).reshape((3, 3))
            depth_scale = np.array(cam_info["depth_scale"])
            batch_geo = {
                "depth": torch.from_numpy(depth).unsqueeze(0).to(device),
                "cam_intrinsic": torch.from_numpy(cam_K).unsqueeze(0).to(device),
                "depth_scale": torch.from_numpy(depth_scale).unsqueeze(0).to(device),
            }
            t0 = time.time() if profile_enabled else None
            image_uv = model.project_template_to_image(
                best_template, pred_idx_objects, batch_geo, detections.masks
            )
            project_time = time.time() - t0 if profile_enabled else None
            _print_perf(
                profile_enabled, "t7", "project_template_to_image", project_time,
                inputs=(("best_template.size()", best_template),
                        ("pred_idx_objects.size()", pred_idx_objects),
                        ("detections.masks.size()", detections.masks),
                        ("batch['depth'].size()", batch_geo["depth"]),
                        ("batch['cam_intrinsic'].size()", batch_geo["cam_intrinsic"]),
                        ("batch['depth_scale'].size()", batch_geo["depth_scale"])),
                outputs=(("image_uv.size()", image_uv),),
            )

            t0 = time.time() if profile_enabled else None
            geometric_score, visible_ratio = model.compute_geometric_score(
                image_uv, detections, query_appe_sel, ref_aux_descriptor,
                visible_thred=model.visible_thred,
            )
            geometric_time = time.time() - t0 if profile_enabled else None
            _print_perf(
                profile_enabled, "t8", "compute_geometric_score", geometric_time,
                inputs=(("image_uv.size()", image_uv),
                        ("detections.masks.size()", detections.masks),
                        ("detections.boxes.size()", detections.boxes),
                        ("query_appe_descriptors.size()", query_appe_sel),
                        ("ref_aux_descriptor.size()", ref_aux_descriptor)),
                outputs=(("geometric_score.size()", geometric_score),
                         ("visible_ratio.size()", visible_ratio)),
            )

            t0 = time.time() if profile_enabled else None
            final_score = (
                semantic_score + appe_scores + geometric_score * visible_ratio
            ) / (1 + 1 + visible_ratio)
            final_time = time.time() - t0 if profile_enabled else None
            _print_perf(
                profile_enabled, "t9", "final score", final_time,
                inputs=(("semantic_score.size()", semantic_score),
                        ("appe_scores.size()", appe_scores),
                        ("geometric_score.size()", geometric_score),
                        ("visible_ratio.size()", visible_ratio)),
                outputs=(("final_score.size()", final_score),),
            )
        else:
            t0 = time.time() if profile_enabled else None
            final_score = (semantic_score + appe_scores) / 2.0
            final_time = time.time() - t0 if profile_enabled else None
            _print_perf(
                profile_enabled, "t9", "final score", final_time,
                inputs=(("semantic_score.size()", semantic_score),
                        ("appe_scores.size()", appe_scores)),
                outputs=(("final_score.size()", final_score),),
            )

        match_time = time.time() - t_match_start if profile_enabled else None

        # ---- Step 6: NMS per object ---------------------------------
        detections.add_attribute("scores", final_score)
        detections.add_attribute("object_ids", pred_idx_objects)
        detections.apply_nms_per_object_id(
            nms_thresh=model.post_processing_config.nms_thresh
        )
        n_dets = len(detections)
        if profile_enabled:
            print(f"    matching: {match_time:.1f}s -> {n_dets} dets", flush=True)
        else:
            print(f"    matching -> {n_dets} dets", flush=True)

        # ---- Convert to numpy ---------------------------------------
        detections.to_numpy()
        pred_masks = detections.masks
        pred_scores = detections.scores
        pred_obj_ids_raw = detections.object_ids
        # Map 0-indexed object indices -> BOP object IDs
        pred_bop_ids = bop_obj_ids[pred_obj_ids_raw]

        # Binarize masks
        pred_masks_bin = (
            np.array([force_binary_mask(m) for m in pred_masks])
            if n_dets > 0
            else np.zeros((0, 1, 1), dtype=np.uint8)
        )

        # ---- Save detections ----------------------------------------
        det_save_path = os.path.join(
            output_dir, f"scene{scene_id}_img{img_id:06d}_detection_ism"
        )
        save_npz(det_save_path, {
            "scene_id": int(scene_id),
            "image_id": img_id,
            "category_id": pred_bop_ids,
            "score": pred_scores,
            "bbox": xyxy_to_xywh(detections.boxes),
            "time": 0,
            "segmentation": pred_masks,
        })

        # ---- Step 7: Per-image mAP (VOC-style, per-threshold) -------
        img_aps_per_thresh = {}
        for t in IOU_THRESHOLDS:
            if n_dets > 0 and len(gt_obj_ids_arr) > 0:
                img_eval = evaluate_image_multiclass(
                    pred_masks_bin, pred_scores, pred_bop_ids,
                    gt_masks_arr, gt_obj_ids_arr, iou_thresh=t,
                )
            else:
                img_eval = {}
                for oid in gt_obj_ids_arr:
                    img_eval[oid] = {
                        "scores": [], "matched": [], "n_gt": 1
                    }

            aps_at_t = {}
            for obj_id, info in img_eval.items():
                ap = compute_global_ap_voc(
                    np.array(info["scores"]) if info["scores"]
                    else np.array([]),
                    info["matched"] if info["matched"] else [],
                    info["n_gt"],
                )
                aps_at_t[obj_id] = ap
                # Accumulate into global counters
                global_per_class[t][obj_id]["scores"].extend(info["scores"])
                global_per_class[t][obj_id]["matched"].extend(
                    info["matched"]
                )
                global_per_class[t][obj_id]["n_gt"] += info["n_gt"]
            img_aps_per_thresh[t] = aps_at_t

        # Per-image summary numbers
        img_aps_50 = img_aps_per_thresh.get(0.5, {})
        img_map_50 = (
            float(np.mean(list(img_aps_50.values())))
            if img_aps_50 else 0.0
        )
        all_thresh_maps = []
        for t in IOU_THRESHOLDS:
            aps = img_aps_per_thresh.get(t, {})
            all_thresh_maps.append(
                float(np.mean(list(aps.values()))) if aps else 0.0
            )
        img_map_5095 = (
            float(np.mean(all_thresh_maps)) if all_thresh_maps else 0.0
        )

        img_time = time.time() - t_img_start if profile_enabled else None
        if profile_enabled:
            timings.append(img_time)

        if profile_enabled:
            print(
                f"    GT: {len(gt_masks)} | dets: {n_dets} | "
                f"mAP@.5={img_map_50:.4f} | "
                f"mAP@[.5:.95]={img_map_5095:.4f} | "
                f"{img_time:.1f}s",
                flush=True,
            )
        else:
            print(
                f"    GT: {len(gt_masks)} | dets: {n_dets} | "
                f"mAP@.5={img_map_50:.4f} | "
                f"mAP@[.5:.95]={img_map_5095:.4f}",
                flush=True,
            )

        image_results.append({
            "scene_id": scene_id,
            "image_id": img_id,
            "num_detections": n_dets,
            "num_gt": len(gt_masks),
            "per_class_ap_0.5": {
                int(k): round(v, 4) for k, v in img_aps_50.items()
            },
            "mAP_0.5": round(img_map_50, 4),
            "mAP_0.5_0.95": round(img_map_5095, 4),
        })
        if profile_enabled:
            image_results[-1]["time_s"] = round(img_time, 2)
        with open(result_path, "w") as f:
            json.dump(image_results[-1], f, indent=2, default=str)

        # Free per-image memory
        del detections, proposals, query_cls, query_appe
        del pred_masks, pred_masks_bin
        gc.collect()

    total_eval_time = time.time() - t_eval_start

    # =================================================================
    #  5. Global VOC-style mAP - across all images
    # =================================================================
    print(f"\n{'='*65}")
    print(
        f"  OVERALL RESULTS  ({evaluated_images}/{n_images} images, "
        f"{total_eval_time:.1f}s)"
    )
    print(f"{'='*65}")

    all_obj_ids_seen = set()
    for t in IOU_THRESHOLDS:
        all_obj_ids_seen.update(global_per_class[t].keys())

    class_aps_per_thresh = {}
    for t in IOU_THRESHOLDS:
        class_aps_per_thresh[t] = {}
        for obj_id in sorted(all_obj_ids_seen):
            info = global_per_class[t][obj_id]
            ap = compute_global_ap_voc(
                np.array(info["scores"]) if info["scores"]
                else np.array([]),
                info["matched"] if info["matched"] else [],
                info["n_gt"],
            )
            class_aps_per_thresh[t][obj_id] = ap

    print(
        f"\n  {'obj':>5s}  {'AP@.5':>8s}  {'AP@.75':>8s}  "
        f"{'AP@[.5:.95]':>12s}  {'GT':>4s}  {'dets':>5s}"
    )
    print(
        f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*4}  {'-'*5}"
    )
    class_ap_5095 = {}
    for obj_id in sorted(all_obj_ids_seen):
        ap50 = class_aps_per_thresh[0.5].get(obj_id, 0.0)
        ap75 = class_aps_per_thresh[0.75].get(obj_id, 0.0)
        ap_avg = float(np.mean(
            [class_aps_per_thresh[t].get(obj_id, 0.0) for t in IOU_THRESHOLDS]
        ))
        class_ap_5095[obj_id] = ap_avg
        n_gt = global_per_class[0.5][obj_id]["n_gt"]
        n_det = len(global_per_class[0.5][obj_id]["scores"])
        print(
            f"  {obj_id:>5d}  {ap50:>8.4f}  {ap75:>8.4f}  "
            f"{ap_avg:>12.4f}  {n_gt:>4d}  {n_det:>5d}"
        )

    overall_map_50 = float(
        np.mean(list(class_aps_per_thresh[0.5].values()))
    ) if class_aps_per_thresh[0.5] else 0.0
    overall_map_75 = float(
        np.mean(list(class_aps_per_thresh[0.75].values()))
    ) if class_aps_per_thresh[0.75] else 0.0
    overall_map_5095 = float(
        np.mean(list(class_ap_5095.values()))
    ) if class_ap_5095 else 0.0

    print(f"\n  ** mAP@0.5       = {overall_map_50:.4f} **")
    print(f"  ** mAP@0.75      = {overall_map_75:.4f} **")
    print(f"  ** mAP@[.5:.95]  = {overall_map_5095:.4f} **")
    print(f"{'='*65}")

    # -- Save summary -------------------------------------------------
    summary = {
        "dataset": dataset_name,
        "num_images": n_images,
        "evaluated_images": evaluated_images,
        "total_runtime_s": round(total_eval_time, 2),
        "ov_device": args.ov_device,
        "segmentor_model": args.segmentor_model,
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "overall_mAP_0.5": round(overall_map_50, 4),
        "overall_mAP_0.75": round(overall_map_75, 4),
        "overall_mAP_0.5_0.95": round(overall_map_5095, 4),
        "per_class_AP_0.5": {
            int(k): round(v, 4)
            for k, v in class_aps_per_thresh[0.5].items()
        },
        "per_class_AP_0.5_0.95": {
            int(k): round(v, 4) for k, v in class_ap_5095.items()
        },
        "per_image": image_results,
        "config": {
            "ov_device": args.ov_device,
            "ext": args.ext,
            "segmentor_model": args.segmentor_model,
            "points_per_side": args.points_per_side,
            "stability_score_thresh": args.stability_score_thresh,
            "obj_ids": target_obj_ids,
        },
    }
    if profile_enabled:
        summary["init_time_s"] = round(init_time, 2)
        summary["ref_time_s"] = round(ref_time, 2)
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved to: {summary_path}")
    print(f"  Results dir: {output_dir}")


if __name__ == "__main__":
    main()
