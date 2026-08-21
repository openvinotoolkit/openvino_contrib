# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
AR evaluation for Pose Estimation Model (PEM)
BOP19 metrics: AR = mean(AR_VSD, AR_MSSD, AR_MSPD)
"""

import os
import json
import time
import numpy as np
from collections import defaultdict


# ---------------------------------------------------------------------------
# BOP19 threshold grids
# ---------------------------------------------------------------------------
MSSD_THRESHOLDS = np.arange(0.05, 0.51, 0.05)  # 10 values, fraction of diameter
MSPD_THRESHOLDS = np.arange(5, 51, 5)            # 10 values, in pixels
VSD_TAUS        = np.arange(0.05, 0.51, 0.05)    # 10 values, fraction of diameter
VSD_CORRECT_TH  = np.arange(0.05, 0.51, 0.05)    # 10 values
VSD_DELTA_MM    = 15.0                             # visibility tolerance [mm]


# ---------------------------------------------------------------------------
# Symmetry helpers
# ---------------------------------------------------------------------------
def _rotation_matrix(angle, axis):
    """Rodrigues rotation."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def load_symmetries(models_info, obj_id):
    """Return list of symmetry transforms [{R: 3x3, t: 3x1}, ...].
    Always includes identity."""
    info = models_info[str(obj_id)]
    syms = [{"R": np.eye(3), "t": np.zeros(3)}]

    for s in info.get("symmetries_discrete", []):
        M = np.array(s).reshape(4, 4)
        syms.append({"R": M[:3, :3], "t": M[:3, 3]})

    for s in info.get("symmetries_continuous", []):
        axis = np.array(s["axis"], dtype=np.float64)
        offset = np.array(s["offset"], dtype=np.float64)
        n_steps = int(np.ceil(np.pi / 0.01))  # ~315 steps
        step = 2 * np.pi / n_steps
        for i in range(1, n_steps):  # skip 0 (identity already added)
            angle = i * step
            R = _rotation_matrix(angle, axis)
            t = -R @ offset + offset
            syms.append({"R": R, "t": t})

    return syms


# ---------------------------------------------------------------------------
# PLY loading
# ---------------------------------------------------------------------------
def load_ply(path):
    """Load a PLY mesh and return (vertices Nx3, faces Mx3) as float64/int32."""
    from plyfile import PlyData
    ply = PlyData.read(path)
    v = ply["vertex"]
    verts = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    f = ply["face"]
    faces = np.vstack(f["vertex_indices"]).astype(np.int32)
    return verts, faces


def make_pyrender_mesh(verts, faces):
    """Create a pyrender Mesh from raw vertex/face arrays."""
    import pyrender
    verts_f32 = verts.astype(np.float32)
    primitive = pyrender.Primitive(positions=verts_f32, indices=faces)
    return pyrender.Mesh(primitives=[primitive])


# ---------------------------------------------------------------------------
# MSSD  (Maximum Symmetry-Aware Surface Distance)
# ---------------------------------------------------------------------------
def compute_mssd(R_est, t_est, R_gt, t_gt, pts, syms):
    """
    MSSD = min_{S in syms} max_{x in pts} ||R_est*x + t_est - (R_gt*S_R*x + R_gt*S_t + t_gt)||
    Returns error in mm.
    """
    pts_est = (R_est @ pts.T + t_est.reshape(3, 1))  # 3 x N

    best = float("inf")
    for sym in syms:
        R_gt_sym = R_gt @ sym["R"]
        t_gt_sym = R_gt @ sym["t"] + t_gt
        pts_gt_sym = (R_gt_sym @ pts.T + t_gt_sym.reshape(3, 1))  # 3 x N
        dists = np.linalg.norm(pts_est - pts_gt_sym, axis=0)
        e = float(dists.max())
        if e < best:
            best = e
    return best


# ---------------------------------------------------------------------------
# MSPD  (Maximum Symmetry-Aware Projection Distance)
# ---------------------------------------------------------------------------
def _project_pts(pts_3d, K, R, t):
    """Project 3D model points to 2D image coordinates.
    Returns Nx2 array of pixel coordinates."""
    pts_cam = (R @ pts_3d.T + t.reshape(3, 1))  # 3 x N
    pts_2d = K @ pts_cam  # 3 x N
    pts_2d = pts_2d[:2, :] / pts_2d[2:, :]  # 2 x N
    return pts_2d.T  # N x 2


def compute_mspd(R_est, t_est, R_gt, t_gt, pts, K, syms):
    """
    MSPD = min_{S in syms} max_{x in pts} ||proj(est, x) - proj(gt*S, x)||
    Returns error in pixels.
    """
    proj_est = _project_pts(pts, K, R_est, t_est)  # N x 2

    best = float("inf")
    for sym in syms:
        R_gt_sym = R_gt @ sym["R"]
        t_gt_sym = R_gt @ sym["t"] + t_gt
        proj_gt_sym = _project_pts(pts, K, R_gt_sym, t_gt_sym)
        dists = np.linalg.norm(proj_est - proj_gt_sym, axis=1)
        e = float(dists.max())
        if e < best:
            best = e
    return best


# ---------------------------------------------------------------------------
# VSD  (Visible Surface Discrepancy)
# ---------------------------------------------------------------------------
def _depth_to_distance(depth, K):
    """Convert z-buffer depth map to Euclidean distance from camera map."""
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = np.arange(w).reshape(1, -1).astype(np.float64)
    v = np.arange(h).reshape(-1, 1).astype(np.float64)
    factor = np.sqrt(((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2 + 1.0)
    return depth * factor


def _render_depth(renderer, mesh_pyrender, K, R, t, w, h):
    """Render z-buffer depth of mesh at pose (R, t) in BOP convention.
    Returns depth map in mm (float64, HxW)."""
    import pyrender

    scene = pyrender.Scene()
    mesh_pose = np.eye(4)
    mesh_pose[:3, :3] = R
    mesh_pose[:3, 3] = t.flatten()
    scene.add(mesh_pyrender, pose=mesh_pose)

    camera = pyrender.IntrinsicsCamera(
        fx=float(K[0, 0]), fy=float(K[1, 1]),
        cx=float(K[0, 2]), cy=float(K[1, 2]),
        znear=0.01, zfar=10000.0
    )
    cam_pose_gl = np.diag([1.0, -1.0, -1.0, 1.0])
    scene.add(camera, pose=cam_pose_gl)

    _, depth = renderer.render(scene)
    return depth.astype(np.float64)


def compute_vsd(R_est, t_est, R_gt, t_gt, depth_test, K, renderer,
                mesh_pyrender, diameter, delta=VSD_DELTA_MM):
    """
    Compute VSD error for all tau values.
    Returns dict: {tau_idx: vsd_error} where lower is better.
    """
    h, w = depth_test.shape

    dist_test = _depth_to_distance(depth_test, K)

    depth_gt = _render_depth(renderer, mesh_pyrender, K, R_gt, t_gt, w, h)
    dist_gt = _depth_to_distance(depth_gt, K)

    depth_est = _render_depth(renderer, mesh_pyrender, K, R_est, t_est, w, h)
    dist_est = _depth_to_distance(depth_est, K)

    visib_gt = (dist_gt > 0) & ((dist_test == 0) | (dist_test >= dist_gt - delta))
    visib_est = (dist_est > 0) & ((dist_test == 0) | (dist_test >= dist_est - delta))

    visib_union = visib_gt | visib_est
    visib_inter = visib_gt & visib_est

    union_count = visib_union.sum()
    if union_count == 0:
        return {i: 1.0 for i in range(len(VSD_TAUS))}

    complement_count = union_count - visib_inter.sum()
    diffs = np.abs(dist_gt[visib_inter] - dist_est[visib_inter])
    diffs_norm = diffs / diameter

    vsd_per_tau = {}
    for ti, tau in enumerate(VSD_TAUS):
        costs = (diffs_norm >= tau).astype(np.float64)
        e = float((costs.sum() + complement_count) / union_count)
        vsd_per_tau[ti] = e

    return vsd_per_tau


# ---------------------------------------------------------------------------
# AR computation
# ---------------------------------------------------------------------------
def compute_ar_mssd(errors, diameters):
    """Compute AR_MSSD: mean recall across 10 thresholds (fraction of diameter)."""
    if not errors:
        return 0.0
    recalls = []
    for th in MSSD_THRESHOLDS:
        correct = sum(1 for e, d in zip(errors, diameters) if e / d < th)
        recalls.append(correct / len(errors))
    return float(np.mean(recalls))


def compute_ar_mspd(errors):
    """Compute AR_MSPD: mean recall across 10 pixel thresholds."""
    if not errors:
        return 0.0
    recalls = []
    for th in MSPD_THRESHOLDS:
        correct = sum(1 for e in errors if e < th)
        recalls.append(correct / len(errors))
    return float(np.mean(recalls))


def compute_ar_vsd(all_vsd_per_tau):
    """Compute AR_VSD: mean recall across 10 taus x 10 correct_th = 100 cells."""
    if not all_vsd_per_tau:
        return 0.0
    recalls = []
    for ti in range(len(VSD_TAUS)):
        for cth in VSD_CORRECT_TH:
            correct = sum(1 for vsd in all_vsd_per_tau if vsd[ti] < cth)
            recalls.append(correct / len(all_vsd_per_tau))
    return float(np.mean(recalls))


# ---------------------------------------------------------------------------
# High-level: evaluate PEM predictions and print AR
# ---------------------------------------------------------------------------
def evaluate_and_print_ar(detection_pem_path, gt_targets, model_data,
                          cam_data, scene_dir, skip_vsd=False):
    """Evaluate PEM predictions against GT and print AR metrics.

    Parameters
    ----------
    detection_pem_path : str
        Path to detection_pem.json (output of PEM inference).
    gt_targets : list of dict
        Each dict has: obj_id, image_id, R (3x3), t (3,) — GT pose.
    model_data : dict
        {obj_id: {"pts": Nx3, "diameter": float, "syms": list,
                   "mesh_pyrender": pyrender.Mesh (optional)}}
    cam_data : dict
        {image_id_str: {"cam_K": list(9), "depth_scale": float}}
    scene_dir : str
        Path to BOP scene dir (for depth images).
    skip_vsd : bool
        If True, skip VSD computation (no renderer needed).

    Returns
    -------
    dict with keys: AR_MSSD, AR_MSPD, AR_VSD (or None), AR (or None),
                    per_object, all_results
    """
    with open(detection_pem_path) as f:
        pem_dets = json.load(f)

    # Build lookup: best detection per (image_id, obj_id) by score
    best_by_key = {}
    for det in pem_dets:
        # PEM detection_pem.json stores R as flat list (9 elements) and t as flat list (3)
        key = (det.get("image_id", 0), det.get("category_id", 1))
        score = det.get("score", 0.0)
        if key not in best_by_key or score > best_by_key[key]["score"]:
            best_by_key[key] = det

    renderer = None
    renderer_size = (0, 0)  # (w, h) — will be initialized on first depth load
    _pyrender = None
    if not skip_vsd:
        try:
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
            import pyrender
            _pyrender = pyrender
        except Exception as e:
            print(f"  [WARN] Cannot import pyrender for VSD: {e}")
            skip_vsd = True

    mssd_errors = []
    mssd_diameters = []
    mspd_errors = []
    vsd_errors = []
    all_results = []
    depth_cache = {}
    t_start = time.time()

    for i, gt in enumerate(gt_targets):
        oid = gt["obj_id"]
        img_id = gt["image_id"]
        gt_R = np.array(gt["R"]).reshape(3, 3)
        gt_t = np.array(gt["t"]).reshape(3)

        key = (img_id, oid)
        det = best_by_key.get(key)
        if det is None:
            continue

        pred_R = np.array(det["R"]).reshape(3, 3)
        pred_t = np.array(det["t"]).reshape(3)

        img_id_str = str(img_id)
        cam_info = cam_data[img_id_str]
        K = np.array(cam_info["cam_K"]).reshape(3, 3)

        pts = model_data[oid]["pts"]
        diameter = model_data[oid]["diameter"]
        syms = model_data[oid]["syms"]

        mssd = compute_mssd(pred_R, pred_t, gt_R, gt_t, pts, syms)
        mssd_errors.append(mssd)
        mssd_diameters.append(diameter)

        mspd = compute_mspd(pred_R, pred_t, gt_R, gt_t, pts, K, syms)
        mspd_errors.append(mspd)

        vsd_per_tau = None
        if not skip_vsd and _pyrender is not None:
            if img_id not in depth_cache:
                # BOP layout expects: <scene_dir>/depth/000000.png
                depth_path_bop = os.path.join(scene_dir, "depth", f"{img_id:06d}.png")

                # Example layout fallback: <scene_dir>/depth.png
                depth_path_example = os.path.join(scene_dir, "depth.png")

                depth_path = depth_path_bop if os.path.exists(depth_path_bop) else depth_path_example

                if os.path.exists(depth_path):
                    from PIL import Image
                    depth_raw = np.array(Image.open(depth_path)).astype(np.float64)
                    depth_scale = cam_info.get("depth_scale", 1.0)
                    depth_cache[img_id] = depth_raw * depth_scale
                else:
                    depth_cache[img_id] = None
            depth_mm = depth_cache[img_id]
            if depth_mm is not None:
                # Initialize/resize renderer to match actual depth image size
                h, w = depth_mm.shape[:2]
                if renderer is None or renderer_size != (w, h):
                    if renderer is not None:
                        renderer.delete()
                    renderer = _pyrender.OffscreenRenderer(w, h)
                    renderer_size = (w, h)

                vsd_per_tau = compute_vsd(
                    pred_R, pred_t, gt_R, gt_t, depth_mm, K,
                    renderer, model_data[oid]["mesh_pyrender"],
                    diameter, delta=VSD_DELTA_MM
                )
                vsd_errors.append(vsd_per_tau)

        result = {
            "obj_id": oid, "image_id": img_id,
            "diameter_mm": diameter,
            "mssd_mm": mssd, "mssd_norm": mssd / diameter,
            "mspd_px": mspd,
        }
        if vsd_per_tau is not None:
            result["vsd_per_tau"] = vsd_per_tau
        all_results.append(result)

        if (i + 1) % 5 == 0 or (i + 1) == len(gt_targets):
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(gt_targets) - i - 1) / rate if rate > 0 else 0
            print(f"\r  [{i+1:>4d}/{len(gt_targets)}]  "
                  f"rate={rate:.1f} tc/s  ETA={eta:.0f}s", end="", flush=True)

    print("\n")
    t_elapsed = time.time() - t_start

    if renderer:
        renderer.delete()

    if not all_results:
        print("  No PEM predictions matched GT targets!")
        return None

    ar_mssd = compute_ar_mssd(mssd_errors, mssd_diameters)
    ar_mspd = compute_ar_mspd(mspd_errors)
    ar_vsd = None
    ar = None
    if vsd_errors:
        ar_vsd = compute_ar_vsd(vsd_errors)
        ar = (ar_vsd + ar_mssd + ar_mspd) / 3.0

    # Per-object breakdown
    by_obj = defaultdict(lambda: {"mssd": [], "mspd": [], "vsd": [],
                                  "diameters": []})
    for r in all_results:
        oid = r["obj_id"]
        by_obj[oid]["mssd"].append(r["mssd_mm"])
        by_obj[oid]["mspd"].append(r["mspd_px"])
        by_obj[oid]["diameters"].append(r["diameter_mm"])
        if "vsd_per_tau" in r:
            by_obj[oid]["vsd"].append(r["vsd_per_tau"])

    per_object = {}
    for oid in sorted(by_obj.keys()):
        obj = by_obj[oid]
        n = len(obj["mssd"])
        obj_ar_mssd = compute_ar_mssd(obj["mssd"], obj["diameters"])
        obj_ar_mspd = compute_ar_mspd(obj["mspd"])
        if obj["vsd"]:
            obj_ar_vsd = compute_ar_vsd(obj["vsd"])
            obj_ar = (obj_ar_vsd + obj_ar_mssd + obj_ar_mspd) / 3.0
        else:
            obj_ar_vsd = None
            obj_ar = None

        per_object[str(oid)] = {
            "n_testcases": n,
            "AR_MSSD": obj_ar_mssd * 100,
            "AR_MSPD": obj_ar_mspd * 100,
            "AR_VSD": obj_ar_vsd * 100 if obj_ar_vsd is not None else None,
            "AR": obj_ar * 100 if obj_ar is not None else None,
        }

        print(f"  Object {oid} ({n} tc, d={obj['diameters'][0]:.0f}mm)"
              f"  MSSD={obj_ar_mssd*100:.1f}%  MSPD={obj_ar_mspd*100:.1f}%",
              end="")
        if obj_ar_vsd is not None:
            print(f"  VSD={obj_ar_vsd*100:.1f}%  AR={obj_ar*100:.1f}%")
        else:
            print()

    print(f"\n  {'='*55}")
    print(f"  PEM AR ({len(all_results)} testcases)")
    print(f"  {'='*55}")
    print(f"    AR_MSSD : {ar_mssd*100:6.2f}%")
    print(f"    AR_MSPD : {ar_mspd*100:6.2f}%")
    if ar_vsd is not None:
        print(f"    AR_VSD  : {ar_vsd*100:6.2f}%")
        print(f"    AR      : {ar*100:6.2f}%")
    else:
        print(f"    AR (MSSD+MSPD only) : {(ar_mssd+ar_mspd)/2*100:6.2f}%")
        print(f"    (VSD skipped — pass skip_vsd=False for full AR)")
    print(f"    Eval time: {t_elapsed:.1f}s")

    return {
        "AR_MSSD": ar_mssd * 100,
        "AR_MSPD": ar_mspd * 100,
        "AR_VSD": ar_vsd * 100 if ar_vsd is not None else None,
        "AR": ar * 100 if ar is not None else None,
        "per_object": per_object,
        "all_results": all_results,
    }
