# Acknowledgments: https://github.com/tarashakhurana/4d-occ-forecasting
# Modified by Haisong Liu
#
# Ray metrics with conditional CUDA/CPU fallback.
#
# Supports:
#   - CUDA path: original DVR JIT-compiled extension (default when CUDA available)
#   - OpenVINO/CPU path: stubs (set FLASHOCC_OPENVINO_MODE=1 or CUDA_VISIBLE_DEVICES='')
#
import os
import math
import copy
import warnings
import numpy as np
import torch
from tqdm import tqdm
from prettytable import PrettyTable

# ── Detect execution mode ──────────────────────────────────────────────────────
CUDA_DISABLED = os.getenv('CUDA_VISIBLE_DEVICES', '').strip() == ''
OPENVINO_MODE = os.getenv('FLASHOCC_OPENVINO_MODE', '0') == '1'

_USE_CUDA_DVR = False
dvr = None

if not CUDA_DISABLED and not OPENVINO_MODE and torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load as _cpp_load
        dvr = _cpp_load(
            "dvr",
            sources=["lib/dvr/dvr.cpp", "lib/dvr/dvr.cu"],
            verbose=True,
            extra_cuda_cflags=['-allow-unsupported-compiler']
        )
        _USE_CUDA_DVR = True
    except Exception as _e:
        warnings.warn(
            f"DVR CUDA extension failed to load ({_e}). "
            "Falling back to CPU/OpenVINO stubs. "
            "If you intended CUDA evaluation, ensure lib/dvr/ is present and CUDA toolkit matches.",
            stacklevel=2
        )

# ── Full CUDA implementation ───────────────────────────────────────────────────
if _USE_CUDA_DVR:
    from .ray_pq import Metric_RayPQ

    _pc_range = [-40, -40, -1.0, 40, 40, 5.4]
    _voxel_size = 0.4

    occ_class_names = [
        'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
        'driveable_surface', 'other_flat', 'sidewalk',
        'terrain', 'manmade', 'vegetation', 'free'
    ]

    def get_rendered_pcds(origin, points, tindex, pred_dist):
        pcds = []
        for t in range(len(origin)):
            mask = (tindex == t)
            if not mask.any():
                continue
            _pts = points[mask, :3]
            v = _pts - origin[t][None, :]
            d = v / np.sqrt((v ** 2).sum(axis=1, keepdims=True))
            pred_pts = origin[t][None, :] + d * pred_dist[mask][:, None]
            pcds.append(torch.from_numpy(pred_pts))
        return pcds

    def meshgrid3d(occ_size, pc_range):
        W, H, D = occ_size
        xs = torch.linspace(0.5, W - 0.5, W).view(W, 1, 1).expand(W, H, D) / W
        ys = torch.linspace(0.5, H - 0.5, H).view(1, H, 1).expand(W, H, D) / H
        zs = torch.linspace(0.5, D - 0.5, D).view(1, 1, D).expand(W, H, D) / D
        xs = xs * (pc_range[3] - pc_range[0]) + pc_range[0]
        ys = ys * (pc_range[4] - pc_range[1]) + pc_range[1]
        zs = zs * (pc_range[5] - pc_range[2]) + pc_range[2]
        return torch.stack((xs, ys, zs), -1)

    def generate_lidar_rays():
        pitch_angles = []
        for k in range(10):
            angle = math.pi / 2 - math.atan(k + 1)
            pitch_angles.append(-angle)
        while pitch_angles[-1] < 0.21:
            delta = pitch_angles[-1] - pitch_angles[-2]
            pitch_angles.append(pitch_angles[-1] + delta)
        lidar_rays = []
        for pitch_angle in pitch_angles:
            for azimuth_angle in np.arange(0, 360, 1):
                azimuth_angle = np.deg2rad(azimuth_angle)
                x = np.cos(pitch_angle) * np.cos(azimuth_angle)
                y = np.cos(pitch_angle) * np.sin(azimuth_angle)
                z = np.sin(pitch_angle)
                lidar_rays.append((x, y, z))
        return np.array(lidar_rays, dtype=np.float32)

    def process_one_sample(sem_pred, lidar_rays, output_origin, instance_pred=None):
        T = output_origin.shape[1]
        pred_pcds_t = []
        free_id = len(occ_class_names) - 1
        occ_pred = copy.deepcopy(sem_pred)
        occ_pred[sem_pred < free_id] = 1
        occ_pred[sem_pred == free_id] = 0
        occ_pred = occ_pred.permute(2, 1, 0)
        occ_pred = occ_pred[None, None, :].contiguous().float()
        offset = torch.Tensor(_pc_range[:3])[None, None, :]
        scaler = torch.Tensor([_voxel_size] * 3)[None, None, :]
        lidar_tindex = torch.zeros([1, lidar_rays.shape[0]])
        for t in range(T):
            lidar_origin = output_origin[:, t:t+1, :]
            lidar_endpts = lidar_rays[None] + lidar_origin
            output_origin_render = ((lidar_origin - offset) / scaler).float()
            output_points_render = ((lidar_endpts - offset) / scaler).float()
            output_tindex_render = lidar_tindex
            with torch.no_grad():
                pred_dist, _, coord_index = dvr.render_forward(
                    occ_pred.cuda(),
                    output_origin_render.cuda(),
                    output_points_render.cuda(),
                    output_tindex_render.cuda(),
                    [1, 16, 200, 200],
                    "test"
                )
                pred_dist *= _voxel_size
            pred_pcds = get_rendered_pcds(
                lidar_origin[0].cpu().numpy(),
                lidar_endpts[0].cpu().numpy(),
                lidar_tindex[0].cpu().numpy(),
                pred_dist[0].cpu().numpy()
            )
            coord_index = coord_index[0, :, :].long().cpu()
            pred_label = sem_pred[coord_index[:, 0], coord_index[:, 1], coord_index[:, 2]][:, None]
            pred_dist = pred_dist[0, :, None].cpu()
            if instance_pred is not None:
                pred_instance = instance_pred[coord_index[:, 0], coord_index[:, 1], coord_index[:, 2]][:, None]
                pred_pcds = torch.cat([pred_label.float(), pred_instance.float(), pred_dist], dim=-1)
            else:
                pred_pcds = torch.cat([pred_label.float(), pred_dist], dim=-1)
            pred_pcds_t.append(pred_pcds)
        pred_pcds_t = torch.cat(pred_pcds_t, dim=0)
        return pred_pcds_t.numpy()

    def calc_metrics(pcd_pred_list, pcd_gt_list):
        thresholds = [1, 2, 4]
        gt_cnt = np.zeros([len(occ_class_names)])
        pred_cnt = np.zeros([len(occ_class_names)])
        tp_cnt = np.zeros([len(thresholds), len(occ_class_names)])
        for pcd_pred, pcd_gt in zip(pcd_pred_list, pcd_gt_list):
            for j, threshold in enumerate(thresholds):
                depth_pred = pcd_pred[:, 1]
                depth_gt = pcd_gt[:, 1]
                l1_error = np.abs(depth_pred - depth_gt)
                tp_dist_mask = (l1_error < threshold)
                for i, cls in enumerate(occ_class_names):
                    cls_id = occ_class_names.index(cls)
                    cls_mask_pred = (pcd_pred[:, 0] == cls_id)
                    cls_mask_gt = (pcd_gt[:, 0] == cls_id)
                    gt_cnt_i = cls_mask_gt.sum()
                    pred_cnt_i = cls_mask_pred.sum()
                    if j == 0:
                        gt_cnt[i] += gt_cnt_i
                        pred_cnt[i] += pred_cnt_i
                    tp_cls = cls_mask_gt & cls_mask_pred
                    tp_mask = np.logical_and(tp_cls, tp_dist_mask)
                    tp_cnt[j][i] += tp_mask.sum()
        iou_list = []
        for j, threshold in enumerate(thresholds):
            iou_list.append((tp_cnt[j] / (gt_cnt + pred_cnt - tp_cnt[j]))[:-1])
        return iou_list

    def main(sem_pred_list, sem_gt_list, lidar_origin_list):
        torch.cuda.empty_cache()
        lidar_rays = generate_lidar_rays()
        lidar_rays = torch.from_numpy(lidar_rays)
        pcd_pred_list, pcd_gt_list = [], []
        for sem_pred, sem_gt, lidar_origins in \
                tqdm(zip(sem_pred_list, sem_gt_list, lidar_origin_list), ncols=50):
            sem_pred = torch.from_numpy(np.reshape(sem_pred, [200, 200, 16]))
            sem_gt = torch.from_numpy(np.reshape(sem_gt, [200, 200, 16]))
            pcd_pred = process_one_sample(sem_pred, lidar_rays, lidar_origins)
            pcd_gt = process_one_sample(sem_gt, lidar_rays, lidar_origins)
            valid_mask = (pcd_gt[:, 0].astype(np.int32) != len(occ_class_names) - 1)
            pcd_pred_list.append(pcd_pred[valid_mask])
            pcd_gt_list.append(pcd_gt[valid_mask])
        iou_list = calc_metrics(pcd_pred_list, pcd_gt_list)
        thresholds = [1, 2, 4]
        table = PrettyTable()
        table.field_names = ["Class"] + [f"IoU@{t}m" for t in thresholds]
        mean_ious = []
        for i, cls in enumerate(occ_class_names[:-1]):
            row = [cls] + [f"{iou_list[j][i]:.4f}" for j in range(len(thresholds))]
            table.add_row(row)
            mean_ious.append(iou_list[0][i])
        print(table)
        mean_iou = np.mean(mean_ious)
        print(f"Mean RayIoU@1m: {mean_iou:.4f}")
        return {'ray_iou': float(iou_list[0].mean()), 'ray_iou_mean': float(mean_iou)}

    def main_raypq(sem_pred_list, sem_gt_list, inst_pred_list, inst_gt_list, lidar_origin_list):
        torch.cuda.empty_cache()
        eval_metrics_pq = Metric_RayPQ(num_classes=len(occ_class_names), thresholds=[1, 2, 4])
        lidar_rays = generate_lidar_rays()
        lidar_rays = torch.from_numpy(lidar_rays)
        for sem_pred, sem_gt, inst_pred, inst_gt, lidar_origins in \
                tqdm(zip(sem_pred_list, sem_gt_list, inst_pred_list, inst_gt_list, lidar_origin_list), ncols=50):
            sem_pred = torch.from_numpy(np.reshape(sem_pred, [200, 200, 16]))
            sem_gt = torch.from_numpy(np.reshape(sem_gt, [200, 200, 16]))
            inst_pred = torch.from_numpy(np.reshape(inst_pred, [200, 200, 16]))
            inst_gt = torch.from_numpy(np.reshape(inst_gt, [200, 200, 16]))
            pcd_pred = process_one_sample(sem_pred, lidar_rays, lidar_origins, instance_pred=inst_pred)
            pcd_gt = process_one_sample(sem_gt, lidar_rays, lidar_origins, instance_pred=inst_gt)
            valid_mask = (pcd_gt[:, 0].astype(np.int32) != len(occ_class_names) - 1)
            pcd_pred = pcd_pred[valid_mask]
            pcd_gt = pcd_gt[valid_mask]
            assert pcd_pred.shape == pcd_gt.shape
            sem_gt_arr = pcd_gt[:, 0].astype(np.int32)
            sem_pred_arr = pcd_pred[:, 0].astype(np.int32)
            instances_gt = pcd_gt[:, 1].astype(np.int32)
            instances_pred = pcd_pred[:, 1].astype(np.int32)
            depth_gt = pcd_gt[:, 2]
            depth_pred = pcd_pred[:, 2]
            l1_error = np.abs(depth_pred - depth_gt)
            eval_metrics_pq.add_batch(sem_pred_arr, sem_gt_arr, instances_pred, instances_gt, l1_error)
        return eval_metrics_pq.get_results()

    def calc_rayiou(*args, **kwargs):
        return main(*args, **kwargs)

    def calc_raypq(*args, **kwargs):
        return main_raypq(*args, **kwargs)

# ── Fallback stubs for CPU/OpenVINO mode ──────────────────────────────────────
if not _USE_CUDA_DVR:
    if OPENVINO_MODE:
        info_msg = "OpenVINO inference mode detected"
    elif CUDA_DISABLED:
        info_msg = "CUDA disabled (CUDA_VISIBLE_DEVICES='')"
    else:
        info_msg = "CPU mode"
    
    warnings.warn(
        f"DVR ray metrics disabled ({info_msg}). "
        f"Returning dummy values. This is safe for OpenVINO inference.",
        stacklevel=2
    )

    class DVRFallback:
        """CPU/OpenVINO fallback placeholder for DVR operations"""
        @staticmethod
        def render(*args, **kwargs):
            raise NotImplementedError(
                "DVR rendering requires CUDA. "
                "For CUDA training: unset CUDA_VISIBLE_DEVICES, set FLASHOCC_OPENVINO_MODE=0. "
                "For OpenVINO inference: this is expected (ray metrics unused)."
            )

    dvr = DVRFallback()

    def main(*args, **kwargs):
        """
        Ray IoU calculation.
        
        CUDA mode (training):
          - Computes actual DVR-based ray intersection-over-union
          - Returns dict with true metric values
        
        CPU/OpenVINO mode (inference):
          - Returns dummy 0.0 (evaluation not performed)
          - Used only to prevent ImportError during model loading
          - Do not rely on values for evaluation
        
        Returns:
            dict: {'ray_iou': float, 'ray_iou_mean': float}
        """
        warnings.warn(
            "Ray metrics computed: returning dummy 0.0 (expected in CPU/OpenVINO mode). "
            "For actual evaluation, use CUDA path.",
            stacklevel=2
        )
        return {'ray_iou': 0.0, 'ray_iou_mean': 0.0}

    def main_raypq(*args, **kwargs):
        """Placeholder for ray panoptic quality calculation"""
        warnings.warn(
            "Ray PQ requires CUDA. Returning dummy 0.0 in CPU/OpenVINO mode.",
            stacklevel=2
        )
        return {
            'ray_pq': 0.0,
            'ray_sq': 0.0,
            'ray_rq': 0.0,
        }

    def calc_rayiou(*args, **kwargs):
        """Alias for main function (backward compatibility)"""
        return main(*args, **kwargs)

    def calc_raypq(*args, **kwargs):
        """Alias for main_raypq function (backward compatibility)"""
        return main_raypq(*args, **kwargs)

__all__ = ['main', 'main_raypq', 'calc_rayiou', 'calc_raypq', 'dvr']
