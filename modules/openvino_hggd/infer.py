# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""HGGD OpenVINO Inference - Self-contained Intel iGPU package.

Uses OpenVINO for neural network inference and native C++ custom ops
for point cloud operations (KNN, FPS, gather) on Intel iGPU.

Usage:
    python infer.py \
        --ov-device GPU --precision-hint default \
        --checkpoint-path <path> --dataset-path <path> --scene-path <path> \
        --scene-l 100 --scene-r 101 \
        --dump-dir ./output/scene_100/pred
"""
import sys
import os

# ── Install OpenVINO Native Extension v3 shim BEFORE any HGGD imports ────────────
xpu_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, xpu_dir)
from ov_gpu_extensions.ov_shim_gpu import install_ov_shim
install_ov_shim(verbose=True)

# ── Device Monkey-Patches: cuda → cpu for data pipeline ───────────────
import torch
import torch.cuda
import functools

def _tensor_cuda_noop(self, *args, **kwargs):
    return self
torch.Tensor.cuda = _tensor_cuda_noop

_orig_to = torch.Tensor.to
def _patched_to(self, *args, **kwargs):
    new_args = list(args)
    for i, a in enumerate(new_args):
        if isinstance(a, str) and 'cuda' in a:
            new_args[i] = 'cpu'
        elif isinstance(a, torch.device) and a.type == 'cuda':
            new_args[i] = torch.device('cpu')
    if 'device' in kwargs:
        dev = kwargs['device']
        if isinstance(dev, str) and 'cuda' in dev:
            kwargs['device'] = 'cpu'
        elif isinstance(dev, torch.device) and getattr(dev, 'type', '') == 'cuda':
            kwargs['device'] = torch.device('cpu')
    return _orig_to(self, *tuple(new_args), **kwargs)
torch.Tensor.to = _patched_to

def _patch_device_arg(orig_fn):
    @functools.wraps(orig_fn)
    def wrapper(*args, **kwargs):
        if 'device' in kwargs:
            dev = kwargs['device']
            if isinstance(dev, str) and 'cuda' in dev:
                kwargs['device'] = 'cpu'
            elif isinstance(dev, torch.device) and getattr(dev, 'type', '') == 'cuda':
                kwargs['device'] = torch.device('cpu')
        return orig_fn(*args, **kwargs)
    return wrapper

for _fn_name in ['zeros', 'ones', 'empty', 'randn', 'randint', 'full',
                  'tensor', 'linspace', 'arange', 'zeros_like', 'ones_like',
                  'empty_like', 'randn_like', 'rand']:
    if hasattr(torch, _fn_name):
        setattr(torch, _fn_name, _patch_device_arg(getattr(torch, _fn_name)))

torch.cuda.synchronize = lambda *a, **k: None
torch.cuda.is_available = lambda: False
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.Tensor.is_cuda = property(lambda self: False)

import types
import open3d as o3d
import numpy as np

cupoch_stub = types.ModuleType("cupoch")
cupoch_stub.geometry = types.ModuleType("cupoch.geometry")
cupoch_stub.visualization = types.ModuleType("cupoch.visualization")
cupoch_stub.visualization.draw_geometries = lambda *a, **k: None
sys.modules["cupoch"] = cupoch_stub
sys.modules["cupoch.geometry"] = cupoch_stub.geometry
sys.modules["cupoch.visualization"] = cupoch_stub.visualization

# ── End Patches ────────────────────────────────────────────────────────

import argparse
import random
import json
import logging
from time import time

import openvino as ov
from tqdm import tqdm
from torch.utils.data import DataLoader

# Model code (customgraspnetAPI, dataset, models) is co-located in this directory
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from customgraspnetAPI import Grasp as GraspNetGrasp
from customgraspnetAPI import GraspGroup as GraspNetGraspGroup
from customgraspnetAPI import GraspNetEval
from dataset.config import camera
from dataset.evaluation import (anchor_output_process, detect_2d_grasp,
                                detect_6d_grasp_multi)
from dataset.graspnet_dataset import GraspnetPointDataset
from dataset.pc_dataset_tools import data_process, feature_fusion
# ── CPU collision detector ─────────────────────────────────────────────
import dataset.collision_detector as collision_module

class ModelFreeCollisionDetectorCPU:
    def __init__(self, scene_points, voxel_size=0.005, mode='regnet'):
        self.mode = mode
        if mode == 'regnet':
            self.height, self.finger_width, self.delta_width = 0.01, 0.01, 0
            self.finger_length, self.depth = 0.06, 0.025
        elif mode == 'graspnet':
            self.height, self.finger_width, self.delta_width = 0.01, 0.01, 0
            self.finger_length, self.depth = 0.04, 0.02
        elif mode == 'test':
            self.height, self.finger_width, self.delta_width = 0.02, 0.01, 0
            self.finger_length, self.depth = 0.04, 0.02
        else:
            raise ValueError(f'Invalid mode: {mode}')

        pts_np = scene_points.cpu().numpy() if isinstance(scene_points, torch.Tensor) else scene_points
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(pts_np.astype(np.float64))
        o3d_pc = o3d_pc.voxel_down_sample(voxel_size)
        self.scene_points = np.asarray(o3d_pc.points, dtype=np.float16)

    def detect(self, grasp_group, approach_dist=0.05):
        device = 'cpu'
        T = torch.from_numpy(grasp_group.translations).to(dtype=torch.float16, device=device)
        R = torch.from_numpy(grasp_group.rotations.reshape(-1, 3, 3)).to(dtype=torch.float16, device=device)
        heights = torch.full((grasp_group.size, 1), self.height, dtype=torch.float16, device=device)
        depths = torch.full((grasp_group.size, 1), self.depth, dtype=torch.float16, device=device)
        widths = torch.from_numpy(grasp_group.widths[:, None]).to(dtype=torch.float16, device=device)
        widths += self.delta_width
        points = torch.from_numpy(self.scene_points[None, ...]).to(dtype=torch.float16, device=device)
        targets = torch.matmul(points - T[:, None, :], R)
        mask1 = ((targets[..., 2] > -heights / 2) & (targets[..., 2] < heights / 2))
        mask2 = ((targets[..., 0] > depths - self.finger_length) & (targets[..., 0] < depths))
        mask3 = (targets[..., 1] > -(widths / 2 + self.finger_width))
        mask4 = (targets[..., 1] < -widths / 2)
        mask5 = (targets[..., 1] < (widths / 2 + self.finger_width))
        mask6 = (targets[..., 1] > widths / 2)
        mask7 = ((targets[..., 0] <= depths - self.finger_length)
                 & (targets[..., 0] > depths - self.finger_length - self.finger_width - approach_dist))
        depth_mask = (mask1 & mask2)
        left_mask = (depth_mask & mask3 & mask4)
        right_mask = (depth_mask & mask5 & mask6)
        shifting_mask = (mask1 & mask3 & mask5 & mask7)
        mask_between = (depth_mask & (~mask4) & (~mask6))
        mask_between = (mask_between.sum(1) > 2)
        finger_mask = (left_mask | right_mask)
        mask_finger = (finger_mask.sum(1) <= 0)
        mask_shift = (shifting_mask.sum(1) <= 0)
        no_collision_mask = (mask_between & mask_finger & mask_shift)
        return no_collision_mask.cpu().numpy()

collision_module.ModelFreeCollisionDetector = ModelFreeCollisionDetectorCPU


def collision_detect_cpu(points_all, pred_gg, mode='regnet'):
    cloud = points_all[:, :3].clone()
    mfcdetector = ModelFreeCollisionDetectorCPU(cloud, voxel_size=0.01, mode=mode)
    no_collision_mask = mfcdetector.detect(pred_gg, approach_dist=0.05)
    collision_free_gg = pred_gg[no_collision_mask]
    return collision_free_gg, no_collision_mask


# ── Argument Parser ────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--ov-device', required=True, choices=['CPU', 'GPU'], help='OpenVINO device')
parser.add_argument('--precision-hint', default='default', choices=['default', 'f16', 'f32'],
                    help='INFERENCE_PRECISION_HINT: default=plugin decides, f16=force FP16, f32=force FP32')
parser.add_argument('--checkpoint-path', required=True)
parser.add_argument('--anchornet-xml', default=None,
                    help='AnchorNet IR XML (default: openvino_models/anchornet_fp16.xml)')
parser.add_argument('--localnet-xml', default=None,
                    help='LocalNet IR XML (default: openvino_models/localnet_fp16.xml)')
parser.add_argument('--dataset-path', required=True)
parser.add_argument('--scene-path', required=True)
parser.add_argument('--scene-l', type=int, required=True)
parser.add_argument('--scene-r', type=int, required=True)
parser.add_argument('--grasp-count', type=int, default=5000)
parser.add_argument('--dump-dir', default='./pred/test')
parser.add_argument('--input-h', type=int, required=True)
parser.add_argument('--input-w', type=int, required=True)
parser.add_argument('--sigma', type=int, default=10)
parser.add_argument('--ratio', type=int, default=8)
parser.add_argument('--anchor-k', type=int, default=6)
parser.add_argument('--anchor-w', type=float, default=50.0)
parser.add_argument('--anchor-z', type=float, default=20.0)
parser.add_argument('--grid-size', type=int, default=8)
parser.add_argument('--anchor-num', type=int, required=True)
parser.add_argument('--all-points-num', type=int, required=True)
parser.add_argument('--center-num', type=int, required=True)
parser.add_argument('--group-num', type=int, required=True)
parser.add_argument('--heatmap-thres', type=float, default=0.01)
parser.add_argument('--local-k', type=int, default=10)
parser.add_argument('--local-thres', type=float, default=0.01)
parser.add_argument('--random-seed', type=int, default=123)
parser.add_argument('--no-gpu-padding', action='store_true',
                    help='Disable GPU padding workaround (to test if OV 2026.1 fixed dynamic shape bug)')
args = parser.parse_args()

# Default IR model paths relative to script dir
if args.anchornet_xml is None:
    args.anchornet_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'openvino_models', 'anchornet_fp16.xml')
if args.localnet_xml is None:
    args.localnet_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'openvino_models', 'localnet_fp16.xml')


def inference():
    core = ov.Core()
    ov_device = args.ov_device
    logging.info(f'OpenVINO {ov.__version__}, device: {ov_device}')
    dev_name = core.get_property(ov_device, "FULL_DEVICE_NAME")
    logging.info(f'Device name: {dev_name}')

    # Build config
    config = {}
    if args.precision_hint == 'f16':
        config["INFERENCE_PRECISION_HINT"] = "f16"
    elif args.precision_hint == 'f32':
        config["INFERENCE_PRECISION_HINT"] = "f32"
    # else: default (plugin decides)

    logging.info(f'Precision hint: {args.precision_hint} -> config={config}')

    logging.info(f'Loading AnchorGraspNet IR: {args.anchornet_xml}')
    anchor_compiled = core.compile_model(args.anchornet_xml, ov_device, config=config)
    logging.info(f'Loading PointMultiGraspNet IR: {args.localnet_xml}')
    local_compiled = core.compile_model(args.localnet_xml, ov_device, config=config)

    # Load anchors from checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    basic_ranges = torch.linspace(-1, 1, args.anchor_num + 1)
    basic_anchors = (basic_ranges[1:] + basic_ranges[:-1]) / 2
    anchors = {'gamma': basic_anchors, 'beta': basic_anchors}
    anchors['gamma'] = checkpoint['gamma'].cpu()
    anchors['beta'] = checkpoint['beta'].cpu()
    logging.info('Loaded saved anchors from checkpoint')

    sceneIds = list(range(args.scene_l, args.scene_r))
    test_dataset = GraspnetPointDataset(args.all_points_num,
                                        args.dataset_path,
                                        args.scene_path,
                                        sceneIds,
                                        sigma=args.sigma,
                                        ratio=args.ratio,
                                        anchor_k=args.anchor_k,
                                        anchor_z=args.anchor_z,
                                        anchor_w=args.anchor_w,
                                        grasp_count=args.grasp_count,
                                        output_size=(args.input_w, args.input_h),
                                        random_rotate=False,
                                        random_zoom=False)

    SCENE_LIST = test_dataset.scene_list()
    test_data = DataLoader(test_dataset, batch_size=1, pin_memory=False, num_workers=0)
    test_data.dataset.unaug()
    test_data.dataset.eval()
    total_frames = len(test_data)

    # Timing breakdown: separate OV inference from post-processing
    time_anchor_ov, time_anchor_post = 0, 0
    time_data = 0
    time_feat_fusion = 0
    time_get_group_pc = 0
    time_local_ov, time_local_post = 0, 0
    time_colli, time_nms = 0, 0
    time_wall = 0

    batch_idx = -1
    with torch.no_grad():
        pbar = tqdm(test_data, desc='Inference', unit='frame', total=total_frames)
        for anchor_data, rgb, ori_depth, grasppaths in pbar:
            batch_idx += 1
            frame_start = time()

            depth = ori_depth.numpy().squeeze()
            depth = torch.from_numpy(depth).float()[None]

            view_points, _, _ = test_data.dataset.helper.to_scene_points(
                rgb, depth, include_rgb=True)
            points = view_points[..., :3]
            xyzs = test_data.dataset.helper.to_xyz_maps(depth)

            # ── AnchorGraspNet ─────────────────────────────────────────
            # Time OV inference separately (for comparison with benchmark_app)
            x, _, _, _, _ = anchor_data
            start_ov = time()
            ov_result = anchor_compiled(x.numpy())
            if batch_idx >= 1:
                time_anchor_ov += time() - start_ov

            # Time post-processing separately
            start_post = time()
            loc_map_t = torch.from_numpy(ov_result[0])
            cls_mask_t = torch.from_numpy(ov_result[1])
            theta_offset_t = torch.from_numpy(ov_result[2])
            depth_offset_t = torch.from_numpy(ov_result[3])
            width_offset_t = torch.from_numpy(ov_result[4])
            perpoint_features = torch.from_numpy(ov_result[5])

            loc_map, cls_mask, theta_offset, height_offset, width_offset = \
                anchor_output_process(loc_map_t, cls_mask_t, theta_offset_t,
                                      depth_offset_t, width_offset_t, sigma=args.sigma)

            rect_gg = detect_2d_grasp(loc_map, cls_mask, theta_offset,
                                      height_offset, width_offset,
                                      ratio=args.ratio,
                                      anchor_k=args.anchor_k,
                                      anchor_w=args.anchor_w,
                                      anchor_z=args.anchor_z,
                                      mask_thre=args.heatmap_thres,
                                      center_num=args.center_num,
                                      grid_size=args.grid_size,
                                      grasp_nms=args.grid_size,
                                      reduce='max')

            if batch_idx >= 1:
                time_anchor_post += time() - start_post

            if rect_gg.size == 0:
                logging.warning(f'Frame {batch_idx}: no 2D grasps detected, skipping')
                continue

            start = time()

            start_ff = time()
            points_all = feature_fusion(points, perpoint_features, xyzs)
            ff_elapsed = time() - start_ff

            rect_ggs = [rect_gg]

            start_dp = time()
            pc_group, valid_local_centers = data_process(
                points_all, depth, rect_ggs, args.center_num,
                args.group_num, (args.input_w, args.input_h),
                min_points=32, is_training=False)
            dp_elapsed = time() - start_dp

            rect_gg = rect_ggs[0]
            points_all = points_all.squeeze()

            grasp_info = np.zeros((0, 3), dtype=np.float32)
            g_thetas = rect_gg.thetas[None]
            g_ws = rect_gg.widths[None]
            g_ds = rect_gg.depths[None]
            cur_info = np.vstack([g_thetas, g_ws, g_ds])
            grasp_info = np.vstack([grasp_info, cur_info.T])
            grasp_info_np = grasp_info.astype(np.float32)

            if batch_idx >= 1:
                time_data += time() - start
                time_feat_fusion += ff_elapsed
                time_get_group_pc += dp_elapsed

            # ── PointMultiGraspNet ─────────────────────────────────────
            # Workaround: pad to fixed batch=center_num to avoid OV GPU
            # dynamic-shape bug that corrupts outputs after ~20-30 calls
            # with varying batch dimensions.
            pc_np = pc_group.numpy()
            info_np = grasp_info_np
            real_batch = pc_np.shape[0]
            use_padding = ov_device == 'GPU' and real_batch < args.center_num and not args.no_gpu_padding
            if use_padding:
                pc_padded = np.zeros((args.center_num, pc_np.shape[1], pc_np.shape[2]), dtype=np.float32)
                pc_padded[:real_batch] = pc_np
                info_padded = np.zeros((args.center_num, info_np.shape[1]), dtype=np.float32)
                info_padded[:real_batch] = info_np
            else:
                pc_padded = pc_np
                info_padded = info_np

            # Time OV inference separately (for comparison with benchmark_app)
            start_ov = time()
            ov_local_result = local_compiled({
                "points": pc_padded,
                "info": info_padded
            })
            if batch_idx >= 1:
                time_local_ov += time() - start_ov

            # Time post-processing separately
            start_post = time()
            pred = torch.from_numpy(ov_local_result["pred"])[:real_batch]
            offset = torch.from_numpy(ov_local_result["offset"])[:real_batch]

            pred_grasp, pred_rect_gg = detect_6d_grasp_multi(
                rect_gg, pred, offset, valid_local_centers,
                (args.input_w, args.input_h), anchors, k=args.local_k)

            if batch_idx >= 1:
                time_local_post += time() - start_post

            start = time()
            pred_grasp_from_rect = pred_rect_gg.to_6d_grasp_group(depth=0.02)
            pred_gg, valid_mask = collision_detect_cpu(
                points_all, pred_grasp_from_rect, mode='graspnet')

            if batch_idx >= 1:
                time_colli += time() - start
            start = time()

            gg = GraspNetGraspGroup()
            for pred_g in pred_gg:
                g = GraspNetGrasp(pred_g.score, pred_g.width, pred_g.height,
                                  pred_g.depth, pred_g.rotation.reshape(9,),
                                  pred_g.translation, -1)
                gg.add(g)
            gg = gg.nms(0.03, 30 / 180 * np.pi)
            gg = gg.sort_by_score()

            if batch_idx >= 1:
                time_nms += time() - start

            if batch_idx >= 1:
                elapsed = time() - frame_start
                pbar.set_postfix(fps=f'{1.0/elapsed:.1f}', grasps_2d=rect_gg.size, grasps_nms=len(gg))

            save_dir = os.path.join(args.dump_dir, SCENE_LIST[batch_idx])
            save_dir = os.path.join(save_dir, camera)
            save_path = os.path.join(save_dir, str(batch_idx % 256).zfill(4) + '.npy')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gg.save_npy(save_path)

            if batch_idx >= 1:
                time_wall += time() - frame_start

    # Time stats
    if batch_idx >= 2:
        n = batch_idx - 1
        # OV inference times (comparable to benchmark_app)
        anchor_ov_avg = time_anchor_ov / n * 1000
        local_ov_avg = time_local_ov / n * 1000
        # Post-processing times
        anchor_post_avg = time_anchor_post / n * 1000
        local_post_avg = time_local_post / n * 1000
        # Other pipeline stages
        time_data_avg = time_data / n * 1000
        time_colli_avg = time_colli / n * 1000
        time_nms_avg = time_nms / n * 1000
        # Data preprocess sub-components
        feat_fusion_avg = time_feat_fusion / n * 1000
        get_group_pc_avg = time_get_group_pc / n * 1000
        grasp_info_avg = time_data_avg - feat_fusion_avg - get_group_pc_avg
        
        # Totals for comparison with old format
        time_2d_total = anchor_ov_avg + anchor_post_avg
        time_6d_total = local_ov_avg + local_post_avg
        total_ov = anchor_ov_avg + local_ov_avg
        total = time_2d_total + time_data_avg + time_6d_total + time_colli_avg + time_nms_avg
        wall_avg = time_wall / n * 1000
        
        logging.info(f'Time stats (OV {ov_device}, precision_hint={args.precision_hint}):')
        logging.info(f'Wall-clock per frame: {wall_avg:.3f} ms  ({1000.0/wall_avg:.1f} FPS)')
        logging.info(f'Component sum:        {total:.3f} ms  (gap: {wall_avg - total:.3f} ms = data load + padding + save + misc)')
        logging.info('')
        logging.info('=== OV Inference Only (for benchmark_app comparison) ===')
        logging.info(f'  AnchorNet OV:  {anchor_ov_avg:.3f} ms')
        logging.info(f'  LocalNet OV:   {local_ov_avg:.3f} ms  (batch={args.center_num})')
        logging.info(f'  Total OV:      {total_ov:.3f} ms')
        logging.info('')
        logging.info('=== Full Pipeline Breakdown ===')
        logging.info(f'  AnchorNet OV:     {anchor_ov_avg:.3f} ms')
        logging.info(f'  AnchorNet post:   {anchor_post_avg:.3f} ms  (convert + anchor_output_process + detect_2d)')
        logging.info(f'  Data preprocess:  {time_data_avg:.3f} ms  (feature_fusion + data_process)')
        logging.info(f'    feature_fusion:   {feat_fusion_avg:.3f} ms  (KNN + gather + max_pool)')
        logging.info(f'    data_process:     {get_group_pc_avg:.3f} ms  (center2dtopc + get_group_pc w/ FPS)')
        logging.info(f'    grasp_info:       {grasp_info_avg:.3f} ms')
        logging.info(f'  LocalNet OV:      {local_ov_avg:.3f} ms')
        logging.info(f'  LocalNet post:    {local_post_avg:.3f} ms  (detect_6d_grasp_multi)')
        logging.info(f'  Collision:        {time_colli_avg:.3f} ms')
        logging.info(f'  NMS:              {time_nms_avg:.3f} ms')
    elif batch_idx == 1:
        wall_avg = time_wall * 1000
        total = (time_anchor_ov + time_anchor_post + time_data + 
                 time_local_ov + time_local_post + time_colli + time_nms) * 1000
        logging.info(f'Wall-clock: {wall_avg:.3f} ms  Component sum: {total:.3f} ms')
    else:
        logging.info('Not enough frames for timing statistics')


def evaluate():
    ge = GraspNetEval(root=args.scene_path,
                      camera=camera,
                      split=(args.scene_l, args.scene_r))
    res, ap, colli = ge.eval_scene_lr(args.dump_dir,
                                      args.scene_l,
                                      args.scene_r,
                                      proc=1)
    result_path = os.path.join(os.path.dirname(args.dump_dir), 'eval_result.npy')
    np.save(result_path, res)
    aps = res.mean(0).mean(0).mean(0)
    logging.info(f'Scene: {args.scene_l} ~ {args.scene_r}')
    logging.info(f'colli == {colli}')
    logging.info(f'ap == {ap}')
    logging.info(f'ap0.8 == {aps[3]}')
    logging.info(f'ap0.4 == {aps[1]}')
    return ap, aps[3], aps[1], colli


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Set up logging: file + console
    output_dir = os.path.dirname(args.dump_dir)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'inference.log')

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler(),
        ])

    # Save command-line args alongside log
    params_path = os.path.join(output_dir, 'commandline_args.json')
    with open(params_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    logging.info(f'Running HGGD inference with OpenVINO {ov.__version__} on {args.ov_device} (precision_hint={args.precision_hint})')
    logging.info(f'Log: {log_path}')

    inference()
    logging.info('Running evaluation...')
    ap, ap08, ap04, colli = evaluate()

    logging.info(f'=== Results: OV {args.ov_device} (precision_hint={args.precision_hint}) ===')
    logging.info(f'AP:      {ap:.4f} ({ap*100:.2f}%)')
    logging.info(f'AP@0.8:  {ap08:.4f} ({ap08*100:.2f}%)')
    logging.info(f'AP@0.4:  {ap04:.4f} ({ap04*100:.2f}%)')
    logging.info(f'Collision rate: {colli:.4f} ({colli*100:.2f}%)')
