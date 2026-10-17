#!/usr/bin/env python
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Evaluate HGGD predictions on the GraspNet benchmark (Collision-Free AP).

Computes Collision-Free AP: the standard GraspNet grasp pipeline (NMS,
object assignment, gripper-box collision + empty-grasp test, AP
aggregation) with a geometric success test — a grasp is a success iff
it does not collide with any scene point and is not "empty" (holds at
least 10 points). Fully standalone: numpy + open3d + transforms3d +
grasp-nms. Requires only the GraspNet dataset root with ``scenes/`` and
``models/%03d/nontextured.ply``.

NOTE: CF-AP has no friction-coefficient semantics and is NOT numerically
comparable to the AP / AP@0.8 / AP@0.4 reported in the HGGD paper.

Usage:
  python evaluate.py --scene-path /path/to/graspnet \
    --pred-dir output/scene_100/pred --scene-l 100 --scene-r 101
"""

import argparse
import logging
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

# ── Collision-Free evaluation ─────────────────────────────────────────────
# The stage 0 (grasp NMS + object assignment + top-K selection) and stage 1
# (gripper-box collision / empty test) below are verbatim ports of the
# generic-numpy parts of graspnetAPI.utils.eval_utils.{eval_grasp,
# collision_detection}; only the quality-scoring stage is omitted.

import open3d as o3d
from grasp_nms import nms_grasp
from transforms3d.euler import euler2mat, quat2euler

CF_GRASP_ARRAY_LEN = 17
CF_HEIGHT = 0.02            # gripper height (collision_detection constant)
CF_DEPTH_BASE = 0.02        # palm/base depth (collision_detection constant)
CF_FINGER_WIDTH = 0.01      # finger width (collision_detection constant)
CF_OUTLIER = 0.05           # workspace crop margin (collision_detection)
CF_EMPTY_THRESH = 10        # min inner points, else "empty" grasp
CF_VOXEL_SIZE = 0.008       # object model voxel size (eval_scene)
CF_TOP_PER_OBJECT = 10      # per-object cap before global top-K (eval_grasp)
CF_NMS_TRANS = 0.03         # NMS translation threshold (eval_grasp)
CF_NMS_ROT = 30.0 / 180.0 * np.pi  # NMS rotation threshold (eval_grasp)


def cf_transform_points(points, trans):
    '''graspnetAPI.utils.utils.transform_points (model->camera).'''
    ones = np.ones([points.shape[0], 1], dtype=points.dtype)
    points_ = np.concatenate([points, ones], axis=-1)
    points_ = np.matmul(trans, points_.T).T
    return points_[:, :3]


def cf_create_table_points(lx=1.0, ly=1.0, lz=0.05,
                           dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008):
    '''graspnetAPI.utils.eval_utils.create_table_points (eval_scene args).'''
    xmap = np.linspace(0, lx, int(lx / grid_size))
    ymap = np.linspace(0, ly, int(ly / grid_size))
    zmap = np.linspace(0, lz, int(lz / grid_size))
    xmap, ymap, zmap = np.meshgrid(xmap, ymap, zmap, indexing='xy')
    xmap += dx
    ymap += dy
    zmap += dz
    points = np.stack([xmap, ymap, zmap], axis=-1)
    return points.reshape([-1, 3])


def cf_voxel_sample_points(points, voxel_size=CF_VOXEL_SIZE):
    '''graspnetAPI.utils.eval_utils.voxel_sample_points.'''
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud = cloud.voxel_down_sample(voxel_size)
    return np.array(cloud.points)


def cf_pose_from_xml(xml_path):
    '''Parse a GraspNet annotation XML into (obj_ids, 4x4 pose mats).

    Mirrors graspnetAPI.utils.xmlhandler.xmlReader.getposevectorlist +
    graspnetAPI.utils.eval_utils.parse_posevector (including the
    rad->deg->rad round trip and float32 matrices).
    '''
    top = ET.parse(xml_path).getroot()
    obj_ids = []
    mats = []
    for obj in top:
        obj_id = int(obj[0].text)
        pos = [float(v) for v in obj[3].text.split()]
        quat = [float(v) for v in obj[4].text.split()]
        alpha, beta, gamma = quat2euler(quat)
        alpha = alpha * (180.0 / np.pi) / 180.0 * np.pi
        beta = beta * (180.0 / np.pi) / 180.0 * np.pi
        gamma = gamma * (180.0 / np.pi) / 180.0 * np.pi
        mat = np.zeros([4, 4], dtype=np.float32)
        mat[:3, :3] = euler2mat(alpha, beta, gamma)
        mat[:3, 3] = pos
        mat[3, 3] = 1
        obj_ids.append(obj_id)
        mats.append(mat)
    return obj_ids, mats


def cf_closest_point_indices(a, b):
    '''graspnetAPI.utils.eval_utils.compute_closest_points.'''
    dists = np.linalg.norm(a[:, np.newaxis, :] - b[np.newaxis, :, :], axis=-1)
    return np.argmin(dists, axis=-1)


def cf_stage0(pred, model_trans_list, max_width=0.1):
    '''Stage 0: width clipping, NMS, object assignment, top-K selection.

    Verbatim port of the generic part of
    graspnetAPI.utils.eval_utils.eval_grasp (grasp_nms + nearest-point
    object assignment + per-object top-10 + global top-50 by score).
    Returns the per-object grasp arrays, same layout as eval_grasp's
    grasp_list.
    '''
    num_models = len(model_trans_list)
    if len(pred) == 0:
        return [np.zeros((0, CF_GRASP_ARRAY_LEN), dtype=np.float64)
                for _ in range(num_models)]

    grasps = pred.copy()
    # clip width to [0, max_width] (as in graspnetAPI eval_scene)
    min_width_mask = (grasps[:, 1] < 0)
    max_width_mask = (grasps[:, 1] > max_width)
    grasps[min_width_mask, 1] = 0
    grasps[max_width_mask, 1] = max_width

    nmsed = nms_grasp(grasps, CF_NMS_TRANS, CF_NMS_ROT)

    # assign each grasp to the nearest object point
    seg_mask = []
    for i in range(num_models):
        seg_mask.extend([i] * len(model_trans_list[i]))
    seg_mask = np.array(seg_mask, dtype=np.int32)
    scene = np.concatenate(model_trans_list, axis=0)
    indices = cf_closest_point_indices(nmsed[:, 13:16], scene)
    model_to_grasp = seg_mask[indices]

    pre_grasp_list = []
    for i in range(num_models):
        grasp_i = nmsed[model_to_grasp == i]
        # GraspGroup.sort_by_score: argsort ascending, then reverse
        grasp_i = grasp_i[np.argsort(grasp_i[:, 0])[::-1]]
        pre_grasp_list.append(grasp_i[:CF_TOP_PER_OBJECT])

    if all(len(g) == 0 for g in pre_grasp_list):
        return [np.zeros((0, CF_GRASP_ARRAY_LEN), dtype=np.float64)
                for _ in range(num_models)]

    all_grasp_list = np.vstack(pre_grasp_list)
    remain_mask = np.argsort(all_grasp_list[:, 0])[::-1]
    min_score = all_grasp_list[remain_mask[min(49, len(remain_mask) - 1)], 0]
    return [g[g[:, 0] >= min_score] for g in pre_grasp_list]


def cf_stage1(grasp_list, model_trans_list, scene_points):
    '''Stage 1: gripper-box collision + empty-grasp mask.

    Verbatim port of graspnetAPI.utils.eval_utils.collision_detection
    (left/right finger + bottom palm boxes, workspace crop, empty
    threshold), minus the quality-scoring stage.
    '''
    collision_mask_list = []
    for i in range(len(model_trans_list)):
        grasps = grasp_list[i]
        if len(grasps) == 0:
            collision_mask_list.append(np.zeros(0, dtype=bool))
            continue

        model = model_trans_list[i]
        grasp_points = grasps[:, 13:16]
        grasp_poses = grasps[:, 4:13].reshape([-1, 3, 3])
        grasp_depths = grasps[:, 3]
        grasp_widths = grasps[:, 1]

        # crop scene, remove outlier
        xmin, xmax = model[:, 0].min(), model[:, 0].max()
        ymin, ymax = model[:, 1].min(), model[:, 1].max()
        zmin, zmax = model[:, 2].min(), model[:, 2].max()
        xlim = ((scene_points[:, 0] > xmin - CF_OUTLIER) &
                (scene_points[:, 0] < xmax + CF_OUTLIER))
        ylim = ((scene_points[:, 1] > ymin - CF_OUTLIER) &
                (scene_points[:, 1] < ymax + CF_OUTLIER))
        zlim = ((scene_points[:, 2] > zmin - CF_OUTLIER) &
                (scene_points[:, 2] < zmax + CF_OUTLIER))
        workspace = scene_points[xlim & ylim & zlim]

        # transform scene to gripper frame
        target = (workspace[np.newaxis, :, :] -
                  grasp_points[:, np.newaxis, :])
        target = np.matmul(target, grasp_poses)

        mask1 = ((target[:, :, 2] > -CF_HEIGHT / 2) &
                 (target[:, :, 2] < CF_HEIGHT / 2))
        mask2 = ((target[:, :, 0] > -CF_DEPTH_BASE) &
                 (target[:, :, 0] < grasp_depths[:, np.newaxis]))
        mask3 = (target[:, :, 1] > -(grasp_widths[:, np.newaxis] / 2
                                     + CF_FINGER_WIDTH))
        mask4 = (target[:, :, 1] < -grasp_widths[:, np.newaxis] / 2)
        mask5 = (target[:, :, 1] < (grasp_widths[:, np.newaxis] / 2
                                    + CF_FINGER_WIDTH))
        mask6 = (target[:, :, 1] > grasp_widths[:, np.newaxis] / 2)
        mask7 = ((target[:, :, 0] > -(CF_DEPTH_BASE + CF_FINGER_WIDTH)) &
                 (target[:, :, 0] < -CF_DEPTH_BASE))

        left_mask = (mask1 & mask2 & mask3 & mask4)
        right_mask = (mask1 & mask2 & mask5 & mask6)
        bottom_mask = (mask1 & mask3 & mask5 & mask7)
        inner_mask = (mask1 & mask2 & (~mask4) & (~mask6))
        collision_mask = np.any((left_mask | right_mask | bottom_mask),
                                axis=-1)
        empty_mask = (np.sum(inner_mask, axis=-1) < CF_EMPTY_THRESH)
        collision_mask = (collision_mask | empty_mask)
        collision_mask_list.append(collision_mask)
    return collision_mask_list


def cf_eval_frame(pred, model_trans_list, camera_pose, align_mat, table,
                  top_k, max_width):
    '''One frame: stage 0 + 1 + per-k CF accuracy (success := collision-free
    and non-empty) + collision mask, in grasp-confidence order.'''
    table_trans = cf_transform_points(
        table, np.linalg.inv(np.matmul(align_mat, camera_pose)))
    grasp_list = cf_stage0(pred, model_trans_list, max_width)
    scene = np.concatenate(model_trans_list, axis=0)
    scene = np.concatenate([scene, table_trans])
    collision_list = cf_stage1(grasp_list, model_trans_list, scene)

    if all(len(g) == 0 for g in grasp_list):
        return np.zeros(top_k), np.zeros(0, dtype=bool)

    all_grasps = np.concatenate(grasp_list)
    all_coll = np.concatenate(collision_list)
    # sort in scene level (highest score first)
    indices = np.argsort(-all_grasps[:, 0])
    success = (~all_coll)[indices]

    acc = np.zeros(top_k)
    for k in range(top_k):
        n = min(k + 1, len(success))
        acc[k] = success[:n].sum() / (k + 1)
    return acc, all_coll[indices]


def cf_eval_scene(scene_id, scene_path, pred_dir, camera, top_k, max_width):
    '''One scene: per-frame CF accuracy tensor (256, top_k) + collision
    counts. Object list/models come from frame 0, poses per frame — same
    convention as graspnetAPI GraspNetEval.'''
    scene_name = 'scene_' + str(scene_id).zfill(4)
    scene_dir = os.path.join(scene_path, 'scenes', scene_name, camera)
    ann_dir = os.path.join(scene_dir, 'annotations')

    obj_ids, _ = cf_pose_from_xml(os.path.join(ann_dir, '0000.xml'))
    model_sampled_list = []
    for obj_idx in obj_ids:
        pcd = o3d.io.read_point_cloud(
            os.path.join(scene_path, 'models', '%03d' % obj_idx,
                         'nontextured.ply'))
        model_sampled_list.append(cf_voxel_sample_points(
            np.array(pcd.points)))

    camera_poses = np.load(os.path.join(scene_dir, 'camera_poses.npy'))
    align_mat = np.load(os.path.join(scene_dir, 'cam0_wrt_table.npy'))
    table = cf_create_table_points()

    scene_acc = np.zeros((256, top_k))
    colli_sum = 0
    total = 0
    for ann_id in range(256):
        _, pose_list = cf_pose_from_xml(os.path.join(ann_dir,
                                                     '%04d.xml' % ann_id))
        model_trans_list = [cf_transform_points(m, p)
                            for m, p in zip(model_sampled_list, pose_list)]
        pred = np.load(os.path.join(pred_dir, scene_name, camera,
                                    '%04d.npy' % ann_id))
        acc, coll = cf_eval_frame(pred, model_trans_list,
                                  camera_poses[ann_id], align_mat, table,
                                  top_k, max_width)
        scene_acc[ann_id] = acc
        colli_sum += int(coll.sum())
        total += len(coll)
        if ann_id % 32 == 0:
            print('\rCF scene %04d: frame %d/256  mean acc %.3f' %
                  (scene_id, ann_id, acc.mean()), end='', flush=True)
    print()
    return scene_acc, colli_sum, total


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate HGGD predictions: Collision-Free AP.')
    parser.add_argument('--scene-path',
                        required=True,
                        help='GraspNet dataset root (contains scenes/ and models/)')
    parser.add_argument('--pred-dir',
                        required=True,
                        help='Prediction dump dir with scene_XXXX/<camera>/NNNN.npy '
                             '(e.g. output/scene_100/pred from infer.py)')
    parser.add_argument('--scene-l', type=int, default=100,
                        help='first scene id (inclusive)')
    parser.add_argument('--scene-r', type=int, default=101,
                        help='last scene id (exclusive)')
    parser.add_argument('--camera', default='realsense',
                        help='camera folder name (default: realsense)')
    parser.add_argument('--top-k', type=int, default=50,
                        help='top-K grasps to evaluate per frame (default: 50)')
    parser.add_argument('--max-width', type=float, default=0.1,
                        help='max gripper width in evaluation (default: 0.1)')
    parser.add_argument('--proc', type=int, default=1,
                        help='number of worker processes (default: 1)')
    parser.add_argument('--result-path', default=None,
                        help='where to save eval_result_cf.npy '
                             '(default: next to the prediction directory)')
    return parser.parse_args()


def main():
    args = parse_args()
    scene_ids = list(range(args.scene_l, args.scene_r))
    if not scene_ids:
        raise ValueError(f'Empty scene range: [{args.scene_l}, {args.scene_r})')

    worker_args = [(sid, args.scene_path, args.pred_dir, args.camera,
                    args.top_k, args.max_width) for sid in scene_ids]
    if args.proc > 1:
        from multiprocessing import Pool
        with Pool(processes=args.proc) as p:
            results = p.map(cf_eval_scene, worker_args)
    else:
        results = [cf_eval_scene(*w) for w in worker_args]

    res = np.stack([r[0] for r in results])
    colli_sum = sum(r[1] for r in results)
    total = sum(r[2] for r in results)
    colli_rate = colli_sum / total if total else 0.0

    result_path = args.result_path or os.path.join(
        os.path.dirname(os.path.abspath(args.pred_dir)),
        'eval_result_cf.npy')
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    np.save(result_path, res)

    logging.info(f'Scene: {args.scene_l} ~ {args.scene_r}')
    logging.info(f'CF-AP == {res.mean()}')
    logging.info(f'colli == {colli_rate}')
    logging.info(f'Saved: {result_path}')
    logging.info('=== Results ===')
    logging.info(f'CF-AP:          {res.mean():.4f} ({res.mean()*100:.2f}%)')
    logging.info(f'Collision rate: {colli_rate:.4f} ({colli_rate*100:.2f}%)')


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%H:%M:%S')
    sys.exit(main())
