#!/usr/bin/env python
"""
xpu_infer.py -- CDPN Inference Class (Intel XPU)

End-to-end 6D pose estimation pipeline:
  preprocessing -> XPU forward pass -> PnP solving -> translation decoding

The inference class is decoupled from dataset paths -- it only needs:
  - Model config (YAML) + checkpoint weights
  - Object 3D model metadata (obj_info dict)
  - Camera intrinsics (optional, defaults to LINEMOD camera matrix, K)

Usage:
    # As a library (direct -- no dataset path needed):
    from xpu_infer import CdpnXpuInference
    infer = CdpnXpuInference(cfg_path, checkpoint_path, obj_info, xpu=0)
    results = infer(rgb_images, boxes, obj_names)

    # Standalone test (--dataset_dir only needed by the test harness):
    python xpu_infer.py --cfg tools/exps_cfg/config_rot_trans.yaml \\
        --load_model checkpoints/stage3.checkpoint --xpu 0 \\
        --dataset_dir dataset/lm_full
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
import numpy as np
import cv2
import torch

# make lib/ importable
_cur = os.path.dirname(os.path.abspath(__file__))
_lib = os.path.join(_cur, 'lib')
if _lib not in sys.path:
    sys.path.insert(0, _lib)

import ref
from utils.img import zoom_in, xyxy_to_xywh
from model import build_model
from config import config as Config


class CdpnXpuInference:
    """
    CDPN 6-DoF Pose Inference on Intel XPU.

    Decoupled from dataset paths -- only needs model weights + object metadata.

    Parameters
    ----------
    cfg_path : str
        Path to YAML config (e.g. 'tools/exps_cfg/config_rot_trans.yaml').
    checkpoint_path : str
        Path to .checkpoint file.
    obj_info : dict
        Object 3D model metadata, keyed by integer object ID.
        Each value: {'diameter': float, 'min_x': float, 'min_y': float, 'min_z': float}.
        Can be obtained via ``LM.load_lm_model_info('models/models_info.txt')``.
    xpu : int
        XPU device id (-1 for CPU).
    camera_matrix : np.ndarray or None
        3x3 camera intrinsics. Defaults to the LINEMOD camera if None.
    """

    def __init__(self, cfg_path, checkpoint_path, obj_info, xpu=0,
                 camera_matrix=None):

        # resolve device
        if xpu > -1:
            if not torch.xpu.is_available():
                raise RuntimeError(
                    'XPU device {} requested but torch.xpu.is_available() '
                    'returned False.  Check Intel GPU driver and PyTorch XPU '
                    'build.'.format(xpu))
            self.device = torch.device('xpu', xpu)
            print(f'Model loaded on XPU {xpu}')
        else:
            self.device = torch.device('cpu')
            print(f'Model loaded on CPU')
        self.xpu = xpu

        # build config (model architecture only, no data paths)
        self.cfg = self._build_cfg(cfg_path, checkpoint_path, xpu)

        # build model and move to device
        self.model, _ = build_model(self.cfg)
        self.model = self.model.to(self.device)
        self.model.eval()

        # object metadata
        self.obj_info = obj_info

        # camera intrinsics and network resolution constants
        self.camera_matrix = camera_matrix if camera_matrix is not None else ref.K.copy()
        self.cfg.dataset.camera_matrix = self.camera_matrix
        self.inp_res = self.cfg.dataiter.inp_res
        self.out_res = self.cfg.dataiter.out_res
        self.pad_ratio = self.cfg.augment.pad_ratio
        self.mask_threshold = self.cfg.test.mask_threshold
        self.pnp_method = self.cfg.test.pnp

    # --------------------- public API ---------------------

    def __call__(self, rgb_list, box_list, obj_list):
        """
        Run end-to-end inference on one or more images.

        Parameters
        ----------
        rgb_list : list[np.ndarray]
            BGR images, each (H, W, 3) uint8  (OpenCV convention).
        box_list : list[tuple/np.ndarray]
            Bounding boxes in **xywh** format: (x, y, w, h).
        obj_list : list[str]
            Object class names, e.g. ['ape', 'can'].

        Returns
        -------
        results : list[dict]  -- one dict per input, with keys:
            'pose_rot'       : (3, 4) pose  [R_pnp | T_pnp]
            'pose_trans'     : (3, 4) pose  [R_pnp | T_trans_head]
            'R'              : (3, 3) rotation from PnP
            'T_pnp'         : (3,)   translation from PnP
            'T_trans'        : (3,)   translation from translation head
            'pred_coor'      : (3, 64, 64) raw coordinate maps
            'pred_conf'      : (64, 64) confidence map
            'pred_trans_raw' : (3,) raw trans head output
            'num_corres'     : int, #correspondences sent to PnP
        """
        assert len(rgb_list) == len(box_list) == len(obj_list)
        num_samples = len(rgb_list)

        # preprocess
        inp_batch, crop_center_batch, crop_scale_batch, box_batch = \
            self._preprocess_batch(rgb_list, box_list)

        # XPU forward pass
        with torch.no_grad():
            model_input = inp_batch.to(self.device).float()
            pred_rot, pred_trans = self.model(model_input)
        if self.xpu > -1:
            torch.xpu.synchronize(self.device)
        pred_rot_np = pred_rot.cpu().numpy()
        pred_trans_np = pred_trans.cpu().numpy()

        # postprocess each sample
        results = []
        for i in range(num_samples):
            result = self._postprocess_single(
                pred_rot_np[i], pred_trans_np[i],
                obj_list[i], crop_center_batch[i],
                crop_scale_batch[i], box_batch[i])
            results.append(result)

        return results

    # --------------------- internal helpers ---------------------

    @staticmethod
    def _build_cfg(cfg_path, checkpoint_path, xpu):
        """Build config for model architecture only (no dataset paths)."""
        sys.argv = ['xpu_infer.py',
                     '--cfg', cfg_path,
                     '--test',
                     '--load_model', checkpoint_path,
                     '--gpu', str(xpu)]
        cfg = Config().parse()
        cfg.pytorch.load_model = checkpoint_path
        return cfg

    @staticmethod
    def _xywh_to_center_scale(box, pad_ratio, max_crop_size):
        """Convert an xywh bounding box to (crop_center, crop_scale).

        Parameters
        ----------
        box : array-like [x, y, w, h]
            Bounding box -- top-left origin, pixel coordinates.
        pad_ratio : float
            Padding multiplier; 1.5 adds 50% margin around the object.
        max_crop_size : float
            Maximum allowed crop size (clamped to max(image_w, image_h)).

        Returns
        -------
        crop_center : (2,) ndarray [col, row]
            Crop centre in image pixels: (x + w/2, y + h/2).
        crop_scale : float
            Square crop side length = min(max(w, h) * pad_ratio, max_crop_size).
        """
        x, y, w, h = box
        crop_center = np.array((x + 0.5 * w, y + 0.5 * h))
        crop_scale = max(w, h) * pad_ratio
        crop_scale = min(crop_scale, max_crop_size)
        return crop_center, crop_scale

    def _preprocess_single(self, rgb, box):
        """
        Crop, resize, and normalise one image for network input.

        Parameters
        ----------
        rgb : (H, W, 3) uint8
            BGR image (OpenCV convention).
        box : array-like [x, y, w, h]
            Bounding box in xywh pixel coordinates.

        Returns
        -------
        inp : (3, 256, 256) float32
            Normalised CHW image patch, pixel values in [0, 1].
        crop_center_out : (2,) [col, row]
            Actual crop centre in image pixels after zoom_in's
            integer truncation.
        crop_scale_out : float
            Square crop side length in pixels, integer-truncated by zoom_in.
        box : (4,) float64
            Input bounding box converted to a float64 array [x, y, w, h].
        """
        box = np.asarray(box, dtype=np.float64)
        crop_center, crop_scale = self._xywh_to_center_scale(
            box, self.pad_ratio, max_crop_size=max(ref.im_w, ref.im_h))
        rgb_crop, center_row, center_col, crop_scale_out = zoom_in(
            rgb, crop_center, crop_scale, self.inp_res)
        inp = rgb_crop.transpose(2, 0, 1).astype(np.float32) / 255.0
        crop_center_out = np.array([center_col, center_row])
        return inp, crop_center_out, crop_scale_out, box

    def _preprocess_batch(self, rgb_list, box_list):
        """
        Preprocess a batch of images into a stacked PyTorch tensor.

        Parameters
        ----------
        rgb_list : list of (H, W, 3) uint8
            BGR images (OpenCV convention).
        box_list : list of array-like [x, y, w, h]
            Bounding boxes in xywh pixel coordinates.

        Returns
        -------
        inp_tensor : torch.Tensor [N, 3, 256, 256] float32
            Stacked normalised CHW image patches.
        crop_centers : list of (2,) ndarray
            Crop centres [col, row] in image pixels (one per sample).
        crop_scales : list of float
            Crop side lengths in pixels (one per sample).
        boxes : list of (4,) float64 ndarray
            Bounding boxes in xywh format (one per sample).
        """
        inps, crop_centers, crop_scales, boxes = [], [], [], []
        for rgb, box in zip(rgb_list, box_list):
            inp, crop_center, crop_scale, box_arr = self._preprocess_single(rgb, box)
            inps.append(inp)
            crop_centers.append(crop_center)
            crop_scales.append(crop_scale)
            boxes.append(box_arr)
        inp_tensor = torch.from_numpy(np.stack(inps, axis=0))
        return inp_tensor, crop_centers, crop_scales, boxes

    def _postprocess_single(self, pred_rot, pred_trans, obj_name,
                            crop_center, crop_scale, box):
        """
        Post-process one sample: coordinate-map to PnP + trans-head decode.

        Parameters
        ----------
        pred_rot : (4, 64, 64)
            RotHead output. Channels: [x, y, z, confidence].
        pred_trans : (3,)
            TransHead output: [ratio_delta_cx, ratio_delta_cy, ratio_depth].
        obj_name : str
            Object class name, e.g. 'ape'.
        crop_center : (2,) [col, row]
            Crop centre in image pixels, as returned by zoom_in
            after integer truncation.
        crop_scale : float
            Square crop side length in pixels (integer-truncated by zoom_in).
        box : (4,) [x, y, w, h]
            Bounding box in xywh pixel coordinates.
        """
        obj_id = ref.obj2idx(obj_name)
        result = {
            'pred_coor': pred_rot[:3].copy(),
            'pred_conf': pred_rot[3].copy(),
            'pred_trans_raw': pred_trans.copy(),
        }

        # build 2D-3D correspondences from coordinate map
        pred_coor = pred_rot[:3].transpose(1, 2, 0).copy()   # (64,64,3)
        pred_coor[:, :, 0] *= abs(self.obj_info[obj_id]['min_x'])
        pred_coor[:, :, 1] *= abs(self.obj_info[obj_id]['min_y'])
        pred_coor[:, :, 2] *= abs(self.obj_info[obj_id]['min_z'])
        pred_coor = pred_coor.tolist()

        pred_conf = pred_rot[3].copy()
        pred_conf = (pred_conf - pred_conf.min()) / (pred_conf.max() - pred_conf.min())
        pred_conf = pred_conf.tolist()

        select_pts_2d = []
        select_pts_3d = []
        crop_col = int(crop_center[0])
        crop_row = int(crop_center[1])
        left_edge = crop_col - crop_scale / 2.0
        top_edge = crop_row - crop_scale / 2.0
        px_per_col = crop_scale * 1.0 / self.out_res
        px_per_row = crop_scale * 1.0 / self.out_res

        min_x = 0.001 * abs(self.obj_info[obj_id]['min_x'])
        min_y = 0.001 * abs(self.obj_info[obj_id]['min_y'])
        min_z = 0.001 * abs(self.obj_info[obj_id]['min_z'])

        for row in range(self.out_res):
            for col in range(self.out_res):
                if pred_conf[row][col] < self.mask_threshold:
                    continue
                if (abs(pred_coor[row][col][0]) < min_x and
                    abs(pred_coor[row][col][1]) < min_y and
                    abs(pred_coor[row][col][2]) < min_z):
                    continue
                select_pts_2d.append([left_edge + col * px_per_col,
                                      top_edge + row * px_per_row])
                select_pts_3d.append(pred_coor[row][col])

        model_points = np.asarray(select_pts_3d, dtype=np.float32)
        image_points = np.asarray(select_pts_2d, dtype=np.float32)
        result['num_corres'] = len(model_points)

        # PnP solve
        R_pnp = np.eye(3)
        T_pnp = np.zeros(3)
        pnp_success = False
        try:
            dist_coeffs = np.zeros((4, 1))
            if self.pnp_method == 'ransac':
                _, R_vec, T_vec, _ = cv2.solvePnPRansac(
                    model_points, image_points,
                    self.camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
            elif self.pnp_method == 'iterPnP':
                _, R_vec, T_vec = cv2.solvePnP(
                    model_points, image_points,
                    self.camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            else:
                raise NotImplementedError(
                    "PnP method '{}' not supported".format(self.pnp_method))
            R_pnp = cv2.Rodrigues(R_vec, jacobian=0)[0]
            T_pnp = np.asarray(T_vec).reshape(3)
            pnp_success = True
        except Exception as e:
            print('[CdpnXpuInference] PnP failed for {}: {}'.format(obj_name, e))

        # translation head decode
        ratio_delta_center = pred_trans[:2]
        ratio_depth = pred_trans[2]
        pred_depth = ratio_depth * (self.out_res / crop_scale)
        pred_center = ratio_delta_center * box[2:] + crop_center
        pred_x = ((pred_center[0] - self.camera_matrix[0, 2]) * pred_depth
                  / self.camera_matrix[0, 0])
        pred_y = ((pred_center[1] - self.camera_matrix[1, 2]) * pred_depth
                  / self.camera_matrix[1, 1])
        T_trans = np.array([pred_x, pred_y, pred_depth])

        # compose pose matrices
        pose_rot = np.concatenate(
            [R_pnp, T_pnp.reshape(3, 1)], axis=1)
        pose_trans = np.concatenate(
            [R_pnp, T_trans.reshape(3, 1)], axis=1)
        result.update({
            'pose_rot': pose_rot,
            'pose_trans': pose_trans,
            'R': R_pnp,
            'T_pnp': T_pnp,
            'T_trans': T_trans,
            'pnp_success': pnp_success,
        })
        return result


# Standalone test
def _collect_test_samples(dataset_dir, obj_names, max_samples=0):
    """
    Collect test sample metadata from a CDPN-format dataset directory.

    Images are not loaded here -- only paths and lightweight
    annotations are stored.  Use _load_batch() to read pixels.

    Parameters
    ----------
    max_samples : int
        Max samples per object (0 = all).

    Returns list of dicts: {rgb_path, box_xywh, pose_gt, obj_name}
    """
    from glob import glob
    samples = []
    for obj in obj_names:
        test_dir = os.path.join(dataset_dir, 'real_test', obj)
        if not os.path.isdir(test_dir):
            print('WARNING: test dir not found: {}'.format(test_dir))
            continue
        rgb_paths = sorted(glob(os.path.join(test_dir, '*-color.png')))
        count = 0
        for rgb_pth in rgb_paths:
            if 0 < max_samples <= count:
                break
            box_xyxy = np.loadtxt(
                rgb_pth.replace('-color.png', '-box_fasterrcnn.txt'))
            samples.append({
                'box_xywh': np.array(xyxy_to_xywh(box_xyxy)),
                'pose_gt': np.loadtxt(
                    rgb_pth.replace('-color.png', '-pose.txt')),
                'obj_name': obj,
                'rgb_path': rgb_pth,
            })
            count += 1
    return samples


def _load_batch(samples):
    """Read images for a batch of samples."""
    return [cv2.imread(sample['rgb_path']) for sample in samples]


def main():
    """
    Batch-driven test harness for CdpnXpuInference.

    Loads test images from --dataset_dir, batches them by --batch_size,
    and feeds each batch to the inference class.
    """
    from datasets.lm import LM
    from utils.eval import calc_rt_dist_m

    parser = argparse.ArgumentParser(description='CDPN XPU Inference Test')
    parser.add_argument('--cfg', required=True, type=str)
    parser.add_argument('--load_model', required=True, type=str)
    parser.add_argument('--xpu', type=int, default=0,
                        help='XPU device id (-1 for CPU)')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Dataset dir with models/ and real_test/')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--obj_name', type=str, default='all',
                        help='Object name(s), comma-separated (default: all)')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Max samples per object (0 = all)')
    args = parser.parse_args()

    print('=' * 70)
    print('CdpnXpuInference -- Batched Test Harness')
    print('=' * 70)

    # load obj_info from dataset
    info_path = os.path.join(args.dataset_dir, 'models', 'models_info.txt')
    obj_info = LM.load_lm_model_info(info_path)

    # initialise inference class
    infer = CdpnXpuInference(
        cfg_path=args.cfg,
        checkpoint_path=args.load_model,
        obj_info=obj_info,
        xpu=args.xpu,
    )

    # determine which objects to test
    if args.obj_name and args.obj_name.lower() != 'all':
        obj_names = [o.strip() for o in args.obj_name.split(',')]
    else:
        obj_names = list(ref.lm_obj)
    print('Objects: {}'.format(obj_names))
    print('Batch size: {}'.format(args.batch_size))
    print('Device: {}'.format(infer.device))

    # collect test sample metadata
    print('\nCollecting test samples from: {}'.format(args.dataset_dir))
    samples = _collect_test_samples(
        args.dataset_dir, obj_names, max_samples=args.max_samples)
    num_samples = len(samples)
    print('Total samples: {}'.format(num_samples))

    # run inference in batches
    all_results = []
    batch_size = args.batch_size

    for start in range(0, num_samples, batch_size):
        batch = samples[start:start + batch_size]
        rgb_list = _load_batch(batch)
        box_list = [sample['box_xywh'] for sample in batch]
        obj_list = [sample['obj_name'] for sample in batch]

        results = infer(rgb_list, box_list, obj_list)

        for sample, result in zip(batch, results):
            result['pose_gt'] = sample['pose_gt']
            result['obj_name'] = sample['obj_name']
            result['rgb_path'] = sample['rgb_path']
            all_results.append(result)

        done = min(start + batch_size, num_samples)
        if done % (batch_size * 50) == 0 or done == num_samples:
            print('  [{}/{}]'.format(done, num_samples))

    # load PLY models for ADD / Proj. 2D metrics
    from utils.io import load_ply_vtx
    obj_vtx = {}
    for obj in obj_names:
        ply_path = os.path.join(
            args.dataset_dir, 'models', obj, '{}.ply'.format(obj))
        if os.path.isfile(ply_path):
            obj_vtx[obj] = load_ply_vtx(ply_path)
    camera_matrix = infer.camera_matrix
    sym_objs = {'eggbox', 'glue'}

    # compute per-object metrics
    obj_metrics = {}
    for result in all_results:
        obj = result['obj_name']
        obj_metrics.setdefault(obj, [])
        pose_est = result['pose_trans']
        pose_gt = result['pose_gt']
        R_est, T_est = pose_est[:, :3], pose_est[:, 3]
        R_gt, T_gt = pose_gt[:, :3], pose_gt[:, 3]

        # 5°5cm
        r_err, t_err = calc_rt_dist_m(pose_est, pose_gt)

        # ADD / ADI
        add_dist = np.inf
        if obj in obj_vtx:
            vtx = obj_vtx[obj]
            vtx_est = (R_est @ vtx.T + T_est.reshape(3, 1)).T
            vtx_gt = (R_gt @ vtx.T + T_gt.reshape(3, 1)).T
            if obj in sym_objs:
                # ADI: closest-point distance
                from scipy.spatial import cKDTree
                tree = cKDTree(vtx_gt)
                dists, _ = tree.query(vtx_est, k=1)
                add_dist = np.mean(dists)
            else:
                add_dist = np.mean(np.linalg.norm(vtx_est - vtx_gt, axis=1))

        # Proj. 2D
        proj2d_dist = np.inf
        if obj in obj_vtx:
            vtx = obj_vtx[obj]
            proj_est = camera_matrix @ (R_est @ vtx.T + T_est.reshape(3, 1))
            proj_est = proj_est[:2] / proj_est[2:3]
            proj_gt = camera_matrix @ (R_gt @ vtx.T + T_gt.reshape(3, 1))
            proj_gt = proj_gt[:2] / proj_gt[2:3]
            proj2d_dist = np.mean(np.linalg.norm(
                proj_est - proj_gt, axis=0))

        obj_metrics[obj].append({
            'r_err': r_err, 't_err': t_err,
            'add': add_dist, 'proj2d': proj2d_dist,
            'pnp_ok': result['pnp_success'],
        })

    # print summary table
    # Get diameters for ADD threshold
    obj_diam = {}
    for obj in obj_names:
        oid = ref.obj2idx(obj)
        if oid in obj_info:
            obj_diam[obj] = obj_info[oid]['diameter']

    print('\n' + '=' * 78)
    print('Results  (batch_size={}, {} samples)'.format(batch_size, num_samples))
    print('-' * 78)
    print('\n{:<14s} {:>6s} {:>6s} {:>8s} {:>8s} {:>10s} {:>8s} {:>8s}'.format(
        'Object', 'Count', 'PnP%', '5°5cm', 'ADD', 'Proj. 2D',
        'MedR(°)', 'MedT(m)'))
    print('-' * 78)

    sum_5d5cm, sum_add, sum_proj = 0.0, 0.0, 0.0
    n_obj = 0
    for obj in obj_names:
        if obj not in obj_metrics:
            continue
        metrics = obj_metrics[obj]
        count = len(metrics)
        pnp_pct = 100.0 * sum(1 for m in metrics if m['pnp_ok']) / count
        # 5°5cm
        acc_5_5 = 100.0 * sum(
            1 for m in metrics if m['r_err'] < 5 and m['t_err'] < 0.05) / count
        # ADD @0.10d
        diam = obj_diam.get(obj, 1.0)
        thresh_add = 0.10 * diam
        acc_add = 100.0 * sum(
            1 for m in metrics if m['add'] < thresh_add) / count
        # Proj. 2D @5px
        acc_proj = 100.0 * sum(
            1 for m in metrics if m['proj2d'] < 5.0) / count
        # Medians
        med_r = np.median([m['r_err'] for m in metrics])
        med_t = np.median([m['t_err'] for m in metrics])

        print('{:<14s} {:>6d} {:>5.1f}% {:>7.2f}% {:>7.2f}% {:>9.2f}% {:>7.2f} {:>8.4f}'.format(
            obj, count, pnp_pct, acc_5_5, acc_add, acc_proj, med_r, med_t))
        sum_5d5cm += acc_5_5
        sum_add += acc_add
        sum_proj += acc_proj
        n_obj += 1

    if n_obj > 0:
        total_count = sum(
            len(obj_metrics[o]) for o in obj_names if o in obj_metrics)
        all_metrics = [m for o in obj_names if o in obj_metrics
                       for m in obj_metrics[o]]
        pnp_pct_all = 100.0 * sum(1 for m in all_metrics if m['pnp_ok']) / len(all_metrics)
        med_r_all = np.median([m['r_err'] for m in all_metrics])
        med_t_all = np.median([m['t_err'] for m in all_metrics])
        print('-' * 78)
        print('{:<14s} {:>6d} {:>5.1f}% {:>7.2f}% {:>7.2f}% {:>9.2f}% {:>7.2f} {:>8.4f}'.format(
            '**Average**', total_count, pnp_pct_all,
            sum_5d5cm / n_obj, sum_add / n_obj, sum_proj / n_obj,
            med_r_all, med_t_all))

    print('=' * 70)


if __name__ == '__main__':
    main()
