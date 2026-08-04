import torch
import numpy as np
import os, sys
from utils.utils import AverageMeter
from utils.eval import calc_all_errs, Evaluation
from utils.img import im_norm_255 
import cv2
import ref
from progress.bar import Bar
import os
import utils.fancy_logger as logger
from utils.tictoc import tic, toc
from builtins import input
from utils.fs import mkdir_p

from scipy.linalg import logm
import numpy.linalg as LA
import time
import matplotlib.pyplot as plt
from numba import jit, njit

def test(epoch, cfg, data_loader, model, obj_vtx, obj_info, criterions):

    model.eval()
    Eval = Evaluation(cfg.dataset, obj_info, obj_vtx)
    if 'trans' in cfg.pytorch.task.lower():
        Eval_trans = Evaluation(cfg.dataset, obj_info, obj_vtx)

    if not cfg.test.ignore_cache_file:
        est_cache_file = cfg.test.cache_file
        # gt_cache_file = cfg.test.cache_file.replace('pose_est', 'pose_gt')
        gt_cache_file = cfg.test.cache_file.replace('_est', '_gt')
        if os.path.exists(est_cache_file) and os.path.exists(gt_cache_file):
            Eval.pose_est_all = np.load(est_cache_file, allow_pickle=True).tolist()
            Eval.pose_gt_all = np.load(gt_cache_file, allow_pickle=True).tolist()
            fig_save_path = os.path.join(cfg.pytorch.save_path, str(epoch))
            mkdir_p(fig_save_path)
            if 'all' in cfg.test.test_mode.lower():
                Eval.evaluate_pose()
                Eval.evaluate_pose_add(fig_save_path)
                Eval.evaluate_pose_arp_2d(fig_save_path)
            elif 'pose' in cfg.test.test_mode.lower():
                Eval.evaluate_pose()
            elif 'add' in cfg.test.test_mode.lower():
                Eval.evaluate_pose_add(fig_save_path)
            elif 'arp' in cfg.test.test_mode.lower():
                Eval.evaluate_pose_arp_2d(fig_save_path)
            else:
                raise Exception("Wrong test mode: {}".format(cfg.test.test_mode))

            return None, None

        else:
            logger.info("test cache file {} and {} not exist!".format(est_cache_file, gt_cache_file))
            userAns = input("Generating cache file from model [Y(y)/N(n)]:")
            if userAns.lower() == 'n':
                sys.exit(0)
            else:
                logger.info("Generating test cache file!")

    preds = {}
    Loss = AverageMeter()
    Loss_rot = AverageMeter()
    Loss_trans = AverageMeter()
    num_iters = len(data_loader)
    bar = Bar('{}'.format(cfg.pytorch.exp_id[-60:]), max=num_iters)

    time_monitor = False
    vis_dir = os.path.join(cfg.pytorch.save_path, 'test_vis_{}'.format(epoch))
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    for i, (obj, obj_id, inp, pose, c_box, s_box, box, trans_local) in enumerate(data_loader):
        if cfg.pytorch.gpu > -1:
            inp_var = inp.cuda(cfg.pytorch.gpu, async=True).float()
        else:
            inp_var = inp.float()

        bs = len(inp)
        # forward propagation
        T_begin = time.time()
        pred_rot, pred_trans = model(inp_var)
        T_end = time.time() - T_begin
        if time_monitor:
            logger.info("time for a batch forward of resnet model is {}".format(T_end))

        if i % cfg.test.disp_interval == 0:
            # input image
            inp_rgb = (inp[0].cpu().numpy().copy() * 255)[[2, 1, 0], :, :].astype(np.uint8)
            cfg.writer.add_image('input_image', inp_rgb, i)
            cv2.imwrite(os.path.join(vis_dir, '{}_inp.png'.format(i)), inp_rgb.transpose(1,2,0)[:, :, ::-1])
            if 'rot' in cfg.pytorch.task.lower():
                # coordinates map
                pred_coor = pred_rot[0, 0:3].data.cpu().numpy().copy()
                pred_coor[0] = im_norm_255(pred_coor[0])
                pred_coor[1] = im_norm_255(pred_coor[1])
                pred_coor[2] = im_norm_255(pred_coor[2])
                pred_coor = np.asarray(pred_coor, dtype=np.uint8)
                cfg.writer.add_image('test_coor_x_pred', np.expand_dims(pred_coor[0], axis=0), i)
                cfg.writer.add_image('test_coor_y_pred', np.expand_dims(pred_coor[1], axis=0), i)
                cfg.writer.add_image('test_coor_z_pred', np.expand_dims(pred_coor[2], axis=0), i)
                # gt_coor = target[0, 0:3].data.cpu().numpy().copy()
                # gt_coor[0] = im_norm_255(gt_coor[0])
                # gt_coor[1] = im_norm_255(gt_coor[1])
                # gt_coor[2] = im_norm_255(gt_coor[2])
                # gt_coor = np.asarray(gt_coor, dtype=np.uint8)
                # cfg.writer.add_image('test_coor_x_gt', np.expand_dims(gt_coor[0], axis=0), i)
                # cfg.writer.add_image('test_coor_y_gt', np.expand_dims(gt_coor[1], axis=0), i)
                # cfg.writer.add_image('test_coor_z_gt', np.expand_dims(gt_coor[2], axis=0), i)
                # confidence map
                pred_conf = pred_rot[0, 3].data.cpu().numpy().copy()
                pred_conf = (im_norm_255(pred_conf)).astype(np.uint8)
                cfg.writer.add_image('test_conf_pred', np.expand_dims(pred_conf, axis=0), i)
                # gt_conf = target[0, 3].data.cpu().numpy().copy()
                # cfg.writer.add_image('test_conf_gt', np.expand_dims(gt_conf, axis=0), i)
            if 'trans' in cfg.pytorch.task.lower():
                pred_trans_ = pred_trans[0].data.cpu().numpy().copy()
                gt_trans_ = trans_local[0].data.cpu().numpy().copy()
                cfg.writer.add_scalar('test_trans_x_gt', gt_trans_[0], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('test_trans_y_gt', gt_trans_[1], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('test_trans_z_gt', gt_trans_[2], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('test_trans_x_pred', pred_trans_[0], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('test_trans_y_pred', pred_trans_[1], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('test_trans_z_pred', pred_trans_[2], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('test_trans_x_err', np.abs(pred_trans_[0]-gt_trans_[0]), i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('test_trans_y_err', np.abs(pred_trans_[1]-gt_trans_[1]), i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('test_trans_z_err', np.abs(pred_trans_[2]-gt_trans_[2]), i + (epoch-1) * num_iters)

        if 'rot' in cfg.pytorch.task.lower():
            pred_coor = pred_rot[:, 0:3].data.cpu().numpy().copy()
            pred_conf = pred_rot[:, 3].data.cpu().numpy().copy()
        else:
            pred_coor = np.zeros(bs)
            pred_conf = np.zeros(bs)

        if 'trans' in cfg.pytorch.task.lower():
            pred_trans = pred_trans.data.cpu().numpy().copy()
        else:
            pred_trans = np.zeros(bs)

        col = list(zip(obj, obj_id.numpy(), pred_coor, pred_conf, pred_trans, pose.numpy(), c_box.numpy(), s_box.numpy(), box.numpy()))
        for idx in range(len(col)):
            obj_, obj_id_, pred_coor_, pred_conf_, pred_trans_, pose_gt, c_box_, s_box_, box_ = col[idx]
            T_begin = time.time()
            if 'rot' in cfg.pytorch.task.lower():
                # building 2D-3D correspondences
                pred_coor_ = pred_coor_.transpose(1, 2, 0)
                pred_coor_[:, :, 0] = pred_coor_[:, :, 0] * abs(obj_info[obj_id_]['min_x'])
                pred_coor_[:, :, 1] = pred_coor_[:, :, 1] * abs(obj_info[obj_id_]['min_y'])
                pred_coor_[:, :, 2] = pred_coor_[:, :, 2] * abs(obj_info[obj_id_]['min_z'])
                pred_coor_= pred_coor_.tolist()
                eroMask = False
                if eroMask:
                    kernel = np.ones((3, 3), np.uint8)
                    pred_conf_ = cv2.erode(pred_conf_, kernel)
                pred_conf_ = (pred_conf_ - pred_conf_.min()) / (pred_conf_.max() - pred_conf_.min())
                pred_conf_ = pred_conf_.tolist()
                
                select_pts_2d = []
                select_pts_3d = []
                c_w = int(c_box_[0])
                c_h = int(c_box_[1])
                s = int(s_box_)
                w_begin = c_w - s / 2.
                h_begin = c_h - s / 2.
                w_unit = s * 1.0 / cfg.dataiter.out_res
                h_unit = s * 1.0 / cfg.dataiter.out_res

                min_x = 0.001 * abs(obj_info[obj_id_]['min_x'])
                min_y = 0.001 * abs(obj_info[obj_id_]['min_y'])
                min_z = 0.001 * abs(obj_info[obj_id_]['min_z'])
                for x in range(cfg.dataiter.out_res):
                    for y in range(cfg.dataiter.out_res):
                        if pred_conf_[x][y] < cfg.test.mask_threshold:
                            continue
                        if abs(pred_coor_[x][y][0]) < min_x  and abs(pred_coor_[x][y][1]) < min_y  and \
                            abs(pred_coor_[x][y][2]) < min_z:
                            continue
                        select_pts_2d.append([w_begin + y * w_unit, h_begin + x * h_unit])
                        select_pts_3d.append(pred_coor_[x][y])

                model_points = np.asarray(select_pts_3d, dtype=np.float32)
                image_points = np.asarray(select_pts_2d, dtype=np.float32)

            if 'trans' in cfg.pytorch.task.lower():
                # compute T from translation head
                ratio_delta_c = pred_trans_[:2]
                ratio_depth = pred_trans_[2]
                pred_depth = ratio_depth * (cfg.dataiter.out_res / s_box_)
                pred_c = ratio_delta_c * box_[2:] + c_box_
                pred_x = (pred_c[0] - cfg.dataset.camera_matrix[0, 2]) * pred_depth / cfg.dataset.camera_matrix[0, 0]
                pred_y = (pred_c[1] - cfg.dataset.camera_matrix[1, 2]) * pred_depth / cfg.dataset.camera_matrix[1, 1]
                T_vector_trans = np.asarray([pred_x, pred_y, pred_depth])
                pose_est_trans = np.concatenate((np.eye(3), np.asarray((T_vector_trans).reshape(3, 1))), axis=1)

            try:
                if 'rot' in cfg.pytorch.task.lower():
                    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
                    if cfg.test.pnp == 'iterPnP': # iterative PnP algorithm
                        success, R_vector, T_vector = cv2.solvePnP(model_points, image_points, cfg.dataset.camera_matrix,
                                                                        dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                    elif cfg.test.pnp == 'ransac': # ransac algorithm
                        _, R_vector, T_vector, inliers = cv2.solvePnPRansac(model_points, image_points,
                                                cfg.dataset.camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
                    else:
                        raise NotImplementedError("Not support PnP algorithm: {}".format(cfg.test.pnp))
                    R_matrix = cv2.Rodrigues(R_vector, jacobian=0)[0]
                    pose_est = np.concatenate((R_matrix, np.asarray(T_vector).reshape(3, 1)), axis=1)
                    if 'trans' in cfg.pytorch.task.lower():
                        pose_est_trans = np.concatenate((R_matrix, np.asarray((T_vector_trans).reshape(3, 1))), axis=1)
                    Eval.pose_est_all[obj_].append(pose_est)
                    Eval.pose_gt_all[obj_].append(pose_gt)
                    Eval.num[obj_] += 1
                    Eval.numAll += 1
                if 'trans' in cfg.pytorch.task.lower():
                    Eval_trans.pose_est_all[obj_].append(pose_est_trans)
                    Eval_trans.pose_gt_all[obj_].append(pose_gt)
                    Eval_trans.num[obj_] += 1
                    Eval_trans.numAll += 1
            except:
                Eval.num[obj_] += 1
                Eval.numAll += 1
                if 'trans' in cfg.pytorch.task.lower():
                    Eval_trans.num[obj_] += 1
                    Eval_trans.numAll += 1
                logger.info('error in solve PnP or Ransac')

            T_end = time.time() - T_begin
            if time_monitor:
                logger.info("time spend on PnP+RANSAC for one image is {}".format(T_end))

        Bar.suffix = 'test Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss {loss.avg:.4f} | Loss_rot {loss_rot.avg:.4f} | Loss_trans {loss_trans.avg:.4f}'.format(
            epoch, i, num_iters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss, loss_rot=Loss_rot, loss_trans=Loss_trans)
        bar.next()

    epoch_save_path = os.path.join(cfg.pytorch.save_path, str(epoch))
    if not os.path.exists(epoch_save_path):
        os.makedirs(epoch_save_path)
    if 'rot' in cfg.pytorch.task.lower():
        logger.info("{} Evaluate of Rotation Branch of Epoch {} {}".format('-'*40, epoch, '-'*40))
        preds['poseGT'] = Eval.pose_gt_all
        preds['poseEst'] = Eval.pose_est_all
        if cfg.pytorch.test:
            np.save(os.path.join(epoch_save_path, 'pose_est_all_test.npy'), Eval.pose_est_all)
            np.save(os.path.join(epoch_save_path, 'pose_gt_all_test.npy'), Eval.pose_gt_all)
        else:
            np.save(os.path.join(epoch_save_path, 'pose_est_all_epoch{}.npy'.format(epoch)), Eval.pose_est_all)
            np.save(os.path.join(epoch_save_path, 'pose_gt_all_epoch{}.npy'.format(epoch)), Eval.pose_gt_all)
        # evaluation
        if 'all' in cfg.test.test_mode.lower():
            Eval.evaluate_pose()
            Eval.evaluate_pose_add(epoch_save_path)
            Eval.evaluate_pose_arp_2d(epoch_save_path)
        else:
            if 'pose' in cfg.test.test_mode.lower():
                Eval.evaluate_pose()
            if 'add' in cfg.test.test_mode.lower():
                Eval.evaluate_pose_add(epoch_save_path)
            if 'arp' in cfg.test.test_mode.lower():
                Eval.evaluate_pose_arp_2d(epoch_save_path)

    if 'trans' in cfg.pytorch.task.lower():
        logger.info("{} Evaluate of Translation Branch of Epoch {} {}".format('-'*40, epoch, '-'*40))
        preds['poseGT'] = Eval_trans.pose_gt_all
        preds['poseEst'] = Eval_trans.pose_est_all
        if cfg.pytorch.test:
            np.save(os.path.join(epoch_save_path, 'pose_est_all_test_trans.npy'), Eval_trans.pose_est_all)
            np.save(os.path.join(epoch_save_path, 'pose_gt_all_test_trans.npy'), Eval_trans.pose_gt_all)
        else:
            np.save(os.path.join(epoch_save_path, 'pose_est_all_trans_epoch{}.npy'.format(epoch)), Eval_trans.pose_est_all)
            np.save(os.path.join(epoch_save_path, 'pose_gt_all_trans_epoch{}.npy'.format(epoch)), Eval_trans.pose_gt_all)
        # evaluation
        if 'all' in cfg.test.test_mode.lower():
            Eval_trans.evaluate_pose()
            Eval_trans.evaluate_pose_add(epoch_save_path)
            Eval_trans.evaluate_pose_arp_2d(epoch_save_path)
        else:
            if 'pose' in cfg.test.test_mode.lower():
                Eval_trans.evaluate_pose()
            if 'add' in cfg.test.test_mode.lower():
                Eval_trans.evaluate_pose_add(epoch_save_path)
            if 'arp' in cfg.test.test_mode.lower():
                Eval_trans.evaluate_pose_arp_2d(epoch_save_path)

    bar.finish()
    return {'Loss': Loss.avg, 'Loss_rot': Loss_rot.avg, 'Loss_trans': Loss_trans.avg}, preds

