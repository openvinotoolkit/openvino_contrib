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
import time
import matplotlib.pyplot as plt

def train(epoch, cfg, data_loader, model, criterions, optimizer=None):
    model.train()
    preds = {}
    Loss = AverageMeter()
    Loss_rot = AverageMeter()
    Loss_trans = AverageMeter()
    num_iters = len(data_loader)
    bar = Bar('{}'.format(cfg.pytorch.exp_id[-60:]), max=num_iters)

    time_monitor = False
    vis_dir = os.path.join(cfg.pytorch.save_path, 'train_vis_{}'.format(epoch))
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    for i, (obj, obj_id, inp, target, loss_msk, trans_local, pose, c_box, s_box, box) in enumerate(data_loader):
        cur_iter = i + (epoch - 1) * num_iters
        if cfg.pytorch.gpu > -1:
            inp_var = inp.cuda(cfg.pytorch.gpu, async=True).float()
            target_var = target.cuda(cfg.pytorch.gpu, async=True).float()
            loss_msk_var  = loss_msk.cuda(cfg.pytorch.gpu, async = True).float()
            trans_local_var = trans_local.cuda(cfg.pytorch.gpu, async=True).float()
            pose_var = pose.cuda(cfg.pytorch.gpu, async=True).float()
            c_box_var = c_box.cuda(cfg.pytorch.gpu, async=True).float()
            s_box_var = s_box.cuda(cfg.pytorch.gpu, async=True).float()
        else:
            inp_var = inp.float()
            target_var = target.float()
            loss_msk_var = loss_msk.float()
            trans_local_var = trans_local.float()
            pose_var = pose.float()
            c_box_var = c_box.float()
            s_box_var = s_box.float()

        bs = len(inp)
        # forward propagation
        T_begin = time.time()
        # import ipdb; ipdb.set_trace()
        pred_rot, pred_trans = model(inp_var)
        T_end = time.time() - T_begin
        if time_monitor:
            logger.info("time for a batch forward of resnet model is {}".format(T_end))

        if i % cfg.test.disp_interval == 0:
            # input image
            inp_rgb = (inp[0].cpu().numpy().copy() * 255)[::-1, :, :].astype(np.uint8)
            cfg.writer.add_image('input_image', inp_rgb, i)
            cv2.imwrite(os.path.join(vis_dir, '{}_inp.png'.format(i)), inp_rgb.transpose(1,2,0)[:, :, ::-1])
            if 'rot' in cfg.pytorch.task.lower():
                # coordinates map
                pred_coor = pred_rot[0, 0:3].data.cpu().numpy().copy()
                pred_coor[0] = im_norm_255(pred_coor[0])
                pred_coor[1] = im_norm_255(pred_coor[1])
                pred_coor[2] = im_norm_255(pred_coor[2])
                pred_coor = np.asarray(pred_coor, dtype=np.uint8)
                cfg.writer.add_image('train_coor_x_pred', np.expand_dims(pred_coor[0], axis=0), i)
                cfg.writer.add_image('train_coor_y_pred', np.expand_dims(pred_coor[1], axis=0), i)
                cfg.writer.add_image('train_coor_z_pred', np.expand_dims(pred_coor[2], axis=0), i)
                cv2.imwrite(os.path.join(vis_dir, '{}_coor_x_pred.png'.format(i)), pred_coor[0])
                cv2.imwrite(os.path.join(vis_dir, '{}_coor_y_pred.png'.format(i)), pred_coor[1])
                cv2.imwrite(os.path.join(vis_dir, '{}_coor_z_pred.png'.format(i)), pred_coor[2])
                gt_coor = target[0, 0:3].data.cpu().numpy().copy()
                gt_coor[0] = im_norm_255(gt_coor[0])
                gt_coor[1] = im_norm_255(gt_coor[1])
                gt_coor[2] = im_norm_255(gt_coor[2])
                gt_coor = np.asarray(gt_coor, dtype=np.uint8)
                cfg.writer.add_image('train_coor_x_gt', np.expand_dims(gt_coor[0], axis=0), i)
                cfg.writer.add_image('train_coor_y_gt', np.expand_dims(gt_coor[1], axis=0), i)
                cfg.writer.add_image('train_coor_z_gt', np.expand_dims(gt_coor[2], axis=0), i)
                cv2.imwrite(os.path.join(vis_dir, '{}_coor_x_gt.png'.format(i)), gt_coor[0])
                cv2.imwrite(os.path.join(vis_dir, '{}_coor_y_gt.png'.format(i)), gt_coor[1])
                cv2.imwrite(os.path.join(vis_dir, '{}_coor_z_gt.png'.format(i)), gt_coor[2])
                # confidence map
                pred_conf = pred_rot[0, 3].data.cpu().numpy().copy()
                pred_conf = (im_norm_255(pred_conf)).astype(np.uint8)
                gt_conf = target[0, 3].data.cpu().numpy().copy()
                cfg.writer.add_image('train_conf_pred', np.expand_dims(pred_conf, axis=0), i)
                cfg.writer.add_image('train_conf_gt', np.expand_dims(gt_conf, axis=0), i)
                cv2.imwrite(os.path.join(vis_dir, '{}_conf_gt.png'.format(i)), gt_conf)
                cv2.imwrite(os.path.join(vis_dir, '{}_conf_pred.png'.format(i)), pred_conf)
            if 'trans' in cfg.pytorch.task.lower():
                pred_trans_ = pred_trans[0].data.cpu().numpy().copy()
                gt_trans_ = trans_local[0].data.cpu().numpy().copy()
                cfg.writer.add_scalar('train_trans_x_gt', gt_trans_[0], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('train_trans_y_gt', gt_trans_[1], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('train_trans_z_gt', gt_trans_[2], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('train_trans_x_pred', pred_trans_[0], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('train_trans_y_pred', pred_trans_[1], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('train_trans_z_pred', pred_trans_[2], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('train_trans_x_err', pred_trans_[0]-gt_trans_[0], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('train_trans_y_err', pred_trans_[1]-gt_trans_[1], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('train_trans_z_err', pred_trans_[2]-gt_trans_[2], i + (epoch-1) * num_iters)

        # loss
        if 'rot' in cfg.pytorch.task.lower() and not cfg.network.rot_head_freeze:
            if cfg.loss.rot_mask_loss:
                loss_rot = criterions[cfg.loss.rot_loss_type](loss_msk_var * pred_rot, loss_msk_var * target_var)
            else:
                loss_rot = criterions[cfg.loss.rot_loss_type](pred_rot, target_var)
        else:
            loss_rot = 0
        if 'trans' in cfg.pytorch.task.lower() and not cfg.network.trans_head_freeze:
            loss_trans = criterions[cfg.loss.trans_loss_type](pred_trans, trans_local_var)
        else:
            loss_trans = 0
        loss = cfg.loss.rot_loss_weight * loss_rot + cfg.loss.trans_loss_weight * loss_trans

        Loss.update(loss.item() if loss != 0 else 0, bs)
        Loss_rot.update(loss_rot.item() if loss_rot != 0 else 0, bs)
        Loss_trans.update(loss_trans.item() if loss_trans != 0 else 0, bs)

        cfg.writer.add_scalar('data/loss_rot_trans', loss.item() if loss != 0 else 0, cur_iter)
        cfg.writer.add_scalar('data/loss_rot', loss_rot.item() if loss_rot != 0 else 0, cur_iter)
        cfg.writer.add_scalar('data/loss_trans', loss_trans.item() if loss_trans != 0 else 0, cur_iter)

        optimizer.zero_grad()
        model.zero_grad()
        T_begin = time.time()
        loss.backward()
        optimizer.step()
        T_end = time.time() - T_begin
        if time_monitor:
            logger.info("time for backward of model: {}".format(T_end))
       
        Bar.suffix = 'train Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss {loss.avg:.4f} | Loss_rot {loss_rot.avg:.4f} | Loss_trans {loss_trans.avg:.4f}'.format(
            epoch, i, num_iters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss, loss_rot=Loss_rot, loss_trans=Loss_trans)
        bar.next()
    bar.finish()
    return {'Loss': Loss.avg, 'Loss_rot': Loss_rot.avg, 'Loss_trans': Loss_trans.avg}, preds
