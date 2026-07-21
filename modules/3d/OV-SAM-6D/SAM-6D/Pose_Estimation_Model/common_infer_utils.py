# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
import heapq

import cv2
import numpy as np
import pycocotools.mask as cocomask
import torch
import torchvision.transforms as transforms
import trimesh
from PIL import Image

from utils.data_utils import (
    get_bbox,
    get_point_cloud_from_depth,
    get_resize_rgb_choose,
    load_im,
)
from utils.draw_utils import draw_detections


rgb_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ConfigNode:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNode(value))
            else:
                setattr(self, key, value)


def load_yaml_config(config_path):
    import yaml

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    return ConfigNode(config_data)


def visualize(rgb, pred_rot, pred_trans, model_points, K, save_path):
    img = draw_detections(rgb, pred_rot, pred_trans, model_points, K, color=(255, 0, 0))
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    prediction = Image.open(save_path)

    rgb_img = Image.fromarray(np.uint8(rgb))
    img_np = np.array(img)
    concat = Image.new("RGB", (img_np.shape[1] + prediction.size[0], img_np.shape[0]))
    concat.paste(rgb_img, (0, 0))
    concat.paste(prediction, (img_np.shape[1], 0))
    return concat


def _normalize_rgb_np(rgb):
    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return rgb.transpose(2, 0, 1)


def _get_template(path, cfg, device, tem_index=1, backend="torch"):
    rgb_path = os.path.join(path, "rgb_" + str(tem_index) + ".png")
    mask_path = os.path.join(path, "mask_" + str(tem_index) + ".png")
    xyz_path = os.path.join(path, "xyz_" + str(tem_index) + ".npy")

    if backend == "numpy":
        rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
        xyz = np.load(xyz_path).astype(np.float32) / 1000.0
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8) == 255

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        mask = mask[y1:y2, x1:x2]

        rgb = rgb[:, :, ::-1][y1:y2, x1:x2, :]
        if cfg.rgb_mask_flag:
            rgb = rgb * (mask[:, :, None] > 0).astype(np.uint8)

        rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = _normalize_rgb_np(rgb)

        xyz = xyz[y1:y2, x1:x2, :].reshape((-1, 3))
        choose = (mask > 0).astype(np.float32).flatten().nonzero()[0]
        if len(choose) <= cfg.n_sample_template_point:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point, replace=False)
        choose = choose[choose_idx]
        xyz = xyz[choose, :]

        h, w = y2 - y1, x2 - x1
        scale_h, scale_w = cfg.img_size / h, cfg.img_size / w
        choose_y = (choose // w).astype(np.float32) * scale_h
        choose_x = (choose % w).astype(np.float32) * scale_w
        rgb_choose = (choose_y * cfg.img_size + choose_x).astype(np.int32)
        return rgb, rgb_choose, xyz

    rgb = load_im(rgb_path).astype(np.uint8)
    xyz = np.load(xyz_path).astype(np.float32) / 1000.0
    mask = load_im(mask_path).astype(np.uint8) == 255

    bbox = get_bbox(mask)
    y1, y2, x1, x2 = bbox
    mask = mask[y1:y2, x1:x2]

    rgb = rgb[:, :, ::-1][y1:y2, x1:x2, :]
    if cfg.rgb_mask_flag:
        rgb = rgb * (mask[:, :, None] > 0).astype(np.uint8)

    rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
    rgb = rgb_transform(np.array(rgb))

    choose = (mask > 0).astype(np.float32).flatten().nonzero()[0]
    if len(choose) <= cfg.n_sample_template_point:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point)
    else:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point, replace=False)
    choose = choose[choose_idx]
    xyz = xyz[y1:y2, x1:x2, :].reshape((-1, 3))[choose, :]

    rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)
    return rgb, rgb_choose, xyz


def get_templates(path, cfg, device=None, backend="torch"):
    n_template_view = cfg.n_template_view
    all_tem = []
    all_tem_choose = []
    all_tem_pts = []

    total_nView = 42
    for v in range(n_template_view):
        i = int(total_nView / n_template_view * v)
        tem, tem_choose, tem_pts = _get_template(path, cfg, device, i, backend=backend)
        if backend == "numpy":
            all_tem.append(tem)
            all_tem_choose.append(tem_choose)
            all_tem_pts.append(np.expand_dims(tem_pts, axis=0))
            continue

        all_tem.append(torch.FloatTensor(tem).unsqueeze(0).to(device))
        all_tem_choose.append(torch.IntTensor(tem_choose).long().unsqueeze(0).to(device))
        all_tem_pts.append(torch.FloatTensor(tem_pts).unsqueeze(0).to(device))
    return all_tem, all_tem_pts, all_tem_choose


def get_test_data(
    rgb_path,
    depth_path,
    cam_path,
    cad_path,
    seg_path,
    det_score_thresh,
    cfg,
    device=None,
    backend="torch",
    topk=None,
    observed_index_mode=None,
):
    dets = []
    with open(seg_path) as f:
        dets_ = json.load(f)

    if topk is not None:
        dets_ = heapq.nlargest(topk, dets_, key=lambda det: det["score"])

    for det in dets_:
        if det["score"] > det_score_thresh:
            dets.append(det)
    del dets_

    cam_info = json.load(open(cam_path))
    K = np.array(cam_info["cam_K"]).reshape(3, 3)

    whole_image = load_im(rgb_path).astype(np.uint8)
    if len(whole_image.shape) == 2:
        whole_image = np.concatenate([whole_image[:, :, None], whole_image[:, :, None], whole_image[:, :, None]], axis=2)
    whole_depth = load_im(depth_path).astype(np.float32) * cam_info["depth_scale"] / 1000.0
    whole_pts = get_point_cloud_from_depth(whole_depth, K)

    mesh = trimesh.load_mesh(cad_path)
    model_points = mesh.sample(cfg.n_sample_model_point).astype(np.float32) / 1000.0
    radius = np.max(np.linalg.norm(model_points, axis=1))

    all_rgb = []
    all_cloud = []
    all_rgb_choose = []
    all_score = []
    all_dets = []
    for inst in dets:
        seg = inst["segmentation"]
        score = inst["score"]

        h, w = seg["size"]
        try:
            rle = cocomask.frPyObjects(seg, h, w)
        except Exception:
            rle = seg
        mask = cocomask.decode(rle)
        mask = np.logical_and(mask > 0, whole_depth > 0)
        if np.sum(mask) > 32:
            bbox = get_bbox(mask)
            y1, y2, x1, x2 = bbox
        else:
            continue
        mask = mask[y1:y2, x1:x2]
        choose = mask.astype(np.float32).flatten().nonzero()[0]

        cloud = whole_pts.copy()[y1:y2, x1:x2, :].reshape(-1, 3)[choose, :]
        center = np.mean(cloud, axis=0)
        tmp_cloud = cloud - center[None, :]
        flag = np.linalg.norm(tmp_cloud, axis=1) < radius * 1.2
        if np.sum(flag) < 4:
            continue
        choose = choose[flag]
        cloud = cloud[flag]

        n_obs = cfg.n_sample_observed_point
        n_pts = len(choose)
        if n_pts == 0:
            continue

        mode = observed_index_mode
        if mode is None:
            mode = "linspace" if backend == "numpy" else "random"

        if mode == "linspace":
            choose_idx = np.linspace(0, n_pts - 1, n_obs).astype(np.int64)
        else:
            if n_pts <= n_obs:
                choose_idx = np.random.choice(np.arange(n_pts), n_obs)
            else:
                choose_idx = np.random.choice(np.arange(n_pts), n_obs, replace=False)

        choose = choose[choose_idx]
        cloud = cloud[choose_idx]

        rgb = whole_image.copy()[y1:y2, x1:x2, :][:, :, ::-1]
        if cfg.rgb_mask_flag:
            rgb = rgb * (mask[:, :, None] > 0).astype(np.uint8)
        rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
        if backend == "numpy":
            rgb = _normalize_rgb_np(np.array(rgb))
        else:
            rgb = rgb_transform(np.array(rgb))
        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)

        if backend == "numpy":
            all_rgb.append(rgb.astype(np.float32))
            all_cloud.append(cloud.astype(np.float32))
            all_rgb_choose.append(rgb_choose.astype(np.int32))
        else:
            all_rgb.append(torch.FloatTensor(rgb))
            all_cloud.append(torch.FloatTensor(cloud))
            all_rgb_choose.append(torch.IntTensor(rgb_choose).long())
        all_score.append(score)
        all_dets.append(inst)

    ret_dict = {}
    if backend == "numpy":
        ret_dict["pts"] = np.stack(all_cloud)
        ret_dict["rgb"] = np.stack(all_rgb)
        ret_dict["rgb_choose"] = np.stack(all_rgb_choose)
        ret_dict["score"] = np.array(all_score, dtype=np.float32)

        ninstance = ret_dict["pts"].shape[0]
        ret_dict["model"] = np.repeat(model_points[np.newaxis, :, :], ninstance, axis=0)
        ret_dict["K"] = np.repeat(K[np.newaxis, :, :], ninstance, axis=0)
    else:
        ret_dict["pts"] = torch.stack(all_cloud).to(device)
        ret_dict["rgb"] = torch.stack(all_rgb).to(device)
        ret_dict["rgb_choose"] = torch.stack(all_rgb_choose).to(device)
        ret_dict["score"] = torch.FloatTensor(all_score).to(device)

        ninstance = ret_dict["pts"].size(0)
        ret_dict["model"] = torch.FloatTensor(model_points).unsqueeze(0).repeat(ninstance, 1, 1).to(device)
        ret_dict["K"] = torch.FloatTensor(K).unsqueeze(0).repeat(ninstance, 1, 1).to(device)
    return ret_dict, whole_image, whole_pts.reshape(-1, 3), model_points, all_dets
