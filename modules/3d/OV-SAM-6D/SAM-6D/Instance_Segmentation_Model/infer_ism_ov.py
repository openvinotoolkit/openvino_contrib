#!/usr/bin/env python3
"""
OpenVINO inference for ISM models.

Example usage:
    python3 infer_ism_ov.py \
        --ov_model_dir ./checkpoints/ov_models \
        --ov_device GPU \
        --ext \
        --segmentor_model fastsam \
        --image SAM-6D/Data/Example/rgb.png \
        --cad SAM-6D/Data/Example/obj_000005.ply \
        --templates_dir SAM-6D/Data/Example/outputs/templates \
        --output_dir SAM-6D/Data/Example/outputs/ref/ism_ov_gpu_fastsam_results \
        --gt_mask SAM-6D/Data/Example/mask_visib
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.ops.boxes import box_area
import imageio
from plyfile import PlyData

import openvino as ov

# ---------------------------------------------------------------------------
# Ensure ISM code is importable
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.utils.amg import (
    MaskData,
    batch_iterator,
    batched_mask_to_box,
    build_all_layer_point_grids,
    calculate_stability_score,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)
from utils.bbox_utils import CropResizePad
from model.utils import BatchedData, Detections, convert_npz_to_json
from utils.inout import save_json_bop23
from PIL import Image
from eval_utils import evaluate_and_print_map


def _calculate_query_translation_ov(proposals, depth,
                                    cam_intrinsic, depth_scale):
    if proposals.ndim == 4:
        proposals = proposals.squeeze(1)
    elif proposals.ndim == 2:
        proposals = proposals.unsqueeze(0)

    n_query, image_h, image_w = proposals.shape
    masks = proposals.to(device=depth.device, dtype=torch.float32).reshape(
        n_query, -1)
    depth_f = depth.to(torch.float32)
    depth_scale_f = depth_scale.to(
        device=depth.device, dtype=torch.float32).reshape(-1)[0]
    cam_intrinsic_f = cam_intrinsic.to(
        device=depth.device, dtype=torch.float32)

    z = depth_f * depth_scale_f / 1000.0
    valid = (z > 0).reshape(-1).to(torch.float32)
    u = torch.arange(image_w, device=depth.device,
                     dtype=torch.float32).view(1, image_w).expand(
                         image_h, image_w)
    v = torch.arange(image_h, device=depth.device,
                     dtype=torch.float32).view(image_h, 1).expand(
                         image_h, image_w)
    x = (u - cam_intrinsic_f[0, 2]) * z / cam_intrinsic_f[0, 0]
    y = (v - cam_intrinsic_f[1, 2]) * z / cam_intrinsic_f[1, 1]

    valid_num = masks.matmul(valid).clamp_min(1e-8)

    avg_x = masks.matmul(x.reshape(-1)) / valid_num
    avg_y = masks.matmul(y.reshape(-1)) / valid_num
    avg_z = masks.matmul(z.reshape(-1)) / valid_num

    return torch.stack((avg_x, avg_y, avg_z), dim=1)


def _project_template_to_image_ov(model, best_pose, pred_object_idx,
                                  batch, proposals):

    pose_r = model.ref_data["poses"][best_pose, 0:3, 0:3]
    select_pc = model.ref_data["pointcloud"][pred_object_idx, ...]
    posed_pc = torch.matmul(pose_r, select_pc.permute(0, 2, 1)).permute(
        0, 2, 1)
    translate = _calculate_query_translation_ov(
        proposals, batch["depth"][0], batch["cam_intrinsic"][0],
        batch["depth_scale"])
    posed_pc = posed_pc + translate[:, None, :]

    cam_intrinsic = batch["cam_intrinsic"][0].to(torch.float32)
    image_homo = torch.matmul(posed_pc, cam_intrinsic.transpose(0, 1))
    image_vu = (image_homo / image_homo[:, :, -1:])[:, :, 0:2].to(torch.int)
    image_h, image_w = batch["depth"][0].shape
    image_vu[:, :, 0].clamp_(min=0, max=image_w - 1)
    image_vu[:, :, 1].clamp_(min=0, max=image_h - 1)

    return image_vu


# ===================================================================
#  OVSamPredictor
# ===================================================================

class OVSamPredictor:
    """OpenVINO ``segment_anything.SamPredictor``.

    Loads two compiled OV models:
      - SAM image encoder
      - SAM mask decoder
    """

    def __init__(
        self,
        encoder_compiled: ov.CompiledModel,
        decoder_compiled: ov.CompiledModel,
        image_size: int = 1024,
    ):
        self.encoder_compiled = encoder_compiled
        self.decoder_compiled = decoder_compiled
        self.image_size = image_size
        self.transform = ResizeLongestSide(image_size)
        self.mask_threshold = 0.0

        self.ov_core = None  # set by init_model_ov to share a Core for unload_plugin

        # SAM pixel normalisation
        self._pixel_mean = np.array(
            [123.675, 116.28, 103.53], dtype=np.float32
        ).reshape(1, 3, 1, 1)
        self._pixel_std = np.array(
            [58.395, 57.12, 57.375], dtype=np.float32
        ).reshape(1, 3, 1, 1)

        # Persistent InferRequest objects - GPU buffers are allocated ONCE
        # at init and reused on every call, preventing buffer accumulation

        self._encoder_request = encoder_compiled.create_infer_request()
        self._decoder_request = decoder_compiled.create_infer_request()

        self.reset_image()

    # ---- compatibility properties ----

    @property
    def device(self) -> torch.device:
        """Tensors live on CPU; OV handles device placement internally."""
        return torch.device("cpu")

    # ---- public API ----

    def set_image(
        self,
        image: np.ndarray,
        image_format: str = "RGB",
    ) -> None:
        """Compute and cache the image embedding via OV image encoder."""
        if image_format != "RGB":
            image = image[..., ::-1]

        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, dtype=torch.float32)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[
            None, :, :, :
        ]
        self.set_torch_image(input_image_torch, image.shape[:2])

    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: tuple,
    ) -> None:
        """Run OV image encoder on the pre-transformed image tensor."""
        self.reset_image()
        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])

        # Pre-process: normalise + pad to 1024x1024
        input_np = transformed_image.float().numpy()
        input_np = (input_np - self._pixel_mean) / self._pixel_std
        h, w = input_np.shape[2], input_np.shape[3]
        padh = self.image_size - h
        padw = self.image_size - w
        if padh > 0 or padw > 0:
            input_np = np.pad(
                input_np,
                ((0, 0), (0, 0), (0, padh), (0, padw)),
                mode="constant",
                constant_values=0,
            )

        # OV image encoder - SINGLE call per image.
        # Use persistent request: reuses GPU buffers, no new allocation.
        self._encoder_request.infer({0: input_np})
        # Copy output immediately so the request buffer is free for reuse.
        self.features = self._encoder_request.get_output_tensor(0).data.copy()
        self.is_image_set = True

    @torch.no_grad()
    def predict_torch(
        self,
        point_coords: torch.Tensor | None,
        point_labels: torch.Tensor | None,
        boxes: torch.Tensor | None = None,
        mask_input: torch.Tensor | None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run OV mask decoder on cached image embedding + point prompts."""
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        coords_np = point_coords.cpu().float().numpy()
        labels_np = point_labels.cpu().to(torch.int32).numpy()

        # OV mask decoder - called N times per image (once per point batch).
        # Use persistent request: one set of GPU buffers reused every call.
        # Outputs are copied immediately so the request is free for next call.
        self._decoder_request.infer([self.features, coords_np, labels_np])
        low_res_masks = torch.from_numpy(
            self._decoder_request.get_output_tensor(0).data.copy()
        )  # [B, 3, 256, 256]
        iou_predictions = torch.from_numpy(
            self._decoder_request.get_output_tensor(1).data.copy()
        )  # [B, 3]

        # Post-process: 256 -> 1024 -> crop to input_size -> original_size
        masks = F.interpolate(
            low_res_masks.float(),
            (self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : self.input_size[0], : self.input_size[1]]
        masks = F.interpolate(
            masks, self.original_size, mode="bilinear", align_corners=False
        )

        if not return_logits:
            masks = masks > self.mask_threshold

        return masks, iou_predictions, low_res_masks

    def get_image_embedding(self) -> torch.Tensor:
        if not self.is_image_set:
            raise RuntimeError("An image must be set first.")
        return torch.from_numpy(np.array(self.features))

    def reset_image(self) -> None:
        self.is_image_set = False
        self.features = None
        self.original_size = None
        self.input_size = None


# ===================================================================
#  OVCustomSamAutomaticMaskGenerator
# ===================================================================

class OVCustomSamAutomaticMaskGenerator:
    """OpenVINO ``model.sam.CustomSamAutomaticMaskGenerator``.
    """

    def __init__(
        self,
        predictor: OVSamPredictor,
        points_per_side: int = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.85,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        min_mask_region_area: int = 0,
        segmentor_width_size: int | None = None,
        ov_device: str = "CPU",
    ):
        self.predictor = predictor
        self.ov_device = ov_device
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.segmentor_width_size = segmentor_width_size

        self.point_grids = build_all_layer_point_grids(
            points_per_side, crop_n_layers, crop_n_points_downscale_factor
        )

        # Lazy-init OV NMS model (built on first call)
        self._nms_compiled = None

    # ---- OV NonMaxSuppression ----

    def _build_nms_model(self):
        """Build and compile an OV NonMaxSuppression-9 model on the
        same device as the SAM predictor."""
        from openvino import opset13 as opset

        boxes_p = opset.parameter(
            [1, -1, 4], np.float32, name="boxes")
        scores_p = opset.parameter(
            [1, 1, -1], np.float32, name="scores")
        max_boxes_p = opset.parameter(
            [], np.int64, name="max_output_boxes_per_class")
        iou_thresh_p = opset.parameter(
            [], np.float32, name="iou_threshold")
        score_thresh_p = opset.parameter(
            [], np.float32, name="score_threshold")

        nms_node = opset.non_max_suppression(
            boxes_p, scores_p, max_boxes_p, iou_thresh_p, score_thresh_p,
            box_encoding="corner",
            sort_result_descending=False,
            output_type="i64",
        )

        model = ov.Model(
            [nms_node.output(0), nms_node.output(1), nms_node.output(2)],
            [boxes_p, scores_p, max_boxes_p, iou_thresh_p, score_thresh_p],
            "ov_nms",
        )

        device = self.ov_device
        core = getattr(self.predictor, "ov_core", None) or ov.Core()
        self._nms_core = core  # keep reference so caller can reset
        config = {}
        if device.upper().startswith("GPU"):
            config = {"PERFORMANCE_HINT": "LATENCY",
                      "NUM_STREAMS": "1",
                      "INFERENCE_PRECISION_HINT": "f32"}  # NMS needs f32 precision
        self._nms_compiled = core.compile_model(model, device, config)

        # Persistent request: buffers reused every NMS call.
        self._nms_request = self._nms_compiled.create_infer_request()
        print(f"OV NMS model compiled on {device}")

    def _ov_nms(self, boxes: torch.Tensor, scores: torch.Tensor,
                iou_threshold: float) -> torch.Tensor:
        """batched_nms using OV NonMaxSuppression-9.

        Args:
            boxes: [N, 4] float tensor (xyxy)
            scores: [N] float tensor
            iou_threshold: IoU threshold

        Returns:
            keep: 1-D int64 tensor of selected box indices
        """
        if self._nms_compiled is None:
            self._build_nms_model()

        n = boxes.shape[0]
        if n == 0:
            return torch.zeros(0, dtype=torch.long)

        boxes_np = boxes.detach().cpu().numpy().reshape(1, n, 4).astype(
            np.float32)
        scores_np = scores.detach().cpu().numpy().reshape(1, 1, n).astype(
            np.float32)

        self._nms_request.infer({
            "boxes": boxes_np,
            "scores": scores_np,
            "max_output_boxes_per_class": np.array(n, dtype=np.int64),
            "iou_threshold": np.array(iou_threshold, dtype=np.float32),
            "score_threshold": np.array(0.0, dtype=np.float32),
        })

        selected_indices = self._nms_request.get_output_tensor(0).data.copy()  # [K, 3]
        valid_count = int(self._nms_request.get_output_tensor(2).data.item())

        # Column 2 is box_index; filter out padding (-1 entries)
        keep = selected_indices[:valid_count, 2].astype(np.int64)

        return torch.from_numpy(keep).to(boxes.device)

    # ---- public API ----

    def preprocess_resize(self, image: np.ndarray) -> np.ndarray:
        orig_size = image.shape[:2]
        height_size = int(self.segmentor_width_size * orig_size[0] / orig_size[1])

        return cv2.resize(
            image.copy(), (self.segmentor_width_size, height_size)
        )

    def postprocess_resize(self, detections: dict, orig_size: tuple) -> dict:
        detections["masks"] = F.interpolate(
            detections["masks"].unsqueeze(1).float(),
            size=(orig_size[0], orig_size[1]),
            mode="bilinear",
            align_corners=False,
        )[:, 0, :, :]

        scale = orig_size[1] / self.segmentor_width_size
        detections["boxes"] = detections["boxes"].float() * scale
        detections["boxes"][:, [0, 2]] = torch.clamp(
            detections["boxes"][:, [0, 2]], 0, orig_size[1] - 1
        )
        detections["boxes"][:, [1, 3]] = torch.clamp(
            detections["boxes"][:, [1, 3]], 0, orig_size[0] - 1
        )

        return detections

    @torch.no_grad()
    def generate_masks(self, image: np.ndarray) -> dict:
        """Generate masks for the input image. Main entry point."""
        if self.segmentor_width_size is not None:
            orig_size = image.shape[:2]
            image = self.preprocess_resize(image)

        mask_data = self._generate_masks(image)

        if self.segmentor_width_size is not None:
            mask_data = self.postprocess_resize(mask_data, orig_size)
        return mask_data

    def _generate_masks(self, image: np.ndarray) -> dict:
        """Core mask generation: crops -> point grids -> SAM decoder -> NMS."""
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(
                image, crop_box, layer_idx, orig_size
            )
            data.cat(crop_data)

        # Remove duplicates across crops (OV NonMaxSuppression-9)
        if len(crop_boxes) > 1:
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = self._ov_nms(
                data["boxes"].float(), scores,
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)

        if "masks" not in data._stats:
            data["masks"] = [torch.from_numpy(rle_to_mask(rle))
                             for rle in data["rles"]]
            data["masks"] = torch.stack(data["masks"])

        return {
            "masks": data["masks"].to(data["boxes"].device),
            "boxes": data["boxes"],
        }

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: list[int],
        crop_layer_idx: int,
        orig_size: tuple[int, ...],
    ) -> MaskData:
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        self.predictor.set_image(cropped_im)

        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Accumulate batch results
        batch_results = []
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(
                points, cropped_im_size, crop_box, orig_size
            )
            batch_results.append(batch_data)
            del batch_data
        self.predictor.reset_image()

        # Combine all batches in one pass
        data = MaskData()
        if batch_results:
            all_keys = batch_results[0]._stats.keys()
            for key in all_keys:
                vals = [b[key] for b in batch_results
                        if key in b._stats and b[key] is not None]

                if not vals:
                    continue

                if isinstance(vals[0], torch.Tensor):
                    data._stats[key] = torch.cat(vals, dim=0)
                elif isinstance(vals[0], np.ndarray):
                    data._stats[key] = np.concatenate(vals, axis=0)
                elif isinstance(vals[0], list):
                    combined = []
                    for v in vals:
                        combined.extend(v)

                    data._stats[key] = combined

            del batch_results

        keep_by_nms = self._ov_nms(
            data["boxes"].float(), data["iou_preds"],
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        n_items = len(data["masks"]) if "masks" in data._stats \
            else len(data["rles"])
        data["crop_boxes"] = torch.tensor(
            [crop_box for _ in range(n_items)]
        )

        return data

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: tuple[int, ...],
        crop_box: list[int],
        orig_size: tuple[int, ...],
    ) -> MaskData:
        orig_h, orig_w = orig_size

        transformed_points = self.predictor.transform.apply_coords(
            points, im_size
        )
        in_points = torch.as_tensor(
            transformed_points, device=self.predictor.device
        )
        in_labels = torch.ones(
            in_points.shape[0], dtype=torch.int, device=in_points.device
        )
        masks, iou_preds, extra = self.predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=True,
            return_logits=True,
        )

        # Check if decoder returned v2 extras (stability_scores, boxes)
        has_v2 = isinstance(extra, tuple) and len(extra) == 2

        if has_v2:
            # v2 decoder: masks are binary, stability_scores + boxes pre-computed
            stability_scores_v2, boxes_v2 = extra
            # masks: [B, 3, H, W] binary bool/f32
            # stability_scores_v2: [B, 3]
            # boxes_v2: [B, 3, 4]
            flat_masks = masks.flatten(0, 1)
            flat_iou_preds = iou_preds.flatten(0, 1)
            flat_stability = stability_scores_v2.flatten(0, 1)
            flat_boxes = boxes_v2.reshape(-1, 4)

            data = MaskData(
                masks=flat_masks,                      # [B*3, H, W]
                iou_preds=flat_iou_preds,              # [B*3]
                points=torch.as_tensor(
                    points.repeat(masks.shape[1], axis=0)),
                stability_score=flat_stability,        # [B*3]
                boxes=flat_boxes,                      # [B*3, 4]
            )
            del masks

            # Combine all metadata filters
            keep_mask = torch.ones_like(flat_iou_preds, dtype=torch.bool)
            if self.pred_iou_thresh > 0.0:
                keep_mask &= flat_iou_preds > self.pred_iou_thresh
            if self.stability_score_thresh > 0.0:
                keep_mask &= flat_stability >= self.stability_score_thresh
            keep_mask &= ~is_box_near_crop_edge(
                flat_boxes, crop_box, [0, 0, orig_w, orig_h]
            )
            if not torch.all(keep_mask):
                data.filter(keep_mask)

            if data["masks"].dtype != torch.bool:
                data["masks"] = data["masks"] > 0.5

        else:
            # v1 decoder: logits returned, compute stability + boxes on CPU
            data = MaskData(
                masks=masks.flatten(0, 1),
                iou_preds=iou_preds.flatten(0, 1),
                points=torch.as_tensor(
                    points.repeat(masks.shape[1], axis=0)),
            )
            del masks

            # Filter by predicted IoU
            if self.pred_iou_thresh > 0.0:
                keep_mask = data["iou_preds"] > self.pred_iou_thresh
                data.filter(keep_mask)

            # Stability score (CPU fallback)
            data["stability_score"] = calculate_stability_score(
                data["masks"],
                self.mask_threshold,
                self.stability_score_offset,
            )
            if self.stability_score_thresh > 0.0:
                keep_mask = data["stability_score"] >= self.stability_score_thresh
                data.filter(keep_mask)

            # Threshold masks
            data["masks"] = data["masks"] > self.mask_threshold

            # Compute boxes (CPU fallback)
            data["boxes"] = batched_mask_to_box(data["masks"])

            # Filter boxes near crop edge
            keep_mask = ~is_box_near_crop_edge(
                data["boxes"], crop_box, [0, 0, orig_w, orig_h]
            )
            if not torch.all(keep_mask):
                data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        if self.crop_n_layers > 0:
            data["rles"] = mask_to_rle_pytorch(data["masks"])
            del data["masks"]

        return data

    @property
    def mask_threshold(self) -> float:
        return self.predictor.mask_threshold


# ===================================================================
#  OVCustomDINOv2
# ===================================================================

class OVCustomDINOv2:
    """OpenVINO ``model.dinov2.CustomDINOv2``.

    Delegates all NN forward calls to an OV compiled DINOv2 model.
    """

    def __init__(
        self,
        dinov2_compiled: ov.CompiledModel,
        model_name: str = "dinov2_vitl14",
        image_size: int = 224,
        chunk_size: int = 16,
        descriptor_width_size: int = 640,
        patch_size: int = 14,
        validpatch_thresh: float = 0.5,
    ):
        self.dinov2_compiled = dinov2_compiled
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.patch_size = patch_size
        self.proposal_size = image_size
        self.descriptor_width_size = descriptor_width_size
        self.validpatch_thresh = validpatch_thresh

        self.model = type(
            "_DummyModel", (), {"device": torch.device("cpu")}
        )()

        self.rgb_normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        self.rgb_proposal_processor = CropResizePad(self.proposal_size)
        self.patch_kernel = torch.nn.AvgPool2d(
            kernel_size=self.patch_size, stride=self.patch_size
        )

        # Persistent InferRequest: one GPU buffer allocation, reused every call.
        self._dinov2_request = dinov2_compiled.create_infer_request()
        print("OVCustomDINOv2 initialised")

    # ---- internal OV call ----

    def _ov_forward(
        self, images_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run DINOv2 OV model -> (cls_token, patch_tokens)."""
        images_np = images_tensor.cpu().float().numpy()
        N = images_np.shape[0]
        if N < self.chunk_size:
            pad = np.zeros(
                (self.chunk_size - N, *images_np.shape[1:]), dtype=images_np.dtype
            )
            images_np = np.concatenate([images_np, pad], axis=0)
        # Persistent request: GPU buffers reused, outputs copied immediately.
        self._dinov2_request.infer({0: images_np})
        cls_token = torch.from_numpy(
            self._dinov2_request.get_output_tensor(0).data[:N].copy()
        )  # [N, 1024]
        patch_tokens = torch.from_numpy(
            self._dinov2_request.get_output_tensor(1).data[:N].copy()
        )  # [N, 256, 1024]
        return cls_token, patch_tokens

    # ---- pre-processing ----

    def process_rgb_proposals(
        self,
        image_np: np.ndarray,
        masks: torch.Tensor,
        boxes: torch.Tensor,
    ) -> torch.Tensor:
        num_proposals = len(masks)
        rgb = self.rgb_normalize(image_np).float()
        rgbs = rgb.unsqueeze(0).repeat(num_proposals, 1, 1, 1)
        masked_rgbs = rgbs * masks.unsqueeze(1)
        return self.rgb_proposal_processor(masked_rgbs, boxes)

    def process_masks_proposals(
        self,
        masks: torch.Tensor,
        boxes: torch.Tensor,
    ) -> torch.Tensor:
        masks.unsqueeze_(1)
        processed = self.rgb_proposal_processor(masks, boxes).squeeze_()
        return processed

    # ---- feature extraction methods ----

    @torch.no_grad()
    def compute_features(
        self, images: torch.Tensor, token_name: str
    ) -> torch.Tensor:
        if token_name == "x_norm_clstoken":
            if images.shape[0] > self.chunk_size:
                return self.forward_by_chunk(images)
            cls_token, _ = self._ov_forward(images)
            return cls_token
        raise NotImplementedError(f"token_name={token_name!r} not supported")

    @torch.no_grad()
    def forward_by_chunk(self, processed_rgbs: torch.Tensor) -> torch.Tensor:
        batch_rgbs = BatchedData(batch_size=self.chunk_size, data=processed_rgbs)
        del processed_rgbs
        features = BatchedData(batch_size=self.chunk_size)

        for idx_batch in range(len(batch_rgbs)):
            feats = self.compute_features(
                batch_rgbs[idx_batch], token_name="x_norm_clstoken"
            )
            features.cat(feats)

        return features.data

    @torch.no_grad()
    def compute_masked_patch_feature(
        self, images: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        if images.shape[0] > self.chunk_size:
            return self.forward_by_chunk_v2(images, masks)
        _, patch_tokens = self._ov_forward(images)
        features_mask = (
            self.patch_kernel(masks).flatten(-2) > self.validpatch_thresh
        )
        features_mask = features_mask.unsqueeze(-1).repeat(
            1, 1, patch_tokens.shape[-1]
        )
        return F.normalize(patch_tokens * features_mask, dim=-1)

    @torch.no_grad()
    def forward_by_chunk_v2(
        self, processed_rgbs: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        batch_rgbs = BatchedData(batch_size=self.chunk_size, data=processed_rgbs)
        batch_masks = BatchedData(batch_size=self.chunk_size, data=masks)
        del processed_rgbs, masks

        features = BatchedData(batch_size=self.chunk_size)

        for idx_batch in range(len(batch_rgbs)):
            feats = self.compute_masked_patch_feature(
                batch_rgbs[idx_batch], batch_masks[idx_batch]
            )
            features.cat(feats)

        return features.data

    @torch.no_grad()
    def compute_cls_and_patch_features(
        self, images: torch.Tensor, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cls_token, patch_tokens = self._ov_forward(images)
        features_mask = (
            self.patch_kernel(masks).flatten(-2) > self.validpatch_thresh
        )
        features_mask = features_mask.unsqueeze(-1).repeat(
            1, 1, patch_tokens.shape[-1]
        )
        patch_tokens = F.normalize(patch_tokens * features_mask, dim=-1)

        return cls_token, patch_tokens

    @torch.no_grad()
    def forward(
        self, image_np: np.ndarray, proposals: Detections
    ) -> tuple[torch.Tensor, torch.Tensor]:
        processed_rgbs = self.process_rgb_proposals(
            image_np, proposals.masks, proposals.boxes
        )
        processed_masks = self.process_masks_proposals(
            proposals.masks, proposals.boxes
        )

        batch_rgbs = BatchedData(
            batch_size=self.chunk_size, data=processed_rgbs
        )
        batch_masks = BatchedData(
            batch_size=self.chunk_size, data=processed_masks
        )
        del processed_rgbs, processed_masks

        cls_features = BatchedData(batch_size=self.chunk_size)
        patch_features = BatchedData(batch_size=self.chunk_size)
        for idx_batch in range(len(batch_rgbs)):
            cls_feats, patch_feats = self.compute_cls_and_patch_features(
                batch_rgbs[idx_batch], batch_masks[idx_batch]
            )
            cls_features.cat(cls_feats)
            patch_features.cat(patch_feats)

        return cls_features.data, patch_features.data

    def __call__(self, image_np, proposals):
        return self.forward(image_np, proposals)


# ===================================================================
#  Extended OV classes
# ===================================================================

class OVSamPredictorExt:
    """Extended OV SAM predictor using custom-op IRs.

    Uses:
      - sam_image_encoder_ext: SamPreprocess baked in
        Input: raw image [1,H,W,3] u8  -> Output: embeddings + input_size
      - sam_mask_decoder_ext: SamMaskPostprocess baked in
        Input: embeddings + coords + labels + postprocess_params
        -> Output: masks + iou_preds + logits (already at original size)
    """

    def __init__(
        self,
        encoder_compiled: ov.CompiledModel,
        decoder_compiled: ov.CompiledModel,
        image_size: int = 1024,
    ):
        self.encoder_compiled = encoder_compiled
        self.decoder_compiled = decoder_compiled
        self.image_size = image_size
        self.transform = ResizeLongestSide(image_size)
        self.mask_threshold = 0.0
        self._encoder_request = encoder_compiled.create_infer_request()
        self._decoder_request = decoder_compiled.create_infer_request()
        self.reset_image()

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def set_image(
        self,
        image: np.ndarray,
        image_format: str = "RGB",
    ) -> None:
        if image_format != "RGB":
            image = image[..., ::-1]
        self.reset_image()
        self.original_size = image.shape[:2]

        # Pass raw u8 image to encoder with SamPreprocess baked in
        input_4d = image[np.newaxis, ...]  # [1, H, W, 3]

        self._encoder_request.infer({'image': input_4d})
        self.features = self._encoder_request.get_output_tensor(
            0).data.copy()                # [1, 256, 64, 64]
        input_size = self._encoder_request.get_output_tensor(
            1).data.copy()                # [1, 1, 1, 2] -> (new_h, new_w)
        self.input_size = (int(input_size.flat[0]), int(input_size.flat[1]))
        self.is_image_set = True

    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: tuple,
    ) -> None:
        raise NotImplementedError(
            "OVSamPredictorExt uses raw images; "
            "call set_image() instead of set_torch_image()")

    @torch.no_grad()
    def predict_torch(
        self,
        point_coords: torch.Tensor | None,
        point_labels: torch.Tensor | None,
        boxes: torch.Tensor | None = None,
        mask_input: torch.Tensor | None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.is_image_set:
            raise RuntimeError("set_image() must be called first.")

        coords_np = point_coords.cpu().float().numpy()
        labels_np = point_labels.cpu().to(torch.int32).numpy()

        self._decoder_request.infer([self.features, coords_np, labels_np])

        n_outputs = len(self.decoder_compiled.outputs)
        if n_outputs >= 4:
            # v2 decoder: masks, iou, stability_scores, boxes
            masks = torch.from_numpy(
                self._decoder_request.get_output_tensor(0).data.copy()
            )                                                   # [B,3,H,W]
            iou_preds = torch.from_numpy(
                self._decoder_request.get_output_tensor(1).data.copy()
            )                                                   # [B, 3]
            stability_scores = torch.from_numpy(
                self._decoder_request.get_output_tensor(2).data.copy()
            )                                                   # [B, 3]
            boxes = torch.from_numpy(
                self._decoder_request.get_output_tensor(3).data.copy()
            )                                                   # [B, 3, 4]

            # v2 - return masks with stability_scores + boxes
            return masks, iou_preds, (stability_scores, boxes)
        else:
            # v1 decoder: masks, iou, logits
            masks = torch.from_numpy(
                self._decoder_request.get_output_tensor(0).data.copy()
            )                                                   # [B,3,H,W]
            iou_preds = torch.from_numpy(
                self._decoder_request.get_output_tensor(1).data.copy()
            )                                                   # [B, 3]
            logits = torch.from_numpy(
                self._decoder_request.get_output_tensor(2).data.copy()
            )                                                   # [B,3,H,W]

            if return_logits:
                return logits, iou_preds, logits
            return masks, iou_preds, logits

    def get_image_embedding(self) -> torch.Tensor:
        if not self.is_image_set:
            raise RuntimeError("set_image() must be called first.")
        return torch.from_numpy(np.array(self.features))

    def reset_image(self) -> None:
        self.is_image_set = False
        self.features = None
        self.original_size = None
        self.input_size = None


class OVCustomDINOv2Ext:
    """Extended OV DINOv2 using custom-op IR.

    Uses dinov2_vitl14_ext: DINOv2PatchPostprocess baked in.
    Input: images[N,3,224,224] + proposal_masks[N,1,224,224]
    -> Output: cls_token[N,1024] + patch_features[N,256,1024]
    """

    def __init__(
        self,
        dinov2_compiled: ov.CompiledModel,
        model_name: str = "dinov2_vitl14",
        image_size: int = 224,
        chunk_size: int = 16,
        descriptor_width_size: int = 640,
        patch_size: int = 14,
        validpatch_thresh: float = 0.5,
    ):
        self.dinov2_compiled = dinov2_compiled
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.patch_size = patch_size
        self.proposal_size = image_size
        self.descriptor_width_size = descriptor_width_size
        self.validpatch_thresh = validpatch_thresh

        self.model = type(
            "_DummyModel", (), {"device": torch.device("cpu")}
        )()

        self.rgb_normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        self.rgb_proposal_processor = CropResizePad(self.proposal_size)
        self._dinov2_request = dinov2_compiled.create_infer_request()
        print("OVCustomDINOv2Ext initialised")

    def _ov_forward_ext(
        self, images_tensor: torch.Tensor, masks_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run DINOv2 ext model -> (cls_token, patch_features)."""
        images_np = images_tensor.cpu().float().numpy()
        masks_np = masks_tensor.cpu().float().numpy()
        if masks_np.ndim == 3:
            masks_np = masks_np[:, np.newaxis, :, :]

        self._dinov2_request.infer([images_np, masks_np])
        cls_token = torch.from_numpy(
            self._dinov2_request.get_output_tensor(0).data.copy()
        )
        patch_features = torch.from_numpy(
            self._dinov2_request.get_output_tensor(1).data.copy()
        )

        return cls_token, patch_features

    def _ov_forward(
        self, images_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run DINOv2 ext model for cls_token only (dummy masks)."""
        images_np = images_tensor.cpu().float().numpy()
        N = images_np.shape[0]
        # For cls_token only, pass all-ones masks
        dummy_masks = np.ones(
            (N, 1, self.proposal_size, self.proposal_size), dtype=np.float32)
        self._dinov2_request.infer([images_np, dummy_masks])
        cls_token = torch.from_numpy(
            self._dinov2_request.get_output_tensor(0).data.copy()
        )

        return cls_token, None

    def process_rgb_proposals(
        self, image_np: np.ndarray, masks: torch.Tensor, boxes: torch.Tensor
    ) -> torch.Tensor:
        num_proposals = len(masks)
        rgb = self.rgb_normalize(image_np).float()
        rgbs = rgb.unsqueeze(0).repeat(num_proposals, 1, 1, 1)
        masked_rgbs = rgbs * masks.unsqueeze(1)

        return self.rgb_proposal_processor(masked_rgbs, boxes)

    def process_masks_proposals(
        self, masks: torch.Tensor, boxes: torch.Tensor
    ) -> torch.Tensor:
        masks.unsqueeze_(1)
        processed = self.rgb_proposal_processor(masks, boxes).squeeze_()

        return processed

    def process_rgb_and_mask_proposals(
        self, image_np: np.ndarray, masks: torch.Tensor, boxes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rgb = self.rgb_normalize(image_np).float()
        target_h, target_w = self.proposal_size, self.proposal_size
        processed_rgbs = []
        processed_masks = []

        for mask, box in zip(masks, boxes):
            x0, y0, x1, y1 = box.long().tolist()
            mask_crop = mask[y0:y1, x0:x1].float().unsqueeze(0)
            rgb_crop = rgb[:, y0:y1, x0:x1] * mask_crop

            crop_h, crop_w = mask_crop.shape[-2:]
            scale = max(target_h, target_w) / max(crop_h, crop_w)
            rgb_crop = F.interpolate(
                rgb_crop.unsqueeze(0), scale_factor=scale)[0]
            mask_crop = F.interpolate(
                mask_crop.unsqueeze(0), scale_factor=scale,
                mode="nearest")[0]

            crop_h, crop_w = rgb_crop.shape[-2:]
            pad_top = max((target_h - crop_h) // 2, 0)
            pad_bottom = max(target_h - crop_h - pad_top, 0)
            pad_left = max((target_w - crop_w) // 2, 0)
            pad_right = max(target_w - crop_w - pad_left, 0)
            if pad_top or pad_bottom or pad_left or pad_right:
                padding = (pad_left, pad_right, pad_top, pad_bottom)
                rgb_crop = F.pad(rgb_crop, padding)
                mask_crop = F.pad(mask_crop, padding)

            if rgb_crop.shape[-1] != target_w or rgb_crop.shape[-2] != target_h:
                final_scale = target_h / rgb_crop.shape[1]
                rgb_crop = F.interpolate(
                    rgb_crop.unsqueeze(0), scale_factor=final_scale)[0]
                mask_crop = F.interpolate(
                    mask_crop.unsqueeze(0), scale_factor=final_scale,
                    mode="nearest")[0]

            processed_rgbs.append(rgb_crop)
            processed_masks.append(mask_crop.squeeze(0))

        return torch.stack(processed_rgbs), torch.stack(processed_masks)

    @torch.no_grad()
    def compute_features(
        self, images: torch.Tensor, token_name: str
    ) -> torch.Tensor:
        if token_name == "x_norm_clstoken":
            if images.shape[0] > self.chunk_size:
                return self.forward_by_chunk(images)
            cls_token, _ = self._ov_forward(images)
            return cls_token
        raise NotImplementedError(f"token_name={token_name!r} not supported")

    @torch.no_grad()
    def forward_by_chunk(self, processed_rgbs: torch.Tensor) -> torch.Tensor:
        batch_rgbs = BatchedData(batch_size=self.chunk_size, data=processed_rgbs)
        del processed_rgbs

        features = BatchedData(batch_size=self.chunk_size)
        for idx_batch in range(len(batch_rgbs)):
            feats = self.compute_features(
                batch_rgbs[idx_batch], token_name="x_norm_clstoken"
            )
            features.cat(feats)

        return features.data

    @torch.no_grad()
    def compute_masked_patch_feature(
        self, images: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        """Uses the extended model - patch postprocess is baked into IR."""
        if images.shape[0] > self.chunk_size:
            return self.forward_by_chunk_v2(images, masks)
        _, patch_features = self._ov_forward_ext(images, masks)

        return patch_features

    @torch.no_grad()
    def forward_by_chunk_v2(
        self, processed_rgbs: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        batch_rgbs = BatchedData(batch_size=self.chunk_size, data=processed_rgbs)
        batch_masks = BatchedData(batch_size=self.chunk_size, data=masks)
        del processed_rgbs, masks

        features = BatchedData(batch_size=self.chunk_size)
        for idx_batch in range(len(batch_rgbs)):
            feats = self.compute_masked_patch_feature(
                batch_rgbs[idx_batch], batch_masks[idx_batch]
            )
            features.cat(feats)

        return features.data

    @torch.no_grad()
    def compute_cls_and_patch_features(
        self, images: torch.Tensor, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cls_token, patch_features = self._ov_forward_ext(images, masks)

        return cls_token, patch_features

    @torch.no_grad()
    def forward(
        self, image_np: np.ndarray, proposals: Detections
    ) -> tuple[torch.Tensor, torch.Tensor]:
        processed_rgbs, processed_masks = self.process_rgb_and_mask_proposals(
            image_np, proposals.masks, proposals.boxes
        )

        batch_rgbs = BatchedData(
            batch_size=self.chunk_size, data=processed_rgbs
        )
        batch_masks = BatchedData(
            batch_size=self.chunk_size, data=processed_masks
        )
        del processed_rgbs, processed_masks

        cls_features = BatchedData(batch_size=self.chunk_size)
        patch_features = BatchedData(batch_size=self.chunk_size)
        for idx_batch in range(len(batch_rgbs)):
            cls_feats, patch_feats = self.compute_cls_and_patch_features(
                batch_rgbs[idx_batch], batch_masks[idx_batch]
            )
            cls_features.cat(cls_feats)
            patch_features.cat(patch_feats)

        return cls_features.data, patch_features.data

    def __call__(self, image_np, proposals):
        return self.forward(image_np, proposals)


# ===================================================================
#  OVFastSAM
# ===================================================================

class OVFastSAM:
    """OpenVINO-based FastSAM mask generator.
    """

    def __init__(self, compiled_model, conf=0.25, iou=0.9, max_det=200,
                 segmentor_width_size=640):
        self.compiled = compiled_model
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.segmentor_width_size = segmentor_width_size
        self.stride = 32  # YOLOv8 stride

        # Import ultralytics post-processing ops
        from ultralytics.yolo.utils.ops import (
            non_max_suppression,
            process_mask,
            scale_boxes,
        )
        from ultralytics.yolo.data.augment import LetterBox
        self._nms = non_max_suppression
        self._process_mask = process_mask
        self._scale_boxes = scale_boxes
        self._letterbox_fn = LetterBox(
            (segmentor_width_size, segmentor_width_size),
            auto=True, stride=self.stride)

    def _preprocess(self, image_rgb):
        """Preprocess: auto-letterbox -> HWC->CHW -> /255 -> float32.

        Uses ``auto=True`` LetterBox (matching CUDA SegmentationPredictor)
        to produce minimal stride-aligned padding.
        """
        img = self._letterbox_fn(image=image_rgb)
        # HWC->CHW, contiguous, float32, normalize
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        img = np.ascontiguousarray(img)[np.newaxis]  # [1, 3, H, W]
        return img

    def generate_masks(self, image):
        """Run FastSAM on an RGB image.

        Args:
            image: np.ndarray [H, W, 3] uint8 RGB

        Returns:
            dict with 'masks' (Tensor [N, H, W]) and 'boxes' (Tensor [N, 4])
        """
        orig_shape = image.shape[:2]  # (H, W)

        # Preprocess
        img_tensor = self._preprocess(image)
        model_h, model_w = img_tensor.shape[2], img_tensor.shape[3]

        # OV inference
        result = self.compiled(img_tensor)
        output0 = np.array(
            result[self.compiled.output(0)], copy=True
        )  # [1, nc+4+nm, anchors]
        output1 = np.array(
            result[self.compiled.output(1)], copy=True
        )  # [1, 32, mh, mw]

        # NMS
        preds = self._nms(
            torch.from_numpy(output0),
            conf_thres=self.conf,
            iou_thres=self.iou,
            max_det=self.max_det,
            nc=1,
            max_time_img=10.0,
        )
        pred = preds[0]  # [N, 6+32] = [N, 38]

        if len(pred) == 0:
            return {
                "masks": torch.zeros((0, orig_shape[0], orig_shape[1])),
                "boxes": torch.zeros((0, 4)),
            }

        # Proto masks
        proto = torch.from_numpy(output1[0])  # [32, mh, mw]
        img_shape = (model_h, model_w)
        masks = self._process_mask(
            proto, pred[:, 6:], pred[:, :4], img_shape, upsample=True)

        # Scale boxes back to original image size
        boxes = pred[:, :4].clone()
        self._scale_boxes(img_shape, boxes, orig_shape)

        # Resize masks to original size
        masks = torch.nn.functional.interpolate(
            masks.unsqueeze(1).float(),
            size=(orig_shape[0], orig_shape[1]),
            mode="bilinear", align_corners=False,
        )[:, 0, :, :]

        return {
            "masks": masks,
            "boxes": boxes,
        }


# ===================================================================
#  init_model_ov
# ===================================================================

def init_model_ov(
    ov_model_dir: str,
    ov_device: str = "CPU",
    segmentor_model_name: str = "sam",
    stability_score_thresh: float = 0.85,
    points_per_side: int = 32,
    points_per_batch: int | None = None,
    chunk_size: int | None = None,
    precision: str = "fp16",
):
    """Load OV IR models and construct the full ISM pipeline.

    Returns 5-tuple:
        (model, cfg, device, selected_poses, proposal_processor)
    """
    from hydra import initialize, compose
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from utils.poses.pose_utils import (
        get_obj_poses_from_template_level,
        load_index_level_in_level2,
    )

    # -- Load Hydra config (needed for scoring parameters) ----------
    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="run_inference.yaml")

    if segmentor_model_name == "sam":
        GlobalHydra.instance().clear()
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name="ISM_sam.yaml")
        cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
        if points_per_batch is not None:
            cfg.model.segmentor_model.points_per_batch = int(points_per_batch)
    elif segmentor_model_name == "fastsam":
        GlobalHydra.instance().clear()
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name="ISM_fastsam.yaml")
    else:
        raise ValueError(
            f"OV export currently supports segmentor_model='sam' or 'fastsam', "
            f"got {segmentor_model_name!r}"
        )
    if chunk_size is not None:
        cfg.model.descriptor_model.chunk_size = int(chunk_size)

    # -- Compile OV models ------------------------------------------
    core = ov.Core()

    print(
        f"Compiling OV models from {ov_model_dir} on device={ov_device}"
    )

    def _compile(core, xml_path, device):
        model_ir = core.read_model(xml_path)
        if device.upper().startswith("GPU"):
            _name = os.path.basename(xml_path)
            if precision == "fp32":
                # Force all models to FP32 for maximum accuracy
                _prec = "f32"
            else:
                _prec = (
                    "f32" if "sam_mask_decoder" in _name or "fastsam" in _name
                    else "f16"
                )
            config = {"PERFORMANCE_HINT": "LATENCY",
                      "NUM_STREAMS": "1",
                      "INFERENCE_PRECISION_HINT": _prec}
            if precision == "fp32":
                config["EXECUTION_MODE_HINT"] = "ACCURACY"
            _cache_dir = os.path.join(ov_model_dir, ".ov_gpu_cache")
            os.makedirs(_cache_dir, exist_ok=True)
            config["CACHE_DIR"] = _cache_dir
        else:
            config = {}

        print(f"Compiling Model: {xml_path}, "
              f"Precision: {config.get('INFERENCE_PRECISION_HINT', 'default')}, "
              f"Device: {device}")

        return core.compile_model(model_ir, device, config)

    if segmentor_model_name == "sam":
        sam_encoder_xml = os.path.join(ov_model_dir, "sam_image_encoder.xml")
        sam_decoder_xml = os.path.join(ov_model_dir, "sam_mask_decoder.xml")
        for p in (sam_encoder_xml, sam_decoder_xml):
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"OV IR not found: {p}  (run export_ism.py first)")
        encoder_compiled = _compile(core, sam_encoder_xml, ov_device)
        decoder_compiled = _compile(core, sam_decoder_xml, ov_device)
    else:  # fastsam
        fastsam_xml = os.path.join(ov_model_dir, "fastsam_x_dynamic.xml")
        if not os.path.isfile(fastsam_xml):
            raise FileNotFoundError(
                f"OV IR not found: {fastsam_xml}  "
                f"(run export_ism.py --only fastsam first)")
        fastsam_compiled = _compile(core, fastsam_xml, ov_device)

    dinov2_xml = os.path.join(ov_model_dir, "dinov2_vitl14.xml")
    if not os.path.isfile(dinov2_xml):
        raise FileNotFoundError(
            f"OV IR not found: {dinov2_xml}  (run export_ism.py first)")
    dinov2_compiled = _compile(core, dinov2_xml, ov_device)

    # Build OV mask generator
    if segmentor_model_name == "sam":
        ov_predictor = OVSamPredictor(encoder_compiled, decoder_compiled)
        ov_predictor.ov_core = core

        seg_cfg = cfg.model.segmentor_model
        ov_mask_generator = OVCustomSamAutomaticMaskGenerator(
            predictor=ov_predictor,
            points_per_side=points_per_side,
            points_per_batch=seg_cfg.points_per_batch,
            pred_iou_thresh=seg_cfg.pred_iou_thresh,
            stability_score_thresh=seg_cfg.stability_score_thresh,
            box_nms_thresh=seg_cfg.box_nms_thresh,
            min_mask_region_area=seg_cfg.min_mask_region_area,
            crop_overlap_ratio=(
                seg_cfg.crop_overlap_ratio
                if seg_cfg.crop_overlap_ratio is not None
                else 512 / 1500
            ),
            segmentor_width_size=seg_cfg.segmentor_width_size,
            ov_device=ov_device,
        )
    else:  # fastsam
        seg_cfg = cfg.model.segmentor_model
        ov_mask_generator = OVFastSAM(
            compiled_model=fastsam_compiled,
            conf=0.25,  # matches CUDA CustomYOLO override
            iou=seg_cfg.config.iou_threshold,
            max_det=getattr(seg_cfg.config, 'max_det', 200),
            segmentor_width_size=cfg.model.segmentor_width_size,
        )

    # Build OV DINOv2 descriptor model
    desc_cfg = cfg.model.descriptor_model
    ov_dinov2 = OVCustomDINOv2(
        dinov2_compiled=dinov2_compiled,
        model_name=desc_cfg.model_name,
        image_size=desc_cfg.image_size,
        chunk_size=desc_cfg.chunk_size,
        descriptor_width_size=desc_cfg.descriptor_width_size,
        patch_size=14,
        validpatch_thresh=desc_cfg.validpatch_thresh,
    )

    # Build the Instance_Segmentation_Model container
    from model.detector import Instance_Segmentation_Model

    model = Instance_Segmentation_Model(
        segmentor_model=ov_mask_generator,
        descriptor_model=ov_dinov2,
        onboarding_config=cfg.model.onboarding_config,
        matching_config=instantiate(cfg.model.matching_config),
        post_processing_config=cfg.model.post_processing_config,
        log_interval=cfg.model.log_interval,
        log_dir=cfg.model.log_dir,
        visible_thred=cfg.model.visible_thred,
        pointcloud_sample_num=cfg.model.pointcloud_sample_num,
    )

    # Torch device for pre/post-processing tensors
    torch_device = torch.device("cpu")

    # -- Pre-compute template pose table ----------------------------
    template_poses = get_obj_poses_from_template_level(
        level=2, pose_distribution="all"
    )
    template_poses[:, :3, 3] *= 0.4
    poses = torch.tensor(template_poses).to(torch.float32).to(torch_device)
    selected_poses = poses[load_index_level_in_level2(0, "all"), :, :]

    proposal_processor = CropResizePad(224)

    print(
        f"OV ISM pipeline ready  (device={ov_device})"
    )
    return model, cfg, torch_device, selected_poses, proposal_processor


def init_model_ov_ext(
    ov_model_dir: str,
    ov_device: str = "CPU",
    segmentor_model_name: str = "sam",
    stability_score_thresh: float = 0.85,
    points_per_side: int = 32,
    points_per_batch: int | None = None,
    chunk_size: int | None = None,
    precision: str = "fp16",
):
    """Load extended OV IR models and construct ISM pipeline.

    Uses *_ext.xml models.
    """
    from hydra import initialize, compose
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from utils.poses.pose_utils import (
        get_obj_poses_from_template_level,
        load_index_level_in_level2,
    )

    # Load Hydra config
    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="run_inference.yaml")

    if segmentor_model_name == "sam":
        GlobalHydra.instance().clear()
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name="ISM_sam.yaml")
        cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
        if points_per_batch is not None:
            cfg.model.segmentor_model.points_per_batch = int(points_per_batch)
    elif segmentor_model_name == "fastsam":
        GlobalHydra.instance().clear()
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name="ISM_fastsam.yaml")
    else:
        raise ValueError(f"Only segmentor_model='sam' or 'fastsam' supported, got {segmentor_model_name!r}")
    if chunk_size is not None:
        cfg.model.descriptor_model.chunk_size = int(chunk_size)

    core = ov.Core()

    def _compile(core, xml_path, device, model_ir=None):
        if model_ir is None:
            model_ir = core.read_model(xml_path)

        if device.upper().startswith("GPU"):
            _name = os.path.basename(xml_path)
            if precision == "fp32":
                _prec = "f32"
            else:
                _prec = (
                    "f32" if "fastsam" in _name
                    else "f16"
                )
            config = {"PERFORMANCE_HINT": "LATENCY",
                      "NUM_STREAMS": "1",
                      "INFERENCE_PRECISION_HINT": _prec}
            if precision == "fp32":
                config["EXECUTION_MODE_HINT"] = "ACCURACY"
            _cache_dir = os.path.join(ov_model_dir, ".ov_gpu_cache")
            os.makedirs(_cache_dir, exist_ok=True)
            config["CACHE_DIR"] = _cache_dir
        else:
            config = {}

        print(f"Compiling Model: {xml_path}, "
              f"Precision: {config.get('INFERENCE_PRECISION_HINT', 'default')}, "
              f"Device: {device}")

        return core.compile_model(model_ir, device, config)

    # Compile extended OV models
    print(f"Compiling extended OV models from {ov_model_dir} on {ov_device}")

    if segmentor_model_name == "sam":
        sam_encoder_xml = os.path.join(ov_model_dir, "sam_image_encoder_ext.xml")
        # Prefer v2 decoder
        sam_decoder_v2_xml = os.path.join(ov_model_dir, "sam_mask_decoder_ext_v2.xml")
        sam_decoder_v1_xml = os.path.join(ov_model_dir, "sam_mask_decoder_ext.xml")
        if os.path.isfile(sam_decoder_v2_xml):
            sam_decoder_xml = sam_decoder_v2_xml
            print("Using v2 decoder")
        else:
            sam_decoder_xml = sam_decoder_v1_xml
            print("Using v1 decoder")

        for p in (sam_encoder_xml, sam_decoder_xml):
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"Extended OV IR not found: {p}  "
                    f"(run export_ism.py --ext first)")

        encoder_compiled = _compile(core, sam_encoder_xml, ov_device)

        _ppb = int(cfg.model.segmentor_model.points_per_batch)
        _dec_ir = core.read_model(sam_decoder_xml)
        _static_shapes = {}
        for _inp in _dec_ir.inputs:
            _shape = []
            for _i, _dim in enumerate(_inp.partial_shape):
                if not _dim.is_dynamic:
                    _shape.append(_dim.get_length())
                elif _i == 0:
                    _shape.append(_ppb)   # batch dim → points_per_batch
                else:
                    _shape.append(1)      # remaining dynamic → num_points_per_prompt
            _static_shapes[_inp.any_name] = _shape
        _dec_ir.reshape(_static_shapes)
        print(f"Decoder static shapes={_static_shapes} (reshaped for GPU performance)")
        decoder_compiled = _compile(core, sam_decoder_xml, ov_device, model_ir=_dec_ir)
    else:  # fastsam
        fastsam_xml = os.path.join(ov_model_dir, "fastsam_x_dynamic.xml")
        if not os.path.isfile(fastsam_xml):
            raise FileNotFoundError(
                f"OV IR not found: {fastsam_xml}  "
                f"(run export_ism.py --only fastsam first)")
        fastsam_compiled = _compile(core, fastsam_xml, ov_device)

    dinov2_xml = os.path.join(ov_model_dir, "dinov2_vitl14_ext.xml")
    if not os.path.isfile(dinov2_xml):
        raise FileNotFoundError(
            f"Extended OV IR not found: {dinov2_xml}  "
            f"(run export_ism.py --ext first)")
    dinov2_compiled = _compile(core, dinov2_xml, ov_device)

    # Build mask generator
    if segmentor_model_name == "sam":
        ov_predictor = OVSamPredictorExt(encoder_compiled, decoder_compiled)
        seg_cfg = cfg.model.segmentor_model
        ov_mask_generator = OVCustomSamAutomaticMaskGenerator(
            predictor=ov_predictor,
            points_per_side=points_per_side,
            points_per_batch=seg_cfg.points_per_batch,
            pred_iou_thresh=seg_cfg.pred_iou_thresh,
            stability_score_thresh=seg_cfg.stability_score_thresh,
            box_nms_thresh=seg_cfg.box_nms_thresh,
            min_mask_region_area=seg_cfg.min_mask_region_area,
            crop_overlap_ratio=(
                seg_cfg.crop_overlap_ratio
                if seg_cfg.crop_overlap_ratio is not None
                else 512 / 1500
            ),
            segmentor_width_size=seg_cfg.segmentor_width_size,
            ov_device=ov_device,
        )
    else:  # fastsam
        seg_cfg = cfg.model.segmentor_model
        ov_mask_generator = OVFastSAM(
            compiled_model=fastsam_compiled,
            conf=0.25,
            iou=seg_cfg.config.iou_threshold,
            max_det=getattr(seg_cfg.config, 'max_det', 200),
            segmentor_width_size=cfg.model.segmentor_width_size,
        )

    # Build extended DINOv2 descriptor model
    desc_cfg = cfg.model.descriptor_model
    ov_dinov2 = OVCustomDINOv2Ext(
        dinov2_compiled=dinov2_compiled,
        model_name=desc_cfg.model_name,
        image_size=desc_cfg.image_size,
        chunk_size=desc_cfg.chunk_size,
        descriptor_width_size=desc_cfg.descriptor_width_size,
        patch_size=14,
        validpatch_thresh=desc_cfg.validpatch_thresh,
    )

    # Build Instance_Segmentation_Model container
    from model.detector import Instance_Segmentation_Model

    model = Instance_Segmentation_Model(
        segmentor_model=ov_mask_generator,
        descriptor_model=ov_dinov2,
        onboarding_config=cfg.model.onboarding_config,
        matching_config=instantiate(cfg.model.matching_config),
        post_processing_config=cfg.model.post_processing_config,
        log_interval=cfg.model.log_interval,
        log_dir=cfg.model.log_dir,
        visible_thred=cfg.model.visible_thred,
        pointcloud_sample_num=cfg.model.pointcloud_sample_num,
    )

    torch_device = torch.device("cpu")

    template_poses = get_obj_poses_from_template_level(
        level=2, pose_distribution="all")
    template_poses[:, :3, 3] *= 0.4
    poses = torch.tensor(template_poses).to(torch.float32).to(torch_device)
    selected_poses = poses[load_index_level_in_level2(0, "all"), :, :]

    proposal_processor = CropResizePad(224)

    print(
        f"OV ISM ext pipeline ready  (device={ov_device})")

    return model, cfg, torch_device, selected_poses, proposal_processor


# ===================================================================
#  Standalone test
# ===================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="OV ISM inference - single-image pipeline"
    )
    parser.add_argument(
        "--ov_model_dir",
        default=os.path.join(BASE_DIR, "checkpoints/ov_models"),
        help="Directory containing OV IR files",
    )
    parser.add_argument(
        "--ov_device",
        default="CPU",
        help="OpenVINO device: CPU or GPU",
    )
    parser.add_argument(
        "--ext", action=argparse.BooleanOptionalAction, default=True,
        help="Use extended models. Default: auto-enable when available.",
    )
    parser.add_argument(
        "--points_per_batch", type=int, default=64,
        help="SAM decoder: point prompts per inference call "
             "default: 64",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=16,
        help="DINOv2: proposal crops per inference call "
             "default: 16",
    )
    parser.add_argument(
        "--image", default=None,
        help="Path to RGB image for single-image ISM inference "
             "(e.g. SAM-6D/Data/Example/rgb.png)",
    )
    parser.add_argument(
        "--cad", default=None,
        help="Path to CAD model (.ply) for single-image ISM inference "
             "(e.g. SAM-6D/Data/Example/obj_000005.ply)",
    )
    parser.add_argument(
        "--templates_dir", default=None,
        help="Path to pre-rendered templates directory (rgb_N.png + mask_N.png + N.npy). "
             "Auto-searched under <image_dir>/outputs/templates/ if not given.",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Output directory for single-image results "
             "(default: <image_dir>/output/results_ov_gpu/ism)",
    )
    parser.add_argument(
        "--stability_score_thresh", type=float, default=0.85,
        help="SAM stability_score_thresh (default: 0.85)",
    )
    parser.add_argument(
        "--segmentor_model", default="sam",
        choices=["sam", "fastsam"],
        help="Segmentor model: sam (SAM ViT-H) or fastsam (YOLOv8x-seg)",
    )
    parser.add_argument(
        "--points_per_side", type=int, default=32,
        help="SAM grid size: NxN point prompts (default: 32)",
    )
    parser.add_argument(
        "--gt_mask", type=str, default=None,
        help="Path to GT mask PNG(s) for mAP@IoU[0.50:0.95] evaluation. "
             "Single file or dir of mask PNGs (mask_visib format).",
    )
    parser.add_argument(
        "--precision", default=os.environ.get("OV_PRECISION", "fp16"),
        choices=["fp32", "fp16"],
        help="Inference precision: fp32 (full accuracy) or fp16 (faster). "
             "Default: OV_PRECISION env var or fp16.",
    )
    args = parser.parse_args()

    # -- random seeds for reproducibility -----------------------
    np.random.seed(42)
    torch.manual_seed(42)

    core = ov.Core()
    print(f"Available OV devices: {core.available_devices}")

    if args.image:
        # -- Single-image ISM inference -----------------------------
        if not os.path.isfile(args.image):
            print(f"[ERROR] Image not found: {args.image}")
            sys.exit(1)

        output_dir = args.output_dir or os.path.join(
            os.path.dirname(args.image), "output", "results_ov_gpu", "ism")
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n=== Single-Image ISM Pipeline ===")
        print(f"  Image:  {args.image}")
        print(f"  Device: {args.ov_device}, Ext: {args.ext}, Precision: {args.precision}")
        print(f"  Output: {output_dir}")

        # Load image
        rgb = np.array(Image.open(args.image).convert("RGB"))
        H, W = rgb.shape[:2]
        print(f"  Image shape: {H}x{W}x3")

        # Init pipeline
        if args.ext:
            model, cfg, device, selected_poses, proposal_processor = \
                init_model_ov_ext(
                    ov_model_dir=args.ov_model_dir,
                    ov_device=args.ov_device,
                    segmentor_model_name=args.segmentor_model,
                    stability_score_thresh=args.stability_score_thresh,
                    points_per_side=args.points_per_side,
                    points_per_batch=args.points_per_batch,
                    chunk_size=args.chunk_size,
                    precision=args.precision,
                )
        else:
            model, cfg, device, selected_poses, proposal_processor = \
                init_model_ov(
                    ov_model_dir=args.ov_model_dir,
                    ov_device=args.ov_device,
                    segmentor_model_name=args.segmentor_model,
                    stability_score_thresh=args.stability_score_thresh,
                    points_per_side=args.points_per_side,
                    points_per_batch=args.points_per_batch,
                    chunk_size=args.chunk_size,
                    precision=args.precision,
                )

        # Print pipeline config
        seg_cfg = cfg.model.segmentor_model
        desc_cfg = cfg.model.descriptor_model

        print(f"\n  Pipeline config:")
        if args.segmentor_model == "sam":
            pts_per_side = args.points_per_side
            pts_per_batch = getattr(seg_cfg, 'points_per_batch', 64)
            chunk_sz = getattr(desc_cfg, 'chunk_size', 16)
            n_grid_points = pts_per_side * pts_per_side
            n_decoder_calls = (n_grid_points + pts_per_batch - 1) // pts_per_batch
            print(f"    points_per_side:       {pts_per_side} "
                  f"({n_grid_points} grid points)")
            print(f"    points_per_batch:      {pts_per_batch} "
                  f"(-> {n_decoder_calls} SAM decoder calls)")
            print(f"    DINOv2 chunk_size:     {chunk_sz}")
            print(f"    pred_iou_thresh:       {seg_cfg.pred_iou_thresh}")
            print(f"    stability_score_thresh: {seg_cfg.stability_score_thresh}")
            print(f"    box_nms_thresh:        {seg_cfg.box_nms_thresh}")
        else:
            print(f"    segmentor:             fastsam (YOLOv8x-seg)")
            print(f"    conf:                  0.25")
            print(f"    iou_threshold:         {seg_cfg.config.iou_threshold}")
            print(f"    segmentor_width_size:  {cfg.model.segmentor_width_size}")

        # Step 1: Segmentation
        print(f"\n  [Step 1] {'SAM' if args.segmentor_model == 'sam' else 'FastSAM'} Segmentation")

        detections = model.segmentor_model.generate_masks(rgb)
        detections = Detections(detections)
        n_masks = len(detections.masks)

        if args.segmentor_model == "sam":
            raw_candidates = n_grid_points * 3
            print(f"    SAM encoder:     1 call (image {H}x{W})")
            print(f"    SAM decoder:     {n_decoder_calls} calls "
                  f"(B={pts_per_batch} each)")
            print(f"    Raw candidates:  {raw_candidates} "
                  f"({n_grid_points} pts x 3 masks)")
        print(f"    After filter+NMS: {n_masks} proposals")

        # Step 2: DINOv2 descriptor extraction
        print(f"\n  [Step 2] DINOv2 Query Descriptors")

        query_cls, query_patch = model.descriptor_model(rgb, detections)

        chunk_sz = getattr(desc_cfg, 'chunk_size', 16)
        n_dino_calls = (n_masks + chunk_sz - 1) // chunk_sz
        print(f"    Proposals: {n_masks} -> {n_dino_calls} DINOv2 calls "
              f"(chunk_size={chunk_sz})")
        print(f"    cls_token:      {tuple(query_cls.shape)}")
        print(f"    patch_features: {tuple(query_patch.shape)}")

        # Save outputs
        # 1. Mask visualization
        vis = rgb.copy()
        masks_np = detections.masks.cpu().numpy() if torch.is_tensor(
            detections.masks) else np.array(detections.masks)

        for i in range(min(n_masks, 50)):
            color = np.random.randint(60, 255, 3).tolist()
            mask = masks_np[i].squeeze().astype(bool)
            vis[mask] = (vis[mask] * 0.4 + np.array(color) * 0.6).astype(
                np.uint8)

        vis_path = os.path.join(output_dir, "segmentation_masks.png")
        cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        # (boxes_np needed for top-detection visualisation in step 3)
        boxes_np = detections.boxes.cpu().numpy() if torch.is_tensor(
            detections.boxes) else np.array(detections.boxes)

        # Step 3: Template matching + scoring (if --cad provided)
        if args.cad and (os.path.isfile(args.cad) or os.path.isdir(args.cad)):
            def _batch_input_data(depth_path, cam_path, device):
                with open(cam_path) as _f:
                    cam_info = json.load(_f)

                depth = np.array(imageio.imread(depth_path)).astype(np.int32)
                cam_K = np.array(cam_info['cam_K']).reshape((3, 3))
                depth_scale = np.array(cam_info['depth_scale'])

                return {
                    "depth": torch.from_numpy(depth).unsqueeze(0).to(device),
                    "cam_intrinsic": torch.from_numpy(cam_K).unsqueeze(0).to(device),
                    "depth_scale": torch.from_numpy(depth_scale).unsqueeze(0).to(device),
                }


            def _sample_mesh_surface(verts, faces, n_pts):
                v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
                areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
                probs = areas / areas.sum()
                tri_idx = np.random.choice(len(faces), size=n_pts, p=probs)
                r1 = np.sqrt(np.random.rand(n_pts, 1))
                r2 = np.random.rand(n_pts, 1)

                return ((1 - r1) * verts[faces[tri_idx, 0]]
                        + r1 * (1 - r2) * verts[faces[tri_idx, 1]]
                        + r1 * r2 * verts[faces[tri_idx, 2]])


            def _load_object_pointcloud(model, cad_path, selected_poses, device):
                model.ref_data["poses"] = selected_poses
                ply = PlyData.read(cad_path)
                verts = np.column_stack([ply["vertex"]["x"],
                                         ply["vertex"]["y"],
                                         ply["vertex"]["z"]]).astype(np.float64)
                faces = np.vstack(ply["face"]["vertex_indices"])
                pts = _sample_mesh_surface(verts, faces, 2048).astype(np.float32) / 1000.0
                model.ref_data["pointcloud"] = (
                    torch.tensor(pts).unsqueeze(0).data.to(device)
                )

            # Locate templates dir
            image_dir = os.path.dirname(args.image)
            candidates = [
                args.templates_dir,
                os.path.join(output_dir, "templates"),
                os.path.join(image_dir, "outputs", "templates"),
                os.path.join(image_dir, "templates"),
            ]
            template_dir = None
            for c in candidates:
                if c and os.path.isdir(c) and \
                   len(glob.glob(os.path.join(c, "*.npy"))) > 0:
                    template_dir = c
                    break

            if template_dir is None:
                print(f"\n  [Step 3] Skipping scoring - no templates found.")
                print(f"    Searched: {[c for c in candidates if c]}")
                print(f"    Render templates first with "
                      f"Render/render_templates_headless.py, or pass "
                      f"--templates_dir <path>")
            else:
                print(f"\n  [Step 3] Template Matching + Scoring")
                n_tpl = len(glob.glob(os.path.join(template_dir, "*.npy")))
                print(f"    Templates dir: {template_dir} ({n_tpl} templates)")

                # Load template images
                tpl_boxes, tpl_masks_list, tpl_images = [], [], []
                for idx in range(n_tpl):
                    im = Image.open(os.path.join(template_dir,
                                                 f'rgb_{idx}.png'))
                    mk = Image.open(os.path.join(template_dir,
                                                 f'mask_{idx}.png'))
                    tpl_boxes.append(mk.getbbox())
                    im_t = torch.from_numpy(
                        np.array(im.convert("RGB")) / 255).float()
                    mk_t = torch.from_numpy(
                        np.array(mk.convert("L")) / 255).float()
                    tpl_images.append(im_t * mk_t[:, :, None])
                    tpl_masks_list.append(mk_t.unsqueeze(-1))

                tpl_imgs = torch.stack(tpl_images).permute(0, 3, 1, 2)
                tpl_msks = torch.stack(tpl_masks_list).permute(0, 3, 1, 2)
                tpl_bxs = torch.tensor(np.array(tpl_boxes))
                tpl_imgs = proposal_processor(
                    images=tpl_imgs, boxes=tpl_bxs).to(device)
                tpl_msks_crop = proposal_processor(
                    images=tpl_msks, boxes=tpl_bxs).to(device)

                # compute_features (template cls)
                model.ref_data = {}
                model.ref_data["descriptors"] = \
                    model.descriptor_model.compute_features(
                        tpl_imgs, token_name="x_norm_clstoken"
                    ).unsqueeze(0).data

                # compute_masked_patch_feature (template patch)
                model.ref_data["appe_descriptors"] = \
                    model.descriptor_model.compute_masked_patch_feature(
                        tpl_imgs, tpl_msks_crop[:, 0, :, :]
                    ).unsqueeze(0).data

                # Load pointcloud + poses
                _load_object_pointcloud(model, args.cad, selected_poses, device)

                # Scoring needs depth + camera
                depth_path = os.path.join(image_dir, "depth.png")
                cam_path = os.path.join(image_dir, "camera.json")
                has_depth = (os.path.isfile(depth_path) and
                             os.path.isfile(cam_path))

                # Clone detections for scoring
                det_score = Detections({
                    key: getattr(detections, key).clone()
                    if torch.is_tensor(getattr(detections, key))
                    else getattr(detections, key)
                    for key in detections.keys
                })

                qcls_score = query_cls.clone()
                qpatch_score = query_patch.clone()

                # compute_semantic_score
                (idx_sel, pred_idx_obj, sem_score, best_tpl) = \
                    model.compute_semantic_score(qcls_score)
                det_score.filter(idx_sel)
                qpatch_sel = qpatch_score[idx_sel, :]
                n_after_sem = len(det_score)

                # compute_appearance_score
                appe_score, ref_aux = model.compute_appearance_score(
                    best_tpl, pred_idx_obj, qpatch_sel)

                if has_depth:
                    batch = _batch_input_data(depth_path, cam_path, device)

                    # project_template_to_image
                    image_uv = _project_template_to_image_ov(
                        model,
                        best_tpl, pred_idx_obj, batch, det_score.masks)

                    # compute_geometric_score
                    geo_score, vis_ratio = model.compute_geometric_score(
                        image_uv, det_score, qpatch_sel, ref_aux,
                        visible_thred=model.visible_thred)

                    # final score
                    final_score = ((sem_score + appe_score +
                                    geo_score * vis_ratio) /
                                   (1 + 1 + vis_ratio))
                    score_label = "semantic + appearance + geometric"
                else:
                    final_score = (sem_score + appe_score) / 2.0
                    score_label = "semantic + appearance (no depth)"

                # Print top-5 proposals
                scores_cpu = final_score.cpu().numpy() if torch.is_tensor(
                    final_score) else np.array(final_score)
                sem_cpu = sem_score.cpu().numpy() if torch.is_tensor(
                    sem_score) else np.array(sem_score)
                appe_cpu = appe_score.cpu().numpy() if torch.is_tensor(
                    appe_score) else np.array(appe_score)
                sort_idx = np.argsort(scores_cpu)[::-1]

                print(f"\n  Score type: {score_label}")
                print(f"  After semantic filter: {n_after_sem}/{n_masks} "
                      f"proposals")
                print(f"\n  Top-5 proposal scores:")
                print(f"  {'Rank':>4}  {'Final':>7}  {'Semantic':>9}  "
                      f"{'Appearance':>11}  {'Template':>9}")
                print(f"  {'-'*4}  {'-'*7}  {'-'*9}  {'-'*11}  {'-'*9}")
                for rank, si in enumerate(sort_idx[:5], 1):
                    tpl_id = int(best_tpl[si].item()) if torch.is_tensor(
                        best_tpl) else int(best_tpl[si])
                    print(f"  {rank:>4}  {scores_cpu[si]:>7.4f}  "
                          f"{sem_cpu[si]:>9.4f}  {appe_cpu[si]:>11.4f}  "
                          f"template_{tpl_id:>3d}")

                best_score = float(scores_cpu[sort_idx[0]])
                print(f"\n  Best match score: {best_score:.4f} "
                      f"({'CONFIDENT' if best_score > 0.5 else 'LOW CONFIDENCE'})")

                # Save detection results
                # Parse obj_id from CAD filename (e.g. obj_000005.ply -> 5)
                cad_stem = os.path.splitext(os.path.basename(args.cad))[0]
                obj_id = int(cad_stem.split("_")[-1]) if "_" in cad_stem else 1

                # Parse image_id from gt_mask path (e.g. 000003_000001.png -> 3)
                frame_id = 0
                gt_mask_src = args.gt_mask
                if not gt_mask_src:
                    img_dir = os.path.dirname(os.path.abspath(args.image))
                    mv_dir = os.path.join(img_dir, "mask_visib")
                    if os.path.isdir(mv_dir):
                        gt_mask_src = mv_dir
                if gt_mask_src:
                    if os.path.isdir(gt_mask_src):
                        fnames = sorted(f for f in os.listdir(gt_mask_src)
                                        if f.endswith(".png"))
                        if fnames:
                            frame_id = int(fnames[0].split("_")[0])
                    elif os.path.isfile(gt_mask_src):
                        frame_id = int(
                            os.path.basename(gt_mask_src).split("_")[0])

                # Save detection results
                obj_ids = torch.full((len(final_score),), obj_id - 1,
                                             dtype=torch.long)
                det_score.add_attribute("scores", final_score)
                det_score.add_attribute("object_ids", obj_ids)
                det_score.apply_nms_per_object_id(
                    nms_thresh=model.post_processing_config.nms_thresh
                )

                det_np = Detections({
                    key: getattr(det_score, key).clone()
                    if torch.is_tensor(getattr(det_score, key))
                    else getattr(det_score, key)
                    for key in det_score.keys
                })
                det_np.to_numpy()
                save_path = os.path.join(output_dir, "detection_ism")
                os.makedirs(output_dir, exist_ok=True)
                det_np.save_to_file(0, frame_id, 0, save_path, "Custom",
                                    return_results=False)

                # Convert npz -> detection_ism.json
                det_list = convert_npz_to_json(
                    idx=0, list_npz_paths=[save_path + ".npz"]
                )
                save_json_bop23(save_path + ".json", det_list)
                print(f"  detection_ism.json ({len(det_list)} detections)")

                # vis_ism.png
                if det_list:
                    best_det = max(det_list, key=lambda d: d["score"])
                    rgb_pil = Image.fromarray(rgb)
                    img_arr = np.array(rgb_pil)

                    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
                    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32)

                    bmask = rle_to_mask(best_det["segmentation"])

                    alpha_v = 0.33
                    overlay[bmask, 0] = np.clip(
                        alpha_v * 255 + (1 - alpha_v) * overlay[bmask, 0], 0, 255)
                    overlay[bmask, 1] = np.clip(
                        alpha_v * 105 + (1 - alpha_v) * overlay[bmask, 1], 0, 255)
                    overlay[bmask, 2] = np.clip(
                        alpha_v * 0   + (1 - alpha_v) * overlay[bmask, 2], 0, 255)

                    pred_pil = Image.fromarray(overlay.astype(np.uint8))
                    h, w = img_arr.shape[:2]

                    concat = Image.new("RGB", (w * 2, h))
                    concat.paste(rgb_pil, (0, 0))
                    concat.paste(pred_pil, (w, 0))

                    vis_path = os.path.join(output_dir, "vis_ism.png")
                    concat.save(vis_path)

                    print(f"  vis_ism.png")

                # Visualize top detection on image
                if len(sort_idx) > 0:
                    top_mask = masks_np[idx_sel.cpu().numpy()
                                        if torch.is_tensor(idx_sel)
                                        else idx_sel][sort_idx[0]]
                    top_mask = top_mask.squeeze().astype(bool)

                    vis_top = rgb.copy()
                    vis_top[top_mask] = (vis_top[top_mask] * 0.3 +
                                         np.array([0, 200, 50]) * 0.7
                                         ).astype(np.uint8)

                    boxes_sel = boxes_np[idx_sel.cpu().numpy()
                                         if torch.is_tensor(idx_sel)
                                         else idx_sel]

                    x1, y1, x2, y2 = map(int, boxes_sel[sort_idx[0]])

                    cv2.rectangle(vis_top, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_top, f"{best_score:.3f}", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    top_path = os.path.join(output_dir, "top_detection.png")

                    cv2.imwrite(top_path,
                                cv2.cvtColor(vis_top, cv2.COLOR_RGB2BGR))

                    print(f"  top_detection.png (score={best_score:.4f})")

                # mAP @ IoU[0.50:0.95] evaluation
                nms_masks = det_np.masks  # (N, H, W) numpy
                nms_scores = det_np.scores  # (N,) numpy
                nms_masks_bin = (nms_masks > 0.5).astype(np.uint8)
                if nms_masks_bin.ndim == 4:
                    nms_masks_bin = nms_masks_bin.squeeze(1)
                evaluate_and_print_map(
                    nms_masks_bin, nms_scores, args.gt_mask, args.image
                )

        elif args.cad:
            print(f"\n  [Step 3] Skipped - CAD file not found: {args.cad}")
        else:
            print(f"\n  [Step 3] Skipped - no --cad provided")

        print(f"\n  Outputs saved to: {output_dir}")
        print(f"    segmentation_masks.png")

    else:
        # test mode (no --image)
        print(f"Device: {args.ov_device}, Ext: {args.ext}")
        print(f"SAM decoder points_per_batch: {args.points_per_batch}")
        print(f"DINOv2 chunk_size: {args.chunk_size}")

        if args.ext:
            print("\n=== SAM Encoder Ext ===")
            enc = core.compile_model(
                core.read_model(os.path.join(args.ov_model_dir,
                                             "sam_image_encoder_ext.xml")),
                args.ov_device)
            dec = core.compile_model(
                core.read_model(os.path.join(args.ov_model_dir,
                                             "sam_mask_decoder_ext.xml")),
                args.ov_device)
            predictor = OVSamPredictorExt(enc, dec)

            dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            predictor.set_image(dummy_img)

            B = args.points_per_batch
            coords = torch.rand(B, 1, 2) * 640
            labels = torch.ones(B, 1, dtype=torch.int)
            masks, iou, logits = predictor.predict_torch(
                coords, labels, return_logits=True)
            print(f"  Encoder+Decoder ext OK  "
                  f"masks={tuple(masks.shape)} iou={tuple(iou.shape)}")
            predictor.reset_image()

            print(f"\n=== DINOv2 Ext (chunk_size={args.chunk_size}) ===")
            dino = core.compile_model(
                core.read_model(os.path.join(args.ov_model_dir,
                                             "dinov2_vitl14_ext.xml")),
                args.ov_device)
            ov_dinov2 = OVCustomDINOv2Ext(dino)
            imgs = torch.randn(args.chunk_size, 3, 224, 224)
            masks_d = torch.ones(args.chunk_size, 1, 224, 224)
            cls_t, patch_f = ov_dinov2._ov_forward_ext(imgs, masks_d)
            print(f"  DINOv2 ext OK  "
                  f"cls={tuple(cls_t.shape)} feats={tuple(patch_f.shape)}")

        else:
            print("\n=== SAM Predictor test ===")
            enc = core.compile_model(
                core.read_model(os.path.join(args.ov_model_dir,
                                             "sam_image_encoder.xml")),
                args.ov_device)
            dec = core.compile_model(
                core.read_model(os.path.join(args.ov_model_dir,
                                             "sam_mask_decoder.xml")),
                args.ov_device)
            predictor = OVSamPredictor(enc, dec)

            dummy_img = np.random.randint(0, 255, (480, 640, 3),
                                          dtype=np.uint8)
            predictor.set_image(dummy_img)

            B = args.points_per_batch
            coords = torch.rand(B, 1, 2) * 640
            labels = torch.ones(B, 1, dtype=torch.int)
            masks, iou_preds, low_res = predictor.predict_torch(
                coords, labels, multimask_output=True, return_logits=True
            )
            print(f"  Encoder+Decoder OK  "
                  f"masks={tuple(masks.shape)} iou={tuple(iou_preds.shape)}")
            predictor.reset_image()

            print("\n=== DINOv2 test ===")
            dino = core.compile_model(
                core.read_model(os.path.join(args.ov_model_dir,
                                             "dinov2_vitl14.xml")),
                args.ov_device)
            ov_dinov2 = OVCustomDINOv2(dino)

            dummy_batch = torch.randn(args.chunk_size, 3, 224, 224)
            cls_tok, patch_tok = ov_dinov2._ov_forward(dummy_batch)
            print(f"  DINOv2 OK  "
                  f"cls={tuple(cls_tok.shape)} "
                  f"patches={tuple(patch_tok.shape)}")

    print("\n=== Done ===")
