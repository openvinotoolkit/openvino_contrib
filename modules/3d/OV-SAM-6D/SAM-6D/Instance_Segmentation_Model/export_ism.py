#!/usr/bin/env python3

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Export ISM PyTorch models directly to OpenVINO IR (.xml/.bin).

cd SAM-6D/Instance_Segmentation_Model && \
ISM_BLOCK_XFORMERS=1 python3 export_ism.py \
  --ext \
  --output_dir ./checkpoints/ov_models \
  --sam_checkpoint_dir SAM-6D/Instance_Segmentation_Model/checkpoints/segment-anything \
  --dinov2_checkpoint_dir SAM-6D/Instance_Segmentation_Model/checkpoints/dinov2 \
  --fastsam_checkpoint SAM-6D/Instance_Segmentation_Model/checkpoints/FastSAM/FastSAM-x.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

import openvino as ov

from openvino import opset13 as opset
from openvino import Type as OVType
from openvino.op import Result as OVResult

# ---------------------------------------------------------------------------
# BASE_DIR
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


# ===================================================================
#  xFormers blocker - prevent import so DINOv2 falls back to standard
#  PyTorch attention and SwiGLU, which are fully ONNX/OV-exportable.
# ===================================================================
def _block_xformers():
    """
        Block xformers import BEFORE any model code is loaded.
    """
    if "xformers" in sys.modules:
        _patch_xformers_flags()
        return

    import importlib

    class _XFormersBlocker(importlib.abc.Loader):
        """Fake loader that always raises ImportError."""
        def create_module(self, spec):
            return None
        def exec_module(self, module):
            raise ImportError("xformers blocked by export_ism.py")

    class _XFormersFinder(importlib.abc.MetaPathFinder):
        def find_module(self, fullname, path=None):
            if fullname == "xformers" or fullname.startswith("xformers."):
                return _XFormersBlocker()
            return None

    sys.meta_path.insert(0, _XFormersFinder())
    print("[export] xFormers import blocked - DINOv2 will use standard ops")


def _patch_xformers_flags():
    """Patch XFORMERS_AVAILABLE=False in already-imported DINOv2 modules."""
    import importlib
    targets = [
        "model.layers.attention",
        "model.layers.swiglu_ffn",
        "model.layers.block",
    ]
    for mod_name in targets:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "XFORMERS_AVAILABLE"):
                mod.XFORMERS_AVAILABLE = False
                print(f"[export] Patched {mod_name}.XFORMERS_AVAILABLE = False")
        except ImportError:
            pass


# Block xformers BEFORE any model imports
_BLOCK_XFORMERS = os.environ.get("ISM_BLOCK_XFORMERS", "0") == "1"
if _BLOCK_XFORMERS:
    _block_xformers()
else:
    print("[export] xFormers NOT blocked (set ISM_BLOCK_XFORMERS=1 to block)")


# ===================================================================
#  Model builders
# ===================================================================

def build_sam(checkpoint_dir: str, model_type: str = "vit_h") -> "Sam":
    """Load the SAM model from checkpoint."""
    from segment_anything import sam_model_registry
    from ov_sam_layer_norm import use_ov_layer_norm2d_for_sam
    weight_map = {
        "vit_h": "sam_vit_h_4b8939.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_b": "sam_vit_b_01ec64.pth",
    }
    ckpt = os.path.join(checkpoint_dir, weight_map[model_type])
    print(f"[export] Loading SAM ({model_type}) from {ckpt}")
    print("[export] Using OpenVINO SAM LayerNorm2d")
    with use_ov_layer_norm2d_for_sam():
        sam = sam_model_registry[model_type](checkpoint=ckpt)
    sam.eval()
    return sam


def build_dinov2(checkpoint_dir: str,
                 model_name: str = "dinov2_vitl14") -> nn.Module:
    """Load the DINOv2 ViT backbone from checkpoint."""

    from model.dinov2 import _make_dinov2_model, descriptor_map

    arch = descriptor_map[model_name]
    ckpt = os.path.join(checkpoint_dir, f"{model_name}_pretrain.pth")

    print(f"[export] Loading DINOv2 ({model_name}, arch={arch}) from {ckpt}")

    model = _make_dinov2_model(arch_name=arch, pretrained=False)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    return model


# ===================================================================
#  SAM Mask Decoder wrapper
# ===================================================================

class SAMMaskDecoderWrapper(nn.Module):
    """Wraps prompt_encoder + mask_decoder into a single traceable module.

    Inputs:
        image_embed : float32 [1, 256, 64, 64]  - from SAM image encoder
        point_coords: float32 [B, N, 2]         - transformed point prompts
        point_labels: int32   [B, N]            - 1 = foreground

    Outputs:
        masks       : float32 [B, 3, 256, 256]   - low-res logit masks
        iou_preds   : float32 [B, 3]             - predicted IoU scores
    """

    def __init__(self, sam_model: "Sam"):
        super().__init__()
        self.prompt_encoder = sam_model.prompt_encoder
        self.mask_decoder = sam_model.mask_decoder

    def forward(self, image_embed, point_coords, point_labels):
        pe = self.prompt_encoder
        B = point_coords.shape[0]

        # Pad: add a dummy point with label=-1 (boxes=None path)
        padding_point = torch.zeros(
            (B, 1, 2), dtype=point_coords.dtype, device=point_coords.device)
        padding_label = torch.full(
            (B, 1), -1, dtype=point_labels.dtype, device=point_labels.device)
        coords = torch.cat([point_coords, padding_point], dim=1)  # [B, N+1, 2]
        labels = torch.cat([point_labels, padding_label], dim=1)  # [B, N+1]

        # Positional encoding
        coords_shifted = coords + 0.5
        point_embedding = pe.pe_layer.forward_with_coords(
            coords_shifted, pe.input_image_size)  # [B, N+1, 256]

        # Replace boolean indexing with torch.where / masked add.
        # label == -1  ->  replace embedding with not_a_point_embed
        # label == 0   ->  add point_embeddings[0]
        # label == 1   ->  add point_embeddings[1]
        mask_neg1 = (labels == -1).unsqueeze(-1)  # [B, N+1, 1]
        not_a_point = pe.not_a_point_embed.weight.unsqueeze(0)  # [1, 1, 256]
        point_embedding = torch.where(mask_neg1, not_a_point, point_embedding)

        mask_0 = (labels == 0).unsqueeze(-1).float()  # [B, N+1, 1]
        point_embedding = point_embedding + mask_0 * pe.point_embeddings[0].weight

        mask_1 = (labels == 1).unsqueeze(-1).float()  # [B, N+1, 1]
        point_embedding = point_embedding + mask_1 * pe.point_embeddings[1].weight

        sparse_embeddings = point_embedding  # [B, N+1, 256]

        # Dense embeddings (no mask prompt)
        dense_embeddings = pe.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            B, -1, pe.image_embedding_size[0], pe.image_embedding_size[1]
        )  # [B, 256, 64, 64]

        # --- mask decoder ---
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embed,
            image_pe=pe.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )

        return low_res_masks, iou_predictions


# ===================================================================
#  DINOv2 wrapper
# ===================================================================

class DINOv2Wrapper(nn.Module):
    """Wraps DinoVisionTransformer to return (cls_token, patch_tokens).

    The ISM pipeline uses both:
      - x_norm_clstoken    (for semantic matching)
      - x_norm_patchtokens (for appearance matching)

    This wrapper calls forward_features and returns both as a tuple.
    """

    def __init__(self, vit_model: nn.Module):
        super().__init__()
        self.vit = vit_model

    def forward(self, x):
        features = self.vit.forward_features(x)
        cls_token = features["x_norm_clstoken"]          # [B, D]
        patch_tokens = features["x_norm_patchtokens"]    # [B, N_patches, D]

        return cls_token, patch_tokens


# ===================================================================
#  Export functions (one per model)
# ===================================================================

def export_sam_image_encoder(sam, output_dir, compress_fp16=False,
                             verify=False):
    """Export SAM image encoder to OV IR."""

    encoder = sam.image_encoder
    encoder.eval()

    # SAM always pads to 1024×1024
    dummy = torch.randn(1, 3, 1024, 1024)
    basename = "sam_image_encoder"
    xml_path = os.path.join(output_dir, basename + ".xml")

    print(f"[export] Converting SAM image encoder ...")

    ov_model = ov.convert_model(
        encoder,
        example_input=dummy,
        input=[ov.PartialShape([1, 3, 1024, 1024])],
    )

    # Name the output
    ov_model.outputs[0].tensor.set_names({"image_embeddings"})
    for i, out in enumerate(ov_model.outputs):
        print(f"  Output {i}: names={out.get_names()}, "
              f"shape={out.get_partial_shape()}")

    ov.save_model(ov_model, xml_path, compress_to_fp16=compress_fp16)

    size_mb = os.path.getsize(xml_path.replace(".xml", ".bin")) / 1e6
    tag = "FP16" if compress_fp16 else "FP32"
    print(f"[export] {tag} IR: {xml_path} ({size_mb:.1f} MB)")

    if verify:
        _verify_model(encoder, ov_model, dummy, "SAM image encoder")

    return xml_path


def export_sam_mask_decoder(sam, output_dir, compress_fp16=False,
                            verify=False):
    """Export SAM prompt_encoder + mask_decoder (fused) to OV IR."""

    wrapper = SAMMaskDecoderWrapper(sam)
    wrapper.eval()

    # Example shapes: 1 image embedding, batch of 64 points, 1 point each
    B = 64
    dummy_embed = torch.randn(1, 256, 64, 64)
    dummy_coords = torch.randn(B, 1, 2)
    dummy_labels = torch.ones(B, 1, dtype=torch.int32)

    basename = "sam_mask_decoder"
    xml_path = os.path.join(output_dir, basename + ".xml")

    print(f"[export] Converting SAM mask decoder ...")
    ov_model = ov.convert_model(
        wrapper,
        example_input=(dummy_embed, dummy_coords, dummy_labels),
        input=[
            ov.PartialShape([1, 256, 64, 64]),
            ov.PartialShape([-1, -1, 2]),
            ov.PartialShape([-1, -1]),
        ],
    )

    ov_model.outputs[0].tensor.set_names({"masks"})
    ov_model.outputs[1].tensor.set_names({"iou_predictions"})
    for i, out in enumerate(ov_model.outputs):
        print(f"  Output {i}: names={out.get_names()}, "
              f"shape={out.get_partial_shape()}")

    ov.save_model(ov_model, xml_path, compress_to_fp16=compress_fp16)

    size_mb = os.path.getsize(xml_path.replace(".xml", ".bin")) / 1e6
    tag = "FP16" if compress_fp16 else "FP32"
    print(f"[export] {tag} IR: {xml_path} ({size_mb:.1f} MB)")

    if verify:
        _verify_model(wrapper, ov_model,
                       (dummy_embed, dummy_coords, dummy_labels),
                       "SAM mask decoder")

    return xml_path


def export_dinov2(dinov2_model, output_dir, compress_fp16=False,
                  verify=False):
    """Export DINOv2 ViT to OV IR."""

    wrapper = DINOv2Wrapper(dinov2_model)
    wrapper.eval()

    # Proposal crops are 224×224; use dynamic batch
    dummy = torch.randn(1, 3, 224, 224)
    basename = "dinov2_vitl14"
    xml_path = os.path.join(output_dir, basename + ".xml")

    print(f"[export] Converting DINOv2 ViT-L/14 ...")

    ov_model = ov.convert_model(
        wrapper,
        example_input=dummy,
        input=[ov.PartialShape([-1, 3, 224, 224])],
    )

    ov_model.outputs[0].tensor.set_names({"cls_token"})
    ov_model.outputs[1].tensor.set_names({"patch_tokens"})
    for i, out in enumerate(ov_model.outputs):
        print(f"  Output {i}: names={out.get_names()}, "
              f"shape={out.get_partial_shape()}")

    ov.save_model(ov_model, xml_path, compress_to_fp16=compress_fp16)

    size_mb = os.path.getsize(xml_path.replace(".xml", ".bin")) / 1e6
    tag = "FP16" if compress_fp16 else "FP32"
    print(f"[export] {tag} IR: {xml_path} ({size_mb:.1f} MB)")

    if verify:
        _verify_model(wrapper, ov_model, dummy, "DINOv2 ViT-L/14")

    return xml_path


# ===================================================================
#  Extended export: bake pre/post-processing custom ops into IR
# ===================================================================

def export_sam_encoder_ext(sam, output_dir, compress_fp16=False,
                           verify=False,
                           im_h=480, im_w=640, image_size=1024):
    """Export SAM image encoder with SamPreprocess custom op baked in.

    Graph: image[1,H,W,3] u8 -> SamPreprocess -> tensor[1,3,1024,1024] f32
           -> NN encoder -> embeddings[1,256,64,64] f32

    Outputs input_size[1,1,1,2] for use by mask postprocess.
    """

    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Get NN body (reuse base IR if available) ---
    base_xml = os.path.join(output_dir, 'sam_image_encoder.xml')
    if not os.path.isfile(base_xml):
        print(f'[export:sam_enc_ext] Step 0: Exporting base IR {base_xml}')
        export_sam_image_encoder(sam, output_dir, compress_fp16=compress_fp16)
    print(f'[export:sam_enc_ext] Step 1: Loading base IR {base_xml}')
    nn_model = ov.Core().read_model(base_xml)

    # --- Step 2: Build extended graph (standard OV ops only) ---
    print('[export:sam_enc_ext] Step 2: Building extended graph ...')

    param_image = opset.parameter(
        ov.PartialShape([1, im_h, im_w, 3]), OVType.u8, name='image')

    # --- Preprocessing with standard OV ops ---
    # (a) Convert u8 -> f32
    image_f32 = opset.convert(param_image.output(0), OVType.f32)  # [1,H,W,3]

    # (b) Transpose HWC -> CHW: [1,H,W,3] -> [1,3,H,W]
    perm = opset.constant(np.array([0, 3, 1, 2], dtype=np.int64))
    image_chw = opset.transpose(image_f32, perm)  # [1,3,H,W]

    # (c) Resize longest side to image_size (bilinear, half_pixel)
    _max_hw = max(im_h, im_w)
    _scale = image_size / _max_hw
    _new_h = int(im_h * _scale + 0.5)
    _new_w = int(im_w * _scale + 0.5)
    target_hw = opset.constant(np.array([_new_h, _new_w], dtype=np.int64))
    axes_hw = opset.constant(np.array([2, 3], dtype=np.int64))
    resized = opset.interpolate(
        image_chw, target_hw,
        mode='linear_onnx',
        shape_calculation_mode='sizes',
        coordinate_transformation_mode='half_pixel',
        axes=axes_hw,
    )  # [1, 3, new_h, new_w]

    # (d) Normalize: (pixel - mean) * inv_std
    mean = opset.constant(
        np.array([123.675, 116.28, 103.53],
                   dtype=np.float32).reshape(1, 3, 1, 1))
    inv_std = opset.constant(
        np.array([1.0 / 58.395, 1.0 / 57.12, 1.0 / 57.375],
                   dtype=np.float32).reshape(1, 3, 1, 1))
    normalized = opset.multiply(opset.subtract(resized, mean), inv_std)

    # (e) Pad to [1, 3, image_size, image_size]
    pad_h = image_size - _new_h
    pad_w = image_size - _new_w
    tensor_out = opset.pad(
        normalized,
        pads_begin=opset.constant(np.array([0, 0, 0, 0], dtype=np.int64)),
        pads_end=opset.constant(
            np.array([0, 0, pad_h, pad_w], dtype=np.int64)),
        arg_pad_value=opset.constant(np.float32(0)),
        pad_mode='constant',
    )  # [1, 3, 1024, 1024]

    # Compute input_size as a Constant (deterministic from im_h, im_w)
    input_size_const = opset.constant(
        np.array([[[[float(_new_h), float(_new_w)]]]], dtype=np.float32),
        name='input_size_const',
    )
    input_size_out = input_size_const.output(0)  # [1, 1, 1, 2]

    # Rewire NN body input
    nn_params = nn_model.get_parameters()
    nn_results = nn_model.get_results()
    nn_input_param = nn_params[0]
    for target_input in nn_input_param.output(0).get_target_inputs():
        target_input.replace_source_output(tensor_out.output(0))

    # NN output: image_embeddings [1, 256, 64, 64]
    embeddings_node = nn_results[0].input(0).get_source_output()

    # --- Step 3: Assemble model ---
    print('[export:sam_enc_ext] Step 3: Creating extended model ...')

    ext_model = ov.Model(
        results=[OVResult(embeddings_node), OVResult(input_size_out)],
        parameters=[param_image],
        name='sam_image_encoder_ext',
    )
    ext_model.outputs[0].tensor.set_names({'image_embeddings'})
    ext_model.outputs[1].tensor.set_names({'input_size'})
    ext_model.inputs[0].tensor.set_names({'image'})

    for i, inp in enumerate(ext_model.inputs):
        print(f'  Input {i}: names={inp.get_names()}, '
              f'shape={inp.get_partial_shape()}, type={inp.get_element_type()}')
    for i, out in enumerate(ext_model.outputs):
        print(f'  Output {i}: names={out.get_names()}, '
              f'shape={out.get_partial_shape()}')

    # --- Step 4: Save ---
    xml_path = os.path.join(output_dir, 'sam_image_encoder_ext.xml')
    ov.save_model(ext_model, xml_path, compress_to_fp16=compress_fp16)
    size_mb = os.path.getsize(xml_path.replace('.xml', '.bin')) / 1e6
    tag = 'FP16' if compress_fp16 else 'FP32'
    print(f'[export:sam_enc_ext] {tag} IR: {xml_path} ({size_mb:.1f} MB)')

    # --- Step 5: Verification ---
    if verify:
        core = ov.Core()
        compiled = core.compile_model(xml_path, 'CPU')
        test_image = np.random.randint(
            0, 255, (1, im_h, im_w, 3), dtype=np.uint8)
        result = compiled({'image': test_image})
        print(f'  embeddings shape: '
              f'{result[compiled.output(0)].shape}')
        print(f'  input_size: '
              f'{result[compiled.output(1)]}')

    return xml_path


def export_sam_decoder_ext(sam, output_dir, compress_fp16=False,
                           verify=False,
                           im_h=480, im_w=640, image_size=1024):
    """Export SAM mask decoder with postprocessing baked in using standard OV ops.

    Graph: image_embed[1,256,64,64] + coords[B,N,2] + labels[B,N]
           -> NN decoder -> low_res_masks[B,3,256,256] + iou_preds[B,3]
           -> Interpolate(256->1024) -> Crop(input_h,input_w) -> Interpolate(orig)
           -> masks (thresholded) + logits (raw)

    All postprocess dimensions are computed from im_h, im_w at export time
    and baked as constants - no runtime params input needed.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Compute fixed postprocess dimensions
    _max_hw = max(im_h, im_w)
    _scale = float(image_size) / float(_max_hw)
    input_h = int(im_h * _scale + 0.5)
    input_w = int(im_w * _scale + 0.5)
    orig_h = im_h
    orig_w = im_w

    # --- Step 1: Get NN body (reuse base IR if available) ---
    base_xml = os.path.join(output_dir, 'sam_mask_decoder.xml')
    if not os.path.isfile(base_xml):
        print(f'[export:sam_dec_ext] Step 0: Exporting base IR {base_xml}')
        export_sam_mask_decoder(sam, output_dir, compress_fp16=compress_fp16)
    print(f'[export:sam_dec_ext] Step 1: Loading base IR {base_xml}')
    nn_model = ov.Core().read_model(base_xml)

    # --- Step 2: Build extended graph ---
    print('[export:sam_dec_ext] Step 2: Building extended graph ...')

    # NN outputs: low_res_masks [B, 3, 256, 256], iou_predictions [B, 3]
    nn_results = nn_model.get_results()
    low_res_masks_node = nn_results[0].input(0).get_source_output()
    iou_preds_node = nn_results[1].input(0).get_source_output()

    # ---- Standard OV ops for postprocessing ----
    # Step A: Interpolate 256->1024 (bilinear, align_corners=False)
    target_1024 = opset.constant(np.array([image_size, image_size], dtype=np.int64))
    axes_hw = opset.constant(np.array([2, 3], dtype=np.int64))
    upsampled = opset.interpolate(
        low_res_masks_node, target_1024,
        mode='linear_onnx',
        shape_calculation_mode='sizes',
        coordinate_transformation_mode='half_pixel',
        axes=axes_hw,
    )

    # Step B: Crop to [input_h, input_w]  (StridedSlice along H,W dims)
    begin_crop = opset.constant(np.array([0, 0, 0, 0], dtype=np.int64))
    end_crop = opset.constant(
        np.array([0, 0, input_h, input_w], dtype=np.int64))
    cropped = opset.strided_slice(
        upsampled, begin_crop, end_crop,
        opset.constant(np.array([1, 1, 1, 1], dtype=np.int64)),
        begin_mask=[1, 1, 0, 0],   # pass-through for B and C
        end_mask=[1, 1, 0, 0],
    )

    # Step C: Interpolate to original size
    target_orig = opset.constant(np.array([orig_h, orig_w], dtype=np.int64))
    logits_node = opset.interpolate(
        cropped, target_orig,
        mode='linear_onnx',
        shape_calculation_mode='sizes',
        coordinate_transformation_mode='half_pixel',
        axes=axes_hw,
    )

    # Step D: Threshold -> masks
    zero_const = opset.constant(np.float32(0.0))
    masks_bool = opset.greater(logits_node, zero_const)
    masks_node = opset.convert(masks_bool, OVType.f32)

    # --- Step 3: Assemble model ---
    print('[export:sam_dec_ext] Step 3: Creating extended model ...')

    nn_params = nn_model.get_parameters()
    ext_model = ov.Model(
        results=[
            OVResult(masks_node.output(0)),
            OVResult(iou_preds_node),
            OVResult(logits_node.output(0)),
        ],
        parameters=list(nn_params),
        name='sam_mask_decoder_ext',
    )
    ext_model.outputs[0].tensor.set_names({'masks'})
    ext_model.outputs[1].tensor.set_names({'iou_predictions'})
    ext_model.outputs[2].tensor.set_names({'logits'})

    for i, inp in enumerate(ext_model.inputs):
        print(f'  Input {i}: names={inp.get_names()}, '
              f'shape={inp.get_partial_shape()}, type={inp.get_element_type()}')
    for i, out in enumerate(ext_model.outputs):
        print(f'  Output {i}: names={out.get_names()}, '
              f'shape={out.get_partial_shape()}')

    # --- Step 4: Save ---
    xml_path = os.path.join(output_dir, 'sam_mask_decoder_ext.xml')
    ov.save_model(ext_model, xml_path, compress_to_fp16=compress_fp16)
    size_mb = os.path.getsize(xml_path.replace('.xml', '.bin')) / 1e6
    tag = 'FP16' if compress_fp16 else 'FP32'
    print(f'[export:sam_dec_ext] {tag} IR: {xml_path} ({size_mb:.1f} MB)')

    # --- Step 5: Verification ---
    if verify:
        core = ov.Core()
        compiled = core.compile_model(xml_path, 'CPU')
        B = 2
        dummy_embed = np.random.randn(1, 256, 64, 64).astype(np.float32)
        dummy_coords = np.random.randn(B, 1, 2).astype(np.float32) * 100
        dummy_labels = np.ones((B, 1), dtype=np.int32)
        result = compiled([dummy_embed, dummy_coords, dummy_labels])
        print(f'  masks: {result[0].shape}, iou: {result[1].shape}, '
              f'logits: {result[2].shape}')

    return xml_path


def export_sam_decoder_ext_v2(sam, output_dir, compress_fp16=False,
                               verify=False,
                               im_h=480, im_w=640, image_size=1024):
    """Export SAM mask decoder v2 with stability_score + mask_to_box on GPU.

    Extends v1 by computing stability_score and bounding boxes from masks
    entirely on GPU using standard OV ops.

    Graph: image_embed + coords + labels
           -> NN decoder -> low_res_masks + iou_preds
           -> Interpolate -> Crop -> Interpolate -> logits [B,3,orig_h,orig_w]
           -> stability_score: (logits > +offset).sum / (logits > -offset).sum -> [B,3]
           -> binary masks: logits > 0 -> [B,3,orig_h,orig_w]
           -> mask_to_box: min/max of row/col indices -> [B,3,4]

    Outputs: masks, iou_predictions, stability_scores, boxes
    Masks are exported as boolean to avoid transferring full-resolution
    binary masks as float32.

    All ops are standard OV opset.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Compute fixed postprocess dimensions
    _max_hw = max(im_h, im_w)
    _scale = float(image_size) / float(_max_hw)
    input_h = int(im_h * _scale + 0.5)
    input_w = int(im_w * _scale + 0.5)
    orig_h = im_h
    orig_w = im_w

    # --- Step 1: Get NN body (reuse base IR if available) ---
    base_xml = os.path.join(output_dir, 'sam_mask_decoder.xml')
    if not os.path.isfile(base_xml):
        print(f'[export:sam_dec_ext_v2] Step 0: Exporting base IR {base_xml}')
        export_sam_mask_decoder(sam, output_dir, compress_fp16=compress_fp16)
    print(f'[export:sam_dec_ext_v2] Step 1: Loading base IR {base_xml}')
    nn_model = ov.Core().read_model(base_xml)

    # --- Step 2: Build extended graph ---
    print('[export:sam_dec_ext_v2] Step 2: Building extended graph ...')

    nn_results = nn_model.get_results()
    low_res_masks_node = nn_results[0].input(0).get_source_output()
    iou_preds_node = nn_results[1].input(0).get_source_output()

    # ---- A: Interpolate 256->1024 ----
    target_1024 = opset.constant(np.array([image_size, image_size], dtype=np.int64))
    axes_hw = opset.constant(np.array([2, 3], dtype=np.int64))
    upsampled = opset.interpolate(
        low_res_masks_node, target_1024,
        mode='linear_onnx',
        shape_calculation_mode='sizes',
        coordinate_transformation_mode='half_pixel',
        axes=axes_hw,
    )

    # ---- B: Crop to [input_h, input_w] ----
    begin_crop = opset.constant(np.array([0, 0, 0, 0], dtype=np.int64))
    end_crop = opset.constant(np.array([0, 0, input_h, input_w], dtype=np.int64))
    cropped = opset.strided_slice(
        upsampled, begin_crop, end_crop,
        opset.constant(np.array([1, 1, 1, 1], dtype=np.int64)),
        begin_mask=[1, 1, 0, 0],
        end_mask=[1, 1, 0, 0],
    )

    # ---- C: Interpolate to original size ----
    target_orig = opset.constant(np.array([orig_h, orig_w], dtype=np.int64))
    logits_node = opset.interpolate(
        cropped, target_orig,
        mode='linear_onnx',
        shape_calculation_mode='sizes',
        coordinate_transformation_mode='half_pixel',
        axes=axes_hw,
    )  # [B, 3, orig_h, orig_w]

    # ---- D: Binary masks ----
    zero_const = opset.constant(np.float32(0.0))
    masks_bool = opset.greater(logits_node, zero_const)
    masks_node = masks_bool  # [B, 3, orig_h, orig_w] bool

    # ---- E: Stability score (using standard OV ops) ----
    # stability = (logits > +offset).sum(H,W) / (logits > -offset).sum(H,W)
    # mask_threshold=0.0, stability_score_offset=1.0
    offset_pos = opset.constant(np.float32(1.0))   # mask_threshold + offset
    offset_neg = opset.constant(np.float32(-1.0))   # mask_threshold - offset
    reduce_hw = opset.constant(np.array([2, 3], dtype=np.int64))

    inter_mask = opset.greater(logits_node, offset_pos)
    inter_f32 = opset.convert(inter_mask, OVType.f32)
    intersections = opset.reduce_sum(inter_f32, reduce_hw, keep_dims=False)  # [B, 3]

    union_mask = opset.greater(logits_node, offset_neg)
    union_f32 = opset.convert(union_mask, OVType.f32)
    unions = opset.reduce_sum(union_f32, reduce_hw, keep_dims=False)  # [B, 3]

    # Avoid division by zero
    eps_const = opset.constant(np.float32(1e-6))
    unions_safe = opset.maximum(unions, eps_const)
    stability_scores_node = opset.divide(intersections, unions_safe)  # [B, 3]

    # ---- F: Mask-to-box (using standard OV ops) ----
    # For each binary mask [H, W], compute bounding box [left, top, right, bottom]
    # masks_node is [B, 3, orig_h, orig_w] bool

    # in_height = max(mask, dim=-1(W))  -> [B, 3, H]
    w_axis = opset.constant(np.array([3], dtype=np.int64))
    masks_f32 = opset.convert(masks_node, OVType.f32)
    in_height = opset.reduce_max(masks_f32, w_axis, keep_dims=False)  # [B, 3, H]

    # height coords: in_height * arange(H)
    h_range = opset.constant(
        np.arange(orig_h, dtype=np.float32).reshape(1, 1, -1))  # [1, 1, H]
    in_height_coords = opset.multiply(in_height, h_range)  # [B, 3, H]

    # bottom_edges = max(in_height_coords, dim=-1) -> [B, 3]
    h_axis = opset.constant(np.array([2], dtype=np.int64))
    bottom_edges = opset.reduce_max(in_height_coords, h_axis, keep_dims=False)

    # top_edges: add H where mask row is empty, then take min
    h_const = opset.constant(np.float32(float(orig_h)))
    ones_const = opset.constant(np.float32(1.0))
    not_in_height = opset.subtract(ones_const, in_height)  # [B, 3, H]
    in_height_coords_top = opset.add(
        in_height_coords,
        opset.multiply(h_const, not_in_height))  # [B, 3, H]
    top_edges = opset.reduce_min(in_height_coords_top, h_axis, keep_dims=False)

    # in_width = max(mask, dim=-2(H))  -> [B, 3, W]
    h_axis_for_width = opset.constant(np.array([2], dtype=np.int64))
    in_width = opset.reduce_max(masks_f32, h_axis_for_width,
                                keep_dims=False)  # [B, 3, W]

    # width coords: in_width * arange(W)
    w_range = opset.constant(
        np.arange(orig_w, dtype=np.float32).reshape(1, 1, -1))  # [1, 1, W]
    in_width_coords = opset.multiply(in_width, w_range)  # [B, 3, W]

    # right_edges = max(in_width_coords, dim=-1)
    w_reduce_axis = opset.constant(np.array([2], dtype=np.int64))
    right_edges = opset.reduce_max(in_width_coords, w_reduce_axis,
                                   keep_dims=False)  # [B, 3]

    # left_edges: add W where mask col is empty, then take min
    w_const = opset.constant(np.float32(float(orig_w)))
    not_in_width = opset.subtract(ones_const, in_width)  # [B, 3, W]
    in_width_coords_left = opset.add(
        in_width_coords,
        opset.multiply(w_const, not_in_width))  # [B, 3, W]
    left_edges = opset.reduce_min(in_width_coords_left, w_reduce_axis,
                                  keep_dims=False)  # [B, 3]

    # Stack [left, top, right, bottom] -> [B, 3, 4]
    unsq_axis = opset.constant(np.array([2], dtype=np.int64))
    left_u = opset.unsqueeze(left_edges, unsq_axis)
    top_u = opset.unsqueeze(top_edges, unsq_axis)
    right_u = opset.unsqueeze(right_edges, unsq_axis)
    bottom_u = opset.unsqueeze(bottom_edges, unsq_axis)
    boxes_raw = opset.concat([left_u, top_u, right_u, bottom_u], axis=2)

    # Zero out boxes for empty masks (right < left or bottom < top)
    empty_lr = opset.less(right_edges, left_edges)
    empty_tb = opset.less(bottom_edges, top_edges)
    empty_mask = opset.logical_or(empty_lr, empty_tb)  # [B, 3]
    empty_expanded = opset.unsqueeze(empty_mask, unsq_axis)  # [B, 3, 1]
    zero_box = opset.constant(
        np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 1, 4))
    boxes_node = opset.select(empty_expanded, zero_box, boxes_raw)  # [B, 3, 4]

    # --- Step 3: Assemble model ---
    print('[export:sam_dec_ext_v2] Step 3: Creating extended model ...')

    nn_params = nn_model.get_parameters()
    ext_model = ov.Model(
        results=[
            OVResult(masks_node.output(0)),
            OVResult(iou_preds_node),
            OVResult(stability_scores_node.output(0)),
            OVResult(boxes_node.output(0)),
        ],
        parameters=list(nn_params),
        name='sam_mask_decoder_ext_v2',
    )
    ext_model.outputs[0].tensor.set_names({'masks'})
    ext_model.outputs[1].tensor.set_names({'iou_predictions'})
    ext_model.outputs[2].tensor.set_names({'stability_scores'})
    ext_model.outputs[3].tensor.set_names({'boxes'})

    for i, inp in enumerate(ext_model.inputs):
        print(f'  Input {i}: names={inp.get_names()}, '
              f'shape={inp.get_partial_shape()}, type={inp.get_element_type()}')
    for i, out in enumerate(ext_model.outputs):
        print(f'  Output {i}: names={out.get_names()}, '
              f'shape={out.get_partial_shape()}')

    # --- Step 4: Save ---
    xml_path = os.path.join(output_dir, 'sam_mask_decoder_ext_v2.xml')
    ov.save_model(ext_model, xml_path, compress_to_fp16=compress_fp16)
    size_mb = os.path.getsize(xml_path.replace('.xml', '.bin')) / 1e6
    tag = 'FP16' if compress_fp16 else 'FP32'
    print(f'[export:sam_dec_ext_v2] {tag} IR: {xml_path} ({size_mb:.1f} MB)')

    if verify:
        core = ov.Core()
        compiled = core.compile_model(xml_path, 'CPU')
        B = 4
        dummy_embed = np.random.randn(1, 256, 64, 64).astype(np.float32)
        dummy_coords = np.random.randn(B, 1, 2).astype(np.float32) * 100
        dummy_labels = np.ones((B, 1), dtype=np.int32)
        result = compiled([dummy_embed, dummy_coords, dummy_labels])
        masks = result[0]  # [B, 3, H, W]
        iou   = result[1]  # [B, 3]
        stab  = result[2]  # [B, 3]
        boxes = result[3]  # [B, 3, 4]
        print(f'  [verify] masks: {masks.shape}, iou: {iou.shape}, '
            f'stab: {stab.shape}, boxes: {boxes.shape}')
        print(f'  [verify] stab range: [{stab.min():.4f}, {stab.max():.4f}]')
        print(f'  [verify] boxes sample: {boxes[0, 0]}')

    return xml_path


def export_dinov2_ext(dinov2_model, output_dir, compress_fp16=False,
                      verify=False):
    """Export DINOv2 with patch postprocessing baked in using standard OV ops.

    Graph: images[N,3,224,224] + masks[N,1,224,224]
           -> NN dinov2 -> cls_token[N,1024] + patch_tokens[N,256,1024]
           -> AvgPool2d on masks -> threshold -> L2 normalize -> features

    Outputs: cls_token, features (masked+normalised patch tokens)
    """

    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Get NN body (reuse base IR if available) ---
    base_xml = os.path.join(output_dir, 'dinov2_vitl14.xml')
    if not os.path.isfile(base_xml):
        print(f'[export:dinov2_ext] Step 0: Exporting base IR {base_xml}')
        export_dinov2(dinov2_model, output_dir, compress_fp16=compress_fp16)
    print(f'[export:dinov2_ext] Step 1: Loading base IR {base_xml}')
    nn_model = ov.Core().read_model(base_xml)

    # --- Step 2: Build extended graph ---
    print('[export:dinov2_ext] Step 2: Building extended graph ...')

    # Add masks input for patch postprocessing
    param_masks = opset.parameter(
        ov.PartialShape([-1, 1, 224, 224]), OVType.f32, name='proposal_masks')

    # NN outputs: cls_token [N, 1024], patch_tokens [N, 256, 1024]
    nn_results = nn_model.get_results()
    cls_token_node = nn_results[0].input(0).get_source_output()
    patch_tokens_node = nn_results[1].input(0).get_source_output()

    # ---- Standard OV ops for patch postprocessing ----
    # Step A: AvgPool2d on masks - [N,1,224,224] -> [N,1,16,16]
    patch_size = 14
    validpatch_thresh = 0.5
    pooled = opset.avg_pool(
        param_masks.output(0),
        strides=[patch_size, patch_size],
        pads_begin=[0, 0],
        pads_end=[0, 0],
        kernel_shape=[patch_size, patch_size],
        exclude_pad=False,
    )  # [N, 1, 16, 16]

    # Step B: Reshape to [N, 256] and threshold
    num_patches = (224 // patch_size) ** 2  # 256
    shape_256 = opset.constant(np.array([0, num_patches], dtype=np.int64))
    pooled_flat = opset.reshape(pooled, shape_256, special_zero=True)  # [N, 256]
    thresh = opset.constant(np.float32(validpatch_thresh))
    valid_mask = opset.greater(pooled_flat, thresh)  # [N, 256] bool
    valid_f32 = opset.convert(valid_mask, OVType.f32)  # [N, 256] f32
    # Unsqueeze to [N, 256, 1] for broadcasting
    valid_expanded = opset.unsqueeze(
        valid_f32, opset.constant(np.array([2], dtype=np.int64)))

    # Step C: L2 normalize patch_tokens along last dim
    eps = opset.constant(np.float32(1e-6))
    reduce_axes = opset.constant(np.array([2], dtype=np.int64))
    l2_norm = opset.reduce_l2(patch_tokens_node, reduce_axes, keep_dims=True)
    l2_norm_safe = opset.maximum(l2_norm, eps)
    normalized = opset.divide(patch_tokens_node, l2_norm_safe)

    # Step D: Apply mask - zero out invalid patches
    features_node = opset.multiply(normalized, valid_expanded)

    # --- Step 3: Assemble model ---
    print('[export:dinov2_ext] Step 3: Creating extended model ...')

    nn_params = nn_model.get_parameters()
    ext_model = ov.Model(
        results=[OVResult(cls_token_node), OVResult(features_node.output(0))],
        parameters=list(nn_params) + [param_masks],
        name='dinov2_vitl14_ext',
    )
    ext_model.outputs[0].tensor.set_names({'cls_token'})
    ext_model.outputs[1].tensor.set_names({'patch_features'})

    for i, inp in enumerate(ext_model.inputs):
        print(f'  Input {i}: names={inp.get_names()}, '
              f'shape={inp.get_partial_shape()}, type={inp.get_element_type()}')
    for i, out in enumerate(ext_model.outputs):
        print(f'  Output {i}: names={out.get_names()}, '
              f'shape={out.get_partial_shape()}')

    # --- Step 4: Save ---
    xml_path = os.path.join(output_dir, 'dinov2_vitl14_ext.xml')
    ov.save_model(ext_model, xml_path, compress_to_fp16=compress_fp16)
    size_mb = os.path.getsize(xml_path.replace('.xml', '.bin')) / 1e6
    tag = 'FP16' if compress_fp16 else 'FP32'
    print(f'[export:dinov2_ext] {tag} IR: {xml_path} ({size_mb:.1f} MB)')

    # --- Step 5: Verification ---
    if verify:
        core = ov.Core()
        compiled = core.compile_model(xml_path, 'CPU')
        N = 2
        dummy_images = np.random.randn(N, 3, 224, 224).astype(np.float32)
        dummy_masks = np.random.rand(N, 1, 224, 224).astype(np.float32)
        result = compiled({'images': dummy_images, 'proposal_masks': dummy_masks})
        print(f'  cls_token: {result[0].shape}, '
              f'patch_features: {result[1].shape}')

    return xml_path


def export_fastsam(checkpoint_path, output_dir, compress_fp16=False,
                   imgsz=640, verify=False):
    """Export FastSAM (YOLOv8x-seg) to OpenVINO IR.

    Steps:
      1. Load FastSAM-x.pt via ultralytics YOLO
      2. Export to ONNX with dynamic=True (variable H/W dims)
      3. Convert ONNX to OV IR using ov.convert_model()
      4. Save as fastsam_x_dynamic.xml/.bin

    Args:
        checkpoint_path: Path to FastSAM-x.pt
        output_dir: Directory for output .xml/.bin
        compress_fp16: Whether to save with FP16 weights
        imgsz: Input image size (default 640, matching config)
        verify: If True, run a quick sanity check after export
    """

    from ultralytics import YOLO

    os.makedirs(output_dir, exist_ok=True)

    print(f'[export:fastsam] Loading {checkpoint_path} ...')
    model = YOLO(checkpoint_path)

    # Step 1: Export to ONNX with dynamic spatial dims
    print(f'[export:fastsam] Step 1: Exporting dynamic ONNX '
          f'(imgsz={imgsz}, dynamic=True) ...')
    onnx_path = model.export(format="onnx", imgsz=imgsz,
                             simplify=False, dynamic=True)
    print(f'[export:fastsam] ONNX exported: {onnx_path}')

    # Step 2: Convert ONNX to OV IR
    print(f'[export:fastsam] Step 2: Converting ONNX -> OpenVINO IR ...')
    ov_model = ov.convert_model(onnx_path)

    # Step 3: Save
    xml_path = os.path.join(output_dir, 'fastsam_x_dynamic.xml')
    ov.save_model(ov_model, xml_path, compress_to_fp16=compress_fp16)
    size_mb = os.path.getsize(xml_path.replace('.xml', '.bin')) / 1e6
    tag = 'FP16' if compress_fp16 else 'FP32'
    print(f'[export:fastsam] {tag} IR: {xml_path} ({size_mb:.1f} MB)')

    # Print model info
    for i, inp in enumerate(ov_model.inputs):
        print(f'  Input {i}: names={inp.get_names()}, '
              f'shape={inp.get_partial_shape()}, type={inp.get_element_type()}')
    for i, out in enumerate(ov_model.outputs):
        print(f'  Output {i}: names={out.get_names()}, '
              f'shape={out.get_partial_shape()}')

    if verify:
        core = ov.Core()
        compiled = core.compile_model(core.read_model(xml_path), 'CPU')

        print(f'[verify:fastsam] Model inputs:')
        for inp in compiled.inputs:
            print(f'  {inp.get_names()}: {inp.get_partial_shape()} '
                f'{inp.get_element_type()}')
        print(f'[verify:fastsam] Model outputs:')
        for out in compiled.outputs:
            print(f'  {out.get_names()}: {out.get_partial_shape()} '
                f'{out.get_element_type()}')

        # Run dummy inference (use square imgsz for dynamic model)
        h = w = imgsz
        dummy = np.random.rand(1, 3, h, w).astype(np.float32)
        result = compiled({0: dummy})
        for i, out in enumerate(compiled.outputs):
            arr = result[out]
            print(f'  Output {i}: shape={arr.shape}, '
                f'min={arr.min():.4f}, max={arr.max():.4f}')
        print('[verify:fastsam] PASS - model compiled and ran successfully')

    return xml_path


# ===================================================================
#  Verification helper
# ===================================================================

def _verify_model(pt_model, ov_model, example_input, label):
    """Run numerical comparison between PyTorch and OV outputs."""

    print(f"\n[verify] === {label} ===")

    core = ov.Core()
    compiled = core.compile_model(ov_model, "CPU")

    # PyTorch forward
    with torch.no_grad():
        if isinstance(example_input, tuple):
            pt_out = pt_model(*example_input)
        else:
            pt_out = pt_model(example_input)

    if not isinstance(pt_out, (tuple, list)):
        pt_out = (pt_out,)

    # OV forward
    if isinstance(example_input, tuple):
        ov_inputs = [x.numpy() if isinstance(x, torch.Tensor) else x
                     for x in example_input]
    else:
        ov_inputs = [example_input.numpy()]

    ov_result = compiled(ov_inputs)

    for i, pt_tensor in enumerate(pt_out):
        ptnp = pt_tensor.detach().cpu().numpy()
        ovnp = ov_result[i]
        max_err = np.max(np.abs(ptnp - ovnp))
        mean_err = np.mean(np.abs(ptnp - ovnp))
        cos_sim = (np.sum(ptnp * ovnp) /
                   (np.linalg.norm(ptnp) * np.linalg.norm(ovnp) + 1e-12))
        status = "PASS" if max_err < 1e-3 else "WARN"
        print(f"  Output {i}: max_err={max_err:.6f}, "
              f"mean_err={mean_err:.6f}, cos_sim={cos_sim:.6f}  [{status}]")


# ===================================================================
#  Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export ISM PyTorch models to OpenVINO IR")
    parser.add_argument("--output_dir", required=True,
                        help="Directory for .xml/.bin output files")
    parser.add_argument("--sam_checkpoint_dir",
                        default=os.path.join(BASE_DIR,
                                             "checkpoints/segment-anything"),
                        help="Directory containing SAM .pth checkpoint")
    parser.add_argument("--sam_model_type", default="vit_h",
                        choices=["vit_h", "vit_l", "vit_b"],
                        help="SAM backbone variant (default: vit_h)")
    parser.add_argument("--dinov2_checkpoint_dir",
                        default=os.path.join(BASE_DIR,
                                             "checkpoints/dinov2"),
                        help="Directory containing DINOv2 .pth checkpoint")
    parser.add_argument("--dinov2_model_name", default="dinov2_vitl14",
                        help="DINOv2 model name (default: dinov2_vitl14)")
    parser.add_argument("--fastsam_checkpoint",
                        default=os.path.join(BASE_DIR,
                                             "checkpoints/FastSAM/FastSAM-x.pt"),
                        help="Path to FastSAM-x.pt checkpoint")
    parser.add_argument("--only", default=None,
                        choices=["sam_encoder", "sam_decoder", "dinov2",
                                 "sam_encoder_ext", "sam_decoder_ext",
                                 "sam_decoder_ext_v2",
                                 "dinov2_ext", "fastsam"],
                        help="Export only one model (default: all)")
    parser.add_argument("--ext", action="store_true",
                        help="Export extended models with custom pre/post ops")
    parser.add_argument("--fp16", action="store_true",
                        help="Compress weights to FP16 in the IR")
    parser.add_argument("--verify", action="store_true",
                        help="Run numerical verification vs PyTorch")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    export_all = args.only is None

    results = {}

    # --- SAM ---
    sam = None
    if export_all or args.only in ("sam_encoder", "sam_decoder",
                                    "sam_encoder_ext", "sam_decoder_ext",
                                    "sam_decoder_ext_v2"):
        sam = build_sam(args.sam_checkpoint_dir, args.sam_model_type)

    if export_all or args.only == "sam_encoder":
        results["sam_encoder"] = export_sam_image_encoder(
            sam, args.output_dir, compress_fp16=args.fp16, verify=args.verify)

    if export_all or args.only == "sam_decoder":
        results["sam_decoder"] = export_sam_mask_decoder(
            sam, args.output_dir, compress_fp16=args.fp16, verify=args.verify)

    # Extended exports (with custom ops)
    if (args.ext and export_all) or args.only == "sam_encoder_ext":
        if sam is None:
            sam = build_sam(args.sam_checkpoint_dir, args.sam_model_type)
        results["sam_encoder_ext"] = export_sam_encoder_ext(
            sam, args.output_dir, compress_fp16=args.fp16, verify=args.verify)

    if (args.ext and export_all) or args.only == "sam_decoder_ext":
        if sam is None:
            sam = build_sam(args.sam_checkpoint_dir, args.sam_model_type)
        results["sam_decoder_ext"] = export_sam_decoder_ext(
            sam, args.output_dir, compress_fp16=args.fp16, verify=args.verify)

    if (args.ext and export_all) or args.only == "sam_decoder_ext_v2":
        if sam is None:
            sam = build_sam(args.sam_checkpoint_dir, args.sam_model_type)
        results["sam_decoder_ext_v2"] = export_sam_decoder_ext_v2(
            sam, args.output_dir, compress_fp16=args.fp16,
            verify=args.verify)

    del sam
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- DINOv2 ---
    if export_all or args.only in ("dinov2", "dinov2_ext"):
        dinov2 = build_dinov2(args.dinov2_checkpoint_dir,
                              args.dinov2_model_name)
        if export_all or args.only == "dinov2":
            results["dinov2"] = export_dinov2(
                dinov2, args.output_dir, compress_fp16=args.fp16,
                verify=args.verify)

        if (args.ext and export_all) or args.only == "dinov2_ext":
            results["dinov2_ext"] = export_dinov2_ext(
                dinov2, args.output_dir, compress_fp16=args.fp16,
                verify=args.verify)
        del dinov2

    # --- FastSAM ---
    if export_all or args.only == "fastsam":
        if os.path.isfile(args.fastsam_checkpoint):
            results["fastsam"] = export_fastsam(
                args.fastsam_checkpoint, args.output_dir,
                compress_fp16=args.fp16, verify=args.verify)
        else:
            print(f"[SKIP] FastSAM checkpoint not found: "
                  f"{args.fastsam_checkpoint}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  Export Summary")
    print(f"{'='*60}")
    for name, path in results.items():
        print(f"  {name:20s} -> {path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
