# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn


class OVLayerNorm2d(nn.Module):
    """LayerNorm for NCHW tensors using a channel-last LayerNorm op."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for old_suffix, new_suffix in (("weight", "norm.weight"),
                                       ("bias", "norm.bias")):
            old_key = prefix + old_suffix
            new_key = prefix + new_suffix
            if old_key in state_dict and new_key not in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


@contextmanager
def use_ov_layer_norm2d_for_sam():
    from segment_anything.modeling import common, image_encoder, mask_decoder, prompt_encoder

    modules = (common, image_encoder, mask_decoder, prompt_encoder)
    original_layer_norms = [module.LayerNorm2d for module in modules]
    for module in modules:
        module.LayerNorm2d = OVLayerNorm2d
    try:
        yield
    finally:
        for module, original_layer_norm in zip(modules, original_layer_norms):
            module.LayerNorm2d = original_layer_norm