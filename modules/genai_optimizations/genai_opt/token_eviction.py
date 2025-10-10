# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Generator, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel

from logging import getLogger

logger = getLogger(__name__)


class KVCacheCompressionMode(Enum):
    H2O = "h2o"
    SNAPKV = "snapkv"


@dataclass
class KVCacheCompressionParameters:
    """
    Contains parameters for KV cache compression algorithm.

    :param algorithm: The KV cache compression algorithm.
    :type algorithm: KVCacheCompressionMode
    :param granularity: The eviction granularity, such as 'per_token', 'per_group'.
    :type granularity: str
    :param group_size: The size of the group for the per_group strategy.
    :type group_size: int
    :param start_tokens: The number of tokens in the beginning of the cache (least recent)
        to be retained when applying KV cache compression.
    :type start_tokens: int
    :param recent_tokens: The number of most recent tokens to be retained when applying KV cache compression.
    :type recent_tokens: int
    :param intermediate_tokens: The number of tokens between the "start" and "recent" areas of KV cache that
        will be considered for eviction.
    :type intermediate_tokens: int
    :param normalize_scores: Whether to normalize the attention scores by the number of times each token was attended to.
    :type normalize_scores: bool
    :param window_size: The size of the importance score aggregation window
        (measured in token positions from the end of the prompt) to compute importance scores.
    :type window_size: Optional[int]
    """
    algorithm: KVCacheCompressionMode = KVCacheCompressionMode.SNAPKV
    granularity: str = "per_group"
    group_size: int = 32
    start_tokens: int = 32
    recent_tokens: int = 128
    intermediate_tokens: int = 512
    normalize_scores: bool = False
    window_size: Optional[int] = None


class KVCacheCompressor:
    def __init__(self, eviction_parameters: KVCacheCompressionParameters = KVCacheCompressionParameters()):
        self.start_tokens = eviction_parameters.start_tokens
        self.recent_tokens = eviction_parameters.recent_tokens
        self.intermediate_tokens = eviction_parameters.intermediate_tokens
        self.normalize_scores = eviction_parameters.normalize_scores
        self.granularity = eviction_parameters.granularity
        self.group_size = eviction_parameters.group_size if self.granularity == "per_group" else 1

        self.algorithm = eviction_parameters.algorithm
        self.window_size = eviction_parameters.window_size
        if self.algorithm == KVCacheCompressionMode.SNAPKV and self.window_size is None:
            self.window_size = 8  # Default window size for SnapKV

        self._scores = []
        self._cache_counter = [] if self.normalize_scores else None

        self._validate_arguments()

    def _validate_arguments(self):
        """
        Validates the arguments for the KV Cache compressor.
        Raises a ValueError at the end if any condition fails.
        """
        error_msg = None
        if self.start_tokens < 0 or self.recent_tokens < 0 or self.intermediate_tokens < 0:
            error_msg = "KV cache sizes must be non-negative integers."
        elif self.start_tokens + self.recent_tokens + self.intermediate_tokens <= 0:
            error_msg = "At least one of the KV cache sizes must be greater than zero."
        elif any(
            size % self.group_size != 0
            for size in (self.start_tokens, self.recent_tokens, self.intermediate_tokens)
        ):
            error_msg = "KV cache part sizes must be divisible by the group size."
        elif self.window_size is not None and self.algorithm == KVCacheCompressionMode.H2O:
            error_msg = "Window size is not supported for H2O algorithm."
        elif self.window_size is not None and self.window_size <= 0:
            error_msg = "Window size must be a positive integer if specified."
        elif self.granularity not in {"per_token", "per_group"}:
            error_msg = f"Granularity {self.granularity} is not supported. Supported granularities: 'per_token', 'per_group'."

        if error_msg:
            raise ValueError(error_msg)

    @property
    def max_cache_size(self) -> int:
        """
        Returns the maximum size of the KV cache.
        """
        return self.start_tokens + self.recent_tokens + self.intermediate_tokens

    def clean(self):
        """
        Resets the scores and cache counter.
        """
        self._scores = []
        self._cache_counter = [] if self.normalize_scores else None

    def aggregate_scores(self, layer_idx, attn_w):
        """
        Updates the scores based on the attention weights.
        """
        layer_scores = self._scores[layer_idx] if len(self._scores) > layer_idx else None

        if self.window_size is not None:
            hh_score = attn_w[..., -self.window_size :, :].sum(dim=(0, 2))  # sum over batch and query length
            if layer_scores is None:
                hh_score = torch.max_pool1d(hh_score, kernel_size=7, padding=7 // 2, stride=1)
        else:
            hh_score = attn_w.sum(dim=(0, 2))  # Sum over batch and query length, shape: (H, seq_len)

        hh_score = hh_score.sum(0, keepdim=True)  # Sum over all heads, shape: (1, seq_len)

        # Skip frozen start tokens in cache
        if hh_score.shape[1] > self.start_tokens:
            hh_score = hh_score[:, self.start_tokens :]

        num_new_tokens = hh_score.shape[-1] - (layer_scores.shape[-1] if layer_scores is not None else 0)
        if layer_scores is not None:
            hh_score[:, :-num_new_tokens] += layer_scores

        layer_scores = hh_score
        layer_counter = self._calculate_layer_counter(layer_idx, num_new_tokens)
        self._update_layer_scores(layer_idx, layer_scores, layer_counter)

    def _calculate_layer_counter(self, layer_idx, num_new_tokens):
        if not self.normalize_scores:
            return None

        new_count_size = num_new_tokens
        if self.window_size is not None:
            new_count_size = min(self.window_size, num_new_tokens)
        new_counters = torch.arange(new_count_size, 0, -1)

        if len(self._cache_counter) > layer_idx:
            layer_counter = self._cache_counter[layer_idx]
            layer_counter += new_count_size
            layer_counter = torch.cat((layer_counter, new_counters), dim=-1)
        else:
            if self.window_size is not None and num_new_tokens > self.window_size:
                full_window = torch.full((num_new_tokens - self.window_size,), self.window_size)
                layer_counter = torch.cat((full_window, new_counters), dim=0)
            else:
                layer_counter = new_counters
        return layer_counter

    def _update_layer_scores(self, layer_idx, layer_scores, layer_counter=None):
        if len(self._scores) <= layer_idx:
            self._scores.append(layer_scores)
            if self.normalize_scores:
                self._cache_counter.append(layer_counter.to(layer_scores.device))
        else:
            self._scores[layer_idx] = layer_scores
            if self.normalize_scores:
                self._cache_counter[layer_idx] = layer_counter.to(layer_scores.device)

    def get_scores(self, layer_idx):
        return (
            self._scores[layer_idx]
            if not self.normalize_scores
            else self._scores[layer_idx] / self._cache_counter[layer_idx]
        )

    def get_intermediate_page_scores(self):
        scores = self.get_scores()

        # Pad cache with zeros to make it multiple by group_size
        pad = scores.shape[-1] % self.group_size
        if pad:
            scores = F.pad(scores, (0, self.group_size - pad), mode="constant", value=0)

        group_scores = scores.view(self.num_heads_to_keep, -1, self.group_size)
        group_scores = group_scores.sum(-1)

        num_recent_groups = self.recent_tokens // self.group_size
        intermediate_group_scores = group_scores[:, :-num_recent_groups]

        return intermediate_group_scores

    def _convert_group_indices(self, group_indices, seq_len):
        heads, num_groups = group_indices.shape
        device = group_indices.device

        # Create relative indices within each group
        relative_idx = torch.arange(self.group_size, device=device).repeat(num_groups)
        relative_idx = relative_idx.view(1, num_groups, self.group_size)

        expanded_groups = group_indices.unsqueeze(-1).expand(-1, -1, self.group_size)
        indices = expanded_groups * self.group_size + relative_idx
        indices = indices.view(heads, -1)

        # Trim padding from the last group if needed
        remainder = seq_len % self.group_size
        if remainder:
            padded = self.group_size - remainder
            indices = indices[:, :-padded]

        return indices

    def get_remaining_indices(self, scores: torch.Tensor, kwargs: dict) -> torch.Tensor:
        """
        Computes the indices of the keep tokens in the KV cache after compression.
        """
        seq_len = self.start_tokens + scores.shape[-1]
        if self.granularity == "per_token":
            start_size = self.start_tokens
            intermediate_size = self.intermediate_tokens
            recent_size = self.recent_tokens
        elif self.granularity == "per_group":
            start_size = self.start_tokens // self.group_size
            intermediate_size = self.intermediate_tokens // self.group_size
            recent_size = self.recent_tokens // self.group_size

            pad = scores.shape[-1] % self.group_size
            if pad:
                scores = F.pad(scores, (0, self.group_size - pad), mode="constant", value=0)
            scores = scores.view(-1, self.group_size).sum(-1)  # Sum token scores inside group

        keep_groups = []
        size = scores.shape[0]
        if start_size > 0:
            keep_past = torch.arange(0, start_size, device=scores.device)
            keep_groups.append(keep_past)

        if intermediate_size > 0:
            intermediate_scores = scores[:size - recent_size]

            _, keep_coarse = torch.topk(intermediate_scores, intermediate_size, dim=-1)
            keep_coarse = keep_coarse.sort().values + start_size
            keep_groups.append(keep_coarse)

        if recent_size > 0:
            keep_recent = (
                torch.arange(size - recent_size, size, device=scores.device) + start_size
            )
            keep_groups.append(keep_recent)

        remaining_idx = torch.cat(keep_groups, dim=-1).unsqueeze(0)
        if self.granularity == "per_group":
            remaining_idx = self._convert_group_indices(remaining_idx, seq_len)

        return remaining_idx

    def compress(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compresses the keys and values of the KV cache based on the attention scores.
        """
        # Compute scores
        scores = self.get_scores(layer_idx)
        indices = self.get_remaining_indices(scores, kwargs)

        # Prune keys and values
        keep_heads = indices.shape[0]
        B, H, seq_len, head_dim = keys.shape
        mask = torch.zeros((keep_heads, seq_len), dtype=torch.bool).to(keys.device)  # shape (keep_heads, seq_len)
        mask = mask.scatter(-1, indices, 1)
        mask = mask.unsqueeze(0).unsqueeze(-1)

        keys = keys.masked_select(mask).view(B, H, -1, head_dim)
        values = values.masked_select(mask).view_as(keys)

        if self.algorithm in [KVCacheCompressionMode.H2O, KVCacheCompressionMode.SNAPKV]:
            score_mask = mask[0, :, self.start_tokens :, 0]  # shape (keep_heads, seq_len - self.start_tokens,)
            self._scores[layer_idx] = self._scores[layer_idx].masked_select(score_mask).view(keep_heads, -1)

            if self.normalize_scores:
                self._cache_counter[layer_idx] = self._cache_counter[layer_idx].masked_select(score_mask[0])

        return keys, values

    @torch.no_grad
    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        layer_idx = module.layer_idx
        cache = kwargs["past_key_values"]
        keys = cache.layers[layer_idx].keys
        values = cache.layers[layer_idx].values

        attn_weights = output[1]
        if attn_weights is None:
            raise RuntimeError(
                "Attention weights are None. Please switch to the `eager` attention implementation "
                "or specify `last_query_size` >= window_size in SparseAttention."
            )

        if layer_idx == 0 and attn_weights.shape[-2] != 1:
            self.clean()

        self.aggregate_scores(layer_idx, attn_weights)

        seq_len = keys.shape[-2]
        if seq_len > self.max_cache_size:
            keys, values = self.compress(layer_idx, keys, values, kwargs)

        cache.layers[layer_idx].keys = keys
        cache.layers[layer_idx].values = values
        return output

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        """
        Context manager to apply a compression method to a model.
        """
        hooks = []
        try:
            llm = model
            if hasattr(llm, "model"):
                llm = llm.model
            if hasattr(llm, "language_model"):
                llm = llm.language_model

            for layer in llm.layers:
                if getattr(layer.self_attn, "is_sliding", False):
                    logger.warning("Compression is skipped for layers with sliding window attention")
                    continue
                hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))
            yield
        finally:
            self.clean()
            for forward_hook in hooks:
                forward_hook.remove()
