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
    RKV = "rkv"


class KVCacheRefinedSelection(Enum):
    KVCRUSH = "kvcrush"
    DIVERSEKV = "diversekv"


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
    :param refined_algorithm: The refined scoring strategy for selecting tokens within the intermediate region.
    :type refined_algorithm: KVCacheRefinedSelection
    :param refined_tokens: The number of tokens within the intermediate region that will be selected
        using a secondary - refined scoring strategy (e.g., KVCrush, DiverseKV algo).
        If set to 0 (default), the entire intermediate region is processed using the primary selection method.
    :type refined_tokens: int
    :param kvcrush_anchor: The anchor point for the KVCrush algorithm,
        which can be "alternate", "random", "zeros", "ones", or "mean". Defaults to "alternate".
    :type kvcrush_anchor: str
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
    refined_algorithm: Optional[KVCacheRefinedSelection] = None
    refined_tokens: int = 64
    kvcrush_anchor: str = "alternate"
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
        if self.algorithm != KVCacheCompressionMode.H2O and self.window_size is None:
            logger.info(f"Set window size for {self.algorithm} to 8")
            self.window_size = 8  # Default window size for SnapKV and RKV

        self.refined_algorithm = eviction_parameters.refined_algorithm
        self.refined_tokens = eviction_parameters.refined_tokens
        self.adaptive_refined_size = self.refined_algorithm is not None and self.refined_tokens == 0
        self.kvcrush_anchor = eviction_parameters.kvcrush_anchor
        self.attn_mass_threshold = 0.9

        self._scores = {}
        self._cache_counter = {} if self.normalize_scores else None

        self._validate_arguments()

    def _validate_arguments(self):
        """
        Validates the arguments for the KV Cache compressor.
        Raises a ValueError at the end if any condition fails.
        """
        error_msg = None
        if any(x < 0 for x in (
            self.start_tokens,
            self.recent_tokens,
            self.intermediate_tokens,
            self.refined_tokens,
        )):
            error_msg = "KV cache sizes must be non-negative integers."
        elif self.refined_tokens > self.intermediate_tokens:
            error_msg = "refined_tokens cannot be greater than intermediate_tokens."
        elif self.start_tokens + self.recent_tokens + self.intermediate_tokens <= 0:
            error_msg = "At least one of the KV cache sizes must be greater than zero."
        elif any(
            size % self.group_size != 0
            for size in (self.start_tokens, self.recent_tokens, self.intermediate_tokens, self.refined_tokens)
        ):
            error_msg = "KV cache part sizes must be divisible by the group size."
        elif self.window_size is not None and self.algorithm == KVCacheCompressionMode.H2O:
            error_msg = "Window size is not supported for H2O algorithm."
        elif self.window_size is not None and self.window_size <= 0:
            error_msg = "Window size must be a positive integer if specified."
        elif self.granularity not in {"per_token", "per_group"}:
            error_msg = f"Granularity {self.granularity} is not supported. Supported granularities: 'per_token', 'per_group'."
        elif self.kvcrush_anchor not in {"random", "zeros", "ones", "mean", "alternate"}:
            error_msg = (
                f"Unknown KVCrush anchor: {self.kvcrush_anchor}. "
                "Supported anchors: 'random', 'zeros', 'ones', 'mean', 'alternate'."
            )

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
        self._scores = {}
        self._cache_counter = {} if self.normalize_scores else None

    def aggregate_scores(self, layer_idx, attn_w):
        """
        Updates the scores based on the attention weights.
        """
        if self.algorithm == KVCacheCompressionMode.RKV:
            return self._update_rkv_scores(layer_idx, attn_w)

        layer_scores = self._scores.get(layer_idx, None)

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
        layer_counter = self._calculate_layer_counter(layer_idx, num_new_tokens, device=attn_w.device)

        self._scores[layer_idx] = layer_scores
        if self.normalize_scores:
            self._cache_counter[layer_idx] = layer_counter

    def _calculate_layer_counter(self, layer_idx, num_new_tokens, device):
        if not self.normalize_scores:
            return None

        new_count_size = num_new_tokens
        if self.window_size is not None:
            new_count_size = min(self.window_size, num_new_tokens)
        new_counters = torch.arange(new_count_size, 0, -1, device=device)

        if len(self._cache_counter) > layer_idx:
            layer_counter = self._cache_counter[layer_idx]
            layer_counter += new_count_size
            layer_counter = torch.cat((layer_counter, new_counters), dim=-1)
        else:
            if self.window_size is not None and num_new_tokens > self.window_size:
                full_window = torch.full((num_new_tokens - self.window_size,), self.window_size, device=device)
                layer_counter = torch.cat((full_window, new_counters), dim=0)
            else:
                layer_counter = new_counters
        return layer_counter

    def _update_rkv_scores(self, layer_idx: int, attn_w: torch.Tensor) -> None:
        """
        Updates the scores for the decoding phase like in R-KV and RPC papers.
        """
        hh_score = attn_w.sum(0)  # Sum over batch, shape: (H, q_len, k_len)

        layer_scores = self._scores.get(layer_idx, None)
        if layer_scores is None:
            self._scores[layer_idx] = hh_score
        else:
            layer_scores = self._scores[layer_idx]
            new_tokens = hh_score.shape[-1] - layer_scores.shape[-1]
            self._scores[layer_idx] = torch.cat(
                (
                    F.pad(layer_scores, (0, new_tokens), mode="constant", value=0),
                    hh_score,
                ),
                dim=-2,
            )

        # Keep only the last `window_size` scores
        if self._scores[layer_idx].shape[1] > self.window_size:
            self._scores[layer_idx] = self._scores[layer_idx][:, -self.window_size :, :]

    def get_scores(self, layer_idx):
        if self._scores[layer_idx].dim() == 2:
            return (
                self._scores[layer_idx]
                if not self.normalize_scores
                else self._scores[layer_idx] / self._cache_counter[layer_idx]
            )

        # Average over query length, shape: (H, k_len)
        scores = self._scores[layer_idx].mean(dim=-2)
        scores = F.max_pool1d(
            scores,
            kernel_size=7,
            padding=7 // 2,
            stride=1,
        )
        del self._scores[layer_idx]  # Clear scores after retrieval
        return scores.mean(0, keepdim=True)[:, self.start_tokens :]  # Average over heads, shape: (1, k_len)

    def _get_keys_similarity(self, key_states):
        keys_normalized = key_states / key_states.norm(dim=-1, keepdim=True)
        similarity = torch.matmul(keys_normalized, keys_normalized.transpose(-1, -2))
        similarity = similarity[:, :, self.start_tokens :, self.start_tokens :]
        # Aggregate over batch
        similarity = similarity.mean(dim=0)

        for h in range(similarity.shape[0]):
            similarity[h].fill_diagonal_(0.0)

        # Zero out values below mean similarity for each head
        head_means = similarity.view(similarity.shape[0], -1).mean(dim=-1, keepdim=True)
        thr = head_means.unsqueeze(-1)
        similarity = torch.where(similarity >= thr, similarity, torch.zeros_like(similarity))

        # Aggregate over heads
        similarity = similarity.mean(dim=0)
        return similarity

    def get_refined_indices(self, scores: torch.Tensor, kwargs: dict) -> torch.Tensor:
        device = scores.device
        refined_size = self.refined_tokens // self.group_size
        if self.refined_algorithm == KVCacheRefinedSelection.KVCRUSH:
            B, _ = scores.shape
            if B != 1:
                error_msg = "KVCacheCompressor with KVCrush algorithm supports only batch size of 1."
                raise ValueError(error_msg)

            scores_flat = scores.view(-1)
            refined_mask = scores_flat != float("-inf")
            keepable_scores = scores_flat[refined_mask]

            # Binary vector: top 50% → 1, bottom 50% → 0
            num_zeros = keepable_scores.numel() // 2
            _, low_idx = torch.topk(keepable_scores, num_zeros, largest=False)
            binary_vector = torch.ones_like(keepable_scores, dtype=torch.int)
            binary_vector[low_idx] = 0

            # Place binary_vector back into full-length binary tensor
            full_binary = torch.zeros_like(scores_flat, dtype=torch.int, device=device)
            full_binary[refined_mask] = binary_vector

            if self.granularity == "per_group":
                full_binary = full_binary.view(-1, self.group_size)
                num_groups = full_binary.shape[0]

                if self.kvcrush_anchor == "random":
                    anchor_point = torch.randint(0, 2, (num_groups,), device=device)
                elif self.kvcrush_anchor == "zeros":
                    anchor_point = torch.zeros(num_groups, device=device)
                elif self.kvcrush_anchor == "ones":
                    anchor_point = torch.ones(num_groups, device=device)
                elif self.kvcrush_anchor == "mean":
                    mean_point = full_binary.float().mean(dim=1)
                    anchor_point = (mean_point > 0.5).int()
                elif self.kvcrush_anchor == "alternate":
                    anchor_point = torch.zeros(num_groups, device=device)
                    anchor_point[1::2] = 1

                hamming_distance = torch.sum(
                    full_binary != anchor_point.unsqueeze(1), dim=1
                ).float()  # shape: [num_groups]
                refined_group_mask = refined_mask.view(-1, self.group_size)[:, 0]
                hamming_distance[~refined_group_mask] = float("-inf")  # Set invalid indices to -inf

                sorted_dist_idx = torch.argsort(hamming_distance, descending=True)

                # Select evenly spaced indices using linspace (representative)
                num_valid = keepable_scores.numel() // self.group_size
                rep_indices = torch.linspace(
                    0, num_valid - 1, steps=refined_size, dtype=torch.long, device=device
                )
                assert rep_indices.numel() == refined_size
                refined_topk = sorted_dist_idx[rep_indices]  # shape: [refined_groups]

                return refined_topk

            # Anchor: shape [L]
            if self.kvcrush_anchor == "random":
                anchor = torch.randint_like(keepable_scores, low=0, high=2, device=device)
            elif self.kvcrush_anchor == "zeros":
                anchor = torch.zeros_like(keepable_scores, dtype=torch.int, device=device)
            elif self.kvcrush_anchor == "ones":
                anchor = torch.ones_like(keepable_scores, dtype=torch.int, device=device)
            elif self.kvcrush_anchor == "mean":  # equal to binary_vector in per-token case
                error_msg = (
                    "Mean anchor is not supported for KVCrush in per-token mode. "
                    "Please use 'random', 'zeros', 'ones' or 'alternate' anchors."
                )
                raise ValueError(error_msg)
            elif self.kvcrush_anchor == "alternate":
                anchor = torch.zeros_like(keepable_scores, dtype=torch.int, device=device)
                anchor[1::2] = 1

            full_anchor = torch.zeros_like(scores_flat, dtype=torch.int)
            full_anchor[refined_mask] = anchor

            # Hamming distance (1D): count bits different from anchor
            hamming_distance = (full_binary != full_anchor).float()
            hamming_distance[~refined_mask] = float("-inf")  # Set invalid indices to -inf

            # Sort valid indices by distance to anchor (more diverse first)
            sorted_dist_idx = torch.argsort(hamming_distance, descending=True)

            # Select evenly spaced indices using linspace (representative)
            num_valid = keepable_scores.numel()
            rep_indices = torch.linspace(
                0, num_valid - 1, steps=refined_size, dtype=torch.long, device=device
            )
            assert rep_indices.numel() == refined_size
            refined_topk = sorted_dist_idx[rep_indices].unsqueeze(0)  # shape: [1, refined_tokens]

        elif self.refined_algorithm == KVCacheRefinedSelection.DIVERSEKV:
            keys = kwargs.get("keys")
            similarity = self._get_keys_similarity(keys)
            n = scores.shape[-1]
            similarity = similarity[:n, :n]  # Only intermediate part

            selected_mask = scores[0] == float("-inf")
            similarity_to_selected = similarity[:, selected_mask]
            diversity = -similarity_to_selected.mean(dim=-1)  # diverse = low sim to selected

            if self.granularity == "per_group":
                diversity = diversity.view(-1, self.group_size).sum(dim=-1)
                scores = scores.view(-1, self.group_size).sum(dim=-1)  # Sum token scores inside group
                # mask for already selected tokens (scores == -inf)
            diversity[scores.view(-1) == float("-inf")] = float("-inf")
            _, refined_topk = torch.topk(diversity, refined_size, dim=-1)

        return refined_topk

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

    def _set_balanced_refined_size(self, intermediate_scores):
        target_mass = self.attn_mass_threshold * intermediate_scores.sum(dim=-1)
        vals, _ = torch.sort(intermediate_scores, descending=True, dim=-1)
        cumsum = vals.cumsum(dim=-1)
        cutoff = (cumsum >= target_mass).nonzero(as_tuple=False)
        # Minimum number of groups to cover the target mass
        k_min = cutoff[0].item() + 1  # +1 because indices are 0-based
        self.refined_tokens = max(0, self.intermediate_tokens - k_min * self.group_size)

    def get_remaining_indices(self, scores: torch.Tensor, kwargs: dict) -> torch.Tensor:
        """
        Computes the indices of the keep tokens in the KV cache after compression.
        """
        seq_len = self.start_tokens + scores.shape[-1]
        start_size = self.start_tokens // self.group_size
        intermediate_size = self.intermediate_tokens // self.group_size
        recent_size = self.recent_tokens // self.group_size

        if self.granularity == "per_group":
            pad = scores.shape[-1] % self.group_size
            if pad:
                scores = F.pad(scores, (0, self.group_size - pad), mode="constant", value=0)
            padded_scores = scores.view(-1, self.group_size).sum(-1)  # Sum token scores inside group
        else:
            padded_scores = scores.squeeze(0)
        intermediate_scores = padded_scores[:-recent_size] if recent_size > 0 else padded_scores

        if self.adaptive_refined_size:
            self._set_balanced_refined_size(intermediate_scores)
        refined_size = self.refined_tokens // self.group_size
        coarse_size = intermediate_size - refined_size

        keep_groups = []
        if start_size > 0:
            keep_past = torch.arange(0, start_size, device=scores.device)
            keep_groups.append(keep_past)

        if intermediate_size > 0:
            if coarse_size > 0:
                _, keep_coarse = torch.topk(intermediate_scores, coarse_size, dim=-1)
                keep_coarse = keep_coarse.sort().values + start_size
                keep_groups.append(keep_coarse)

            if refined_size > 0:
                refined_scores = scores[:, :len(intermediate_scores) * self.group_size]

                if coarse_size > 0:
                    coarse_idx = keep_coarse.unsqueeze(0) - start_size
                    mask = torch.zeros_like(refined_scores, dtype=torch.bool)

                    if self.granularity == "per_group":
                        coarse_idx = self._convert_group_indices(
                            coarse_idx, coarse_idx.shape[-1] * self.group_size
                        )

                    mask.scatter_(1, coarse_idx, True)  # Ensure no OOB here
                    refined_scores = refined_scores.masked_fill(mask, float("-inf"))

                refined_topk = self.get_refined_indices(refined_scores, kwargs) + start_size
                keep_groups.append(refined_topk)

        if recent_size > 0:
            padded_len = padded_scores.shape[0]
            keep_recent = (
                torch.arange(padded_len - recent_size, padded_len, device=scores.device) + start_size
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
            if self.refined_algorithm == KVCacheRefinedSelection.DIVERSEKV:
                kwargs["keys"] = keys

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
