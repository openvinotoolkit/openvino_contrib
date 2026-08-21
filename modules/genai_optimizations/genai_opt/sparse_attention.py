# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math
from collections.abc import Callable, Generator
from contextlib import contextmanager
from enum import Enum
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.models.phi3.modeling_phi3 import (
    apply_rotary_pos_emb as phi3_apply_rotary_pos_emb,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    apply_multimodal_rotary_pos_emb,
)

BLOCK_SPARSE_KERNEL_BLOCK_SIZE = 128


class AttentionMode(str, Enum):
    DENSE = "dense"
    TRI_SHAPE = "tri-shape"

    @classmethod
    def from_string(cls, value: str) -> "AttentionMode":
        try:
            return cls(value)
        except ValueError as error:
            raise ValueError(f"{value} is not a valid {cls.__name__}") from error


class SparseAttention:
    def __init__(
        self,
        algorithm: str,
        output_attentions: bool = False,
        last_query_size: int = 100,
        recent_size: int = 1024,
        block_size: int = 128,
    ):
        self.algorithm = AttentionMode.from_string(algorithm)
        self.output_attentions = output_attentions
        self.last_query_size = last_query_size
        self.recent_size = recent_size
        self.block_size = block_size
        self._validate_parameters()
        self._patched_layers = []
        self._original_attention_implementation = None

    def _validate_parameters(self) -> None:
        if self.block_size <= 0:
            raise ValueError(
                f"block_size must be greater than zero, got {self.block_size}"
            )
        if self.recent_size <= 0 or self.recent_size % self.block_size:
            raise ValueError(
                "recent_size must be a positive multiple of block_size, got "
                f"{self.recent_size} and {self.block_size}"
            )
        if self.last_query_size < 0:
            raise ValueError(
                f"last_query_size must be non-negative, got {self.last_query_size}"
            )
        if (
            self.algorithm is AttentionMode.TRI_SHAPE
            and self.block_size != BLOCK_SPARSE_KERNEL_BLOCK_SIZE
        ):
            raise ValueError(
                "tri-shape attention requires block_size=128 to match the "
                "Block-Sparse-Attention kernel"
            )

    def _attention_implementation(self) -> Callable:
        implementations = {
            AttentionMode.DENSE: self._dense_attention,
            AttentionMode.TRI_SHAPE: self._tri_shape_attention,
        }
        return implementations[self.algorithm]

    @staticmethod
    def _expand_kv(
        module: nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ):
        if key.shape[1] != query.shape[1]:
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        return key, value

    def _dense_attention(
        self,
        module: nn.Module,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float | None = None,
        **_kwargs,
    ):
        key_states, value_states = self._expand_kv(
            module, query_states, key_states, value_states
        )
        scale = scaling if scaling is not None else module.head_dim**-0.5
        weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * scale
        if attention_mask is not None:
            mask = attention_mask[..., : key_states.shape[-2]]
            if mask.ndim == 2:
                mask = mask[:, None, None, :]
            if mask.dtype == torch.bool:
                weights = weights.masked_fill(~mask, torch.finfo(weights.dtype).min)
            else:
                weights = weights + mask
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        output = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
        return output, weights if self.output_attentions else None

    @staticmethod
    def _has_additional_masking(
        attention_mask: torch.Tensor | None,
        q_len: int,
        k_len: int,
    ) -> bool:
        """Return whether a mask blocks or biases causally visible positions."""
        if attention_mask is None:
            return False

        mask = attention_mask[..., -q_len:, :k_len]
        if mask.shape[-2] not in (1, q_len):
            return True

        query_positions = torch.arange(q_len, device=mask.device) + max(
            k_len - q_len, 0
        )
        key_positions = torch.arange(k_len, device=mask.device)
        causally_visible = key_positions <= query_positions.unsqueeze(-1)
        causally_visible = causally_visible.view(
            *((1,) * (mask.ndim - 2)), q_len, k_len
        )

        if mask.dtype == torch.bool:
            additionally_masked = ~mask
        else:
            additionally_masked = mask != 0
        return bool(torch.any(additionally_masked & causally_visible).item())

    def _block_sparse_attention(
        self,
        module: nn.Module,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float | None,
        block_mask: torch.Tensor,
    ):
        key_states, value_states = self._expand_kv(
            module, query_states, key_states, value_states
        )
        batch_size, head_count, q_len, head_dim = query_states.shape
        k_len = key_states.shape[-2]

        returned_weights = None
        if self.output_attentions:
            tail_size = min(self.last_query_size or q_len, q_len)
            tail_mask = (
                attention_mask[..., -tail_size:, :]
                if attention_mask is not None
                else None
            )
            _, returned_weights = self._dense_attention(
                module,
                query_states[..., -tail_size:, :],
                key_states,
                value_states,
                tail_mask,
                scaling,
            )

        try:
            from block_sparse_attn import block_sparse_attn_func
        except ImportError as error:
            raise RuntimeError(
                "block_sparse_attn is required for sparse attention modes"
            ) from error

        query = query_states.transpose(1, 2).reshape(
            batch_size * q_len, head_count, head_dim
        )
        key = (
            key_states.transpose(1, 2)
            .reshape(batch_size * k_len, head_count, head_dim)
            .to(query.device)
        )
        value = (
            value_states.transpose(1, 2)
            .reshape(batch_size * k_len, head_count, head_dim)
            .to(query.device)
        )
        q_offsets = (
            torch.arange(batch_size + 1, dtype=torch.int32, device=query.device) * q_len
        )
        k_offsets = (
            torch.arange(batch_size + 1, dtype=torch.int32, device=query.device) * k_len
        )
        head_mask_type = torch.ones(head_count, dtype=torch.int32, device=query.device)

        output = block_sparse_attn_func(
            query,
            key,
            value,
            q_offsets,
            k_offsets,
            head_mask_type,
            None,
            block_mask.contiguous(),
            q_len,
            k_len,
            p_dropout=0.0,
            deterministic=True,
            softmax_scale=scaling,
            is_causal=True,
            return_attn_probs=False,
        )
        return output.view(batch_size, q_len, head_count, head_dim), returned_weights

    def _tri_shape_attention(
        self,
        module: nn.Module,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float | None = None,
        **_kwargs,
    ):
        q_len = query_states.shape[-2]
        k_len = key_states.shape[-2]
        if q_len <= self.recent_size:
            return self._dense_attention(
                module, query_states, key_states, value_states, attention_mask, scaling
            )
        if self._has_additional_masking(attention_mask, q_len, k_len):
            return self._dense_attention(
                module, query_states, key_states, value_states, attention_mask, scaling
            )

        q_blocks = math.ceil(q_len / self.block_size)
        k_blocks = math.ceil(k_len / self.block_size)
        q_indices = torch.arange(q_blocks, device=query_states.device).unsqueeze(1)
        k_indices = torch.arange(k_blocks, device=query_states.device).unsqueeze(0)
        diagonal_offset = max(k_blocks - q_blocks, 0)
        causal = k_indices <= q_indices + diagonal_offset
        recent_blocks = self.recent_size // self.block_size
        mask = causal & (k_indices > q_indices + diagonal_offset - recent_blocks)
        mask[:, 0] = True

        if self.last_query_size:
            padding = q_blocks * self.block_size - q_len
            dense_tail = math.ceil(
                (min(self.last_query_size, q_len) + padding) / self.block_size
            )
            mask[-dense_tail:] = causal[-dense_tail:]
        mask = mask.view(1, 1, q_blocks, k_blocks).expand(
            query_states.shape[0], query_states.shape[1], -1, -1
        )
        return self._block_sparse_attention(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
            mask,
        )

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        language_model = model
        if hasattr(language_model, "model"):
            language_model = language_model.model
        if hasattr(language_model, "language_model"):
            language_model = language_model.language_model

        adapter = get_custom_attn_forward(language_model)
        self._original_attention_implementation = (
            language_model.config._attn_implementation
        )
        language_model.config._attn_implementation = "eager"
        sentinel = object()
        try:
            for layer in language_model.layers:
                attention = layer.self_attn
                previous_interface = getattr(attention, "attn_interface", sentinel)
                self._patched_layers.append(
                    (attention, attention.forward, previous_interface, sentinel)
                )
                attention.attn_interface = self._attention_implementation()
                attention.forward = partial(adapter, module=attention)
            yield
        finally:
            for (
                attention,
                previous_forward,
                previous_interface,
                missing,
            ) in self._patched_layers:
                attention.forward = previous_forward
                if previous_interface is missing:
                    del attention.attn_interface
                else:
                    attention.attn_interface = previous_interface
            self._patched_layers.clear()
            language_model.config._attn_implementation = (
                self._original_attention_implementation
            )


def qwen2_vl_forward(
    module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: torch.LongTensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs,
):
    del position_ids, output_attentions, use_cache, kwargs
    batch_size, q_len, _ = hidden_states.shape
    query = (
        module.q_proj(hidden_states)
        .view(batch_size, q_len, -1, module.head_dim)
        .transpose(1, 2)
    )
    key = (
        module.k_proj(hidden_states)
        .view(batch_size, q_len, -1, module.head_dim)
        .transpose(1, 2)
    )
    value = (
        module.v_proj(hidden_states)
        .view(batch_size, q_len, -1, module.head_dim)
        .transpose(1, 2)
    )
    cos, sin = position_embeddings
    query, key = apply_multimodal_rotary_pos_emb(
        query, key, cos, sin, module.rope_scaling["mrope_section"]
    )
    if past_key_values is not None:
        key, value = past_key_values.update(
            key,
            value,
            module.layer_idx,
            {"sin": sin, "cos": cos, "cache_position": cache_position},
        )
    output, weights = module.attn_interface(
        module,
        query_states=query,
        key_states=key,
        value_states=value,
        attention_mask=attention_mask,
        scaling=module.scaling,
    )
    return module.o_proj(output.reshape(batch_size, q_len, -1).contiguous()), weights


def llama_forward(
    module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
):
    del kwargs
    input_shape = hidden_states.shape[:-1]
    projection_shape = (*input_shape, -1, module.head_dim)
    query = module.q_proj(hidden_states).view(projection_shape).transpose(1, 2)
    key = module.k_proj(hidden_states).view(projection_shape).transpose(1, 2)
    value = module.v_proj(hidden_states).view(projection_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    if past_key_values is not None:
        key, value = past_key_values.update(
            key,
            value,
            module.layer_idx,
            {"sin": sin, "cos": cos, "cache_position": cache_position},
        )
    output, weights = module.attn_interface(
        module,
        query_states=query,
        key_states=key,
        value_states=value,
        attention_mask=attention_mask,
        scaling=module.scaling,
    )
    return module.o_proj(output.reshape(*input_shape, -1).contiguous()), weights


def qwen3_forward(
    module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
):
    del kwargs
    input_shape = hidden_states.shape[:-1]
    projection_shape = (*input_shape, -1, module.head_dim)
    query = module.q_norm(
        module.q_proj(hidden_states).view(projection_shape)
    ).transpose(1, 2)
    key = module.k_norm(module.k_proj(hidden_states).view(projection_shape)).transpose(
        1, 2
    )
    value = module.v_proj(hidden_states).view(projection_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    if past_key_values is not None:
        key, value = past_key_values.update(
            key,
            value,
            module.layer_idx,
            {"sin": sin, "cos": cos, "cache_position": cache_position},
        )
    output, weights = module.attn_interface(
        module,
        query_states=query,
        key_states=key,
        value_states=value,
        attention_mask=attention_mask,
        scaling=module.scaling,
    )
    return module.o_proj(output.reshape(*input_shape, -1).contiguous()), weights


def phi3_forward(
    module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
):
    del kwargs
    input_shape = hidden_states.shape[:-1]
    projection_shape = (*input_shape, -1, module.head_dim)
    qkv = module.qkv_proj(hidden_states)
    query_end = module.config.num_attention_heads * module.head_dim
    key_end = query_end + module.num_key_value_heads * module.head_dim
    query = qkv[..., :query_end].view(projection_shape).transpose(1, 2)
    key = qkv[..., query_end:key_end].view(projection_shape).transpose(1, 2)
    value = qkv[..., key_end:].view(projection_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query, key = phi3_apply_rotary_pos_emb(query, key, cos, sin)
    if past_key_values is not None:
        key, value = past_key_values.update(
            key,
            value,
            module.layer_idx,
            {"sin": sin, "cos": cos, "cache_position": cache_position},
        )
    output, weights = module.attn_interface(
        module,
        query_states=query,
        key_states=key,
        value_states=value,
        attention_mask=attention_mask,
        scaling=module.scaling,
    )
    return module.o_proj(output.reshape(*input_shape, -1).contiguous()), weights


CUSTOM_ATTENTION_FORWARDS = {
    "LlamaModel": llama_forward,
    "MistralModel": llama_forward,
    "Qwen2Model": llama_forward,
    "Qwen3Model": qwen3_forward,
    "Phi3Model": phi3_forward,
    "Qwen2VLTextModel": qwen2_vl_forward,
    "Qwen2_5_VLTextModel": qwen2_vl_forward,
    "Qwen2VLForConditionalGeneration": qwen2_vl_forward,
    "Qwen2_5_VLForConditionalGeneration": qwen2_vl_forward,
    "LlamaForCausalLM": llama_forward,
    "MistralForCausalLM": llama_forward,
    "Qwen2ForCausalLM": llama_forward,
    "Qwen3ForCausalLM": qwen3_forward,
    "Phi3ForCausalLM": phi3_forward,
}


def get_custom_attn_forward(model: PreTrainedModel):
    architectures = getattr(model.config, "architectures", None)
    text_config = getattr(model.config, "text_config", None)
    text_architectures = getattr(text_config, "architectures", None)
    candidates = []
    if architectures:
        candidates.append(architectures[0])
    if text_architectures:
        candidates.append(text_architectures[0])
    candidates.append(type(model).__name__)
    for architecture in candidates:
        if architecture in CUSTOM_ATTENTION_FORWARDS:
            return CUSTOM_ATTENTION_FORWARDS[architecture]
    raise ValueError(f"Unsupported model class: {candidates[0]}")
