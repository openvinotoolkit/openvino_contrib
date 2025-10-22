# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# This logic is largely copied from the https://github.com/mit-han-lab/x-attention

from functools import partial
import string
from typing import Any, Callable, Generator, Optional, Tuple
from enum import Enum
from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import repeat_kv
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.models.phi3.modeling_phi3 import apply_rotary_pos_emb as phi3_apply_rotary_pos_emb
from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb

from block_sparse_attn import block_sparse_attn_func


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return str(self.value)

    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list[Any]) -> Any:
        return name.lower()

    @classmethod
    def from_string(cls, value: str) -> "StrEnum":
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"{value} is not a valid {cls.__name__}")


class AttentionMode(StrEnum):
    """
    Defines attention modes.
    :param DENSE: Standard dense eager attention.
    :param TRI_SHAPE: Tri-shaped sparse attention.
    :param XATTN: Dynamic block sparse attention (X-Attention https://arxiv.org/pdf/2503.16428).
    """

    DENSE = "dense"
    TRI_SHAPE = "tri-shape"
    XATTENTION = "x-attention"


class SparseAttention:
    def __init__(
        self,
        algorithm: string,
        output_attentions: bool = False,
        threshold: float = 0.8,
        last_query_size: int = 100,
        recent_size: int = 1024,
        block_size: int = 128,
    ):
        """
        :param algorithm: The attention algorithm to use.
        :param output_attentions: Whether to return attention weights.
        :param threshold: The threshold for attention mass for X-Attention mode.
        :param last_query_size: The number of the last tokens to apply dense attention.
        :param recent_size: The size of the recent window size.
        :param block_size: The size of the blocks for block sparse attention.
        """
        self.algorithm = AttentionMode.from_string(algorithm)
        self.output_attentions = output_attentions

        self.last_query_size = last_query_size # used in tri-shape and x-attention
        self.block_size = block_size # used in tri-shape and x-attention
        self.recent_size = recent_size # used in tri-shape attention
        self.threshold = threshold
        self._validate_params()

        self._original_attn_forwards = []
        self._original_attn_implementation = None

    def _validate_params(self):
        if self.recent_size <= 0:
            raise ValueError(f"recent_size should be greater than 0, but got recent_size={self.recent_size}")
        if self.recent_size % self.block_size != 0:
            raise ValueError(f"recent_size should be multiple of block_size, but got recent_size={self.recent_size}, block_size={self.block_size}")
        if self.algorithm == AttentionMode.XATTENTION and (self.threshold <= 0 or self.threshold > 1):
            raise ValueError(f"A threshold value used to determine the minimum attention weight sum should be greater than 0 and less than or equal to 1, but found {self.threshold}")
        if self.last_query_size < 0:
            raise ValueError(f"last_query_size should be non-negative, but got last_query_size={self.last_query_size}")

    def _get_custom_impl(self) -> Callable:
        if self.algorithm == AttentionMode.XATTENTION:
            return partial(self._xattention_forward, output_attentions=self.output_attentions)
        elif self.algorithm == AttentionMode.TRI_SHAPE:
            return partial(self._trishape_forward, output_attentions=self.output_attentions)
        elif self.algorithm == AttentionMode.DENSE:
            return partial(self._dense_forward, output_attentions=self.output_attentions)
        else:
            error_msg = f"Unsupported KV cache prefill mode: {self.algorithm}"
            raise ValueError(error_msg)

    @staticmethod
    def _prepare_kv(
            module: nn.Module,
            query_states: torch.Tensor,
            key_states: torch.Tensor,
            value_states: torch.Tensor
    ):
        if key_states.shape[1] != query_states.shape[1]:
            key_states = repeat_kv(key_states, module.num_key_value_groups)
            value_states = repeat_kv(value_states, module.num_key_value_groups)
        return key_states, value_states

    def _dense_forward(
        self,
        module: nn.Module,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: Optional[float] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        key_states, value_states = SparseAttention._prepare_kv(module, query_states, key_states, value_states)

        if scaling is None:
            scaling = module.head_dim**-0.5

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights += attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
        return attn_output, attn_weights if self.output_attentions else None

    def _sparse_forward_common(
        self,
        module: nn.Module,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # [optional] compute attn_scores
        attn_weights = None
        if self.output_attentions:
            q2 = query_states[:, :, -self.last_query_size:]
            mask2 = attention_mask[:, :, -self.last_query_size:] if attention_mask is not None else None
            _, attn_weights = self._dense_forward(module, q2, key_states, value_states, mask2, scaling)

        batch_size, num_heads, q_len, head_dim = query_states.shape
        k_len = key_states.shape[-2]
        q_block_num = (q_len + self.block_size - 1) // self.block_size
        k_block_num = (k_len + self.block_size - 1) // self.block_size

        # reshape Q/K/V to [seq, head, dim]
        q = query_states.transpose(1, 2).reshape(q_len, num_heads, head_dim)
        k = key_states.transpose(1, 2).reshape(k_len, num_heads, head_dim).to(q.device)
        v = value_states.transpose(1, 2).reshape(k_len, num_heads, head_dim).to(q.device)

        q_cu_seq_lens = torch.tensor([0, q_len], dtype=torch.int32, device=q.device)
        k_cu_seq_lens = torch.tensor([0, k_len], dtype=torch.int32, device=q.device)
        head_mask_type = torch.ones(num_heads, dtype=torch.int32, device=q.device)

        attn_output = block_sparse_attn_func(
            q, k, v,
            q_cu_seq_lens, k_cu_seq_lens,
            head_mask_type, None,
            mask[:, :, :q_block_num, :k_block_num].contiguous(),
            q_len, k_len,
            p_dropout=0.0,
            deterministic=True,
            is_causal=True,
            return_attn_probs=False,
        )

        attn_output = attn_output.view(batch_size, q_len, num_heads, head_dim)
        return attn_output, attn_weights

    def _trishape_forward(
        self,
        module: nn.Module,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        **kwargs,
    ):
        _, num_heads, q_len, _ = query_states.shape
        # decoding → fallback to dense
        if q_len <= self.recent_size:
            return self._dense_forward(module, query_states, key_states, value_states, attention_mask, scaling)

        key_states, value_states = SparseAttention._prepare_kv(module, query_states, key_states, value_states)

        # build tri-shape mask
        k_block_num = q_block_num = (q_len + self.block_size - 1) // self.block_size
        device = query_states.device

        mask = torch.zeros((1, num_heads, q_block_num, k_block_num), dtype=torch.bool, device=device)

        # keep sink tokens
        mask[..., 0] = True

        # keep recent
        local_block_num = (self.recent_size + self.block_size - 1) // self.block_size
        q_idx = torch.arange(q_block_num, device=device).unsqueeze(1)
        k_idx = torch.arange(k_block_num, device=device).unsqueeze(0)
        recent_mask = ((k_idx <= q_idx) & (k_idx > q_idx - local_block_num)).to(device).unsqueeze(0).unsqueeze(0).expand(1, num_heads, -1, -1)
        mask = mask | recent_mask

        # keep last queries
        padded_len = q_block_num * self.block_size - q_len
        last_blocks = (self.last_query_size + padded_len + self.block_size - 1) // self.block_size
        last_mask = (k_idx <= q_idx) & (q_idx >= q_block_num - last_blocks).to(device).unsqueeze(0).unsqueeze(0).expand(1, num_heads, -1, -1)
        mask = mask | last_mask

        return self._sparse_forward_common(module, query_states, key_states, value_states, attention_mask, scaling, mask)

    def _xattention_forward(
        self,
        module: nn.Module,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        **kwargs,
    ):
        # decoding → fallback to dense
        if query_states.shape[-2] == 1:
            return self._dense_forward(module, query_states, key_states, value_states, attention_mask, scaling)

        key_states, value_states = SparseAttention._prepare_kv(module, query_states, key_states, value_states)
        k_len = key_states.shape[2]
        chunk_size = int(
            max(
                min(
                    max(2048, 1 << (k_len - 1).bit_length()),  # next power of two ≥ k_len
                    128 * 1024 * 2048 // (1 << (k_len - 1).bit_length()),  # # upper bound
                ),
                2048,  # lower bound
            )
        )

        _, mask = self._xattn_estimate(
            query_states,
            key_states,
            stride=16,
            chunk_size=chunk_size,
            keep_sink=False,
            keep_recent=False,
        )

        num_heads = query_states.shape[1]
        if mask.shape[1] != num_heads:
            mask = mask.expand(-1, num_heads, -1, -1)

        return self._sparse_forward_common(module, query_states, key_states, value_states, attention_mask, scaling, mask)

    def _find_blocks_chunked(
        self,
        input_tensor: torch.Tensor,
        current_index: int,
        decoding: bool,
        mode: str = "both",
        causal: bool = True
    ):
        """
        Finds and selects relevant blocks of attention for transformer-based models based on a
        threshold or a predefined number of blocks.

        Parameters
        ----------
        - input_tensor (torch.Tensor): The input tensor of shape (batch_size, head_num, chunk_num, block_num).
        - current_index (int): The current index in the sequence processing.
        - decoding (bool): If True, operates in decoding mode; otherwise, it's in encoding mode.
        - mode (str): Defines the processing mode, either 'both', 'prefill', or 'decode'.
        - causal (bool): If True, applies causal masking to prevent future information leakage.

        Returns
        -------
        - torch.Tensor: A boolean mask of shape (batch_size, head_num, chunk_num, block_num),
        indicating which blocks should be attended to.
        """
        batch_size, head_num, chunk_num, block_num = input_tensor.shape

        if mode == "prefill" and decoding:
            return torch.ones_like(input_tensor, dtype=torch.bool)
        if mode == "decode" and not decoding:
            mask = torch.ones_like(input_tensor, dtype=torch.bool)
            if causal:
                mask[:, :, :, current_index : current_index + chunk_num] = torch.tril(
                    torch.ones(1, head_num, chunk_num, chunk_num, device=input_tensor.device)
                )
                mask[:, :, current_index + chunk_num :, :] = 0
                return torch.cat(
                    [
                        torch.ones_like(input_tensor, dtype=torch.bool)[:, :, 0 : current_index + 1],
                        torch.zeros_like(input_tensor, dtype=torch.bool)[:, :, current_index + 1 :],
                    ],
                    dim=-1,
                )
            else:
                return mask
        input_tensor = input_tensor.to(float)

        total_sum = input_tensor.sum(dim=-1, keepdim=True)
        if isinstance(self.threshold, torch.Tensor):
            self.threshold = self.threshold.to(float)
            required_sum = total_sum * self.threshold.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(
                (batch_size, head_num, chunk_num, 1)
            ).to(input_tensor.device)
        else:
            required_sum = total_sum * self.threshold
        if causal:
            mask = torch.zeros_like(input_tensor, dtype=torch.bool)
            mask[:, :, :, 0] = 1  # keep the first key block
            mask[:, :, :, current_index : current_index + chunk_num] = (  # keep the diagonal block
                torch.eye(chunk_num, device=mask.device).unsqueeze(0).unsqueeze(0).expand(1, head_num, chunk_num, chunk_num)
            )
            other_values = input_tensor.masked_fill(mask, 0)
            sorted_values, _ = torch.sort(other_values, dim=-1, descending=True)
            sorted_values = sorted_values.to(input_tensor.device)

            sorted_values = torch.cat(
                [
                    torch.zeros((batch_size, head_num, chunk_num, 1), device=input_tensor.device),  # cumulative logic
                    torch.where(mask, input_tensor, 0).sum(dim=-1, keepdim=True),  # diagonal values
                    sorted_values[:, :, :, :-2],
                ],
                dim=-1,
            )

            _, index = torch.sort(
                torch.where(mask, 100000 * (1 + input_tensor), input_tensor),
                dim=-1,
                descending=True,
            )
            cumulative_sum_without_self = torch.cat(
                [
                    torch.zeros((batch_size, head_num, chunk_num, 1), device=input_tensor.device),
                    sorted_values[:, :, :, 0:-1],
                ],
                dim=-1,
            ).cumsum(dim=-1)

            index_mask = cumulative_sum_without_self < required_sum
            index = torch.where(index_mask, index, 0)
            mask = mask.view(batch_size, head_num * chunk_num, block_num)
            index = index.view(batch_size, head_num * chunk_num, block_num)
            mask[:, torch.arange(mask.shape[1], device=mask.device).unsqueeze(dim=-1), index] = True
            mask = mask.view(batch_size, head_num, chunk_num, block_num)
        else:
            mask = torch.zeros_like(input_tensor, dtype=torch.bool)
            sorted_values, index = torch.sort(input_tensor, dim=-1, descending=True)
            sorted_values = sorted_values.to(input_tensor.device)
            cumulative_sum_without_self = torch.cat(
                [
                    torch.zeros((batch_size, head_num, chunk_num, 1), device=input_tensor.device),
                    sorted_values[:, :, :, 0:-1],
                ],
                dim=-1,
            ).cumsum(dim=-1)
            index_mask = cumulative_sum_without_self < required_sum
            index = torch.where(index_mask, index, 0)
            mask = mask.view(batch_size, head_num * chunk_num, block_num)
            index = index.view(batch_size, head_num * chunk_num, block_num)
            mask[
                :,
                torch.arange(mask.shape[1], device=mask.device).unsqueeze(dim=-1),
                index,
            ] = True
            mask = mask.view(batch_size, head_num, chunk_num, block_num)

        try:
            if causal:
                assert (~mask[:, :, :, current_index + chunk_num :]).all()
        except Exception:
            mask[:, :, :, current_index + chunk_num :] = False

        if causal:
            if decoding:
                assert mask[:, :, :, 0].all() and mask[:, :, :, -1].all()
            else:
                lambda_mask = torch.zeros_like(input_tensor, dtype=bool, device=input_tensor.device)
                lambda_mask[:, :, :, 0] = 1
                lambda_mask[:, :, :, current_index : current_index + chunk_num] = (
                    torch.eye(chunk_num, device=lambda_mask.device)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(1, head_num, chunk_num, chunk_num)
                )
                assert torch.where(lambda_mask, mask, True).all()

        return mask

    def _xattn_estimate(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        stride: int = 16,
        chunk_size: int = 2048,
        keep_sink: bool = False,
        keep_recent: bool = False,
    ) -> torch.Tensor:
        batch_size, num_kv_head, k_len, head_dim = key_states.shape
        batch_size, num_q_head, q_len, head_dim = query_states.shape
        assert num_q_head == num_kv_head

        k_num_to_pad = ((k_len + chunk_size - 1) // chunk_size) * chunk_size - k_len
        q_num_to_pad = ((q_len + chunk_size - 1) // chunk_size) * chunk_size - q_len
        k_chunk_num = (k_len + k_num_to_pad) // chunk_size
        k_block_num = (k_len + k_num_to_pad) // self.block_size
        q_chunk_num = (q_len + q_num_to_pad) // chunk_size
        q_block_num = (q_len + q_num_to_pad) // self.block_size

        pad_key_states = F.pad(key_states, (0, 0, 0, k_num_to_pad), value=0) if k_num_to_pad > 0 else key_states
        pad_query_states = F.pad(query_states, (0, 0, 0, q_num_to_pad), value=0) if q_num_to_pad > 0 else query_states

        assert num_kv_head == num_q_head
        attn_sum_list = []
        simple_mask_list = []

        reshaped_chunk_size = chunk_size // stride
        reshaped_block_size = self.block_size // stride
        k_reshaped_num_to_pad = k_num_to_pad // stride
        k_reshaped_seq_len = (k_len + k_num_to_pad) // stride
        q_reshaped_num_to_pad = q_num_to_pad // stride
        num_blocks_per_chunk = reshaped_chunk_size // reshaped_block_size

        reshaped_key = torch.cat([(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1)
        reshaped_query = torch.cat(
            [(pad_query_states[:, :, (stride - 1 - q) :: stride, :]) for q in range(stride)],
            dim=-1,
        )
        assert reshaped_key.shape[-2] == k_reshaped_seq_len

        for chunk_idx in range(q_chunk_num):
            chunked_query = reshaped_query[
                :,
                :,
                (chunk_idx * reshaped_chunk_size) : (chunk_idx * reshaped_chunk_size + reshaped_chunk_size),
                :,
            ]
            attn_weights_slice = torch.matmul(chunked_query, reshaped_key.transpose(2, 3))

            attn_weights_slice = attn_weights_slice / (head_dim**0.5) / stride

            causal_mask = torch.zeros(
                (
                    batch_size,
                    num_q_head,
                    reshaped_chunk_size,
                    reshaped_chunk_size * k_chunk_num,
                ),
                device=key_states.device,
            )
            causal_mask[:, :, :, (-k_reshaped_num_to_pad):] = float("-inf")
            chunk_start = chunk_idx * reshaped_chunk_size
            chunk_end = chunk_start + reshaped_chunk_size
            causal_mask[:, :, :, chunk_start:chunk_end] = torch.triu(
                torch.ones(
                    1,
                    num_q_head,
                    reshaped_chunk_size,
                    reshaped_chunk_size,
                    device=key_states.device,
                )
                * float("-inf"),
                diagonal=1,
            )

            if chunk_idx == q_chunk_num - 1 and q_reshaped_num_to_pad != 0:
                causal_mask[:, :, (-(q_reshaped_num_to_pad)):, :] = float("-inf")

            causal_mask[:, :, :, chunk_end:] = float("-inf")
            attn_weights_slice = attn_weights_slice + causal_mask.to(attn_weights_slice.device)

            attn_weights_slice = F.softmax(attn_weights_slice, dim=-1, dtype=torch.float32).to(pad_query_states.dtype)

            if chunk_idx == q_chunk_num - 1 and q_reshaped_num_to_pad != 0:
                attn_weights_slice[:, :, -q_reshaped_num_to_pad:, :] = 0

            attn_sum = (
                attn_weights_slice.view(
                    batch_size,
                    num_kv_head,
                    num_blocks_per_chunk,
                    reshaped_block_size,
                    -1,
                    reshaped_block_size,
                )
                .sum(dim=-1)
                .sum(dim=-2)
                .sum(dim=1, keepdim=True)  # aggregation accross heads
            )  # attention mass per block
            del chunked_query

            simple_mask = self._find_blocks_chunked(
                input_tensor=attn_sum,
                current_index=k_block_num - q_block_num + chunk_idx * num_blocks_per_chunk,
                decoding=False,
                mode="prefill",
                causal=True,
            )

            attn_sum_list.append(attn_sum)
            simple_mask_list.append(simple_mask)

            del attn_weights_slice

        del reshaped_query, reshaped_key
        attn_sums = torch.cat(attn_sum_list, dim=-2)
        simple_masks = torch.cat(simple_mask_list, dim=-2)

        simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
            torch.tril(
                torch.ones(q_block_num, q_block_num, dtype=bool, device=key_states.device),
                diagonal=0,
            ),
            simple_masks[:, :, -q_block_num:, -q_block_num:],
            False,
        )
        if keep_sink:
            simple_masks[..., 0] = True

        if keep_recent:
            eye_matrix = torch.eye(q_block_num, device=simple_masks.device, dtype=bool)
            num_aggregated_heads = simple_masks.shape[1]
            eye_matrix_expanded = eye_matrix.unsqueeze(0).unsqueeze(0).expand(1, num_aggregated_heads, q_block_num, q_block_num)
            simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
                eye_matrix_expanded, True, simple_masks[:, :, -q_block_num:, -q_block_num:]
            )

        if self.last_query_size > 0:
            q_blocks_to_keep = (self.last_query_size + self.block_size - 1) // self.block_size
            q_rows = torch.arange(q_block_num, device=simple_masks.device)
            q_keep_mask = q_rows >= (q_block_num - q_blocks_to_keep)
            q_keep_mask = q_keep_mask.view(1, 1, q_block_num, 1)

            num_aggregated_heads = simple_masks.shape[1]
            q_keep_mask = q_keep_mask.expand(1, num_aggregated_heads, q_block_num, k_block_num)
            simple_masks[:, :, -q_block_num:, :] = torch.where(q_keep_mask, True, simple_masks[:, :, -q_block_num:, :])

        return attn_sums, simple_masks

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        try:
            llm = model
            if hasattr(llm, "model"):
                llm = llm.model
            if hasattr(llm, "language_model"):
                llm = llm.language_model

            attention_interface = self._get_custom_impl()
            attn_forward = get_custom_attn_forward(llm)
            self._original_attn_implementation = llm.config._attn_implementation
            llm.config._attn_implementation = "eager"
            for layer_idx, layer in enumerate(llm.layers):
                # replace the self-attention module with our custom module
                assert hasattr(layer, "self_attn"), "The model does not have a self-attention module."
                self._original_attn_forwards.append(layer.self_attn.forward)
                layer.self_attn.attn_interface = attention_interface
                layer.self_attn.forward = partial(attn_forward, module=layer.self_attn)
            yield
        except Exception as e:
            raise e
        finally:
            for layer_idx, layer in enumerate(llm.layers):
                # restore the original self-attention forward
                layer.self_attn.forward = self._original_attn_forwards[layer_idx]
                del layer.self_attn.attn_interface
            self._original_attn_forwards = []
            llm.config._attn_implementation = self._original_attn_implementation


def qwen2_vl_forward(
    module,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = module.q_proj(hidden_states)
    key_states = module.k_proj(hidden_states)
    value_states = module.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, module.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, module.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, module.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, module.rope_scaling["mrope_section"]
    )

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_values.update(key_states, value_states, module.layer_idx, cache_kwargs)

    attn_output, attn_weights = module.attn_interface(
        module,
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        attention_mask=attention_mask,
        scaling=module.scaling,
        dropout=module.attention_dropout if module.training else 0.0,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = module.o_proj(attn_output)
    return attn_output, attn_weights


def llama_forward(
    module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)

    query_states = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, module.layer_idx, cache_kwargs)

    attn_output, attn_weights = module.attn_interface(
        module,
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        attention_mask=attention_mask,
        dropout=module.attention_dropout if module.training else 0.0,
        scaling=module.scaling,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = module.o_proj(attn_output)
    return attn_output, attn_weights


def qwen3_forward(
    module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)

    query_states = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, module.layer_idx, cache_kwargs)

    attn_output, attn_weights = module.attn_interface(
        module,
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        attention_mask=attention_mask,
        dropout=module.attention_dropout if module.training else 0.0,
        scaling=module.scaling,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = module.o_proj(attn_output)
    return attn_output, attn_weights


def phi_forward(
    module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)

    qkv = module.qkv_proj(hidden_states)
    query_pos = module.config.num_attention_heads * module.head_dim
    query_states = qkv[..., :query_pos]
    key_states = qkv[..., query_pos : query_pos + module.num_key_value_heads * module.head_dim]
    value_states = qkv[..., query_pos + module.num_key_value_heads * module.head_dim :]

    query_states = query_states.view(hidden_shape).transpose(1, 2)
    key_states = key_states.view(hidden_shape).transpose(1, 2)
    value_states = value_states.view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = phi3_apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, module.layer_idx, cache_kwargs)

    attn_output, attn_weights = module.attn_interface(
        module,
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        attention_mask=attention_mask,
        dropout=module.attention_dropout if module.training else 0.0,
        scaling=module.scaling,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = module.o_proj(attn_output)
    return attn_output, attn_weights


CUSTOM_ATTENTION_FORWARDS = {
    "Qwen2VLForConditionalGeneration": qwen2_vl_forward,
    "Qwen2_5_VLForConditionalGeneration": qwen2_vl_forward,
    "LlamaForCausalLM": llama_forward,
    "MistralForCausalLM": llama_forward,
    "Qwen2ForCausalLM": llama_forward,
    "Qwen3ForCausalLM": qwen3_forward,
    "Phi3ForCausalLM": phi_forward,
}

def get_custom_attn_forward(model: PreTrainedModel):
    """
    Get the custom attention forward function for the given model.
    """
    if model.config.architectures[0] in CUSTOM_ATTENTION_FORWARDS:
        return CUSTOM_ATTENTION_FORWARDS[model.config.architectures[0]]
    if hasattr(model.config, "text_config") and model.config.text_config.architectures[0] in CUSTOM_ATTENTION_FORWARDS:
        return CUSTOM_ATTENTION_FORWARDS[model.config.text_config.architectures[0]]

    error_msg = f"Unsupported model class for: {model.config.architectures[0]}"
    raise ValueError(error_msg)
