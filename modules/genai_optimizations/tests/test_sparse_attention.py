# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace

import pytest
import torch

transformers = pytest.importorskip("transformers")

from genai_opt import SparseAttention


def test_dense_patch_matches_model_and_restores_forwards():
    config = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = transformers.LlamaForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    with torch.no_grad():
        baseline = model(input_ids).logits
        original_forwards = [layer.self_attn.forward for layer in model.model.layers]
        with SparseAttention("dense")(model):
            patched = model(input_ids).logits
            assert all(
                layer.self_attn.forward != original
                for layer, original in zip(
                    model.model.layers, original_forwards, strict=True
                )
            )
        assert all(
            layer.self_attn.forward == original
            for layer, original in zip(
                model.model.layers, original_forwards, strict=True
            )
        )

    assert torch.allclose(baseline, patched, atol=1e-5, rtol=1e-5)


def test_tri_shape_builds_and_dispatches_block_mask(monkeypatch):
    captured = {}

    def fake_block_sparse_attention(
        query,
        key,
        value,
        query_offsets,
        key_offsets,
        head_mask_type,
        _unused,
        block_mask,
        max_query_length,
        max_key_length,
        **kwargs,
    ):
        captured["mask"] = block_mask
        assert query.shape == key.shape == value.shape == (256, 2, 4)
        assert query_offsets.tolist() == key_offsets.tolist() == [0, 256]
        assert head_mask_type.tolist() == [1, 1]
        assert max_query_length == max_key_length == 256
        assert kwargs["is_causal"] is True
        return query

    monkeypatch.setitem(
        sys.modules,
        "block_sparse_attn",
        SimpleNamespace(block_sparse_attn_func=fake_block_sparse_attention),
    )
    generator = torch.Generator().manual_seed(23)
    query = torch.randn(1, 2, 256, 4, generator=generator)
    key = torch.randn(1, 2, 256, 4, generator=generator)
    value = torch.randn(1, 2, 256, 4, generator=generator)
    attention_mask = torch.full((1, 1, 256, 256), torch.finfo(query.dtype).min)
    attention_mask.masked_fill_(
        torch.ones(256, 256, dtype=torch.bool).tril().view(1, 1, 256, 256),
        0,
    )
    module = SimpleNamespace(num_key_value_groups=1, head_dim=4)
    attention = SparseAttention(
        "tri-shape",
        last_query_size=0,
        recent_size=128,
        block_size=128,
    )

    output, weights = attention._tri_shape_attention(
        module,
        query,
        key,
        value,
        attention_mask=attention_mask,
    )

    assert output.shape == (1, 256, 2, 4)
    assert weights is None
    assert captured["mask"].shape == (1, 2, 2, 2)
    assert not torch.triu(captured["mask"], diagonal=1).any()


def test_tri_shape_keeps_every_block_intersecting_dense_tail(monkeypatch):
    captured = {}

    def fake_block_sparse_attention(*args, **_kwargs):
        captured["mask"] = args[7]
        return args[0]

    monkeypatch.setitem(
        sys.modules,
        "block_sparse_attn",
        SimpleNamespace(block_sparse_attn_func=fake_block_sparse_attention),
    )
    query = torch.zeros(1, 1, 385, 2)
    module = SimpleNamespace(num_key_value_groups=1, head_dim=2)
    attention = SparseAttention(
        "tri-shape",
        last_query_size=64,
        recent_size=128,
        block_size=128,
    )

    attention._tri_shape_attention(module, query, query, query, attention_mask=None)

    assert captured["mask"][0, 0].tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ]


def test_tri_shape_falls_back_to_dense_for_padding_mask(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        pytest.fail("block-sparse kernel must not receive a token-level padding mask")

    monkeypatch.setitem(
        sys.modules,
        "block_sparse_attn",
        SimpleNamespace(block_sparse_attn_func=fail_if_called),
    )
    query = torch.zeros(1, 1, 256, 2)
    module = SimpleNamespace(num_key_value_groups=1, head_dim=2)
    attention = SparseAttention(
        "tri-shape",
        last_query_size=0,
        recent_size=128,
        block_size=128,
    )
    attention_mask = torch.full((1, 1, 256, 256), torch.finfo(query.dtype).min)
    attention_mask.masked_fill_(
        torch.ones(256, 256, dtype=torch.bool).tril().view(1, 1, 256, 256),
        0,
    )
    attention_mask[..., 0] = torch.finfo(query.dtype).min

    output, weights = attention._tri_shape_attention(
        module,
        query,
        query,
        query,
        attention_mask=attention_mask,
    )

    assert output.shape == (1, 256, 1, 2)
    assert weights is None


def test_sparse_attention_returns_weights_when_dense_tail_is_zero(monkeypatch):
    def fake_block_sparse_attention(*args, **_kwargs):
        return args[0]

    monkeypatch.setitem(
        sys.modules,
        "block_sparse_attn",
        SimpleNamespace(block_sparse_attn_func=fake_block_sparse_attention),
    )
    query = torch.zeros(1, 1, 256, 2)
    module = SimpleNamespace(num_key_value_groups=1, head_dim=2)
    attention = SparseAttention(
        "tri-shape",
        output_attentions=True,
        last_query_size=0,
        recent_size=128,
        block_size=128,
    )

    _, weights = attention._tri_shape_attention(
        module, query, query, query, attention_mask=None
    )

    assert weights.shape == (1, 1, 256, 256)


def test_unsupported_attention_mode_is_rejected():
    with pytest.raises(ValueError, match="not a valid AttentionMode"):
        SparseAttention("unsupported")


def test_tri_shape_rejects_block_size_unsupported_by_kernel():
    with pytest.raises(ValueError, match="requires block_size=128"):
        SparseAttention("tri-shape", recent_size=64, block_size=64)
