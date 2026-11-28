// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <cuda/runtime.hpp>
#include <kernels/paged_attention.hpp>

using namespace ov::nvidia_gpu;

namespace {

std::mt19937 rng(123);
float frand() { return std::uniform_real_distribution<float>(-1.f, 1.f)(rng); }

// CPU golden: same math as the kernels (scale Q, causal softmax over paged K/V).
void cpu_golden(const std::vector<float>& query, const std::vector<float>& key_cache,
                const std::vector<float>& value_cache, const std::vector<int>& block_indices,
                const std::vector<int>& block_indices_begins, const std::vector<int>& past_lens,
                const std::vector<int>& subseq_begins, const std::vector<int>& row_to_subseq,
                const std::vector<float>& alibi, std::vector<float>& out,
                const kernel::PagedAttentionParams& p, int num_tokens) {
    const int S = p.head_size, SV = p.v_head_size, BS = p.block_size;
    for (int row = 0; row < num_tokens; ++row) {
        const int i = row_to_subseq[row];
        const int cur_kv_len = past_lens[i] + (row - subseq_begins[i]) + 1;
        const int blk_begin = block_indices_begins[i];
        for (int h = 0; h < p.num_heads; ++h) {
            const int hk = h / (p.num_heads / p.num_kv_heads);
            std::vector<float> scores;
            std::vector<int> ppos;
            for (int pc = 0; pc < cur_kv_len; ++pc) {
                if (p.sliding_window > 0 && pc < cur_kv_len - p.sliding_window) continue;
                const int bn = block_indices[blk_begin + pc / BS];
                if (bn < 0) continue;
                const int off = pc % BS;
                double dot = 0;
                for (int d = 0; d < S; ++d)
                    dot += (double)(query[row * (p.num_heads * S) + h * S + d] * p.scale) *
                           key_cache[((size_t)bn * p.num_kv_heads + hk) * BS * S + (size_t)off * S + d];
                float s = (float)dot;
                if (p.has_alibi) s += alibi[h] * (float)(-(cur_kv_len - 1 - pc));
                scores.push_back(s);
                ppos.push_back(pc);
            }
            float m = -1e30f;
            for (float s : scores) m = std::max(m, s);
            double l = 0;
            for (float s : scores) l += std::exp((double)s - m);
            for (int d = 0; d < SV; ++d) {
                double a = 0;
                for (size_t j = 0; j < scores.size(); ++j) {
                    const int bn = block_indices[blk_begin + ppos[j] / BS], off = ppos[j] % BS;
                    a += std::exp((double)scores[j] - m) / l *
                         value_cache[((size_t)bn * p.num_kv_heads + hk) * BS * SV + (size_t)off * SV + d];
                }
                out[row * (p.num_heads * SV) + h * SV + d] = (float)a;
            }
        }
    }
}

struct Case {
    int num_heads, num_kv_heads, head_size, v_head_size, sliding_window;
    bool alibi;
    std::vector<int> q_lens, past;
};

double run_case(const Case& c) {
    kernel::PagedAttentionParams p{c.num_heads,     c.num_kv_heads, c.head_size,      c.v_head_size, 32,
                                   1.0f / std::sqrt((float)c.head_size), c.sliding_window, c.alibi};
    const int nseq = c.q_lens.size();
    std::vector<int> subseq_begins(nseq + 1, 0), past_lens(c.past), row_to_subseq;
    for (int i = 0; i < nseq; ++i) subseq_begins[i + 1] = subseq_begins[i] + c.q_lens[i];
    const int num_tokens = subseq_begins[nseq];
    for (int i = 0; i < nseq; ++i)
        for (int j = 0; j < c.q_lens[i]; ++j) row_to_subseq.push_back(i);
    std::vector<int> block_indices, block_indices_begins(nseq + 1, 0);
    int next_block = 0;
    for (int i = 0; i < nseq; ++i) {
        int kv_len = past_lens[i] + c.q_lens[i], nblk = (kv_len + p.block_size - 1) / p.block_size;
        for (int b = 0; b < nblk; ++b) block_indices.push_back(next_block++);
        block_indices_begins[i + 1] = block_indices.size();
    }
    const int num_blocks = next_block + 2;
    std::vector<float> key_cache((size_t)num_blocks * c.num_kv_heads * p.block_size * c.head_size, 0.f);
    std::vector<float> value_cache((size_t)num_blocks * c.num_kv_heads * p.block_size * c.v_head_size, 0.f);
    for (int i = 0; i < nseq; ++i)
        for (int pos = 0; pos < past_lens[i]; ++pos) {
            int bn = block_indices[block_indices_begins[i] + pos / p.block_size], off = pos % p.block_size;
            for (int hk = 0; hk < c.num_kv_heads; ++hk) {
                for (int d = 0; d < c.head_size; ++d)
                    key_cache[((size_t)bn * c.num_kv_heads + hk) * p.block_size * c.head_size +
                              (size_t)off * c.head_size + d] = frand();
                for (int d = 0; d < c.v_head_size; ++d)
                    value_cache[((size_t)bn * c.num_kv_heads + hk) * p.block_size * c.v_head_size +
                                (size_t)off * c.v_head_size + d] = frand();
            }
        }
    std::vector<float> query((size_t)num_tokens * c.num_heads * c.head_size);
    std::vector<float> key((size_t)num_tokens * c.num_kv_heads * c.head_size);
    std::vector<float> value((size_t)num_tokens * c.num_kv_heads * c.v_head_size);
    for (auto& x : query) x = frand();
    for (auto& x : key) x = frand();
    for (auto& x : value) x = frand();
    std::vector<float> alibi(c.num_heads);
    for (int h = 0; h < c.num_heads; ++h) alibi[h] = c.alibi ? -std::pow(2.f, -(h + 1)) : 0.f;
    std::vector<int> slot_mapping(num_tokens);
    for (int i = 0; i < nseq; ++i)
        for (int j = 0; j < c.q_lens[i]; ++j) {
            int pos = past_lens[i] + j;
            int bn = block_indices[block_indices_begins[i] + pos / p.block_size], off = pos % p.block_size;
            slot_mapping[subseq_begins[i] + j] = bn * p.block_size + off;
        }

    const auto& ds = CUDA::DefaultStream::stream();
    CUDA::Stream cuda_stream{};
    auto stream = cuda_stream.get();
    auto up = [&](const void* h, size_t bytes) {
        auto a = ds.malloc(std::max(bytes, size_t{1}));
        if (bytes) ds.upload(CUDA::DevicePointer<void*>{a.get()}, h, bytes);
        return a;
    };
    auto d_key = up(key.data(), key.size() * 4), d_val = up(value.data(), value.size() * 4);
    auto d_kc = up(key_cache.data(), key_cache.size() * 4), d_vc = up(value_cache.data(), value_cache.size() * 4);
    auto d_q = up(query.data(), query.size() * 4);
    auto d_slot = up(slot_mapping.data(), slot_mapping.size() * 4);
    auto d_bi = up(block_indices.data(), block_indices.size() * 4);
    auto d_bib = up(block_indices_begins.data(), block_indices_begins.size() * 4);
    auto d_past = up(past_lens.data(), past_lens.size() * 4);
    auto d_sb = up(subseq_begins.data(), subseq_begins.size() * 4);
    auto d_r2s = up(row_to_subseq.data(), row_to_subseq.size() * 4);
    auto d_alibi = up(alibi.data(), alibi.size() * 4);
    auto d_out = ds.malloc((size_t)num_tokens * c.num_heads * c.v_head_size * 4);

    kernel::reshape_and_cache(stream, kernel::Type_t::f32, d_key.get(), d_val.get(), d_kc.get(), d_vc.get(),
                              static_cast<const int32_t*>(d_slot.get()), num_tokens, c.num_kv_heads, c.head_size,
                              c.v_head_size, p.block_size);
    kernel::paged_attention(stream, kernel::Type_t::f32, d_q.get(), d_kc.get(), d_vc.get(),
                            static_cast<const int32_t*>(d_bi.get()), static_cast<const int32_t*>(d_bib.get()),
                            static_cast<const int32_t*>(d_past.get()), static_cast<const int32_t*>(d_sb.get()),
                            static_cast<const int32_t*>(d_r2s.get()),
                            c.alibi ? static_cast<const float*>(d_alibi.get()) : nullptr, d_out.get(), p, num_tokens);

    std::vector<float> out_gpu((size_t)num_tokens * c.num_heads * c.v_head_size);
    ds.download(out_gpu.data(), CUDA::DevicePointer<const void*>{d_out.get()}, out_gpu.size() * 4);

    // golden: apply the same K/V writes to a cache copy, then attention.
    auto kc2 = key_cache, vc2 = value_cache;
    for (int token = 0; token < num_tokens; ++token) {
        int slot = slot_mapping[token], bn = slot / p.block_size, off = slot % p.block_size;
        for (int hk = 0; hk < c.num_kv_heads; ++hk) {
            for (int d = 0; d < c.head_size; ++d)
                kc2[((size_t)bn * c.num_kv_heads + hk) * p.block_size * c.head_size + (size_t)off * c.head_size + d] =
                    key[token * (c.num_kv_heads * c.head_size) + hk * c.head_size + d];
            for (int d = 0; d < c.v_head_size; ++d)
                vc2[((size_t)bn * c.num_kv_heads + hk) * p.block_size * c.v_head_size + (size_t)off * c.v_head_size +
                    d] = value[token * (c.num_kv_heads * c.v_head_size) + hk * c.v_head_size + d];
        }
    }
    std::vector<float> out_cpu((size_t)num_tokens * c.num_heads * c.v_head_size);
    cpu_golden(query, kc2, vc2, block_indices, block_indices_begins, past_lens, subseq_begins, row_to_subseq, alibi,
               out_cpu, p, num_tokens);
    double maxdiff = 0;
    for (size_t j = 0; j < out_cpu.size(); ++j) maxdiff = std::max(maxdiff, (double)std::abs(out_cpu[j] - out_gpu[j]));
    return maxdiff;
}

}  // namespace

class PagedAttentionKernelTest : public testing::TestWithParam<Case> {};

TEST_P(PagedAttentionKernelTest, MatchesCpuGolden) { ASSERT_LT(run_case(GetParam()), 1e-3); }

INSTANTIATE_TEST_SUITE_P(
    PagedAttention, PagedAttentionKernelTest,
    testing::Values(Case{12, 12, 64, 64, 0, false, {1}, {5}},        // decode
                    Case{12, 12, 64, 64, 0, false, {7}, {0}},        // prefill
                    Case{8, 8, 64, 64, 0, false, {5, 1}, {0, 10}},   // mixed prefill+decode
                    Case{4, 4, 64, 64, 0, false, {1}, {40}},         // cross-block kv>32
                    Case{12, 4, 64, 64, 0, false, {3}, {8}},         // GQA
                    Case{8, 8, 64, 32, 0, false, {1}, {6}},          // S != SV
                    Case{8, 8, 64, 64, 4, false, {1}, {20}},         // sliding window
                    Case{8, 8, 64, 64, 0, true, {1}, {12}},          // alibi
                    Case{8, 2, 64, 64, 0, false, {1, 4, 1}, {33, 7, 0}}));  // batch
