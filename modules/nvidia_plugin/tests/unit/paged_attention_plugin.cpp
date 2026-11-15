// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Plugin-level tests of PagedAttentionExtension: run a single-PagedAttention model through the
// plugin's infer request for several steps, modifying the KV cache host tensors between steps
// the way openvino.genai's ContinuousBatchingPipeline does (copy-on-write `copy_blocks`, cache
// growth with `set_tensor`), and check the outputs against a CPU reference and the host caches.

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "cuda_plugin.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/make_tensor.hpp"

using namespace ov::nvidia_gpu;

namespace {

constexpr size_t kHeads = 2;
constexpr size_t kHeadSize = 8;
constexpr size_t kBlock = 32;
constexpr size_t kHidden = kHeads * kHeadSize;

using Vec = std::vector<float>;  // one token: [kHeads * kHeadSize]

std::shared_ptr<ov::op::v0::Parameter> param(ov::element::Type type, const ov::PartialShape& shape, const char* name) {
    auto p = std::make_shared<ov::op::v0::Parameter>(type, shape);
    p->set_friendly_name(name);
    p->get_output_tensor(0).set_names({name});
    return p;
}

template <typename T>
std::shared_ptr<ov::op::v0::Constant> constant(ov::element::Type type, const ov::Shape& shape, std::vector<T> v) {
    return std::make_shared<ov::op::v0::Constant>(type, shape, v);
}

/// The model SDPAToPagedAttention would produce for one attention layer.
std::shared_ptr<ov::Model> make_model() {
    const auto dyn = ov::Dimension::dynamic();
    auto q = param(ov::element::f32, {dyn, kHidden}, "q");
    auto k = param(ov::element::f32, {dyn, kHidden}, "k");
    auto v = param(ov::element::f32, {dyn, kHidden}, "v");
    auto key_cache = param(ov::element::dynamic, ov::PartialShape::dynamic(4), "key_cache.0");
    auto value_cache = param(ov::element::dynamic, ov::PartialShape::dynamic(4), "value_cache.0");
    auto past_lens = param(ov::element::i32, {dyn}, "past_lens");
    auto subsequence_begins = param(ov::element::i32, {dyn}, "subsequence_begins");
    auto block_indices = param(ov::element::i32, {dyn}, "block_indices");
    auto block_indices_begins = param(ov::element::i32, {dyn}, "block_indices_begins");
    const auto i32 = ov::element::i32;
    const auto f32 = ov::element::f32;
    const ov::OutputVector inputs{
        q,
        k,
        v,
        key_cache,
        value_cache,
        past_lens,
        subsequence_begins,
        block_indices,
        block_indices_begins,
        constant<float>(f32, {}, {1.0f / std::sqrt(static_cast<float>(kHeadSize))}),  // scale
        constant<int32_t>(i32, {}, {0}),                                               // sliding_window
        constant<float>(f32, {0}, {}),                                                 // alibi_slopes
        constant<int32_t>(i32, {}, {1024}),                                            // max_context_len
        constant<int32_t>(i32, {}, {0}),     // score_aggregation_window
        constant<int32_t>(i32, {0}, {}),     // rotated_block_indices
        constant<int32_t>(i32, {0}, {}),     // rotation_deltas
        constant<float>(f32, {0}, {}),       // rotation_trig_lut
        constant<float>(f32, {0}, {}),       // xattention_threshold
        constant<int32_t>(i32, {}, {64}),    // xattention_block_size
        constant<int32_t>(i32, {}, {8}),     // xattention_stride
        constant<float>(f32, {0}, {}),       // sinks
        constant<int32_t>(i32, {}, {0}),     // adaptive_rkv_start_size
        constant<int32_t>(i32, {0}, {}),     // adaptive_rkv_evictable_sizes
        constant<int32_t>(i32, {0}, {}),     // adaptive_rkv_diversity_block_set_indices
        constant<int32_t>(i32, {0}, {}),     // adaptive_rkv_diversity_block_set_indices_begins
    };
    auto pa = std::make_shared<ov::op::PagedAttentionExtension>(inputs);
    pa->get_rt_info()["num_k_heads"] = kHeads;
    pa->get_rt_info()["k_head_size"] = kHeadSize;
    pa->get_rt_info()["num_v_heads"] = kHeads;
    pa->get_rt_info()["v_head_size"] = kHeadSize;
    return std::make_shared<ov::Model>(
        ov::OutputVector{pa->output(0)},
        ov::ParameterVector{q, k, v, key_cache, value_cache, past_lens, subsequence_begins, block_indices,
                            block_indices_begins},
        "PagedAttention");
}

Vec random_token(std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Vec v(kHidden);
    for (auto& x : v) x = dist(rng);
    return v;
}

/// Attention of one query token over the given key/value tokens (all visible).
Vec reference_attention(const Vec& q, const std::vector<Vec>& keys, const std::vector<Vec>& values) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadSize));
    Vec out(kHidden, 0.0f);
    for (size_t h = 0; h < kHeads; ++h) {
        std::vector<float> s(keys.size());
        float max_s = -INFINITY;
        for (size_t j = 0; j < keys.size(); ++j) {
            float dot = 0;
            for (size_t d = 0; d < kHeadSize; ++d) dot += q[h * kHeadSize + d] * keys[j][h * kHeadSize + d];
            s[j] = dot * scale;
            max_s = std::max(max_s, s[j]);
        }
        float sum = 0;
        for (auto& x : s) sum += (x = std::exp(x - max_s));
        for (size_t j = 0; j < keys.size(); ++j)
            for (size_t d = 0; d < kHeadSize; ++d) out[h * kHeadSize + d] += s[j] / sum * values[j][h * kHeadSize + d];
    }
    return out;
}

/// One cache slot ([kHeads, kHeadSize]) of a host cache tensor [blocks, kHeads, kBlock, kHeadSize].
Vec cache_slot(const ov::Tensor& cache, size_t block, size_t offset) {
    Vec v(kHidden);
    const float* p = cache.data<const float>();
    for (size_t h = 0; h < kHeads; ++h)
        for (size_t d = 0; d < kHeadSize; ++d)
            v[h * kHeadSize + d] = p[((block * kHeads + h) * kBlock + offset) * kHeadSize + d];
    return v;
}

void copy_block(ov::Tensor& cache, size_t src, size_t dst) {
    const size_t block_elems = kHeads * kBlock * kHeadSize;
    std::memcpy(cache.data<float>() + dst * block_elems,
                cache.data<float>() + src * block_elems,
                block_elems * sizeof(float));
}

ov::Tensor make_cache(size_t num_blocks, float fill) {
    ov::Tensor t(ov::element::f32, ov::Shape{num_blocks, kHeads, kBlock, kHeadSize});
    std::fill_n(t.data<float>(), t.get_size(), fill);
    return t;
}

ov::Tensor tokens_tensor(const std::vector<Vec>& tokens) {
    ov::Tensor t(ov::element::f32, ov::Shape{tokens.size(), kHidden});
    for (size_t i = 0; i < tokens.size(); ++i) std::copy(tokens[i].begin(), tokens[i].end(), t.data<float>() + i * kHidden);
    return t;
}

ov::Tensor i32_tensor(std::vector<int32_t> v) {
    ov::Tensor t(ov::element::i32, ov::Shape{v.size()});
    std::copy(v.begin(), v.end(), t.data<int32_t>());
    return t;
}

class PagedAttentionPluginTest : public testing::Test {
protected:
    void SetUp() override {
        model_ = make_model();
        auto plugin = std::make_shared<Plugin>();
        compiled_ = plugin->compile_model(model_, {{ov::device::id.name(), "0"}, {ov::hint::inference_precision.name(), "f32"}});
        request_ = compiled_->create_infer_request();
    }

    void set(size_t input, const ov::Tensor& t) {
        request_->set_tensor(compiled_->inputs()[input], ov::get_tensor_impl(t));
    }

    /// Runs one step for a single sequence and returns the output rows.
    std::vector<Vec> step(const std::vector<Vec>& q,
                          const std::vector<Vec>& k,
                          const std::vector<Vec>& v,
                          int32_t past_len,
                          std::vector<int32_t> blocks) {
        set(0, tokens_tensor(q));
        set(1, tokens_tensor(k));
        set(2, tokens_tensor(v));
        set(3, key_cache_);
        set(4, value_cache_);
        set(5, i32_tensor({past_len}));
        set(6, i32_tensor({0, static_cast<int32_t>(q.size())}));
        set(7, i32_tensor(blocks));
        set(8, i32_tensor({0, static_cast<int32_t>(blocks.size())}));
        request_->infer();
        auto out = request_->get_tensor(compiled_->outputs()[0]);
        EXPECT_EQ(out->get_shape(), (ov::Shape{q.size(), kHidden}));
        std::vector<Vec> rows(q.size());
        const float* p = static_cast<const float*>(out->data());
        for (size_t i = 0; i < q.size(); ++i) rows[i].assign(p + i * kHidden, p + (i + 1) * kHidden);
        return rows;
    }

    static void expect_near(const Vec& actual, const Vec& expected, const char* what) {
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], expected[i], 1e-4f) << what << " at element " << i;
        }
    }

    std::shared_ptr<ov::Model> model_;
    std::shared_ptr<ov::ICompiledModel> compiled_;
    std::shared_ptr<ov::IAsyncInferRequest> request_;
    ov::Tensor key_cache_ = make_cache(2, 0.0f);
    ov::Tensor value_cache_ = make_cache(2, 0.0f);
};

}  // namespace

TEST_F(PagedAttentionPluginTest, CopyOnWriteAndGrowthStayCoherentWithHostCaches) {
    std::mt19937 rng(7);
    const auto tokens = [&](size_t n) {
        std::vector<Vec> t(n);
        for (auto& x : t) x = random_token(rng);
        return t;
    };

    // Step 1: prefill of sequence A (5 tokens) into block 0.
    const auto q1 = tokens(5), k1 = tokens(5), v1 = tokens(5);
    const auto out1 = step(q1, k1, v1, 0, {0});
    for (size_t r = 0; r < q1.size(); ++r) {
        const std::vector<Vec> keys(k1.begin(), k1.begin() + r + 1), values(v1.begin(), v1.begin() + r + 1);
        expect_near(out1[r], reference_attention(q1[r], keys, values), "prefill output");
        // write-through: the new K/V are in the host caches when infer() returns
        expect_near(cache_slot(key_cache_, 0, r), k1[r], "host key cache after prefill");
        expect_near(cache_slot(value_cache_, 0, r), v1[r], "host value cache after prefill");
    }

    // Step 2: GenAI forks A into B (beam search / parallel sampling). B's partially filled last
    // block is copied on the host only (CacheManager::copy_blocks: block 0 -> block 1), then B
    // decodes one token into block 1. The device must pick up the host-side copy.
    copy_block(key_cache_, 0, 1);
    copy_block(value_cache_, 0, 1);
    const auto q2 = tokens(1), k2 = tokens(1), v2 = tokens(1);
    const auto out2 = step(q2, k2, v2, 5, {1});
    auto keys_b = k1, values_b = v1;
    keys_b.push_back(k2[0]);
    values_b.push_back(v2[0]);
    expect_near(out2[0], reference_attention(q2[0], keys_b, values_b), "decode output after copy-on-write");
    expect_near(cache_slot(key_cache_, 1, 5), k2[0], "host key cache after decode into the copied block");

    // Step 3: GenAI grows the cache: bigger host tensors with the old contents in front, bound
    // with set_tensor. Sequence A decodes into block 0; the new K/V must land in the grown tensor.
    const auto old_key_cache = key_cache_;
    auto grown_k = make_cache(4, 123.0f), grown_v = make_cache(4, 123.0f);
    std::memcpy(grown_k.data(), key_cache_.data(), key_cache_.get_byte_size());
    std::memcpy(grown_v.data(), value_cache_.data(), value_cache_.get_byte_size());
    key_cache_ = grown_k;
    value_cache_ = grown_v;
    const auto q3 = tokens(1), k3 = tokens(1), v3 = tokens(1);
    const auto out3 = step(q3, k3, v3, 5, {0});
    auto keys_a = k1, values_a = v1;
    keys_a.push_back(k3[0]);
    values_a.push_back(v3[0]);
    expect_near(out3[0], reference_attention(q3[0], keys_a, values_a), "decode output after cache growth");
    expect_near(cache_slot(key_cache_, 0, 5), k3[0], "grown host key cache after decode");
    // the replaced host tensor is not written any more
    expect_near(cache_slot(old_key_cache, 0, 5), Vec(kHidden, 0.0f), "replaced host key cache");
}

TEST_F(PagedAttentionPluginTest, GrowthKeepsFullBlocksOnTheDevice) {
    // A full block is never refreshed from the host (nothing is appended to it), so after the
    // cache grows its contents can only come from the device data kept across the reallocation.
    std::mt19937 rng(5);
    const auto tokens = [&](size_t n) {
        std::vector<Vec> t(n);
        for (auto& x : t) x = random_token(rng);
        return t;
    };
    const size_t prompt = kBlock + 2;  // block 0 full, block 1 partially filled
    const auto q1 = tokens(prompt), k1 = tokens(prompt), v1 = tokens(prompt);
    step(q1, k1, v1, 0, {0, 1});

    auto grown_k = make_cache(5, 0.0f), grown_v = make_cache(5, 0.0f);
    std::memcpy(grown_k.data(), key_cache_.data(), key_cache_.get_byte_size());
    std::memcpy(grown_v.data(), value_cache_.data(), value_cache_.get_byte_size());
    key_cache_ = grown_k;
    value_cache_ = grown_v;

    const auto q2 = tokens(1), k2 = tokens(1), v2 = tokens(1);
    const auto out = step(q2, k2, v2, static_cast<int32_t>(prompt), {0, 1});
    auto keys = k1, values = v1;
    keys.push_back(k2[0]);
    values.push_back(v2[0]);
    expect_near(out[0], reference_attention(q2[0], keys, values), "decode output after cache growth");
}

TEST_F(PagedAttentionPluginTest, InvalidBlockTableThrows) {
    std::mt19937 rng(3);
    const auto t = random_token(rng);
    // block 5 is outside the 2-block cache
    EXPECT_ANY_THROW(step({t}, {t}, {t}, 0, {5}));
}
