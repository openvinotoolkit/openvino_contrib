// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gfx_runtime_model_runner.hpp"
#include "gfx_runtime_scenario.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"

namespace {

using RuntimeScenarioPtr = std::shared_ptr<const ov::test::gfx::RuntimeScenario>;
using RuntimeModelBuilder = ov::test::gfx::RuntimeModelBuilder;
using RuntimeInputBuilder = ov::test::gfx::RuntimeInputBuilder;

std::shared_ptr<ov::op::v0::Constant> f32_constant(const ov::Shape& shape, std::vector<float> values) {
    return std::make_shared<ov::op::v0::Constant>(ov::element::f32, shape, values);
}

std::shared_ptr<ov::op::v0::Constant> i64_constant(const ov::Shape& shape, std::vector<int64_t> values) {
    return std::make_shared<ov::op::v0::Constant>(ov::element::i64, shape, values);
}

ov::Tensor filled_f32(const ov::Shape& shape, int modulus, int shift, float scale, int offset = 0) {
    ov::Tensor tensor(ov::element::f32, shape);
    auto* data = tensor.data<float>();
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        data[i] = static_cast<float>((static_cast<int>((i + offset) % modulus) - shift)) * scale;
    }
    return tensor;
}

ov::Tensor zero_f32(const ov::Shape& shape) {
    ov::Tensor tensor(ov::element::f32, shape);
    std::fill_n(tensor.data<float>(), tensor.get_size(), 0.0f);
    return tensor;
}

std::vector<ov::Tensor> small_pair_inputs() {
    return {filled_f32({1, 4, 2}, 17, 8, 0.125f),
            filled_f32({1, 4, 2}, 19, 9, 0.125f, 5)};
}

std::vector<ov::Tensor> small_attention_inputs() {
    return {filled_f32({1, 4, 2}, 17, 8, 0.125f),
            filled_f32({1, 4, 2}, 19, 9, 0.125f, 5),
            filled_f32({1, 4, 2}, 23, 11, 0.125f, 11)};
}

std::vector<ov::Tensor> large_pair_inputs() {
    return {filled_f32({1, 4, 32, 400}, 251, 125, 0.03125f),
            filled_f32({1, 4, 32, 400}, 257, 128, 0.03125f, 17)};
}

std::vector<ov::Tensor> large_pair_with_zero_inputs() {
    auto inputs = large_pair_inputs();
    inputs.push_back(zero_f32({1, 4, 32, 400}));
    return inputs;
}

std::vector<ov::Tensor> variadic_split_input() {
    return {filled_f32({1, 4, 96, 400}, 251, 125, 0.03125f)};
}

RuntimeScenarioPtr scenario(std::string name,
                            RuntimeModelBuilder model_builder,
                            RuntimeInputBuilder input_builder,
                            int timeout = 20,
                            float atol = 1e-4f,
                            size_t infer_count = 1) {
    return ov::test::gfx::runtime_scenario(std::move(name),
                                           std::move(model_builder),
                                           std::move(input_builder),
                                           timeout,
                                           atol,
                                           0.f,
                                           infer_count);
}

std::vector<RuntimeScenarioPtr> matmul_runtime_scenarios() {
    return {
        scenario("DirectBatched",
                 [] {
                     auto lhs =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
                     auto rhs =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
                     auto matmul = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, true);
                     auto res = std::make_shared<ov::op::v0::Result>(matmul);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{lhs, rhs},
                                                        "matmul_runtime");
                 },
                 small_pair_inputs,
                 15,
                 1e-5f),
        scenario("LargeTransposed",
                 [] {
                     auto lhs =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
                     auto rhs =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
                     auto matmul = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, true, false);
                     auto res = std::make_shared<ov::op::v0::Result>(matmul);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{lhs, rhs},
                                                        "large_matmul_runtime");
                 },
                 large_pair_inputs),
        scenario("MultiplyMatMul",
                 [] {
                     auto lhs =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
                     auto rhs =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
                     auto scaled = std::make_shared<ov::op::v1::Multiply>(
                         lhs,
                         f32_constant({1}, std::vector<float>{0.5f}));
                     auto matmul = std::make_shared<ov::op::v0::MatMul>(scaled, rhs, false, true);
                     auto res = std::make_shared<ov::op::v0::Result>(matmul);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{lhs, rhs},
                                                        "multiply_matmul_runtime");
                 },
                 small_pair_inputs,
                 15,
                 1e-5f),
        scenario("SoftmaxMatMul",
                 [] {
                     auto probs =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 4});
                     auto values =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
                     auto softmax = std::make_shared<ov::op::v1::Softmax>(probs, 2);
                     auto matmul = std::make_shared<ov::op::v0::MatMul>(softmax, values, false, false);
                     auto res = std::make_shared<ov::op::v0::Result>(matmul);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{probs, values},
                                                        "softmax_matmul_runtime");
                 },
                 [] {
                     return std::vector<ov::Tensor>{filled_f32({1, 4, 4}, 19, 9, 0.125f),
                                                    filled_f32({1, 4, 2}, 23, 11, 0.125f, 11)};
                 },
                 15,
                 1e-5f),
        scenario("AttentionCoreNoSplit",
                 [] {
                     auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
                     auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
                     auto v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
                     auto q_scaled = std::make_shared<ov::op::v1::Multiply>(
                         q,
                         f32_constant({1}, std::vector<float>{0.5f}));
                     auto scores = std::make_shared<ov::op::v0::MatMul>(q_scaled, k, false, true);
                     auto probs = std::make_shared<ov::op::v1::Softmax>(scores, 2);
                     auto attn = std::make_shared<ov::op::v0::MatMul>(probs, v, false, false);
                     auto res = std::make_shared<ov::op::v0::Result>(attn);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{q, k, v},
                                                        "attn_core_no_split");
                 },
                 small_attention_inputs,
                 15,
                 1e-5f),
        scenario("AttentionPreScale",
                 [] {
                     auto param =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 6});
                     auto split = std::make_shared<ov::op::v1::Split>(
                         param,
                         i64_constant({}, std::vector<int64_t>{2}),
                         3);
                     auto q_scaled = std::make_shared<ov::op::v1::Multiply>(
                         split->output(0),
                         f32_constant({1}, std::vector<float>{0.5f}));
                     auto scores = std::make_shared<ov::op::v0::MatMul>(q_scaled, split->output(1), false, true);
                     auto probs = std::make_shared<ov::op::v1::Softmax>(scores, 2);
                     auto attn = std::make_shared<ov::op::v0::MatMul>(probs, split->output(2), false, false);
                     auto res = std::make_shared<ov::op::v0::Result>(attn);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{param},
                                                        "attn_prescale_runtime");
                 },
                 [] { return std::vector<ov::Tensor>{filled_f32({1, 4, 6}, 19, 9, 0.125f)}; },
                 20,
                 1e-5f),
        scenario("AttentionScoreScale",
                 [] {
                     auto q =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
                     auto k =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
                     auto scores = std::make_shared<ov::op::v0::MatMul>(q, k, true, false);
                     auto mul = std::make_shared<ov::op::v1::Multiply>(
                         scores,
                         f32_constant({1, 1, 1, 1}, std::vector<float>{0.176776695f}));
                     auto res = std::make_shared<ov::op::v0::Result>(mul);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{q, k},
                                                        "attn_score_scale_matmul");
                 },
                 large_pair_inputs),
        scenario("AttentionLayoutSplitScale",
                 [] {
                     auto param =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 400, 384});
                     auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                         param,
                         i64_constant({4}, std::vector<int64_t>{1, 400, 4, 96}),
                         true);
                     auto transposed = std::make_shared<ov::op::v1::Transpose>(
                         reshaped,
                         i64_constant({4}, std::vector<int64_t>{0, 2, 3, 1}));
                     auto split = std::make_shared<ov::op::v1::VariadicSplit>(
                         transposed,
                         i64_constant({}, std::vector<int64_t>{2}),
                         i64_constant({3}, std::vector<int64_t>{32, 32, 32}));
                     auto scores = std::make_shared<ov::op::v0::MatMul>(split->output(0),
                                                                         split->output(1),
                                                                         true,
                                                                         false);
                     auto mul = std::make_shared<ov::op::v1::Multiply>(
                         scores,
                         f32_constant({1, 1, 1, 1}, std::vector<float>{0.176776695f}));
                     auto res = std::make_shared<ov::op::v0::Result>(mul);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{param},
                                                        "attn_layout_split_scale");
                 },
                 [] { return std::vector<ov::Tensor>{filled_f32({1, 400, 384}, 251, 125, 0.03125f)}; }),
        scenario("VariadicSplitMatMul",
                 [] {
                     auto param =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 400});
                     auto split = std::make_shared<ov::op::v1::VariadicSplit>(
                         param,
                         i64_constant({}, std::vector<int64_t>{2}),
                         i64_constant({3}, std::vector<int64_t>{32, 32, 32}));
                     auto scores = std::make_shared<ov::op::v0::MatMul>(split->output(0),
                                                                         split->output(1),
                                                                         true,
                                                                         false);
                     auto res = std::make_shared<ov::op::v0::Result>(scores);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{param},
                                                        "variadic_split_matmul");
                 },
                 variadic_split_input),
        scenario("VariadicSplitLhsMatMul",
                 [] {
                     auto split_in =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 400});
                     auto rhs_in =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
                     auto split = std::make_shared<ov::op::v1::VariadicSplit>(
                         split_in,
                         i64_constant({}, std::vector<int64_t>{2}),
                         i64_constant({3}, std::vector<int64_t>{32, 32, 32}));
                     auto scores = std::make_shared<ov::op::v0::MatMul>(split->output(0),
                                                                         rhs_in,
                                                                         true,
                                                                         false);
                     auto res = std::make_shared<ov::op::v0::Result>(scores);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{split_in, rhs_in},
                                                        "variadic_split_lhs_matmul");
                 },
                 [] {
                     return std::vector<ov::Tensor>{filled_f32({1, 4, 96, 400}, 251, 125, 0.03125f),
                                                    filled_f32({1, 4, 32, 400}, 257, 128, 0.03125f, 17)};
                 }),
        scenario("VariadicSplitRhsMatMul",
                 [] {
                     auto lhs_in =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
                     auto split_in =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 400});
                     auto split = std::make_shared<ov::op::v1::VariadicSplit>(
                         split_in,
                         i64_constant({}, std::vector<int64_t>{2}),
                         i64_constant({3}, std::vector<int64_t>{32, 32, 32}));
                     auto scores = std::make_shared<ov::op::v0::MatMul>(lhs_in,
                                                                         split->output(1),
                                                                         true,
                                                                         false);
                     auto res = std::make_shared<ov::op::v0::Result>(scores);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{lhs_in, split_in},
                                                        "variadic_split_rhs_matmul");
                 },
                 [] {
                     return std::vector<ov::Tensor>{filled_f32({1, 4, 32, 400}, 251, 125, 0.03125f),
                                                    filled_f32({1, 4, 96, 400}, 257, 128, 0.03125f, 17)};
                 }),
        scenario("VariadicSplitScaledMatMul",
                 [] {
                     auto param =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 400});
                     auto split = std::make_shared<ov::op::v1::VariadicSplit>(
                         param,
                         i64_constant({}, std::vector<int64_t>{2}),
                         i64_constant({3}, std::vector<int64_t>{32, 32, 32}));
                     auto mul = std::make_shared<ov::op::v1::Multiply>(
                         split->output(1),
                         f32_constant({1, 1, 1, 1}, std::vector<float>{0.176776695f}));
                     auto scores = std::make_shared<ov::op::v0::MatMul>(split->output(0), mul, true, false);
                     auto res = std::make_shared<ov::op::v0::Result>(scores);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{param},
                                                        "variadic_split_scaled_matmul");
                 },
                 variadic_split_input),
        scenario("LargeBroadcastMultiplyMatMul",
                 [] {
                     auto lhs =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
                     auto rhs =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
                     auto mul = std::make_shared<ov::op::v1::Multiply>(
                         rhs,
                         f32_constant({1, 1, 1, 1}, std::vector<float>{0.176776695f}));
                     auto scores = std::make_shared<ov::op::v0::MatMul>(lhs, mul, true, false);
                     auto res = std::make_shared<ov::op::v0::Result>(scores);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{lhs, rhs},
                                                        "large_broadcast_multiply_matmul");
                 },
                 large_pair_inputs),
        scenario("DualProducedBroadcastMultiplyMatMul",
                 [] {
                     auto lhs_in =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
                     auto rhs_in =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
                     auto lhs = std::make_shared<ov::op::v1::Multiply>(
                         lhs_in,
                         f32_constant({1, 1, 1, 1}, std::vector<float>{1.0f}));
                     auto rhs = std::make_shared<ov::op::v1::Multiply>(
                         rhs_in,
                         f32_constant({1, 1, 1, 1}, std::vector<float>{0.176776695f}));
                     auto scores = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, true, false);
                     auto res = std::make_shared<ov::op::v0::Result>(scores);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{lhs_in, rhs_in},
                                                        "dual_produced_broadcast_multiply_matmul");
                 },
                 large_pair_inputs),
        scenario("ProducedLhsMatMul",
                 [] {
                     auto lhs_in =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
                     auto rhs_in =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
                     auto zero =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
                     auto lhs = std::make_shared<ov::op::v1::Add>(lhs_in, zero);
                     auto scores = std::make_shared<ov::op::v0::MatMul>(lhs, rhs_in, true, false);
                     auto res = std::make_shared<ov::op::v0::Result>(scores);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{lhs_in, rhs_in, zero},
                                                        "produced_lhs_matmul");
                 },
                 large_pair_with_zero_inputs),
        scenario("ProducedLhsAndMatMulOutputs",
                 [] {
                     auto lhs_in =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
                     auto rhs_in =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
                     auto zero =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
                     auto lhs = std::make_shared<ov::op::v1::Add>(lhs_in, zero);
                     auto scores = std::make_shared<ov::op::v0::MatMul>(lhs, rhs_in, true, false);
                     ov::ResultVector results{std::make_shared<ov::op::v0::Result>(lhs),
                                              std::make_shared<ov::op::v0::Result>(scores)};
                     return std::make_shared<ov::Model>(results,
                                                        ov::ParameterVector{lhs_in, rhs_in, zero},
                                                        "produced_lhs_and_matmul_outputs");
                 },
                 large_pair_with_zero_inputs),
        scenario("ProducedLhsAndMatMulOutputsRepeated",
                 [] {
                     auto lhs_in =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
                     auto rhs_in =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
                     auto zero =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
                     auto lhs = std::make_shared<ov::op::v1::Add>(lhs_in, zero);
                     auto scores = std::make_shared<ov::op::v0::MatMul>(lhs, rhs_in, true, false);
                     ov::ResultVector results{std::make_shared<ov::op::v0::Result>(lhs),
                                              std::make_shared<ov::op::v0::Result>(scores)};
                     return std::make_shared<ov::Model>(results,
                                                        ov::ParameterVector{lhs_in, rhs_in, zero},
                                                        "produced_lhs_and_matmul_outputs_repeat");
                 },
                 large_pair_with_zero_inputs,
                 30,
                 1e-4f,
                 4),
        scenario("VariadicSplitDualScaledMatMul",
                 [] {
                     auto param =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 400});
                     auto split = std::make_shared<ov::op::v1::VariadicSplit>(
                         param,
                         i64_constant({}, std::vector<int64_t>{2}),
                         i64_constant({3}, std::vector<int64_t>{32, 32, 32}));
                     auto lhs = std::make_shared<ov::op::v1::Multiply>(
                         split->output(0),
                         f32_constant({1, 1, 1, 1}, std::vector<float>{1.0f}));
                     auto rhs = std::make_shared<ov::op::v1::Multiply>(
                         split->output(1),
                         f32_constant({1, 1, 1, 1}, std::vector<float>{0.176776695f}));
                     auto scores = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, true, false);
                     auto res = std::make_shared<ov::op::v0::Result>(scores);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{param},
                                                        "variadic_split_dual_scaled_matmul");
                 },
                 variadic_split_input),
        scenario("VariadicSplitAddScaledMatMul",
                 [] {
                     auto param =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 400});
                     auto zero =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
                     auto split = std::make_shared<ov::op::v1::VariadicSplit>(
                         param,
                         i64_constant({}, std::vector<int64_t>{2}),
                         i64_constant({3}, std::vector<int64_t>{32, 32, 32}));
                     auto lhs = std::make_shared<ov::op::v1::Add>(split->output(0), zero);
                     auto rhs = std::make_shared<ov::op::v1::Multiply>(
                         split->output(1),
                         f32_constant({1, 1, 1, 1}, std::vector<float>{0.176776695f}));
                     auto scores = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, true, false);
                     auto res = std::make_shared<ov::op::v0::Result>(scores);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{param, zero},
                                                        "variadic_split_add_scaled_matmul");
                 },
                 [] {
                     return std::vector<ov::Tensor>{filled_f32({1, 4, 96, 400}, 251, 125, 0.03125f),
                                                    zero_f32({1, 4, 32, 400})};
                 }),
        scenario("VariadicSplitAddScaledMatMulRepeated",
                 [] {
                     auto param =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 400});
                     auto zero =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
                     auto split = std::make_shared<ov::op::v1::VariadicSplit>(
                         param,
                         i64_constant({}, std::vector<int64_t>{2}),
                         i64_constant({3}, std::vector<int64_t>{32, 32, 32}));
                     auto lhs = std::make_shared<ov::op::v1::Add>(split->output(0), zero);
                     auto rhs = std::make_shared<ov::op::v1::Multiply>(
                         split->output(1),
                         f32_constant({1, 1, 1, 1}, std::vector<float>{0.176776695f}));
                     auto scores = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, true, false);
                     auto res = std::make_shared<ov::op::v0::Result>(scores);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{param, zero},
                                                        "variadic_split_add_scaled_matmul_repeat");
                 },
                 [] {
                     return std::vector<ov::Tensor>{filled_f32({1, 4, 96, 400}, 251, 125, 0.03125f),
                                                    zero_f32({1, 4, 32, 400})};
                 },
                 30,
                 1e-4f,
                 4),
        scenario("VariadicSplitAddAndMatMulOutputs",
                 [] {
                     auto param =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 400});
                     auto zero =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
                     auto split = std::make_shared<ov::op::v1::VariadicSplit>(
                         param,
                         i64_constant({}, std::vector<int64_t>{2}),
                         i64_constant({3}, std::vector<int64_t>{32, 32, 32}));
                     auto lhs = std::make_shared<ov::op::v1::Add>(split->output(0), zero);
                     auto rhs = std::make_shared<ov::op::v1::Multiply>(
                         split->output(1),
                         f32_constant({1, 1, 1, 1}, std::vector<float>{0.176776695f}));
                     auto scores = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, true, false);
                     ov::ResultVector results{std::make_shared<ov::op::v0::Result>(lhs),
                                              std::make_shared<ov::op::v0::Result>(scores)};
                     return std::make_shared<ov::Model>(results,
                                                        ov::ParameterVector{param, zero},
                                                        "variadic_split_add_and_matmul_outputs");
                 },
                 [] {
                     return std::vector<ov::Tensor>{filled_f32({1, 4, 96, 400}, 251, 125, 0.03125f),
                                                    zero_f32({1, 4, 32, 400})};
                 }),
        scenario("VariadicSplitAttentionAndValueLayout",
                 [] {
                     auto param =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 400});
                     auto split = std::make_shared<ov::op::v1::VariadicSplit>(
                         param,
                         i64_constant({}, std::vector<int64_t>{2}),
                         i64_constant({3}, std::vector<int64_t>{32, 32, 32}));
                     auto scaled_k = std::make_shared<ov::op::v1::Multiply>(
                         split->output(1),
                         f32_constant({1, 1, 1, 1}, std::vector<float>{0.176776695f}));
                     auto scores = std::make_shared<ov::op::v0::MatMul>(split->output(0), scaled_k, true, false);
                     auto probs = std::make_shared<ov::op::v1::Softmax>(scores, 3);
                     auto attn = std::make_shared<ov::op::v0::MatMul>(probs, split->output(2), false, true);
                     auto value_transpose = std::make_shared<ov::op::v1::Transpose>(
                         split->output(2),
                         i64_constant({4}, std::vector<int64_t>{0, 3, 1, 2}));
                     auto value_reshape = std::make_shared<ov::op::v1::Reshape>(
                         value_transpose,
                         i64_constant({3}, std::vector<int64_t>{1, 400, 128}),
                         true);
                     ov::ResultVector results{std::make_shared<ov::op::v0::Result>(attn),
                                              std::make_shared<ov::op::v0::Result>(value_reshape)};
                     return std::make_shared<ov::Model>(results,
                                                        ov::ParameterVector{param},
                                                        "variadic_split_attention_and_value_layout");
                 },
                 variadic_split_input),
        scenario("VariadicSplitScaledMatMulAndValueLayout",
                 [] {
                     auto param =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 400});
                     auto split = std::make_shared<ov::op::v1::VariadicSplit>(
                         param,
                         i64_constant({}, std::vector<int64_t>{2}),
                         i64_constant({3}, std::vector<int64_t>{32, 32, 32}));
                     auto scaled_k = std::make_shared<ov::op::v1::Multiply>(
                         split->output(1),
                         f32_constant({1, 1, 1, 1}, std::vector<float>{0.176776695f}));
                     auto scores = std::make_shared<ov::op::v0::MatMul>(split->output(0),
                                                                         scaled_k,
                                                                         true,
                                                                         false);
                     auto value_transpose = std::make_shared<ov::op::v1::Transpose>(
                         split->output(2),
                         i64_constant({4}, std::vector<int64_t>{0, 3, 1, 2}));
                     auto value_reshape = std::make_shared<ov::op::v1::Reshape>(
                         value_transpose,
                         i64_constant({3}, std::vector<int64_t>{1, 400, 128}),
                         true);
                     ov::ResultVector results{std::make_shared<ov::op::v0::Result>(scores),
                                              std::make_shared<ov::op::v0::Result>(value_reshape)};
                     return std::make_shared<ov::Model>(results,
                                                        ov::ParameterVector{param},
                                                        "variadic_split_scaled_matmul_and_value_layout");
                 },
                 variadic_split_input),
        scenario("SplitMatMul",
                 [] {
                     auto param =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 400});
                     auto split = std::make_shared<ov::op::v1::Split>(
                         param,
                         i64_constant({}, std::vector<int64_t>{2}),
                         3);
                     auto scores = std::make_shared<ov::op::v0::MatMul>(split->output(0),
                                                                         split->output(1),
                                                                         true,
                                                                         false);
                     auto res = std::make_shared<ov::op::v0::Result>(scores);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{param},
                                                        "split_matmul");
                 },
                 variadic_split_input),
        scenario("AttentionLayoutSplitMatMul",
                 [] {
                     auto param =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 400, 384});
                     auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                         param,
                         i64_constant({4}, std::vector<int64_t>{1, 400, 4, 96}),
                         true);
                     auto transposed = std::make_shared<ov::op::v1::Transpose>(
                         reshaped,
                         i64_constant({4}, std::vector<int64_t>{0, 2, 3, 1}));
                     auto split = std::make_shared<ov::op::v1::VariadicSplit>(
                         transposed,
                         i64_constant({}, std::vector<int64_t>{2}),
                         i64_constant({3}, std::vector<int64_t>{32, 32, 32}));
                     auto mul = std::make_shared<ov::op::v1::Multiply>(
                         split->output(1),
                         f32_constant({1, 1, 1, 1}, std::vector<float>{0.176776695f}));
                     auto scores = std::make_shared<ov::op::v0::MatMul>(split->output(0), mul, true, false);
                     auto res = std::make_shared<ov::op::v0::Result>(scores);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{param},
                                                        "attn_layout_matmul");
                 },
                 [] { return std::vector<ov::Tensor>{filled_f32({1, 400, 384}, 251, 125, 0.03125f)}; }),
    };
}

class MatMulRuntimeTest : public ::testing::TestWithParam<RuntimeScenarioPtr> {};

TEST_P(MatMulRuntimeTest, MatchesTemplate) {
    const auto& scenario = *GetParam();
    ov::test::gfx::RuntimeModelRunner runner;
    if (scenario.infer_count() > 1) {
        runner.compare_model_repeated_infer(scenario.make_model(),
                                            scenario.make_inputs(),
                                            scenario.infer_count(),
                                            scenario.timeout_seconds(),
                                            scenario.atol(),
                                            scenario.rtol());
        return;
    }
    runner.compare_model(scenario.make_model(),
                         scenario.make_inputs(),
                         scenario.timeout_seconds(),
                         scenario.atol(),
                         scenario.rtol());
}

INSTANTIATE_TEST_SUITE_P(MatMul,
                         MatMulRuntimeTest,
                         ::testing::ValuesIn(matmul_runtime_scenarios()),
                         [](const auto& info) {
                             return info.param->name();
                         });

TEST(GfxMatMulRuntime, CompileModelSucceeds) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, true);
    auto res = std::make_shared<ov::op::v0::Result>(matmul);
    auto model =
        std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{lhs, rhs}, "matmul_compile_only");

    ov::test::gfx::RuntimeModelRunner runner;
    runner.with_gfx_core([&](ov::Core& child_core) {
        auto gfx_cm = child_core.compile_model(model, "GFX", ov::test::gfx::fp16_compile_config());
        if (!static_cast<bool>(gfx_cm)) {
            throw std::runtime_error("compile_model returned empty compiled model");
        }
    }, 15);
}

TEST(GfxMatMulRuntime, CreateInferRequestSucceeds) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, true);
    auto res = std::make_shared<ov::op::v0::Result>(matmul);
    auto model =
        std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{lhs, rhs}, "matmul_create_request");

    ov::test::gfx::RuntimeModelRunner runner;
    runner.with_gfx_core([&](ov::Core& child_core) {
        auto gfx_cm = child_core.compile_model(model, "GFX", ov::test::gfx::fp16_compile_config());
        auto gfx_req = gfx_cm.create_infer_request();
        if (!static_cast<bool>(gfx_req)) {
            throw std::runtime_error("create_infer_request returned empty request");
        }
    }, 15);
}

}  // namespace
