// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gfx_runtime_model_runner.hpp"
#include "gfx_runtime_scenario.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"

namespace {

using RuntimeScenarioPtr = std::shared_ptr<const ov::test::gfx::RuntimeScenario>;

std::shared_ptr<ov::op::v0::Constant> f32_constant(const ov::Shape& shape, std::vector<float> values) {
    return std::make_shared<ov::op::v0::Constant>(ov::element::f32, shape, values);
}

std::shared_ptr<ov::op::v0::Constant> i64_constant(const ov::Shape& shape, std::vector<int64_t> values) {
    return std::make_shared<ov::op::v0::Constant>(ov::element::i64, shape, values);
}

ov::Tensor filled_f32(const ov::Shape& shape, int modulus, int shift, float scale) {
    ov::Tensor tensor(ov::element::f32, shape);
    auto* data = tensor.data<float>();
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        data[i] = static_cast<float>((static_cast<int>(i % modulus) - shift)) * scale;
    }
    return tensor;
}

ov::Tensor zero_f32(const ov::Shape& shape) {
    ov::Tensor tensor(ov::element::f32, shape);
    std::fill_n(tensor.data<float>(), tensor.get_size(), 0.0f);
    return tensor;
}

std::vector<ov::Tensor> split_1x4x6_input() {
    return {filled_f32({1, 4, 6}, 23, 11, 0.125f)};
}

std::vector<ov::Tensor> attention_layout_input() {
    return {filled_f32({1, 400, 384}, 251, 125, 0.03125f)};
}

std::vector<ov::Tensor> variadic_split_input() {
    return {filled_f32({1, 4, 96, 400}, 251, 125, 0.03125f)};
}

std::vector<ov::Tensor> variadic_split_with_zero_input() {
    auto inputs = variadic_split_input();
    inputs.push_back(zero_f32({1, 4, 32, 400}));
    return inputs;
}

std::shared_ptr<ov::op::v1::VariadicSplit> make_three_way_variadic_split(
    const ov::Output<ov::Node>& input,
    std::vector<int64_t> lengths = {32, 32, 32}) {
    auto split_axis = i64_constant({}, {2});
    auto split_lengths = i64_constant({3}, std::move(lengths));
    return std::make_shared<ov::op::v1::VariadicSplit>(input, split_axis, split_lengths);
}

std::shared_ptr<ov::Model> make_attention_layout_split_model(
    const std::function<ov::ResultVector(const std::shared_ptr<ov::op::v1::VariadicSplit>&)>& results_builder,
    const std::string& name) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 400, 384});
    auto reshape_shape = i64_constant({4}, {1, 400, 4, 96});
    auto reshaped = std::make_shared<ov::op::v1::Reshape>(param, reshape_shape, true);
    auto perm = i64_constant({4}, {0, 2, 3, 1});
    auto transposed = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
    auto split = make_three_way_variadic_split(transposed);
    return std::make_shared<ov::Model>(results_builder(split), ov::ParameterVector{param}, name);
}

RuntimeScenarioPtr scenario(std::string name,
                            ov::test::gfx::RuntimeModelBuilder model_builder,
                            ov::test::gfx::RuntimeInputBuilder input_builder,
                            int timeout = 20,
                            float atol = 1e-4f) {
    return ov::test::gfx::runtime_scenario(std::move(name),
                                           std::move(model_builder),
                                           std::move(input_builder),
                                           timeout,
                                           atol);
}

std::vector<RuntimeScenarioPtr> split_runtime_scenarios() {
    return {
        scenario("SplitDirectOutputs",
                 [] {
                     auto param =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 6});
                     auto axis = i64_constant({}, {2});
                     auto split = std::make_shared<ov::op::v1::Split>(param, axis, 3);
                     ov::ResultVector results;
                     for (size_t i = 0; i < 3; ++i) {
                         results.push_back(std::make_shared<ov::op::v0::Result>(split->output(i)));
                     }
                     return std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "split_runtime");
                 },
                 split_1x4x6_input,
                 15,
                 1e-5f),
        scenario("VariadicSplitAttentionLayoutScaledOperand",
                 [] {
                     return make_attention_layout_split_model(
                         [](const std::shared_ptr<ov::op::v1::VariadicSplit>& split) {
                             auto scale = f32_constant({1, 1, 1, 1}, {0.176776695f});
                             auto mul = std::make_shared<ov::op::v1::Multiply>(split->output(1), scale);
                             return ov::ResultVector{std::make_shared<ov::op::v0::Result>(mul)};
                         },
                         "attn_layout_scaled");
                 },
                 attention_layout_input),
        scenario("VariadicSplitAttentionLayoutOutputs",
                 [] {
                     return make_attention_layout_split_model(
                         [](const std::shared_ptr<ov::op::v1::VariadicSplit>& split) {
                             return ov::ResultVector{
                                 std::make_shared<ov::op::v0::Result>(split->output(0)),
                                 std::make_shared<ov::op::v0::Result>(split->output(1)),
                                 std::make_shared<ov::op::v0::Result>(split->output(2)),
                             };
                         },
                         "attn_layout_split_outputs");
                 },
                 attention_layout_input),
        scenario("VariadicSplitScaledOutputs",
                 [] {
                     auto param =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 400});
                     auto split = make_three_way_variadic_split(param);
                     auto scale = f32_constant({1, 1, 1, 1}, {0.176776695f});
                     auto mul = std::make_shared<ov::op::v1::Multiply>(split->output(1), scale);
                     ov::ResultVector results{
                         std::make_shared<ov::op::v0::Result>(split->output(0)),
                         std::make_shared<ov::op::v0::Result>(mul),
                     };
                     return std::make_shared<ov::Model>(results,
                                                        ov::ParameterVector{param},
                                                        "variadic_split_scaled_outputs");
                 },
                 variadic_split_input),
        scenario("VariadicSplitInferredLengthOutputs",
                 [] {
                     auto param =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 16});
                     auto split = make_three_way_variadic_split(param, {24, -1, 24});
                     ov::ResultVector results{
                         std::make_shared<ov::op::v0::Result>(split->output(0)),
                         std::make_shared<ov::op::v0::Result>(split->output(1)),
                         std::make_shared<ov::op::v0::Result>(split->output(2)),
                     };
                     return std::make_shared<ov::Model>(results,
                                                        ov::ParameterVector{param},
                                                        "variadic_split_inferred_length");
                 },
                 [] {
                     return std::vector<ov::Tensor>{filled_f32({1, 4, 96, 16}, 251, 125, 0.03125f)};
                 }),
        scenario("VariadicSplitAddOnly",
                 [] {
                     auto param =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 400});
                     auto zero =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
                     auto split = make_three_way_variadic_split(param);
                     auto lhs = std::make_shared<ov::op::v1::Add>(split->output(0), zero);
                     auto res = std::make_shared<ov::op::v0::Result>(lhs);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{param, zero},
                                                        "variadic_split_add_only");
                 },
                 variadic_split_with_zero_input),
        scenario("VariadicSplitValueLayout",
                 [] {
                     auto param =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 96, 400});
                     auto split = make_three_way_variadic_split(param);
                     auto value_perm = i64_constant({4}, {0, 3, 1, 2});
                     auto value_transpose = std::make_shared<ov::op::v1::Transpose>(split->output(2), value_perm);
                     auto value_shape = i64_constant({3}, {1, 400, 128});
                     auto value_reshape = std::make_shared<ov::op::v1::Reshape>(value_transpose, value_shape, true);
                     auto res = std::make_shared<ov::op::v0::Result>(value_reshape);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{param},
                                                        "variadic_split_value_layout");
                 },
                 variadic_split_input),
    };
}

class SplitRuntimeTest : public ::testing::TestWithParam<RuntimeScenarioPtr> {};

TEST_P(SplitRuntimeTest, MatchesTemplate) {
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

INSTANTIATE_TEST_SUITE_P(Split,
                         SplitRuntimeTest,
                         ::testing::ValuesIn(split_runtime_scenarios()),
                         [](const auto& info) {
                             return info.param->name();
                         });

}  // namespace
