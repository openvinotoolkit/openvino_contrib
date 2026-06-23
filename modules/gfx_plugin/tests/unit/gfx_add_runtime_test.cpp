// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gfx_runtime_model_runner.hpp"
#include "gfx_runtime_scenario.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace {

using RuntimeScenarioPtr = std::shared_ptr<const ov::test::gfx::RuntimeScenario>;

std::shared_ptr<ov::op::v0::Constant> f32_constant(const ov::Shape& shape, std::vector<float> values) {
    return std::make_shared<ov::op::v0::Constant>(ov::element::f32, shape, std::move(values));
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

std::vector<RuntimeScenarioPtr> add_runtime_scenarios() {
    return {
        scenario("BroadcastBias",
                 [] {
                     auto param =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 256, 20, 20});
                     std::vector<float> bias_vals(256, 0.0f);
                     for (size_t i = 0; i < bias_vals.size(); ++i) {
                         bias_vals[i] = static_cast<float>((static_cast<int>(i % 13) - 6)) * 0.125f;
                     }
                     auto bias = f32_constant({1, 256, 1, 1}, std::move(bias_vals));
                     auto add = std::make_shared<ov::op::v1::Add>(param, bias);
                     auto res = std::make_shared<ov::op::v0::Result>(add);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{param},
                                                        "add_broadcast_bias_runtime");
                 },
                 [] {
                     return std::vector<ov::Tensor>{filled_f32({1, 256, 20, 20}, 37, 18, 0.0625f)};
                 },
                 15,
                 1e-5f),
        scenario("ProducedLhs",
                 [] {
                     auto lhs_in =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
                     auto zero =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
                     auto lhs = std::make_shared<ov::op::v1::Add>(lhs_in, zero);
                     auto res = std::make_shared<ov::op::v0::Result>(lhs);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{lhs_in, zero},
                                                        "add_produced_lhs_runtime");
                 },
                 [] {
                     return std::vector<ov::Tensor>{filled_f32({1, 4, 32, 400}, 251, 125, 0.03125f),
                                                    zero_f32({1, 4, 32, 400})};
                 }),
    };
}

class AddRuntimeTest : public ::testing::TestWithParam<RuntimeScenarioPtr> {};

TEST_P(AddRuntimeTest, MatchesTemplate) {
    const auto& scenario = *GetParam();
    ov::test::gfx::RuntimeModelRunner runner;
    runner.compare_model(scenario.make_model(),
                         scenario.make_inputs(),
                         scenario.timeout_seconds(),
                         scenario.atol(),
                         scenario.rtol());
}

INSTANTIATE_TEST_SUITE_P(Add,
                         AddRuntimeTest,
                         ::testing::ValuesIn(add_runtime_scenarios()),
                         [](const auto& info) {
                             return info.param->name();
                         });

}  // namespace
