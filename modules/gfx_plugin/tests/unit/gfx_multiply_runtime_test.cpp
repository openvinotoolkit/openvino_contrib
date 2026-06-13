// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gfx_runtime_model_runner.hpp"
#include "gfx_runtime_scenario.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
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

std::vector<RuntimeScenarioPtr> multiply_runtime_scenarios() {
    return {
        scenario("Broadcast400x400",
                 [] {
                     auto param =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 400, 400});
                     auto scale = f32_constant({1, 1, 1, 1}, {0.176776695f});
                     auto mul = std::make_shared<ov::op::v1::Multiply>(param, scale);
                     auto res = std::make_shared<ov::op::v0::Result>(mul);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{param},
                                                        "multiply_broadcast_400x400_runtime");
                 },
                 [] {
                     return std::vector<ov::Tensor>{filled_f32({1, 4, 400, 400}, 251, 125, 0.03125f)};
                 },
                 15,
                 1e-5f),
        scenario("LargeBroadcast",
                 [] {
                     auto input =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 32, 400});
                     auto scale = f32_constant({1, 1, 1, 1}, {0.176776695f});
                     auto mul = std::make_shared<ov::op::v1::Multiply>(input, scale);
                     auto res = std::make_shared<ov::op::v0::Result>(mul);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{input},
                                                        "multiply_large_broadcast_runtime");
                 },
                 [] {
                     return std::vector<ov::Tensor>{filled_f32({1, 4, 32, 400}, 251, 125, 0.03125f)};
                 }),
    };
}

class MultiplyRuntimeTest : public ::testing::TestWithParam<RuntimeScenarioPtr> {};

TEST_P(MultiplyRuntimeTest, MatchesTemplate) {
    const auto& scenario = *GetParam();
    ov::test::gfx::RuntimeModelRunner runner;
    runner.compare_model(scenario.make_model(),
                         scenario.make_inputs(),
                         scenario.timeout_seconds(),
                         scenario.atol(),
                         scenario.rtol());
}

INSTANTIATE_TEST_SUITE_P(Multiply,
                         MultiplyRuntimeTest,
                         ::testing::ValuesIn(multiply_runtime_scenarios()),
                         [](const auto& info) {
                             return info.param->name();
                         });

}  // namespace
