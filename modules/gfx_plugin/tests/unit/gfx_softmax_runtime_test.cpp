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
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/softmax.hpp"

namespace {

using RuntimeScenarioPtr = std::shared_ptr<const ov::test::gfx::RuntimeScenario>;

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
                            int timeout = 15,
                            float atol = 1e-5f) {
    return ov::test::gfx::runtime_scenario(std::move(name),
                                           std::move(model_builder),
                                           std::move(input_builder),
                                           timeout,
                                           atol);
}

std::vector<RuntimeScenarioPtr> softmax_runtime_scenarios() {
    return {
        scenario("SoftmaxV1LastAxis3D",
                 [] {
                     auto param =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 4});
                     auto softmax = std::make_shared<ov::op::v1::Softmax>(param, 2);
                     auto res = std::make_shared<ov::op::v0::Result>(softmax);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{param},
                                                        "softmax_v1_last_axis_3d");
                 },
                 [] {
                     return std::vector<ov::Tensor>{filled_f32({1, 4, 4}, 19, 9, 0.125f)};
                 }),
        scenario("SoftmaxV8NegativeAxis4D",
                 [] {
                     auto param =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 3, 5});
                     auto softmax = std::make_shared<ov::op::v8::Softmax>(param, -1);
                     auto res = std::make_shared<ov::op::v0::Result>(softmax);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{param},
                                                        "softmax_v8_negative_axis_4d");
                 },
                 [] {
                     return std::vector<ov::Tensor>{filled_f32({1, 2, 3, 5}, 29, 14, 0.0625f)};
                 }),
    };
}

class SoftmaxRuntimeTest : public ::testing::TestWithParam<RuntimeScenarioPtr> {};

TEST_P(SoftmaxRuntimeTest, MatchesTemplate) {
    const auto& scenario = *GetParam();
    ov::test::gfx::RuntimeModelRunner runner;
    runner.compare_model(scenario.make_model(),
                         scenario.make_inputs(),
                         scenario.timeout_seconds(),
                         scenario.atol(),
                         scenario.rtol());
}

INSTANTIATE_TEST_SUITE_P(Softmax,
                         SoftmaxRuntimeTest,
                         ::testing::ValuesIn(softmax_runtime_scenarios()),
                         [](const auto& info) {
                             return info.param->name();
                         });

}  // namespace
