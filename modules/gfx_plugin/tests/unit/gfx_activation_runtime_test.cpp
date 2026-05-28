// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gfx_runtime_model_runner.hpp"
#include "gfx_runtime_scenario.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/hsigmoid.hpp"
#include "openvino/op/hswish.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/op/softsign.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/tanh.hpp"

namespace {

using RuntimeScenarioPtr = std::shared_ptr<const ov::test::gfx::RuntimeScenario>;
using ActivationBuilder =
    std::function<std::shared_ptr<ov::Node>(const ov::Output<ov::Node>&)>;

ov::Tensor filled_f32(const ov::Shape& shape) {
    ov::Tensor tensor(ov::element::f32, shape);
    auto* data = tensor.data<float>();
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        const int signed_value = static_cast<int>(i % 23) - 11;
        data[i] = static_cast<float>(signed_value) * 0.25f;
    }
    return tensor;
}

ov::Tensor scalar_f32(float value) {
    ov::Tensor tensor(ov::element::f32, ov::Shape{});
    tensor.data<float>()[0] = value;
    return tensor;
}

RuntimeScenarioPtr activation_scenario(std::string name,
                                       ActivationBuilder activation_builder,
                                       float atol = 1e-4f,
                                       float rtol = 1e-4f) {
    return ov::test::gfx::runtime_scenario(
        std::move(name),
        [activation_builder = std::move(activation_builder)] {
            auto input =
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                        ov::Shape{1, 4, 8, 8});
            auto activation = activation_builder(input);
            auto result = std::make_shared<ov::op::v0::Result>(activation);
            return std::make_shared<ov::Model>(
                ov::ResultVector{result},
                ov::ParameterVector{input},
                "activation_runtime");
        },
        [] {
            return std::vector<ov::Tensor>{filled_f32({1, 4, 8, 8})};
        },
        90,
        atol,
        rtol);
}

RuntimeScenarioPtr dynamic_swish_scenario() {
    return ov::test::gfx::runtime_scenario(
        "SwishDynamicBeta",
        [] {
            auto input =
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                        ov::Shape{1, 4, 8, 8});
            auto beta =
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                        ov::Shape{});
            auto activation = std::make_shared<ov::op::v4::Swish>(input, beta);
            auto result = std::make_shared<ov::op::v0::Result>(activation);
            return std::make_shared<ov::Model>(
                ov::ResultVector{result},
                ov::ParameterVector{input, beta},
                "activation_runtime_dynamic_swish");
        },
        [] {
            return std::vector<ov::Tensor>{filled_f32({1, 4, 8, 8}),
                                           scalar_f32(0.5f)};
        },
        90,
        1e-4f,
        1e-4f);
}

std::vector<RuntimeScenarioPtr> activation_runtime_scenarios() {
    return {
        activation_scenario("Relu", [](const ov::Output<ov::Node>& input) {
            return std::make_shared<ov::op::v0::Relu>(input);
        }),
        activation_scenario("Sigmoid", [](const ov::Output<ov::Node>& input) {
            return std::make_shared<ov::op::v0::Sigmoid>(input);
        }),
        activation_scenario("Tanh", [](const ov::Output<ov::Node>& input) {
            return std::make_shared<ov::op::v0::Tanh>(input);
        }),
        activation_scenario("Elu", [](const ov::Output<ov::Node>& input) {
            return std::make_shared<ov::op::v0::Elu>(input, 0.5);
        }),
        activation_scenario("Clamp", [](const ov::Output<ov::Node>& input) {
            return std::make_shared<ov::op::v0::Clamp>(input, -0.25, 0.75);
        }),
        activation_scenario("HSwish", [](const ov::Output<ov::Node>& input) {
            return std::make_shared<ov::op::v4::HSwish>(input);
        }),
        activation_scenario("HSigmoid", [](const ov::Output<ov::Node>& input) {
            return std::make_shared<ov::op::v5::HSigmoid>(input);
        }),
        activation_scenario("SoftPlus", [](const ov::Output<ov::Node>& input) {
            return std::make_shared<ov::op::v4::SoftPlus>(input);
        }),
        activation_scenario("SwishDefaultBeta", [](const ov::Output<ov::Node>& input) {
            return std::make_shared<ov::op::v4::Swish>(input);
        }),
        activation_scenario("SwishStaticBeta", [](const ov::Output<ov::Node>& input) {
            const auto beta = ov::op::v0::Constant::create(
                ov::element::f32, ov::Shape{}, {0.5f});
            return std::make_shared<ov::op::v4::Swish>(input, beta);
        }),
        dynamic_swish_scenario(),
        activation_scenario("SoftSign", [](const ov::Output<ov::Node>& input) {
            return std::make_shared<ov::op::v9::SoftSign>(input);
        }),
        activation_scenario("Sign", [](const ov::Output<ov::Node>& input) {
            return std::make_shared<ov::op::v0::Sign>(input);
        }),
        activation_scenario("RoundEven", [](const ov::Output<ov::Node>& input) {
            return std::make_shared<ov::op::v5::Round>(
                input, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
        }),
        activation_scenario("RoundAway", [](const ov::Output<ov::Node>& input) {
            return std::make_shared<ov::op::v5::Round>(
                input, ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
        }),
    };
}

class ActivationRuntimeTest : public ::testing::TestWithParam<RuntimeScenarioPtr> {};

TEST_P(ActivationRuntimeTest, MatchesTemplate) {
    const auto& scenario = *GetParam();
    ov::test::gfx::RuntimeModelRunner runner;
    runner.compare_model(scenario.make_model(),
                         scenario.make_inputs(),
                         scenario.timeout_seconds(),
                         scenario.atol(),
                         scenario.rtol());
}

INSTANTIATE_TEST_SUITE_P(Activation,
                         ActivationRuntimeTest,
                         ::testing::ValuesIn(activation_runtime_scenarios()),
                         [](const auto& info) {
                             return info.param->name();
                         });

}  // namespace
