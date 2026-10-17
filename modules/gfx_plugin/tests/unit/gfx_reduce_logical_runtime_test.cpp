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
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/result.hpp"

namespace {

using RuntimeScenarioPtr = std::shared_ptr<const ov::test::gfx::RuntimeScenario>;

std::shared_ptr<ov::op::v0::Constant> i64_constant(const ov::Shape& shape, std::vector<int64_t> values) {
    return std::make_shared<ov::op::v0::Constant>(ov::element::i64, shape, std::move(values));
}

ov::Tensor reduce_and_input() {
    ov::Tensor tensor(ov::element::boolean, ov::Shape{2, 3, 4});
    auto* data = tensor.data<uint8_t>();
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        data[i] = static_cast<uint8_t>((i % 5) != 2);
    }
    return tensor;
}

ov::Tensor reduce_or_input() {
    ov::Tensor tensor(ov::element::boolean, ov::Shape{2, 3, 4});
    auto* data = tensor.data<uint8_t>();
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        data[i] = static_cast<uint8_t>((i % 7) == 3 || i == 22);
    }
    return tensor;
}

RuntimeScenarioPtr scenario(std::string name,
                            ov::test::gfx::RuntimeModelBuilder model_builder,
                            ov::test::gfx::RuntimeInputBuilder input_builder,
                            int timeout = 15) {
    return ov::test::gfx::runtime_scenario(std::move(name),
                                           std::move(model_builder),
                                           std::move(input_builder),
                                           timeout,
                                           0.f);
}

std::vector<RuntimeScenarioPtr> reduce_logical_runtime_scenarios() {
    return {
        scenario("ReduceLogicalAnd",
                 [] {
                     auto param =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::Shape{2, 3, 4});
                     auto axes = i64_constant({1}, {1});
                     auto reduce = std::make_shared<ov::op::v1::ReduceLogicalAnd>(param, axes, false);
                     auto res = std::make_shared<ov::op::v0::Result>(reduce);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{param},
                                                        "reduce_logical_and_runtime");
                 },
                 [] {
                     return std::vector<ov::Tensor>{reduce_and_input()};
                 }),
        scenario("ReduceLogicalOrKeepDims",
                 [] {
                     auto param =
                         std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::Shape{2, 3, 4});
                     auto axes = i64_constant({2}, {1, 2});
                     auto reduce = std::make_shared<ov::op::v1::ReduceLogicalOr>(param, axes, true);
                     auto res = std::make_shared<ov::op::v0::Result>(reduce);
                     return std::make_shared<ov::Model>(ov::ResultVector{res},
                                                        ov::ParameterVector{param},
                                                        "reduce_logical_or_keepdims_runtime");
                 },
                 [] {
                     return std::vector<ov::Tensor>{reduce_or_input()};
                 }),
    };
}

class ReduceLogicalRuntimeTest : public ::testing::TestWithParam<RuntimeScenarioPtr> {};

TEST_P(ReduceLogicalRuntimeTest, MatchesTemplate) {
    const auto& scenario = *GetParam();
    ov::test::gfx::RuntimeModelRunner runner;
    runner.compare_model(scenario.make_model(),
                         scenario.make_inputs(),
                         scenario.timeout_seconds(),
                         scenario.atol(),
                         scenario.rtol());
}

INSTANTIATE_TEST_SUITE_P(ReduceLogical,
                         ReduceLogicalRuntimeTest,
                         ::testing::ValuesIn(reduce_logical_runtime_scenarios()),
                         [](const auto& info) {
                             return info.param->name();
                         });

}  // namespace
