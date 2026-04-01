// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/assign.hpp"

#include "cuda_test_constants.hpp"

namespace {
using namespace ov::test;
using namespace ov::test::utils;

using ReadValueAssignParams = std::tuple<
    InputShape,         // input shapes
    ov::element::Type   // input precision
>;

/**
 * Builds a simple stateful model:
 *   Parameter -> ReadValue(v0) -> Add(ReadValue, Parameter) -> Result
 *                                                           -> Assign(v0)
 *
 * On each inference, the state accumulates: output = state + input.
 * State is updated to the new output after each inference.
 */
class ReadValueAssignNVIDIATest : virtual public ov::test::SubgraphBaseTest,
                                   public testing::WithParamInterface<ReadValueAssignParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReadValueAssignParams>& obj) {
        const auto& [input_shapes, input_precision] = obj.param;

        std::ostringstream result;
        result << "IS=" << ov::test::utils::partialShape2str({input_shapes.first}) << "_";
        result << "TS=";
        for (const auto& shape : input_shapes.second) {
            result << ov::test::utils::partialShape2str({shape}) << "_";
        }
        result << "Precision=" << input_precision;
        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [input_shapes, input_precision] = GetParam();
        targetDevice = DEVICE_NVIDIA;

        init_input_shapes({input_shapes});

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(input_precision, shape));
        }
        auto read_value = std::make_shared<ov::op::v3::ReadValue>(params.at(0), "v0");
        auto add = std::make_shared<ov::op::v1::Add>(read_value, params.at(0));
        auto assign = std::make_shared<ov::op::v3::Assign>(add, "v0");
        auto res = std::make_shared<ov::op::v0::Result>(add);
        function = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::SinkVector{assign}, params);
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        auto data_tensor = ov::Tensor{funcInputs[0].get_element_type(), targetInputStaticShapes[0]};
        auto precision = funcInputs[0].get_element_type();

        if (precision == ov::element::f32) {
            auto data = data_tensor.data<float>();
            auto len = ov::shape_size(targetInputStaticShapes[0]);
            for (size_t i = 0; i < len; i++) {
                data[i] = static_cast<float>(i % 10 + 1);
            }
        } else if (precision == ov::element::f16) {
            auto data = data_tensor.data<ov::float16>();
            auto len = ov::shape_size(targetInputStaticShapes[0]);
            for (size_t i = 0; i < len; i++) {
                data[i] = ov::float16(static_cast<float>(i % 10 + 1));
            }
        } else if (precision == ov::element::i32) {
            auto data = data_tensor.data<int32_t>();
            auto len = ov::shape_size(targetInputStaticShapes[0]);
            for (size_t i = 0; i < len; i++) {
                data[i] = static_cast<int32_t>(i % 10 + 1);
            }
        }

        inputs.insert({funcInputs[0].get_node_shared_ptr(), data_tensor});
    }
};

TEST_P(ReadValueAssignNVIDIATest, Inference) {
    run();
}

// Static shapes
INSTANTIATE_TEST_SUITE_P(smoke_ReadValueAssign_Static_f32,
                         ReadValueAssignNVIDIATest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                {{1, 4, 20, 20}})),
                                            ::testing::Values(ov::element::f32)),
                         ReadValueAssignNVIDIATest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReadValueAssign_Static_f16,
                         ReadValueAssignNVIDIATest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                {{1, 4, 20, 20}})),
                                            ::testing::Values(ov::element::f16)),
                         ReadValueAssignNVIDIATest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReadValueAssign_Static_i32,
                         ReadValueAssignNVIDIATest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                {{7, 4, 20, 20}})),
                                            ::testing::Values(ov::element::i32)),
                         ReadValueAssignNVIDIATest::getTestCaseName);

}  // namespace
