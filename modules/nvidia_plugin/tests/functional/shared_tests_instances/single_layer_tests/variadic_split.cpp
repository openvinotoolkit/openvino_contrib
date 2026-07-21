// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "cuda_test_constants.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "single_op_tests/variadic_split.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/variadic_split.hpp"

namespace {

using namespace ov::test;
using namespace ov::test::utils;
using ov::test::VariadicSplitLayerTest;

const std::vector<ov::element::Type> model_types = {
    ov::element::f32,
    ov::element::f16
};

// Sum of elements numSplits = inputShapes[Axis]
const std::vector<std::vector<size_t>> num_splits = {
    {1, 16, 5, 8},
    {2, 19, 5, 4},
    {7, 13, 2, 8},
    {5, 8, 12, 5},
    {4, 11, 6, 9},
};

const std::vector<int64_t> axis = {-3, -2, -1, 0, 1, 2, 3};

INSTANTIATE_TEST_CASE_P(num_splitsCheck,
                        VariadicSplitLayerTest,
                        ::testing::Combine(::testing::ValuesIn(num_splits),
                                           ::testing::ValuesIn(axis),
                                           ::testing::ValuesIn(model_types),
                                           ::testing::Values(static_shapes_to_test_representation({{30, 30, 30, 30}})),
                                           ::testing::Values(DEVICE_NVIDIA)),
                        VariadicSplitLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(yolov5_NumSplitsCheck_shape0_axis1,
                        VariadicSplitLayerTest,
                        ::testing::Combine(::testing::Values(std::vector<size_t>{1, 1, 1}),
                                           ::testing::Values(1),
                                           ::testing::ValuesIn(model_types),
                                           ::testing::Values(static_shapes_to_test_representation({{1, 3, 40, 40, 85}})),
                                           ::testing::Values(DEVICE_NVIDIA)),
                        VariadicSplitLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(yolov5_NumSplitsCheck_shape0_axis2,
                        VariadicSplitLayerTest,
                        ::testing::Combine(::testing::Values(std::vector<size_t>{10, 20, 10}),
                                           ::testing::Values(2),
                                           ::testing::ValuesIn(model_types),
                                           ::testing::Values(static_shapes_to_test_representation({{1, 3, 40, 40, 85}})),
                                           ::testing::Values(DEVICE_NVIDIA)),
                        VariadicSplitLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(yolov5_NumSplitsCheck_shape0_axis3,
                        VariadicSplitLayerTest,
                        ::testing::Combine(::testing::Values(std::vector<size_t>{5, 5, 30}),
                                           ::testing::Values(3),
                                           ::testing::ValuesIn(model_types),
                                           ::testing::Values(static_shapes_to_test_representation({{1, 3, 40, 40, 85}})),
                                           ::testing::Values(DEVICE_NVIDIA)),
                        VariadicSplitLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(yolov5_NumSplitsCheck_shape0_axis4,
                        VariadicSplitLayerTest,
                        ::testing::Combine(::testing::Values(std::vector<size_t>{20, 60, 5}),
                                           ::testing::Values(4),
                                           ::testing::ValuesIn(model_types),
                                           ::testing::Values(static_shapes_to_test_representation({{1, 3, 40, 40, 85}})),
                                           ::testing::Values(DEVICE_NVIDIA)),
                        VariadicSplitLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(yolov5_NumSplitsCheck_shape1_axis1,
                        VariadicSplitLayerTest,
                        ::testing::Combine(::testing::Values(std::vector<size_t>{1, 1, 1}),
                                           ::testing::Values(1),
                                           ::testing::ValuesIn(model_types),
                                           ::testing::Values(static_shapes_to_test_representation({{1, 3, 20, 20, 85}})),
                                           ::testing::Values(DEVICE_NVIDIA)),
                        VariadicSplitLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(yolov5_NumSplitsCheck_shape1_axis2,
                        VariadicSplitLayerTest,
                        ::testing::Combine(::testing::Values(std::vector<size_t>{5, 12, 3}),
                                           ::testing::Values(2),
                                           ::testing::ValuesIn(model_types),
                                           ::testing::Values(static_shapes_to_test_representation({{1, 3, 20, 20, 85}})),
                                           ::testing::Values(DEVICE_NVIDIA)),
                        VariadicSplitLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(yolov5_NumSplitsCheck_shape1_axis3,
                        VariadicSplitLayerTest,
                        ::testing::Combine(::testing::Values(std::vector<size_t>{2, 8, 10}),
                                           ::testing::Values(3),
                                           ::testing::ValuesIn(model_types),
                                           ::testing::Values(static_shapes_to_test_representation({{1, 3, 20, 20, 85}})),
                                           ::testing::Values(DEVICE_NVIDIA)),
                        VariadicSplitLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(yolov5_NumSplitsCheck_shape1_axis4,
                        VariadicSplitLayerTest,
                        ::testing::Combine(::testing::Values(std::vector<size_t>{65, 3, 17}),
                                           ::testing::Values(4),
                                           ::testing::ValuesIn(model_types),
                                           ::testing::Values(static_shapes_to_test_representation({{1, 3, 20, 20, 85}})),
                                           ::testing::Values(DEVICE_NVIDIA)),
                        VariadicSplitLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(yolov5_NumSplitsCheck_shape2_axis1,
                        VariadicSplitLayerTest,
                        ::testing::Combine(::testing::Values(std::vector<size_t>{1, 1, 1}),
                                           ::testing::Values(1),
                                           ::testing::ValuesIn(model_types),
                                           ::testing::Values(static_shapes_to_test_representation({{1, 3, 80, 80, 85}})),
                                           ::testing::Values(DEVICE_NVIDIA)),
                        VariadicSplitLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(yolov5_NumSplitsCheck_shape2_axis2,
                        VariadicSplitLayerTest,
                        ::testing::Combine(::testing::Values(std::vector<size_t>{13, 13, 54}),
                                           ::testing::Values(2),
                                           ::testing::ValuesIn(model_types),
                                           ::testing::Values(static_shapes_to_test_representation({{1, 3, 80, 80, 85}})),
                                           ::testing::Values(DEVICE_NVIDIA)),
                        VariadicSplitLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(yolov5_NumSplitsCheck_shape2_axis3,
                        VariadicSplitLayerTest,
                        ::testing::Combine(::testing::Values(std::vector<size_t>{7, 3, 70}),
                                           ::testing::Values(3),
                                           ::testing::ValuesIn(model_types),
                                           ::testing::Values(static_shapes_to_test_representation({{1, 3, 80, 80, 85}})),
                                           ::testing::Values(DEVICE_NVIDIA)),
                        VariadicSplitLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(yolov5_NumSplitsCheck_shape2_axis4,
                        VariadicSplitLayerTest,
                        ::testing::Combine(::testing::Values(std::vector<size_t>{1, 1, 83}),
                                           ::testing::Values(4),
                                           ::testing::ValuesIn(model_types),
                                           ::testing::Values(static_shapes_to_test_representation({{1, 3, 80, 80, 85}})),
                                           ::testing::Values(DEVICE_NVIDIA)),
                        VariadicSplitLayerTest::getTestCaseName);

// The shared VariadicSplitLayerTest takes split lengths as unsigned values, so
// it can never exercise a "-1" (remaining) length. GPT-2's QKV split uses
// {768, 768, -1}; the custom test below covers that.
using VariadicSplitRemainingParams = std::tuple<size_t,                // size of the split axis
                                                std::vector<int64_t>>;  // split lengths, one of them -1

/**
 * Regression: VariadicSplit with a "-1" (remaining) split length.
 * buildSplitIndexHelper must build its index table from the resolved per-output
 * sizes; using the raw split_lengths lets the -1 decrease the running offset and
 * then index split_lengths out of bounds, so the table holds split indices >=
 * the number of outputs and the kernel reads its metadata/output-pointer arrays
 * out of bounds (illegal memory access at inference).
 */
class VariadicSplitRemainingNVIDIATest : public ov::test::SubgraphBaseTest,
                                         public testing::WithParamInterface<VariadicSplitRemainingParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<VariadicSplitRemainingParams>& obj) {
        const auto& [axis_size, lengths] = obj.param;
        std::ostringstream result;
        result << "axisSize=" << axis_size << "_lengths=";
        for (auto l : lengths) {
            result << l << ".";
        }
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = DEVICE_NVIDIA;
        const auto& [axis_size, lengths] = GetParam();
        const int64_t axis_value = 1;
        const ov::Shape data_shape{2, axis_size, 3};

        init_input_shapes(ov::test::static_shapes_to_test_representation({data_shape}));
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0]);
        auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{axis_value});
        auto lengths_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{lengths.size()}, lengths);
        auto variadic_split = std::make_shared<ov::op::v1::VariadicSplit>(param, axis, lengths_const);

        ov::ResultVector results;
        for (size_t i = 0; i < variadic_split->get_output_size(); ++i) {
            results.push_back(std::make_shared<ov::op::v0::Result>(variadic_split->output(i)));
        }
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "VariadicSplitRemaining");
    }
};

TEST_P(VariadicSplitRemainingNVIDIATest, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit_RemainingPart,
                         VariadicSplitRemainingNVIDIATest,
                         ::testing::Values(VariadicSplitRemainingParams{4, {1, 1, -1}},   // GPT-2-like: -1 last
                                           VariadicSplitRemainingParams{6, {2, -1, 2}}),  // -1 in the middle
                         VariadicSplitRemainingNVIDIATest::getTestCaseName);
}  // namespace
