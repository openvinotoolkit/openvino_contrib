// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/log_softmax.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

#include "cuda_test_constants.hpp"

namespace {
using namespace ov::test;
using namespace ov::test::utils;

// precision, data shape, axis
using LogSoftmaxParams = std::tuple<ov::element::Type, ov::Shape, std::int64_t>;

class LogSoftmaxNVIDIATest : public ov::test::SubgraphBaseTest,
                             public testing::WithParamInterface<LogSoftmaxParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LogSoftmaxParams>& obj) {
        const auto& [precision, data_shape, axis] = obj.param;
        std::ostringstream result;
        result << "prec=" << precision << "_shape=" << ov::test::utils::vec2str(data_shape) << "_axis=" << axis;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = DEVICE_NVIDIA;
        const auto& [precision, data_shape, axis] = GetParam();
        abs_threshold = (precision == ov::element::f16) ? 2e-2 : 2e-3;

        init_input_shapes(ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>{data_shape}));
        auto data = std::make_shared<ov::op::v0::Parameter>(precision, inputDynamicShapes[0]);
        auto log_softmax = std::make_shared<ov::op::v5::LogSoftmax>(data, axis);
        auto result = std::make_shared<ov::op::v0::Result>(log_softmax);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{data}, "LogSoftmax");
    }
};

TEST_P(LogSoftmaxNVIDIATest, Inference) {
    run();
}

const std::vector<ov::element::Type> precisions = {ov::element::f32, ov::element::f16};

// ASR/CTC heads reduce over the vocabulary (last axis); also cover middle/first.
INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax_lastAxis,
                         LogSoftmaxNVIDIATest,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::ValuesIn(std::vector<ov::Shape>{{4, 29}, {2, 50, 64}, {8}}),
                                            ::testing::Values(std::int64_t{-1})),
                         LogSoftmaxNVIDIATest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax_otherAxis,
                         LogSoftmaxNVIDIATest,
                         ::testing::Values(LogSoftmaxParams{ov::element::f32, ov::Shape{3, 5, 7}, 0},
                                           LogSoftmaxParams{ov::element::f32, ov::Shape{3, 5, 7}, 1},
                                           LogSoftmaxParams{ov::element::f16, ov::Shape{2, 8, 4}, 1}),
                         LogSoftmaxNVIDIATest::getTestCaseName);

}  // namespace
