// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

#include "cuda_test_constants.hpp"

namespace {
using namespace ov::test;
using namespace ov::test::utils;

// precision, data shape, axis, exclusive, reverse
using CumSumParams = std::tuple<ov::element::Type, ov::Shape, std::int64_t, bool, bool>;

class CumSumNVIDIATest : public ov::test::SubgraphBaseTest, public testing::WithParamInterface<CumSumParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CumSumParams>& obj) {
        const auto& [precision, data_shape, axis, exclusive, reverse] = obj.param;
        std::ostringstream result;
        result << "prec=" << precision << "_shape=" << ov::test::utils::vec2str(data_shape) << "_axis=" << axis
               << "_excl=" << exclusive << "_rev=" << reverse;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = DEVICE_NVIDIA;
        const auto& [precision, data_shape, axis, exclusive, reverse] = GetParam();
        abs_threshold = (precision == ov::element::f16) ? 5e-2 : 2e-3;

        init_input_shapes(ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>{data_shape}));
        auto data = std::make_shared<ov::op::v0::Parameter>(precision, inputDynamicShapes[0]);
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
        auto cum_sum = std::make_shared<ov::op::v0::CumSum>(data, axis_const, exclusive, reverse);
        auto result = std::make_shared<ov::op::v0::Result>(cum_sum);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{data}, "CumSum");
    }
};

TEST_P(CumSumNVIDIATest, Inference) {
    run();
}

const std::vector<ov::element::Type> precisions = {ov::element::f32, ov::element::f16, ov::element::i32};

INSTANTIATE_TEST_SUITE_P(smoke_CumSum,
                         CumSumNVIDIATest,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::ValuesIn(std::vector<ov::Shape>{{8}, {4, 5}, {2, 3, 4}}),
                                            ::testing::Values(std::int64_t{-1}),
                                            ::testing::Values(false, true),   // exclusive
                                            ::testing::Values(false, true)),  // reverse
                         CumSumNVIDIATest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CumSum_axis,
                         CumSumNVIDIATest,
                         ::testing::Values(CumSumParams{ov::element::f32, ov::Shape{3, 5, 7}, 0, false, false},
                                           CumSumParams{ov::element::f32, ov::Shape{3, 5, 7}, 1, true, false},
                                           CumSumParams{ov::element::i32, ov::Shape{4, 6}, 0, false, true}),
                         CumSumNVIDIATest::getTestCaseName);

}  // namespace
