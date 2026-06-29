// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/shape_of.hpp"

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/result.hpp"

#include "common_test_utils/test_constants.hpp"
#include "cuda_test_constants.hpp"

namespace {
using namespace ov::test;
using namespace ov::test::utils;

const std::vector<ov::element::Type> model_precisions = {
    ov::element::f32,
    ov::element::f16,
    ov::element::i32,
};

const std::vector<std::vector<ov::Shape>> input_shapes_static = {
    {{1}},
    {{1, 2}},
    {{1, 2, 3}},
    {{1, 2, 3, 4}},
    {{1, 2, 3, 4, 5}},
    {{10, 20, 30}},
};

INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf_i64,
                         ShapeOfLayerTest,
                         ::testing::Combine(::testing::ValuesIn(model_precisions),
                                            ::testing::Values(ov::element::i64),
                                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                                            ::testing::Values(DEVICE_NVIDIA)),
                         ShapeOfLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf_i32,
                         ShapeOfLayerTest,
                         ::testing::Combine(::testing::ValuesIn(model_precisions),
                                            ::testing::Values(ov::element::i32),
                                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                                            ::testing::Values(DEVICE_NVIDIA)),
                         ShapeOfLayerTest::getTestCaseName);

// Dynamic ShapeOf: drives the dynamic-input path (the input shape is taken from the
// DynamicBufferContext at runtime). Covers the fix that no longer falls back to an
// uninitialized static_input_shape_ for a dynamic input.
using ShapeOfDynamicParams = std::tuple<ov::element::Type,  // input precision
                                        ov::element::Type,  // output (shape) type
                                        InputShape>;        // dynamic input shape + targets

class ShapeOfDynamicNVIDIATest : public ov::test::SubgraphBaseTest,
                                 public testing::WithParamInterface<ShapeOfDynamicParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ShapeOfDynamicParams>& obj) {
        const auto& [in_prec, out_type, shapes] = obj.param;
        std::ostringstream result;
        result << "inPrc=" << in_prec << "_outType=" << out_type
               << "_IS=" << ov::test::utils::partialShape2str({shapes.first});
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = DEVICE_NVIDIA;
        const auto& [in_prec, out_type, shapes] = GetParam();
        init_input_shapes({shapes});
        auto param = std::make_shared<ov::op::v0::Parameter>(in_prec, inputDynamicShapes[0]);
        auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(param, out_type);
        auto res = std::make_shared<ov::op::v0::Result>(shape_of);
        function = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "ShapeOfDynamic");
    }
};

TEST_P(ShapeOfDynamicNVIDIATest, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_ShapeOf_Dynamic,
    ShapeOfDynamicNVIDIATest,
    ::testing::Combine(
        ::testing::Values(ov::element::f32),
        ::testing::Values(ov::element::i64, ov::element::i32),
        ::testing::Values(InputShape{{-1, -1, -1}, {ov::Shape{2, 3, 4}, ov::Shape{1, 5, 7}}},
                          InputShape{{-1, 4}, {ov::Shape{3, 4}}})),
    ShapeOfDynamicNVIDIATest::getTestCaseName);
}  // namespace
