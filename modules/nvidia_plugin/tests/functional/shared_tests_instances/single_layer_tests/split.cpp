// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/split.hpp"

#include "cuda_test_constants.hpp"

namespace {
using namespace ov::test;
using namespace ov::test::utils;

/**
 * Regression: a Split whose axis constant is a narrow integer type. GPT-2 emits
 * an i32 Split axis, and the plugin lowers index/axis constants to i32 anyway,
 * so SplitOp must read the axis with cast_vector. The pre-fix code used
 * get_data_ptr<int64_t>(), which over-reads the 4-byte constant and aborts
 * compile_model with "Buffer over-read".
 */
class SplitNVIDIATest : public ov::test::SubgraphBaseTest,
                        public testing::WithParamInterface<ov::element::Type> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ov::element::Type>& obj) {
        return "modelType=" + obj.param.to_string();
    }

protected:
    void SetUp() override {
        targetDevice = DEVICE_NVIDIA;
        const ov::element::Type model_type = GetParam();
        const ov::Shape data_shape{2, 6, 4};
        const int32_t axis_value = 1;  // split the dim of size 6
        const size_t num_splits = 3;

        init_input_shapes(ov::test::static_shapes_to_test_representation({data_shape}));
        auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0]);
        // Narrow (i32) axis constant — the scenario the pre-fix code over-read.
        auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int32_t>{axis_value});
        auto split = std::make_shared<ov::op::v1::Split>(param, axis, num_splits);

        ov::ResultVector results;
        for (size_t i = 0; i < split->get_output_size(); ++i) {
            results.push_back(std::make_shared<ov::op::v0::Result>(split->output(i)));
        }
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "Split");
    }
};

TEST_P(SplitNVIDIATest, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_Split_NarrowAxis,
                         SplitNVIDIATest,
                         ::testing::Values(ov::element::f32, ov::element::f16),
                         SplitNVIDIATest::getTestCaseName);

}  // namespace
