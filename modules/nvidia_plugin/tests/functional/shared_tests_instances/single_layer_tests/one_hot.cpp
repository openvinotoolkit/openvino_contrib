// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/one_hot.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

#include "cuda_test_constants.hpp"

namespace {
using namespace ov::test;
using namespace ov::test::utils;

// output precision, indices shape, depth, axis
using OneHotParams = std::tuple<ov::element::Type, ov::Shape, std::int64_t, std::int64_t>;

class OneHotNVIDIATest : public ov::test::SubgraphBaseTest, public testing::WithParamInterface<OneHotParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<OneHotParams>& obj) {
        const auto& [precision, indices_shape, depth, axis] = obj.param;
        std::ostringstream result;
        result << "prec=" << precision << "_idx=" << ov::test::utils::vec2str(indices_shape) << "_depth=" << depth
               << "_axis=" << axis;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = DEVICE_NVIDIA;
        const auto& [precision, indices_shape, depth, axis] = GetParam();

        init_input_shapes(ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>{indices_shape}));
        auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, inputDynamicShapes[0]);
        auto depth_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {depth});
        auto on_value = ov::op::v0::Constant::create(precision, ov::Shape{}, {1});
        auto off_value = ov::op::v0::Constant::create(precision, ov::Shape{}, {0});
        auto one_hot = std::make_shared<ov::op::v1::OneHot>(indices, depth_const, on_value, off_value, axis);
        auto result = std::make_shared<ov::op::v0::Result>(one_hot);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{indices}, "OneHot");
    }

    // Indices in [0, depth) plus an out-of-range value (must yield all off).
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto funcInputs = function->inputs();
        const auto& funcInput = funcInputs[0];
        const auto depth = std::get<2>(GetParam());
        ov::Tensor tensor{funcInput.get_element_type(), targetInputStaticShapes[0]};
        auto* p = tensor.data<int32_t>();
        const size_t count = ov::shape_size(targetInputStaticShapes[0]);
        for (size_t k = 0; k < count; ++k) {
            // cycle 0..depth, where 'depth' itself is out of range -> all off_value
            p[k] = static_cast<int32_t>(k % (static_cast<size_t>(depth) + 1));
        }
        inputs.insert({funcInput.get_node_shared_ptr(), tensor});
    }
};

TEST_P(OneHotNVIDIATest, Inference) {
    run();
}

const std::vector<ov::element::Type> precisions = {ov::element::f32, ov::element::f16, ov::element::i32};

INSTANTIATE_TEST_SUITE_P(smoke_OneHot,
                         OneHotNVIDIATest,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::ValuesIn(std::vector<ov::Shape>{{6}, {2, 3}, {2, 2, 2}}),
                                            ::testing::Values(std::int64_t{5}),
                                            ::testing::Values(std::int64_t{-1})),
                         OneHotNVIDIATest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OneHot_axis,
                         OneHotNVIDIATest,
                         ::testing::Values(OneHotParams{ov::element::f32, ov::Shape{4}, 5, 0},
                                           OneHotParams{ov::element::f32, ov::Shape{2, 3}, 4, 1},
                                           OneHotParams{ov::element::i32, ov::Shape{3}, 7, 0}),
                         OneHotNVIDIATest::getTestCaseName);

}  // namespace
