// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

#include "cuda_test_constants.hpp"

namespace {
using namespace ov::test;
using namespace ov::test::utils;

// data precision, data shape, indices shape, axis
using GatherElementsParams = std::tuple<ov::element::Type, ov::Shape, ov::Shape, std::int64_t>;

class GatherElementsNVIDIATest : public ov::test::SubgraphBaseTest,
                                 public testing::WithParamInterface<GatherElementsParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherElementsParams>& obj) {
        const auto& [precision, data_shape, indices_shape, axis] = obj.param;
        std::ostringstream result;
        result << "prec=" << precision << "_data=" << ov::test::utils::vec2str(data_shape)
               << "_idx=" << ov::test::utils::vec2str(indices_shape) << "_axis=" << axis;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = DEVICE_NVIDIA;
        const auto& [precision, data_shape, indices_shape, axis] = GetParam();
        axis_ = axis < 0 ? axis + static_cast<std::int64_t>(data_shape.size()) : axis;
        data_axis_dim_ = data_shape[axis_];

        init_input_shapes(ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>{data_shape}));
        auto data = std::make_shared<ov::op::v0::Parameter>(precision, inputDynamicShapes[0]);
        // indices are a parameter so the framework compares the full op; filled
        // in generate_inputs with valid values along the gathered axis.
        indices_shape_ = indices_shape;
        auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, indices_shape);
        auto gather = std::make_shared<ov::op::v6::GatherElements>(data, indices, axis);
        auto result = std::make_shared<ov::op::v0::Result>(gather);
        function = std::make_shared<ov::Model>(ov::ResultVector{result},
                                               ov::ParameterVector{data, indices},
                                               "GatherElements");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto funcInputs = function->inputs();
        // data: spread values
        {
            const auto& fi = funcInputs[0];
            ov::Tensor t{fi.get_element_type(), targetInputStaticShapes[0]};
            const size_t n = ov::shape_size(targetInputStaticShapes[0]);
            if (fi.get_element_type() == ov::element::f16) {
                auto* p = t.data<ov::float16>();
                for (size_t k = 0; k < n; ++k) p[k] = ov::float16(static_cast<float>(k % 23) - 11.0f);
            } else {
                auto* p = t.data<float>();
                for (size_t k = 0; k < n; ++k) p[k] = static_cast<float>(k % 23) - 11.0f;
            }
            inputs.insert({fi.get_node_shared_ptr(), t});
        }
        // indices: valid coordinates in [0, data_axis_dim)
        {
            const auto& fi = funcInputs[1];
            ov::Tensor t{fi.get_element_type(), indices_shape_};
            auto* p = t.data<int32_t>();
            const size_t n = ov::shape_size(indices_shape_);
            for (size_t k = 0; k < n; ++k) p[k] = static_cast<int32_t>(k % static_cast<size_t>(data_axis_dim_));
            inputs.insert({fi.get_node_shared_ptr(), t});
        }
    }

private:
    std::int64_t axis_{};
    std::int64_t data_axis_dim_{};
    ov::Shape indices_shape_{};
};

TEST_P(GatherElementsNVIDIATest, Inference) {
    run();
}

const std::vector<ov::element::Type> precisions = {ov::element::f32, ov::element::f16};

INSTANTIATE_TEST_SUITE_P(
    smoke_GatherElements,
    GatherElementsNVIDIATest,
    ::testing::Values(GatherElementsParams{ov::element::f32, ov::Shape{3, 4}, ov::Shape{3, 2}, 1},
                      GatherElementsParams{ov::element::f32, ov::Shape{3, 4}, ov::Shape{5, 4}, 0},
                      GatherElementsParams{ov::element::f16, ov::Shape{2, 3, 4}, ov::Shape{2, 3, 6}, 2},
                      GatherElementsParams{ov::element::f16, ov::Shape{2, 3, 4}, ov::Shape{2, 5, 4}, 1},
                      GatherElementsParams{ov::element::f32, ov::Shape{4, 5}, ov::Shape{4, 5}, -1}),
    GatherElementsNVIDIATest::getTestCaseName);

}  // namespace
