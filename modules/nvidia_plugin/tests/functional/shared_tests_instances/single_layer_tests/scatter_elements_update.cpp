// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_elements_update.hpp"

#include "cuda_test_constants.hpp"

namespace {
using namespace ov::test;
using namespace ov::test::utils;

// precision, data shape, updates/indices shape, axis
using ScatterElementsUpdateParams = std::tuple<ov::element::Type, ov::Shape, ov::Shape, std::int64_t>;

class ScatterElementsUpdateNVIDIATest : public ov::test::SubgraphBaseTest,
                                        public testing::WithParamInterface<ScatterElementsUpdateParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ScatterElementsUpdateParams>& obj) {
        const auto& [precision, data_shape, updates_shape, axis] = obj.param;
        std::ostringstream result;
        result << "prec=" << precision << "_data=" << ov::test::utils::vec2str(data_shape)
               << "_upd=" << ov::test::utils::vec2str(updates_shape) << "_axis=" << axis;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = DEVICE_NVIDIA;
        const auto& [precision, data_shape, updates_shape, axis] = GetParam();
        const auto a = axis < 0 ? axis + static_cast<std::int64_t>(data_shape.size()) : axis;
        data_axis_dim_ = data_shape[a];
        updates_shape_ = updates_shape;

        init_input_shapes(ov::test::static_shapes_to_test_representation(
            std::vector<ov::Shape>{data_shape, updates_shape}));
        auto data = std::make_shared<ov::op::v0::Parameter>(precision, inputDynamicShapes[0]);
        auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, updates_shape);
        auto updates = std::make_shared<ov::op::v0::Parameter>(precision, inputDynamicShapes[1]);
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
        auto scatter = std::make_shared<ov::op::v3::ScatterElementsUpdate>(data, indices, updates, axis_const);
        auto result = std::make_shared<ov::op::v0::Result>(scatter);
        function = std::make_shared<ov::Model>(ov::ResultVector{result},
                                               ov::ParameterVector{data, indices, updates},
                                               "ScatterElementsUpdate");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto funcInputs = function->inputs();
        auto fillFloat = [](const ov::Output<ov::Node>& fi, const ov::Shape& shape, float bias) {
            ov::Tensor t{fi.get_element_type(), shape};
            const size_t n = ov::shape_size(shape);
            if (fi.get_element_type() == ov::element::f16) {
                auto* p = t.data<ov::float16>();
                for (size_t k = 0; k < n; ++k) p[k] = ov::float16(static_cast<float>(k % 19) + bias);
            } else {
                auto* p = t.data<float>();
                for (size_t k = 0; k < n; ++k) p[k] = static_cast<float>(k % 19) + bias;
            }
            return t;
        };
        inputs.insert({funcInputs[0].get_node_shared_ptr(), fillFloat(funcInputs[0], targetInputStaticShapes[0], 0.f)});
        // indices: valid axis positions in [0, data_axis_dim)
        {
            const auto& fi = funcInputs[1];
            ov::Tensor t{fi.get_element_type(), updates_shape_};
            auto* p = t.data<int32_t>();
            const size_t n = ov::shape_size(updates_shape_);
            for (size_t k = 0; k < n; ++k) p[k] = static_cast<int32_t>(k % static_cast<size_t>(data_axis_dim_));
            inputs.insert({fi.get_node_shared_ptr(), t});
        }
        inputs.insert(
            {funcInputs[2].get_node_shared_ptr(), fillFloat(funcInputs[2], targetInputStaticShapes[1], 100.f)});
    }

private:
    std::int64_t data_axis_dim_{};
    ov::Shape updates_shape_{};
};

TEST_P(ScatterElementsUpdateNVIDIATest, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_ScatterElementsUpdate,
    ScatterElementsUpdateNVIDIATest,
    ::testing::Values(
        ScatterElementsUpdateParams{ov::element::f32, ov::Shape{4, 3}, ov::Shape{2, 3}, 0},
        ScatterElementsUpdateParams{ov::element::f32, ov::Shape{3, 5}, ov::Shape{3, 2}, 1},
        ScatterElementsUpdateParams{ov::element::f16, ov::Shape{2, 3, 4}, ov::Shape{2, 3, 2}, 2},
        ScatterElementsUpdateParams{ov::element::f16, ov::Shape{2, 4, 4}, ov::Shape{2, 4, 4}, 1},  // duplicates
        ScatterElementsUpdateParams{ov::element::f32, ov::Shape{6}, ov::Shape{6}, -1}),
    ScatterElementsUpdateNVIDIATest::getTestCaseName);

}  // namespace
