// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/gather_nd.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

#include "cuda_test_constants.hpp"

namespace {
using namespace ov::test;
using namespace ov::test::utils;

// data precision, data shape, indices shape, batch_dims
using GatherNDParams = std::tuple<ov::element::Type, ov::Shape, ov::Shape, std::size_t>;

class GatherNDNVIDIATest : public ov::test::SubgraphBaseTest, public testing::WithParamInterface<GatherNDParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherNDParams>& obj) {
        const auto& [precision, data_shape, indices_shape, batch_dims] = obj.param;
        std::ostringstream result;
        result << "prec=" << precision << "_data=" << ov::test::utils::vec2str(data_shape)
               << "_idx=" << ov::test::utils::vec2str(indices_shape) << "_batch=" << batch_dims;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = DEVICE_NVIDIA;
        const auto& [precision, data_shape, indices_shape, batch_dims] = GetParam();
        data_shape_ = data_shape;
        indices_shape_ = indices_shape;
        batch_dims_ = batch_dims;

        init_input_shapes(ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>{data_shape}));
        auto data = std::make_shared<ov::op::v0::Parameter>(precision, inputDynamicShapes[0]);
        auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, indices_shape);
        auto gather = std::make_shared<ov::op::v8::GatherND>(data, indices, batch_dims);
        auto result = std::make_shared<ov::op::v0::Result>(gather);
        function =
            std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{data, indices}, "GatherND");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto funcInputs = function->inputs();
        // data
        {
            const auto& fi = funcInputs[0];
            ov::Tensor t{fi.get_element_type(), targetInputStaticShapes[0]};
            const size_t n = ov::shape_size(targetInputStaticShapes[0]);
            if (fi.get_element_type() == ov::element::f16) {
                auto* p = t.data<ov::float16>();
                for (size_t k = 0; k < n; ++k) p[k] = ov::float16(static_cast<float>(k % 23) - 11.0f);
            } else if (fi.get_element_type() == ov::element::i32) {
                auto* p = t.data<int32_t>();
                for (size_t k = 0; k < n; ++k) p[k] = static_cast<int32_t>(k);
            } else {
                auto* p = t.data<float>();
                for (size_t k = 0; k < n; ++k) p[k] = static_cast<float>(k % 23) - 11.0f;
            }
            inputs.insert({fi.get_node_shared_ptr(), t});
        }
        // indices: each L-tuple component valid for its indexed data dim
        {
            const auto& fi = funcInputs[1];
            ov::Tensor t{fi.get_element_type(), indices_shape_};
            auto* p = t.data<int32_t>();
            const size_t n = ov::shape_size(indices_shape_);
            const size_t L = indices_shape_.back();
            for (size_t k = 0; k < n; ++k) {
                const size_t l = k % L;
                const size_t dim = data_shape_[batch_dims_ + l];
                p[k] = static_cast<int32_t>((k / L + l) % dim);
            }
            inputs.insert({fi.get_node_shared_ptr(), t});
        }
    }

private:
    ov::Shape data_shape_{};
    ov::Shape indices_shape_{};
    std::size_t batch_dims_{};
};

TEST_P(GatherNDNVIDIATest, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_GatherND,
    GatherNDNVIDIATest,
    ::testing::Values(GatherNDParams{ov::element::f32, ov::Shape{4, 3}, ov::Shape{2, 1}, 0},     // slice rows
                      GatherNDParams{ov::element::f32, ov::Shape{4, 3}, ov::Shape{2, 2}, 0},     // full index
                      GatherNDParams{ov::element::f16, ov::Shape{2, 3, 4}, ov::Shape{2, 2}, 0},  // index 2 dims
                      GatherNDParams{ov::element::f32, ov::Shape{2, 3, 4}, ov::Shape{5, 3}, 0},  // index all 3
                      GatherNDParams{ov::element::f32, ov::Shape{2, 3, 4}, ov::Shape{2, 1}, 1},  // batch_dims=1
                      GatherNDParams{ov::element::i32, ov::Shape{6}, ov::Shape{3, 1}, 0}),
    GatherNDNVIDIATest::getTestCaseName);

}  // namespace
