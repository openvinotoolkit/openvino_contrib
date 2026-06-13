// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/group_normalization.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

#include "cuda_test_constants.hpp"

namespace {
using namespace ov::test;
using namespace ov::test::utils;

// precision, data shape, num_groups, epsilon
using GroupNormalizationParams = std::tuple<ov::element::Type, ov::Shape, std::int64_t, double>;

/**
 * GroupNormalization (opset12): splits the channel dim into num_groups groups,
 * normalizes each group over its channels and spatial extent, then applies a
 * per-channel scale and bias. The NVIDIA plugin implements it natively (the
 * GroupNormalizationDecomposition pass is disabled in the transformer).
 *
 * A custom SubgraphBaseTest is used so the case names stay free of the word
 * "dynamic" (the shared GroupNormalizationTest encodes the undefined in/out
 * precision as "dynamic", which the plugin's skip config filters out).
 */
class GroupNormalizationNVIDIATest : public ov::test::SubgraphBaseTest,
                                     public testing::WithParamInterface<GroupNormalizationParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GroupNormalizationParams>& obj) {
        const auto& [precision, data_shape, num_groups, epsilon] = obj.param;
        std::ostringstream result;
        result << "prec=" << precision << "_shape=" << ov::test::utils::vec2str(data_shape) << "_groups=" << num_groups
               << "_eps=" << epsilon;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = DEVICE_NVIDIA;
        const auto& [precision, data_shape, num_groups, epsilon] = GetParam();
        const ov::Shape channel_shape{data_shape.at(1)};

        // Normalization amplifies tiny per-element differences when a group's
        // variance approaches zero (1/sqrt(var+eps)), so f16 rounding alone can
        // exceed the framework's auto threshold. Relax it per precision.
        abs_threshold = (precision == ov::element::f16) ? 2e-2 : 2e-3;
        rel_threshold = (precision == ov::element::f16) ? 1e-2 : 1e-3;

        init_input_shapes(ov::test::static_shapes_to_test_representation(
            std::vector<ov::Shape>{data_shape, channel_shape, channel_shape}));

        auto data = std::make_shared<ov::op::v0::Parameter>(precision, inputDynamicShapes[0]);
        auto scale = std::make_shared<ov::op::v0::Parameter>(precision, inputDynamicShapes[1]);
        auto bias = std::make_shared<ov::op::v0::Parameter>(precision, inputDynamicShapes[2]);
        auto group_norm =
            std::make_shared<ov::op::v12::GroupNormalization>(data, scale, bias, num_groups, epsilon);
        auto result = std::make_shared<ov::op::v0::Result>(group_norm);
        function = std::make_shared<ov::Model>(ov::ResultVector{result},
                                               ov::ParameterVector{data, scale, bias},
                                               "GroupNormalization");
    }

    // Deterministic, well-spread inputs so every group has a variance safely
    // away from zero (a contiguous slice of the -2..2 ramp), keeping the
    // comparison stable regardless of group size. The range is kept small so
    // the f16 reference's sum-of-squares does not overflow for large groups
    // (8192 * 8^2 would exceed the f16 max, 65504).
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t in = 0; in < funcInputs.size(); ++in) {
            const auto& funcInput = funcInputs[in];
            ov::Tensor tensor{funcInput.get_element_type(), targetInputStaticShapes[in]};
            const size_t count = ov::shape_size(targetInputStaticShapes[in]);
            auto value = [in](size_t k) -> double {
                if (in == 0) return (static_cast<double>(k % 17) - 8.0) * 0.25;  // data: spread -2..2
                if (in == 1) return 0.5 + static_cast<double>(k % 5) * 0.25;     // scale: 0.5..1.5
                return static_cast<double>(k % 7) * 0.1 - 0.3;                   // bias: -0.3..0.3
            };
            if (funcInput.get_element_type() == ov::element::f16) {
                auto* p = tensor.data<ov::float16>();
                for (size_t k = 0; k < count; ++k) p[k] = ov::float16(static_cast<float>(value(k)));
            } else {
                auto* p = tensor.data<float>();
                for (size_t k = 0; k < count; ++k) p[k] = static_cast<float>(value(k));
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(GroupNormalizationNVIDIATest, Inference) {
    run();
}

const std::vector<ov::element::Type> precisions = {ov::element::f32, ov::element::f16};
const std::vector<double> epsilons = {1e-5, 1e-6};

// num_groups must divide the channel count (dim 1), so each shape is paired with
// its own valid group counts.
INSTANTIATE_TEST_SUITE_P(smoke_GroupNormalization_4d,
                         GroupNormalizationNVIDIATest,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::Values(ov::Shape{2, 32, 16, 16}),
                                            ::testing::ValuesIn(std::vector<std::int64_t>{1, 2, 4, 32}),
                                            ::testing::ValuesIn(epsilons)),
                         GroupNormalizationNVIDIATest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupNormalization_sd,  // Stable Diffusion UNet-like
                         GroupNormalizationNVIDIATest,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::Values(ov::Shape{1, 320, 8, 8}),
                                            ::testing::ValuesIn(std::vector<std::int64_t>{4, 32}),
                                            ::testing::ValuesIn(epsilons)),
                         GroupNormalizationNVIDIATest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupNormalization_3d,
                         GroupNormalizationNVIDIATest,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::Values(ov::Shape{3, 6, 5}),
                                            ::testing::ValuesIn(std::vector<std::int64_t>{1, 2, 3, 6}),
                                            ::testing::ValuesIn(epsilons)),
                         GroupNormalizationNVIDIATest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupNormalization_2d,
                         GroupNormalizationNVIDIATest,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::Values(ov::Shape{2, 8}),
                                            ::testing::ValuesIn(std::vector<std::int64_t>{1, 2, 4, 8}),
                                            ::testing::ValuesIn(epsilons)),
                         GroupNormalizationNVIDIATest::getTestCaseName);

}  // namespace
