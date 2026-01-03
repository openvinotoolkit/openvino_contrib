// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/convolution.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "integration/test_constants.hpp"
#include "shared_tests_instances/test_utils.hpp"

using ov::test::ConvolutionLayerTest;

class GfxConvolutionLayerTest : public ov::test::utils::GfxVsTemplateLayerTest<ConvolutionLayerTest> {
protected:
    void SetUp() override {
        GfxVsTemplateLayerTest::SetUp();
        const auto& params = GetParam();
        const auto& model_type = std::get<1>(params);
        if (model_type == ov::element::f16) {
            this->abs_threshold = 0.05f;
            this->rel_threshold = 0.05f;
        }
    }
};

TEST_P(GfxConvolutionLayerTest, CompareWithTemplate) {
    run_compare();
}

namespace {

const std::vector<ov::element::Type> model_types = {
    ov::element::f32,
    ov::element::f16,
};

/* ============= 2D Convolution ============= */

const std::vector<std::vector<size_t>> kernels = {{3, 3}, {3, 5}};
const std::vector<std::vector<size_t>> strides = {{1, 1}, {1, 3}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 0}, {0, 3}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 0}, {0, 3}};
const std::vector<std::vector<size_t>> dilations = {{1, 1}, {3, 1}};
const std::vector<size_t> numOutChannels = {1, 5};

const auto conv2DParams_ExplicitPadding = ::testing::Combine(::testing::ValuesIn(kernels),
                                                             ::testing::ValuesIn(strides),
                                                             ::testing::ValuesIn(padBegins),
                                                             ::testing::ValuesIn(padEnds),
                                                             ::testing::ValuesIn(dilations),
                                                             ::testing::ValuesIn(numOutChannels),
                                                             ::testing::Values(ov::op::PadType::EXPLICIT));

const auto conv2DParams_AutoPadValid = ::testing::Combine(::testing::ValuesIn(kernels),
                                                          ::testing::ValuesIn(strides),
                                                          ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                                                          ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                                                          ::testing::ValuesIn(dilations),
                                                          ::testing::ValuesIn(numOutChannels),
                                                          ::testing::Values(ov::op::PadType::VALID));

INSTANTIATE_TEST_SUITE_P(
    Gfx_Convolution2D_ExplicitPadding,
    GfxConvolutionLayerTest,
    ::testing::Combine(conv2DParams_ExplicitPadding,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation({{1, 3, 30, 30}})),
                       ::testing::Values(ov::test::utils::DEVICE_GFX)),
    ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    Gfx_Convolution2D_AutoPadValid,
    GfxConvolutionLayerTest,
    ::testing::Combine(conv2DParams_AutoPadValid,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation({{1, 3, 30, 30}})),
                       ::testing::Values(ov::test::utils::DEVICE_GFX)),
    ConvolutionLayerTest::getTestCaseName);

/* ============= 3D Convolution ============= */

const std::vector<std::vector<size_t>> kernels3d = {{3, 3, 3}, {3, 5, 3}};
const std::vector<std::vector<size_t>> strides3d = {{1, 1, 1}, {1, 2, 1}};
const std::vector<std::vector<ptrdiff_t>> paddings3d = {{0, 0, 0}, {0, 2, 0}};
const std::vector<std::vector<size_t>> dilations3d = {{1, 1, 1}, {1, 2, 1}};

const auto conv3DParams_ExplicitPadding = ::testing::Combine(::testing::ValuesIn(kernels3d),
                                                             ::testing::ValuesIn(strides3d),
                                                             ::testing::ValuesIn(paddings3d),
                                                             ::testing::ValuesIn(paddings3d),
                                                             ::testing::ValuesIn(dilations3d),
                                                             ::testing::Values(5),
                                                             ::testing::Values(ov::op::PadType::EXPLICIT));

const auto conv3DParams_AutoPadValid = ::testing::Combine(::testing::ValuesIn(kernels3d),
                                                          ::testing::ValuesIn(strides3d),
                                                          ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
                                                          ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
                                                          ::testing::ValuesIn(dilations3d),
                                                          ::testing::Values(5),
                                                          ::testing::Values(ov::op::PadType::VALID));

INSTANTIATE_TEST_SUITE_P(
    Gfx_smoke_Convolution3D_ExplicitPadding,
    GfxConvolutionLayerTest,
    ::testing::Combine(conv3DParams_ExplicitPadding,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation({{1, 3, 10, 10, 10}})),
                       ::testing::Values(ov::test::utils::DEVICE_GFX)),
    ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    Gfx_nightly_Convolution3D_AutoPadValid,
    GfxConvolutionLayerTest,
    ::testing::Combine(conv3DParams_AutoPadValid,
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::static_shapes_to_test_representation({{1, 3, 10, 10, 10}})),
                       ::testing::Values(ov::test::utils::DEVICE_GFX)),
    ConvolutionLayerTest::getTestCaseName);

}  // namespace
