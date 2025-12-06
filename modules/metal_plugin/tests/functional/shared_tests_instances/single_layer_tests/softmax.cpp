// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/softmax.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "../../test_constants.hpp"
#include "../../metal_test_utils.hpp"

using ov::test::subgraph::SoftMax8LayerTest;

class MetalSoftmaxLayerTest : public ov::test::utils::MetalVsTemplateLayerTest<SoftMax8LayerTest> {
protected:
    void SetUp() override {
        ov::test::utils::MetalVsTemplateLayerTest<SoftMax8LayerTest>::SetUp();
        // Softmax falls back to CPU path in METAL backend; align tolerances with CPU/GPU
        // to avoid tiny numeric deltas vs TEMPLATE reference.
        if (abs_threshold == 0.f) {
            abs_threshold = 5e-4f;
        }
        if (rel_threshold == 0.f) {
            rel_threshold = 5e-4f;
        }
    }

    void run_compare() {
        // Ensure thresholds are applied even if the base class overrides them later.
        abs_threshold = 5e-4f;
        rel_threshold = 5e-4f;
        ov::test::utils::MetalVsTemplateLayerTest<SoftMax8LayerTest>::run_compare();
    }
};

TEST_P(MetalSoftmaxLayerTest, CompareWithTemplate) {
    run_compare();
}

TEST_P(MetalSoftmaxLayerTest, CompareQueryModel) {
    this->query_model();
}

namespace {

const std::vector<ov::test::ElementType> netPrecisions = {
    ov::element::f32,
    ov::element::f16,
};

const std::vector<ov::Shape> inputStaticShape2D = {
    {1, 100},
    {100, 1},
    {10, 10},
};

const std::vector<ov::test::InputShape> inputDynamicShape2D = {
    {{ov::Dimension::dynamic(), 10}, {{1, 10}, {2, 10}, {10, 10}}},
    {{ov::Dimension(1, 10), 10}, {{1, 10}, {2, 10}, {10, 10}}},
    {{10, ov::Dimension::dynamic()}, {{10, 1}, {10, 5}, {10, 10}}},
    {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 10}, {2, 10}, {10, 10}}}};

const std::vector<int64_t> axis2D = {-2, -1, 0, 1};

const auto params2D_static =
    testing::Combine(testing::ValuesIn(netPrecisions),
                     ::testing::Values(ov::element::dynamic),
                     ::testing::Values(ov::element::dynamic),
                     testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputStaticShape2D)),
                     testing::ValuesIn(axis2D),
                     testing::Values(ov::test::utils::DEVICE_METAL),
                     testing::Values(ov::AnyMap()));

const auto params2D_dynamic = testing::Combine(testing::ValuesIn(netPrecisions),
                                               ::testing::Values(ov::element::dynamic),
                                               ::testing::Values(ov::element::dynamic),
                                               testing::ValuesIn(inputDynamicShape2D),
                                               testing::ValuesIn(axis2D),
                                               testing::Values(ov::test::utils::DEVICE_METAL),
                                               testing::Values(ov::AnyMap()));

INSTANTIATE_TEST_SUITE_P(Metal_smoke_SoftMax2D_static,
                         MetalSoftmaxLayerTest,
                         params2D_static,
                         SoftMax8LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Metal_smoke_SoftMax2D_dynamic,
                         MetalSoftmaxLayerTest,
                         params2D_dynamic,
                         SoftMax8LayerTest::getTestCaseName);

const std::vector<ov::Shape> inputStaticShape4D = {
    {1, 100, 1, 1},
    {50, 100, 4, 1},
    {2, 100, 10, 1},
};

const std::vector<ov::test::InputShape> inputDynamicShape4D = {
    {{ov::Dimension::dynamic(), 100, ov::Dimension(1, 10), 1}, {{1, 100, 1, 1}, {100, 100, 5, 1}}},
    {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()},
     {{1, 100, 1, 1}, {50, 100, 4, 1}, {2, 100, 10, 1}}},
};

const std::vector<ov::test::ElementType> netPrecisions4D = {
    ov::element::f32,
};

const std::vector<int64_t> axis4D = {0, 1, 2, 3, -1, -2, -3, -4};

const auto params4Dstatic =
    testing::Combine(testing::ValuesIn(netPrecisions4D),
                     ::testing::Values(ov::element::dynamic),
                     ::testing::Values(ov::element::dynamic),
                     testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputStaticShape4D)),
                     testing::ValuesIn(axis4D),
                     testing::Values(ov::test::utils::DEVICE_METAL),
                     testing::Values(ov::AnyMap()));

const auto params4Ddynamic = testing::Combine(testing::ValuesIn(netPrecisions4D),
                                              ::testing::Values(ov::element::dynamic),
                                              ::testing::Values(ov::element::dynamic),
                                              testing::ValuesIn(inputDynamicShape4D),
                                              testing::ValuesIn(axis4D),
                                              testing::Values(ov::test::utils::DEVICE_METAL),
                                              testing::Values(ov::AnyMap()));

INSTANTIATE_TEST_SUITE_P(Metal_smoke_SoftMax4D_static,
                         MetalSoftmaxLayerTest,
                         params4Dstatic,
                         SoftMax8LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Metal_smoke_SoftMax4D_dynamic,
                         MetalSoftmaxLayerTest,
                         params4Ddynamic,
                         SoftMax8LayerTest::getTestCaseName);

const std::vector<ov::Shape> inputStaticShape5D = {
    {1, 100, 1, 1, 1},
    {50, 100, 4, 1, 1},
    {2, 100, 10, 1, 1},
};

const std::vector<ov::test::InputShape> inputDynamicShape5D = {
    {{ov::Dimension::dynamic(), 100, ov::Dimension(1, 10), 1, 1}, {{1, 100, 1, 1, 1}, {100, 100, 5, 1, 1}}},
    {{ov::Dimension::dynamic(),
      ov::Dimension::dynamic(),
      ov::Dimension::dynamic(),
      ov::Dimension::dynamic(),
      ov::Dimension::dynamic()},
     {{1, 100, 1, 1, 1}, {50, 100, 4, 1, 1}, {2, 100, 10, 1, 1}}},
};

const std::vector<ov::test::ElementType> netPrecisions5D = {
    ov::element::f32,
};

const std::vector<int64_t> axis5D = {0, 1, 2, 3, 4, -1, -2, -3, -4, -5};

const auto params5Dstatic =
    testing::Combine(testing::ValuesIn(netPrecisions5D),
                     ::testing::Values(ov::element::dynamic),
                     ::testing::Values(ov::element::dynamic),
                     testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputStaticShape5D)),
                     testing::ValuesIn(axis5D),
                     testing::Values(ov::test::utils::DEVICE_METAL),
                     testing::Values(ov::AnyMap()));

const auto params5Ddynamic = testing::Combine(testing::ValuesIn(netPrecisions5D),
                                              ::testing::Values(ov::element::dynamic),
                                              ::testing::Values(ov::element::dynamic),
                                              testing::ValuesIn(inputDynamicShape5D),
                                              testing::ValuesIn(axis5D),
                                              testing::Values(ov::test::utils::DEVICE_METAL),
                                              testing::Values(ov::AnyMap()));

INSTANTIATE_TEST_SUITE_P(Metal_smoke_SoftMax5D_static,
                         MetalSoftmaxLayerTest,
                         params5Dstatic,
                         SoftMax8LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Metal_smoke_SoftMax5D_dynamic,
                         MetalSoftmaxLayerTest,
                         params5Ddynamic,
                         SoftMax8LayerTest::getTestCaseName);

}  // namespace
