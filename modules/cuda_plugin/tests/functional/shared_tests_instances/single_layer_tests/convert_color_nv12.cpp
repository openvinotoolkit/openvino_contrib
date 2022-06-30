// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <cuda_test_constants.hpp>

#include "single_layer_tests/convert_color_nv12.hpp"

using namespace LayerTestsDefinitions;

namespace {

class ConvertColorNV12CUDALayerTest : public ConvertColorNV12LayerTest {
 private:
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 255);
    }
};

TEST_P(ConvertColorNV12CUDALayerTest, CompareWithRefs) { Run(); }

const std::vector<ov::Shape> inShapes_nhwc = {
    {1,       10,     10,   1},
    {1,       50,     10,   1},
    {1,      100,     10,   1},
    {2,       10,     10,   1},
    {2,       50,     10,   1},
    {2,      100,     10,   1},
    {5,       10,     10,   1},
    {5,       50,     10,   1},
    {5,      100,     10,   1},
    {1,      96,      16,   1},
};

const std::vector<ov::element::Type> inTypes = {
    ov::element::u8,
    ov::element::f32,
    ov::element::f16,
};

const auto testCase_values = ::testing::Combine(
    ::testing::ValuesIn(inShapes_nhwc),
    ::testing::ValuesIn(inTypes),
    ::testing::Bool(),
    ::testing::Bool(),
    ::testing::Values(CommonTestUtils::DEVICE_CUDA)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsConvertColorNV12, ConvertColorNV12CUDALayerTest, testCase_values, ConvertColorNV12CUDALayerTest::getTestCaseName);

class ConvertColorNV12CUDAAccuracyTest : public ConvertColorNV12AccuracyTest {
 private:
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 255);
    }
};

TEST_P(ConvertColorNV12CUDAAccuracyTest, CompareWithRefs) { Run(); }

const auto testCase_accuracy_values = ::testing::Combine(
    ::testing::Values(ov::Shape{1, 96, 16, 1}),
    ::testing::Values(ov::element::u8),
    ::testing::Values(false),
    ::testing::Values(true),
    ::testing::Values(CommonTestUtils::DEVICE_CUDA)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsConvertColorNV12_acc,
                         ConvertColorNV12CUDAAccuracyTest,
                         testCase_accuracy_values,
                         ConvertColorNV12LayerTest::getTestCaseName);

const auto testCase_accuracy_values_nightly = ::testing::Combine(
    ::testing::Values(ov::Shape{1, 65536, 256, 1}),
    ::testing::Values(ov::element::u8),
    ::testing::Values(false),
    ::testing::Values(true),
    ::testing::Values(CommonTestUtils::DEVICE_CUDA)
);

INSTANTIATE_TEST_SUITE_P(nightly_TestsConvertColorNV12_acc,
                         ConvertColorNV12CUDAAccuracyTest,
                         testCase_accuracy_values_nightly,
                         ConvertColorNV12LayerTest::getTestCaseName);

}  // namespace
