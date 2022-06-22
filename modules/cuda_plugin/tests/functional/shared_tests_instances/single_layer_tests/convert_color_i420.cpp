// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/convert_color_i420.hpp"

#include <cuda_test_constants.hpp>
#include <vector>

using namespace LayerTestsDefinitions;

namespace {

class ConvertColorI420CUDALayerTest : public ConvertColorI420LayerTest {
private:
    void SetUp() override {
        ov::Shape inputShape;
        ov::element::Type ngPrc;
        bool conversionToRGB, singlePlane;
        abs_threshold = 1.0f;  // I420 conversion can use various algorithms, thus some absolute deviation is allowed
        threshold = 1.f;       // Ignore relative comparison for I420 convert (allow 100% relative deviation)
        std::tie(inputShape, ngPrc, conversionToRGB, singlePlane, targetDevice) = GetParam();
        if (singlePlane) {
            inputShape[1] = inputShape[1] * 3 / 2;
            auto param = std::make_shared<ov::op::v0::Parameter>(ngPrc, inputShape);
            std::shared_ptr<ov::Node> convert_color;
            if (conversionToRGB) {
                convert_color = std::make_shared<ov::op::v8::I420toRGB>(param);
            } else {
                convert_color = std::make_shared<ov::op::v8::I420toRGB>(param);
            }
            function = std::make_shared<ov::Model>(
                std::make_shared<ov::op::v0::Result>(convert_color), ov::ParameterVector{param}, "ConvertColorI420");
        } else {
            auto uvShape = ov::Shape{inputShape[0], inputShape[1] / 2, inputShape[2] / 2, 1};
            auto param_y = std::make_shared<ov::op::v0::Parameter>(ngPrc, inputShape);
            auto param_u = std::make_shared<ov::op::v0::Parameter>(ngPrc, uvShape);
            auto param_v = std::make_shared<ov::op::v0::Parameter>(ngPrc, uvShape);
            std::shared_ptr<ov::Node> convert_color;
            if (conversionToRGB) {
                convert_color = std::make_shared<ov::op::v8::I420toRGB>(param_y, param_u, param_v);
            } else {
                convert_color = std::make_shared<ov::op::v8::I420toBGR>(param_y, param_u, param_v);
            }
            function = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(convert_color),
                                                   ov::ParameterVector{param_y, param_u, param_v},
                                                   "ConvertColorI420");
        }
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 255);
    }
};

TEST_P(ConvertColorI420CUDALayerTest, CompareWithRefs) { Run(); }

class ConvertColorI420CUDAAccuracyTest : public ConvertColorI420AccuracyTest {
private:
    void SetUp() override {
        ov::Shape inputShape;
        ov::element::Type ngPrc;
        bool conversionToRGB, singlePlane;
        abs_threshold = 1.0f;  // I420 conversion can use various algorithms, thus some absolute deviation is allowed
        threshold = 1.f;       // Ignore relative comparison for I420 convert (allow 100% relative deviation)
        std::tie(inputShape, ngPrc, conversionToRGB, singlePlane, targetDevice) = GetParam();
        if (singlePlane) {
            inputShape[1] = inputShape[1] * 3 / 2;
            auto param = std::make_shared<ov::op::v0::Parameter>(ngPrc, inputShape);
            std::shared_ptr<ov::Node> convert_color;
            if (conversionToRGB) {
                convert_color = std::make_shared<ov::op::v8::I420toRGB>(param);
            } else {
                convert_color = std::make_shared<ov::op::v8::I420toRGB>(param);
            }
            function = std::make_shared<ov::Model>(
                std::make_shared<ov::op::v0::Result>(convert_color), ov::ParameterVector{param}, "ConvertColorI420");
        } else {
            auto uvShape = ov::Shape{inputShape[0], inputShape[1] / 2, inputShape[2] / 2, 1};
            auto param_y = std::make_shared<ov::op::v0::Parameter>(ngPrc, inputShape);
            auto param_u = std::make_shared<ov::op::v0::Parameter>(ngPrc, uvShape);
            auto param_v = std::make_shared<ov::op::v0::Parameter>(ngPrc, uvShape);
            std::shared_ptr<ov::Node> convert_color;
            if (conversionToRGB) {
                convert_color = std::make_shared<ov::op::v8::I420toRGB>(param_y, param_u, param_v);
            } else {
                convert_color = std::make_shared<ov::op::v8::I420toBGR>(param_y, param_u, param_v);
            }
            function = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(convert_color),
                                                   ov::ParameterVector{param_y, param_u, param_v},
                                                   "ConvertColorI420");
        }
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 255);
    }
};

TEST_P(ConvertColorI420CUDAAccuracyTest, CompareWithRefs) { Run(); }

const std::vector<ov::Shape> inShapes_nhwc = {
    {1, 10, 10, 1},
    {1, 50, 10, 1},
    {1, 100, 10, 1},
    {2, 10, 10, 1},
    {2, 50, 10, 1},
    {2, 100, 10, 1},
    {5, 10, 10, 1},
    {5, 50, 10, 1},
    {5, 100, 10, 1},
    {1, 96, 16, 1},
};

const std::vector<ov::element::Type> inTypes = {
    ov::element::u8,
    ov::element::f32,
    ov::element::f16,
};

const auto testCase_values = ::testing::Combine(::testing::ValuesIn(inShapes_nhwc),
                                                ::testing::ValuesIn(inTypes),
                                                ::testing::Bool(),
                                                ::testing::Bool(),
                                                ::testing::Values(CommonTestUtils::DEVICE_CUDA));

INSTANTIATE_TEST_SUITE_P(smoke_TestsConvertColorI420,
                         ConvertColorI420CUDALayerTest,
                         testCase_values,
                         ConvertColorI420CUDALayerTest::getTestCaseName);

const auto testCase_accuracy_values = ::testing::Combine(::testing::Values(ov::Shape{1, 96, 16, 1}),
                                                         ::testing::Values(ov::element::u8),
                                                         ::testing::Values(false),
                                                         ::testing::Values(true),
                                                         ::testing::Values(CommonTestUtils::DEVICE_CUDA));

INSTANTIATE_TEST_SUITE_P(smoke_TestsConvertColorI420_acc,
                         ConvertColorI420CUDAAccuracyTest,
                         testCase_accuracy_values,
                         ConvertColorI420CUDALayerTest::getTestCaseName);

const auto testCase_accuracy_values_nightly = ::testing::Combine(::testing::Values(ov::Shape{1, 65536, 256, 1}),
                                                                 ::testing::Values(ov::element::u8),
                                                                 ::testing::Values(false),
                                                                 ::testing::Values(true),
                                                                 ::testing::Values(CommonTestUtils::DEVICE_CUDA));

INSTANTIATE_TEST_SUITE_P(nightly_TestsConvertColorI420_acc,
                         ConvertColorI420CUDAAccuracyTest,
                         testCase_accuracy_values_nightly,
                         ConvertColorI420CUDALayerTest::getTestCaseName);

}  // namespace
