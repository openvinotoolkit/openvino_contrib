// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_test_constants.hpp>
#include <ngraph_functions/builders.hpp>
#include <vector>

#include "finite_comparer.hpp"

namespace LayerTestsDefinitions {

struct FullyConnectedShapeRelatedParams {
  std::pair<InferenceEngine::SizeVector, bool> input1, input2;
  InferenceEngine::SizeVector input3;
};

typedef std::tuple<
    FullyConnectedShapeRelatedParams,
    InferenceEngine::Precision,        // Network precision
    InferenceEngine::Precision,        // Input precision
    InferenceEngine::Precision,        // Output precision
    InferenceEngine::Layout,           // Input layout
    ngraph::helpers::InputLayerType,   // Secondary input type
    LayerTestsUtils::TargetDevice,     // Device name
    std::map<std::string, std::string> // Additional network configuration
> FullyConnectedLayerTestParamsSet;

class FullyConnectedLayerTest : public testing::WithParamInterface<FullyConnectedLayerTestParamsSet>
                              , public FiniteLayerComparer {
 public:
  static std::string getTestCaseName(const testing::TestParamInfo<FullyConnectedLayerTestParamsSet> &obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout;
    FullyConnectedShapeRelatedParams shapeRelatedParams;
    ngraph::helpers::InputLayerType secondaryInputType;
    std::string targetDevice;
    std::map<std::string, std::string> additionalConfig;
    std::tie(shapeRelatedParams, netPrecision, inPrc, outPrc, inLayout, secondaryInputType, targetDevice, additionalConfig) =
        obj.param;

    std::ostringstream result;
    result << "IS0=" << CommonTestUtils::vec2str(shapeRelatedParams.input1.first) << "_";
    result << "IS1=" << CommonTestUtils::vec2str(shapeRelatedParams.input2.first) << "_";
    result << "transpose_a=" << shapeRelatedParams.input1.second << "_";
    result << "transpose_b=" << shapeRelatedParams.input2.second << "_";
    result << "secondaryInputType=" << secondaryInputType << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "trgDev=" << targetDevice;
    result << "config=(";
    for (const auto configEntry : additionalConfig) {
      result << configEntry.first << ", " << configEntry.second << ":";
    }
    result << ")";
    return result.str();
  }

 protected:
  void SetUp() override {
    FullyConnectedShapeRelatedParams shapeRelatedParams;
    ngraph::helpers::InputLayerType secondaryInputType;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::map<std::string, std::string> additionalConfig;
    std::tie(shapeRelatedParams, netPrecision, inPrc, outPrc, inLayout, secondaryInputType, targetDevice, additionalConfig) =
        this->GetParam();

    configuration.insert(additionalConfig.begin(), additionalConfig.end());

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {shapeRelatedParams.input1.first});

    auto secondaryInput = ngraph::builder::makeInputLayer(ngPrc, secondaryInputType, shapeRelatedParams.input2.first);
    if (secondaryInputType == ngraph::helpers::InputLayerType::PARAMETER) {
      params.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(secondaryInput));
    }
    auto thirdInput = ngraph::builder::makeInputLayer(ngPrc, ngraph::helpers::InputLayerType::CONSTANT, shapeRelatedParams.input3);
    auto paramOuts = ngraph::helpers::convert2OutputVector(
        ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto MatMul = std::dynamic_pointer_cast<ngraph::opset3::MatMul>(
        ngraph::builder::makeMatMul(paramOuts[0], secondaryInput, shapeRelatedParams.input1.second, shapeRelatedParams.input2.second));
    auto Add = std::dynamic_pointer_cast<ngraph::opset3::Add>(
        ngraph::builder::makeEltwise(MatMul, thirdInput, ngraph::helpers::EltwiseTypes::ADD));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(Add)};
    function = std::make_shared<ngraph::Function>(results, params, "FullyConnected");
  }
};

TEST_P(FullyConnectedLayerTest, CompareWithRefs) {
  SKIP_IF_CURRENT_TEST_IS_DISABLED()

  auto params = GetParam();
  inPrc = std::get<1>(params);
  outPrc = std::get<2>(params);

  Run();
}

namespace {

const std::vector<InferenceEngine::Precision> inputFP16Precisions = {
    InferenceEngine::Precision::FP16,
};

const std::vector<InferenceEngine::Precision> inputFP32Precisions = {
    InferenceEngine::Precision::FP32,
};

// General
const std::vector<FullyConnectedShapeRelatedParams> smokeShapeRelatedParams = {
    // ND x ND: [B, ..., X, Y] x [B, ..., Y, Z] => [B, ..., X, Z]
    { { {2, 10, 10}, false }, { {2, 10, 20}, false }, {10, 20} },
    { { {2, 10, 10}, true }, { {2, 10, 20}, false }, {10, 20} },
    { { {2, 10, 20}, false }, { {2, 10, 20}, true }, {10, 10} },
    { { {2, 20, 10}, true }, { {2, 10, 20}, true }, {10, 10} },

    // ND x ND: [B, ..., X, Y] x [B, ..., Y, Z] => [B, ..., X, Z]
    { { {10, 10}, false }, { {10, 20}, false }, {10, 20} },
    { { {10, 10}, true }, { {10, 20}, false }, {10, 20} },
    { { {10, 20}, false }, { {10, 20}, true }, {10, 10} },
    { { {20, 10}, true }, { {10, 20}, true }, {10, 10} },

    // 1D x ND: [X] x [B, ..., X, Y] -> [1, X] x [B, ..., X, Y] -> [B, ..., 1, Y] => [B, ..., Y]
    { { {10}, false }, { {10, 20}, false }, {1, 20} },
    { { {10}, true }, { {10, 20}, false }, {1, 20} },
    { { {20}, false }, { {10, 20}, true }, {10} },
    { { {20}, true }, { {10, 20}, true }, {10} },
};

// NOTE: Resnet-50 shapes
const std::vector<FullyConnectedShapeRelatedParams> resnet50ShapeRelatedParams = {
    { { {1, 2048}, false }, { {2048, 1001}, false }, {1, 1001}  },
    { { {1, 2048}, false }, { {1001, 2048}, true }, {1, 1001}  },
};

// NOTE: VGG-16 shapes
const std::vector<FullyConnectedShapeRelatedParams> vgg16ShapeRelatedParams = {
    { { {1, 25088}, false }, { {4096, 25088}, true }, {1, 4096} },
    { { {1, 25088}, false }, { {25088, 4096}, false }, {1, 4096} },
    { { {1, 4096}, false }, { {4096, 4096}, true }, {1, 4096} },
    { { {1, 4096}, false }, { {4096, 4096}, false }, {1, 4096} },
    { { {1, 4096}, false }, { {1000, 4096}, true }, {1, 1000} },
    { { {1, 4096}, false }, { {4096, 1000}, false }, {1, 1000} },
};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
    ngraph::helpers::InputLayerType::CONSTANT,
    ngraph::helpers::InputLayerType::PARAMETER,
};

std::map<std::string, std::string> additional_config = {};

INSTANTIATE_TEST_CASE_P(smoke_MatMulFP16, FullyConnectedLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(smokeShapeRelatedParams),
                            ::testing::ValuesIn(inputFP16Precisions),
                            ::testing::Values(InferenceEngine::Precision::FP16),
                            ::testing::Values(InferenceEngine::Precision::FP16),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::ValuesIn(secondaryInputTypes),
                            ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                            ::testing::Values(additional_config)),
                        FullyConnectedLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_MatMulFP32, FullyConnectedLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(smokeShapeRelatedParams),
                            ::testing::ValuesIn(inputFP32Precisions),
                            ::testing::Values(InferenceEngine::Precision::FP32),
                            ::testing::Values(InferenceEngine::Precision::FP32),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::ValuesIn(secondaryInputTypes),
                            ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                            ::testing::Values(additional_config)),
                        FullyConnectedLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(MatMul_Resnet50FP16, FullyConnectedLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(resnet50ShapeRelatedParams),
                            ::testing::ValuesIn(inputFP16Precisions),
                            ::testing::Values(InferenceEngine::Precision::FP16),
                            ::testing::Values(InferenceEngine::Precision::FP16),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::ValuesIn(secondaryInputTypes),
                            ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                            ::testing::Values(additional_config)),
                        FullyConnectedLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(MatMul_Resnet50FP32, FullyConnectedLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(resnet50ShapeRelatedParams),
                            ::testing::ValuesIn(inputFP32Precisions),
                            ::testing::Values(InferenceEngine::Precision::FP32),
                            ::testing::Values(InferenceEngine::Precision::FP32),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::ValuesIn(secondaryInputTypes),
                            ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                            ::testing::Values(additional_config)),
                        FullyConnectedLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(MatMul_VGG16FP16, FullyConnectedLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(vgg16ShapeRelatedParams),
                            ::testing::ValuesIn(inputFP16Precisions),
                            ::testing::Values(InferenceEngine::Precision::FP16),
                            ::testing::Values(InferenceEngine::Precision::FP16),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::ValuesIn(secondaryInputTypes),
                            ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                            ::testing::Values(additional_config)),
                        FullyConnectedLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(MatMul_VGG16FP32, FullyConnectedLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(vgg16ShapeRelatedParams),
                            ::testing::ValuesIn(inputFP32Precisions),
                            ::testing::Values(InferenceEngine::Precision::FP32),
                            ::testing::Values(InferenceEngine::Precision::FP32),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::ValuesIn(secondaryInputTypes),
                            ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                            ::testing::Values(additional_config)),
                        FullyConnectedLayerTest::getTestCaseName);

} // namespace
} // namespace LayerTestsDefinitions
