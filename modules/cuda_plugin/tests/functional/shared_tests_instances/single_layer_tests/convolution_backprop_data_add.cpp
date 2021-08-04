// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/convolution_backprop_data.hpp"

#include <ie_common.h>

#include <ie_precision.hpp>
#include <ngraph/node.hpp>
#include <vector>

#include "cuda_test_constants.hpp"
#include "finite_comparer.hpp"
#include "ngraph_functions/builders.hpp"

using namespace LayerTestsDefinitions;

namespace {

using convBackpropDataExtendedLayerTestParamsSet = std::tuple<
    convBackpropDataSpecificParams,
    InferenceEngine::Precision,     // Net precision
    InferenceEngine::Precision,     // Input precision
    InferenceEngine::Precision,     // Output precision
    InferenceEngine::Layout,        // Input layout
    InferenceEngine::Layout,        // Output layout
    InferenceEngine::SizeVector,    // Input shapes
    InferenceEngine::SizeVector,    // Output shape data
    LayerTestsUtils::TargetDevice   // Device name
    >;

class ConvolutionBackpropDataAddExtendedLayerTest
    : public testing::WithParamInterface<
    convBackpropDataExtendedLayerTestParamsSet>
    , virtual public LayerTestsUtils::LayerTestsCommon {
 public:
  static std::string getTestCaseName(testing::TestParamInfo<convBackpropDataExtendedLayerTestParamsSet> obj) {
    convBackpropDataSpecificParams convBackpropDataParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::SizeVector outputShapeData;
    std::string targetDevice;
    std::tie(convBackpropDataParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, outputShapeData, targetDevice) = obj.param;
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convBackpropDataParams;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "OSD=" << CommonTestUtils::vec2str(outputShapeData) << "_";
    result << "K" << CommonTestUtils::vec2str(kernel) << "_";
    result << "S" << CommonTestUtils::vec2str(stride) << "_";
    result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
    result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
    result << "D=" << CommonTestUtils::vec2str(dilation) << "_";
    result << "O=" << convOutChannels << "_";
    result << "AP=" << padType << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
  }

 protected:
  std::shared_ptr<ngraph::Node> makeConvolutionBackpropData(
      const ngraph::Output<ngraph::Node> &in,
      const ngraph::Output<ngraph::Node> &output,
      const ngraph::element::Type &type,
      const std::vector<size_t> &filterSize,
      const std::vector<size_t> &strides,
      const std::vector<ptrdiff_t> &padsBegin,
      const std::vector<ptrdiff_t> &padsEnd,
      const std::vector<size_t> &dilations,
      const ngraph::op::PadType &autoPad,
      size_t numOutChannels,
      bool addBiases = false,
      const std::vector<float> &filterWeights = {},
      const std::vector<float> &biasesWeights = {}) {
    bool randomFilterWeights = filterWeights.empty();
    auto shape = in.get_shape();
    std::vector<size_t> filterWeightsShape = {shape[1], numOutChannels};
    filterWeightsShape.insert(filterWeightsShape.end(), filterSize.begin(), filterSize.end());
    auto filterWeightsNode = ngraph::builder::makeConstant(type, filterWeightsShape, filterWeights, randomFilterWeights);

    return makeConvolutionBackpropData(in, filterWeightsNode, output, type, strides, padsBegin, padsEnd, dilations, autoPad, addBiases, biasesWeights);
  }

  std::shared_ptr<ngraph::Node> makeConvolutionBackpropData(
      const ngraph::Output<ngraph::Node> &in,
      const ngraph::Output<ngraph::Node> &weights,
      const ngraph::Output<ngraph::Node> &output,
      const ngraph::element::Type &type,
      const std::vector<size_t> &strides,
      const std::vector<ptrdiff_t> &padsBegin,
      const std::vector<ptrdiff_t> &padsEnd,
      const std::vector<size_t> &dilations,
      const ngraph::op::PadType &autoPad,
      bool addBiases = false,
      const std::vector<float> &biasesWeights = {}) {
    return std::make_shared<ngraph::opset1::ConvolutionBackpropData>(
        in, weights, output, strides, padsBegin, padsEnd, dilations, autoPad);
  }

  void SetUp() override {
    convBackpropDataSpecificParams convBackpropDataParams;
    std::vector<size_t> inputShape;
    std::vector<size_t> outputShapeData;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(convBackpropDataParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, outputShapeData, targetDevice) = this->GetParam();
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convBackpropDataParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto outputShapeNode = ngraph::builder::makeConstant(ngraph::element::Type_t::i64, {outputShapeData.size()}, outputShapeData);
    auto paramOuts = ngraph::helpers::convert2OutputVector(
        ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto convBackpropData = std::dynamic_pointer_cast<ngraph::opset1::ConvolutionBackpropData>(
        makeConvolutionBackpropData(paramOuts[0], outputShapeNode, ngPrc, kernel, stride, padBegin,
                                    padEnd, dilation, padType, convOutChannels));
    auto addConstant = ngraph::builder::makeConstant(ngPrc, outputShapeData, outputShapeData, true);
    auto add = ngraph::builder::makeEltwise(convBackpropData, addConstant, ngraph::helpers::EltwiseTypes::ADD);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(add)};
    function = std::make_shared<ngraph::Function>(results, params, "convolutionBackpropData");
  }
};

TEST_P(ConvolutionBackpropDataAddExtendedLayerTest, CompareWithRefs) {
  SKIP_IF_CURRENT_TEST_IS_DISABLED()

  auto params = GetParam();
  inPrc = std::get<2>(params);
  outPrc = std::get<3>(params);

  Run();
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32,
    };

/* ============= ConvolutionBackpropData params (2D) ============= */
//Attributes: {'dilations': '1,1', 'strides': '2,2'}
//Inputs: {1, 64, 34, 34}, {64, 32, 2, 2}, {2}
//Outputs: {1, 32, 64, 64}
const InferenceEngine::SizeVector input2D_group_0 = { 1, 64, 34, 34 };
const InferenceEngine::SizeVector output2D_group_0 = { 64, 64 };
const size_t numOutChannels_group_0 = 32;
const InferenceEngine::SizeVector kernels2D_group_0 = { 2, 2 };
const InferenceEngine::SizeVector strides2D_group_0 = { 2, 2 };
const std::vector<ptrdiff_t> padBegins2D_group_0 = { 2, 2 };
const std::vector<ptrdiff_t> padEnds2D_group_0 = { 2, 2 };
const InferenceEngine::SizeVector dilations2D_group_0 = { 1, 1 };
const auto conv2DParams_group_0 = ::testing::Combine(
    ::testing::Values(kernels2D_group_0),
    ::testing::Values(strides2D_group_0),
    ::testing::Values(padBegins2D_group_0),
    ::testing::Values(padEnds2D_group_0),
    ::testing::Values(dilations2D_group_0),
    ::testing::Values(numOutChannels_group_0),
    ::testing::Values(ngraph::op::PadType::NOTSET)
    );
INSTANTIATE_TEST_CASE_P(
    smoke_Convolution2D_group_0, ConvolutionBackpropDataAddExtendedLayerTest,
    ::testing::Combine(
        conv2DParams_group_0,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(input2D_group_0),
        ::testing::Values(output2D_group_0),
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ConvolutionBackpropDataAddExtendedLayerTest::getTestCaseName);

//Attributes: {'auto_pad': 'same_lower', 'dilations': '1,1', 'strides': '2,2'}
//Inputs: {1, 64, 34, 34}, {64, 32, 2, 2}, {2}
//Outputs: {1, 32, 64, 64}
const InferenceEngine::SizeVector input2D_group_1 = { 1, 64, 32, 32 };
const InferenceEngine::SizeVector output2D_group_1 = { 64, 64 };
const size_t numOutChannels_group_1 = 32;
const InferenceEngine::SizeVector kernels2D_group_1 = { 2, 2 };
const InferenceEngine::SizeVector strides2D_group_1 = { 2, 2 };
const std::vector<ptrdiff_t> padBegins2D_group_1 = { 0, 0 };
const std::vector<ptrdiff_t> padEnds2D_group_1 = { 0, 0 };
const InferenceEngine::SizeVector dilations2D_group_1 = { 1, 1 };
const auto conv2DParams_group_1 = ::testing::Combine(
    ::testing::Values(kernels2D_group_1),
    ::testing::Values(strides2D_group_1),
    ::testing::Values(padBegins2D_group_1),
    ::testing::Values(padEnds2D_group_1),
    ::testing::Values(dilations2D_group_1),
    ::testing::Values(numOutChannels_group_1),
    ::testing::Values(ngraph::op::PadType::SAME_LOWER)
    );
INSTANTIATE_TEST_CASE_P(
    smoke_Convolution2D_group_1, ConvolutionBackpropDataAddExtendedLayerTest,
    ::testing::Combine(
        conv2DParams_group_1,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(input2D_group_1),
        ::testing::Values(output2D_group_1),
        ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
        ConvolutionBackpropDataAddExtendedLayerTest::getTestCaseName);

} // namespace
