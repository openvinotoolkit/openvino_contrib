// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_weights.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"


namespace ngraph {
namespace builder {
namespace subgraph {
static std::shared_ptr<ngraph::Function> FakeQuantizeAndArmGroupConvolutionFunctionGet(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData,
    const FakeQuantizeOnWeights& fqOnWeights,
    const FakeQuantizeOnData& fqOnOutput,
    const bool withRelu) {

    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    const auto fakeQuantizeOnActivations = fqOnData.empty() ?
        nullptr :
        ngraph::builder::makeFakeQuantize(
            input, precision, fqOnData.quantizationLevel, fqOnData.constantShape,
            fqOnData.inputLowValues, fqOnData.inputHighValues, fqOnData.outputLowValues, fqOnData.outputHighValues);

    const size_t inputChannelsCount = inputShape[1];
    const auto weights = ngraph::opset1::Constant::create(
        precision,
        ngraph::Shape{inputChannelsCount, 1, 1, 1 },
        std::vector<float>(inputChannelsCount, 1));

    const auto convolution = std::make_shared<ngraph::opset1::GroupConvolution>(
        fqOnData.empty() ? input : fakeQuantizeOnActivations,
        std::make_shared<ngraph::op::v1::Reshape>(
            fqOnWeights.empty() ? weights->output(0) :
            ngraph::builder::makeFakeQuantize(
                weights, precision, fqOnWeights.quantizationLevel, fqOnWeights.constantShape,
                fqOnWeights.inputLowValues, fqOnWeights.inputHighValues, fqOnWeights.outputLowValues, fqOnWeights.outputHighValues),
            ngraph::opset1::Constant::create(
                ngraph::element::i32, ngraph::Shape{5},
                std::vector<std::int32_t>{static_cast<std::int32_t>(inputChannelsCount), 1, 1, 1, 1}), false),
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });
    convolution->set_friendly_name("convolution");

    std::shared_ptr<ngraph::Node> beforeFqNode = convolution;
    if (withRelu) {
        beforeFqNode = std::make_shared<ngraph::op::Relu>(convolution);
    }

    const auto fakeQuantizeOnOutput = fqOnOutput.empty() ?
        beforeFqNode :
        ngraph::builder::makeFakeQuantize(
            beforeFqNode, precision,
            fqOnOutput.quantizationLevel, fqOnOutput.constantShape,
            fqOnOutput.inputLowValues, fqOnOutput.inputHighValues, fqOnOutput.outputLowValues, fqOnOutput.outputHighValues);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(fakeQuantizeOnOutput) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FakeQuantizeAndArmGroupConvolutionFunction");
}
}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph

namespace LayerTestsDefinitions {

struct ArmGroupConvolutionTransformationParam {
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    bool asymmetricQuantizationOnData;
    ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
    bool asymmetricQuantizationOnWeights;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnOutput;
    std::string layerName;
    std::string expectedKernelType;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    std::string,
    ngraph::pass::low_precision::LayerTransformation::Params,
    ArmGroupConvolutionTransformationParam,
    bool // with Relu
> ArmGroupConvolutionTransformationParams;


class ArmGroupConvolutionTransformation :
    public testing::WithParamInterface<ArmGroupConvolutionTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ArmGroupConvolutionTransformationParams> obj) {
        ngraph::element::Type netPrecision;
        ngraph::Shape inputShape;
        std::string targetDevice;
        ngraph::pass::low_precision::LayerTransformation::Params params;
        ArmGroupConvolutionTransformationParam param;
        bool withRelu;
        std::tie(netPrecision, inputShape, targetDevice, params, param, withRelu) = obj.param;

        std::ostringstream result;
        result << getTestCaseNameByParams(netPrecision, inputShape, targetDevice, params) << "_" <<
            param.fakeQuantizeOnData << "_" <<
            param.fakeQuantizeOnWeights << "_" <<
            param.fakeQuantizeOnOutput << "_" <<
            "withRelu=" << withRelu;
        return result.str();
    }
protected:
    void SetUp() override {
        threshold = 0.1f;

        ngraph::element::Type netPrecision;
        ngraph::Shape inputShape;
        ngraph::pass::low_precision::LayerTransformation::Params params;
        ArmGroupConvolutionTransformationParam param;
        bool withRelu;
        std::tie(netPrecision, inputShape, targetDevice, params, param, withRelu) = this->GetParam();

        function = ngraph::builder::subgraph::FakeQuantizeAndArmGroupConvolutionFunctionGet(
            netPrecision,
            inputShape,
            // TODO: pass from test parameters
            param.fakeQuantizeOnData,
            param.fakeQuantizeOnWeights,
            param.fakeQuantizeOnOutput,
            withRelu);
    }

    void Run() override {
        LayerTestsCommon::Run();

        const auto params = std::get<4>(GetParam());
        const auto actualPrecision = getRuntimePrecisionByType(params.layerName);
        auto expectedPrecision = params.expectedKernelType;
        if (expectedPrecision == "FP32" && std::get<0>(GetParam()) == ngraph::element::f16) {
            expectedPrecision = "FP16";
        }
        EXPECT_EQ(actualPrecision, expectedPrecision);
    }
};

TEST_P(ArmGroupConvolutionTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<ngraph::pass::low_precision::LayerTransformation::Params> transformationParamValues = {
    // LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams().setUpdatePrecisions(true),
    ngraph::pass::low_precision::LayerTransformation::Params(true)
    // LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams().setUpdatePrecisions(false),
};

const std::vector<LayerTestsDefinitions::ArmGroupConvolutionTransformationParam> params = {
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        true,
        { 255ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
        true,
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -128.f }, { 127.f }, { -128.f }, { 127.f } },
        "ArmGroupConvolution",
        "I8"
    },
    {
        { 256ul, ngraph::Shape { 1, 3, 1, 1 }, { 0.f, 0.f, 0.f }, { 255.f, 255.f, 255.f }, { 0.f, 0.f, 0.f }, { 25.5f, 25.5f, 25.5f } },
        true,
        { 255ul, ngraph::Shape { 3, 1, 1, 1 }, { 0.f, 0.f, 0.f}, { 254.f, 254.f, 254.f },
                                               { -12.7f, -12.7f*2, -12.7f*4 },
                                               { 12.7f, 12.7f*2, 12.7f*4 } },
        true,
        { 256ul, ngraph::Shape { 1 }, { -128.f }, { 127.f }, { -12.8f }, { 12.7f }},
        "ArmGroupConvolution",
        "I8"
    },
    {
        { 256ul, ngraph::Shape { 1, 3, 1, 1 }, { 0.f, 0.f, 0.f }, { 255.f, 255.f/2, 255.f/4 }, { -1.27f }, { 1.28 } },
        true,
        { 255ul, ngraph::Shape { 3, 1, 1, 1 }, { 0.f, 0.f, 0.f },
                                               { 254.f, 254.f, 254.f },
                                               { -12.7f, -12.7f*2, -12.7f*4 },
                                               { 12.7f, 12.7f*2, 12.7f*4 } },
        true,
        { 256ul, ngraph::Shape { 1 }, { -128.f }, { 127.f }, { -12.8f }, { 12.7f }},
        "ArmGroupConvolution",
        "I8"
    },
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 3, 2, 2 },
};

using namespace LayerTestsDefinitions;

INSTANTIATE_TEST_CASE_P(smoke_LPT, ArmGroupConvolutionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(shapes),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(transformationParamValues),
        ::testing::ValuesIn(params),
        ::testing::Values(true, false)),
    ArmGroupConvolutionTransformation::getTestCaseName);

}  // namespace
