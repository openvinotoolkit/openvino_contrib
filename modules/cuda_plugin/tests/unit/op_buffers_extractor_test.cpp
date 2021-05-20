// Copyright (C) 2021 Intel Corporation

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

#include <cuda_op_buffers_extractor.hpp>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <details/ie_exception.hpp>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

/*
 * TODO: To be moved to functional tests once they are enabled for CUDAPlugin
 * */

class OperationBufferExtractorTest: public testing::Test {
    /**
     * Creates a graph with the following structure (left to right):
     * ```
     * Parameter           ________________________________________________
     *          \         /                                                \
     *            Multiply                                                  Add -----> Reshape --> Result
     *          /         \                                                /          /
     *   Constant          \                                              /          /
     *    (Multiplier)       Add ----> Unsqueeze --> Relu --> Squeeze -->/    Constant
     *                     /            /                    /
     *                    /            /                    /
     *             Constant     Constant             Constant
     *               (Bias)
     * ```
     */
    void SetUp() override {
        auto input = std::make_shared<ngraph::opset1::Parameter>(
                ngraph::element::f32, ngraph::Shape({ 3 }));

        std::vector<float> multiplier_values = { 0.23, 0.23, 0.23 };
        auto multiplier = std::make_shared<ngraph::opset1::Constant>(
                ngraph::element::f32, ngraph::Shape { 3 }, multiplier_values);

        auto multiply = std::make_shared<ngraph::opset1::Multiply>(input,
                multiplier);

        std::vector<float> bias_values = { -0.03, 0.13, 0.65 };
        auto bias = std::make_shared<ngraph::opset1::Constant>(
                ngraph::element::f32, ngraph::Shape { 3 }, bias_values);

        auto add_0 = std::make_shared<ngraph::opset1::Add>(multiply, bias);

        std::vector<int32_t> unsqueese_axes_values = { 1 };
        auto unsqueese_axes = std::make_shared<ngraph::opset1::Constant>(
                ngraph::element::i32, ngraph::Shape { 1 }, unsqueese_axes_values);
        auto unsqueese = std::make_shared<ngraph::opset1::Unsqueeze>(add_0, unsqueese_axes);

        auto relu = std::make_shared<ngraph::opset1::Relu>(unsqueese);

        std::vector<int32_t> squeese_axes_values = { 1 };
        auto squeese_axes = std::make_shared<ngraph::opset1::Constant>(
                ngraph::element::i32, ngraph::Shape { 1 }, squeese_axes_values);
        auto squeese = std::make_shared<ngraph::opset1::Squeeze>(relu, squeese_axes);

        auto add_1 = std::make_shared<ngraph::opset1::Add>(squeese, multiply);

        std::vector<int32_t> reshape_pattern_values = { 0, 1 };
        auto reshape_pattern = std::make_shared<ngraph::opset1::Constant>(
                ngraph::element::i32, ngraph::Shape { 2 }, reshape_pattern_values);
        auto reshape = std::make_shared<ngraph::opset1::Reshape>(add_1, reshape_pattern, true);

        ngraph::ParameterVector inputs { input };
        ngraph::NodeVector outputs { reshape };
        ngraph_function_ = std::make_unique<ngraph::Function>(outputs, inputs,
                "SimpleGraph");

        exec_sequence_ = ngraph_function_->get_ordered_ops();
        extractor_ = std::make_unique<CUDAPlugin::OperationBuffersExtractor>(exec_sequence_);
    }

protected:
    struct OpIndex {
      using Type = size_t;
      constexpr static Type Parameter = 0;
      constexpr static Type Constant_Multiplier = 1;
      constexpr static Type Multiply = 2;
      constexpr static Type Constant_Bias = 3;
      constexpr static Type Add_Bias = 4;
      constexpr static Type Constant_Unsqueeze_Axes = 5;
      constexpr static Type Unsqueeze = 6;
      constexpr static Type Relu = 7;
      constexpr static Type Constant_Squeeze_Axes = 8;
      constexpr static Type Squeeze = 9;
      constexpr static Type Add_Squeeze_Multiply = 10;
      constexpr static Type Constant_Reshape_Pattern = 11;
      constexpr static Type Reshape = 12;
      constexpr static Type Result = 13;
    };

    struct OutputBufferIndex {
      using Type = unsigned;
      constexpr static Type Parameter = 0;
      constexpr static Type Constant_Multiplier = 1;
      constexpr static Type Multiply = 2;
      constexpr static Type Constant_Bias = 3;
      constexpr static Type Add_Bias = 4;
      constexpr static Type Constant_Unsqueeze_Axes = 5;
      constexpr static Type Relu = 6;
      constexpr static Type Constant_Squeeze_Axes = 7;
      constexpr static Type Add_Squeeze_Multiply = 8;
      constexpr static Type Constant_Reshape_Pattern = 9;
    };

    std::vector<unsigned> inputBufferIndices(OpIndex::Type op_idx) {
        return extractor_->inputBufferIndices(*exec_sequence_.at(op_idx));
    }

    std::vector<unsigned> outputBufferIndices(OpIndex::Type op_idx) {
        return extractor_->outputBufferIndices(*exec_sequence_.at(op_idx));
    }

    template<typename T>
    std::vector<T> immutableBuffer(OutputBufferIndex::Type idx) {
      auto span = extractor_->immutableBuffer(idx);
      const T* begin = reinterpret_cast<const T*>(span.data());
      EXPECT_EQ(span.size() % sizeof(T), 0);
      const size_t size = span.size() / sizeof(T);
      return std::vector<T>(begin, begin + size);
    }

protected:
    std::unique_ptr<ngraph::Function> ngraph_function_;
    std::vector<std::shared_ptr<ngraph::Node>> exec_sequence_;
    std::unique_ptr<CUDAPlugin::OperationBuffersExtractor> extractor_;
};


TEST_F(OperationBufferExtractorTest, CheckTestIntegrity) {
    // auto expect_node = [this]()
    using namespace ngraph;
    EXPECT_TRUE(is_type<opset1::Parameter>(exec_sequence_.at(OpIndex::Parameter)));
    EXPECT_TRUE(is_type<opset1::Constant>(exec_sequence_.at(OpIndex::Constant_Multiplier)));
    EXPECT_TRUE(is_type<opset1::Multiply>(exec_sequence_.at(OpIndex::Multiply)));
    EXPECT_TRUE(is_type<opset1::Constant>(exec_sequence_.at(OpIndex::Constant_Bias)));
    EXPECT_TRUE(is_type<opset1::Add>(exec_sequence_.at(OpIndex::Add_Bias)));
    EXPECT_TRUE(is_type<opset1::Constant>(exec_sequence_.at(OpIndex::Constant_Unsqueeze_Axes)));
    EXPECT_TRUE(is_type<opset1::Unsqueeze>(exec_sequence_.at(OpIndex::Unsqueeze)));
    EXPECT_TRUE(is_type<opset1::Relu>(exec_sequence_.at(OpIndex::Relu)));
    EXPECT_TRUE(is_type<opset1::Constant>(exec_sequence_.at(OpIndex::Constant_Squeeze_Axes)));
    EXPECT_TRUE(is_type<opset1::Squeeze>(exec_sequence_.at(OpIndex::Squeeze)));
    EXPECT_TRUE(is_type<opset1::Add>(exec_sequence_.at(OpIndex::Add_Squeeze_Multiply)));
    EXPECT_TRUE(is_type<opset1::Constant>(exec_sequence_.at(OpIndex::Constant_Reshape_Pattern)));
    EXPECT_TRUE(is_type<opset1::Reshape>(exec_sequence_.at(OpIndex::Reshape)));
    EXPECT_TRUE(is_type<opset1::Result>(exec_sequence_.at(OpIndex::Result)));
}


TEST_F(OperationBufferExtractorTest, CheckMutableBuffersIndices) {
    using ::testing::ElementsAre;
    auto buffer_indices = extractor_->mutableBuffersIndices();
    std::sort(buffer_indices.begin(), buffer_indices.end());
    ASSERT_THAT(buffer_indices, ElementsAre(
        OutputBufferIndex::Parameter,
        OutputBufferIndex::Multiply,
        OutputBufferIndex::Add_Bias,
        OutputBufferIndex::Relu,
        OutputBufferIndex::Add_Squeeze_Multiply));
}


TEST_F(OperationBufferExtractorTest, CheckImmutableBuffersIndices) {
    using ::testing::ElementsAre;
    auto buffer_indices = extractor_->immutableBuffersIndices();
    std::sort(buffer_indices.begin(), buffer_indices.end());
    ASSERT_THAT(buffer_indices, ElementsAre(
        OutputBufferIndex::Constant_Multiplier,
        OutputBufferIndex::Constant_Bias,
        OutputBufferIndex::Constant_Unsqueeze_Axes,
        OutputBufferIndex::Constant_Squeeze_Axes,
        OutputBufferIndex::Constant_Reshape_Pattern));
}


TEST_F(OperationBufferExtractorTest, CheckNodeInputsAreValid) {
    using ::testing::ElementsAre;

    ASSERT_TRUE(inputBufferIndices(OpIndex::Parameter).empty());
    ASSERT_TRUE(inputBufferIndices(OpIndex::Constant_Multiplier).empty());
    ASSERT_THAT(inputBufferIndices(OpIndex::Multiply),
                ElementsAre(OutputBufferIndex::Parameter, OutputBufferIndex::Constant_Multiplier));
    ASSERT_TRUE(inputBufferIndices(OpIndex::Constant_Bias).empty());
    ASSERT_THAT(inputBufferIndices(OpIndex::Add_Bias),
                ElementsAre(OutputBufferIndex::Multiply, OutputBufferIndex::Constant_Bias));
    ASSERT_TRUE(inputBufferIndices(OpIndex::Constant_Unsqueeze_Axes).empty());
    ASSERT_THAT(inputBufferIndices(OpIndex::Unsqueeze),
                ElementsAre(OutputBufferIndex::Add_Bias, OutputBufferIndex::Constant_Unsqueeze_Axes));
    ASSERT_THAT(inputBufferIndices(OpIndex::Relu),
                ElementsAre(OutputBufferIndex::Add_Bias));
    ASSERT_TRUE(inputBufferIndices(OpIndex::Constant_Squeeze_Axes).empty());
    ASSERT_THAT(inputBufferIndices(OpIndex::Squeeze),
                ElementsAre(OutputBufferIndex::Relu, OutputBufferIndex::Constant_Squeeze_Axes));
    ASSERT_THAT(inputBufferIndices(OpIndex::Add_Squeeze_Multiply),
                ElementsAre(OutputBufferIndex::Relu, OutputBufferIndex::Multiply));
    ASSERT_TRUE(inputBufferIndices(OpIndex::Constant_Reshape_Pattern).empty());
    ASSERT_THAT(inputBufferIndices(OpIndex::Reshape),
                ElementsAre(OutputBufferIndex::Add_Squeeze_Multiply, OutputBufferIndex::Constant_Reshape_Pattern));
    ASSERT_THAT(inputBufferIndices(OpIndex::Result),
                ElementsAre(OutputBufferIndex::Add_Squeeze_Multiply));
}


TEST_F(OperationBufferExtractorTest, CheckNodeOutputsAreValid) {
    using ::testing::ElementsAre;

    ASSERT_THAT(outputBufferIndices(OpIndex::Parameter), ElementsAre(OutputBufferIndex::Parameter));
    ASSERT_THAT(outputBufferIndices(OpIndex::Constant_Multiplier), ElementsAre(OutputBufferIndex::Constant_Multiplier));
    ASSERT_THAT(outputBufferIndices(OpIndex::Multiply), ElementsAre(OutputBufferIndex::Multiply));
    ASSERT_THAT(outputBufferIndices(OpIndex::Constant_Bias), ElementsAre(OutputBufferIndex::Constant_Bias));
    ASSERT_THAT(outputBufferIndices(OpIndex::Add_Bias), ElementsAre(OutputBufferIndex::Add_Bias));
    ASSERT_THAT(outputBufferIndices(OpIndex::Constant_Unsqueeze_Axes), ElementsAre(OutputBufferIndex::Constant_Unsqueeze_Axes));
    ASSERT_THAT(outputBufferIndices(OpIndex::Unsqueeze), ElementsAre(OutputBufferIndex::Add_Bias));
    ASSERT_THAT(outputBufferIndices(OpIndex::Relu), ElementsAre(OutputBufferIndex::Relu));
    ASSERT_THAT(outputBufferIndices(OpIndex::Constant_Squeeze_Axes), ElementsAre(OutputBufferIndex::Constant_Squeeze_Axes));
    ASSERT_THAT(outputBufferIndices(OpIndex::Squeeze), ElementsAre(OutputBufferIndex::Relu));
    ASSERT_THAT(outputBufferIndices(OpIndex::Add_Squeeze_Multiply), ElementsAre(OutputBufferIndex::Add_Squeeze_Multiply));
    ASSERT_THAT(outputBufferIndices(OpIndex::Constant_Reshape_Pattern), ElementsAre(OutputBufferIndex::Constant_Reshape_Pattern));
    ASSERT_THAT(outputBufferIndices(OpIndex::Reshape), ElementsAre(OutputBufferIndex::Add_Squeeze_Multiply));
    ASSERT_TRUE(outputBufferIndices(OpIndex::Result).empty());
}


TEST_F(OperationBufferExtractorTest, CheckMutableBuffersLifespanStart) {
    auto lifespanStart = [this](OutputBufferIndex::Type idx) {
      return extractor_->mutableBufferLifespanStart(idx);
    };

    ASSERT_EQ(lifespanStart(OutputBufferIndex::Parameter), OpIndex::Parameter);
    ASSERT_EQ(lifespanStart(OutputBufferIndex::Multiply), OpIndex::Multiply);
    ASSERT_EQ(lifespanStart(OutputBufferIndex::Add_Bias), OpIndex::Add_Bias);
    ASSERT_EQ(lifespanStart(OutputBufferIndex::Relu), OpIndex::Relu);
    ASSERT_EQ(lifespanStart(OutputBufferIndex::Add_Squeeze_Multiply), OpIndex::Add_Squeeze_Multiply);
}


TEST_F(OperationBufferExtractorTest, CheckMutableBuffersLifespanEnd) {
    auto lifespanEnd = [this](OutputBufferIndex::Type idx) {
      return extractor_->mutableBufferLifespanEnd(idx);
    };

    ASSERT_EQ(lifespanEnd(OutputBufferIndex::Parameter), OpIndex::Multiply);
    ASSERT_EQ(lifespanEnd(OutputBufferIndex::Multiply), OpIndex::Add_Squeeze_Multiply);
    ASSERT_EQ(lifespanEnd(OutputBufferIndex::Add_Bias), OpIndex::Relu);
    ASSERT_EQ(lifespanEnd(OutputBufferIndex::Relu), OpIndex::Add_Squeeze_Multiply);
    ASSERT_EQ(lifespanEnd(OutputBufferIndex::Add_Squeeze_Multiply), OpIndex::Result);
}


TEST_F(OperationBufferExtractorTest, CheckMutableBuffersSizes) {
    ASSERT_EQ(extractor_->mutableBufferSize(OutputBufferIndex::Parameter), 12);
    ASSERT_EQ(extractor_->mutableBufferSize(OutputBufferIndex::Multiply), 12);
    ASSERT_EQ(extractor_->mutableBufferSize(OutputBufferIndex::Add_Bias), 12);
    ASSERT_EQ(extractor_->mutableBufferSize(OutputBufferIndex::Relu), 12);
    ASSERT_EQ(extractor_->mutableBufferSize(OutputBufferIndex::Add_Squeeze_Multiply), 12);
}


TEST_F(OperationBufferExtractorTest, CheckImmutableBuffersSizes) {
    ASSERT_EQ(extractor_->immutableBuffer(OutputBufferIndex::Constant_Multiplier).size(), 12);
    ASSERT_EQ(extractor_->immutableBuffer(OutputBufferIndex::Constant_Bias).size(), 12);
    ASSERT_EQ(extractor_->immutableBuffer(OutputBufferIndex::Constant_Unsqueeze_Axes).size(), 4);
    ASSERT_EQ(extractor_->immutableBuffer(OutputBufferIndex::Constant_Squeeze_Axes).size(), 4);
    ASSERT_EQ(extractor_->immutableBuffer(OutputBufferIndex::Constant_Reshape_Pattern).size(), 8);
}


TEST_F(OperationBufferExtractorTest, CheckWrongBufferInfexBehavior) {
    EXPECT_THROW(extractor_->mutableBufferSize(128), InferenceEngine::details::InferenceEngineException);
    EXPECT_THROW(extractor_->mutableBufferLifespanEnd(128), InferenceEngine::details::InferenceEngineException);
    EXPECT_THROW(extractor_->mutableBufferLifespanStart(128), InferenceEngine::details::InferenceEngineException);
    EXPECT_THROW(extractor_->immutableBuffer(128), InferenceEngine::details::InferenceEngineException);
}


TEST_F(OperationBufferExtractorTest, CheckImmutableBufferContent) {
    using ::testing::ElementsAre;
    EXPECT_THAT(immutableBuffer<float>(OutputBufferIndex::Constant_Multiplier), ElementsAre(0.23, 0.23, 0.23));
    EXPECT_THAT(immutableBuffer<float>(OutputBufferIndex::Constant_Bias), ElementsAre(-0.03, 0.13, 0.65));
    EXPECT_THAT(immutableBuffer<int32_t>(OutputBufferIndex::Constant_Unsqueeze_Axes), ElementsAre(1));
    EXPECT_THAT(immutableBuffer<int32_t>(OutputBufferIndex::Constant_Squeeze_Axes), ElementsAre(1));
    EXPECT_THAT(immutableBuffer<int32_t>(OutputBufferIndex::Constant_Reshape_Pattern), ElementsAre(0, 1));
}


TEST_F(OperationBufferExtractorTest, CheckSameInputOutputForReshapeOnlyOps) {
    EXPECT_EQ(inputBufferIndices(OpIndex::Unsqueeze).at(0), outputBufferIndices(OpIndex::Unsqueeze).at(0));
    EXPECT_EQ(inputBufferIndices(OpIndex::Squeeze).at(0), outputBufferIndices(OpIndex::Squeeze).at(0));
    EXPECT_EQ(inputBufferIndices(OpIndex::Reshape).at(0), outputBufferIndices(OpIndex::Reshape).at(0));
}
