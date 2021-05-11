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
     * Creates a graph with the following structure:
     * relu( input * multiplier + bias ) + (input * multiplier)
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

        auto relu = std::make_shared<ngraph::opset1::Relu>(add_0);

        auto add_1 = std::make_shared<ngraph::opset1::Add>(relu, multiply);

        ngraph::ParameterVector inputs { input };
        ngraph::NodeVector outputs { add_1 };
        ngraph_function_ = std::make_unique<ngraph::Function>(outputs, inputs,
                "SimpleGraph");
    }

    void TearDown() override {
        ngraph_function_.reset();
    }

protected:
    std::unique_ptr<ngraph::Function> ngraph_function_;
};


TEST_F(OperationBufferExtractorTest, CheckMutableBuffersIndices) {
    using ::testing::ElementsAre;
    auto ordered_nodes = ngraph_function_->get_ordered_ops();
    CUDAPlugin::OperationBuffersExtractor extractor { ordered_nodes };
    auto buffer_indices = extractor.mutableBuffersIndices();
    std::sort(buffer_indices.begin(), buffer_indices.end());
    ASSERT_THAT(buffer_indices, ElementsAre(0, 2, 4, 5, 6));
}


TEST_F(OperationBufferExtractorTest, CheckImmutableBuffersIndices) {
    using ::testing::ElementsAre;
    auto ordered_nodes = ngraph_function_->get_ordered_ops();
    CUDAPlugin::OperationBuffersExtractor extractor { ordered_nodes };
    auto buffer_indices = extractor.immutableBuffersIndices();
    std::sort(buffer_indices.begin(), buffer_indices.end());
    ASSERT_THAT(buffer_indices, ElementsAre(1, 3));
}


TEST_F(OperationBufferExtractorTest, CheckNodeInputsAreValid) {
    using ::testing::ElementsAre;
    auto ordered_nodes = ngraph_function_->get_ordered_ops();
    CUDAPlugin::OperationBuffersExtractor extractor { ordered_nodes };
    ASSERT_TRUE(extractor.inputBufferIndices(*ordered_nodes[0]).empty());
    ASSERT_TRUE(extractor.inputBufferIndices(*ordered_nodes[1]).empty());
    ASSERT_THAT(extractor.inputBufferIndices(*ordered_nodes[2]), ElementsAre(0, 1));
    ASSERT_TRUE(extractor.inputBufferIndices(*ordered_nodes[3]).empty());
    ASSERT_THAT(extractor.inputBufferIndices(*ordered_nodes[4]), ElementsAre(2, 3));
    ASSERT_THAT(extractor.inputBufferIndices(*ordered_nodes[5]), ElementsAre(4));
    ASSERT_THAT(extractor.inputBufferIndices(*ordered_nodes[6]), ElementsAre(5, 2));
    ASSERT_THAT(extractor.inputBufferIndices(*ordered_nodes[7]), ElementsAre(6));
}


TEST_F(OperationBufferExtractorTest, CheckNodeOutputsAreValid) {
    using ::testing::ElementsAre;
    auto ordered_nodes = ngraph_function_->get_ordered_ops();
    CUDAPlugin::OperationBuffersExtractor extractor { ordered_nodes };
    ASSERT_THAT(extractor.outputBufferIndices(*ordered_nodes[0]), ElementsAre(0));
    ASSERT_THAT(extractor.outputBufferIndices(*ordered_nodes[1]), ElementsAre(1));
    ASSERT_THAT(extractor.outputBufferIndices(*ordered_nodes[2]), ElementsAre(2));
    ASSERT_THAT(extractor.outputBufferIndices(*ordered_nodes[3]), ElementsAre(3));
    ASSERT_THAT(extractor.outputBufferIndices(*ordered_nodes[4]), ElementsAre(4));
    ASSERT_THAT(extractor.outputBufferIndices(*ordered_nodes[5]), ElementsAre(5));
    ASSERT_THAT(extractor.outputBufferIndices(*ordered_nodes[6]), ElementsAre(6));
    ASSERT_TRUE(extractor.outputBufferIndices(*ordered_nodes[7]).empty());
}


TEST_F(OperationBufferExtractorTest, CheckMutableBuffersLifespanStart) {
    auto ordered_nodes = ngraph_function_->get_ordered_ops();
    CUDAPlugin::OperationBuffersExtractor extractor { ordered_nodes };
    ASSERT_EQ(extractor.mutableBufferLifespanStart(0), 0);
    ASSERT_EQ(extractor.mutableBufferLifespanStart(2), 2);
    ASSERT_EQ(extractor.mutableBufferLifespanStart(4), 4);
    ASSERT_EQ(extractor.mutableBufferLifespanStart(5), 5);
    ASSERT_EQ(extractor.mutableBufferLifespanStart(6), 6);
}


TEST_F(OperationBufferExtractorTest, CheckMutableBuffersLifespanEnd) {
    using ::testing::ElementsAre;
    auto ordered_nodes = ngraph_function_->get_ordered_ops();
    CUDAPlugin::OperationBuffersExtractor extractor { ordered_nodes };
    ASSERT_EQ(extractor.mutableBufferLifespanEnd(0), 2);
    ASSERT_EQ(extractor.mutableBufferLifespanEnd(2), 6);
    ASSERT_EQ(extractor.mutableBufferLifespanEnd(4), 5);
    ASSERT_EQ(extractor.mutableBufferLifespanEnd(5), 6);
    ASSERT_EQ(extractor.mutableBufferLifespanEnd(6), 7);
}


TEST_F(OperationBufferExtractorTest, CheckMutableBuffersSizes) {
    auto ordered_nodes = ngraph_function_->get_ordered_ops();
    CUDAPlugin::OperationBuffersExtractor extractor { ordered_nodes };
    ASSERT_EQ(extractor.mutableBufferSize(0), 12);
    ASSERT_EQ(extractor.mutableBufferSize(2), 12);
    ASSERT_EQ(extractor.mutableBufferSize(4), 12);
    ASSERT_EQ(extractor.mutableBufferSize(5), 12);
    ASSERT_EQ(extractor.mutableBufferSize(6), 12);
}


TEST_F(OperationBufferExtractorTest, CheckImmutableBuffersSizes) {
    using ::testing::ElementsAre;
    auto ordered_nodes = ngraph_function_->get_ordered_ops();
    CUDAPlugin::OperationBuffersExtractor extractor { ordered_nodes };
    ASSERT_EQ(extractor.immutableBuffer(1).size(), 12);
    ASSERT_EQ(extractor.immutableBuffer(3).size(), 12);
}


TEST_F(OperationBufferExtractorTest, CheckWrongBufferInfexBehavior) {
    auto ordered_nodes = ngraph_function_->get_ordered_ops();
    CUDAPlugin::OperationBuffersExtractor extractor { ordered_nodes };
    EXPECT_THROW(extractor.mutableBufferSize(128), InferenceEngine::details::InferenceEngineException);
    EXPECT_THROW(extractor.mutableBufferLifespanEnd(128), InferenceEngine::details::InferenceEngineException);
    EXPECT_THROW(extractor.mutableBufferLifespanStart(128), InferenceEngine::details::InferenceEngineException);
    EXPECT_THROW(extractor.immutableBuffer(128), InferenceEngine::details::InferenceEngineException);
}


TEST_F(OperationBufferExtractorTest, CheckImmutableBufferContent) {
    using ::testing::ElementsAre;
    using namespace CUDAPlugin;
    auto ordered_nodes = ngraph_function_->get_ordered_ops();
    OperationBuffersExtractor extractor { ordered_nodes };
    auto span_1 = extractor.immutableBuffer(1);
    auto span_3 = extractor.immutableBuffer(3);
    std::vector<float> actual_buffer_1(reinterpret_cast<const float*>(span_1.data()), reinterpret_cast<const float*>(span_1.data()) + 3);
    std::vector<float> actual_buffer_3(reinterpret_cast<const float*>(span_3.data()), reinterpret_cast<const float*>(span_3.data()) + 3);
    ASSERT_THAT(actual_buffer_1, ElementsAre(0.23, 0.23, 0.23));
    ASSERT_THAT(actual_buffer_3, ElementsAre(-0.03, 0.13, 0.65));
}
