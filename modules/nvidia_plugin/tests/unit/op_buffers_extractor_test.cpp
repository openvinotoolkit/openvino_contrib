// Copyright (C) 2021-2023 Intel Corporation

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

#include "cuda_op_buffers_extractor.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "transformer/nodes/concat_optimized.hpp"

/*
 * TODO: To be moved to functional tests once they are enabled for nvidia_gpu
 * */

class OperationBufferExtractorTest : public testing::Test {
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
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape({3}));

        std::vector<float> multiplier_values = {0.23, 0.23, 0.23};
        auto multiplier = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{3}, multiplier_values);

        auto multiply = std::make_shared<ov::op::v1::Multiply>(input, multiplier);

        std::vector<float> bias_values = {-0.03, 0.13, 0.65};
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{3}, bias_values);

        auto add_0 = std::make_shared<ov::op::v1::Add>(multiply, bias);

        std::vector<int32_t> unsqueeze_axes_values = {1};
        auto unsqueeze_axes =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, unsqueeze_axes_values);
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(add_0, unsqueeze_axes);

        auto relu = std::make_shared<ov::op::v0::Relu>(unsqueeze);

        std::vector<int32_t> squeeze_axes_values = {1};
        auto squeeze_axes = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, squeeze_axes_values);
        auto squeeze = std::make_shared<ov::op::v0::Squeeze>(relu, squeeze_axes);
        auto squeeze_id = squeeze->get_instance_id();

        auto add_1 = std::make_shared<ov::op::v1::Add>(squeeze, multiply);

        std::vector<int32_t> reshape_pattern_values = {0, 1};
        auto reshape_pattern =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2}, reshape_pattern_values);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(add_1, reshape_pattern, true);

        ov::ParameterVector inputs{input};
        ov::OutputVector outputs{reshape};
        model_ = std::make_unique<ov::Model>(outputs, inputs, "SimpleGraph");

        exec_sequence_ = model_->get_ordered_ops();
        extractor_ = std::make_unique<ov::nvidia_gpu::OperationBuffersExtractor>(exec_sequence_);
        ov::nvidia_gpu::WorkbufferRequest request{{128}, {256}};
        buffer_indices_ = extractor_->processWorkbufferRequest(squeeze_id, request);
    }

protected:
    using TensorID = ov::nvidia_gpu::TensorID;

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
        constexpr static Type Squeeze_Imutable_Workbuffer = 10;
        constexpr static Type Squeeze_Mutable_Workbuffer = 11;
    };

    std::vector<TensorID> inputBufferIndices(OpIndex::Type op_idx) {
        return extractor_->inputTensorIds(*exec_sequence_.at(op_idx));
    }

    std::vector<TensorID> outputBufferIndices(OpIndex::Type op_idx) {
        return extractor_->outputTensorIds(*exec_sequence_.at(op_idx));
    }

    template <typename T>
    std::vector<T> immutableBuffer(OutputBufferIndex::Type idx) {
        auto span = extractor_->immutableBuffer(idx);
        const T* begin = reinterpret_cast<const T*>(span.data());
        EXPECT_EQ(span.size() % sizeof(T), 0);
        const size_t size = span.size() / sizeof(T);
        return std::vector<T>(begin, begin + size);
    }

protected:
    std::unique_ptr<ov::Model> model_;
    std::vector<std::shared_ptr<ov::Node>> exec_sequence_;
    std::unique_ptr<ov::nvidia_gpu::OperationBuffersExtractor> extractor_;
    ov::nvidia_gpu::WorkbufferIds buffer_indices_;
};

TEST_F(OperationBufferExtractorTest, CheckTestIntegrity) {
    // auto expect_node = [this]()
    EXPECT_TRUE(ov::is_type<ov::op::v0::Parameter>(exec_sequence_.at(OpIndex::Parameter)));
    EXPECT_TRUE(ov::is_type<ov::op::v0::Constant>(exec_sequence_.at(OpIndex::Constant_Multiplier)));
    EXPECT_TRUE(ov::is_type<ov::op::v1::Multiply>(exec_sequence_.at(OpIndex::Multiply)));
    EXPECT_TRUE(ov::is_type<ov::op::v0::Constant>(exec_sequence_.at(OpIndex::Constant_Bias)));
    EXPECT_TRUE(ov::is_type<ov::op::v1::Add>(exec_sequence_.at(OpIndex::Add_Bias)));
    EXPECT_TRUE(ov::is_type<ov::op::v0::Constant>(exec_sequence_.at(OpIndex::Constant_Unsqueeze_Axes)));
    EXPECT_TRUE(ov::is_type<ov::op::v0::Unsqueeze>(exec_sequence_.at(OpIndex::Unsqueeze)));
    EXPECT_TRUE(ov::is_type<ov::op::v0::Relu>(exec_sequence_.at(OpIndex::Relu)));
    EXPECT_TRUE(ov::is_type<ov::op::v0::Constant>(exec_sequence_.at(OpIndex::Constant_Squeeze_Axes)));
    EXPECT_TRUE(ov::is_type<ov::op::v0::Squeeze>(exec_sequence_.at(OpIndex::Squeeze)));
    EXPECT_TRUE(ov::is_type<ov::op::v1::Add>(exec_sequence_.at(OpIndex::Add_Squeeze_Multiply)));
    EXPECT_TRUE(ov::is_type<ov::op::v0::Constant>(exec_sequence_.at(OpIndex::Constant_Reshape_Pattern)));
    EXPECT_TRUE(ov::is_type<ov::op::v1::Reshape>(exec_sequence_.at(OpIndex::Reshape)));
    EXPECT_TRUE(ov::is_type<ov::op::v0::Result>(exec_sequence_.at(OpIndex::Result)));
}

TEST_F(OperationBufferExtractorTest, CheckMutableBuffersIndices) {
    using ::testing::ElementsAre;
    auto buffer_indices = extractor_->mutableBuffersIds();
    std::sort(buffer_indices.begin(), buffer_indices.end());
    ASSERT_THAT(buffer_indices,
                ElementsAre(OutputBufferIndex::Parameter,
                            OutputBufferIndex::Multiply,
                            OutputBufferIndex::Add_Bias,
                            OutputBufferIndex::Relu,
                            OutputBufferIndex::Add_Squeeze_Multiply,
                            OutputBufferIndex::Squeeze_Mutable_Workbuffer));
}

TEST_F(OperationBufferExtractorTest, CheckImmutableBuffersIndices) {
    using ::testing::ElementsAre;
    auto buffer_indices = extractor_->immutableBuffersIds();
    std::sort(buffer_indices.begin(), buffer_indices.end());
    ASSERT_THAT(buffer_indices,
                ElementsAre(OutputBufferIndex::Constant_Multiplier,
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
                ElementsAre(TensorID{OutputBufferIndex::Parameter}, TensorID{OutputBufferIndex::Constant_Multiplier}));
    ASSERT_TRUE(inputBufferIndices(OpIndex::Constant_Bias).empty());
    ASSERT_THAT(inputBufferIndices(OpIndex::Add_Bias),
                ElementsAre(TensorID{OutputBufferIndex::Multiply}, TensorID{OutputBufferIndex::Constant_Bias}));
    ASSERT_TRUE(inputBufferIndices(OpIndex::Constant_Unsqueeze_Axes).empty());
    ASSERT_THAT(
        inputBufferIndices(OpIndex::Unsqueeze),
        ElementsAre(TensorID{OutputBufferIndex::Add_Bias}, TensorID{OutputBufferIndex::Constant_Unsqueeze_Axes}));
    ASSERT_THAT(inputBufferIndices(OpIndex::Relu), ElementsAre(TensorID{OutputBufferIndex::Add_Bias}));
    ASSERT_TRUE(inputBufferIndices(OpIndex::Constant_Squeeze_Axes).empty());
    ASSERT_THAT(inputBufferIndices(OpIndex::Squeeze),
                ElementsAre(TensorID{OutputBufferIndex::Relu}, TensorID{OutputBufferIndex::Constant_Squeeze_Axes}));
    ASSERT_THAT(inputBufferIndices(OpIndex::Add_Squeeze_Multiply),
                ElementsAre(TensorID{OutputBufferIndex::Relu}, TensorID{OutputBufferIndex::Multiply}));
    ASSERT_TRUE(inputBufferIndices(OpIndex::Constant_Reshape_Pattern).empty());
    ASSERT_THAT(inputBufferIndices(OpIndex::Reshape),
                ElementsAre(TensorID{OutputBufferIndex::Add_Squeeze_Multiply},
                            TensorID{OutputBufferIndex::Constant_Reshape_Pattern}));
    ASSERT_THAT(inputBufferIndices(OpIndex::Result), ElementsAre(TensorID{OutputBufferIndex::Add_Squeeze_Multiply}));
}

TEST_F(OperationBufferExtractorTest, CheckNodeOutputsAreValid) {
    using ::testing::ElementsAre;

    ASSERT_THAT(outputBufferIndices(OpIndex::Parameter), ElementsAre(TensorID{OutputBufferIndex::Parameter}));
    ASSERT_THAT(outputBufferIndices(OpIndex::Constant_Multiplier),
                ElementsAre(TensorID{OutputBufferIndex::Constant_Multiplier}));
    ASSERT_THAT(outputBufferIndices(OpIndex::Multiply), ElementsAre(TensorID{OutputBufferIndex::Multiply}));
    ASSERT_THAT(outputBufferIndices(OpIndex::Constant_Bias), ElementsAre(TensorID{OutputBufferIndex::Constant_Bias}));
    ASSERT_THAT(outputBufferIndices(OpIndex::Add_Bias), ElementsAre(TensorID{OutputBufferIndex::Add_Bias}));
    ASSERT_THAT(outputBufferIndices(OpIndex::Constant_Unsqueeze_Axes),
                ElementsAre(TensorID{OutputBufferIndex::Constant_Unsqueeze_Axes}));
    ASSERT_THAT(outputBufferIndices(OpIndex::Unsqueeze), ElementsAre(TensorID{OutputBufferIndex::Add_Bias}));
    ASSERT_THAT(outputBufferIndices(OpIndex::Relu), ElementsAre(TensorID{OutputBufferIndex::Relu}));
    ASSERT_THAT(outputBufferIndices(OpIndex::Constant_Squeeze_Axes),
                ElementsAre(TensorID{OutputBufferIndex::Constant_Squeeze_Axes}));
    ASSERT_THAT(outputBufferIndices(OpIndex::Squeeze), ElementsAre(TensorID{OutputBufferIndex::Relu}));
    ASSERT_THAT(outputBufferIndices(OpIndex::Add_Squeeze_Multiply),
                ElementsAre(TensorID{OutputBufferIndex::Add_Squeeze_Multiply}));
    ASSERT_THAT(outputBufferIndices(OpIndex::Constant_Reshape_Pattern),
                ElementsAre(TensorID{OutputBufferIndex::Constant_Reshape_Pattern}));
    ASSERT_THAT(outputBufferIndices(OpIndex::Reshape), ElementsAre(TensorID{OutputBufferIndex::Add_Squeeze_Multiply}));
    ASSERT_TRUE(outputBufferIndices(OpIndex::Result).empty());
}

TEST_F(OperationBufferExtractorTest, CheckMutableBuffersLifespanStart) {
    auto lifespanStart = [this](OutputBufferIndex::Type idx) { return extractor_->mutableBufferLifespanStart(idx); };

    ASSERT_EQ(lifespanStart(OutputBufferIndex::Parameter), OpIndex::Parameter);
    ASSERT_EQ(lifespanStart(OutputBufferIndex::Multiply), OpIndex::Multiply);
    ASSERT_EQ(lifespanStart(OutputBufferIndex::Add_Bias), OpIndex::Add_Bias);
    ASSERT_EQ(lifespanStart(OutputBufferIndex::Relu), OpIndex::Relu);
    ASSERT_EQ(lifespanStart(OutputBufferIndex::Add_Squeeze_Multiply), OpIndex::Add_Squeeze_Multiply);
}

TEST_F(OperationBufferExtractorTest, CheckMutableBuffersLifespanEnd) {
    auto lifespanEnd = [this](OutputBufferIndex::Type idx) { return extractor_->mutableBufferLifespanEnd(idx); };

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
    EXPECT_THROW(extractor_->mutableBufferSize(128), ov::Exception);
    EXPECT_THROW(extractor_->mutableBufferLifespanEnd(128), ov::Exception);
    EXPECT_THROW(extractor_->mutableBufferLifespanStart(128), ov::Exception);
    EXPECT_THROW(extractor_->immutableBuffer(128), ov::Exception);
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

TEST_F(OperationBufferExtractorTest, CheckMutableWorkbufferIndices) {
    using ::testing::ElementsAre;
    ASSERT_THAT(buffer_indices_.mutableIds, ElementsAre(OutputBufferIndex::Squeeze_Mutable_Workbuffer));
}

TEST_F(OperationBufferExtractorTest, CheckImmutableWorkbufferIndices) {
    using ov::nvidia_gpu::WorkbufferRequest;
    using ::testing::ElementsAre;
    WorkbufferRequest request{{246}, {}};
    ASSERT_THAT(buffer_indices_.immutableIds, ElementsAre(OutputBufferIndex::Squeeze_Imutable_Workbuffer));
}

class OperationBufferExtractorConcatOptimizedTest : public testing::Test {
    /**
     * Creates a graph with the following structure (left to right):
     * ```
     * Parameter           ___________
     *          \         /           \
     *            Multiply --> Add --> ConcatOptimized --> Add --> Reshape --> Result
     *          /             /                           /
     *   Constant         Constant                      Constant
     *    (Multiplier)     (Adder_0)                     (Adder_1)
     *
     * ```
     */
    void SetUp() override {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape({1, 8, 16, 16}));

        std::vector<float> multiplier_values = {0.23};
        auto multiplier = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, multiplier_values);

        auto multiply = std::make_shared<ov::op::v1::Multiply>(input, multiplier);

        std::vector<float> adder_0_values = {0.33};
        auto adder_0 =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 8, 16, 16}, adder_0_values);
        auto add_0 = std::make_shared<ov::op::v1::Add>(multiplier, adder_0);

        auto concat = std::make_shared<ov::nvidia_gpu::nodes::ConcatOptimized>(ov::OutputVector{multiply, add_0}, 1);

        std::vector<float> adder_1_values = {0.55};
        auto adder_1 =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 16, 16, 16}, adder_1_values);
        auto add_1 = std::make_shared<ov::op::v1::Add>(adder_1, concat);

        std::vector<int32_t> reshape_pattern_values = {0, 1};
        auto reshape_pattern =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2}, reshape_pattern_values);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(add_1, reshape_pattern, true);

        ov::ParameterVector inputs{input};
        ov::OutputVector outputs{reshape};
        model_ = std::make_unique<ov::Model>(outputs, inputs, "ConcatOptimizedGraph");

        exec_sequence_ = model_->get_ordered_ops();
        extractor_ = std::make_unique<ov::nvidia_gpu::OperationBuffersExtractor>(exec_sequence_);
    }

protected:
    using TensorID = ov::nvidia_gpu::TensorID;

    struct OpIndex {
        using Type = size_t;
        constexpr static Type Parameter = 0;
        constexpr static Type Constant_Multiplier = 1;
        constexpr static Type Constant_Adder_0 = 2;
        constexpr static Type Multiply = 3;
        constexpr static Type Constant_Adder_1 = 4;
        constexpr static Type Add_0 = 5;
        constexpr static Type ConcatOptimized = 6;
        constexpr static Type Add_1 = 7;
        constexpr static Type Constant_Reshape_Pattern = 8;
        constexpr static Type Reshape = 9;
        constexpr static Type Result = 10;
    };

    struct OutputBufferIndex {
        using Type = unsigned;
        constexpr static Type Parameter = 0;
        constexpr static Type Constant_Multiplier = 1;
        constexpr static Type Constant_Adder_0 = 2;
        constexpr static Type Constant_Adder_1 = 4;
        constexpr static Type ConcatOptimized = 6;
        constexpr static Type Add_1 = 7;
        constexpr static Type Constant_Reshape_Pattern = 8;
    };

    std::vector<TensorID> inputBufferIndices(OpIndex::Type op_idx) {
        return extractor_->inputTensorIds(*exec_sequence_.at(op_idx));
    }

    std::vector<TensorID> outputBufferIndices(OpIndex::Type op_idx) {
        return extractor_->outputTensorIds(*exec_sequence_.at(op_idx));
    }

    template <typename T>
    std::vector<T> immutableBuffer(OutputBufferIndex::Type idx) {
        auto span = extractor_->immutableBuffer(idx);
        const T* begin = reinterpret_cast<const T*>(span.data());
        EXPECT_EQ(span.size() % sizeof(T), 0);
        const size_t size = span.size() / sizeof(T);
        return std::vector<T>(begin, begin + size);
    }

protected:
    std::unique_ptr<ov::Model> model_;
    std::vector<std::shared_ptr<ov::Node>> exec_sequence_;
    std::unique_ptr<ov::nvidia_gpu::OperationBuffersExtractor> extractor_;
};

TEST_F(OperationBufferExtractorConcatOptimizedTest, CheckTestIntegrity) {
    EXPECT_TRUE(ov::is_type<ov::op::v0::Parameter>(exec_sequence_.at(OpIndex::Parameter)));
    EXPECT_TRUE(ov::is_type<ov::op::v0::Constant>(exec_sequence_.at(OpIndex::Constant_Multiplier)));
    EXPECT_TRUE(ov::is_type<ov::op::v0::Constant>(exec_sequence_.at(OpIndex::Constant_Adder_0)));
    EXPECT_TRUE(ov::is_type<ov::op::v1::Multiply>(exec_sequence_.at(OpIndex::Multiply)));
    EXPECT_TRUE(ov::is_type<ov::op::v0::Constant>(exec_sequence_.at(OpIndex::Constant_Adder_1)));
    EXPECT_TRUE(ov::is_type<ov::op::v1::Add>(exec_sequence_.at(OpIndex::Add_0)));
    EXPECT_TRUE(ov::is_type<ov::nvidia_gpu::nodes::ConcatOptimized>(exec_sequence_.at(OpIndex::ConcatOptimized)));
    EXPECT_TRUE(ov::is_type<ov::op::v1::Add>(exec_sequence_.at(OpIndex::Add_1)));
    EXPECT_TRUE(ov::is_type<ov::op::v0::Constant>(exec_sequence_.at(OpIndex::Constant_Reshape_Pattern)));
    EXPECT_TRUE(ov::is_type<ov::op::v1::Reshape>(exec_sequence_.at(OpIndex::Reshape)));
    EXPECT_TRUE(ov::is_type<ov::op::v0::Result>(exec_sequence_.at(OpIndex::Result)));
}

TEST_F(OperationBufferExtractorConcatOptimizedTest, CheckMutableBuffersIndices) {
    using ::testing::ElementsAre;
    auto buffer_indices = extractor_->mutableBuffersIds();
    std::sort(buffer_indices.begin(), buffer_indices.end());
    ASSERT_THAT(
        buffer_indices,
        ElementsAre(OutputBufferIndex::Parameter, OutputBufferIndex::ConcatOptimized, OutputBufferIndex::Add_1));
}

TEST_F(OperationBufferExtractorConcatOptimizedTest, CheckImmutableBuffersIndices) {
    using ::testing::ElementsAre;
    auto immutable_buffer_indices = extractor_->immutableBuffersIds();
    std::sort(immutable_buffer_indices.begin(), immutable_buffer_indices.end());
    ASSERT_THAT(immutable_buffer_indices,
                ElementsAre(OutputBufferIndex::Constant_Multiplier,
                            OutputBufferIndex::Constant_Adder_0,
                            OutputBufferIndex::Constant_Adder_1,
                            OutputBufferIndex::Constant_Reshape_Pattern));
}

class OperationBufferExtractorConcatOptimizedV2Test : public testing::Test {
    /**
     * Creates a graph with the following structure (left to right):
     * ```
     * Parameter           --> Reshape -->
     *          \         /               \
     *            Multiply --> Add --> ConcatOptimized --> Add --> Reshape --> Result
     *          /             /                           /
     *   Constant         Constant                      Constant
     *    (Multiplier)     (Adder_0)                     (Adder_1)
     *
     * ```
     */
    void SetUp() override {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape({1, 8, 16, 16}));

        std::vector<float> multiplier_values = {0.23};
        auto multiplier = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, multiplier_values);

        auto multiply = std::make_shared<ov::op::v1::Multiply>(input, multiplier);

        std::vector<float> adder_0_values = {0.33};
        auto adder_0 =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 8, 16, 16}, adder_0_values);
        auto add_0 = std::make_shared<ov::op::v1::Add>(multiplier, adder_0);

        auto reshape_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{1, 8, 16, 16});
        auto reshape0 = std::make_shared<ov::op::v1::Reshape>(multiply, reshape_const, true);

        auto concat = std::make_shared<ov::nvidia_gpu::nodes::ConcatOptimized>(ov::OutputVector{reshape0, add_0}, 1);

        std::vector<float> adder_1_values = {0.55};
        auto adder_1 =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 16, 16, 16}, adder_1_values);
        auto add_1 = std::make_shared<ov::op::v1::Add>(adder_1, concat);

        std::vector<int32_t> reshape_pattern_values = {0, 1};
        auto reshape_pattern =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2}, reshape_pattern_values);
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(add_1, reshape_pattern, true);

        ov::ParameterVector inputs{input};
        ov::OutputVector outputs{reshape1};
        model_ = std::make_unique<ov::Model>(outputs, inputs, "ConcatOptimizedGraph");

        exec_sequence_ = model_->get_ordered_ops();
        extractor_ = std::make_unique<ov::nvidia_gpu::OperationBuffersExtractor>(exec_sequence_);
    }

protected:
    using TensorID = ov::nvidia_gpu::TensorID;

    struct OpIndex {
        using Type = size_t;
        constexpr static Type Parameter = 0;
        constexpr static Type Constant_Multiplier = 1;
        constexpr static Type Constant_Adder_0 = 2;
        constexpr static Type Multiply = 3;
        constexpr static Type Constant_Reshape_0 = 4;
        constexpr static Type Reshape_0 = 5;
        constexpr static Type Constant_Adder_1 = 6;
        constexpr static Type Add_0 = 7;
        constexpr static Type ConcatOptimized = 8;
        constexpr static Type Add_1 = 9;
        constexpr static Type Constant_Reshape_Pattern = 10;
        constexpr static Type Reshape1 = 11;
        constexpr static Type Result = 12;
    };

    struct OutputBufferIndex {
        using Type = unsigned;
        constexpr static Type Parameter = 0;
        constexpr static Type Constant_Multiplier = 1;
        constexpr static Type Constant_Adder_0 = 2;
        constexpr static Type Constant_Reshape_0_Pattern = 4;
        constexpr static Type Constant_Adder_1 = 5;
        constexpr static Type ConcatOptimized = 7;
        constexpr static Type Add_1 = 8;
        constexpr static Type Constant_Reshape_1_Pattern = 9;
    };

    std::vector<TensorID> inputBufferIndices(OpIndex::Type op_idx) {
        return extractor_->inputTensorIds(*exec_sequence_.at(op_idx));
    }

    std::vector<TensorID> outputBufferIndices(OpIndex::Type op_idx) {
        return extractor_->outputTensorIds(*exec_sequence_.at(op_idx));
    }

    template <typename T>
    std::vector<T> immutableBuffer(OutputBufferIndex::Type idx) {
        auto span = extractor_->immutableBuffer(idx);
        const T* begin = reinterpret_cast<const T*>(span.data());
        EXPECT_EQ(span.size() % sizeof(T), 0);
        const size_t size = span.size() / sizeof(T);
        return std::vector<T>(begin, begin + size);
    }

protected:
    std::unique_ptr<ov::Model> model_;
    std::vector<std::shared_ptr<ov::Node>> exec_sequence_;
    std::unique_ptr<ov::nvidia_gpu::OperationBuffersExtractor> extractor_;
};

TEST_F(OperationBufferExtractorConcatOptimizedV2Test, CheckTestIntegrity) {
    EXPECT_TRUE(ov::is_type<ov::op::v0::Parameter>(exec_sequence_.at(OpIndex::Parameter)));
    EXPECT_TRUE(ov::is_type<ov::op::v0::Constant>(exec_sequence_.at(OpIndex::Constant_Multiplier)));
    EXPECT_TRUE(ov::is_type<ov::op::v0::Constant>(exec_sequence_.at(OpIndex::Constant_Adder_0)));
    EXPECT_TRUE(ov::is_type<ov::op::v1::Multiply>(exec_sequence_.at(OpIndex::Multiply)));
    EXPECT_TRUE(ov::is_type<ov::op::v0::Constant>(exec_sequence_.at(OpIndex::Constant_Reshape_0)));
    EXPECT_TRUE(ov::is_type<ov::op::v1::Reshape>(exec_sequence_.at(OpIndex::Reshape_0)));
    EXPECT_TRUE(ov::is_type<ov::op::v0::Constant>(exec_sequence_.at(OpIndex::Constant_Adder_1)));
    EXPECT_TRUE(ov::is_type<ov::op::v1::Add>(exec_sequence_.at(OpIndex::Add_0)));
    EXPECT_TRUE(ov::is_type<ov::nvidia_gpu::nodes::ConcatOptimized>(exec_sequence_.at(OpIndex::ConcatOptimized)));
    EXPECT_TRUE(ov::is_type<ov::op::v1::Add>(exec_sequence_.at(OpIndex::Add_1)));
    EXPECT_TRUE(ov::is_type<ov::op::v0::Constant>(exec_sequence_.at(OpIndex::Constant_Reshape_Pattern)));
    EXPECT_TRUE(ov::is_type<ov::op::v1::Reshape>(exec_sequence_.at(OpIndex::Reshape1)));
    EXPECT_TRUE(ov::is_type<ov::op::v0::Result>(exec_sequence_.at(OpIndex::Result)));
}

TEST_F(OperationBufferExtractorConcatOptimizedV2Test, CheckMutableBuffersIndices) {
    using ::testing::ElementsAre;
    auto buffer_indices = extractor_->mutableBuffersIds();
    std::sort(buffer_indices.begin(), buffer_indices.end());
    ASSERT_THAT(
        buffer_indices,
        ElementsAre(OutputBufferIndex::Parameter, OutputBufferIndex::ConcatOptimized, OutputBufferIndex::Add_1));
}

TEST_F(OperationBufferExtractorConcatOptimizedV2Test, CheckImmutableBuffersIndices) {
    using ::testing::ElementsAre;
    auto immutable_buffer_indices = extractor_->immutableBuffersIds();
    std::sort(immutable_buffer_indices.begin(), immutable_buffer_indices.end());
    ASSERT_THAT(immutable_buffer_indices,
                ElementsAre(OutputBufferIndex::Constant_Multiplier,
                            OutputBufferIndex::Constant_Adder_0,
                            OutputBufferIndex::Constant_Reshape_0_Pattern,
                            OutputBufferIndex::Constant_Adder_1,
                            OutputBufferIndex::Constant_Reshape_1_Pattern));
}
