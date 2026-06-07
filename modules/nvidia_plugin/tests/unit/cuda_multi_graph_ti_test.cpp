// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <random>

#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/node_builders/gru_cell.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "cuda_graph_topology_runner.hpp"
#include "cuda_simple_execution_delegator.hpp"
#include "gtest/gtest.h"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace ov::nvidia_gpu;
using namespace testing;
using ov::test::utils::EltwiseTypes;

namespace {

constexpr int TO = 10;
constexpr int FROM = 0;
constexpr int SEED = 1;

constexpr std::size_t INPUTS_COUNT = 2;
constexpr int64_t CONCAT_AXIS = 0;

constexpr float THRESHOLD = 0.01f;

using CalcType = float;
constexpr auto CALC_ELEMENT_TYPE = ov::element::Type_t::f32;

inline CalcType* getMutablePtr(ov::Tensor& tensor) { return static_cast<CalcType*>(tensor.data()); }

inline const CalcType* getConstPtr(const ov::Tensor& tensor) { return static_cast<const CalcType*>(tensor.data()); }

void generateInput(ov::Tensor& tensor, int to = TO, int from = FROM, int seed = SEED) {
    EXPECT_EQ(tensor.get_element_type(), CALC_ELEMENT_TYPE);
    auto* ptr = getMutablePtr(tensor);
    std::mt19937 engine(seed);
    std::uniform_real_distribution<float> dist(from, to);
    std::generate(ptr, ptr + tensor.get_size(), [&dist, &engine]() { return CalcType{dist(engine)}; });
}

ov::TensorVector calcRefs(std::shared_ptr<ov::Model> model, const std::vector<std::shared_ptr<ov::Tensor>>& inputs) {
    auto ref_model = model->clone();

    ov::TensorVector inputs_ref;
    for (auto& input : inputs) {
        inputs_ref.push_back(*input);
    }

    return ov::test::utils::infer_on_template(ref_model, inputs_ref);
}

void validateOutput(const ov::Tensor& tensor, const ov::Tensor& ref_tensor, float threshold) {
    EXPECT_EQ(tensor.get_element_type(), CALC_ELEMENT_TYPE);
    EXPECT_EQ(ref_tensor.get_element_type(), CALC_ELEMENT_TYPE);
    const auto size = tensor.get_size();
    EXPECT_EQ(size, ref_tensor.get_size());
    const auto* ptr = getConstPtr(tensor);
    const auto* ref_ptr = getConstPtr(ref_tensor);
    bool areEqual = std::equal(
        ptr, ptr + size, ptr, [threshold](auto val1, auto val2) { return std::abs(val1 - val2) < threshold; });
    EXPECT_TRUE(areEqual);
}
}  // namespace

class GRUTI {
public:
    static std::shared_ptr<ov::Model> createNetwork() {
        constexpr size_t seqLengths = 20;
        constexpr size_t batch = 1;
        constexpr size_t hidden_size = 10;
        constexpr size_t inputSize = 10;
        constexpr size_t seqAxis = 1;
        constexpr float clip = 0.0;
        constexpr ov::element::Type ngPrc = CALC_ELEMENT_TYPE;

        auto tensorIterator = std::make_shared<ov::op::v0::TensorIterator>();
        auto axis = std::make_shared<ov::op::v0::Constant>(
            ov::element::Type_t::i64, ov::Shape{1}, std::vector<int64_t>{static_cast<int64_t>(seqAxis)});
        std::vector<std::vector<size_t>> inputShapes = {
            {{batch, seqLengths, inputSize},
             {batch, hidden_size},
             {3 * hidden_size, inputSize},
             {3 * hidden_size, hidden_size},
             {3 * hidden_size}},
        };
        ov::ParameterVector outerParams{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[0])),
                                        std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[1]))};

        inputShapes[0][seqAxis] = 1;  // sliced dimension
        ov::ParameterVector bodyParams{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[0])),
                                       std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes[1]))};

        std::vector<ov::Shape> WRB = {inputShapes[2], inputShapes[3], inputShapes[4]};
        auto squeeze = std::make_shared<ov::op::v0::Squeeze>(bodyParams[0], axis);
        ov::OutputVector out_vector = {squeeze, bodyParams[1]};
        auto gru_cell =
            ov::test::utils::make_gru(out_vector, WRB, hidden_size, {"sigmoid", "tanh"}, {}, {}, clip, false);
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(gru_cell->output(0), axis);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(gru_cell->output(0)),
                                 std::make_shared<ov::op::v0::Result>(unsqueeze)};
        auto body = std::make_shared<ov::Model>(results, bodyParams, "gru_cell");
        tensorIterator->set_function(body);

        tensorIterator->set_sliced_input(bodyParams[0], outerParams[0], -1, -1, 1, 0, seqAxis);
        tensorIterator->get_concatenated_slices(results[1], -1, -1, 1, 0, seqAxis);

        tensorIterator->set_merged_input(bodyParams[1], outerParams[1], results[0]);
        tensorIterator->get_iter_value(results[0]);

        return std::make_shared<ov::Model>(ov::OutputVector{tensorIterator->output(0), tensorIterator->output(1)},
                                           outerParams);
    }

    static void checkContext(CudaGraphContext& cudaGraphContext) {
        // TI has always a separate graph in CudaGraphContext
        // Single-graph TI version uses CudaGraphInfo object with 1 graph
        // Total graph count should be 3
        EXPECT_EQ(cudaGraphContext.get_graphs_count(), 3);
        EXPECT_TRUE(cudaGraphContext.is_nested());

        cudaGraphContext.select_current_graph(1);
        const auto& tiGraph = cudaGraphContext.get_current_graph();
        EXPECT_FALSE(tiGraph.is_nested());
        EXPECT_EQ(tiGraph.get_graphs_count(), 1);
    }

    static void checkRunner(const CudaGraphTopologyRunner& runner) {
        // CudaGraphTopologyRunner always puts a TI into a separate SubGraph
        // Single-graph TI version doesn't use nested CudaGraphTopologyRunner objects and uses 1 graph
        // Total graph count should be 3
        EXPECT_EQ(runner.GetSubGraph().GetCudaGraphCompatibility(), CudaGraphCompatibility::SPECIAL);
        EXPECT_EQ(runner.GetCudaGraphsCount(), 3);
        EXPECT_FALSE(runner.hasNestedRunners());
    }
};

class SplitConcatAddTI {
public:
    static void createNetworkInternal(std::shared_ptr<ov::Model>& model) {
        constexpr size_t seqLengths = 20;
        constexpr size_t batch = 1;
        constexpr size_t inputSize = 10;
        constexpr size_t seqAxis = 1;
        constexpr float clip = 0.0;
        ov::element::Type ngPrc = CALC_ELEMENT_TYPE;

        auto tensorIterator = std::make_shared<ov::op::v0::TensorIterator>();
        auto axisConstant = std::make_shared<ov::op::v0::Constant>(
            ov::element::Type_t::i64, ov::Shape{1}, std::vector<int64_t>{static_cast<int64_t>(seqAxis)});
        std::vector<size_t> outerShape = {{batch, seqLengths, inputSize}};
        std::vector<std::vector<size_t>> bodyShapes;
        for (std::size_t i = 0; i < INPUTS_COUNT; ++i) {
            bodyShapes.emplace_back(std::vector<size_t>{batch, 1, inputSize});
        }
        ov::ParameterVector outerParams;
        outerParams.emplace_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{outerShape}));
        for (std::size_t i = 1; i < INPUTS_COUNT; ++i) {
            outerParams.emplace_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{bodyShapes[i]}));
        }

        for (std::size_t i = 0; i < INPUTS_COUNT; ++i) {
            ASSERT_EQ(outerShape.size(), bodyShapes[i].size());
            for (std::size_t j = 0; j < bodyShapes[i].size(); ++j) {
                if (j == seqAxis) {
                    ASSERT_EQ(bodyShapes[i][j], 1);
                    ASSERT_EQ(outerShape[j], seqLengths);
                } else {
                    ASSERT_EQ(bodyShapes[i][j], outerShape[j]);
                }
            }
        }
        ov::ParameterVector bodyParams;
        for (std::size_t i = 0; i < INPUTS_COUNT; ++i) {
            bodyParams.emplace_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{bodyShapes[i]}));
        }

        auto squeeze = std::make_shared<ov::op::v0::Squeeze>(bodyParams[0], axisConstant);
        const auto split_axis_op =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1});
        const auto split = std::make_shared<ov::op::v1::Split>(squeeze, split_axis_op, 2);
        const auto concat =
            std::make_shared<ov::op::v0::Concat>(ov::OutputVector{split->output(0), split->output(1)}, 1);
        const auto add0 = ov::test::utils::make_eltwise(concat->output(0), bodyParams[1], EltwiseTypes::ADD);

        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(add0->output(0), axisConstant);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(add0->output(0)),
                                 std::make_shared<ov::op::v0::Result>(unsqueeze)};

        auto body = std::make_shared<ov::Model>(results, bodyParams, "AddConcat");
        tensorIterator->set_function(body);

        tensorIterator->set_sliced_input(bodyParams[0], outerParams[0], -1, -1, 1, 0, seqAxis);
        tensorIterator->get_concatenated_slices(results[1], -1, -1, 1, 0, seqAxis);

        tensorIterator->set_merged_input(bodyParams[1], outerParams[1], results[0]);
        tensorIterator->get_iter_value(results[0]);

        model = std::make_shared<ov::Model>(ov::OutputVector{tensorIterator->output(0), tensorIterator->output(1)},
                                            outerParams);
    }

    static std::shared_ptr<ov::Model> createNetwork() {
        std::shared_ptr<ov::Model> model;
        createNetworkInternal(model);
        return model;
    }

    static void checkContext(CudaGraphContext& cudaGraphContext) {
        // TI has always a separate graph in CudaGraphContext
        // Multi-graph TI version uses CudaGraphPack object with 3 graphs
        // Total graph count should be 5
        EXPECT_EQ(cudaGraphContext.get_graphs_count(), 5);
        EXPECT_TRUE(cudaGraphContext.is_nested());
        cudaGraphContext.select_current_graph(1);
        const auto& tiGraph = cudaGraphContext.get_current_graph();
        EXPECT_TRUE(tiGraph.is_nested());
        EXPECT_EQ(tiGraph.get_graphs_count(), 3);
    }

    static void checkRunner(const CudaGraphTopologyRunner& runner) {
        // CudaGraphTopologyRunner always puts a TI into a separate SubGraph
        // Multi-graph TI version uses nested CudaGraphTopologyRunner and uses 3 graphs
        // Total graph count should be 5
        EXPECT_EQ(runner.GetSubGraph().GetCudaGraphCompatibility(), CudaGraphCompatibility::SPECIAL);
        EXPECT_EQ(runner.GetCudaGraphsCount(), 5);
        EXPECT_TRUE(runner.hasNestedRunners());
    }
};

template <typename Network>
class CudaMultiGraphTest : public Test {
protected:
    static std::map<std::string, std::size_t> populateInputIndices(std::shared_ptr<ov::Model> model) {
        std::map<std::string, std::size_t> inputIndices;
        for (const auto& parameter : model->get_parameters()) {
            const auto& parameter_index = model->get_parameter_index(parameter);
            inputIndices.emplace(ParameterOp::GetInputTensorName(*parameter), parameter_index);
        }
        return inputIndices;
    }

    static std::map<std::string, std::size_t> populateOutputIndices(std::shared_ptr<ov::Model> model) {
        std::map<std::string, std::size_t> outputIndices;
        for (auto& result : model->get_results()) {
            const auto& result_index = model->get_result_index(result->input_value(0));
            for (const auto& outputName : ResultOp::GetOutputTensorName(*result)) {
                outputIndices.emplace(outputName, result_index);
            }
        }
        return outputIndices;
    }

    static std::vector<std::shared_ptr<ov::Tensor>> populateTensors(const std::vector<ov::Output<ov::Node>>& nodes) {
        std::vector<std::shared_ptr<ov::Tensor>> result;
        for (const auto& node : nodes) {
            result.push_back(std::make_shared<ov::Tensor>(node.get_element_type(), node.get_shape()));
        }
        return result;
    }

    void generateInputs() {
        for (auto& input : inputTensors_) {
            generateInput(*input, TO, FROM, currentSeed_);
            ++currentSeed_;
        }
    }

    void updateContext() { runner_.UpdateContext(*inferRequestContext_, deviceMemBlock_); }

    void checkConditions() {
        Network::checkContext(cudaGraphContext_);
        Network::checkRunner(runner_);
    }

    void run() { runner_.Run(*inferRequestContext_, deviceMemBlock_); }

    void calcRefs() { refOutputTensors_ = ::calcRefs(model_, inputTensors_); }

    void validate(float threshold = THRESHOLD) {
        const auto size = outputTensors_.size();
        EXPECT_EQ(size, refOutputTensors_.size());
        for (std::size_t i = 0; i < size; ++i) {
            validateOutput(*outputTensors_[i], refOutputTensors_[i], THRESHOLD);
        }
    }

    void updateTensors() {
        inputTensors_ = {populateTensors(model_->inputs())};
        outputTensors_ = {populateTensors(model_->outputs())};
        inferRequestContext_ = std::make_unique<InferenceRequestContext>(inputTensors_,
                                                                         inputIndices_,
                                                                         outputTensors_,
                                                                         outputIndices_,
                                                                         threadContext_,
                                                                         cancellationToken_,
                                                                         simpleExecutionDelegator_,
                                                                         cudaGraphContext_,
                                                                         false);
    }

    void runTest() {
        generateInputs();
        updateContext();
        checkConditions();
        run();
        calcRefs();
        validate();

        updateTensors();
        generateInputs();
        updateContext();
        checkConditions();
        run();
        calcRefs();
        validate();
    }

    std::shared_ptr<ov::Model> model_{Network::createNetwork()};
    CreationContext creationContext_{{}, false};
    ThreadContext threadContext_{{}};
    CancellationToken cancellationToken_{};
    CudaGraphContext cudaGraphContext_{};
    CudaGraphTopologyRunner runner_{creationContext_, model_};
    SimpleExecutionDelegator simpleExecutionDelegator_{};
    std::vector<std::shared_ptr<ov::Tensor>> inputTensors_{populateTensors(model_->inputs())};
    std::vector<std::shared_ptr<ov::Tensor>> outputTensors_{populateTensors(model_->outputs())};
    ov::TensorVector refOutputTensors_;
    std::map<std::string, std::size_t> inputIndices_{populateInputIndices(model_)};
    std::map<std::string, std::size_t> outputIndices_{populateOutputIndices(model_)};
    std::unique_ptr<InferenceRequestContext> inferRequestContext_ =
        std::make_unique<InferenceRequestContext>(inputTensors_,
                                                  inputIndices_,
                                                  outputTensors_,
                                                  outputIndices_,
                                                  threadContext_,
                                                  cancellationToken_,
                                                  simpleExecutionDelegator_,
                                                  cudaGraphContext_,
                                                  false);
    DeviceMemBlock deviceMemBlock_{runner_.GetSubGraph().memoryManager()->mutableTensorsMemoryModel()};

    int currentSeed_ = SEED;
};

using GRUTIMultiGraphTest = CudaMultiGraphTest<GRUTI>;

TEST_F(GRUTIMultiGraphTest, CudaMultiGraphTest) { runTest(); }

using SplitConcatAddTIMultiGraphTest = CudaMultiGraphTest<SplitConcatAddTI>;

TEST_F(SplitConcatAddTIMultiGraphTest, CudaMultiGraphTest) { runTest(); }
