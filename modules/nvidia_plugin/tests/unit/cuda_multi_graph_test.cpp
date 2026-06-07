// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <random>
#include <gtest/gtest.h>

#include "common_test_utils/node_builders/eltwise.hpp"
#include "cuda_graph_topology_runner.hpp"
#include "cuda_simple_execution_delegator.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "ops/parameter.hpp"
#include "ops/result.hpp"

using namespace ov::nvidia_gpu;
using namespace testing;
using ov::test::utils::EltwiseTypes;

namespace {

constexpr int TO = 10;
constexpr int FROM = -10;
constexpr int SEED = 1;

constexpr std::size_t INPUTS_COUNT = 4;
constexpr int64_t CONCAT_AXIS = 0;

constexpr float THRESHOLD = 0.01f;

inline ov::float16* getMutablePtr(ov::Tensor& tensor) { return static_cast<ov::float16*>(tensor.data()); }

inline const ov::float16* getConstPtr(const ov::Tensor& tensor) {
    return static_cast<const ov::float16*>(tensor.data());
}

void generateInput(ov::Tensor& tensor, int to = TO, int from = FROM, int seed = SEED) {
    // This test supports only FP16 precision
    EXPECT_EQ(tensor.get_element_type(), ov::element::Type_t::f16);
    auto* ptr = getMutablePtr(tensor);
    std::mt19937 engine(seed);
    std::uniform_real_distribution<float> dist(from, to);
    std::generate(ptr, ptr + tensor.get_size(), [&dist, &engine]() { return ov::float16{dist(engine)}; });
}

void validateOutput(const ov::Tensor& tensor, const std::vector<ov::float16>& refVector, float threshold) {
    // This test supports only FP16 precision
    EXPECT_EQ(tensor.get_element_type(), ov::element::Type_t::f16);
    const auto size = tensor.get_size();
    EXPECT_EQ(size, refVector.size());
    const auto* ptr = getConstPtr(tensor);
    bool areEqual = std::equal(ptr, ptr + size, refVector.cbegin(), [threshold](auto val1, auto val2) {
        return std::abs(val1 - val2) < threshold;
    });
    EXPECT_TRUE(areEqual);
}

}  // namespace

class AddMul {
public:
    static std::shared_ptr<ov::Model> createNetwork() {
        ov::element::Type prc = ov::element::Type_t::f16;
        ov::Shape shape{1, 2, 3, 4};
        ov::ParameterVector params;
        for (std::size_t i = 0; i < INPUTS_COUNT; ++i) {
            params.emplace_back(std::make_shared<ov::op::v0::Parameter>(prc, shape));
        }
        const auto add0 = ov::test::utils::make_eltwise(params[0], params[1], EltwiseTypes::ADD);
        const auto add1 = ov::test::utils::make_eltwise(params[2], params[3], EltwiseTypes::ADD);

        const auto mul = ov::test::utils::make_eltwise(add0, add1, EltwiseTypes::MULTIPLY);
        const auto result = std::make_shared<ov::op::v0::Result>(mul);
        return std::make_shared<ov::Model>(result, params, "AddMul");
    }

    static void checkContext(const CudaGraphContext& cudaGraphContext) {
        // AddMul network should have a single CUDA Graph
        EXPECT_EQ(cudaGraphContext.get_graphs_count(), 1);
    }

    static void checkSubGraph(const SubGraph& subGraph) {
        // Original SubGraph for AddMul network should be CUDA Graph compatible
        EXPECT_EQ(subGraph.GetCudaGraphCompatibility(), CudaGraphCompatibility::FULL);
    }

    static std::vector<std::vector<ov::float16>> calcRefs(
        const std::vector<std::shared_ptr<ov::Tensor>>& inputTensors) {
        EXPECT_EQ(inputTensors.size(), INPUTS_COUNT);
        const auto size = inputTensors[0]->get_size();
        std::vector<std::vector<ov::float16>> result{std::vector<ov::float16>(size)};
        std::array<const ov::float16*, INPUTS_COUNT> inputs;
        for (std::size_t i = 0; i < INPUTS_COUNT; ++i) {
            inputs[i] = getConstPtr(*inputTensors[i]);
        }
        EXPECT_EQ(result.size(), 1);
        auto& output = result[0];
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = (inputs[0][i] + inputs[1][i]) * ((inputs[2][i] + inputs[3][i]));
        }
        return result;
    }
};

class AddConcat {
public:
    static std::shared_ptr<ov::Model> createNetwork() {
        ov::element::Type prc = ov::element::Type_t::f16;
        ov::Shape shape{1, 2, 3, 4};
        ov::ParameterVector params;
        for (std::size_t i = 0; i < INPUTS_COUNT; ++i) {
            params.emplace_back(std::make_shared<ov::op::v0::Parameter>(prc, shape));
        }
        const auto add0 = ov::test::utils::make_eltwise(params[0], params[1], EltwiseTypes::ADD);
        const auto add1 = ov::test::utils::make_eltwise(params[2], params[3], EltwiseTypes::ADD);

        constexpr int64_t axis = CONCAT_AXIS;
        const auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{add0, add1}, axis);
        const auto result = std::make_shared<ov::op::v0::Result>(concat);
        return std::make_shared<ov::Model>(result, params, "AddConcat");
    }

    static void checkContext(const CudaGraphContext& cudaGraphContext) {
        // AddConcat network should have a more than one CUDA Graph
        EXPECT_GT(cudaGraphContext.get_graphs_count(), 1);
    }

    static void checkSubGraph(const SubGraph& subGraph) {
        // Original SubGraph for AddConcat network should not be CUDA Graph compatible
        EXPECT_EQ(subGraph.GetCudaGraphCompatibility(), CudaGraphCompatibility::NONE);
    }

    static std::vector<std::vector<ov::float16>> calcRefs(
        const std::vector<std::shared_ptr<ov::Tensor>>& inputTensors) {
        EXPECT_EQ(inputTensors.size(), INPUTS_COUNT);
        const auto size = inputTensors[0]->get_size();
        std::vector<std::vector<ov::float16>> result{std::vector<ov::float16>(2 * size)};
        std::array<const ov::float16*, INPUTS_COUNT> inputs;
        for (std::size_t i = 0; i < INPUTS_COUNT; ++i) {
            inputs[i] = getConstPtr(*inputTensors[i]);
        }
        std::vector<ov::float16> addResult0(size);
        std::vector<ov::float16> addResult1(size);
        for (std::size_t i = 0; i < size; ++i) {
            addResult0[i] = (inputs[0][i] + inputs[1][i]);
            addResult1[i] = (inputs[2][i] + inputs[3][i]);
        }
        EXPECT_EQ(result.size(), 1);
        auto& output = result[0];
        std::copy(addResult0.cbegin(), addResult0.cend(), output.begin());
        std::copy(addResult1.cbegin(), addResult1.cend(), output.begin() + size);
        return result;
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
        Network::checkSubGraph(runner_.GetSubGraph());
    }

    void run() { runner_.Run(*inferRequestContext_, deviceMemBlock_); }

    void calcRefs() { refOutputs_ = Network::calcRefs(inputTensors_); }

    void validate(float threshold = THRESHOLD) {
        const auto size = outputTensors_.size();
        EXPECT_EQ(size, refOutputs_.size());
        for (std::size_t i = 0; i < size; ++i) {
            validateOutput(*outputTensors_[i], refOutputs_[i], THRESHOLD);
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

    std::vector<std::vector<ov::float16>> refOutputs_;
    int currentSeed_ = SEED;
};

using AddMulMultiGraphTest = CudaMultiGraphTest<AddMul>;

TEST_F(AddMulMultiGraphTest, AddMulTest) { runTest(); }

using AddConcatMultiGraphTest = CudaMultiGraphTest<AddConcat>;

TEST_F(AddConcatMultiGraphTest, AddConcatTest) { runTest(); }
