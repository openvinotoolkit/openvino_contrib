// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cuda_graph_context.hpp>
#include <cuda_op_buffers_extractor.hpp>
#include <cuda_operation_registry.hpp>
#include <cuda_simple_execution_delegator.hpp>
#include <ops/result.hpp>
#include <typeinfo>

#include "common_test_utils/data_utils.hpp"
#include "nodes/parameter_stub_node.hpp"
#include "nodes/result_stub_node.hpp"

using namespace ov::nvidia_gpu;
using devptr_t = DevicePointer<void*>;
using cdevptr_t = DevicePointer<const void*>;

template <>
class ov::Output<ParameterStubNode> : public ov::Output<ov::Node> {
public:
    explicit Output<ParameterStubNode>(std::shared_ptr<ParameterStubNode> node) : ov::Output<ov::Node>(node, 0) {
        auto tensor = std::make_shared<ov::descriptor::Tensor>(
            ov::element::Type{},
            ov::PartialShape{1},
            std::unordered_set<std::string>{ParameterStubNode::get_type_info_static().name});
        node->get_output_descriptor(0).set_tensor_ptr(tensor);
    }
};

class ResultRegistryTest : public testing::Test {
    void SetUp() override {}

    void TearDown() override {}
};

struct ResultTest : testing::Test {
    static constexpr size_t size = 16 * 1024;
    void SetUp() override {
        CUDA::Device device{};
        const bool optimizeOption = false;
        auto& registry{OperationRegistry::getInstance()};
        auto paramNode = std::make_shared<ParameterStubNode>();
        paramNode->set_friendly_name(ParameterStubNode::get_type_info_static().name);
        auto resultNode = std::make_shared<ResultStubNode>();
        auto outputParameterNode = std::make_shared<ov::Output<ParameterStubNode>>(paramNode);
        resultNode->set_argument(0, *outputParameterNode);
        auto inputIDs = std::vector<TensorID>{TensorID{0}};
        auto outputIDs = std::vector<TensorID>{};
        resultNode->set_friendly_name(ResultStubNode::get_type_info_static().name);
        ASSERT_TRUE(registry.hasOperation(resultNode));
        operation = registry.createOperation(CreationContext{device, optimizeOption}, resultNode, inputIDs, outputIDs);
        ASSERT_TRUE(operation);
        auto resultOp = dynamic_cast<ResultOp*>(operation.get());
        ASSERT_TRUE(resultOp);
        tensor = std::make_shared<ov::Tensor>(ov::element::u8, ov::Shape{size});
        ov::test::utils::fill_tensor_random(*tensor.get());
        tensors_mapping[paramNode->get_friendly_name()] = 0;
        tensors.push_back(tensor);
    }
    ThreadContext threadContext{{}};
    CUDA::Allocation inAlloc = threadContext.stream().malloc(size);
    OperationBase::Ptr operation;
    std::vector<cdevptr_t> inputs{inAlloc};
    IOperationExec::Outputs outputs;
    std::shared_ptr<ov::Tensor> tensor;
    std::vector<std::shared_ptr<ov::Tensor>> tensors;
    std::map<std::string, std::size_t> tensors_mapping;
    std::vector<std::shared_ptr<ov::Tensor>> empty_tensor;
    std::map<std::string, std::size_t> empty_mapping;
};

TEST_F(ResultRegistryTest, GetOperationBuilder_Available) {
    ASSERT_TRUE(OperationRegistry::getInstance().hasOperation(std::make_shared<ResultStubNode>()));
}

TEST_F(ResultTest, canExecuteSync) {
    CancellationToken token{};
    SimpleExecutionDelegator simpleExecutionDelegator{};
    ov::nvidia_gpu::CudaGraphContext cudaGraphContext{};
    InferenceRequestContext context{empty_tensor,
                                    empty_mapping,
                                    tensors,
                                    tensors_mapping,
                                    threadContext,
                                    token,
                                    simpleExecutionDelegator,
                                    cudaGraphContext};
    auto& stream = context.getThreadContext().stream();
    stream.upload(inputs[0].as_mutable(), tensor->data(), size);
    operation->Execute(context, inputs, outputs, {});
    auto data = std::make_unique<uint8_t[]>(size);
    stream.download(data.get(), inputs[0], size);
    stream.synchronize();
    ASSERT_EQ(0, memcmp(data.get(), tensor->data(), size));
}

TEST_F(ResultTest, canExecuteAsync) {
    CancellationToken token{};
    SimpleExecutionDelegator simpleExecutionDelegator{};
    ov::nvidia_gpu::CudaGraphContext cudaGraphContext{};
    InferenceRequestContext context{empty_tensor,
                                    empty_mapping,
                                    tensors,
                                    tensors_mapping,
                                    threadContext,
                                    token,
                                    simpleExecutionDelegator,
                                    cudaGraphContext};
    auto& stream = context.getThreadContext().stream();
    stream.upload(inputs[0].as_mutable(), tensor->data(), size);
    operation->Execute(context, inputs, outputs, {});
    auto data = std::make_unique<uint8_t[]>(size);
    stream.download(data.get(), inputs[0], size);
    ASSERT_NO_THROW(stream.synchronize());
    ASSERT_EQ(0, memcmp(data.get(), tensor->data(), size));
}
