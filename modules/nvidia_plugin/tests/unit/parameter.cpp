// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cuda_graph_context.hpp>
#include <cuda_op_buffers_extractor.hpp>
#include <cuda_operation_registry.hpp>
#include <cuda_simple_execution_delegator.hpp>
#include <ops/parameter.hpp>
#include <typeinfo>

#include "common_test_utils/data_utils.hpp"
#include "nodes/parameter_stub_node.hpp"

using namespace ov::nvidia_gpu;
using devptr_t = DevicePointer<void*>;
class ParameterRegistryTest : public testing::Test {
    void SetUp() override {}

    void TearDown() override {}
};

struct ParameterTest : testing::Test {
    static constexpr size_t size = 16 * 1024;
    void SetUp() override {
        CUDA::Device device{};
        const bool optimizeOption = false;
        auto& registry{OperationRegistry::getInstance()};
        auto node = std::make_shared<ParameterStubNode>();
        auto inputIDs = std::vector<ov::nvidia_gpu::TensorID>{};
        auto outputIDs = std::vector<ov::nvidia_gpu::TensorID>{ov::nvidia_gpu::TensorID{0}};
        node->set_friendly_name(ParameterStubNode::get_type_info_static().name);
        ASSERT_TRUE(registry.hasOperation(node));
        operation = registry.createOperation(CreationContext{device, optimizeOption}, node, inputIDs, outputIDs);
        ASSERT_TRUE(operation);
        auto parameterOp = dynamic_cast<ParameterOp*>(operation.get());
        ASSERT_TRUE(parameterOp);
        tensor = std::make_shared<ov::Tensor>(ov::element::u8, ov::Shape{size});
        ov::test::utils::fill_tensor_random(*tensor.get());
        tensors_mapping[node->get_friendly_name()] = 0;
        tensors.push_back(tensor);
    }
    ThreadContext threadContext{{}};
    CUDA::Allocation outAlloc = threadContext.stream().malloc(size);
    OperationBase::Ptr operation;
    IOperationExec::Inputs inputs;
    std::vector<devptr_t> outputs{outAlloc};
    std::shared_ptr<ov::Tensor> tensor;
    std::vector<std::shared_ptr<ov::Tensor>> tensors;
    std::map<std::string, std::size_t> tensors_mapping;
    std::vector<std::shared_ptr<ov::Tensor>> empty_tensor;
    std::map<std::string, std::size_t> empty_mapping;
};

TEST_F(ParameterRegistryTest, GetOperationBuilder_Available) {
    ASSERT_TRUE(OperationRegistry::getInstance().hasOperation(std::make_shared<ParameterStubNode>()));
}

TEST_F(ParameterTest, canExecuteSync) {
    CancellationToken token{};
    SimpleExecutionDelegator simpleExecutionDelegator{};
    ov::nvidia_gpu::CudaGraphContext cudaGraphContext{};
    InferenceRequestContext context{tensors,
                                    tensors_mapping,
                                    empty_tensor,
                                    empty_mapping,
                                    threadContext,
                                    token,
                                    simpleExecutionDelegator,
                                    cudaGraphContext};
    auto& stream = context.getThreadContext().stream();
    operation->Execute(context, inputs, outputs, {});
    auto data = std::make_unique<uint8_t[]>(size);
    stream.download(data.get(), outputs[0], size);
    stream.synchronize();
    ASSERT_EQ(0, memcmp(data.get(), tensor->data(), size));
}

TEST_F(ParameterTest, canExecuteAsync) {
    CancellationToken token{};
    ov::nvidia_gpu::SimpleExecutionDelegator simpleExecutionDelegator{};
    ov::nvidia_gpu::CudaGraphContext cudaGraphContext{};
    InferenceRequestContext context{tensors,
                                    tensors_mapping,
                                    empty_tensor,
                                    empty_mapping,
                                    threadContext,
                                    token,
                                    simpleExecutionDelegator,
                                    cudaGraphContext};
    auto& stream = context.getThreadContext().stream();
    operation->Execute(context, inputs, outputs, {});
    auto data = std::make_unique<uint8_t[]>(size);
    stream.download(data.get(), outputs[0], size);
    ASSERT_NO_THROW(stream.synchronize());
    ASSERT_EQ(0, memcmp(data.get(), tensor->data(), size));
}
