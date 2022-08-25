// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cuda_op_buffers_extractor.hpp>
#include <cuda_operation_registry.hpp>
#include <cuda_profiler.hpp>
#include <ngraph/node.hpp>
#include <ops/parameter.hpp>
#include <typeinfo>

#include "nodes/parameter_stub_node.hpp"

using namespace InferenceEngine;
using namespace ov::nvidia_gpu;
using devptr_t = DevicePointer<void*>;

/**
 * @brief Fill InferenceEngine blob with random values
 */
template <typename T>
void fillBlobRandom(Blob::Ptr& inputBlob) {
    MemoryBlob::Ptr minput = as<MemoryBlob>(inputBlob);
    // locked memory holder should be alive all time while access to its buffer happens
    auto minputHolder = minput->wmap();

    auto inputBlobData = minputHolder.as<T*>();
    for (size_t i = 0; i < inputBlob->size(); i++) {
        auto rand_max = RAND_MAX;
        inputBlobData[i] = (T)rand() / static_cast<T>(rand_max) * 10;
    }
}

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
        node->set_friendly_name(ParameterStubNode::type_info.name);
        ASSERT_TRUE(registry.hasOperation(node));
        operation = registry.createOperation(CreationContext{device, optimizeOption}, node, inputIDs, outputIDs);
        ASSERT_TRUE(operation);
        auto parameterOp = dynamic_cast<ParameterOp*>(operation.get());
        ASSERT_TRUE(parameterOp);
        allocate();
        fillBlobRandom<uint8_t>(blob);
        blobsMapping[node->get_friendly_name()] = 0;
        blobs.push_back(std::make_shared<ngraph::HostTensor>(
            ngraph::element::Type_t::u8, blob->getTensorDesc().getDims(), blob->buffer().as<uint8_t*>()));
    }
    void allocate() {
        TensorDesc desc{Precision::U8, {size}, Layout::C};
        blob = InferenceEngine::make_shared_blob<uint8_t>(desc);
        blob->allocate();
    }
    ThreadContext threadContext{{}};
    CUDA::Allocation outAlloc = threadContext.stream().malloc(size);
    OperationBase::Ptr operation;
    IOperationExec::Inputs inputs;
    std::vector<devptr_t> outputs{outAlloc};
    Blob::Ptr blob;
    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> blobs;
    std::map<std::string, std::size_t> blobsMapping;
    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> emptyTensor;
    std::map<std::string, std::size_t> emptyMapping;
};

TEST_F(ParameterRegistryTest, GetOperationBuilder_Available) {
    ASSERT_TRUE(OperationRegistry::getInstance().hasOperation(std::make_shared<ParameterStubNode>()));
}

TEST_F(ParameterTest, canExecuteSync) {
    CancellationToken token{};
    CudaGraph graph{CreationContext{CUDA::Device{}, false}, {}};
    Profiler profiler{false, graph};
    InferenceRequestContext context{blobs, blobsMapping, emptyTensor, emptyMapping, threadContext, token, profiler};
    auto& stream = context.getThreadContext().stream();
    operation->Execute(context, inputs, outputs, {});
    auto data = std::make_unique<uint8_t[]>(size);
    stream.download(data.get(), outputs[0], size);
    stream.synchronize();
    auto mem = blob->as<MemoryBlob>()->rmap();
    ASSERT_EQ(0, memcmp(data.get(), mem, size));
}

TEST_F(ParameterTest, canExecuteAsync) {
    CancellationToken token{};
    ov::nvidia_gpu::CudaGraph graph{CreationContext{CUDA::Device{}, false}, {}};
    ov::nvidia_gpu::Profiler profiler{false, graph};
    InferenceRequestContext context{blobs, blobsMapping, emptyTensor, emptyMapping, threadContext, token, profiler};
    auto& stream = context.getThreadContext().stream();
    operation->Execute(context, inputs, outputs, {});
    auto data = std::make_unique<uint8_t[]>(size);
    stream.download(data.get(), outputs[0], size);
    ASSERT_NO_THROW(stream.synchronize());
    auto mem = blob->as<MemoryBlob>()->rmap();
    ASSERT_EQ(0, memcmp(data.get(), mem, size));
}
