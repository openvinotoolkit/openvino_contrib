// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cuda_operation_registry.hpp>
#include <ngraph/node.hpp>
#include <ops/parameter.hpp>
#include <typeinfo>

#include "nodes/parameter_stub_node.hpp"

using namespace InferenceEngine::gpu;
using namespace InferenceEngine;
using namespace CUDAPlugin;
using devptr_t = CUDA::DevicePointer<void*>;

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
        auto inputIDs = std::vector<CUDAPlugin::TensorID>{};
        auto outputIDs = std::vector<CUDAPlugin::TensorID>{CUDAPlugin::TensorID{0}};
        node->set_friendly_name(ParameterStubNode::type_info.name);
        ASSERT_TRUE(registry.hasOperation(node));
        operation = registry.createOperation(CUDA::CreationContext{device, optimizeOption}, node, inputIDs, outputIDs);
        ASSERT_TRUE(operation);
        auto parameterOp = dynamic_cast<ParameterOp*>(operation.get());
        ASSERT_TRUE(parameterOp);
        allocate();
        fillBlobRandom<uint8_t>(blob);
        blobs.insert({node->get_friendly_name(), blob});
    }
    void allocate() {
        TensorDesc desc{Precision::U8, {size}, Layout::C};
        blob = InferenceEngine::make_shared_blob<uint8_t>(desc);
        blob->allocate();
    }
    CUDA::ThreadContext threadContext{{}};
    CUDA::Allocation outAlloc = threadContext.stream().malloc(size);
    OperationBase::Ptr operation;
    IOperationExec::Inputs inputs;
    std::vector<devptr_t> outputs{outAlloc};
    Blob::Ptr blob;
    InferenceEngine::BlobMap blobs;
    InferenceEngine::BlobMap empty;
};

TEST_F(ParameterRegistryTest, GetOperationBuilder_Available) {
    ASSERT_TRUE(OperationRegistry::getInstance().hasOperation(std::make_shared<ParameterStubNode>()));
}

TEST_F(ParameterTest, canExecuteSync) {
    InferenceRequestContext context{blobs, empty, threadContext};
    auto& stream = context.getThreadContext().stream();
    operation->Execute(context, inputs, outputs, {});
    auto data = std::make_unique<uint8_t[]>(size);
    stream.download(data.get(), outputs[0], size);
    stream.synchronize();
    auto mem = blob->as<MemoryBlob>()->rmap();
    ASSERT_EQ(0, memcmp(data.get(), mem, size));
}

TEST_F(ParameterTest, canExecuteAsync) {
    InferenceRequestContext context{blobs, empty, threadContext};
    auto& stream = context.getThreadContext().stream();
    operation->Execute(context, inputs, outputs, {});
    auto data = std::make_unique<uint8_t[]>(size);
    stream.download(data.get(), outputs[0], size);
    ASSERT_NO_THROW(stream.synchronize());
    auto mem = blob->as<MemoryBlob>()->rmap();
    ASSERT_EQ(0, memcmp(data.get(), mem, size));
}
