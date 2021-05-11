// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_runtime.h>
#include <cuda_operation_registry.hpp>
#include <ngraph/node.hpp>
#include <ops/parameter.hpp>
#include <typeinfo>
#include <gtest/gtest.h>

using namespace InferenceEngine::gpu;
using namespace InferenceEngine;
using namespace CUDAPlugin;
using devptr_t = DevicePointer<void*>;

/**
 * @brief Fill InferenceEngine blob with random values
 */
template<typename T>
void fillBlobRandom(Blob::Ptr& inputBlob) {
  MemoryBlob::Ptr minput = as<MemoryBlob>(inputBlob);
  // locked memory holder should be alive all time while access to its buffer happens
  auto minputHolder = minput->wmap();

  auto inputBlobData = minputHolder.as<T *>();
  for (size_t i = 0; i < inputBlob->size(); i++) {
    auto rand_max = RAND_MAX;
    inputBlobData[i] = (T) rand() / static_cast<T>(rand_max) * 10;
  }
}

class ParameterRegistryTest : public testing::Test {
  void SetUp() override {
  }

  void TearDown() override {
  }
};

struct ParameterStubNode : ngraph::Node {
  static constexpr type_info_t type_info{"Parameter", 0};
  const type_info_t& get_type_info() const override {
    return type_info;
  }

  std::shared_ptr<ngraph::Node>
  clone_with_new_inputs(const ngraph::OutputVector& inputs) const override {
    return std::make_shared<ParameterStubNode>();
  }
};

constexpr ngraph::Node::type_info_t ParameterStubNode::type_info;

struct ParameterTest : testing::Test {
  static constexpr size_t size = 16*1024;
  void SetUp() override {
    auto& registry { OperationRegistry::getInstance() };
    auto node = std::make_shared<ParameterStubNode>();
    auto inputIDs  = std::vector<unsigned>{};
    auto outputIDs = std::vector<unsigned>{0};
    node->set_friendly_name(ParameterStubNode::type_info.name);
    ASSERT_TRUE(registry.hasOperation(node));
    operation = registry.createOperation(node, inputIDs, outputIDs);
    ASSERT_TRUE(operation);
    auto parameterOp = dynamic_cast<ParameterOp*>(operation.get());
    ASSERT_TRUE(parameterOp);
    allocate();
    fillBlobRandom<uint8_t>(blob);
    blobs.insert({parameterOp->GetName(), blob});
  }
  void TearDown() override {
    if (outputs.size() > 0)
      cudaFree(outputs[0].get());
    operation.reset();
  }
  void allocate() {
    void* buffer {};
    auto success = cudaMalloc(&buffer, size);
    ASSERT_TRUE((success == cudaSuccess && buffer != nullptr));
    outputs.push_back({buffer});
    TensorDesc desc {Precision::U8, {size}, Layout::C };
    blob = InferenceEngine::make_shared_blob<uint8_t>(desc);
    blob->allocate();
  }
  OperationBase::Ptr operation;
  IOperationExec::Inputs inputs {};
  std::vector<devptr_t> outputs {};
  Blob::Ptr blob;
  InferenceEngine::BlobMap blobs;
  InferenceEngine::BlobMap empty;
};

TEST_F(ParameterRegistryTest, GetOperationBuilder_Available) {
  ASSERT_TRUE(OperationRegistry::getInstance().hasOperation(std::make_shared<ParameterStubNode>()));
}

TEST_F(ParameterTest, canExecuteSync) {
  InferenceRequestContext context{std::make_shared<CudaStream>(), blobs, empty};
  operation->Execute(context, inputs, outputs);
  auto data = std::make_unique<uint8_t[]>(size);
  cudaMemcpy(data.get(), outputs[0], size, cudaMemcpyDeviceToHost);
  auto mem = blob->as<MemoryBlob>()->rmap();
  ASSERT_EQ(0, memcmp(data.get(), mem, size));
}

TEST_F(ParameterTest, canExecuteAsync) {
  std::shared_ptr<CudaStream> stream;
  ASSERT_NO_THROW(stream = std::make_shared<CudaStream>());
  InferenceRequestContext context{stream, blobs, empty};
  operation->Execute(context, inputs, outputs);
  auto data = std::make_unique<uint8_t[]>(size);
  stream->memcpyAsync(data.get(), outputs[0].cast<const void*>(), size);
  auto mem = blob->as<MemoryBlob>()->rmap();
  ASSERT_NO_THROW(stream->synchronize());
  ASSERT_EQ(0, memcmp(data.get(), mem, size));
}
