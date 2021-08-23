// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_runtime.h>
#include <cuda_operation_registry.hpp>
#include <ngraph/node.hpp>
#include <ops/result.hpp>
#include <typeinfo>
#include <gtest/gtest.h>
#include <ngraph/function.hpp>

#include "nodes/result_stub_node.hpp"
#include "nodes/parameter_stub_node.hpp"

using namespace InferenceEngine::gpu;
using namespace InferenceEngine;
using namespace CUDAPlugin;
using devptr_t = DevicePointer<void*>;
using cdevptr_t = DevicePointer<const void*>;

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

template<>
class ngraph::Output<ParameterStubNode> : public ngraph::Output<ngraph::Node> {
 public:
    explicit Output<ParameterStubNode>(std::shared_ptr<ParameterStubNode> node)
        : ngraph::Output<ngraph::Node>(node, 0) {
        auto tensor = std::make_shared<ngraph::descriptor::Tensor>(
            ngraph::element::Type{}, ngraph::PartialShape{1}, ParameterStubNode::type_info.name);
        node->m_outputs.emplace_back(node.get(), 0, tensor);
    }
};

class ResultRegistryTest : public testing::Test {
  void SetUp() override {
  }

  void TearDown() override {
  }
};

struct ResultTest : testing::Test {
  static constexpr size_t size = 16*1024;
  void SetUp() override {
    CUDA::Device device{};
    const bool optimizeOption = false;
    auto& registry { OperationRegistry::getInstance() };
    auto paramNode = std::make_shared<ParameterStubNode>();
    paramNode->set_friendly_name(ParameterStubNode::type_info.name);
    auto resultNode = std::make_shared<ResultStubNode>();
    auto outputParameterNode = std::make_shared<ngraph::Output<ParameterStubNode>>(paramNode);
    resultNode->set_argument(0, *outputParameterNode);
    auto inputIDs  = std::vector<CUDAPlugin::TensorID>{0};
    auto outputIDs = std::vector<CUDAPlugin::TensorID>{};
    resultNode->set_friendly_name(ResultStubNode::type_info.name);
    ASSERT_TRUE(registry.hasOperation(resultNode));
    operation = registry.createOperation(CUDA::CreationContext{device, optimizeOption}, resultNode, inputIDs, outputIDs);
    ASSERT_TRUE(operation);
    auto resultOp = dynamic_cast<ResultOp*>(operation.get());
    ASSERT_TRUE(resultOp);
    allocate();
    fillBlobRandom<uint8_t>(blob);
    blobs.insert({paramNode->get_friendly_name(), blob});
  }
  void allocate() {
    TensorDesc desc {Precision::U8, {size}, Layout::C };
    blob = InferenceEngine::make_shared_blob<uint8_t>(desc);
    blob->allocate();
  }
  CUDA::ThreadContext threadContext{{}};
  CUDA::Allocation inAlloc = threadContext.stream().malloc(size);
  OperationBase::Ptr operation;
  std::vector<cdevptr_t> inputs{inAlloc};
  IOperationExec::Outputs outputs;
  Blob::Ptr blob;
  InferenceEngine::BlobMap blobs;
  InferenceEngine::BlobMap empty;
};

TEST_F(ResultRegistryTest, GetOperationBuilder_Available) {
  ASSERT_TRUE(OperationRegistry::getInstance().hasOperation(std::make_shared<ResultStubNode>()));
}

TEST_F(ResultTest, canExecuteSync) {
  InferenceRequestContext context{empty, blobs, threadContext};
  auto mem = blob->as<MemoryBlob>()->rmap();
  auto& stream = context.getThreadContext().stream();
  stream.upload(inputs[0].as_mutable(), mem, size);
  operation->Execute(context, inputs, outputs, {});
  auto data = std::make_unique<uint8_t[]>(size);
  stream.download(data.get(), inputs[0], size);
  stream.synchronize();
  ASSERT_EQ(0, memcmp(data.get(), mem, size));
}

TEST_F(ResultTest, canExecuteAsync) {
  InferenceRequestContext context{empty, blobs, threadContext};
  auto& stream = context.getThreadContext().stream();
  auto mem = blob->as<MemoryBlob>()->rmap();
  stream.upload(inputs[0].as_mutable(), mem, size);
  operation->Execute(context, inputs, outputs, {});
  auto data = std::make_unique<uint8_t[]>(size);
  stream.download(data.get(), inputs[0], size);
  ASSERT_NO_THROW(stream.synchronize());
  ASSERT_EQ(0, memcmp(data.get(), mem, size));
}
