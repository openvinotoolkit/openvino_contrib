// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>
#include <cuda_operation_registry.hpp>
#include <ops/concat.hpp>
#include <typeinfo>
#include <gtest/gtest.h>
#include <ngraph/function.hpp>
#include <ngraph/output_vector.hpp>

using namespace InferenceEngine::gpu;
using namespace InferenceEngine;
using namespace CUDAPlugin;
using devptr_t = DevicePointer<void*>;
using cdevptr_t = DevicePointer<const void*>;

template<typename T, std::size_t Size>
std::ostream& operator<<(std::ostream& out, gsl::span<T, Size> data) {
  const char *dlm = "";
  for (const auto& i : data) { out << dlm << i; dlm = ","; }
  return out;
}


/**
 * @brief Fill InferenceEngine blob with random values
 */
template<typename T>
void fillBlob(Blob::Ptr& inputBlob, T value) {
  MemoryBlob::Ptr minput = as<MemoryBlob>(inputBlob);
  // locked memory holder should be alive all time while access to its buffer happens
  auto minputHolder = minput->wmap();

  auto inputBlobData = minputHolder.as<T *>();
  for (size_t i = 0; i < inputBlob->size(); i++) {
    inputBlobData[i] = value;
  }
}

struct ConcatTest : testing::Test {
  const std::array<std::array<ngraph::Shape, 3>, 2> shapes {
    std::array<ngraph::Shape, 3>{ngraph::Shape{2, 2}, ngraph::Shape{3, 2}, ngraph::Shape{4, 2}},
    std::array<ngraph::Shape, 3>{ngraph::Shape{2, 2}, ngraph::Shape{2, 3}, ngraph::Shape{2, 4}}
  };
  static constexpr float masters[2][18] = {
    { 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4 },
    { 2, 2, 3, 3, 3, 4, 4, 4, 4, 2, 2, 3, 3, 3, 4, 4, 4, 4 }
  };
  void SetUp() override { }
  void TearDown() override { }
  void run(size_t axis) {
    CUDA::Device device{};
    const bool optimizeOption = false;
    allocate(axis);
    auto& registry { OperationRegistry::getInstance() };
    auto concatNode = std::make_shared<ngraph::op::Concat>(params, axis);
    auto inputIDs = std::vector<TensorID>{TensorID{0}, TensorID{1}, TensorID{2}};
    auto outputIDs = std::vector<TensorID>{TensorID{0}};
    ASSERT_TRUE(registry.hasOperation(concatNode));
    auto operation = registry.createOperation(CUDA::CreationContext{device, optimizeOption}, concatNode, inputIDs, outputIDs);
    ASSERT_TRUE(operation);
    auto concatOp = dynamic_cast<ConcatOp*>(operation.get());
    ASSERT_TRUE(concatOp);
    InferenceRequestContext context{empty, empty, threadContext};
    const auto& stream = threadContext.stream();
    std::vector<cdevptr_t> inputs {};
    std::vector<devptr_t> outputs {};
    std::vector<CUDA::Allocation> mem {};
    for (const auto& blob : blobs) {
      mem.emplace_back(stream.malloc(blob->byteSize()));
      inputs.emplace_back(cdevptr_t { mem.back().get()});
      stream.upload(inputs.back().as_mutable(), blob->as<MemoryBlob>()->rmap(), blob->byteSize());
    }
    mem.emplace_back(stream.malloc(output_size));
    outputs.emplace_back(devptr_t { mem.back().get()});
    auto wb_request = operation->GetWorkBufferRequest();
    ASSERT_EQ(wb_request.immutable_sizes.size(), 1);
    ASSERT_EQ(wb_request.mutable_sizes.size(), 1);
    auto& immutable_wb = mem.emplace_back(stream.malloc(wb_request.immutable_sizes[0]));
    auto& mutable_wb = mem.emplace_back(stream.malloc(wb_request.mutable_sizes[0]));
    operation->InitSharedImmutableWorkbuffers({immutable_wb});
    operation->Execute(context, inputs, outputs,
                       {{immutable_wb}, {mutable_wb}});
    auto data = std::make_unique<float[]>(output_size / sizeof(float));
    stream.synchronize();
    stream.download(data.get(), outputs[0], output_size);
    stream.synchronize();
    ASSERT_EQ(0, memcmp(data.get(), masters[axis], output_size));
  }

  void allocate(size_t axis) {
    for (int i = 0; i < blobs.size(); i++) {
      TensorDesc desc {Precision::FP32, shapes[axis][i], Layout::ANY };
      blobs[i] = InferenceEngine::make_shared_blob<float>(desc);
      blobs[i]->allocate();
      fillBlob<float>(blobs[i], i + 2.0);
      output_size += blobs[i]->byteSize();
      params.emplace_back(std::make_shared<ngraph::op::Parameter>(ngraph::element::Type{ngraph::element::Type_t::f32}, shapes[axis][i]));
    }
  }
  CUDA::ThreadContext threadContext{{}};
  std::array<Blob::Ptr, 3> blobs;
  InferenceEngine::BlobMap empty;
  size_t output_size {};
  ngraph::OutputVector params {};
};

TEST_F(ConcatTest, axis_0) {
  run(0);
}

TEST_F(ConcatTest, axis_1) {
  run(1);
}
