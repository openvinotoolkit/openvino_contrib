// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cuda_config.hpp>
#include <cuda_operation_registry.hpp>
#include <ngraph/node.hpp>
#include <ngraph/op/parameter.hpp>
#include <ngraph/op/relu.hpp>
#include <ops/parameter.hpp>
#include <typeinfo>
namespace {

using devptr_t = InferenceEngine::gpu::DevicePointer<void*>;
using cdevptr_t = InferenceEngine::gpu::DevicePointer<const void*>;
template <typename F>
auto assertToThrow(F&& f, const std::experimental::source_location& loc =
                              std::experimental::source_location::current()) {
  bool success = false;
  std::forward<F>(f)(success);
  if (!success)
    CUDA::throwIEException("pathetic google test failed in non-void function",
                           loc);
}

#define TASSERT_TRUE(condition)              \
  assertToThrow([&](bool& tassert_success) { \
    ASSERT_TRUE(condition);                  \
    tassert_success = true;                  \
  })

struct ReluTest : testing::Test {
  using ElementType = float;
  static constexpr int length = 5;
  static constexpr size_t size = length * sizeof(ElementType);
  CUDA::ThreadContext threadContext{{}};
  CUDA::Allocation inAlloc = threadContext.stream().malloc(size);
  CUDA::Allocation outAlloc = threadContext.stream().malloc(size);
  std::vector<cdevptr_t> inputs{inAlloc.get()};
  std::vector<devptr_t> outputs{outAlloc.get()};
  InferenceEngine::BlobMap empty;
  CUDAPlugin::OperationBase::Ptr operation = [this] {
    CUDA::Device device{};
    const bool optimizeOption = false;
    auto param = std::make_shared<ngraph::op::v0::Parameter>(
        ngraph::element::f32, ngraph::PartialShape{length});
    auto node = std::make_shared<ngraph::op::v0::Relu>(param->output(0));
    auto& registry = CUDAPlugin::OperationRegistry::getInstance();
    TASSERT_TRUE(registry.hasOperation(node));
    auto op = registry.createOperation(
        CUDA::CreationContext{device, optimizeOption},
        node, std::vector<CUDAPlugin::TensorID>{0u}, std::vector<CUDAPlugin::TensorID>{0u});
    TASSERT_TRUE(op);
    return op;
  }();
};

TEST_F(ReluTest, canExecuteSync) {
  InferenceEngine::gpu::InferenceRequestContext context{empty, empty,
                                                        threadContext};
  auto& stream = context.getThreadContext().stream();
  std::array<ElementType, length> in{-1, 1, -5, 5, 0};
  std::array<ElementType, length> correct;
  for (std::size_t i = 0; i < in.size(); i++)
    if (auto c = in[i]; c < 0)
      correct[i] = 0;
    else
      correct[i] = c;
  stream.upload(inAlloc, in.data(), size);
  operation->Execute(context, inputs, outputs, {});
  std::array<ElementType, length> out;
  out.fill(-1);
  stream.download(out.data(), outputs[0], size);
  stream.synchronize();
  for (std::size_t i = 0; i < out.size(); i++)
    EXPECT_EQ(out[i], correct[i]) << "at i == " << i;
  ASSERT_EQ(std::memcmp(out.data(), correct.data(), sizeof out), 0);
}

}  // namespace
