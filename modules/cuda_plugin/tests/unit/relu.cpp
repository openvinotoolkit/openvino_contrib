// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cuda_config.hpp>
#include <cuda_op_buffers_extractor.hpp>
#include <cuda_operation_registry.hpp>
#include <cuda_profiler.hpp>
#include <ngraph/node.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/relu.hpp>
#include <ops/parameter.hpp>
#include <typeinfo>
namespace {

using devptr_t = CUDA::DevicePointer<void*>;
using cdevptr_t = CUDA::DevicePointer<const void*>;
template <typename F>
auto assertToThrow(F&& f,
                   const std::experimental::source_location& loc = std::experimental::source_location::current()) {
    bool success = false;
    std::forward<F>(f)(success);
    if (!success) CUDAPlugin::throwIEException("pathetic google test failed in non-void function", loc);
}

#define TASSERT_TRUE(condition)                \
    assertToThrow([&](bool& tassert_success) { \
        ASSERT_TRUE(condition);                \
        tassert_success = true;                \
    })

struct ReluTest : testing::Test {
    using TensorID = CUDAPlugin::TensorID;
    using ElementType = float;
    static constexpr int length = 5;
    static constexpr size_t size = length * sizeof(ElementType);
    CUDAPlugin::ThreadContext threadContext{{}};
    CUDA::Allocation inAlloc = threadContext.stream().malloc(size);
    CUDA::Allocation outAlloc = threadContext.stream().malloc(size);
    std::vector<cdevptr_t> inputs{inAlloc};
    std::vector<devptr_t> outputs{outAlloc};
    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> emptyTensor;
    std::map<std::string, std::size_t> emptyMapping;
    CUDAPlugin::OperationBase::Ptr operation = [this] {
        CUDA::Device device{};
        const bool optimizeOption = false;
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{length});
        auto node = std::make_shared<ov::op::v0::Relu>(param->output(0));
        auto& registry = CUDAPlugin::OperationRegistry::getInstance();
        TASSERT_TRUE(registry.hasOperation(node));
        auto op = registry.createOperation(CUDAPlugin::CreationContext{device, optimizeOption},
                                           node,
                                           std::vector<TensorID>{TensorID{0u}},
                                           std::vector<TensorID>{TensorID{0u}});
        TASSERT_TRUE(op);
        return op;
    }();
};

TEST_F(ReluTest, canExecuteSync) {
    CUDAPlugin::CancellationToken token{};
    CUDAPlugin::CudaGraph graph{CUDAPlugin::CreationContext{CUDA::Device{}, false}, {}};
    CUDAPlugin::Profiler profiler{false, graph};
    CUDAPlugin::InferenceRequestContext context{
        emptyTensor, emptyMapping, emptyTensor, emptyMapping, threadContext, token, profiler};
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
    for (std::size_t i = 0; i < out.size(); i++) EXPECT_EQ(out[i], correct[i]) << "at i == " << i;
    ASSERT_EQ(std::memcmp(out.data(), correct.data(), sizeof out), 0);
}

}  // namespace
