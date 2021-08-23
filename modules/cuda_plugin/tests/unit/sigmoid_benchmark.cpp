// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cuda_config.hpp>
#include <cuda_operation_registry.hpp>
#include <ngraph/node.hpp>
#include <ngraph/op/parameter.hpp>
#include <ngraph/op/sigmoid.hpp>
#include <ops/parameter.hpp>
#include <typeinfo>
#include <array>
#include <random>
#include <chrono>
#include <iomanip>

namespace {

using devptr_t = InferenceEngine::gpu::DevicePointer<void*>;
using cdevptr_t = InferenceEngine::gpu::DevicePointer<const void*>;

struct SigmoidTest: testing::Test {
    using ElementType = float;
    static constexpr int length = 1024;
    static constexpr size_t size = length * sizeof(ElementType);
    CUDA::ThreadContext threadContext { { } };
    CUDA::Allocation inAlloc = threadContext.stream().malloc(size);
    CUDA::Allocation outAlloc = threadContext.stream().malloc(size);
    std::vector<cdevptr_t> inputs{inAlloc};
    std::vector<devptr_t> outputs{outAlloc};
    InferenceEngine::BlobMap empty;
    CUDAPlugin::OperationBase::Ptr operation =
            [this] {
                const bool optimizeOption = false;
                auto param = std::make_shared<ngraph::op::v0::Parameter>(
                        ngraph::element::f32, ngraph::PartialShape {length});
                auto node = std::make_shared<ngraph::op::v0::Sigmoid>(param->output(0));
                auto& registry = CUDAPlugin::OperationRegistry::getInstance();
                auto op = registry.createOperation(
                        CUDA::CreationContext {threadContext.device(), optimizeOption},
                        node, std::vector<CUDAPlugin::TensorID> {0u}, std::vector<CUDAPlugin::TensorID> {0u});
                return op;
            }();
};


TEST_F(SigmoidTest, DISABLED_benchmark) {
    using microseconds = std::chrono::duration<double, std::micro>;
    constexpr int kNumAttempts = 20;
    InferenceEngine::gpu::InferenceRequestContext context { empty, empty,
            threadContext };
    auto& stream = context.getThreadContext().stream();
    std::array<ElementType, length> in;
    std::random_device r_device;
    std::mt19937 mersenne_engine {r_device()};
    std::uniform_int_distribution<int> dist {std::numeric_limits<int>::min(),
        std::numeric_limits<int>::max()};
    auto gen = [&dist, &mersenne_engine](){
        return 10.f * dist(mersenne_engine) / std::numeric_limits<int>::max();
    };
    std::generate(in.begin(), in.end(), gen);
    stream.upload(inAlloc, in.data(), size);
    CUDAPlugin::Workbuffers workbuffers {};
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < kNumAttempts; i++) {
        operation->Execute(context, inputs, outputs, workbuffers);
        stream.synchronize();
    }
    auto end = std::chrono::steady_clock::now();
    microseconds average_exec_time = (end - start) / kNumAttempts;
    std::cout << std::fixed <<
            std::setfill('0') << "Sigmoid execution time: " << average_exec_time.count() << " microseconds\n";
}

} // namespace
