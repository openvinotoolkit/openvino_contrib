// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cuda/device_pointers.hpp>
#include <cuda_config.hpp>
#include <cuda_operation_registry.hpp>
#include <cuda_profiler.hpp>
#include <iomanip>
#include <ngraph/node.hpp>
#include <ngraph/op/not.hpp>
#include <ngraph/op/parameter.hpp>
#include <ops/parameter.hpp>
#include <random>
#include <typeinfo>
#include <vector>

namespace {

using devptr_t = CUDA::DevicePointer<void*>;
using cdevptr_t = CUDA::DevicePointer<const void*>;

struct LogicalNotBenchmark : testing::Test {
    using TensorID = CUDAPlugin::TensorID;
    using ElementType = std::uint8_t;
    static constexpr int length = 10 * 1024;
    static constexpr size_t size = length * sizeof(ElementType);
    CUDAPlugin::ThreadContext threadContext{{}};
    CUDA::Allocation in_alloc = threadContext.stream().malloc(size);
    CUDA::Allocation out_alloc = threadContext.stream().malloc(size);
    std::vector<cdevptr_t> inputs{in_alloc};
    std::vector<devptr_t> outputs{out_alloc};
    InferenceEngine::BlobMap empty;
    CUDAPlugin::OperationBase::Ptr operation = [this] {
        const bool optimizeOption = false;
        auto param = std::make_shared<ngraph::op::v0::Parameter>(ngraph::element::f32, ngraph::PartialShape{length});
        auto node = std::make_shared<ngraph::op::v1::LogicalNot>(param->output(0));
        auto& registry = CUDAPlugin::OperationRegistry::getInstance();
        auto op = registry.createOperation(CUDAPlugin::CreationContext{threadContext.device(), optimizeOption},
                                           node,
                                           std::vector<TensorID>{TensorID{0u}},
                                           std::vector<TensorID>{TensorID{0u}});
        return op;
    }();
};

TEST_F(LogicalNotBenchmark, DISABLED_benchmark) {
    constexpr int kNumAttempts = 20;
    CUDAPlugin::CancellationToken token{};
    CUDAPlugin::CudaGraph graph{CUDAPlugin::CreationContext{CUDA::Device{}, false}, {}};
    CUDAPlugin::Profiler profiler{false, graph};
    CUDAPlugin::InferenceRequestContext context{empty, empty, threadContext, token, profiler};
    auto& stream = context.getThreadContext().stream();
    std::vector<ElementType> in(length);
    std::random_device r_device;
    std::mt19937 mersenne_engine{r_device()};
    std::uniform_int_distribution<std::uint8_t> dist{std::numeric_limits<std::uint8_t>::min(),
                                                     std::numeric_limits<std::uint8_t>::max()};
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine) / std::numeric_limits<std::uint8_t>::max(); };
    std::generate(in.begin(), in.end(), gen);
    stream.upload(in_alloc, in.data(), size);
    CUDAPlugin::Workbuffers workbuffers{};
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float totalExecTime{};
    for (int i = 0; i < kNumAttempts; i++) {
        cudaEventRecord(start, stream.get());
        operation->Execute(context, inputs, outputs, workbuffers);
        cudaEventRecord(stop, stream.get());
        cudaEventSynchronize(stop);
        float execTime{};
        cudaEventElapsedTime(&execTime, start, stop);
        totalExecTime += execTime;
    }
    stream.synchronize();
    std::cout << std::fixed << std::setfill('0') << "LogicalNot execution time: " << totalExecTime * 1000 / kNumAttempts
              << " microseconds\n";
}

}  // namespace
