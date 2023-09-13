// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cuda/device_pointers.hpp>
#include <cuda_config.hpp>
#include <cuda_graph_context.hpp>
#include <cuda_operation_registry.hpp>
#include <cuda_simple_execution_delegator.hpp>
#include <iomanip>
#include <openvino/op/logical_not.hpp>
#include <openvino/op/parameter.hpp>
#include <ops/parameter.hpp>
#include <random>
#include <typeinfo>
#include <vector>

namespace {

using devptr_t = CUDA::DevicePointer<void*>;
using cdevptr_t = CUDA::DevicePointer<const void*>;

struct LogicalNotBenchmark : testing::Test {
    using TensorID = ov::nvidia_gpu::TensorID;
    using ElementType = std::uint8_t;
    static constexpr int length = 10 * 1024;
    static constexpr size_t size = length * sizeof(ElementType);
    ov::nvidia_gpu::ThreadContext threadContext{{}};
    CUDA::Allocation in_alloc = threadContext.stream().malloc(size);
    CUDA::Allocation out_alloc = threadContext.stream().malloc(size);
    std::vector<cdevptr_t> inputs{in_alloc};
    std::vector<devptr_t> outputs{out_alloc};
    std::vector<std::shared_ptr<ov::Tensor>> emptyTensor;
    std::map<std::string, std::size_t> emptyMapping;
    ov::nvidia_gpu::OperationBase::Ptr operation = [this] {
        const bool optimizeOption = false;
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{length});
        auto node = std::make_shared<ov::op::v1::LogicalNot>(param->output(0));
        auto& registry = ov::nvidia_gpu::OperationRegistry::getInstance();
        auto op = registry.createOperation(ov::nvidia_gpu::CreationContext{threadContext.device(), optimizeOption},
                                           node,
                                           std::vector<TensorID>{TensorID{0u}},
                                           std::vector<TensorID>{TensorID{0u}});
        return op;
    }();
};

TEST_F(LogicalNotBenchmark, DISABLED_benchmark) {
    constexpr int kNumAttempts = 20;
    ov::nvidia_gpu::CancellationToken token{};
    ov::nvidia_gpu::SimpleExecutionDelegator simpleExecutionDelegator{};
    ov::nvidia_gpu::CudaGraphContext cudaGraphContext{};
    ov::nvidia_gpu::InferenceRequestContext context{emptyTensor,
                                                    emptyMapping,
                                                    emptyTensor,
                                                    emptyMapping,
                                                    threadContext,
                                                    token,
                                                    simpleExecutionDelegator,
                                                    cudaGraphContext};
    auto& stream = context.getThreadContext().stream();
    std::vector<ElementType> in(length);
    std::random_device r_device;
    std::mt19937 mersenne_engine{r_device()};
    std::uniform_int_distribution<> dist{std::numeric_limits<std::uint8_t>::min(),
                                         std::numeric_limits<std::uint8_t>::max()};
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    std::generate(in.begin(), in.end(), gen);
    stream.upload(in_alloc, in.data(), size);
    ov::nvidia_gpu::Workbuffers workbuffers{};
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
