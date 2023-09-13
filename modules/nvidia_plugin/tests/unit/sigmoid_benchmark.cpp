// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cuda_config.hpp>
#include <cuda_graph_context.hpp>
#include <cuda_op_buffers_extractor.hpp>
#include <cuda_operation_registry.hpp>
#include <cuda_simple_execution_delegator.hpp>
#include <iomanip>
#include <openvino/op/parameter.hpp>
#include <openvino/op/sigmoid.hpp>
#include <ops/parameter.hpp>
#include <random>
#include <typeinfo>

namespace {

using devptr_t = CUDA::DevicePointer<void*>;
using cdevptr_t = CUDA::DevicePointer<const void*>;

struct SigmoidTest : testing::Test {
    using TensorID = ov::nvidia_gpu::TensorID;
    using ElementType = float;
    static constexpr int length = 1024;
    static constexpr size_t size = length * sizeof(ElementType);
    ov::nvidia_gpu::ThreadContext threadContext{{}};
    CUDA::Allocation inAlloc = threadContext.stream().malloc(size);
    CUDA::Allocation outAlloc = threadContext.stream().malloc(size);
    std::vector<cdevptr_t> inputs{inAlloc};
    std::vector<devptr_t> outputs{outAlloc};
    std::vector<std::shared_ptr<ov::Tensor>> emptyTensor;
    std::map<std::string, std::size_t> emptyMapping;
    ov::nvidia_gpu::OperationBase::Ptr operation = [this] {
        const bool optimizeOption = false;
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{length});
        auto node = std::make_shared<ov::op::v0::Sigmoid>(param->output(0));
        auto& registry = ov::nvidia_gpu::OperationRegistry::getInstance();
        auto op = registry.createOperation(ov::nvidia_gpu::CreationContext{threadContext.device(), optimizeOption},
                                           node,
                                           std::vector<TensorID>{TensorID{0u}},
                                           std::vector<TensorID>{TensorID{0u}});
        return op;
    }();
};

TEST_F(SigmoidTest, DISABLED_benchmark) {
    using microseconds = std::chrono::duration<double, std::micro>;
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
    std::array<ElementType, length> in;
    std::random_device r_device;
    std::mt19937 mersenne_engine{r_device()};
    std::uniform_int_distribution<int> dist{std::numeric_limits<int>::min(), std::numeric_limits<int>::max()};
    auto gen = [&dist, &mersenne_engine]() { return 10.f * dist(mersenne_engine) / std::numeric_limits<int>::max(); };
    std::generate(in.begin(), in.end(), gen);
    stream.upload(inAlloc, in.data(), size);
    ov::nvidia_gpu::Workbuffers workbuffers{};
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < kNumAttempts; i++) {
        operation->Execute(context, inputs, outputs, workbuffers);
        stream.synchronize();
    }
    auto end = std::chrono::steady_clock::now();
    microseconds average_exec_time = (end - start) / kNumAttempts;
    std::cout << std::fixed << std::setfill('0') << "Sigmoid execution time: " << average_exec_time.count()
              << " microseconds\n";
}

}  // namespace
