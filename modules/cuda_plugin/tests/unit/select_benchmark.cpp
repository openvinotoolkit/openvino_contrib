// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <chrono>
#include <cuda_config.hpp>
#include <cuda_operation_registry.hpp>
#include <cuda_profiler.hpp>
#include <iomanip>
#include <ngraph/node.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/select.hpp>
#include <ops/parameter.hpp>
#include <random>

namespace {

using devptr_t = CUDA::DevicePointer<void*>;
using cdevptr_t = CUDA::DevicePointer<const void*>;

struct SelectTest : testing::Test {
    static constexpr auto kNumOfDim = 5u;
    using OffsetType = size_t;
    static constexpr auto kOffsetBufferSize = kNumOfDim * sizeof(OffsetType);

    const ov::Shape tensorShape{32, 256, 256};
    // const ov::Shape tensorShape{1, 8, 129};
    const size_t bufferLength = ov::shape_size(tensorShape);

    const size_t conditionBufferSize = bufferLength * sizeof(uint8_t);
    const size_t thenBufferSize = bufferLength * sizeof(float);
    const size_t elseBufferSize = bufferLength * sizeof(float);
    const size_t outputBufferSize = bufferLength * sizeof(float);

    CUDAPlugin::ThreadContext threadContext{CUDA::Device{}};
    CUDA::Allocation conditionAlloc = threadContext.stream().malloc(conditionBufferSize);
    CUDA::Allocation thenAlloc = threadContext.stream().malloc(thenBufferSize);
    CUDA::Allocation elseAlloc = threadContext.stream().malloc(elseBufferSize);
    CUDA::Allocation outputAlloc = threadContext.stream().malloc(outputBufferSize);
    std::vector<cdevptr_t> inputs{conditionAlloc, conditionAlloc, elseAlloc};
    std::vector<devptr_t> outputs{outputAlloc};
    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> emptyTensor;
    std::map<std::string, std::size_t> emptyMapping;
    std::function<std::shared_ptr<ov::op::v1::Select>()> create_node = [this]() {
        auto condition =
            std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::PartialShape{tensorShape});
        auto then_flow =
            std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{tensorShape});
        auto else_flow =
            std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{tensorShape});

        auto node =
            std::make_shared<ov::op::v1::Select>(condition->output(0), then_flow->output(0), else_flow->output(0));
        return node;
    };

    CUDAPlugin::OperationBase::Ptr operation = [this] {
        const bool optimizeOption = false;
        auto& registry = CUDAPlugin::OperationRegistry::getInstance();
        return registry.createOperation(CUDAPlugin::CreationContext{threadContext.device(), optimizeOption},
                                        create_node(),
                                        std::vector<CUDAPlugin::TensorID>{CUDAPlugin::TensorID{0u}},
                                        std::vector<CUDAPlugin::TensorID>{CUDAPlugin::TensorID{0u}});
    }();
};

namespace {
template <typename T>
void fillArrayWithRandomData(std::vector<T>& v) {
    std::random_device r_device;
    std::mt19937 mersenne_engine{r_device()};
    std::uniform_int_distribution<int> dist;
    std::function<T()> generator;
    if constexpr (std::is_same<T, uint8_t>::value) {
        dist = std::uniform_int_distribution<int>{0, 1};
        generator = [&dist, &mersenne_engine]() { return static_cast<T>(dist(mersenne_engine)); };
    } else {
        dist = std::uniform_int_distribution<int>{std::numeric_limits<int>::min(), std::numeric_limits<int>::max()};
        generator = [&dist, &mersenne_engine]() {
            return static_cast<T>(10) * dist(mersenne_engine) / std::numeric_limits<int>::max();
        };
    }
    std::generate(v.begin(), v.end(), generator);
}
}  // namespace

TEST_F(SelectTest, DISABLED_benchmark) {
    using microseconds = std::chrono::duration<double, std::micro>;
    constexpr int kNumAttempts = 20000;
    CUDAPlugin::CudaGraph graph{CUDAPlugin::CreationContext{CUDA::Device{}, false}, {}};
    CUDAPlugin::CancellationToken token{};
    CUDAPlugin::Profiler profiler{false, graph};
    CUDAPlugin::InferenceRequestContext context{emptyTensor, emptyMapping, emptyTensor, emptyMapping, threadContext, token, profiler};
    auto& stream = context.getThreadContext().stream();

    std::vector<uint8_t> conditions(bufferLength);
    fillArrayWithRandomData(conditions);
    stream.upload(conditionAlloc, conditions.data(), conditionBufferSize);

    std::vector<float> then_flow(bufferLength);
    fillArrayWithRandomData(then_flow);
    stream.upload(thenAlloc, then_flow.data(), thenBufferSize);

    std::vector<float> else_flow(bufferLength);
    fillArrayWithRandomData(else_flow);
    stream.upload(elseAlloc, else_flow.data(), elseBufferSize);
    stream.synchronize();

    CUDA::Allocation condOffsetAlloc = threadContext.stream().malloc(kOffsetBufferSize);
    CUDA::Allocation thenOffsetAlloc = threadContext.stream().malloc(kOffsetBufferSize);
    CUDA::Allocation elseOffsetAlloc = threadContext.stream().malloc(kOffsetBufferSize);
    CUDA::Allocation outputSizesAlloc = threadContext.stream().malloc(kOffsetBufferSize);
    CUDAPlugin::Workbuffers workbuffers{};
    workbuffers.immutable_buffers = {condOffsetAlloc, thenOffsetAlloc, elseOffsetAlloc, outputSizesAlloc};
    operation->InitSharedImmutableWorkbuffers({condOffsetAlloc, thenOffsetAlloc, elseOffsetAlloc, outputSizesAlloc});

    float elapsedTime = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < kNumAttempts; i++) {
        cudaEventRecord(start, stream.get());
        operation->Execute(context, inputs, outputs, workbuffers);
        cudaEventRecord(stop, stream.get());
        stream.synchronize();
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        elapsedTime += milliseconds;
    }
    std::cout << std::fixed << std::setfill('0') << "Sigmoid execution time: " << elapsedTime * 1000 / kNumAttempts
              << " microseconds\n";
}

}  // namespace
