// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <chrono>
#include <cuda_config.hpp>
#include <cuda_operation_registry.hpp>
#include <iomanip>
#include <ngraph/node.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/parameter.hpp>
#include <ngraph/op/strided_slice.hpp>
#include <ops/parameter.hpp>
#include <random>

namespace {

using devptr_t = InferenceEngine::gpu::DevicePointer<void*>;
using cdevptr_t = InferenceEngine::gpu::DevicePointer<const void*>;

struct StridedSliceTest : testing::Test {
    using ElementType = float;
    using AuxilaryElementType = int64_t;

    const ngraph::Shape inputTensorShape{3, 256, 256};
    const int inputBufferLength = ngraph::shape_size(inputTensorShape);
    const size_t inputBufferSize = inputBufferLength * sizeof(ElementType);

    const ngraph::Shape outputTensorShape{3, 128, 128};
    const size_t ouputBufferSize = ngraph::shape_size(outputTensorShape) * sizeof(ElementType);

    const ngraph::Shape constTensorShape{3};
    const size_t constantTensorSize = ngraph::shape_size(constTensorShape) * sizeof(AuxilaryElementType);

    CUDA::ThreadContext threadContext{{}};
    CUDA::Allocation inAlloc = threadContext.stream().malloc(inputBufferSize);
    CUDA::Allocation inBeginAlloc = threadContext.stream().malloc(constantTensorSize);
    CUDA::Allocation inEndAlloc = threadContext.stream().malloc(constantTensorSize);
    CUDA::Allocation inStrideAlloc = threadContext.stream().malloc(constantTensorSize);
    CUDA::Allocation srcShapeSizesAlloc = threadContext.stream().malloc(constantTensorSize);
    CUDA::Allocation dstShapeSizesAlloc = threadContext.stream().malloc(constantTensorSize);
    CUDA::Allocation outAlloc = threadContext.stream().malloc(ouputBufferSize);
    std::vector<cdevptr_t> inputs{inAlloc, inBeginAlloc, inEndAlloc, inStrideAlloc};
    std::vector<devptr_t> outputs{outAlloc};
    InferenceEngine::BlobMap empty;
    std::function<std::shared_ptr<ngraph::op::v1::StridedSlice>()> create_node = [this]() {
        auto param =
            std::make_shared<ngraph::op::v0::Parameter>(ngraph::element::f32, ngraph::PartialShape{inputTensorShape});
        std::vector<int64_t> shapeBegin{0, 64, -65};
        auto begin_input =
            std::make_shared<ngraph::op::v0::Constant>(ngraph::element::i64, ngraph::Shape{3}, shapeBegin);
        std::vector<int64_t> shapeEnd{3, 192, -193};
        auto end_input = std::make_shared<ngraph::op::v0::Constant>(ngraph::element::i64, ngraph::Shape{3}, shapeEnd);
        std::vector<int64_t> stride{1, 1, -1};
        auto stride_input = std::make_shared<ngraph::op::v0::Constant>(ngraph::element::i64, ngraph::Shape{3}, stride);
        auto node = std::make_shared<ngraph::op::v1::StridedSlice>(param->output(0),
                                                                   begin_input->output(0),
                                                                   end_input->output(0),
                                                                   stride_input->output(0),
                                                                   std::vector<int64_t>{},
                                                                   std::vector<int64_t>{});
        return node;
    };

    CUDAPlugin::OperationBase::Ptr operation = [this] {
        const bool optimizeOption = false;
        auto& registry = CUDAPlugin::OperationRegistry::getInstance();
        return registry.createOperation(CUDA::CreationContext{threadContext.device(), optimizeOption},
                                        create_node(),
                                        std::vector<CUDAPlugin::TensorID>{CUDAPlugin::TensorID{0u}},
                                        std::vector<CUDAPlugin::TensorID>{CUDAPlugin::TensorID{0u}});
    }();
};

TEST_F(StridedSliceTest, DISABLED_benchmark) {
    using microseconds = std::chrono::duration<double, std::micro>;
    constexpr int kNumAttempts = 20000;
    InferenceEngine::gpu::InferenceRequestContext context{empty, empty, threadContext};
    auto& stream = context.getThreadContext().stream();
    std::vector<ElementType> in(inputBufferLength);
    std::random_device r_device;
    std::mt19937 mersenne_engine{r_device()};
    std::uniform_int_distribution<int> dist{std::numeric_limits<int>::min(), std::numeric_limits<int>::max()};
    auto gen = [&dist, &mersenne_engine]() { return 10.f * dist(mersenne_engine) / std::numeric_limits<int>::max(); };
    std::generate(in.begin(), in.end(), gen);
    stream.upload(inAlloc, in.data(), inputBufferSize);

    auto wb_request = operation->GetWorkBufferRequest();
    ASSERT_EQ(wb_request.immutable_sizes.size(), 5);

    CUDAPlugin::Workbuffers workbuffers;
    workbuffers.immutable_buffers = {srcShapeSizesAlloc, dstShapeSizesAlloc, inBeginAlloc, inEndAlloc, inStrideAlloc};
    operation->InitSharedImmutableWorkbuffers(
        {srcShapeSizesAlloc, dstShapeSizesAlloc, inBeginAlloc, inEndAlloc, inStrideAlloc});
    stream.synchronize();
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < kNumAttempts; i++) {
        operation->Execute(context, inputs, outputs, workbuffers);
        stream.synchronize();
    }
    auto end = std::chrono::steady_clock::now();
    microseconds average_exec_time = (end - start) / kNumAttempts;
    std::cout << std::fixed << std::setfill('0') << "Strided slice execution time: " << average_exec_time.count()
              << " microseconds\n";
}

}  // namespace
