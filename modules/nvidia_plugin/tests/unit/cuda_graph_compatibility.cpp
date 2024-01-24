// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <random>

#include "cuda/graph.hpp"
#include "cuda_compiled_model.hpp"
#include "cuda_operation_registry.hpp"
#include "cuda_profiler.hpp"
#include "cuda_runtime.h"
#include "cuda_simple_execution_delegator.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"

namespace {

using namespace ov::nvidia_gpu;
using DevPtr = CUDA::DevicePointer<void*>;
using CDevPtr = CUDA::DevicePointer<const void*>;

struct CudaGraphCompatibilityTest : testing::Test {
    template <typename T, typename C>
    static void generate(C& c) {
        std::random_device randDevice;
        std::mt19937 randEngine{randDevice()};
        std::uniform_int_distribution<int> dist{std::numeric_limits<int>::min(), std::numeric_limits<int>::max()};
        auto gen_input = [&dist, &randEngine]() {
            return static_cast<T>(10.f * dist(randEngine) / static_cast<float>(std::numeric_limits<int>::max()));
        };
        std::generate(c.begin(), c.end(), gen_input);
    }

    static bool checkAndRun(const OperationBase::Ptr& operation,
                            InferenceRequestContext& context,
                            OperationBase::Inputs inputs,
                            OperationBase::Outputs outputs,
                            const Workbuffers& workbuffers) {
        auto& stream = context.getThreadContext().stream();
        if (operation->GetCudaGraphCompatibility() == CudaGraphCompatibility::FULL) {
            stream.synchronize();
            CUDA::GraphCapture capture{stream};
            {
                auto scope = capture.getScope();
                operation->Execute(context, inputs, outputs, workbuffers);
            }
            CUDA::GraphExec exec{capture.getGraph()};
            stream.synchronize();
            std::cout << "--- Operation compatible. Running with CudaGraph ---\n";
            exec.launch(stream);
            return true;
        }
        std::cout << "--- Operation isn't compatible. Running without CudaGraph ---\n";
        operation->Execute(context, inputs, outputs, workbuffers);
        return false;
    }
};

struct ReluCudaGraphCompatibilityTest : CudaGraphCompatibilityTest {
    void run() {
        using ElementType = float;

        ov::Shape shape{1, 2, 3, 4};

        // Prepare environment
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::from<ElementType>(), shape);
        auto node = std::make_shared<ov::op::v0::Relu>(param);

        CUDA::Device device{};
        constexpr bool optimizeOption = false;
        CreationContext creationContext{device, optimizeOption};
        auto operation = OperationRegistry::getInstance().createOperation(
            creationContext, node, std::array{TensorID{0}}, std::array{TensorID{0}});

        const auto inSize = ov::shape_size(shape);
        const auto inSizeBytes = inSize * sizeof(ElementType);
        const auto outSize = ov::shape_size(shape);
        const auto outSizeBytes = outSize * sizeof(ElementType);

        ThreadContext threadContext{device};
        auto& stream = threadContext.stream();
        CUDA::Allocation inAlloc = stream.malloc(inSizeBytes);
        CUDA::Allocation outAlloc = stream.malloc(outSizeBytes);
        std::vector<CDevPtr> inputs{inAlloc};
        std::vector<DevPtr> outputs{outAlloc};

        CancellationToken token{};
        SimpleExecutionDelegator simpleExecutionDelegator{};
        std::vector<std::shared_ptr<ov::Tensor>> emptyTensor;
        std::map<std::string, std::size_t> emptyMapping;
        ov::nvidia_gpu::CudaGraphContext cudaGraphContext{};
        InferenceRequestContext context{emptyTensor,
                                        emptyMapping,
                                        emptyTensor,
                                        emptyMapping,
                                        threadContext,
                                        token,
                                        simpleExecutionDelegator,
                                        cudaGraphContext};

        // Generate input
        std::vector<ElementType> input(inSize);
        generate<ElementType>(input);

        // Upload input
        stream.upload(inAlloc, input.data(), inSizeBytes);

        Workbuffers workbuffers{};

        // Run with or without CudaGraph usage
        bool isCompatible = checkAndRun(operation, context, inputs, outputs, workbuffers);
        ASSERT_TRUE(isCompatible);

        // Download output
        std::vector<ElementType> output(outSize);
        stream.download(output.data(), outAlloc, outSizeBytes);
        stream.synchronize();

        // Calculate reference output
        std::vector<ElementType> refOutput(outSize);
        ASSERT_EQ(input.size(), refOutput.size());
        std::transform(input.cbegin(), input.cend(), refOutput.begin(), [](ElementType el) { return el > 0 ? el : 0; });

        // Validate result
        ASSERT_EQ(output.size(), refOutput.size());
        ASSERT_TRUE(std::equal(output.cbegin(), output.cend(), refOutput.cbegin()));
    }
};

TEST_F(ReluCudaGraphCompatibilityTest, Compatibile) { run(); }

struct ConcatCudaGraphCompatibilityTest : CudaGraphCompatibilityTest {
    void run() {
        using ElementType = float;

        // Prepare environment
        ov::Shape shape1{1, 2, 3, 4};
        ov::Shape shape2{2, 2, 3, 4};
        std::vector<std::shared_ptr<ov::op::v0::Parameter>> params{
            std::make_shared<ov::op::v0::Parameter>(ov::element::from<ElementType>(), shape1),
            std::make_shared<ov::op::v0::Parameter>(ov::element::from<ElementType>(), shape2)};

        constexpr int64_t axis = 0;
        auto node =
            std::make_shared<ov::op::v0::Concat>(ov::OutputVector{params[0]->output(0), params[1]->output(0)}, axis);

        CUDA::Device device{};
        constexpr bool optimizeOption = false;
        CreationContext creationContext{device, optimizeOption};
        auto operation = OperationRegistry::getInstance().createOperation(
            creationContext, node, std::array{TensorID{0}, TensorID{0}}, std::array{TensorID{0}});

        const auto inSize1 = ov::shape_size(shape1);
        const auto inSizeBytes1 = inSize1 * sizeof(ElementType);
        const auto inSize2 = ov::shape_size(shape2);
        const auto inSizeBytes2 = inSize2 * sizeof(ElementType);

        const auto& outputShape = node->get_output_shape(0);
        const auto outSize = ov::shape_size(outputShape);
        const auto outSizeBytes = outSize * sizeof(ElementType);

        ThreadContext threadContext{device};
        auto& stream = threadContext.stream();
        CUDA::Allocation inAlloc1 = stream.malloc(inSizeBytes1);
        CUDA::Allocation inAlloc2 = stream.malloc(inSizeBytes2);
        CUDA::Allocation outAlloc = stream.malloc(outSizeBytes);
        std::vector<CDevPtr> inputs{inAlloc1, inAlloc2};
        std::vector<DevPtr> outputs{outAlloc};

        CancellationToken token{};
        SimpleExecutionDelegator simpleExecutionDelegator{};
        std::vector<std::shared_ptr<ov::Tensor>> emptyTensor;
        std::map<std::string, std::size_t> emptyMapping;
        ov::nvidia_gpu::CudaGraphContext cudaGraphContext{};
        InferenceRequestContext context{emptyTensor,
                                        emptyMapping,
                                        emptyTensor,
                                        emptyMapping,
                                        threadContext,
                                        token,
                                        simpleExecutionDelegator,
                                        cudaGraphContext};

        // Generate inputs
        std::vector<ElementType> input1(inSize1);
        std::vector<ElementType> input2(inSize2);
        generate<ElementType>(input1);
        generate<ElementType>(input2);

        // Upload inputs
        stream.upload(inAlloc1, input1.data(), inSizeBytes1);
        stream.upload(inAlloc2, input2.data(), inSizeBytes2);

        WorkbufferRequest wbRequest{operation->GetWorkBufferRequest()};

        ASSERT_EQ(wbRequest.immutable_sizes.size(), 1);
        ASSERT_EQ(wbRequest.mutable_sizes.size(), 1);

        auto immutAlloc = stream.malloc(wbRequest.immutable_sizes[0]);
        auto mutAlloc = stream.malloc(wbRequest.mutable_sizes[0]);
        auto immutPtr = DevPtr{immutAlloc};
        auto mutPtr = DevPtr{mutAlloc};

        operation->InitSharedImmutableWorkbuffers({immutPtr});
        Workbuffers workbuffers{{immutPtr.cast<const void*>()}, {mutPtr}};

        // Run with or without CudaGraph usage
        bool isCompatible = checkAndRun(operation, context, inputs, outputs, workbuffers);
        ASSERT_FALSE(isCompatible);

        // Download output
        std::vector<ElementType> output(outSize);
        stream.download(output.data(), outAlloc, outSizeBytes);
        stream.synchronize();

        // Calculate reference output
        std::vector<ElementType> refOutput(outSize);
        ASSERT_EQ(input1.size() + input2.size(), refOutput.size());
        std::copy(input1.cbegin(), input1.cend(), refOutput.begin());
        std::copy(input2.cbegin(), input2.cend(), refOutput.begin() + input1.size());

        // Validate result
        ASSERT_EQ(output.size(), refOutput.size());
        ASSERT_TRUE(std::equal(output.cbegin(), output.cend(), refOutput.cbegin()));
    }
};

TEST_F(ConcatCudaGraphCompatibilityTest, NotCompatible) { run(); }

}  // namespace
