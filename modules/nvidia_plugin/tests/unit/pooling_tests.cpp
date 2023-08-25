// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cudnn_ops_infer.h>
#include <gtest/gtest.h>

#include <cuda_graph_context.hpp>
#include <cuda_op_buffers_extractor.hpp>
#include <cuda_operation_registry.hpp>
#include <cuda_eager_topology_runner.hpp>
#include <cuda_profiler.hpp>
#include <gsl/span>
#include <openvino/op/avg_pool.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/max_pool.hpp>
#include <ops/avgpool.hpp>
#include <ops/maxpool.hpp>
#include <type_traits>

using namespace ov::nvidia_gpu;

static const ov::Shape min_supported_pooling_shape{1, 1, 4, 4};
static const ov::Shape dummy_kernel{2, 2};
static const ov::Shape dummy_padding{1, 1};
static const ov::Strides dummy_strides{1, 1};
static const std::vector<TensorID> dummy_index{TensorID{0}};
static const auto default_data_type{ov::element::f32};
static const bool exclude_padding_from_pooling{true};

using NodePtr = std::shared_ptr<ov::Node>;
using ov::Shape;
using ov::Strides;

template <class NGraphPoolingNode>
std::shared_ptr<NGraphPoolingNode> build_ngraph_pooling_node(
    NodePtr input, const Strides& strides, const Shape& pads_begin, const Shape& pads_end, const Shape& kernel) {}

template <>
std::shared_ptr<ov::op::v1::MaxPool> build_ngraph_pooling_node(
    NodePtr input, const Strides& strides, const Shape& pads_begin, const Shape& pads_end, const Shape& kernel) {
    auto node = std::make_shared<ov::op::v1::MaxPool>(input->output(0), strides, pads_begin, pads_end, kernel);
    return node;
}

template <>
std::shared_ptr<ov::op::v1::AvgPool> build_ngraph_pooling_node(
    NodePtr input, const Strides& strides, const Shape& pads_begin, const Shape& pads_end, const Shape& kernel) {
    auto node = std::make_shared<ov::op::v1::AvgPool>(
        input->output(0), strides, pads_begin, pads_end, kernel, exclude_padding_from_pooling);
    return node;
}

template <class NGraphPoolingNode>
std::shared_ptr<NGraphPoolingNode> build_ngraph_pooling_dummy() {
    auto const_input_node = std::make_shared<ov::op::v0::Constant>(default_data_type, min_supported_pooling_shape);
    return build_ngraph_pooling_node<NGraphPoolingNode>(
        const_input_node, dummy_strides, dummy_padding, dummy_padding, dummy_kernel);
}

template <class NGraphPoolingNode, class CudnnPoolingNode>
class PoolingRegistryTest : public testing::Test {
    void SetUp() override {
        CUDA::Device device{};
        const bool optimizeOption = false;
        // A CUDA Plugin's Op registration requires at least 1 explicit
        // construction of the node.
        auto ngraph_node_dummy = build_ngraph_pooling_dummy<NGraphPoolingNode>();
        CudnnPoolingNode registration_dummy(CreationContext{device, optimizeOption},
                                            ngraph_node_dummy,
                                            std::vector<TensorID>{dummy_index},
                                            std::vector<TensorID>{dummy_index});
    }

    void TearDown() override {}
};

template <class NGraphPoolingNode>
struct PoolingTest : testing::Test {
    void initializeCudaBuffers(gsl::span<const float> in, gsl::span<const float> out) {
        allocs.push_back(threadContext.stream().malloc(in.size_bytes()));
        threadContext.stream().upload(allocs.back(), in.data(), in.size_bytes());
        inputs.push_back(allocs.back());
        allocs.push_back(threadContext.stream().malloc(out.size_bytes()));
        outputs.push_back(allocs.back());
    }
    void test(gsl::span<const float> input, std::vector<size_t> in_shape, gsl::span<const float> output) {
        CUDA::Device device{};
        const bool optimizeOption = false;
        CancellationToken token{};
        EagerTopologyRunner graph{CreationContext{CUDA::Device{}, false}, {}};
        Profiler profiler{false, graph};
        ov::nvidia_gpu::CudaGraphContext cudaGraphContext{};
        InferenceRequestContext context{
            empty_tensor, empty_mapping, empty_tensor, empty_mapping, threadContext, token, profiler, cudaGraphContext};
        auto& registry{OperationRegistry::getInstance()};
        auto const_input = std::make_shared<ov::op::v0::Constant>(ov::element::f32, Shape{in_shape});
        const size_t spatial_dims = in_shape.size() - 2;
        const Strides strides(spatial_dims, spatial_stride);
        const Shape pads_begin(spatial_dims, padding);
        const Shape pads_end(spatial_dims, padding);
        const Shape kernel(spatial_dims, kernel_side);

        auto node = build_ngraph_pooling_node<NGraphPoolingNode>(const_input, strides, pads_begin, pads_end, kernel);

        auto operation = registry.createOperation(CreationContext{device, optimizeOption}, node, inputIDs, outputIDs);
        initializeCudaBuffers(input, output);

        operation->Execute(context, {inputs}, {outputs}, {});
        threadContext.stream().synchronize();

        auto data = std::make_unique<float[]>(output.size());
        auto result = data.get();
        cudaMemcpy(result, outputs[0].get(), output.size_bytes(), cudaMemcpyDeviceToHost);
        ASSERT_EQ(0, memcmp(result, output.data(), output.size_bytes()));
    }
    const std::vector<TensorID> inputIDs{TensorID{0}};
    const std::vector<TensorID> outputIDs{TensorID{0}};
    const size_t padding{0};
    const size_t kernel_side{2};
    const size_t spatial_stride{2};
    ThreadContext threadContext{{}};
    std::vector<CUDA::Allocation> allocs;
    std::vector<CUDA::DevicePointer<const void*>> inputs;
    std::vector<CUDA::DevicePointer<void*>> outputs;
    std::shared_ptr<ov::Tensor> tensor;
    std::vector<std::shared_ptr<ov::Tensor>> tensors;
    std::map<std::string, std::size_t> tensors_mapping;
    std::vector<std::shared_ptr<ov::Tensor>> empty_tensor;
    std::map<std::string, std::size_t> empty_mapping;
};

class MaxPoolRegistryTest : public PoolingRegistryTest<ov::op::v1::MaxPool, MaxPoolOp> {};

TEST_F(MaxPoolRegistryTest, GetOperationBuilder_Available) {
    ASSERT_TRUE(OperationRegistry::getInstance().hasOperation(std::make_shared<ov::op::v1::MaxPool>()));
}

class MaxPoolTest : public PoolingTest<ov::op::v1::MaxPool> {};

TEST_F(MaxPoolTest, canExecuteOnFloat1DData) {
    // Input [4]
    // Kernel 2, strides [2], padding [[0,0]]
    // Result shape is [2]
    static const std::vector<size_t> in_shape = {1, 1, 4};
    static constexpr float in_data[] = {1.f, 2.f, 3.f, 4.f};
    static constexpr float result[] = {2.f, 4.f};
    EXPECT_NO_THROW(test(in_data, in_shape, result));
}

TEST_F(MaxPoolTest, canExecuteOnFloat2DData) {
    // Input [4,4]
    // Kernel 2x2, strides [2,2], padding [[0,0],[0,0]]
    // Result shape is [2, 2]
    static const std::vector<size_t> in_shape{1, 1, 4, 4};
    static constexpr float in_data[] = {
        -1.f,
        2.f,
        3.f,
        4.f,  //
        1.f,
        2.f,
        -3.f,
        4.f,  //
        -1.f,
        2.f,
        3.f,
        4.f,  //
        1.f,
        3.f,
        -3.f,
        5.f  //
    };
    static constexpr float result[] = {
        2.f,
        4.f,  //
        3.f,
        5.f  //
    };
    EXPECT_NO_THROW(test(in_data, in_shape, result));
}

TEST_F(MaxPoolTest, canExecuteOnFloat3DData) {
    // Input [4,4,4]
    // Kernel 2x2x2, strides [2, 2, 2], padding [[0,0],[0,0],[0,0]]
    // Result shape is [2, 2, 2]
    static const std::vector<size_t> in_shape{1, 1, 4, 4, 4};
    static constexpr float in_data[] = {
        -1.f,
        2.f,
        3.f,
        4.f,  //
        1.f,
        2.f,
        -3.f,
        4.f,  //
        -1.f,
        3.f,
        3.f,
        5.f,  //
        1.f,
        2.f,
        -3.f,
        4.f,  //
        //
        -1.f,
        2.f,
        3.f,
        4.f,  //
        1.f,
        2.f,
        -3.f,
        4.f,  //
        -1.f,
        2.f,
        3.f,
        4.f,  //
        1.f,
        2.f,
        -3.f,
        4.f,  //
        //
        -1.f,
        2.f,
        3.f,
        4.f,  //
        1.f,
        2.f,
        -3.f,
        4.f,  //
        -1.f,
        2.f,
        3.f,
        4.f,  //
        1.f,
        2.f,
        -3.f,
        4.f,  //
        //
        -1.f,
        2.f,
        3.f,
        4.f,  //
        1.f,
        2.f,
        -3.f,
        4.f,  //
        -1.f,
        3.f,
        3.f,
        5.f,  //
        1.f,
        2.f,
        -3.f,
        4.f  //
    };
    static constexpr float result[] = {
        2.f,
        4.f,  //
        3.f,
        5.f,  //
        //
        2.f,
        4.f,  //
        3.f,
        5.f  //
    };

    EXPECT_NO_THROW(test(in_data, in_shape, result));
}

class AvgPoolRegistryTest : public PoolingRegistryTest<ov::op::v1::AvgPool, AvgPoolOp> {};

TEST_F(AvgPoolRegistryTest, GetOperationBuilder_Available) {
    ASSERT_TRUE(OperationRegistry::getInstance().hasOperation(std::make_shared<ov::op::v1::AvgPool>()));
}

class AvgPoolTest : public PoolingTest<ov::op::v1::AvgPool> {};

TEST_F(AvgPoolTest, canExecuteOnFloat1DData) {
    // Input [4]
    // Kernel 2, strides [2], padding [[0,0]]
    // Result shape is [2]
    static const std::vector<size_t> in_shape = {1, 1, 4};
    static constexpr float in_data[] = {1.f, 3.f, 5.f, 3.f};
    static constexpr float result[] = {2.f, 4.f};
    EXPECT_NO_THROW(test(in_data, in_shape, result));
}

TEST_F(AvgPoolTest, canExecuteOnFloat2DData) {
    // Input [4,4]
    // Kernel 2x2, strides [2,2], padding [[0,0],[0,0]]
    // Result shape is [2, 2]
    static const std::vector<size_t> in_shape{1, 1, 4, 4};
    static constexpr float in_data[] = {
        0.f,
        4.f,
        4.f,
        4.f,  //
        0.f,
        4.f,
        4.f,
        4.f,  //
        4.f,
        2.f,
        0.f,
        2.f,  //
        2.f,
        4.f,
        0.f,
        2.f  //
    };
    static constexpr float result[] = {
        2.f,
        4.f,  //
        3.f,
        1.f  //
    };
    EXPECT_NO_THROW(test(in_data, in_shape, result));
}

TEST_F(AvgPoolTest, canExecuteOnFloat3DData) {
    // Input [4,4,4]
    // Kernel 2x2x2, strides [2, 2, 2], padding [[0,0],[0,0],[0,0]]
    // Result shape is [2, 2, 2]
    static const std::vector<size_t> in_shape{1, 1, 4, 4, 4};
    static constexpr float in_data[] = {
        0.f,
        4.f,
        4.f,
        4.f,  //
        0.f,
        4.f,
        4.f,
        4.f,  //
        4.f,
        2.f,
        0.f,
        2.f,  //
        2.f,
        4.f,
        0.f,
        2.f,  //
        //
        2.f,
        6.f,
        8.f,
        8.f,  //
        2.f,
        6.f,
        8.f,
        8.f,  //
        6.f,
        4.f,
        2.f,
        4.f,  //
        4.f,
        6.f,
        2.f,
        4.f,  //
        //
        -0.f,
        -4.f,
        -4.f,
        -4.f,  //
        -0.f,
        -4.f,
        -4.f,
        -4.f,  //
        -4.f,
        -2.f,
        -0.f,
        -2.f,  //
        -2.f,
        -4.f,
        -0.f,
        -2.f,  //
        //
        -2.f,
        -6.f,
        -8.f,
        -8.f,  //
        -2.f,
        -6.f,
        -8.f,
        -8.f,  //
        -6.f,
        -4.f,
        -2.f,
        -4.f,  //
        -4.f,
        -6.f,
        -2.f,
        -4.f  //
    };
    static constexpr float result[] = {
        3.f,
        6.f,  //
        4.f,
        2.f,  //
        //
        -3.f,
        -6.f,  //
        -4.f,
        -2.f  //
    };

    EXPECT_NO_THROW(test(in_data, in_shape, result));
}
