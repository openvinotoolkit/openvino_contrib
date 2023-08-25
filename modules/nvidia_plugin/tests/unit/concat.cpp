// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cuda_graph_context.hpp>
#include <cuda_op_buffers_extractor.hpp>
#include <cuda_operation_registry.hpp>
#include <cuda_eager_topology_runner.hpp>
#include <cuda_profiler.hpp>
#include <numeric>
#include <ops/concat.hpp>
#include <typeinfo>

using namespace ov::nvidia_gpu;
using devptr_t = DevicePointer<void*>;
using cdevptr_t = DevicePointer<const void*>;

template <typename T, std::size_t Size>
std::ostream& operator<<(std::ostream& out, gsl::span<T, Size> data) {
    const char* dlm = "";
    for (const auto& i : data) {
        out << dlm << i;
        dlm = ",";
    }
    return out;
}

struct ConcatTest : testing::Test {
    const std::array<std::array<ov::Shape, 3>, 2> shapes{
        std::array<ov::Shape, 3>{ov::Shape{2, 2}, ov::Shape{3, 2}, ov::Shape{4, 2}},
        std::array<ov::Shape, 3>{ov::Shape{2, 2}, ov::Shape{2, 3}, ov::Shape{2, 4}}};
    static constexpr float masters[2][18] = {{2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4},
                                             {2, 2, 3, 3, 3, 4, 4, 4, 4, 2, 2, 3, 3, 3, 4, 4, 4, 4}};
    void SetUp() override {}
    void TearDown() override {}
    void run(size_t axis) {
        CUDA::Device device{};
        const bool optimizeOption = false;
        allocate(axis);
        auto& registry{OperationRegistry::getInstance()};
        auto concatNode = std::make_shared<ov::op::v0::Concat>(params, axis);
        auto inputIDs = std::vector<TensorID>{TensorID{0}, TensorID{1}, TensorID{2}};
        auto outputIDs = std::vector<TensorID>{TensorID{0}};
        ASSERT_TRUE(registry.hasOperation(concatNode));
        auto operation =
            registry.createOperation(CreationContext{device, optimizeOption}, concatNode, inputIDs, outputIDs);
        ASSERT_TRUE(operation);
        auto concatOp = dynamic_cast<ConcatOp*>(operation.get());
        ASSERT_TRUE(concatOp);
        CancellationToken token{};
        EagerTopologyRunner graph{CreationContext{CUDA::Device{}, false}, {}};
        Profiler profiler{false, graph};
        ov::nvidia_gpu::CudaGraphContext cudaGraphContext{};
        InferenceRequestContext context{
            empty_tensor, empty_mapping, empty_tensor, empty_mapping, threadContext, token, profiler, cudaGraphContext};
        const auto& stream = threadContext.stream();
        std::vector<cdevptr_t> inputs{};
        std::vector<devptr_t> outputs{};
        std::vector<CUDA::Allocation> mem{};
        for (const auto& tensor : tensors) {
            mem.emplace_back(stream.malloc(tensor->get_byte_size()));
            inputs.emplace_back(cdevptr_t{mem.back().get()});
            stream.upload(inputs.back().as_mutable(), tensor->data(), tensor->get_byte_size());
        }
        mem.emplace_back(stream.malloc(output_size));
        outputs.emplace_back(devptr_t{mem.back().get()});
        auto wb_request = operation->GetWorkBufferRequest();
        ASSERT_EQ(wb_request.immutable_sizes.size(), 1);
        ASSERT_EQ(wb_request.mutable_sizes.size(), 1);
        auto& immutable_wb = mem.emplace_back(stream.malloc(wb_request.immutable_sizes[0]));
        auto& mutable_wb = mem.emplace_back(stream.malloc(wb_request.mutable_sizes[0]));
        operation->InitSharedImmutableWorkbuffers({immutable_wb});
        operation->Execute(context, inputs, outputs, {{immutable_wb}, {mutable_wb}});
        auto data = std::make_unique<float[]>(output_size / sizeof(float));
        stream.synchronize();
        stream.download(data.get(), outputs[0], output_size);
        stream.synchronize();
        ASSERT_EQ(0, memcmp(data.get(), masters[axis], output_size));
    }
    template <typename T>
    void fill_tensor(ov::Tensor& tensor, T value) const {
        T* data = tensor.data<T>();
        for (size_t i = 0; i < tensor.get_size(); i++)
            data[i] = static_cast<T>(value);
    }
    void allocate(size_t axis) {
        for (int i = 0; i < tensors.size(); i++) {
            tensors[i] = std::make_shared<ov::Tensor>(ov::element::f32, shapes[axis][i]);
            fill_tensor<float>(*tensors[i].get(), i + 2.0);
            output_size += tensors[i]->get_byte_size();
            params.emplace_back(
                std::make_shared<ov::op::v0::Parameter>(ov::element::Type{ov::element::Type_t::f32}, shapes[axis][i]));
        }
    }
    ThreadContext threadContext{{}};
    std::array<std::shared_ptr<ov::Tensor>, 3> tensors;
    std::vector<std::shared_ptr<ov::Tensor>> empty_tensor;
    std::map<std::string, std::size_t> empty_mapping;
    size_t output_size{};
    ov::OutputVector params{};
};

TEST_F(ConcatTest, axis_0) { run(0); }

TEST_F(ConcatTest, axis_1) { run(1); }
