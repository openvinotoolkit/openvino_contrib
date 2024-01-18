// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cuda/graph.hpp>
#include <cuda/runtime.hpp>
#include <kernels/insert.hpp>

using namespace ov::nvidia_gpu;

class KernelNodeTest : public testing::Test {};

TEST_F(KernelNodeTest, InsertKernel) {
    constexpr size_t in_size = 2 * 1 * 4;
    constexpr size_t out_size = 2 * 3 * 4;
    kernel::Type_t element_type = kernel::Type_t::i32;
    kernel::Insert::Props props;
    props.old_shape[0] = 2;
    props.old_shape[1] = 1;
    props.old_shape[2] = 4;
    props.new_shape[0] = 2;
    props.new_shape[1] = 3;
    props.new_shape[2] = 4;
    props.axe = 1;
    const size_t start = 1;
    auto insert = kernel::Insert(element_type, props, CUDA::Device{}.props().maxThreadsPerBlock);

    CUDA::Stream stream{};
    auto iwb = stream.malloc(insert.getImmutableWorkbufferSize());
    insert.setImmutableWorkbuffer(iwb.get());

    // Regular kernel + graph with KernelNode
    const int32_t in_arr1[2][1][4] = {{{1, 42, 38, 17}}, {{1, 2, 18, 17}}};
    auto src1 = stream.malloc(sizeof(int32_t) * in_size);
    auto dst1 = stream.malloc(sizeof(int32_t) * out_size);
    auto host_out_arr1 = std::make_unique<int32_t[]>(out_size);

    stream.upload(src1, in_arr1, sizeof(int32_t) * in_size);
    insert(stream.get(), src1.get(), dst1.get(), start);
    stream.download(host_out_arr1.get(), dst1, sizeof(int32_t) * out_size);

    auto dst1_graph = stream.malloc(sizeof(int32_t) * out_size);

    std::optional<CUDA::KernelNode> kernel_node;
    CUDA::GraphCapture capture{stream};
    {
        auto scope = capture.getScope();
        CUDA::CaptureInfo captureInfo{stream};
        kernel_node.emplace(captureInfo.addKernelNode(insert.getKernel(),
                                                      insert.getNumBlocks(),
                                                      insert.getThreadsPerBlock(),
                                                      insert.getPropsPtr(),
                                                      start,
                                                      insert.getSize(),
                                                      src1.get(),
                                                      dst1_graph.get()));
    }
    CUDA::GraphExec graph_exec{capture.getGraph()};
    graph_exec.launch(stream);

    auto host_out_arr1_graph = std::make_unique<int32_t[]>(out_size);
    stream.download(host_out_arr1_graph.get(), dst1_graph, sizeof(int32_t) * out_size);
    stream.synchronize();

    ASSERT_TRUE(std::equal(host_out_arr1.get(), host_out_arr1.get() + out_size, host_out_arr1_graph.get()));

    // Regular kernel + updated graph with KernelNode
    const int32_t in_arr2[2][1][4] = {{{31, 2, 8, 10}}, {{20, 12, 1, 7}}};

    auto src2 = stream.malloc(sizeof(int32_t) * in_size);
    auto dst2 = stream.malloc(sizeof(int32_t) * out_size);
    auto host_out_arr2 = std::make_unique<int32_t[]>(out_size);

    stream.upload(src2, in_arr2, sizeof(int32_t) * in_size);
    insert(stream.get(), src2.get(), dst2.get(), start);
    stream.download(host_out_arr2.get(), dst2, sizeof(int32_t) * out_size);

    auto dst2_graph = stream.malloc(sizeof(int32_t) * out_size);
    auto host_out_arr2_graph = std::make_unique<int32_t[]>(out_size);

    kernel_node.value().update_params(
        graph_exec, insert.getPropsPtr(), start, insert.getSize(), src2.get(), dst2_graph.get());
    graph_exec.launch(stream);
    stream.download(host_out_arr2_graph.get(), dst2_graph, sizeof(int32_t) * out_size);
    stream.synchronize();

    ASSERT_TRUE(std::equal(host_out_arr2.get(), host_out_arr2.get() + out_size, host_out_arr2_graph.get()));
}

class TransferNodeTest : public testing::Test {};

TEST_F(TransferNodeTest, Transfer) {
    constexpr size_t size = 2 * 1 * 4;
    const int32_t host_arr1[2][1][4] = {{{1, 42, 38, 17}}, {{1, 2, 18, 17}}};
    CUDA::Stream stream{};

    // Transfer with graph and TransferNode
    auto src1 = stream.malloc(sizeof(int32_t) * size);
    auto dst1 = stream.malloc(sizeof(int32_t) * size);
    const auto host_out_arr1 = std::make_unique<int32_t[]>(size);

    stream.upload(src1, host_arr1, sizeof(int32_t) * size);

    std::optional<CUDA::TransferNode> transfer_node;
    CUDA::GraphCapture capture{stream};
    {
        auto scope = capture.getScope();
        CUDA::CaptureInfo captureInfo{stream};
        transfer_node.emplace(captureInfo.addTransferNode(dst1, src1, sizeof(int32_t) * size));
    }
    CUDA::GraphExec graph_exec{capture.getGraph()};
    graph_exec.launch(stream);

    stream.download(host_out_arr1.get(), dst1, sizeof(int32_t) * size);
    stream.synchronize();

    const auto* src_ptr1 = static_cast<const int32_t*>(static_cast<const void*>(host_arr1));
    ASSERT_TRUE(std::equal(src_ptr1, src_ptr1 + size, host_out_arr1.get()));

    // Transfer with graph and updated TransferNode
    const int32_t host_arr2[2][1][4] = {{{31, 2, 8, 10}}, {{20, 12, 1, 7}}};

    auto src2 = stream.malloc(sizeof(int32_t) * size);
    auto dst2 = stream.malloc(sizeof(int32_t) * size);
    auto host_out_arr2 = std::make_unique<int32_t[]>(size);

    stream.upload(src2, host_arr2, sizeof(int32_t) * size);

    transfer_node.value().update_ptrs(graph_exec, dst2, src2);
    graph_exec.launch(stream);

    stream.download(host_out_arr2.get(), dst2, sizeof(int32_t) * size);
    stream.synchronize();

    const auto* src_ptr2 = static_cast<const int32_t*>(static_cast<const void*>(host_arr2));
    ASSERT_TRUE(std::equal(src_ptr2, src_ptr2 + size, host_out_arr2.get()));
}
