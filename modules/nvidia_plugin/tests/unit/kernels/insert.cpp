// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cuda/runtime.hpp>
#include <kernels/insert.hpp>

using namespace ov::nvidia_gpu;

class InsertKernelTest : public testing::Test {
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(InsertKernelTest, InsertKernel) {
    {
        constexpr size_t slice_arr_size = 2 * 1 * 4;
        constexpr size_t join_arr_size = 2 * 3 * 4;
        int32_t slice_arr[2][1][4] = {{{1, 42, 38, 17}}, {{1, 2, 18, 17}}};
        int32_t join_arr[2][3][4] = {{{0, 0, 0, 0}, {1, 42, 38, 17}, {0, 0, 0, 0}},
                                     {{0, 0, 0, 0}, {1, 2, 18, 17}, {0, 0, 0, 0}}};
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

        const auto& defaultStream = CUDA::DefaultStream::stream();
        auto src = defaultStream.malloc(sizeof(int32_t) * slice_arr_size);
        defaultStream.upload(src, slice_arr, sizeof(int32_t) * slice_arr_size);

        auto dst = defaultStream.malloc(sizeof(int32_t) * join_arr_size);
        auto immutableWorkbuffer = defaultStream.malloc(insert.getImmutableWorkbufferSize());
        insert.setImmutableWorkbuffer(immutableWorkbuffer.get());

        insert(stream.get(), src.get(), dst.get(), start);
        const auto hostSliceArr = std::make_unique<int32_t[]>(join_arr_size);
        defaultStream.download(hostSliceArr.get(), dst, sizeof(int32_t) * join_arr_size);
        ASSERT_TRUE(
            std::equal(hostSliceArr.get(), hostSliceArr.get() + join_arr_size, reinterpret_cast<int32_t*>(join_arr)));
    }
}

TEST_F(InsertKernelTest, InsertKernel2) {
    {
        float slice_arr[1][1][1] = {{{0.76131253}}};
        float join_arr[20][1][1] = {
            {{0.76131253}}, {{0}}, {{0}}, {{0}}, {{0}}, {{0}}, {{0}}, {{0}}, {{0}}, {{0}},
            {{0}},          {{0}}, {{0}}, {{0}}, {{0}}, {{0}}, {{0}}, {{0}}, {{0}}, {{0}},
        };
        constexpr size_t slice_arr_size = sizeof(slice_arr) / sizeof(float);
        constexpr size_t join_arr_size = sizeof(join_arr) / sizeof(float);
        kernel::Type_t element_type = kernel::Type_t::f32;
        kernel::Insert::Props props;
        props.old_shape[0] = 1;
        props.old_shape[1] = 1;
        props.old_shape[2] = 1;
        props.new_shape[0] = 20;
        props.new_shape[1] = 1;
        props.new_shape[2] = 1;
        props.axe = 0;
        const size_t start = 0;
        auto insert = kernel::Insert(element_type, props, CUDA::Device{}.props().maxThreadsPerBlock);
        CUDA::Stream stream{};
        const auto& defaultStream = CUDA::DefaultStream::stream();

        auto src = defaultStream.malloc(sizeof(float) * slice_arr_size);
        defaultStream.memset(src, 0, sizeof(float) * slice_arr_size);
        defaultStream.upload(src, slice_arr, sizeof(float) * slice_arr_size);

        auto dst = defaultStream.malloc(sizeof(float) * join_arr_size);
        defaultStream.memset(dst, 0, sizeof(float) * join_arr_size);

        auto immutableWorkbuffer = defaultStream.malloc(insert.getImmutableWorkbufferSize());
        insert.setImmutableWorkbuffer(immutableWorkbuffer.get());
        insert(stream.get(), src.get(), dst.get(), start);

        const auto hostSliceArr = std::make_unique<float[]>(join_arr_size);
        std::fill(hostSliceArr.get(), hostSliceArr.get() + join_arr_size, 0);
        defaultStream.download(hostSliceArr.get(), dst, sizeof(float) * join_arr_size);

        ASSERT_TRUE(
            std::equal(hostSliceArr.get(), hostSliceArr.get() + join_arr_size, reinterpret_cast<float*>(join_arr)));
    }
}
