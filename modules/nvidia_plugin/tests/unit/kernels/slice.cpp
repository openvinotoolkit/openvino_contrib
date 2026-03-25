// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cuda/runtime.hpp>
#include <kernels/details/tensor_helpers.hpp>
#include <kernels/slice.hpp>

using namespace ov::nvidia_gpu;

class SliceKernelTest : public testing::Test {
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(SliceKernelTest, Rank) {
    {
        kernel::Shape<size_t, 3> shape0{0, 0, 0};
        kernel::Shape<size_t, 3> shape1{1, 0, 0};
        kernel::Shape<size_t, 3> shape2{1, 2, 0};
        ASSERT_EQ(0, kernel::rank(shape0));
        ASSERT_EQ(1, kernel::rank(shape1));
        ASSERT_EQ(2, kernel::rank(shape2));
    }
    {
        kernel::Shape<size_t, 5> shape0{0, 0, 0, 0, 0};
        kernel::Shape<size_t, 5> shape1{1, 0, 0, 0, 0};
        kernel::Shape<size_t, 5> shape2{1, 2, 0, 0, 0};
        kernel::Shape<size_t, 5> shape3{1, 2, 5, 0, 0};
        kernel::Shape<size_t, 5> shape4{1, 2, 5, 8, 0};
        kernel::Shape<size_t, 5> shape5{1, 2, 5, 8, 10};
        ASSERT_EQ(0, kernel::rank(shape0));
        ASSERT_EQ(1, kernel::rank(shape1));
        ASSERT_EQ(2, kernel::rank(shape2));
        ASSERT_EQ(3, kernel::rank(shape3));
        ASSERT_EQ(4, kernel::rank(shape4));
        ASSERT_EQ(5, kernel::rank(shape5));
    }
}

TEST_F(SliceKernelTest, ShapeIndexes) {
    {
        kernel::Shape<size_t, 5> shape{1, 2, 3, 0, 0};
        kernel::Shape<size_t, 5> indexes{};
        kernel::shape_indices(shape, 10, indexes);
        kernel::Shape<size_t, 5> expectedIndexes{1, 1, 1, 0, 0};
        ASSERT_TRUE(std::equal(std::begin(indexes), std::end(indexes), std::begin(expectedIndexes)));
    }
    {
        kernel::Shape<size_t, 5> shape{1, 2, 3, 8, 0};
        kernel::Shape<size_t, 5> indexes{};
        kernel::shape_indices(shape, 10, indexes);
        kernel::Shape<size_t, 5> expectedIndexes{0, 0, 1, 2, 0};
        ASSERT_TRUE(std::equal(std::begin(indexes), std::end(indexes), std::begin(expectedIndexes)));
    }
}

TEST_F(SliceKernelTest, FlatAddress) {
    {
        kernel::Shape<size_t, 5> shape{1, 2, 3, 0, 0};
        kernel::Shape<size_t, 5> indexes{1, 1, 1, 0, 0};
        ASSERT_EQ(10, kernel::flat_address_by_shape(shape, indexes));
    }
    {
        kernel::Shape<size_t, 5> shape{1, 2, 3, 8, 0};
        kernel::Shape<size_t, 5> indexes{0, 0, 1, 2, 0};
        ASSERT_EQ(10, kernel::flat_address_by_shape(shape, indexes));
    }
}

TEST_F(SliceKernelTest, SliceKernel) {
    {
        constexpr size_t arr_size = 2 * 3 * 4;
        constexpr size_t slice_size = 2 * 1 * 4;
        int32_t arr[2][3][4] = {{{8, 2, 18, 17}, {1, 42, 38, 17}, {1, 2, 58, 17}},
                                {{1, 2, 28, 17}, {1, 2, 18, 17}, {51, 42, 8, 17}}};
        int32_t slice_arr[2][1][4] = {{{1, 42, 38, 17}}, {{1, 2, 18, 17}}};
        kernel::Type_t element_type = kernel::Type_t::i32;
        kernel::Slice::Props props;
        props.old_shape[0] = 2;
        props.old_shape[1] = 3;
        props.old_shape[2] = 4;
        props.new_shape[0] = 2;
        props.new_shape[1] = 1;
        props.new_shape[2] = 4;
        props.axe = 1;
        const size_t start = 1;
        auto slice = kernel::Slice(element_type, props, 1024);
        CUDA::Stream stream{};

        const auto& defaultStream = CUDA::DefaultStream::stream();
        auto src = defaultStream.malloc(sizeof(int32_t) * arr_size);
        defaultStream.upload(src, arr, sizeof(int32_t) * arr_size);

        auto dst = defaultStream.malloc(sizeof(int32_t) * slice_size);
        auto immutableWorkbuffer = defaultStream.malloc(slice.getImmutableWorkbufferSize());
        slice.setImmutableWorkbuffer(immutableWorkbuffer.get());

        slice(stream.get(), src.get(), dst.get(), start);
        const auto hostSliceArr = std::make_unique<int32_t[]>(slice_size);
        defaultStream.download(hostSliceArr.get(), dst, sizeof(int32_t) * slice_size);
        ASSERT_TRUE(
            std::equal(hostSliceArr.get(), hostSliceArr.get() + slice_size, reinterpret_cast<int32_t*>(slice_arr)));
    }
}
