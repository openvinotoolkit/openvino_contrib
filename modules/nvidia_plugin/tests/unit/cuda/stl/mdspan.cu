// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cuda/runtime.hpp>
#include <cuda/stl/mdspan.cuh>

using namespace ov::nvidia_gpu;

class MDSpanTest : public testing::Test {
    void SetUp() override {}

    void TearDown() override {}

public:
    template <typename TMDSpan, typename... TIndexes>
    auto getOffset(TMDSpan span, TIndexes... idx) {
        return span.template get_offset<false>(idx...);
    }
    template <typename TMDSpan, typename... TIndexes>
    auto getOffsetAt(TMDSpan span, TIndexes... idx) {
        return span.template get_offset<true>(idx...);
    }
};

namespace {

__global__ void verify_extents(CUDA::MDSpan<int, CUDA::DExtents<2>> span) {
    assert(span.extent(0) == 100);
    assert(span.extent(1) == 512);
    assert(gridDim.x == 100);
    assert(blockDim.x == 512);
}

__global__ void assign(CUDA::MDSpan<int, CUDA::DExtents<2>> span) {
    assert(span.extent(0) == 100);
    assert(span.extent(1) == 512);
    assert(gridDim.x == 100);
    assert(blockDim.x == 512);
    const size_t y = blockIdx.x;
    const size_t x = threadIdx.x;
    span(y, x) = 1000 * y + x;
}

__global__ void verify(CUDA::MDSpan<int, CUDA::DExtents<2>> span) {
    assert(span.extent(0) == 100);
    assert(span.extent(1) == 512);
    assert(gridDim.x == 100);
    assert(blockDim.x == 512);
    const size_t y = blockIdx.x;
    const size_t x = threadIdx.x;
    assert(span(y, x) == 1000 * y + x);
    assert(*(span.data() + 512 * y + x) == 1000 * y + x);
}

}  // namespace

TEST_F(MDSpanTest, MDSpan_VerifyExtents) {
    using MDSpanTestType = CUDA::MDSpan<int, CUDA::DExtents<2>>;

    CUDA::Stream stream{};
    auto src = stream.malloc(MDSpanTestType::size_of(100, 512));
    auto span = MDSpanTestType(src.get(), 100, 512);
    verify_extents<<<100, 512, 0, stream.get()>>>(span);
    ASSERT_NO_THROW(stream.synchronize());
}

TEST_F(MDSpanTest, MDSpan_VerifyOffsets) {
    using MDSpanTestType = CUDA::MDSpan<int, CUDA::DExtents<2>>;

    CUDA::Stream stream{};
    auto src = stream.malloc(MDSpanTestType::size_of(100, 512));
    auto span = MDSpanTestType(src.get(), 100, 512);
    ASSERT_EQ(getOffset(span, 0, 0), 0);
    ASSERT_EQ(getOffset(span, 0, 1), 1);
    ASSERT_EQ(getOffset(span, 1, 1), 512 + 1);
    ASSERT_EQ(getOffset(span, 2, 1), 2 * 512 + 1);
    ASSERT_EQ(getOffset(span, 32, 0), 32 * 512);
    ASSERT_EQ(getOffset(span, 32, 10), 32 * 512 + 10);
    for (size_t yi = 0; yi < 100; ++yi) {
        for (size_t xi = 0; xi < 512; ++xi) {
            ASSERT_EQ(getOffset(span, yi, xi), 512 * yi + xi);
        }
    }
}

TEST_F(MDSpanTest, MDSpan_VerifyOffsetsAt_0) {
    using MDSpanTestType = CUDA::MDSpan<int, CUDA::DExtents<2>>;

    CUDA::Stream stream{};
    auto src = stream.malloc(MDSpanTestType::size_of(100, 512));
    auto span = MDSpanTestType(src.get(), 100, 512);
    ASSERT_EQ(getOffsetAt(span, 0, 0), 0);
    ASSERT_EQ(getOffsetAt(span, 0, 1), 1);
    ASSERT_EQ(getOffsetAt(span, 1, 1), 512 + 1);
    ASSERT_EQ(getOffsetAt(span, 2, 1), 2 * 512 + 1);
    ASSERT_EQ(getOffsetAt(span, 32, 0), 32 * 512);
    ASSERT_EQ(getOffsetAt(span, 32, 0), 32 * 512);
#ifndef NDEBUG
    ASSERT_DEATH(getOffsetAt(span, 99, 512), "idx is out of range");
#endif
}

TEST_F(MDSpanTest, MDSpan_VerifyOffsetsAt_1) {
    using MDSpanTestType = CUDA::MDSpan<int, CUDA::DExtents<2>>;

    CUDA::Stream stream{};
    auto src = stream.malloc(MDSpanTestType::size_of(100, 512));
    auto span = MDSpanTestType(src.get(), 100, 512);
    ASSERT_EQ(getOffsetAt(span, 0, 0), 0);
    ASSERT_EQ(getOffsetAt(span, 0, 1), 1);
    ASSERT_EQ(getOffsetAt(span, 1, 1), 512 + 1);
    ASSERT_EQ(getOffsetAt(span, 2, 1), 2 * 512 + 1);
    ASSERT_EQ(getOffsetAt(span, 32, 0), 32 * 512);
    ASSERT_EQ(getOffsetAt(span, 32, 0), 32 * 512);
#ifndef NDEBUG
    ASSERT_DEATH(getOffsetAt(span, 120, 10), "idx is out of range");
#endif
}

TEST_F(MDSpanTest, MDSpan_Verify) {
    using MDSpanTestType = CUDA::MDSpan<int, CUDA::DExtents<2>>;

    CUDA::Stream stream{};
    auto src = stream.malloc(MDSpanTestType::size_of(100, 512));
    auto span = MDSpanTestType(src.get(), 100, 512);
    assign<<<100, 512, 0, stream.get()>>>(span);
    verify<<<100, 512, 0, stream.get()>>>(span);
    ASSERT_NO_THROW(stream.synchronize());
}
