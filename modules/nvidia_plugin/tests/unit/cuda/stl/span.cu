// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cuda/runtime.hpp>
#include <cuda/stl/atomic.cuh>
#include <cuda/stl/span.cuh>

using namespace ov::nvidia_gpu;

class SpanTest : public testing::Test {
    void SetUp() override {}

    void TearDown() override {}
};

namespace {

template <typename T>
__global__ void verify_extents(CUDA::Span<T> span) {
    assert(span.size() == 101);
    assert(blockDim.x == 101);
}

template <typename T>
__global__ void assign(CUDA::Span<T> span) {
    assert(span.size() == 101);
    assert(blockDim.x == 101);
    const size_t x = threadIdx.x;
    span[x] = x;
}

template <typename T>
__global__ void verify(CUDA::Span<T> span) {
    assert(span.size() == 101);
    assert(blockDim.x == 101);
    const size_t x = threadIdx.x;
    assert(span[x] == x);
    assert(*(span.data() + x) == x);
}

}  // namespace

TEST_F(SpanTest, Span_VerifyExtents) {
    using SpanTestType = CUDA::Span<int>;

    CUDA::Stream stream{};
    auto src = stream.malloc(SpanTestType::size_of(101));
    auto span = SpanTestType(src.get(), 101);
    verify_extents<<<1, 101, 0, stream.get()>>>(span);
    ASSERT_NO_THROW(stream.synchronize());
}

TEST_F(SpanTest, Span_Verify) {
    using SpanTestType = CUDA::Span<int>;

    CUDA::Stream stream{};
    auto src = stream.malloc(SpanTestType::size_of(101));
    auto span = SpanTestType(src.get(), 101);
    assign<<<1, 101, 0, stream.get()>>>(span);
    verify<<<1, 101, 0, stream.get()>>>(span);
    ASSERT_NO_THROW(stream.synchronize());
}
