// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstddef>
#include <cuda/runtime.hpp>
#include <cuda/stl/atomic.cuh>
#include <cuda/stl/span.cuh>

using namespace ov::nvidia_gpu;

class SpanTest : public testing::Test {
    void SetUp() override {}

    void TearDown() override {}
};

namespace {

// CUDA 11.8 generates an exit(1) host stub when a dynamic-extent Span is passed by value.
// Keep it out of the kernel ABI and construct it on the device instead.
template <typename T>
__global__ void verify_extents(T* data, std::size_t size) {
    CUDA::Span<T> span{data, size};
    assert(span.size() == 101);
    assert(blockDim.x == 101);
}

template <typename T>
__global__ void assign(T* data, std::size_t size) {
    CUDA::Span<T> span{data, size};
    assert(span.size() == 101);
    assert(blockDim.x == 101);
    const size_t x = threadIdx.x;
    span[x] = x;
}

template <typename T>
__global__ void verify(T* data, std::size_t size) {
    CUDA::Span<T> span{data, size};
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
    verify_extents<<<1, 101, 0, stream.get()>>>(static_cast<int*>(src.get()), 101);
    ASSERT_NO_THROW(stream.synchronize());
}

TEST_F(SpanTest, Span_Verify) {
    using SpanTestType = CUDA::Span<int>;

    CUDA::Stream stream{};
    auto src = stream.malloc(SpanTestType::size_of(101));
    assign<<<1, 101, 0, stream.get()>>>(static_cast<int*>(src.get()), 101);
    verify<<<1, 101, 0, stream.get()>>>(static_cast<int*>(src.get()), 101);
    ASSERT_NO_THROW(stream.synchronize());
}
