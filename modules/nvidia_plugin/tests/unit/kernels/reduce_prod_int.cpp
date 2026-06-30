// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include <cuda/runtime.hpp>
#include <kernels/reduce_prod_int.cuh>

using namespace ov::nvidia_gpu;

namespace {

// Runs the integer ReduceProd kernel, which performs a FULL reduction of all
// elements into a single scalar output[0] (it takes no axes argument).
int32_t runReduceProdInt(const std::vector<int32_t>& input) {
    CUDA::Stream stream{};
    const auto& ds = CUDA::DefaultStream::stream();
    auto d_in = ds.malloc(input.size() * sizeof(int32_t));
    ds.upload(d_in, input.data(), input.size() * sizeof(int32_t));
    auto d_out = ds.malloc(sizeof(int32_t));

    kernel::reduce_prod_int32(stream.get(),
                              static_cast<const int32_t*>(d_in.get()),
                              static_cast<int32_t*>(d_out.get()),
                              input.size());

    int32_t got = 0;
    ds.download(&got, d_out, sizeof(int32_t));
    return got;
}

}  // namespace

class ReduceProdIntKernelTest : public testing::Test {};

TEST_F(ReduceProdIntKernelTest, FullReduction) {
    ASSERT_EQ(runReduceProdInt({2, 3, 4, 5}), 120);
}

TEST_F(ReduceProdIntKernelTest, SingleElement) {
    ASSERT_EQ(runReduceProdInt({7}), 7);
}

// Sign is preserved by integer multiplication (odd number of negatives -> negative).
TEST_F(ReduceProdIntKernelTest, NegativeFactors) {
    ASSERT_EQ(runReduceProdInt({-2, 3, -4, -1}), -24);
}
