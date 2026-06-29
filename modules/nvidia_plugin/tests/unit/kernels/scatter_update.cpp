// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include <cuda/runtime.hpp>
#include <kernels/scatter_update.hpp>

using namespace ov::nvidia_gpu;

namespace {

// Runs the ScatterUpdate kernel over a 1-D `data` (axis 0, inner_size 1) and
// returns the resulting output (which starts as a copy of `data`).
std::vector<int32_t> runScatterUpdate1D(const std::vector<int32_t>& data,
                                        const std::vector<int32_t>& indices,
                                        const std::vector<int32_t>& updates) {
    const size_t axis_dim = data.size();
    const size_t indices_size = indices.size();
    const size_t num_columns = updates.size() / indices_size;  // == 1 for these 1-D cases
    kernel::ScatterUpdate su(kernel::Type_t::i32,
                             kernel::Type_t::i32,
                             /*num_input_elements*/ data.size(),
                             /*num_update_elements*/ updates.size(),
                             /*indices_size*/ indices_size,
                             /*inner_size*/ 1,
                             /*axis_dim*/ axis_dim,
                             /*num_blocks*/ 1,
                             /*num_threads*/ std::max<size_t>(num_columns, 1));

    CUDA::Stream stream{};
    const auto& ds = CUDA::DefaultStream::stream();
    auto d_in = ds.malloc(data.size() * sizeof(int32_t));
    ds.upload(d_in, data.data(), data.size() * sizeof(int32_t));
    auto d_idx = ds.malloc(indices.size() * sizeof(int32_t));
    ds.upload(d_idx, indices.data(), indices.size() * sizeof(int32_t));
    auto d_upd = ds.malloc(updates.size() * sizeof(int32_t));
    ds.upload(d_upd, updates.data(), updates.size() * sizeof(int32_t));
    auto d_out = ds.malloc(data.size() * sizeof(int32_t));

    su(stream.get(), d_in.get(), d_idx.get(), d_upd.get(), d_out.get());

    std::vector<int32_t> got(data.size());
    ds.download(got.data(), d_out, data.size() * sizeof(int32_t));
    return got;
}

}  // namespace

class ScatterUpdateKernelTest : public testing::Test {};

// Regression: in-range indices write the expected slices (last-write-wins).
TEST_F(ScatterUpdateKernelTest, InRangeUpdate) {
    const std::vector<int32_t> data{10, 20, 30, 40};
    const std::vector<int32_t> indices{1, 3};
    const std::vector<int32_t> updates{100, 300};
    const std::vector<int32_t> expected{10, 100, 30, 300};
    ASSERT_EQ(runScatterUpdate1D(data, indices, updates), expected);
}

// The fix: an index >= axis_dim must be skipped, NOT written past the output
// buffer. Without the bounds guard this writes to output[7] of a 4-element buffer
// (out-of-bounds device write / memory corruption).
TEST_F(ScatterUpdateKernelTest, OutOfRangeIndexIsSkipped) {
    const std::vector<int32_t> data{10, 20, 30, 40};
    const std::vector<int32_t> indices{1, 7};  // 7 is out of range for axis_dim == 4
    const std::vector<int32_t> updates{100, 700};
    const std::vector<int32_t> expected{10, 100, 30, 40};  // only index 1 applied; 7 skipped
    ASSERT_EQ(runScatterUpdate1D(data, indices, updates), expected);
}

// A negative index that is still out of range after the +axis_dim wrap
// (e.g. -5 -> -1 for axis_dim == 4) must also be skipped.
TEST_F(ScatterUpdateKernelTest, NegativeOutOfRangeIndexIsSkipped) {
    const std::vector<int32_t> data{10, 20, 30, 40};
    const std::vector<int32_t> indices{-1, -5};  // -1 -> 3 (valid); -5 -> -1 (invalid)
    const std::vector<int32_t> updates{300, 999};
    const std::vector<int32_t> expected{10, 20, 30, 300};  // only -1 (->3) applied
    ASSERT_EQ(runScatterUpdate1D(data, indices, updates), expected);
}
