// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/gfx_opencl_source_artifacts.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "kernel_ir/gfx_custom_kernel_families.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/util/scatter_elements_update_base.hpp"
#include "openvino/op/variadic_split.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

constexpr const char* kOpenClBaselineSource = R"CLC(
static inline float gfx_unary_f32(float x, uint op) {
    switch (op) {
    case 16u: return fmax(x, 0.0f);
    case 17u: return 1.0f / (1.0f + exp(-x));
    case 18u: return tanh(x);
    case 19u: return fabs(x);
    case 20u: return -x;
    case 21u: return exp(x);
    case 22u: return log(x);
    case 23u: return sqrt(x);
    case 24u: return floor(x);
    case 25u: return ceil(x);
    default: return x;
    }
}

static inline float gfx_binary_f32(float lhs, float rhs, uint op) {
    switch (op) {
    case 1u: return lhs + rhs;
    case 2u: return lhs - rhs;
    case 3u: return lhs * rhs;
    case 4u: return lhs / rhs;
    case 5u: return fmax(lhs, rhs);
    case 6u: return fmin(lhs, rhs);
    case 7u: return pow(lhs, rhs);
    case 8u: {
        const float diff = lhs - rhs;
        return diff * diff;
    }
    default: return lhs;
    }
}

static inline uchar gfx_compare_f32(float lhs, float rhs, uint op) {
    uint result = 0u;
    if (op == 32u) {
        result = lhs == rhs ? 1u : 0u;
    } else if (op == 33u) {
        result = lhs != rhs ? 1u : 0u;
    } else if (op == 34u) {
        result = lhs > rhs ? 1u : 0u;
    } else if (op == 35u) {
        result = lhs >= rhs ? 1u : 0u;
    } else if (op == 36u) {
        result = lhs < rhs ? 1u : 0u;
    } else if (op == 37u) {
        result = lhs <= rhs ? 1u : 0u;
    }
    return (uchar)result;
}

__kernel void gfx_opencl_baseline_unary_f32(__global const float* src,
                                            __global float* dst,
                                            uint count,
                                            uint op) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    dst[gid] = gfx_unary_f32(src[gid], op);
}

__kernel void gfx_opencl_baseline_convert_f32_to_f32(__global const float* src,
                                                     __global float* dst,
                                                     uint count) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    dst[gid] = src[gid];
}

__kernel void gfx_opencl_baseline_convert_f32_to_i32(__global const float* src,
                                                     __global int* dst,
                                                     uint count) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    dst[gid] = (int)src[gid];
}

__kernel void gfx_opencl_baseline_convert_f32_to_i64(__global const float* src,
                                                     __global long* dst,
                                                     uint count) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    dst[gid] = (long)src[gid];
}

__kernel void gfx_opencl_baseline_convert_i32_to_f32(__global const int* src,
                                                     __global float* dst,
                                                     uint count) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    dst[gid] = (float)src[gid];
}

__kernel void gfx_opencl_baseline_convert_i32_to_i32(__global const int* src,
                                                     __global int* dst,
                                                     uint count) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    dst[gid] = src[gid];
}

__kernel void gfx_opencl_baseline_convert_i32_to_i64(__global const int* src,
                                                     __global long* dst,
                                                     uint count) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    dst[gid] = (long)src[gid];
}

__kernel void gfx_opencl_baseline_convert_i64_to_f32(__global const long* src,
                                                     __global float* dst,
                                                     uint count) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    dst[gid] = (float)src[gid];
}

__kernel void gfx_opencl_baseline_convert_i64_to_i32(__global const long* src,
                                                     __global int* dst,
                                                     uint count) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    dst[gid] = (int)src[gid];
}

__kernel void gfx_opencl_baseline_convert_i64_to_i64(__global const long* src,
                                                     __global long* dst,
                                                     uint count) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    dst[gid] = src[gid];
}

__kernel void gfx_opencl_baseline_binary_f32(__global const float* lhs,
                                             __global const float* rhs,
                                             __global float* dst,
                                             uint count,
                                             uint op) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    dst[gid] = gfx_binary_f32(lhs[gid], rhs[gid], op);
}

__kernel void gfx_opencl_baseline_binary_broadcast_f32(__global const float* lhs,
                                                       __global const float* rhs,
                                                       __global float* dst,
                                                       uint count,
                                                       uint op,
                                                       uint rank,
                                                       uint out_dim0,
                                                       uint out_dim1,
                                                       uint out_dim2,
                                                       uint out_dim3,
                                                       uint lhs_stride0,
                                                       uint lhs_stride1,
                                                       uint lhs_stride2,
                                                       uint lhs_stride3,
                                                       uint rhs_stride0,
                                                       uint rhs_stride1,
                                                       uint rhs_stride2,
                                                       uint rhs_stride3) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }

    uint coord0 = 0u;
    uint coord1 = 0u;
    uint coord2 = 0u;
    uint coord3 = 0u;
    if (rank == 1u) {
        coord0 = gid;
    } else if (rank == 2u) {
        coord0 = gid / out_dim1;
        coord1 = gid - coord0 * out_dim1;
    } else if (rank == 3u) {
        const uint plane0 = out_dim1 * out_dim2;
        const uint rem0 = gid - (gid / plane0) * plane0;
        coord0 = gid / plane0;
        coord1 = rem0 / out_dim2;
        coord2 = rem0 - coord1 * out_dim2;
    } else {
        const uint plane0 = out_dim1 * out_dim2 * out_dim3;
        const uint rem0 = gid - (gid / plane0) * plane0;
        const uint plane1 = out_dim2 * out_dim3;
        const uint rem1 = rem0 - (rem0 / plane1) * plane1;
        coord0 = gid / plane0;
        coord1 = rem0 / plane1;
        coord2 = rem1 / out_dim3;
        coord3 = rem1 - coord2 * out_dim3;
    }
    const uint lhs_offset = coord0 * lhs_stride0 + coord1 * lhs_stride1 +
                            coord2 * lhs_stride2 + coord3 * lhs_stride3;
    const uint rhs_offset = coord0 * rhs_stride0 + coord1 * rhs_stride1 +
                            coord2 * rhs_stride2 + coord3 * rhs_stride3;
    dst[gid] = gfx_binary_f32(lhs[lhs_offset], rhs[rhs_offset], op);
}

__kernel void gfx_opencl_baseline_matmul_f32(__global const float* lhs,
                                             __global const float* rhs,
                                             __global float* dst,
                                             uint count,
                                             uint m,
                                             uint n,
                                             uint k_dim,
                                             uint lhs_batch_stride,
                                             uint rhs_batch_stride,
                                             uint lhs_row_stride,
                                             uint lhs_col_stride,
                                             uint rhs_row_stride,
                                             uint rhs_col_stride) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint mn = m * n;
    const uint batch_idx = gid / mn;
    const uint rem = gid - batch_idx * mn;
    const uint row = rem / n;
    const uint col = rem - row * n;
    const uint lhs_base = lhs_batch_stride == 0u ? 0u : batch_idx * lhs_batch_stride;
    const uint rhs_base = rhs_batch_stride == 0u ? 0u : batch_idx * rhs_batch_stride;

    float acc = 0.0f;
    for (uint k = 0u; k < k_dim; ++k) {
        const uint lhs_idx = lhs_base + row * lhs_row_stride + k * lhs_col_stride;
        const uint rhs_idx = rhs_base + k * rhs_row_stride + col * rhs_col_stride;
        acc += lhs[lhs_idx] * rhs[rhs_idx];
    }
    dst[gid] = acc;
}

__kernel void gfx_opencl_baseline_binary_scalar_f32(__global const float* lhs,
                                                    __global const float* rhs,
                                                    __global float* dst,
                                                    uint count,
                                                    uint op,
                                                    uint input_mode) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    const float l = input_mode == 2u ? lhs[0] : lhs[gid];
    const float r = input_mode == 1u ? rhs[0] : rhs[gid];
    dst[gid] = gfx_binary_f32(l, r, op);
}

__kernel void gfx_opencl_baseline_binary_const_f32(__global const float* tensor,
                                                   __global float* dst,
                                                   uint count,
                                                   uint op,
                                                   uint input_mode,
                                                   float scalar_value) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    const float l = input_mode == 4u ? scalar_value : tensor[gid];
    const float r = input_mode == 3u ? scalar_value : tensor[gid];
    dst[gid] = gfx_binary_f32(l, r, op);
}

__kernel void gfx_opencl_baseline_compare_f32(__global const float* lhs,
                                              __global const float* rhs,
                                              __global uchar* dst,
                                              uint count,
                                              uint op) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    dst[gid] = gfx_compare_f32(lhs[gid], rhs[gid], op);
}

__kernel void gfx_opencl_baseline_select_f32(__global const uchar* cond,
                                             __global const float* then_data,
                                             __global const float* else_data,
                                             __global float* dst,
                                             uint count) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    dst[gid] = cond[gid] ? then_data[gid] : else_data[gid];
}

__kernel void gfx_opencl_baseline_transpose_f32(__global const float* src,
                                                __global float* dst,
                                                uint count,
                                                uint rank,
                                                uint out_dim0,
                                                uint out_dim1,
                                                uint out_dim2,
                                                uint out_dim3,
                                                uint in_stride0,
                                                uint in_stride1,
                                                uint in_stride2,
                                                uint in_stride3,
                                                uint perm0,
                                                uint perm1,
                                                uint perm2,
                                                uint perm3) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }

    const uint out_dim[4] = {out_dim0, out_dim1, out_dim2, out_dim3};
    const uint in_stride[4] = {in_stride0, in_stride1, in_stride2, in_stride3};
    const uint perm[4] = {perm0, perm1, perm2, perm3};
    uint coord[4] = {0u, 0u, 0u, 0u};
    uint rem = gid;
    for (uint axis = 0u; axis < rank; ++axis) {
        uint suffix = 1u;
        for (uint inner = axis + 1u; inner < rank; ++inner) {
            suffix *= out_dim[inner];
        }
        coord[axis] = suffix == 0u ? 0u : rem / suffix;
        rem = suffix == 0u ? 0u : rem - coord[axis] * suffix;
    }

    uint src_offset = 0u;
    for (uint axis = 0u; axis < rank; ++axis) {
        src_offset += coord[axis] * in_stride[perm[axis]];
    }
    dst[gid] = src[src_offset];
}

__kernel void gfx_opencl_baseline_slice_f32(__global const float* src,
                                            __global float* dst,
                                            uint count,
                                            uint rank,
                                            uint out_dim0,
                                            uint out_dim1,
                                            uint out_dim2,
                                            uint out_dim3,
                                            uint in_stride0,
                                            uint in_stride1,
                                            uint in_stride2,
                                            uint in_stride3,
                                            uint begin0,
                                            uint begin1,
                                            uint begin2,
                                            uint begin3,
                                            uint step0,
                                            uint step1,
                                            uint step2,
                                            uint step3) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint out_dim[4] = {out_dim0, out_dim1, out_dim2, out_dim3};
    const uint in_stride[4] = {in_stride0, in_stride1, in_stride2, in_stride3};
    const uint begin[4] = {begin0, begin1, begin2, begin3};
    const uint step[4] = {step0, step1, step2, step3};
    uint coord[4] = {0u, 0u, 0u, 0u};
    uint rem = gid;
    for (uint axis = 0u; axis < rank; ++axis) {
        uint suffix = 1u;
        for (uint inner_axis = axis + 1u; inner_axis < rank; ++inner_axis) {
            suffix *= out_dim[inner_axis];
        }
        coord[axis] = suffix == 0u ? 0u : rem / suffix;
        rem = suffix == 0u ? 0u : rem - coord[axis] * suffix;
    }

    uint src_offset = 0u;
    for (uint axis = 0u; axis < rank; ++axis) {
        src_offset += (begin[axis] + coord[axis] * step[axis]) * in_stride[axis];
    }
    dst[gid] = src[src_offset];
}

__kernel void gfx_opencl_baseline_range_f32(__global const float* start,
                                            __global const float* stop,
                                            __global const float* step,
                                            __global float* dst,
                                            uint count) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    (void)stop;
    dst[gid] = start[0] + (float)gid * step[0];
}

__kernel void gfx_opencl_baseline_range_i32(__global const int* start,
                                            __global const int* stop,
                                            __global const int* step,
                                            __global int* dst,
                                            uint count) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    (void)stop;
    dst[gid] = start[0] + (int)gid * step[0];
}

__kernel void gfx_opencl_baseline_range_i64(__global const long* start,
                                            __global const long* stop,
                                            __global const long* step,
                                            __global long* dst,
                                            uint count) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    (void)stop;
    dst[gid] = start[0] + (long)gid * step[0];
}

__kernel void gfx_opencl_baseline_tile_f32(__global const float* src,
                                           __global float* dst,
                                           uint count,
                                           uint rank,
                                           uint out_dim0,
                                           uint out_dim1,
                                           uint out_dim2,
                                           uint out_dim3,
                                           uint in_dim0,
                                           uint in_dim1,
                                           uint in_dim2,
                                           uint in_dim3,
                                           uint out_stride0,
                                           uint out_stride1,
                                           uint out_stride2,
                                           uint out_stride3,
                                           uint in_stride0,
                                           uint in_stride1,
                                           uint in_stride2,
                                           uint in_stride3) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint out_dim[4] = {out_dim0, out_dim1, out_dim2, out_dim3};
    const uint in_dim[4] = {in_dim0, in_dim1, in_dim2, in_dim3};
    const uint out_stride[4] = {out_stride0, out_stride1, out_stride2, out_stride3};
    const uint in_stride[4] = {in_stride0, in_stride1, in_stride2, in_stride3};

    uint src_offset = 0u;
    for (uint axis = 0u; axis < rank; ++axis) {
        const uint out_axis_coord =
            out_stride[axis] == 0u ? 0u : (gid / out_stride[axis]) % out_dim[axis];
        const uint in_axis_coord =
            in_dim[axis] == 0u ? 0u : out_axis_coord % in_dim[axis];
        src_offset += in_axis_coord * in_stride[axis];
    }
    dst[gid] = src[src_offset];
}

__kernel void gfx_opencl_baseline_gather_i32_f32(__global const float* data,
                                                 __global const int* indices,
                                                 __global float* dst,
                                                 uint count,
                                                 uint outer,
                                                 uint inner,
                                                 uint axis_dim,
                                                 uint indices_count) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    uint tmp = gid;
    const uint inner_idx = tmp % inner;
    tmp /= inner;
    const uint idx_idx = tmp % indices_count;
    tmp /= indices_count;
    const uint outer_idx = tmp;

    int index = indices[idx_idx];
    if (index < 0) {
        index += (int)axis_dim;
    }
    if (index < 0) {
        index = 0;
    }
    if (index >= (int)axis_dim) {
        index = (int)axis_dim - 1;
    }
    const uint src_idx = ((outer_idx * axis_dim + (uint)index) * inner) + inner_idx;
    dst[gid] = data[src_idx];
}

__kernel void gfx_opencl_baseline_gather_i64_f32(__global const float* data,
                                                 __global const long* indices,
                                                 __global float* dst,
                                                 uint count,
                                                 uint outer,
                                                 uint inner,
                                                 uint axis_dim,
                                                 uint indices_count) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    uint tmp = gid;
    const uint inner_idx = tmp % inner;
    tmp /= inner;
    const uint idx_idx = tmp % indices_count;
    tmp /= indices_count;
    const uint outer_idx = tmp;

    long index = indices[idx_idx];
    if (index < 0) {
        index += (long)axis_dim;
    }
    if (index < 0) {
        index = 0;
    }
    if (index >= (long)axis_dim) {
        index = (long)axis_dim - 1;
    }
    const uint src_idx = ((outer_idx * axis_dim + (uint)index) * inner) + inner_idx;
    dst[gid] = data[src_idx];
}

__kernel void gfx_opencl_baseline_gather_elements_i32_f32(__global const float* data,
                                                          __global const int* indices,
                                                          __global float* dst,
                                                          uint count,
                                                          uint rank,
                                                          uint axis,
                                                          uint out_dim0,
                                                          uint out_dim1,
                                                          uint out_dim2,
                                                          uint out_dim3,
                                                          uint out_stride0,
                                                          uint out_stride1,
                                                          uint out_stride2,
                                                          uint out_stride3,
                                                          uint data_dim0,
                                                          uint data_dim1,
                                                          uint data_dim2,
                                                          uint data_dim3,
                                                          uint data_stride0,
                                                          uint data_stride1,
                                                          uint data_stride2,
                                                          uint data_stride3) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint out_dim[4] = {out_dim0, out_dim1, out_dim2, out_dim3};
    const uint out_stride[4] = {out_stride0, out_stride1, out_stride2, out_stride3};
    const uint data_dim[4] = {data_dim0, data_dim1, data_dim2, data_dim3};
    const uint data_stride[4] = {data_stride0, data_stride1, data_stride2, data_stride3};
    uint coord[4] = {0u, 0u, 0u, 0u};
    for (uint i = 0u; i < rank; ++i) {
        const uint stride = out_stride[i];
        coord[i] = stride == 0u ? 0u : (gid / stride) % out_dim[i];
    }

    int index = indices[gid];
    const int axis_dim = (int)data_dim[axis];
    if (index < 0) {
        index += axis_dim;
    }
    if (index < 0) {
        index = 0;
    }
    if (index >= axis_dim) {
        index = axis_dim - 1;
    }

    uint src_idx = 0u;
    for (uint i = 0u; i < rank; ++i) {
        const uint c = i == axis ? (uint)index : coord[i];
        src_idx += c * data_stride[i];
    }
    dst[gid] = data[src_idx];
}

__kernel void gfx_opencl_baseline_gather_elements_i64_f32(__global const float* data,
                                                          __global const long* indices,
                                                          __global float* dst,
                                                          uint count,
                                                          uint rank,
                                                          uint axis,
                                                          uint out_dim0,
                                                          uint out_dim1,
                                                          uint out_dim2,
                                                          uint out_dim3,
                                                          uint out_stride0,
                                                          uint out_stride1,
                                                          uint out_stride2,
                                                          uint out_stride3,
                                                          uint data_dim0,
                                                          uint data_dim1,
                                                          uint data_dim2,
                                                          uint data_dim3,
                                                          uint data_stride0,
                                                          uint data_stride1,
                                                          uint data_stride2,
                                                          uint data_stride3) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint out_dim[4] = {out_dim0, out_dim1, out_dim2, out_dim3};
    const uint out_stride[4] = {out_stride0, out_stride1, out_stride2, out_stride3};
    const uint data_dim[4] = {data_dim0, data_dim1, data_dim2, data_dim3};
    const uint data_stride[4] = {data_stride0, data_stride1, data_stride2, data_stride3};
    uint coord[4] = {0u, 0u, 0u, 0u};
    for (uint i = 0u; i < rank; ++i) {
        const uint stride = out_stride[i];
        coord[i] = stride == 0u ? 0u : (gid / stride) % out_dim[i];
    }

    long index = indices[gid];
    const long axis_dim = (long)data_dim[axis];
    if (index < 0) {
        index += axis_dim;
    }
    if (index < 0) {
        index = 0;
    }
    if (index >= axis_dim) {
        index = axis_dim - 1;
    }

    uint src_idx = 0u;
    for (uint i = 0u; i < rank; ++i) {
        const uint c = i == axis ? (uint)index : coord[i];
        src_idx += c * data_stride[i];
    }
    dst[gid] = data[src_idx];
}

__kernel void gfx_opencl_baseline_gather_nd_i32_f32(__global const float* data,
                                                    __global const int* indices,
                                                    __global float* dst,
                                                    uint count,
                                                    uint index_depth,
                                                    uint slice_rank,
                                                    uint slice_size,
                                                    uint data_dim0,
                                                    uint data_dim1,
                                                    uint data_dim2,
                                                    uint data_dim3,
                                                    uint data_stride0,
                                                    uint data_stride1,
                                                    uint data_stride2,
                                                    uint data_stride3) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint data_dim[4] = {data_dim0, data_dim1, data_dim2, data_dim3};
    const uint data_stride[4] = {data_stride0, data_stride1, data_stride2, data_stride3};
    const uint tuple_idx = slice_size == 0u ? 0u : gid / slice_size;
    const uint slice_gid = slice_size == 0u ? 0u : gid - tuple_idx * slice_size;

    uint src_idx = 0u;
    for (uint i = 0u; i < index_depth; ++i) {
        int index = indices[tuple_idx * index_depth + i];
        const int dim = (int)data_dim[i];
        if (index < 0) {
            index += dim;
        }
        if (index < 0) {
            index = 0;
        }
        if (index >= dim) {
            index = dim - 1;
        }
        src_idx += (uint)index * data_stride[i];
    }

    for (uint i = 0u; i < slice_rank; ++i) {
        const uint axis = index_depth + i;
        const uint stride = data_stride[axis];
        const uint coord = stride == 0u ? 0u : (slice_gid / stride) % data_dim[axis];
        src_idx += coord * stride;
    }
    dst[gid] = data[src_idx];
}

__kernel void gfx_opencl_baseline_gather_nd_i64_f32(__global const float* data,
                                                    __global const long* indices,
                                                    __global float* dst,
                                                    uint count,
                                                    uint index_depth,
                                                    uint slice_rank,
                                                    uint slice_size,
                                                    uint data_dim0,
                                                    uint data_dim1,
                                                    uint data_dim2,
                                                    uint data_dim3,
                                                    uint data_stride0,
                                                    uint data_stride1,
                                                    uint data_stride2,
                                                    uint data_stride3) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint data_dim[4] = {data_dim0, data_dim1, data_dim2, data_dim3};
    const uint data_stride[4] = {data_stride0, data_stride1, data_stride2, data_stride3};
    const uint tuple_idx = slice_size == 0u ? 0u : gid / slice_size;
    const uint slice_gid = slice_size == 0u ? 0u : gid - tuple_idx * slice_size;

    uint src_idx = 0u;
    for (uint i = 0u; i < index_depth; ++i) {
        long index = indices[tuple_idx * index_depth + i];
        const long dim = (long)data_dim[i];
        if (index < 0) {
            index += dim;
        }
        if (index < 0) {
            index = 0;
        }
        if (index >= dim) {
            index = dim - 1;
        }
        src_idx += (uint)index * data_stride[i];
    }

    for (uint i = 0u; i < slice_rank; ++i) {
        const uint axis = index_depth + i;
        const uint stride = data_stride[axis];
        const uint coord = stride == 0u ? 0u : (slice_gid / stride) % data_dim[axis];
        src_idx += coord * stride;
    }
    dst[gid] = data[src_idx];
}

__kernel void gfx_opencl_baseline_scatter_update_i32_f32(__global const float* data,
                                                         __global const int* indices,
                                                         __global const float* updates,
                                                         __global float* dst,
                                                         uint count,
                                                         uint data_rank,
                                                         uint idx_rank,
                                                         uint update_rank,
                                                         uint axis,
                                                         uint idx_total,
                                                         uint data_dim0,
                                                         uint data_dim1,
                                                         uint data_dim2,
                                                         uint data_dim3,
                                                         uint data_stride0,
                                                         uint data_stride1,
                                                         uint data_stride2,
                                                         uint data_stride3,
                                                         uint idx_dim0,
                                                         uint idx_dim1,
                                                         uint idx_dim2,
                                                         uint idx_dim3,
                                                         uint idx_stride0,
                                                         uint idx_stride1,
                                                         uint idx_stride2,
                                                         uint idx_stride3,
                                                         uint update_stride0,
                                                         uint update_stride1,
                                                         uint update_stride2,
                                                         uint update_stride3,
                                                         uint update_stride4,
                                                         uint update_stride5,
                                                         uint update_stride6) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint data_dim[4] = {data_dim0, data_dim1, data_dim2, data_dim3};
    const uint data_stride[4] = {data_stride0, data_stride1, data_stride2, data_stride3};
    const uint idx_dim[4] = {idx_dim0, idx_dim1, idx_dim2, idx_dim3};
    const uint idx_stride[4] = {idx_stride0, idx_stride1, idx_stride2, idx_stride3};
    const uint update_stride[7] = {
        update_stride0, update_stride1, update_stride2, update_stride3,
        update_stride4, update_stride5, update_stride6};
    uint coord[4] = {0u, 0u, 0u, 0u};
    uint rem = gid;
    for (uint d = 0u; d < data_rank; ++d) {
        const uint stride = data_stride[d];
        coord[d] = stride == 0u ? 0u : rem / stride;
        rem = stride == 0u ? 0u : rem - coord[d] * stride;
    }

    float value = data[gid];
    for (uint linear = 0u; linear < idx_total; ++linear) {
        uint idx_coord[4] = {0u, 0u, 0u, 0u};
        for (uint d = 0u; d < idx_rank; ++d) {
            const uint stride = idx_stride[d];
            idx_coord[d] = stride == 0u ? 0u : (linear / stride) % idx_dim[d];
        }
        int index = indices[linear];
        const int axis_dim = (int)data_dim[axis];
        if (index < 0) {
            index += axis_dim;
        }
        if (index < 0) {
            index = 0;
        }
        if (index >= axis_dim) {
            index = axis_dim - 1;
        }
        if ((uint)index != coord[axis]) {
            continue;
        }
        uint update_offset = 0u;
        uint update_dim = 0u;
        for (uint d = 0u; d < axis; ++d) {
            update_offset += coord[d] * update_stride[update_dim++];
        }
        for (uint d = 0u; d < idx_rank; ++d) {
            update_offset += idx_coord[d] * update_stride[update_dim++];
        }
        for (uint d = axis + 1u; d < data_rank; ++d) {
            update_offset += coord[d] * update_stride[update_dim++];
        }
        (void)update_rank;
        value = updates[update_offset];
    }
    dst[gid] = value;
}

__kernel void gfx_opencl_baseline_scatter_update_i64_f32(__global const float* data,
                                                         __global const long* indices,
                                                         __global const float* updates,
                                                         __global float* dst,
                                                         uint count,
                                                         uint data_rank,
                                                         uint idx_rank,
                                                         uint update_rank,
                                                         uint axis,
                                                         uint idx_total,
                                                         uint data_dim0,
                                                         uint data_dim1,
                                                         uint data_dim2,
                                                         uint data_dim3,
                                                         uint data_stride0,
                                                         uint data_stride1,
                                                         uint data_stride2,
                                                         uint data_stride3,
                                                         uint idx_dim0,
                                                         uint idx_dim1,
                                                         uint idx_dim2,
                                                         uint idx_dim3,
                                                         uint idx_stride0,
                                                         uint idx_stride1,
                                                         uint idx_stride2,
                                                         uint idx_stride3,
                                                         uint update_stride0,
                                                         uint update_stride1,
                                                         uint update_stride2,
                                                         uint update_stride3,
                                                         uint update_stride4,
                                                         uint update_stride5,
                                                         uint update_stride6) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint data_dim[4] = {data_dim0, data_dim1, data_dim2, data_dim3};
    const uint data_stride[4] = {data_stride0, data_stride1, data_stride2, data_stride3};
    const uint idx_dim[4] = {idx_dim0, idx_dim1, idx_dim2, idx_dim3};
    const uint idx_stride[4] = {idx_stride0, idx_stride1, idx_stride2, idx_stride3};
    const uint update_stride[7] = {
        update_stride0, update_stride1, update_stride2, update_stride3,
        update_stride4, update_stride5, update_stride6};
    uint coord[4] = {0u, 0u, 0u, 0u};
    uint rem = gid;
    for (uint d = 0u; d < data_rank; ++d) {
        const uint stride = data_stride[d];
        coord[d] = stride == 0u ? 0u : rem / stride;
        rem = stride == 0u ? 0u : rem - coord[d] * stride;
    }

    float value = data[gid];
    for (uint linear = 0u; linear < idx_total; ++linear) {
        uint idx_coord[4] = {0u, 0u, 0u, 0u};
        for (uint d = 0u; d < idx_rank; ++d) {
            const uint stride = idx_stride[d];
            idx_coord[d] = stride == 0u ? 0u : (linear / stride) % idx_dim[d];
        }
        long index = indices[linear];
        const long axis_dim = (long)data_dim[axis];
        if (index < 0) {
            index += axis_dim;
        }
        if (index < 0) {
            index = 0;
        }
        if (index >= axis_dim) {
            index = axis_dim - 1;
        }
        if ((uint)index != coord[axis]) {
            continue;
        }
        uint update_offset = 0u;
        uint update_dim = 0u;
        for (uint d = 0u; d < axis; ++d) {
            update_offset += coord[d] * update_stride[update_dim++];
        }
        for (uint d = 0u; d < idx_rank; ++d) {
            update_offset += idx_coord[d] * update_stride[update_dim++];
        }
        for (uint d = axis + 1u; d < data_rank; ++d) {
            update_offset += coord[d] * update_stride[update_dim++];
        }
        (void)update_rank;
        value = updates[update_offset];
    }
    dst[gid] = value;
}

__kernel void gfx_opencl_baseline_scatter_elements_i32_f32(__global const float* data,
                                                           __global const int* indices,
                                                           __global const float* updates,
                                                           __global float* dst,
                                                           uint count,
                                                           uint rank,
                                                           uint axis,
                                                           uint update_count,
                                                           uint update_dim0,
                                                           uint update_dim1,
                                                           uint update_dim2,
                                                           uint update_dim3,
                                                           uint update_stride0,
                                                           uint update_stride1,
                                                           uint update_stride2,
                                                           uint update_stride3,
                                                           uint data_dim0,
                                                           uint data_dim1,
                                                           uint data_dim2,
                                                           uint data_dim3,
                                                           uint data_stride0,
                                                           uint data_stride1,
                                                           uint data_stride2,
                                                           uint data_stride3) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint update_dim[4] = {update_dim0, update_dim1, update_dim2, update_dim3};
    const uint update_stride[4] = {update_stride0, update_stride1, update_stride2, update_stride3};
    const uint data_dim[4] = {data_dim0, data_dim1, data_dim2, data_dim3};
    const uint data_stride[4] = {data_stride0, data_stride1, data_stride2, data_stride3};
    float value = data[gid];
    for (uint linear = 0u; linear < update_count; ++linear) {
        uint coord[4] = {0u, 0u, 0u, 0u};
        for (uint i = 0u; i < rank; ++i) {
            const uint stride = update_stride[i];
            coord[i] = stride == 0u ? 0u : (linear / stride) % update_dim[i];
        }
        int index = indices[linear];
        const int axis_dim = (int)data_dim[axis];
        if (index < 0) {
            index += axis_dim;
        }
        if (index < 0) {
            index = 0;
        }
        if (index >= axis_dim) {
            index = axis_dim - 1;
        }
        uint out_idx = 0u;
        for (uint i = 0u; i < rank; ++i) {
            const uint c = i == axis ? (uint)index : coord[i];
            out_idx += c * data_stride[i];
        }
        if (out_idx == gid) {
            value = updates[linear];
        }
    }
    dst[gid] = value;
}

__kernel void gfx_opencl_baseline_scatter_elements_i64_f32(__global const float* data,
                                                           __global const long* indices,
                                                           __global const float* updates,
                                                           __global float* dst,
                                                           uint count,
                                                           uint rank,
                                                           uint axis,
                                                           uint update_count,
                                                           uint update_dim0,
                                                           uint update_dim1,
                                                           uint update_dim2,
                                                           uint update_dim3,
                                                           uint update_stride0,
                                                           uint update_stride1,
                                                           uint update_stride2,
                                                           uint update_stride3,
                                                           uint data_dim0,
                                                           uint data_dim1,
                                                           uint data_dim2,
                                                           uint data_dim3,
                                                           uint data_stride0,
                                                           uint data_stride1,
                                                           uint data_stride2,
                                                           uint data_stride3) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint update_dim[4] = {update_dim0, update_dim1, update_dim2, update_dim3};
    const uint update_stride[4] = {update_stride0, update_stride1, update_stride2, update_stride3};
    const uint data_dim[4] = {data_dim0, data_dim1, data_dim2, data_dim3};
    const uint data_stride[4] = {data_stride0, data_stride1, data_stride2, data_stride3};
    float value = data[gid];
    for (uint linear = 0u; linear < update_count; ++linear) {
        uint coord[4] = {0u, 0u, 0u, 0u};
        for (uint i = 0u; i < rank; ++i) {
            const uint stride = update_stride[i];
            coord[i] = stride == 0u ? 0u : (linear / stride) % update_dim[i];
        }
        long index = indices[linear];
        const long axis_dim = (long)data_dim[axis];
        if (index < 0) {
            index += axis_dim;
        }
        if (index < 0) {
            index = 0;
        }
        if (index >= axis_dim) {
            index = axis_dim - 1;
        }
        uint out_idx = 0u;
        for (uint i = 0u; i < rank; ++i) {
            const uint c = i == axis ? (uint)index : coord[i];
            out_idx += c * data_stride[i];
        }
        if (out_idx == gid) {
            value = updates[linear];
        }
    }
    dst[gid] = value;
}

__kernel void gfx_opencl_baseline_scatter_nd_i32_f32(__global const float* data,
                                                     __global const int* indices,
                                                     __global const float* updates,
                                                     __global float* dst,
                                                     uint count,
                                                     uint index_depth,
                                                     uint slice_size,
                                                     uint tuple_count,
                                                     uint data_dim0,
                                                     uint data_dim1,
                                                     uint data_dim2,
                                                     uint data_dim3,
                                                     uint data_stride0,
                                                     uint data_stride1,
                                                     uint data_stride2,
                                                     uint data_stride3) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint data_dim[4] = {data_dim0, data_dim1, data_dim2, data_dim3};
    const uint data_stride[4] = {data_stride0, data_stride1, data_stride2, data_stride3};
    float value = data[gid];
    for (uint tuple_idx = 0u; tuple_idx < tuple_count; ++tuple_idx) {
        uint base = 0u;
        for (uint i = 0u; i < index_depth; ++i) {
            int index = indices[tuple_idx * index_depth + i];
            const int dim = (int)data_dim[i];
            if (index < 0) {
                index += dim;
            }
            if (index < 0) {
                index = 0;
            }
            if (index >= dim) {
                index = dim - 1;
            }
            base += (uint)index * data_stride[i];
        }
        if (gid >= base && gid < base + slice_size) {
            value = updates[tuple_idx * slice_size + (gid - base)];
        }
    }
    dst[gid] = value;
}

__kernel void gfx_opencl_baseline_scatter_nd_i64_f32(__global const float* data,
                                                     __global const long* indices,
                                                     __global const float* updates,
                                                     __global float* dst,
                                                     uint count,
                                                     uint index_depth,
                                                     uint slice_size,
                                                     uint tuple_count,
                                                     uint data_dim0,
                                                     uint data_dim1,
                                                     uint data_dim2,
                                                     uint data_dim3,
                                                     uint data_stride0,
                                                     uint data_stride1,
                                                     uint data_stride2,
                                                     uint data_stride3) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint data_dim[4] = {data_dim0, data_dim1, data_dim2, data_dim3};
    const uint data_stride[4] = {data_stride0, data_stride1, data_stride2, data_stride3};
    float value = data[gid];
    for (uint tuple_idx = 0u; tuple_idx < tuple_count; ++tuple_idx) {
        uint base = 0u;
        for (uint i = 0u; i < index_depth; ++i) {
            long index = indices[tuple_idx * index_depth + i];
            const long dim = (long)data_dim[i];
            if (index < 0) {
                index += dim;
            }
            if (index < 0) {
                index = 0;
            }
            if (index >= dim) {
                index = dim - 1;
            }
            base += (uint)index * data_stride[i];
        }
        if (gid >= base && gid < base + slice_size) {
            value = updates[tuple_idx * slice_size + (gid - base)];
        }
    }
    dst[gid] = value;
}

__kernel void gfx_opencl_baseline_shapeof_i32(__global const uchar* src,
                                              __global int* dst,
                                              uint count,
                                              uint dim0,
                                              uint dim1,
                                              uint dim2,
                                              uint dim3,
                                              uint dim4,
                                              uint dim5,
                                              uint dim6,
                                              uint dim7) {
    (void)src;
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint dims[8] = {dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7};
    dst[gid] = (int)dims[gid];
}

__kernel void gfx_opencl_baseline_shapeof_i64(__global const uchar* src,
                                              __global long* dst,
                                              uint count,
                                              uint dim0,
                                              uint dim1,
                                              uint dim2,
                                              uint dim3,
                                              uint dim4,
                                              uint dim5,
                                              uint dim6,
                                              uint dim7) {
    (void)src;
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint dims[8] = {dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7};
    dst[gid] = (long)dims[gid];
}

static inline uint gfx_concat_load_f32(__global const float* src,
                                       __private float* value,
                                       uint axis_idx,
                                       uint outer_idx,
                                       uint inner_idx,
                                       uint inner,
                                       uint axis_offset,
                                       uint axis_len) {
    if (axis_idx < axis_offset || axis_idx >= axis_offset + axis_len) {
        return 0u;
    }
    const uint src_axis_idx = axis_idx - axis_offset;
    const uint src_idx = (outer_idx * axis_len + src_axis_idx) * inner + inner_idx;
    *value = src[src_idx];
    return 1u;
}

__kernel void gfx_opencl_baseline_concat2_f32(__global const float* src0,
                                              __global const float* src1,
                                              __global float* dst,
                                              uint count,
                                              uint out_axis,
                                              uint inner,
                                              uint src0_axis_offset,
                                              uint src0_axis_len,
                                              uint src1_axis_offset,
                                              uint src1_axis_len) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint inner_idx = gid % inner;
    const uint axis_idx = (gid / inner) % out_axis;
    const uint outer_idx = gid / (out_axis * inner);
    float value = 0.0f;
    (void)(gfx_concat_load_f32(src0, &value, axis_idx, outer_idx, inner_idx, inner,
                              src0_axis_offset, src0_axis_len) ||
           gfx_concat_load_f32(src1, &value, axis_idx, outer_idx, inner_idx, inner,
                              src1_axis_offset, src1_axis_len));
    dst[gid] = value;
}

__kernel void gfx_opencl_baseline_concat3_f32(__global const float* src0,
                                              __global const float* src1,
                                              __global const float* src2,
                                              __global float* dst,
                                              uint count,
                                              uint out_axis,
                                              uint inner,
                                              uint src0_axis_offset,
                                              uint src0_axis_len,
                                              uint src1_axis_offset,
                                              uint src1_axis_len,
                                              uint src2_axis_offset,
                                              uint src2_axis_len) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint inner_idx = gid % inner;
    const uint axis_idx = (gid / inner) % out_axis;
    const uint outer_idx = gid / (out_axis * inner);
    float value = 0.0f;
    (void)(gfx_concat_load_f32(src0, &value, axis_idx, outer_idx, inner_idx, inner,
                              src0_axis_offset, src0_axis_len) ||
           gfx_concat_load_f32(src1, &value, axis_idx, outer_idx, inner_idx, inner,
                              src1_axis_offset, src1_axis_len) ||
           gfx_concat_load_f32(src2, &value, axis_idx, outer_idx, inner_idx, inner,
                              src2_axis_offset, src2_axis_len));
    dst[gid] = value;
}

__kernel void gfx_opencl_baseline_concat4_f32(__global const float* src0,
                                              __global const float* src1,
                                              __global const float* src2,
                                              __global const float* src3,
                                              __global float* dst,
                                              uint count,
                                              uint out_axis,
                                              uint inner,
                                              uint src0_axis_offset,
                                              uint src0_axis_len,
                                              uint src1_axis_offset,
                                              uint src1_axis_len,
                                              uint src2_axis_offset,
                                              uint src2_axis_len,
                                              uint src3_axis_offset,
                                              uint src3_axis_len) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint inner_idx = gid % inner;
    const uint axis_idx = (gid / inner) % out_axis;
    const uint outer_idx = gid / (out_axis * inner);
    float value = 0.0f;
    (void)(gfx_concat_load_f32(src0, &value, axis_idx, outer_idx, inner_idx, inner,
                              src0_axis_offset, src0_axis_len) ||
           gfx_concat_load_f32(src1, &value, axis_idx, outer_idx, inner_idx, inner,
                              src1_axis_offset, src1_axis_len) ||
           gfx_concat_load_f32(src2, &value, axis_idx, outer_idx, inner_idx, inner,
                              src2_axis_offset, src2_axis_len) ||
           gfx_concat_load_f32(src3, &value, axis_idx, outer_idx, inner_idx, inner,
                              src3_axis_offset, src3_axis_len));
    dst[gid] = value;
}

static inline void gfx_split_store_f32(__global const float* src,
                                       __global float* dst,
                                       uint gid,
                                       uint axis_idx,
                                       uint outer_idx,
                                       uint inner_idx,
                                       uint inner,
                                       uint axis_offset,
                                       uint axis_len) {
    if (axis_idx < axis_offset || axis_idx >= axis_offset + axis_len) {
        return;
    }
    const uint dst_axis_idx = axis_idx - axis_offset;
    const uint dst_idx = (outer_idx * axis_len + dst_axis_idx) * inner + inner_idx;
    dst[dst_idx] = src[gid];
}

__kernel void gfx_opencl_baseline_split2_f32(__global const float* src,
                                             __global float* dst0,
                                             __global float* dst1,
                                             uint count,
                                             uint axis_total,
                                             uint inner,
                                             uint offset0,
                                             uint len0,
                                             uint offset1,
                                             uint len1) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint inner_idx = gid % inner;
    const uint axis_idx = (gid / inner) % axis_total;
    const uint outer_idx = gid / (axis_total * inner);
    gfx_split_store_f32(src, dst0, gid, axis_idx, outer_idx, inner_idx, inner, offset0, len0);
    gfx_split_store_f32(src, dst1, gid, axis_idx, outer_idx, inner_idx, inner, offset1, len1);
}

__kernel void gfx_opencl_baseline_split3_f32(__global const float* src,
                                             __global float* dst0,
                                             __global float* dst1,
                                             __global float* dst2,
                                             uint count,
                                             uint axis_total,
                                             uint inner,
                                             uint offset0,
                                             uint len0,
                                             uint offset1,
                                             uint len1,
                                             uint offset2,
                                             uint len2) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint inner_idx = gid % inner;
    const uint axis_idx = (gid / inner) % axis_total;
    const uint outer_idx = gid / (axis_total * inner);
    gfx_split_store_f32(src, dst0, gid, axis_idx, outer_idx, inner_idx, inner, offset0, len0);
    gfx_split_store_f32(src, dst1, gid, axis_idx, outer_idx, inner_idx, inner, offset1, len1);
    gfx_split_store_f32(src, dst2, gid, axis_idx, outer_idx, inner_idx, inner, offset2, len2);
}

__kernel void gfx_opencl_baseline_split4_f32(__global const float* src,
                                             __global float* dst0,
                                             __global float* dst1,
                                             __global float* dst2,
                                             __global float* dst3,
                                             uint count,
                                             uint axis_total,
                                             uint inner,
                                             uint offset0,
                                             uint len0,
                                             uint offset1,
                                             uint len1,
                                             uint offset2,
                                             uint len2,
                                             uint offset3,
                                             uint len3) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint inner_idx = gid % inner;
    const uint axis_idx = (gid / inner) % axis_total;
    const uint outer_idx = gid / (axis_total * inner);
    gfx_split_store_f32(src, dst0, gid, axis_idx, outer_idx, inner_idx, inner, offset0, len0);
    gfx_split_store_f32(src, dst1, gid, axis_idx, outer_idx, inner_idx, inner, offset1, len1);
    gfx_split_store_f32(src, dst2, gid, axis_idx, outer_idx, inner_idx, inner, offset2, len2);
    gfx_split_store_f32(src, dst3, gid, axis_idx, outer_idx, inner_idx, inner, offset3, len3);
}
)CLC";

constexpr const char* kOpenClMatMulSource = R"CLC(
__kernel void gfx_opencl_baseline_matmul_f32(__global const float* lhs,
                                             __global const float* rhs,
                                             __global float* dst,
                                             uint count,
                                             uint m,
                                             uint n,
                                             uint k_dim,
                                             uint lhs_batch_stride,
                                             uint rhs_batch_stride,
                                             uint lhs_row_stride,
                                             uint lhs_col_stride,
                                             uint rhs_row_stride,
                                             uint rhs_col_stride) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint mn = m * n;
    const uint batch_idx = gid / mn;
    const uint rem = gid - batch_idx * mn;
    const uint row = rem / n;
    const uint col = rem - row * n;
    const uint lhs_base = lhs_batch_stride == 0u ? 0u : batch_idx * lhs_batch_stride;
    const uint rhs_base = rhs_batch_stride == 0u ? 0u : batch_idx * rhs_batch_stride;

    float acc = 0.0f;
    for (uint k = 0u; k < k_dim; ++k) {
        const uint lhs_idx = lhs_base + row * lhs_row_stride + k * lhs_col_stride;
        const uint rhs_idx = rhs_base + k * rhs_row_stride + col * rhs_col_stride;
        acc += lhs[lhs_idx] * rhs[rhs_idx];
    }
    dst[gid] = acc;
}
)CLC";

constexpr const char* kOpenClSoftmaxSource = R"CLC(
__kernel void gfx_opencl_baseline_softmax_f32(__global const float* src,
                                              __global float* dst,
                                              uint count,
                                              uint outer,
                                              uint axis_dim,
                                              uint inner) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint plane = axis_dim * inner;
    const uint outer_idx = gid / plane;
    if (outer_idx >= outer) {
        return;
    }
    const uint inner_idx = gid % inner;
    const uint base = outer_idx * plane + inner_idx;

    float max_value = src[base];
    for (uint axis_idx = 1u; axis_idx < axis_dim; ++axis_idx) {
        max_value = fmax(max_value, src[base + axis_idx * inner]);
    }

    float denom = 0.0f;
    for (uint axis_idx = 0u; axis_idx < axis_dim; ++axis_idx) {
        denom += exp(src[base + axis_idx * inner] - max_value);
    }
    dst[gid] = exp(src[gid] - max_value) / denom;
}
)CLC";

constexpr const char* kOpenClShapeOfSource = R"CLC(
__kernel void gfx_opencl_baseline_shapeof_i32(__global const uchar* src,
                                              __global int* dst,
                                              uint count,
                                              uint dim0,
                                              uint dim1,
                                              uint dim2,
                                              uint dim3,
                                              uint dim4,
                                              uint dim5,
                                              uint dim6,
                                              uint dim7) {
    (void)src;
    const uint gid = (uint)get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint dims[8] = {dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7};
    dst[gid] = (int)dims[gid];
}

__kernel void gfx_opencl_baseline_shapeof_i64(__global const uchar* src,
                                              __global uint* dst,
                                              uint count,
                                              uint dim0,
                                              uint dim1,
                                              uint dim2,
                                              uint dim3,
                                              uint dim4,
                                              uint dim5,
                                              uint dim6,
                                              uint dim7) {
    (void)src;
    const uint gid = (uint)get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint dims[8] = {dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7};
    const uint word = gid * 2u;
    dst[word] = dims[gid];
    dst[word + 1u] = 0u;
}
)CLC";

constexpr const char* kOpenClUnaryF32Source = R"CLC(
static inline float gfx_unary_f32(float x, uint op) {
    switch (op) {
    case 16u: return fmax(x, 0.0f);
    case 17u: return 1.0f / (1.0f + exp(-x));
    case 18u: return tanh(x);
    case 19u: return fabs(x);
    case 20u: return -x;
    case 21u: return exp(x);
    case 22u: return log(x);
    case 23u: return sqrt(x);
    case 24u: return floor(x);
    case 25u: return ceil(x);
    default: return x;
    }
}

__kernel void gfx_opencl_baseline_unary_f32(__global const float* src,
                                            __global float* dst,
                                            uint count,
                                            uint op) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= count) {
        return;
    }
    dst[gid] = gfx_unary_f32(src[gid], op);
}
)CLC";

constexpr const char* kOpenClBinaryF32Source = R"CLC(
static inline float gfx_binary_f32(float lhs, float rhs, uint op) {
    switch (op) {
    case 1u: return lhs + rhs;
    case 2u: return lhs - rhs;
    case 3u: return lhs * rhs;
    case 4u: return lhs / rhs;
    case 5u: return fmax(lhs, rhs);
    case 6u: return fmin(lhs, rhs);
    case 7u: return pow(lhs, rhs);
    case 8u: {
        const float diff = lhs - rhs;
        return diff * diff;
    }
    default: return lhs;
    }
}

__kernel void gfx_opencl_baseline_binary_f32(__global const float* lhs,
                                             __global const float* rhs,
                                             __global float* dst,
                                             uint count,
                                             uint op) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= count) {
        return;
    }
    dst[gid] = gfx_binary_f32(lhs[gid], rhs[gid], op);
}
)CLC";

constexpr const char* kOpenClBinaryScalarF32Source = R"CLC(
static inline float gfx_binary_f32(float lhs, float rhs, uint op) {
    switch (op) {
    case 1u: return lhs + rhs;
    case 2u: return lhs - rhs;
    case 3u: return lhs * rhs;
    case 4u: return lhs / rhs;
    case 5u: return fmax(lhs, rhs);
    case 6u: return fmin(lhs, rhs);
    case 7u: return pow(lhs, rhs);
    case 8u: {
        const float diff = lhs - rhs;
        return diff * diff;
    }
    default: return lhs;
    }
}

__kernel void gfx_opencl_baseline_binary_scalar_f32(__global const float* lhs,
                                                    __global const float* rhs,
                                                    __global float* dst,
                                                    uint count,
                                                    uint op,
                                                    uint input_mode) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= count) {
        return;
    }
    const float l = input_mode == 2u ? lhs[0] : lhs[gid];
    const float r = input_mode == 1u ? rhs[0] : rhs[gid];
    dst[gid] = gfx_binary_f32(l, r, op);
}
)CLC";

constexpr const char* kOpenClBinaryConstF32Source = R"CLC(
static inline float gfx_binary_f32(float lhs, float rhs, uint op) {
    switch (op) {
    case 1u: return lhs + rhs;
    case 2u: return lhs - rhs;
    case 3u: return lhs * rhs;
    case 4u: return lhs / rhs;
    case 5u: return fmax(lhs, rhs);
    case 6u: return fmin(lhs, rhs);
    case 7u: return pow(lhs, rhs);
    case 8u: {
        const float diff = lhs - rhs;
        return diff * diff;
    }
    default: return lhs;
    }
}

__kernel void gfx_opencl_baseline_binary_const_f32(__global const float* tensor,
                                                   __global float* dst,
                                                   uint count,
                                                   uint op,
                                                   uint input_mode,
                                                   float scalar_value) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= count) {
        return;
    }
    const float l = input_mode == 4u ? scalar_value : tensor[gid];
    const float r = input_mode == 3u ? scalar_value : tensor[gid];
    dst[gid] = gfx_binary_f32(l, r, op);
}
)CLC";

constexpr const char* kOpenClCompareF32Source = R"CLC(
static inline uchar gfx_compare_f32(float lhs, float rhs, uint op) {
    uint result = 0u;
    if (op == 32u) {
        result = lhs == rhs ? 1u : 0u;
    } else if (op == 33u) {
        result = lhs != rhs ? 1u : 0u;
    } else if (op == 34u) {
        result = lhs > rhs ? 1u : 0u;
    } else if (op == 35u) {
        result = lhs >= rhs ? 1u : 0u;
    } else if (op == 36u) {
        result = lhs < rhs ? 1u : 0u;
    } else if (op == 37u) {
        result = lhs <= rhs ? 1u : 0u;
    }
    return (uchar)result;
}

__kernel void gfx_opencl_baseline_compare_f32(__global const float* lhs,
                                              __global const float* rhs,
                                              __global uchar* dst,
                                              uint count,
                                              uint op) {
    const uint word_idx = (uint)get_global_id(0);
    const uint base = word_idx * 4u;
    if (base >= count) {
        return;
    }
    uint packed = 0u;
    if (base < count) {
        packed |= ((uint)gfx_compare_f32(lhs[base], rhs[base], op)) << 0u;
    }
    if (base + 1u < count) {
        packed |= ((uint)gfx_compare_f32(lhs[base + 1u], rhs[base + 1u], op)) << 8u;
    }
    if (base + 2u < count) {
        packed |= ((uint)gfx_compare_f32(lhs[base + 2u], rhs[base + 2u], op)) << 16u;
    }
    if (base + 3u < count) {
        packed |= ((uint)gfx_compare_f32(lhs[base + 3u], rhs[base + 3u], op)) << 24u;
    }
    ((__global uint*)dst)[word_idx] = packed;
}
)CLC";

constexpr const char* kOpenClCompareBroadcastF32Source = R"CLC(
static inline uchar gfx_compare_f32(float lhs, float rhs, uint op) {
    uint result = 0u;
    if (op == 32u) {
        result = lhs == rhs ? 1u : 0u;
    } else if (op == 33u) {
        result = lhs != rhs ? 1u : 0u;
    } else if (op == 34u) {
        result = lhs > rhs ? 1u : 0u;
    } else if (op == 35u) {
        result = lhs >= rhs ? 1u : 0u;
    } else if (op == 36u) {
        result = lhs < rhs ? 1u : 0u;
    } else if (op == 37u) {
        result = lhs <= rhs ? 1u : 0u;
    }
    return (uchar)result;
}

static inline uint gfx_broadcast_offset(uint idx,
                                        uint rank,
                                        uint out_dim1,
                                        uint out_dim2,
                                        uint out_dim3,
                                        uint stride0,
                                        uint stride1,
                                        uint stride2,
                                        uint stride3) {
    uint coord0 = 0u;
    uint coord1 = 0u;
    uint coord2 = 0u;
    uint coord3 = 0u;
    if (rank == 1u) {
        coord0 = idx;
    } else if (rank == 2u) {
        coord0 = idx / out_dim1;
        coord1 = idx - coord0 * out_dim1;
    } else if (rank == 3u) {
        const uint plane0 = out_dim1 * out_dim2;
        const uint rem0 = idx - (idx / plane0) * plane0;
        coord0 = idx / plane0;
        coord1 = rem0 / out_dim2;
        coord2 = rem0 - coord1 * out_dim2;
    } else {
        const uint plane0 = out_dim1 * out_dim2 * out_dim3;
        const uint rem0 = idx - (idx / plane0) * plane0;
        const uint plane1 = out_dim2 * out_dim3;
        const uint rem1 = rem0 - (rem0 / plane1) * plane1;
        coord0 = idx / plane0;
        coord1 = rem0 / plane1;
        coord2 = rem1 / out_dim3;
        coord3 = rem1 - coord2 * out_dim3;
    }
    return coord0 * stride0 + coord1 * stride1 + coord2 * stride2 + coord3 * stride3;
}

__kernel void gfx_opencl_baseline_compare_broadcast_f32(__global const float* lhs,
                                                        __global const float* rhs,
                                                        __global uchar* dst,
                                                        uint count,
                                                        uint op,
                                                        uint rank,
                                                        uint out_dim0,
                                                        uint out_dim1,
                                                        uint out_dim2,
                                                        uint out_dim3,
                                                        uint lhs_stride0,
                                                        uint lhs_stride1,
                                                        uint lhs_stride2,
                                                        uint lhs_stride3,
                                                        uint rhs_stride0,
                                                        uint rhs_stride1,
                                                        uint rhs_stride2,
                                                        uint rhs_stride3) {
    const uint word_idx = (uint)get_global_id(0);
    const uint base = word_idx * 4u;
    if (base >= count) {
        return;
    }
    (void)out_dim0;

    uint packed = 0u;
    if (base < count) {
        const uint lhs_offset = gfx_broadcast_offset(base, rank, out_dim1, out_dim2, out_dim3,
                                                     lhs_stride0, lhs_stride1, lhs_stride2, lhs_stride3);
        const uint rhs_offset = gfx_broadcast_offset(base, rank, out_dim1, out_dim2, out_dim3,
                                                     rhs_stride0, rhs_stride1, rhs_stride2, rhs_stride3);
        packed |= ((uint)gfx_compare_f32(lhs[lhs_offset], rhs[rhs_offset], op)) << 0u;
    }
    if (base + 1u < count) {
        const uint idx = base + 1u;
        const uint lhs_offset = gfx_broadcast_offset(idx, rank, out_dim1, out_dim2, out_dim3,
                                                     lhs_stride0, lhs_stride1, lhs_stride2, lhs_stride3);
        const uint rhs_offset = gfx_broadcast_offset(idx, rank, out_dim1, out_dim2, out_dim3,
                                                     rhs_stride0, rhs_stride1, rhs_stride2, rhs_stride3);
        packed |= ((uint)gfx_compare_f32(lhs[lhs_offset], rhs[rhs_offset], op)) << 8u;
    }
    if (base + 2u < count) {
        const uint idx = base + 2u;
        const uint lhs_offset = gfx_broadcast_offset(idx, rank, out_dim1, out_dim2, out_dim3,
                                                     lhs_stride0, lhs_stride1, lhs_stride2, lhs_stride3);
        const uint rhs_offset = gfx_broadcast_offset(idx, rank, out_dim1, out_dim2, out_dim3,
                                                     rhs_stride0, rhs_stride1, rhs_stride2, rhs_stride3);
        packed |= ((uint)gfx_compare_f32(lhs[lhs_offset], rhs[rhs_offset], op)) << 16u;
    }
    if (base + 3u < count) {
        const uint idx = base + 3u;
        const uint lhs_offset = gfx_broadcast_offset(idx, rank, out_dim1, out_dim2, out_dim3,
                                                     lhs_stride0, lhs_stride1, lhs_stride2, lhs_stride3);
        const uint rhs_offset = gfx_broadcast_offset(idx, rank, out_dim1, out_dim2, out_dim3,
                                                     rhs_stride0, rhs_stride1, rhs_stride2, rhs_stride3);
        packed |= ((uint)gfx_compare_f32(lhs[lhs_offset], rhs[rhs_offset], op)) << 24u;
    }
    ((__global uint*)dst)[word_idx] = packed;
}
)CLC";

constexpr const char* kOpenClSelectF32Source = R"CLC(
__kernel void gfx_opencl_baseline_select_f32(__global const uchar* cond,
                                             __global const float* then_data,
                                             __global const float* else_data,
                                             __global float* dst,
                                             uint count) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= count) {
        return;
    }
    const float then_value = then_data[gid];
    const float else_value = else_data[gid];
    const float mask = convert_float(cond[gid]);
    dst[gid] = else_value + mask * (then_value - else_value);
}
)CLC";

constexpr const char* kOpenClSelectBroadcastF32Source = R"CLC(
__kernel void gfx_opencl_baseline_select_broadcast_f32(__global const uchar* cond,
                                                       __global const float* then_data,
                                                       __global const float* else_data,
                                                       __global float* dst,
                                                       uint count,
                                                       uint rank,
                                                       uint out_dim0,
                                                       uint out_dim1,
                                                       uint out_dim2,
                                                       uint out_dim3,
                                                       uint cond_stride0,
                                                       uint cond_stride1,
                                                       uint cond_stride2,
                                                       uint cond_stride3,
                                                       uint then_stride0,
                                                       uint then_stride1,
                                                       uint then_stride2,
                                                       uint then_stride3,
                                                       uint else_stride0,
                                                       uint else_stride1,
                                                       uint else_stride2,
                                                       uint else_stride3) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= count) {
        return;
    }
    (void)out_dim0;

    uint coord0 = 0u;
    uint coord1 = 0u;
    uint coord2 = 0u;
    uint coord3 = 0u;
    if (rank == 1u) {
        coord0 = gid;
    } else if (rank == 2u) {
        coord0 = gid / out_dim1;
        coord1 = gid - coord0 * out_dim1;
    } else if (rank == 3u) {
        const uint plane0 = out_dim1 * out_dim2;
        const uint rem0 = gid - (gid / plane0) * plane0;
        coord0 = gid / plane0;
        coord1 = rem0 / out_dim2;
        coord2 = rem0 - coord1 * out_dim2;
    } else {
        const uint plane0 = out_dim1 * out_dim2 * out_dim3;
        const uint rem0 = gid - (gid / plane0) * plane0;
        const uint plane1 = out_dim2 * out_dim3;
        const uint rem1 = rem0 - (rem0 / plane1) * plane1;
        coord0 = gid / plane0;
        coord1 = rem0 / plane1;
        coord2 = rem1 / out_dim3;
        coord3 = rem1 - coord2 * out_dim3;
    }
    const uint cond_offset = coord0 * cond_stride0 + coord1 * cond_stride1 +
                             coord2 * cond_stride2 + coord3 * cond_stride3;
    const uint then_offset = coord0 * then_stride0 + coord1 * then_stride1 +
                             coord2 * then_stride2 + coord3 * then_stride3;
    const uint else_offset = coord0 * else_stride0 + coord1 * else_stride1 +
                             coord2 * else_stride2 + coord3 * else_stride3;
    const float then_value = then_data[then_offset];
    const float else_value = else_data[else_offset];
    const float mask = convert_float(cond[cond_offset]);
    dst[gid] = else_value + mask * (then_value - else_value);
}
)CLC";

constexpr const char* kOpenClDynamicDataMovementF16Source = R"CLC(
#define GFX_LOW_U32_SHAPE_VALUE(words, idx) ((words)[(idx) * 2u])
#define GFX_LOAD_I32_SHAPE_VALUE(words, idx) ((int)GFX_LOW_U32_SHAPE_VALUE((words), (idx)))
#define GFX_LOAD_BOOL_MASK(src, idx) \
    (0u - (((((src)[(idx) >> 2u]) >> (((idx) & 3u) * 8u)) & 255u) != 0u))
#define GFX_LOAD_F16_BITS(src, idx) \
    (((idx) & 1u) == 0u ? ((src)[(idx) >> 1u] & 65535u) : (((src)[(idx) >> 1u] >> 16u) & 65535u))
#define GFX_STORE_F16_PAIR(dst, word_idx, lo, hi) \
    ((dst)[(word_idx)] = ((lo) & 65535u) | (((hi) & 65535u) << 16u))
#define GFX_SELECT_F16_BITS(mask, then_bits, else_bits) \
    (((else_bits) & ~(mask)) | ((then_bits) & (mask)))

static inline uint gfx_offset_from_dims(uint idx,
                                        uint rank,
                                        uint dim0,
                                        uint dim1,
                                        uint dim2,
                                        uint dim3,
                                        uint stride0,
                                        uint stride1,
                                        uint stride2,
                                        uint stride3) {
    uint coord0 = 0u;
    uint coord1 = 0u;
    uint coord2 = 0u;
    uint coord3 = 0u;
    if (rank == 1u) {
        coord0 = idx;
    } else if (rank == 2u) {
        coord0 = idx / dim1;
        coord1 = idx - coord0 * dim1;
    } else if (rank == 3u) {
        const uint plane0 = dim1 * dim2;
        const uint rem0 = idx - (idx / plane0) * plane0;
        coord0 = idx / plane0;
        coord1 = rem0 / dim2;
        coord2 = rem0 - coord1 * dim2;
    } else {
        const uint plane0 = dim1 * dim2 * dim3;
        const uint rem0 = idx - (idx / plane0) * plane0;
        const uint plane1 = dim2 * dim3;
        const uint rem1 = rem0 - (rem0 / plane1) * plane1;
        coord0 = idx / plane0;
        coord1 = rem0 / plane1;
        coord2 = rem1 / dim3;
        coord3 = rem1 - coord2 * dim3;
    }
    return coord0 * stride0 + coord1 * stride1 + coord2 * stride2 + coord3 * stride3;
}

__kernel void gfx_opencl_baseline_select_f16(__global const uint* cond,
                                             __global const uint* then_data,
                                             __global const uint* else_data,
                                             __global uint* dst,
                                             uint count) {
    const uint word_idx = (uint)get_global_id(0);
    const uint elem0 = word_idx * 2u;
    if (elem0 >= count) {
        return;
    }
    const uint lo_mask = GFX_LOAD_BOOL_MASK(cond, elem0);
    const uint lo = GFX_SELECT_F16_BITS(lo_mask,
                                        GFX_LOAD_F16_BITS(then_data, elem0),
                                        GFX_LOAD_F16_BITS(else_data, elem0));
    uint hi = 0u;
    if (elem0 + 1u < count) {
        const uint hi_mask = GFX_LOAD_BOOL_MASK(cond, elem0 + 1u);
        hi = GFX_SELECT_F16_BITS(hi_mask,
                                 GFX_LOAD_F16_BITS(then_data, elem0 + 1u),
                                 GFX_LOAD_F16_BITS(else_data, elem0 + 1u));
    }
    GFX_STORE_F16_PAIR(dst, word_idx, lo, hi);
}

#define GFX_CONCAT2_LOAD_F16_BITS(out_idx, dst_value, src0, src1, inner, src0_axis_len, src1_axis_len) \
    do { \
        const uint out_axis__ = (src0_axis_len) + (src1_axis_len); \
        const uint inner_idx__ = (out_idx) % (inner); \
        const uint axis_idx__ = ((out_idx) / (inner)) % out_axis__; \
        const uint outer_idx__ = (out_idx) / (out_axis__ * (inner)); \
        if (axis_idx__ < (src0_axis_len)) { \
            (dst_value) = GFX_LOAD_F16_BITS((src0), (outer_idx__ * (src0_axis_len) + axis_idx__) * (inner) + inner_idx__); \
        } else { \
            const uint src_axis__ = axis_idx__ - (src0_axis_len); \
            (dst_value) = GFX_LOAD_F16_BITS((src1), (outer_idx__ * (src1_axis_len) + src_axis__) * (inner) + inner_idx__); \
        } \
    } while (0)

__kernel void gfx_opencl_baseline_concat2_f16(__global const uint* src0,
                                              __global const uint* src1,
                                              __global uint* dst,
                                              uint count,
                                              uint inner,
                                              uint src0_axis_len,
                                              uint src1_axis_len) {
    const uint word_idx = (uint)get_global_id(0);
    const uint elem0 = word_idx * 2u;
    if (elem0 >= count) {
        return;
    }
    uint lo = 0u;
    GFX_CONCAT2_LOAD_F16_BITS(elem0, lo, src0, src1, inner, src0_axis_len, src1_axis_len);
    uint hi = 0u;
    if (elem0 + 1u < count) {
        GFX_CONCAT2_LOAD_F16_BITS(elem0 + 1u, hi, src0, src1, inner, src0_axis_len, src1_axis_len);
    }
    GFX_STORE_F16_PAIR(dst, word_idx, lo, hi);
}

#define GFX_CONCAT3_LOAD_F16_BITS(out_idx, dst_value, src0, src1, src2, inner, src0_axis_len, src1_axis_len, src2_axis_len) \
    do { \
        const uint out_axis__ = (src0_axis_len) + (src1_axis_len) + (src2_axis_len); \
        const uint inner_idx__ = (out_idx) % (inner); \
        const uint axis_idx__ = ((out_idx) / (inner)) % out_axis__; \
        const uint outer_idx__ = (out_idx) / (out_axis__ * (inner)); \
        if (axis_idx__ < (src0_axis_len)) { \
            (dst_value) = GFX_LOAD_F16_BITS((src0), (outer_idx__ * (src0_axis_len) + axis_idx__) * (inner) + inner_idx__); \
        } else if (axis_idx__ < (src0_axis_len) + (src1_axis_len)) { \
            const uint src_axis__ = axis_idx__ - (src0_axis_len); \
            (dst_value) = GFX_LOAD_F16_BITS((src1), (outer_idx__ * (src1_axis_len) + src_axis__) * (inner) + inner_idx__); \
        } else { \
            const uint src_axis__ = axis_idx__ - (src0_axis_len) - (src1_axis_len); \
            (dst_value) = GFX_LOAD_F16_BITS((src2), (outer_idx__ * (src2_axis_len) + src_axis__) * (inner) + inner_idx__); \
        } \
    } while (0)

__kernel void gfx_opencl_baseline_concat3_f16(__global const uint* src0,
                                              __global const uint* src1,
                                              __global const uint* src2,
                                              __global uint* dst,
                                              uint count,
                                              uint inner,
                                              uint src0_axis_len,
                                              uint src1_axis_len,
                                              uint src2_axis_len) {
    const uint word_idx = (uint)get_global_id(0);
    const uint elem0 = word_idx * 2u;
    if (elem0 >= count) {
        return;
    }
    uint lo = 0u;
    GFX_CONCAT3_LOAD_F16_BITS(elem0, lo, src0, src1, src2, inner, src0_axis_len, src1_axis_len, src2_axis_len);
    uint hi = 0u;
    if (elem0 + 1u < count) {
        GFX_CONCAT3_LOAD_F16_BITS(elem0 + 1u,
                                  hi,
                                  src0,
                                  src1,
                                  src2,
                                  inner,
                                  src0_axis_len,
                                  src1_axis_len,
                                  src2_axis_len);
    }
    GFX_STORE_F16_PAIR(dst, word_idx, lo, hi);
}

#define GFX_CONCAT4_LOAD_F16_BITS(out_idx, dst_value, src0, src1, src2, src3, inner, src0_axis_len, src1_axis_len, src2_axis_len, src3_axis_len) \
    do { \
        const uint out_axis__ = (src0_axis_len) + (src1_axis_len) + (src2_axis_len) + (src3_axis_len); \
        const uint inner_idx__ = (out_idx) % (inner); \
        const uint axis_idx__ = ((out_idx) / (inner)) % out_axis__; \
        const uint outer_idx__ = (out_idx) / (out_axis__ * (inner)); \
        if (axis_idx__ < (src0_axis_len)) { \
            (dst_value) = GFX_LOAD_F16_BITS((src0), (outer_idx__ * (src0_axis_len) + axis_idx__) * (inner) + inner_idx__); \
        } else if (axis_idx__ < (src0_axis_len) + (src1_axis_len)) { \
            const uint src_axis__ = axis_idx__ - (src0_axis_len); \
            (dst_value) = GFX_LOAD_F16_BITS((src1), (outer_idx__ * (src1_axis_len) + src_axis__) * (inner) + inner_idx__); \
        } else if (axis_idx__ < (src0_axis_len) + (src1_axis_len) + (src2_axis_len)) { \
            const uint src_axis__ = axis_idx__ - (src0_axis_len) - (src1_axis_len); \
            (dst_value) = GFX_LOAD_F16_BITS((src2), (outer_idx__ * (src2_axis_len) + src_axis__) * (inner) + inner_idx__); \
        } else { \
            const uint src_axis__ = axis_idx__ - (src0_axis_len) - (src1_axis_len) - (src2_axis_len); \
            (dst_value) = GFX_LOAD_F16_BITS((src3), (outer_idx__ * (src3_axis_len) + src_axis__) * (inner) + inner_idx__); \
        } \
    } while (0)

__kernel void gfx_opencl_baseline_concat4_f16(__global const uint* src0,
                                              __global const uint* src1,
                                              __global const uint* src2,
                                              __global const uint* src3,
                                              __global uint* dst,
                                              uint count,
                                              uint inner,
                                              uint src0_axis_len,
                                              uint src1_axis_len,
                                              uint src2_axis_len,
                                              uint src3_axis_len) {
    const uint word_idx = (uint)get_global_id(0);
    const uint elem0 = word_idx * 2u;
    if (elem0 >= count) {
        return;
    }
    uint lo = 0u;
    GFX_CONCAT4_LOAD_F16_BITS(elem0,
                              lo,
                              src0,
                              src1,
                              src2,
                              src3,
                              inner,
                              src0_axis_len,
                              src1_axis_len,
                              src2_axis_len,
                              src3_axis_len);
    uint hi = 0u;
    if (elem0 + 1u < count) {
        GFX_CONCAT4_LOAD_F16_BITS(elem0 + 1u,
                                  hi,
                                  src0,
                                  src1,
                                  src2,
                                  src3,
                                  inner,
                                  src0_axis_len,
                                  src1_axis_len,
                                  src2_axis_len,
                                  src3_axis_len);
    }
    GFX_STORE_F16_PAIR(dst, word_idx, lo, hi);
}

__kernel void gfx_opencl_baseline_broadcast_f16_i64shape(__global const uint* src,
                                                         __global const uint* target_shape_words,
                                                         __global uint* dst,
                                                         uint count,
                                                         uint rank,
                                                         uint input_rank,
                                                         uint input_dim0,
                                                         uint input_dim1,
                                                         uint input_dim2,
                                                         uint input_dim3) {
    const uint word_idx = (uint)get_global_id(0);
    const uint elem0 = word_idx * 2u;
    if (elem0 >= count) {
        return;
    }
    const uint out_dim0 = rank > 0u ? GFX_LOW_U32_SHAPE_VALUE(target_shape_words, 0u) : 1u;
    const uint out_dim1 = rank > 1u ? GFX_LOW_U32_SHAPE_VALUE(target_shape_words, 1u) : 1u;
    const uint out_dim2 = rank > 2u ? GFX_LOW_U32_SHAPE_VALUE(target_shape_words, 2u) : 1u;
    const uint out_dim3 = rank > 3u ? GFX_LOW_U32_SHAPE_VALUE(target_shape_words, 3u) : 1u;
    const uint raw_in_dim[4] = {input_dim0, input_dim1, input_dim2, input_dim3};
    const uint rank_offset = rank >= input_rank ? rank - input_rank : 0u;
    uint in_dim[4] = {1u, 1u, 1u, 1u};
    for (uint axis = 0u; axis < input_rank && axis + rank_offset < 4u; ++axis) {
        in_dim[axis + rank_offset] = raw_in_dim[axis];
    }
    uint in_stride[4] = {1u, 1u, 1u, 1u};
    for (int axis = 2; axis >= 0; --axis) {
        in_stride[(uint)axis] = in_stride[(uint)axis + 1u] * in_dim[(uint)axis + 1u];
    }
    if (in_dim[0] == 1u) {
        in_stride[0] = 0u;
    }
    if (in_dim[1] == 1u) {
        in_stride[1] = 0u;
    }
    if (in_dim[2] == 1u) {
        in_stride[2] = 0u;
    }
    if (in_dim[3] == 1u) {
        in_stride[3] = 0u;
    }
    const uint src_offset0 = gfx_offset_from_dims(elem0,
                                                  rank,
                                                  out_dim0,
                                                  out_dim1,
                                                  out_dim2,
                                                  out_dim3,
                                                  in_stride[0],
                                                  in_stride[1],
                                                  in_stride[2],
                                                  in_stride[3]);
    const uint lo = GFX_LOAD_F16_BITS(src, src_offset0);
    uint hi = 0u;
    if (elem0 + 1u < count) {
        const uint src_offset1 = gfx_offset_from_dims(elem0 + 1u,
                                                      rank,
                                                      out_dim0,
                                                      out_dim1,
                                                      out_dim2,
                                                      out_dim3,
                                                      in_stride[0],
                                                      in_stride[1],
                                                      in_stride[2],
                                                      in_stride[3]);
        hi = GFX_LOAD_F16_BITS(src, src_offset1);
    }
    GFX_STORE_F16_PAIR(dst, word_idx, lo, hi);
}

__kernel void gfx_opencl_baseline_slice_f16(__global const uint* src,
                                            __global const uint* end_words,
                                            __global uint* dst,
                                            uint count,
                                            uint rank,
                                            uint out_dim0,
                                            uint out_dim1,
                                            uint out_dim2,
                                            uint out_dim3,
                                            uint input_dim0,
                                            uint input_dim1,
                                            uint input_dim2,
                                            uint input_dim3,
                                            uint begin0,
                                            uint begin1,
                                            uint begin2,
                                            uint begin3,
                                            uint step0,
                                            uint step1,
                                            uint step2,
                                            uint step3) {
    (void)end_words;
    const uint word_idx = (uint)get_global_id(0);
    const uint elem0 = word_idx * 2u;
    if (elem0 >= count) {
        return;
    }
    const uint out_dim[4] = {out_dim0, out_dim1, out_dim2, out_dim3};
    const uint input_dim[4] = {input_dim0, input_dim1, input_dim2, input_dim3};
    const uint begin[4] = {begin0, begin1, begin2, begin3};
    const uint step[4] = {step0, step1, step2, step3};
    uint in_stride[4] = {1u, 1u, 1u, 1u};
    for (int axis = 2; axis >= 0; --axis) {
        in_stride[(uint)axis] = in_stride[(uint)axis + 1u] * input_dim[(uint)axis + 1u];
    }
    uint src_offset0 = 0u;
    uint rem0 = elem0;
    for (uint axis = 0u; axis < rank; ++axis) {
        uint suffix = 1u;
        for (uint inner_axis = axis + 1u; inner_axis < rank; ++inner_axis) {
            suffix *= out_dim[inner_axis];
        }
        const uint coord = suffix == 0u ? 0u : rem0 / suffix;
        rem0 = suffix == 0u ? 0u : rem0 - coord * suffix;
        src_offset0 += (begin[axis] + coord * step[axis]) * in_stride[axis];
    }
    const uint lo = GFX_LOAD_F16_BITS(src, src_offset0);
    uint hi = 0u;
    if (elem0 + 1u < count) {
        uint src_offset1 = 0u;
        uint rem1 = elem0 + 1u;
        for (uint axis = 0u; axis < rank; ++axis) {
            uint suffix = 1u;
            for (uint inner_axis = axis + 1u; inner_axis < rank; ++inner_axis) {
                suffix *= out_dim[inner_axis];
            }
            const uint coord = suffix == 0u ? 0u : rem1 / suffix;
            rem1 = suffix == 0u ? 0u : rem1 - coord * suffix;
            src_offset1 += (begin[axis] + coord * step[axis]) * in_stride[axis];
        }
        hi = GFX_LOAD_F16_BITS(src, src_offset1);
    }
    GFX_STORE_F16_PAIR(dst, word_idx, lo, hi);
}

__kernel void gfx_opencl_baseline_slice_v8_f16(__global const uint* src,
                                               __global const uint* starts_words,
                                               __global const uint* ends_words,
                                               __global const uint* steps_words,
                                               __global uint* dst,
                                               uint count,
                                               uint rank,
                                               uint out_dim0,
                                               uint out_dim1,
                                               uint out_dim2,
                                               uint out_dim3,
                                               uint input_dim0,
                                               uint input_dim1,
                                               uint input_dim2,
                                               uint input_dim3) {
    (void)ends_words;
    const uint word_idx = (uint)get_global_id(0);
    const uint elem0 = word_idx * 2u;
    if (elem0 >= count) {
        return;
    }
    const uint out_dim[4] = {out_dim0, out_dim1, out_dim2, out_dim3};
    const uint input_dim[4] = {input_dim0, input_dim1, input_dim2, input_dim3};
    int begin[4] = {0, 0, 0, 0};
    int step[4] = {1, 1, 1, 1};
    for (uint axis = 0u; axis < rank; ++axis) {
        begin[axis] = GFX_LOAD_I32_SHAPE_VALUE(starts_words, axis);
        step[axis] = GFX_LOAD_I32_SHAPE_VALUE(steps_words, axis);
        if (step[axis] == 0) {
            step[axis] = 1;
        }
        if (begin[axis] < 0) {
            begin[axis] += (int)input_dim[axis];
        }
        const int upper = input_dim[axis] == 0u ? 0 : (int)input_dim[axis] - 1;
        if (step[axis] < 0) {
            if (begin[axis] < 0) {
                begin[axis] = 0;
            }
            if (begin[axis] > upper) {
                begin[axis] = upper;
            }
        } else {
            if (begin[axis] < 0) {
                begin[axis] = 0;
            }
            if (begin[axis] > (int)input_dim[axis]) {
                begin[axis] = (int)input_dim[axis];
            }
        }
    }
    uint in_stride[4] = {1u, 1u, 1u, 1u};
    for (int axis = 2; axis >= 0; --axis) {
        in_stride[(uint)axis] = in_stride[(uint)axis + 1u] * input_dim[(uint)axis + 1u];
    }
    uint src_offset0 = 0u;
    uint rem0 = elem0;
    for (uint axis = 0u; axis < rank; ++axis) {
        uint suffix = 1u;
        for (uint inner_axis = axis + 1u; inner_axis < rank; ++inner_axis) {
            suffix *= out_dim[inner_axis];
        }
        const uint coord = suffix == 0u ? 0u : rem0 / suffix;
        rem0 = suffix == 0u ? 0u : rem0 - coord * suffix;
        int src_coord = begin[axis] + (int)coord * step[axis];
        if (src_coord < 0) {
            src_coord = 0;
        }
        if (input_dim[axis] != 0u && src_coord >= (int)input_dim[axis]) {
            src_coord = (int)input_dim[axis] - 1;
        }
        src_offset0 += (uint)src_coord * in_stride[axis];
    }
    const uint lo = GFX_LOAD_F16_BITS(src, src_offset0);
    uint hi = 0u;
    if (elem0 + 1u < count) {
        uint src_offset1 = 0u;
        uint rem1 = elem0 + 1u;
        for (uint axis = 0u; axis < rank; ++axis) {
            uint suffix = 1u;
            for (uint inner_axis = axis + 1u; inner_axis < rank; ++inner_axis) {
                suffix *= out_dim[inner_axis];
            }
            const uint coord = suffix == 0u ? 0u : rem1 / suffix;
            rem1 = suffix == 0u ? 0u : rem1 - coord * suffix;
            int src_coord = begin[axis] + (int)coord * step[axis];
            if (src_coord < 0) {
                src_coord = 0;
            }
            if (input_dim[axis] != 0u && src_coord >= (int)input_dim[axis]) {
                src_coord = (int)input_dim[axis] - 1;
            }
            src_offset1 += (uint)src_coord * in_stride[axis];
        }
        hi = GFX_LOAD_F16_BITS(src, src_offset1);
    }
    GFX_STORE_F16_PAIR(dst, word_idx, lo, hi);
}

__kernel void gfx_opencl_baseline_range_i64_unit(__global const uint* stop_words,
                                                 __global uint* dst,
                                                 uint count) {
    (void)stop_words;
    const uint gid = (uint)get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint word = gid * 2u;
    dst[word] = gid;
    dst[word + 1u] = 0u;
}
)CLC";

constexpr const char* kOpenClLogicalUnaryBoolSource = R"CLC(
static inline uint gfx_load_bool(__global const uchar* src, uint idx) {
    const uint word = ((__global const uint*)src)[idx >> 2u];
    return ((word >> ((idx & 3u) * 8u)) & 255u) == 0u ? 0u : 1u;
}

static inline uint gfx_logical_unary_bool(uint value, uint op) {
    if (op == 48u) {
        return value == 0u ? 1u : 0u;
    }
    return value == 0u ? 0u : 1u;
}

__kernel void gfx_opencl_baseline_logical_unary_bool(__global const uchar* src,
                                                     __global uchar* dst,
                                                     uint count,
                                                     uint op) {
    const uint word_idx = (uint)get_global_id(0);
    const uint base = word_idx * 4u;
    if (base >= count) {
        return;
    }
    uint packed = 0u;
    if (base < count) {
        packed |= gfx_logical_unary_bool(gfx_load_bool(src, base), op) << 0u;
    }
    if (base + 1u < count) {
        packed |= gfx_logical_unary_bool(gfx_load_bool(src, base + 1u), op) << 8u;
    }
    if (base + 2u < count) {
        packed |= gfx_logical_unary_bool(gfx_load_bool(src, base + 2u), op) << 16u;
    }
    if (base + 3u < count) {
        packed |= gfx_logical_unary_bool(gfx_load_bool(src, base + 3u), op) << 24u;
    }
    ((__global uint*)dst)[word_idx] = packed;
}
)CLC";

constexpr const char* kOpenClLogicalBinaryBoolSource = R"CLC(
static inline uint gfx_load_bool(__global const uchar* src, uint idx) {
    const uint word = ((__global const uint*)src)[idx >> 2u];
    return ((word >> ((idx & 3u) * 8u)) & 255u) == 0u ? 0u : 1u;
}

static inline uint gfx_logical_binary_bool(uint l, uint r, uint op) {
    uint result = 0u;
    if (op == 49u) {
        result = l & r;
    } else if (op == 50u) {
        result = l | r;
    } else if (op == 51u) {
        result = l ^ r;
    }
    return result;
}

__kernel void gfx_opencl_baseline_logical_binary_bool(__global const uchar* lhs,
                                                      __global const uchar* rhs,
                                                      __global uchar* dst,
                                                      uint count,
                                                      uint op) {
    const uint word_idx = (uint)get_global_id(0);
    const uint base = word_idx * 4u;
    if (base >= count) {
        return;
    }
    uint packed = 0u;
    if (base < count) {
        packed |= gfx_logical_binary_bool(gfx_load_bool(lhs, base),
                                          gfx_load_bool(rhs, base),
                                          op) << 0u;
    }
    if (base + 1u < count) {
        packed |= gfx_logical_binary_bool(gfx_load_bool(lhs, base + 1u),
                                          gfx_load_bool(rhs, base + 1u),
                                          op) << 8u;
    }
    if (base + 2u < count) {
        packed |= gfx_logical_binary_bool(gfx_load_bool(lhs, base + 2u),
                                          gfx_load_bool(rhs, base + 2u),
                                          op) << 16u;
    }
    if (base + 3u < count) {
        packed |= gfx_logical_binary_bool(gfx_load_bool(lhs, base + 3u),
                                          gfx_load_bool(rhs, base + 3u),
                                          op) << 24u;
    }
    ((__global uint*)dst)[word_idx] = packed;
}
)CLC";

constexpr const char* kOpenClLogicalBinaryBroadcastBoolSource = R"CLC(
static inline uint gfx_load_bool(__global const uchar* src, uint idx) {
    const uint word = ((__global const uint*)src)[idx >> 2u];
    return ((word >> ((idx & 3u) * 8u)) & 255u) == 0u ? 0u : 1u;
}

static inline uint gfx_logical_binary_bool(uint l, uint r, uint op) {
    uint result = 0u;
    if (op == 49u) {
        result = l & r;
    } else if (op == 50u) {
        result = l | r;
    } else if (op == 51u) {
        result = l ^ r;
    }
    return result;
}

static inline uint gfx_broadcast_offset(uint idx,
                                        uint rank,
                                        uint out_dim1,
                                        uint out_dim2,
                                        uint out_dim3,
                                        uint stride0,
                                        uint stride1,
                                        uint stride2,
                                        uint stride3) {
    uint coord0 = 0u;
    uint coord1 = 0u;
    uint coord2 = 0u;
    uint coord3 = 0u;
    if (rank == 1u) {
        coord0 = idx;
    } else if (rank == 2u) {
        coord0 = idx / out_dim1;
        coord1 = idx - coord0 * out_dim1;
    } else if (rank == 3u) {
        const uint plane0 = out_dim1 * out_dim2;
        const uint rem0 = idx - (idx / plane0) * plane0;
        coord0 = idx / plane0;
        coord1 = rem0 / out_dim2;
        coord2 = rem0 - coord1 * out_dim2;
    } else {
        const uint plane0 = out_dim1 * out_dim2 * out_dim3;
        const uint rem0 = idx - (idx / plane0) * plane0;
        const uint plane1 = out_dim2 * out_dim3;
        const uint rem1 = rem0 - (rem0 / plane1) * plane1;
        coord0 = idx / plane0;
        coord1 = rem0 / plane1;
        coord2 = rem1 / out_dim3;
        coord3 = rem1 - coord2 * out_dim3;
    }
    return coord0 * stride0 + coord1 * stride1 + coord2 * stride2 + coord3 * stride3;
}

__kernel void gfx_opencl_baseline_logical_binary_broadcast_bool(__global const uchar* lhs,
                                                                __global const uchar* rhs,
                                                                __global uchar* dst,
                                                                uint count,
                                                                uint op,
                                                                uint rank,
                                                                uint out_dim0,
                                                                uint out_dim1,
                                                                uint out_dim2,
                                                                uint out_dim3,
                                                                uint lhs_stride0,
                                                                uint lhs_stride1,
                                                                uint lhs_stride2,
                                                                uint lhs_stride3,
                                                                uint rhs_stride0,
                                                                uint rhs_stride1,
                                                                uint rhs_stride2,
                                                                uint rhs_stride3) {
    const uint word_idx = (uint)get_global_id(0);
    const uint base = word_idx * 4u;
    if (base >= count) {
        return;
    }
    (void)out_dim0;

    uint packed = 0u;
    if (base < count) {
        const uint lhs_offset = gfx_broadcast_offset(base, rank, out_dim1, out_dim2, out_dim3,
                                                     lhs_stride0, lhs_stride1, lhs_stride2, lhs_stride3);
        const uint rhs_offset = gfx_broadcast_offset(base, rank, out_dim1, out_dim2, out_dim3,
                                                     rhs_stride0, rhs_stride1, rhs_stride2, rhs_stride3);
        packed |= gfx_logical_binary_bool(gfx_load_bool(lhs, lhs_offset),
                                          gfx_load_bool(rhs, rhs_offset),
                                          op) << 0u;
    }
    if (base + 1u < count) {
        const uint idx = base + 1u;
        const uint lhs_offset = gfx_broadcast_offset(idx, rank, out_dim1, out_dim2, out_dim3,
                                                     lhs_stride0, lhs_stride1, lhs_stride2, lhs_stride3);
        const uint rhs_offset = gfx_broadcast_offset(idx, rank, out_dim1, out_dim2, out_dim3,
                                                     rhs_stride0, rhs_stride1, rhs_stride2, rhs_stride3);
        packed |= gfx_logical_binary_bool(gfx_load_bool(lhs, lhs_offset),
                                          gfx_load_bool(rhs, rhs_offset),
                                          op) << 8u;
    }
    if (base + 2u < count) {
        const uint idx = base + 2u;
        const uint lhs_offset = gfx_broadcast_offset(idx, rank, out_dim1, out_dim2, out_dim3,
                                                     lhs_stride0, lhs_stride1, lhs_stride2, lhs_stride3);
        const uint rhs_offset = gfx_broadcast_offset(idx, rank, out_dim1, out_dim2, out_dim3,
                                                     rhs_stride0, rhs_stride1, rhs_stride2, rhs_stride3);
        packed |= gfx_logical_binary_bool(gfx_load_bool(lhs, lhs_offset),
                                          gfx_load_bool(rhs, rhs_offset),
                                          op) << 16u;
    }
    if (base + 3u < count) {
        const uint idx = base + 3u;
        const uint lhs_offset = gfx_broadcast_offset(idx, rank, out_dim1, out_dim2, out_dim3,
                                                     lhs_stride0, lhs_stride1, lhs_stride2, lhs_stride3);
        const uint rhs_offset = gfx_broadcast_offset(idx, rank, out_dim1, out_dim2, out_dim3,
                                                     rhs_stride0, rhs_stride1, rhs_stride2, rhs_stride3);
        packed |= gfx_logical_binary_bool(gfx_load_bool(lhs, lhs_offset),
                                          gfx_load_bool(rhs, rhs_offset),
                                          op) << 24u;
    }
    ((__global uint*)dst)[word_idx] = packed;
}
)CLC";

constexpr const char* kOpenClReduceLogicalBoolSource = R"CLC(
static inline uint gfx_load_bool(__global const uchar* src, uint idx) {
    const uint word = ((__global const uint*)src)[idx >> 2u];
    return ((word >> ((idx & 3u) * 8u)) & 255u) == 0u ? 0u : 1u;
}

static inline uint gfx_reduce_output_coord(uint input_axis,
                                           uint out_axis0,
                                           uint out_axis1,
                                           uint out_axis2,
                                           uint out_axis3,
                                           uint o0,
                                           uint o1,
                                           uint o2,
                                           uint o3) {
    if (out_axis0 == input_axis) {
        return o0;
    }
    if (out_axis1 == input_axis) {
        return o1;
    }
    if (out_axis2 == input_axis) {
        return o2;
    }
    if (out_axis3 == input_axis) {
        return o3;
    }
    return 0u;
}

static inline uint gfx_reduce_logical_bool_at(__global const uchar* src,
                                              uint out_idx,
                                              uint op,
                                              uint out_rank,
                                              uint in_dim0,
                                              uint in_dim1,
                                              uint in_dim2,
                                              uint in_dim3,
                                              uint out_dim1,
                                              uint out_dim2,
                                              uint out_dim3,
                                              uint reduce_mask,
                                              uint out_axis0,
                                              uint out_axis1,
                                              uint out_axis2,
                                              uint out_axis3) {
    uint o0 = 0u;
    uint o1 = 0u;
    uint o2 = 0u;
    uint o3 = 0u;
    if (out_rank == 1u) {
        o0 = out_idx;
    } else if (out_rank == 2u) {
        o0 = out_idx / out_dim1;
        o1 = out_idx - o0 * out_dim1;
    } else if (out_rank == 3u) {
        const uint plane0 = out_dim1 * out_dim2;
        const uint rem0 = out_idx - (out_idx / plane0) * plane0;
        o0 = out_idx / plane0;
        o1 = rem0 / out_dim2;
        o2 = rem0 - o1 * out_dim2;
    } else if (out_rank == 4u) {
        const uint plane0 = out_dim1 * out_dim2 * out_dim3;
        const uint rem0 = out_idx - (out_idx / plane0) * plane0;
        const uint plane1 = out_dim2 * out_dim3;
        const uint rem1 = rem0 - (rem0 / plane1) * plane1;
        o0 = out_idx / plane0;
        o1 = rem0 / plane1;
        o2 = rem1 / out_dim3;
        o3 = rem1 - o2 * out_dim3;
    }

    const uint base0 = gfx_reduce_output_coord(0u, out_axis0, out_axis1, out_axis2, out_axis3,
                                               o0, o1, o2, o3);
    const uint base1 = gfx_reduce_output_coord(1u, out_axis0, out_axis1, out_axis2, out_axis3,
                                               o0, o1, o2, o3);
    const uint base2 = gfx_reduce_output_coord(2u, out_axis0, out_axis1, out_axis2, out_axis3,
                                               o0, o1, o2, o3);
    const uint base3 = gfx_reduce_output_coord(3u, out_axis0, out_axis1, out_axis2, out_axis3,
                                               o0, o1, o2, o3);

    const uint r0_count = (reduce_mask & 1u) != 0u ? in_dim0 : 1u;
    const uint r1_count = (reduce_mask & 2u) != 0u ? in_dim1 : 1u;
    const uint r2_count = (reduce_mask & 4u) != 0u ? in_dim2 : 1u;
    const uint r3_count = (reduce_mask & 8u) != 0u ? in_dim3 : 1u;
    uint acc = op == 64u ? 1u : 0u;
    for (uint r0 = 0u; r0 < r0_count; ++r0) {
        const uint c0 = (reduce_mask & 1u) != 0u ? r0 : base0;
        for (uint r1 = 0u; r1 < r1_count; ++r1) {
            const uint c1 = (reduce_mask & 2u) != 0u ? r1 : base1;
            for (uint r2 = 0u; r2 < r2_count; ++r2) {
                const uint c2 = (reduce_mask & 4u) != 0u ? r2 : base2;
                for (uint r3 = 0u; r3 < r3_count; ++r3) {
                    const uint c3 = (reduce_mask & 8u) != 0u ? r3 : base3;
                    const uint input_offset = ((c0 * in_dim1 + c1) * in_dim2 + c2) * in_dim3 + c3;
                    const uint v = gfx_load_bool(src, input_offset);
                    if (op == 64u) {
                        acc = acc & v;
                    } else {
                        acc = acc | v;
                    }
                }
            }
        }
    }
    return acc;
}

__kernel void gfx_opencl_baseline_reduce_logical_bool(__global const uchar* src,
                                                      __global uchar* dst,
                                                      uint count,
                                                      uint op,
                                                      uint rank,
                                                      uint out_rank,
                                                      uint in_dim0,
                                                      uint in_dim1,
                                                      uint in_dim2,
                                                      uint in_dim3,
                                                      uint out_dim0,
                                                      uint out_dim1,
                                                      uint out_dim2,
                                                      uint out_dim3,
                                                      uint reduce_mask,
                                                      uint out_axis0,
                                                      uint out_axis1,
                                                      uint out_axis2,
                                                      uint out_axis3) {
    const uint word_idx = (uint)get_global_id(0);
    const uint base = word_idx * 4u;
    if (base >= count) {
        return;
    }
    (void)rank;
    (void)out_dim0;

    uint packed = 0u;
    if (base < count) {
        packed |= gfx_reduce_logical_bool_at(src, base, op, out_rank,
                                             in_dim0, in_dim1, in_dim2, in_dim3,
                                             out_dim1, out_dim2, out_dim3,
                                             reduce_mask, out_axis0, out_axis1,
                                             out_axis2, out_axis3) << 0u;
    }
    if (base + 1u < count) {
        packed |= gfx_reduce_logical_bool_at(src, base + 1u, op, out_rank,
                                             in_dim0, in_dim1, in_dim2, in_dim3,
                                             out_dim1, out_dim2, out_dim3,
                                             reduce_mask, out_axis0, out_axis1,
                                             out_axis2, out_axis3) << 8u;
    }
    if (base + 2u < count) {
        packed |= gfx_reduce_logical_bool_at(src, base + 2u, op, out_rank,
                                             in_dim0, in_dim1, in_dim2, in_dim3,
                                             out_dim1, out_dim2, out_dim3,
                                             reduce_mask, out_axis0, out_axis1,
                                             out_axis2, out_axis3) << 16u;
    }
    if (base + 3u < count) {
        packed |= gfx_reduce_logical_bool_at(src, base + 3u, op, out_rank,
                                             in_dim0, in_dim1, in_dim2, in_dim3,
                                             out_dim1, out_dim2, out_dim3,
                                             reduce_mask, out_axis0, out_axis1,
                                             out_axis2, out_axis3) << 24u;
    }
    ((__global uint*)dst)[word_idx] = packed;
}
)CLC";

constexpr const char* kOpenClBinaryBroadcastF32Source = R"CLC(
static inline float gfx_binary_f32(float lhs, float rhs, uint op) {
    switch (op) {
    case 1u: return lhs + rhs;
    case 2u: return lhs - rhs;
    case 3u: return lhs * rhs;
    case 4u: return lhs / rhs;
    case 5u: return fmax(lhs, rhs);
    case 6u: return fmin(lhs, rhs);
    case 7u: return pow(lhs, rhs);
    case 8u: {
        const float diff = lhs - rhs;
        return diff * diff;
    }
    default: return lhs;
    }
}

__kernel void gfx_opencl_baseline_binary_broadcast_f32(__global const float* lhs,
                                                       __global const float* rhs,
                                                       __global float* dst,
                                                       uint count,
                                                       uint op,
                                                       uint rank,
                                                       uint out_dim0,
                                                       uint out_dim1,
                                                       uint out_dim2,
                                                       uint out_dim3,
                                                       uint lhs_stride0,
                                                       uint lhs_stride1,
                                                       uint lhs_stride2,
                                                       uint lhs_stride3,
                                                       uint rhs_stride0,
                                                       uint rhs_stride1,
                                                       uint rhs_stride2,
                                                       uint rhs_stride3) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= count) {
        return;
    }
    (void)out_dim0;

    uint coord0 = 0u;
    uint coord1 = 0u;
    uint coord2 = 0u;
    uint coord3 = 0u;
    if (rank == 1u) {
        coord0 = gid;
    } else if (rank == 2u) {
        coord0 = gid / out_dim1;
        coord1 = gid - coord0 * out_dim1;
    } else if (rank == 3u) {
        const uint plane0 = out_dim1 * out_dim2;
        const uint rem0 = gid - (gid / plane0) * plane0;
        coord0 = gid / plane0;
        coord1 = rem0 / out_dim2;
        coord2 = rem0 - coord1 * out_dim2;
    } else {
        const uint plane0 = out_dim1 * out_dim2 * out_dim3;
        const uint rem0 = gid - (gid / plane0) * plane0;
        const uint plane1 = out_dim2 * out_dim3;
        const uint rem1 = rem0 - (rem0 / plane1) * plane1;
        coord0 = gid / plane0;
        coord1 = rem0 / plane1;
        coord2 = rem1 / out_dim3;
        coord3 = rem1 - coord2 * out_dim3;
    }
    const uint lhs_offset = coord0 * lhs_stride0 + coord1 * lhs_stride1 +
                            coord2 * lhs_stride2 + coord3 * lhs_stride3;
    const uint rhs_offset = coord0 * rhs_stride0 + coord1 * rhs_stride1 +
                            coord2 * rhs_stride2 + coord3 * rhs_stride3;
    dst[gid] = gfx_binary_f32(lhs[lhs_offset], rhs[rhs_offset], op);
}
)CLC";

bool is_f32_tensor_type(const ov::element::Type& type) {
    return type == ov::element::f32;
}

bool is_f16_tensor_type(const ov::element::Type& type) {
    return type == ov::element::f16;
}

bool is_bool_tensor_type(const ov::element::Type& type) {
    return type == ov::element::boolean;
}

bool is_i32_tensor_type(const ov::element::Type& type) {
    return type == ov::element::i32;
}

bool is_i64_tensor_type(const ov::element::Type& type) {
    return type == ov::element::i64;
}

bool is_opencl_range_tensor_type(const ov::element::Type& type) {
    return is_f32_tensor_type(type) || is_i32_tensor_type(type) ||
           is_i64_tensor_type(type);
}

bool is_opencl_convert_tensor_type(const ov::element::Type& type) {
    return is_f32_tensor_type(type) || is_i32_tensor_type(type) ||
           is_i64_tensor_type(type);
}

const char* opencl_scalar_type_suffix(const ov::element::Type& type) {
    if (is_f32_tensor_type(type)) {
        return "f32";
    }
    if (is_i32_tensor_type(type)) {
        return "i32";
    }
    if (is_i64_tensor_type(type)) {
        return "i64";
    }
    return "unknown";
}

const char* opencl_range_type_suffix(const ov::element::Type& type) {
    return opencl_scalar_type_suffix(type);
}

bool same_static_shape(const std::shared_ptr<const ov::Node>& node,
                       size_t input_a,
                       size_t input_b) {
    if (!node || input_a >= node->get_input_size() ||
        input_b >= node->get_input_size()) {
        return false;
    }
    if (!node->get_input_partial_shape(input_a).is_static() ||
        !node->get_input_partial_shape(input_b).is_static()) {
        return false;
    }
    return node->get_input_shape(input_a) == node->get_input_shape(input_b);
}

bool same_static_element_count_input_output(const std::shared_ptr<const ov::Node>& node,
                                            size_t input_idx,
                                            size_t output_idx) {
    if (!node || input_idx >= node->get_input_size() ||
        output_idx >= node->get_output_size()) {
        return false;
    }
    if (!node->get_input_partial_shape(input_idx).is_static() ||
        !node->get_output_partial_shape(output_idx).is_static()) {
        return false;
    }
    return ov::shape_size(node->get_input_shape(input_idx)) ==
           ov::shape_size(node->get_output_shape(output_idx));
}

std::optional<GfxOpenClBaselineOp> unary_op_code(std::string_view type) {
    if (type == "Relu") return GfxOpenClBaselineOp::Relu;
    if (type == "Sigmoid") return GfxOpenClBaselineOp::Sigmoid;
    if (type == "Tanh") return GfxOpenClBaselineOp::Tanh;
    if (type == "Abs") return GfxOpenClBaselineOp::Abs;
    if (type == "Negative") return GfxOpenClBaselineOp::Negative;
    if (type == "Exp") return GfxOpenClBaselineOp::Exp;
    if (type == "Log") return GfxOpenClBaselineOp::Log;
    if (type == "Sqrt") return GfxOpenClBaselineOp::Sqrt;
    if (type == "Floor") return GfxOpenClBaselineOp::Floor;
    if (type == "Ceiling") return GfxOpenClBaselineOp::Ceiling;
    return std::nullopt;
}

std::optional<GfxOpenClBaselineOp> binary_op_code(std::string_view type) {
    if (type == "Add") return GfxOpenClBaselineOp::Add;
    if (type == "Subtract") return GfxOpenClBaselineOp::Subtract;
    if (type == "Multiply") return GfxOpenClBaselineOp::Multiply;
    if (type == "Divide") return GfxOpenClBaselineOp::Divide;
    if (type == "Maximum") return GfxOpenClBaselineOp::Maximum;
    if (type == "Minimum") return GfxOpenClBaselineOp::Minimum;
    if (type == "Power") return GfxOpenClBaselineOp::Power;
    if (type == "SquaredDifference") return GfxOpenClBaselineOp::SquaredDifference;
    return std::nullopt;
}

std::optional<GfxOpenClBaselineOp> compare_op_code(std::string_view type) {
    if (type == "Equal") return GfxOpenClBaselineOp::Equal;
    if (type == "NotEqual") return GfxOpenClBaselineOp::NotEqual;
    if (type == "Greater") return GfxOpenClBaselineOp::Greater;
    if (type == "GreaterEqual") return GfxOpenClBaselineOp::GreaterEqual;
    if (type == "Less") return GfxOpenClBaselineOp::Less;
    if (type == "LessEqual") return GfxOpenClBaselineOp::LessEqual;
    return std::nullopt;
}

std::optional<GfxOpenClBaselineOp> logical_unary_op_code(std::string_view type) {
    if (type == "LogicalNot") return GfxOpenClBaselineOp::LogicalNot;
    return std::nullopt;
}

std::optional<GfxOpenClBaselineOp> logical_binary_op_code(std::string_view type) {
    if (type == "LogicalAnd") return GfxOpenClBaselineOp::LogicalAnd;
    if (type == "LogicalOr") return GfxOpenClBaselineOp::LogicalOr;
    if (type == "LogicalXor" || type == "Xor") return GfxOpenClBaselineOp::LogicalXor;
    return std::nullopt;
}

std::optional<GfxOpenClBaselineOp> reduce_logical_op_code(std::string_view type) {
    if (type == "ReduceLogicalAnd") return GfxOpenClBaselineOp::ReduceLogicalAnd;
    if (type == "ReduceLogicalOr") return GfxOpenClBaselineOp::ReduceLogicalOr;
    return std::nullopt;
}

bool is_linear_copy_op(std::string_view type) {
    return type == "Reshape" || type == "Squeeze" || type == "Unsqueeze";
}

bool is_static_scalar_like_input(const std::shared_ptr<const ov::Node>& node,
                                 size_t input_idx) {
    if (!node || input_idx >= node->get_input_size() ||
        !node->get_input_partial_shape(input_idx).is_static()) {
        return false;
    }
    return ov::shape_size(node->get_input_shape(input_idx)) == 1;
}

bool input_static_element_count_matches_output(const std::shared_ptr<const ov::Node>& node,
                                               size_t input_idx,
                                               size_t output_idx) {
    if (!node || input_idx >= node->get_input_size() ||
        output_idx >= node->get_output_size()) {
        return false;
    }
    if (!node->get_input_partial_shape(input_idx).is_static() ||
        !node->get_output_partial_shape(output_idx).is_static()) {
        return false;
    }
    return ov::shape_size(node->get_input_shape(input_idx)) ==
           ov::shape_size(node->get_output_shape(output_idx));
}

std::optional<float> scalar_f32_constant_input(const std::shared_ptr<const ov::Node>& node,
                                               size_t input_idx) {
    if (!node || input_idx >= node->get_input_size()) {
        return std::nullopt;
    }
    auto constant = ov::as_type_ptr<const ov::op::v0::Constant>(
        node->input_value(input_idx).get_node_shared_ptr());
    if (!constant ||
        !is_f32_tensor_type(constant->get_output_element_type(0)) ||
        !constant->get_output_partial_shape(0).is_static() ||
        ov::shape_size(constant->get_output_shape(0)) != 1) {
        return std::nullopt;
    }
    const auto values = constant->cast_vector<float>();
    if (values.empty()) {
        return std::nullopt;
    }
    return values.front();
}

std::optional<std::vector<int64_t>> constant_i64_vector_input(
    const std::shared_ptr<const ov::Node>& node,
    size_t input_idx) {
    if (!node || input_idx >= node->get_input_size()) {
        return std::nullopt;
    }
    auto constant = ov::as_type_ptr<const ov::op::v0::Constant>(
        node->input_value(input_idx).get_node_shared_ptr());
    if (!constant) {
        return std::nullopt;
    }
    return constant->cast_vector<int64_t>();
}

std::optional<size_t> normalize_axis(int64_t axis, size_t rank);

bool checked_u32(uint64_t value, uint32_t& out) {
    if (value > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    out = static_cast<uint32_t>(value);
    return true;
}

uint64_t shape_product_range(const ov::Shape& shape, size_t begin, size_t end) {
    uint64_t product = 1;
    for (size_t axis = begin; axis < end; ++axis) {
        product *= shape[axis];
    }
    return product;
}

bool append_shape_u32(const ov::Shape& shape,
                      size_t max_rank,
                      std::vector<uint32_t>& values);

std::optional<uint32_t> aligned_broadcast_stride(const ov::Shape& input_shape,
                                                 const ov::Shape& output_shape,
                                                 size_t output_axis) {
    const size_t output_rank = output_shape.size();
    const size_t input_rank = input_shape.size();
    if (output_axis >= output_rank || input_rank > output_rank) {
        return std::nullopt;
    }
    if (output_axis < output_rank - input_rank) {
        return 0u;
    }
    const size_t input_axis = output_axis - (output_rank - input_rank);
    const size_t input_dim = input_shape[input_axis];
    const size_t output_dim = output_shape[output_axis];
    if (input_dim != output_dim && input_dim != 1) {
        return std::nullopt;
    }
    if (input_dim == 1 && output_dim != 1) {
        return 0u;
    }
    uint32_t stride = 0;
    if (!checked_u32(shape_product_range(input_shape, input_axis + 1, input_rank), stride)) {
        return std::nullopt;
    }
    return stride;
}

bool append_aligned_broadcast_strides_u32(const ov::Shape& input_shape,
                                          const ov::Shape& output_shape,
                                          size_t max_rank,
                                          std::vector<uint32_t>& values) {
    if (output_shape.size() > max_rank || input_shape.size() > output_shape.size()) {
        return false;
    }
    for (size_t axis = 0; axis < output_shape.size(); ++axis) {
        const auto stride = aligned_broadcast_stride(input_shape, output_shape, axis);
        if (!stride) {
            return false;
        }
        values.push_back(*stride);
    }
    values.insert(values.end(), max_rank - output_shape.size(), 0u);
    return true;
}

bool output_type_matches(const std::shared_ptr<const ov::Node>& node,
                         const ov::element::Type& output_type) {
    return node && node->get_output_size() == 1 &&
           node->get_output_element_type(0) == output_type;
}

std::optional<std::vector<uint32_t>> broadcast_static_u32_scalars(
    const std::shared_ptr<const ov::Node>& node,
    const std::vector<ov::element::Type>& input_types,
    const ov::element::Type& output_type) {
    if (!node ||
        node->get_input_size() != input_types.size() ||
        node->get_output_size() != 1 ||
        !node->get_output_partial_shape(0).is_static() ||
        !output_type_matches(node, output_type)) {
        return std::nullopt;
    }

    for (size_t input_idx = 0; input_idx < input_types.size(); ++input_idx) {
        if (!node->get_input_partial_shape(input_idx).is_static() ||
            node->get_input_element_type(input_idx) != input_types[input_idx]) {
            return std::nullopt;
        }
    }

    const auto& output_shape = node->get_output_shape(0);
    const size_t rank = output_shape.size();
    if (rank == 0 || rank > 4 || ov::shape_size(output_shape) == 0) {
        return std::nullopt;
    }

    for (size_t input_idx = 0; input_idx < input_types.size(); ++input_idx) {
        const auto& input_shape = node->get_input_shape(input_idx);
        if (input_shape.size() > rank || ov::shape_size(input_shape) == 0) {
            return std::nullopt;
        }
    }

    for (size_t axis = 0; axis < rank; ++axis) {
        if (output_shape[axis] == 0) {
            return std::nullopt;
        }
        for (size_t input_idx = 0; input_idx < input_types.size(); ++input_idx) {
            if (!aligned_broadcast_stride(node->get_input_shape(input_idx),
                                          output_shape,
                                          axis)) {
                return std::nullopt;
            }
        }
    }

    uint32_t total = 0;
    if (!checked_u32(ov::shape_size(output_shape), total)) {
        return std::nullopt;
    }
    (void)total;

    std::vector<uint32_t> scalars;
    scalars.reserve(1 + 4 + 4 * input_types.size());
    scalars.push_back(static_cast<uint32_t>(rank));
    if (!append_shape_u32(output_shape, 4, scalars)) {
        return std::nullopt;
    }
    for (size_t input_idx = 0; input_idx < input_types.size(); ++input_idx) {
        if (!append_aligned_broadcast_strides_u32(node->get_input_shape(input_idx),
                                                 output_shape,
                                                 4,
                                                 scalars)) {
            return std::nullopt;
        }
    }
    return scalars;
}

std::optional<std::vector<uint32_t>> binary_broadcast_static_u32_scalars(
    const std::shared_ptr<const ov::Node>& node) {
    return broadcast_static_u32_scalars(node,
                                        {ov::element::f32, ov::element::f32},
                                        ov::element::f32);
}

std::optional<std::vector<uint32_t>> compare_broadcast_static_u32_scalars(
    const std::shared_ptr<const ov::Node>& node) {
    return broadcast_static_u32_scalars(node,
                                        {ov::element::f32, ov::element::f32},
                                        ov::element::boolean);
}

std::optional<std::vector<uint32_t>> select_broadcast_static_u32_scalars(
    const std::shared_ptr<const ov::Node>& node) {
    return broadcast_static_u32_scalars(node,
                                        {ov::element::boolean,
                                         ov::element::f32,
                                         ov::element::f32},
                                        ov::element::f32);
}

std::optional<std::vector<uint32_t>> logical_binary_broadcast_static_u32_scalars(
    const std::shared_ptr<const ov::Node>& node) {
    return broadcast_static_u32_scalars(node,
                                        {ov::element::boolean,
                                         ov::element::boolean},
                                        ov::element::boolean);
}

std::optional<std::vector<uint32_t>> reduce_logical_static_u32_scalars(
    const std::shared_ptr<const ov::Node>& node) {
    if (!node ||
        node->get_input_size() != 2 ||
        node->get_output_size() != 1 ||
        !is_bool_tensor_type(node->get_input_element_type(0)) ||
        !is_bool_tensor_type(node->get_output_element_type(0)) ||
        !node->get_input_partial_shape(0).is_static() ||
        !node->get_output_partial_shape(0).is_static()) {
        return std::nullopt;
    }

    bool keep_dims = false;
    if (auto reduce = ov::as_type_ptr<const ov::op::v1::ReduceLogicalAnd>(node)) {
        if (!reduce->reduction_axes_constant()) {
            return std::nullopt;
        }
        keep_dims = reduce->get_keep_dims();
    } else if (auto reduce = ov::as_type_ptr<const ov::op::v1::ReduceLogicalOr>(node)) {
        if (!reduce->reduction_axes_constant()) {
            return std::nullopt;
        }
        keep_dims = reduce->get_keep_dims();
    } else {
        return std::nullopt;
    }

    const auto& input_shape = node->get_input_shape(0);
    const auto& output_shape = node->get_output_shape(0);
    const size_t rank = input_shape.size();
    const size_t out_rank = output_shape.size();
    if (rank == 0 || rank > 4 || out_rank > 4 ||
        ov::shape_size(input_shape) == 0 ||
        ov::shape_size(output_shape) == 0) {
        return std::nullopt;
    }
    for (const auto dim : input_shape) {
        if (dim == 0 || dim > std::numeric_limits<uint32_t>::max()) {
            return std::nullopt;
        }
    }
    for (const auto dim : output_shape) {
        if (dim == 0 || dim > std::numeric_limits<uint32_t>::max()) {
            return std::nullopt;
        }
    }
    uint32_t input_count = 0;
    uint32_t output_count = 0;
    if (!checked_u32(ov::shape_size(input_shape), input_count) ||
        !checked_u32(ov::shape_size(output_shape), output_count)) {
        return std::nullopt;
    }
    (void)input_count;
    (void)output_count;

    const auto axes_i64 = constant_i64_vector_input(node, 1);
    if (!axes_i64 || axes_i64->size() > rank) {
        return std::nullopt;
    }

    std::vector<bool> reduce_axes(rank, false);
    uint32_t reduce_mask = 0;
    for (const auto axis_value : *axes_i64) {
        const auto axis = normalize_axis(axis_value, rank);
        if (!axis || reduce_axes[*axis]) {
            return std::nullopt;
        }
        reduce_axes[*axis] = true;
        reduce_mask |= 1u << static_cast<uint32_t>(*axis);
    }

    ov::Shape expected_output;
    if (keep_dims) {
        expected_output.reserve(rank);
        for (size_t axis = 0; axis < rank; ++axis) {
            expected_output.push_back(reduce_axes[axis] ? 1 : input_shape[axis]);
        }
    } else {
        for (size_t axis = 0; axis < rank; ++axis) {
            if (!reduce_axes[axis]) {
                expected_output.push_back(input_shape[axis]);
            }
        }
    }
    if (expected_output != output_shape) {
        return std::nullopt;
    }

    std::vector<uint32_t> out_axis_to_input_axis;
    out_axis_to_input_axis.reserve(4);
    if (keep_dims) {
        for (size_t axis = 0; axis < rank; ++axis) {
            out_axis_to_input_axis.push_back(reduce_axes[axis]
                                                 ? 4u
                                                 : static_cast<uint32_t>(axis));
        }
    } else {
        for (size_t axis = 0; axis < rank; ++axis) {
            if (!reduce_axes[axis]) {
                out_axis_to_input_axis.push_back(static_cast<uint32_t>(axis));
            }
        }
    }
    out_axis_to_input_axis.resize(4, 4u);

    std::vector<uint32_t> scalars;
    scalars.reserve(15);
    scalars.push_back(static_cast<uint32_t>(rank));
    scalars.push_back(static_cast<uint32_t>(out_rank));
    if (!append_shape_u32(input_shape, 4, scalars) ||
        !append_shape_u32(output_shape, 4, scalars)) {
        return std::nullopt;
    }
    scalars.push_back(reduce_mask);
    scalars.insert(scalars.end(),
                   out_axis_to_input_axis.begin(),
                   out_axis_to_input_axis.end());
    return scalars;
}

std::optional<std::vector<uint32_t>> transpose_static_u32_scalars(
    const std::shared_ptr<const ov::Node>& node) {
    if (!node || node->get_input_size() != 2 ||
        !node->get_input_partial_shape(0).is_static() ||
        !node->get_output_partial_shape(0).is_static()) {
        return std::nullopt;
    }
    const auto& input_shape = node->get_input_shape(0);
    const auto& output_shape = node->get_output_shape(0);
    const size_t rank = input_shape.size();
    if (rank == 0 || rank > 4 || output_shape.size() != rank ||
        ov::shape_size(input_shape) != ov::shape_size(output_shape)) {
        return std::nullopt;
    }
    const auto perm_i64 = constant_i64_vector_input(node, 1);
    if (!perm_i64 || perm_i64->size() != rank) {
        return std::nullopt;
    }
    std::vector<bool> seen(rank, false);
    std::vector<uint32_t> perm(rank, 0);
    for (size_t axis = 0; axis < rank; ++axis) {
        const int64_t value = (*perm_i64)[axis];
        if (value < 0 || static_cast<size_t>(value) >= rank ||
            seen[static_cast<size_t>(value)]) {
            return std::nullopt;
        }
        seen[static_cast<size_t>(value)] = true;
        perm[axis] = static_cast<uint32_t>(value);
    }
    std::vector<uint32_t> out_dims(4, 1);
    std::vector<uint32_t> input_strides(4, 1);
    for (size_t axis = 0; axis < rank; ++axis) {
        if (output_shape[axis] > std::numeric_limits<uint32_t>::max() ||
            input_shape[axis] > std::numeric_limits<uint32_t>::max()) {
            return std::nullopt;
        }
        out_dims[axis] = static_cast<uint32_t>(output_shape[axis]);
    }
    uint64_t stride = 1;
    for (size_t rev = rank; rev-- > 0;) {
        if (stride > std::numeric_limits<uint32_t>::max()) {
            return std::nullopt;
        }
        input_strides[rev] = static_cast<uint32_t>(stride);
        stride *= input_shape[rev];
    }
    if (stride > std::numeric_limits<uint32_t>::max()) {
        return std::nullopt;
    }
    perm.resize(4, 0);

    std::vector<uint32_t> scalars;
    scalars.reserve(13);
    scalars.push_back(static_cast<uint32_t>(rank));
    scalars.insert(scalars.end(), out_dims.begin(), out_dims.end());
    scalars.insert(scalars.end(), input_strides.begin(), input_strides.end());
    scalars.insert(scalars.end(), perm.begin(), perm.end());
    return scalars;
}

std::optional<std::vector<uint32_t>> slice_static_u32_scalars(
    const std::shared_ptr<const ov::Node>& node) {
    auto slice = ov::as_type_ptr<const ov::op::v8::Slice>(node);
    if (!slice ||
        slice->get_input_size() < 4 ||
        slice->get_input_size() > 5 ||
        !slice->get_input_partial_shape(0).is_static() ||
        !slice->get_output_partial_shape(0).is_static() ||
        !is_f32_tensor_type(slice->get_input_element_type(0)) ||
        !is_f32_tensor_type(slice->get_output_element_type(0))) {
        return std::nullopt;
    }

    const auto& input_shape = slice->get_input_shape(0);
    const auto& output_shape = slice->get_output_shape(0);
    const size_t rank = input_shape.size();
    if (rank == 0 || rank > 4 || output_shape.size() != rank ||
        ov::shape_size(output_shape) == 0) {
        return std::nullopt;
    }

    const auto starts = constant_i64_vector_input(node, 1);
    const auto ends = constant_i64_vector_input(node, 2);
    const auto steps = constant_i64_vector_input(node, 3);
    if (!starts || !ends || !steps ||
        starts->size() != ends->size() ||
        starts->size() != steps->size()) {
        return std::nullopt;
    }

    std::vector<int64_t> axes;
    if (slice->get_input_size() == 5) {
        auto axes_i64 = constant_i64_vector_input(node, 4);
        if (!axes_i64 || axes_i64->size() != starts->size()) {
            return std::nullopt;
        }
        axes = std::move(*axes_i64);
    } else {
        if (starts->size() != rank) {
            return std::nullopt;
        }
        axes.reserve(rank);
        for (size_t axis = 0; axis < rank; ++axis) {
            axes.push_back(static_cast<int64_t>(axis));
        }
    }

    std::vector<uint32_t> out_dims(4, 1);
    std::vector<uint32_t> input_strides(4, 1);
    std::vector<uint32_t> begin(4, 0);
    std::vector<uint32_t> slice_steps(4, 1);

    for (size_t axis = 0; axis < rank; ++axis) {
        if (input_shape[axis] == 0 ||
            input_shape[axis] > std::numeric_limits<uint32_t>::max() ||
            output_shape[axis] == 0 ||
            output_shape[axis] > std::numeric_limits<uint32_t>::max()) {
            return std::nullopt;
        }
        out_dims[axis] = static_cast<uint32_t>(output_shape[axis]);
    }

    uint64_t stride = 1;
    for (size_t rev = rank; rev-- > 0;) {
        if (stride > std::numeric_limits<uint32_t>::max()) {
            return std::nullopt;
        }
        input_strides[rev] = static_cast<uint32_t>(stride);
        stride *= input_shape[rev];
    }
    if (stride > std::numeric_limits<uint32_t>::max()) {
        return std::nullopt;
    }

    std::vector<bool> seen(rank, false);
    for (size_t i = 0; i < starts->size(); ++i) {
        const auto axis = normalize_axis(axes[i], rank);
        if (!axis || seen[*axis]) {
            return std::nullopt;
        }
        seen[*axis] = true;
        const int64_t step = (*steps)[i];
        if (step <= 0 ||
            step > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
            return std::nullopt;
        }
        const int64_t dim = static_cast<int64_t>(input_shape[*axis]);
        const int64_t raw_start = (*starts)[i] < 0 ? (*starts)[i] + dim : (*starts)[i];
        if (raw_start < 0 || raw_start >= dim) {
            return std::nullopt;
        }
        int64_t raw_end = (*ends)[i] < 0 ? (*ends)[i] + dim : (*ends)[i];
        if (raw_end < 0) {
            raw_end = 0;
        }
        if (raw_end > dim) {
            raw_end = dim;
        }
        const size_t expected_axis_len =
            raw_end <= raw_start
                ? 0
                : static_cast<size_t>((raw_end - raw_start + step - 1) / step);
        if (expected_axis_len != output_shape[*axis]) {
            return std::nullopt;
        }
        const size_t last_coord = output_shape[*axis] - 1;
        const uint64_t last_input_coord =
            static_cast<uint64_t>(raw_start) +
            static_cast<uint64_t>(last_coord) * static_cast<uint64_t>(step);
        if (last_input_coord >= input_shape[*axis]) {
            return std::nullopt;
        }
        begin[*axis] = static_cast<uint32_t>(raw_start);
        slice_steps[*axis] = static_cast<uint32_t>(step);
    }
    for (size_t axis = 0; axis < rank; ++axis) {
        if (!seen[axis] && output_shape[axis] != input_shape[axis]) {
            return std::nullopt;
        }
    }

    std::vector<uint32_t> scalars;
    scalars.reserve(17);
    scalars.push_back(static_cast<uint32_t>(rank));
    scalars.insert(scalars.end(), out_dims.begin(), out_dims.end());
    scalars.insert(scalars.end(), input_strides.begin(), input_strides.end());
    scalars.insert(scalars.end(), begin.begin(), begin.end());
    scalars.insert(scalars.end(), slice_steps.begin(), slice_steps.end());
    return scalars;
}

bool mask_has_non_zero_past_rank(const std::vector<int64_t>& mask, size_t rank) {
    if (mask.size() <= rank) {
        return false;
    }
    for (size_t axis = rank; axis < mask.size(); ++axis) {
        if (mask[axis] != 0) {
            return true;
        }
    }
    return false;
}

std::optional<std::vector<uint32_t>> strided_slice_static_u32_scalars(
    const std::shared_ptr<const ov::Node>& node) {
    auto slice = ov::as_type_ptr<const ov::op::v1::StridedSlice>(node);
    if (!slice ||
        slice->get_input_size() < 3 ||
        slice->get_input_size() > 4 ||
        !slice->get_input_partial_shape(0).is_static() ||
        !slice->get_output_partial_shape(0).is_static() ||
        !is_f32_tensor_type(slice->get_input_element_type(0)) ||
        !is_f32_tensor_type(slice->get_output_element_type(0))) {
        return std::nullopt;
    }
    for (const int64_t value : slice->get_new_axis_mask()) {
        if (value != 0) {
            return std::nullopt;
        }
    }
    for (const int64_t value : slice->get_shrink_axis_mask()) {
        if (value != 0) {
            return std::nullopt;
        }
    }
    for (const int64_t value : slice->get_ellipsis_mask()) {
        if (value != 0) {
            return std::nullopt;
        }
    }

    const auto& input_shape = slice->get_input_shape(0);
    const auto& output_shape = slice->get_output_shape(0);
    const size_t rank = input_shape.size();
    if (rank == 0 || rank > 4 || output_shape.size() != rank ||
        ov::shape_size(output_shape) == 0 ||
        mask_has_non_zero_past_rank(slice->get_begin_mask(), rank) ||
        mask_has_non_zero_past_rank(slice->get_end_mask(), rank)) {
        return std::nullopt;
    }

    const auto begin_values = constant_i64_vector_input(node, 1);
    const auto end_values = constant_i64_vector_input(node, 2);
    if (!begin_values || !end_values ||
        begin_values->size() > rank ||
        end_values->size() > rank) {
        return std::nullopt;
    }
    std::vector<int64_t> strides(rank, 1);
    if (slice->get_input_size() == 4) {
        const auto stride_values = constant_i64_vector_input(node, 3);
        if (!stride_values || stride_values->size() > rank) {
            return std::nullopt;
        }
        for (size_t axis = 0; axis < stride_values->size(); ++axis) {
            strides[axis] = (*stride_values)[axis];
        }
    }

    std::vector<uint32_t> out_dims(4, 1);
    std::vector<uint32_t> input_strides(4, 1);
    std::vector<uint32_t> begin(4, 0);
    std::vector<uint32_t> slice_steps(4, 1);

    for (size_t axis = 0; axis < rank; ++axis) {
        if (input_shape[axis] == 0 ||
            input_shape[axis] > std::numeric_limits<uint32_t>::max() ||
            output_shape[axis] == 0 ||
            output_shape[axis] > std::numeric_limits<uint32_t>::max()) {
            return std::nullopt;
        }
        out_dims[axis] = static_cast<uint32_t>(output_shape[axis]);
    }

    uint64_t stride = 1;
    for (size_t rev = rank; rev-- > 0;) {
        if (stride > std::numeric_limits<uint32_t>::max()) {
            return std::nullopt;
        }
        input_strides[rev] = static_cast<uint32_t>(stride);
        stride *= input_shape[rev];
    }
    if (stride > std::numeric_limits<uint32_t>::max()) {
        return std::nullopt;
    }

    const auto& begin_mask = slice->get_begin_mask();
    const auto& end_mask = slice->get_end_mask();
    for (size_t axis = 0; axis < rank; ++axis) {
        const bool masked_begin = axis < begin_mask.size() && begin_mask[axis] != 0;
        const bool masked_end = axis < end_mask.size() && end_mask[axis] != 0;
        const int64_t step = strides[axis];
        if (step <= 0 ||
            step > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
            return std::nullopt;
        }
        const int64_t dim = static_cast<int64_t>(input_shape[axis]);
        int64_t raw_start = axis < begin_values->size() ? (*begin_values)[axis] : 0;
        raw_start = masked_begin ? 0 : (raw_start < 0 ? raw_start + dim : raw_start);
        if (raw_start < 0 || raw_start >= dim) {
            return std::nullopt;
        }
        int64_t raw_end = axis < end_values->size() ? (*end_values)[axis] : dim;
        raw_end = masked_end ? dim : (raw_end < 0 ? raw_end + dim : raw_end);
        if (raw_end < 0) {
            raw_end = 0;
        }
        if (raw_end > dim) {
            raw_end = dim;
        }

        const size_t expected_axis_len =
            raw_end <= raw_start
                ? 0
                : static_cast<size_t>((raw_end - raw_start + step - 1) / step);
        if (expected_axis_len != output_shape[axis]) {
            return std::nullopt;
        }
        const size_t last_coord = output_shape[axis] - 1;
        const uint64_t last_input_coord =
            static_cast<uint64_t>(raw_start) +
            static_cast<uint64_t>(last_coord) * static_cast<uint64_t>(step);
        if (last_input_coord >= input_shape[axis]) {
            return std::nullopt;
        }
        begin[axis] = static_cast<uint32_t>(raw_start);
        slice_steps[axis] = static_cast<uint32_t>(step);
    }

    std::vector<uint32_t> scalars;
    scalars.reserve(17);
    scalars.push_back(static_cast<uint32_t>(rank));
    scalars.insert(scalars.end(), out_dims.begin(), out_dims.end());
    scalars.insert(scalars.end(), input_strides.begin(), input_strides.end());
    scalars.insert(scalars.end(), begin.begin(), begin.end());
    scalars.insert(scalars.end(), slice_steps.begin(), slice_steps.end());
    return scalars;
}

std::optional<std::vector<uint32_t>> gather_static_u32_scalars(
    const std::shared_ptr<const ov::Node>& node) {
    auto gather = ov::as_type_ptr<const ov::op::util::GatherBase>(node);
    if (!gather ||
        gather->get_input_size() != 3 ||
        !gather->get_input_partial_shape(0).is_static() ||
        !gather->get_input_partial_shape(1).is_static() ||
        !gather->get_output_partial_shape(0).is_static() ||
        !is_f32_tensor_type(gather->get_input_element_type(0)) ||
        !is_f32_tensor_type(gather->get_output_element_type(0))) {
        return std::nullopt;
    }
    int64_t batch_dims = 0;
    if (auto gather_v7 = ov::as_type_ptr<const ov::op::v7::Gather>(node)) {
        batch_dims = gather_v7->get_batch_dims();
    } else if (auto gather_v8 = ov::as_type_ptr<const ov::op::v8::Gather>(node)) {
        batch_dims = gather_v8->get_batch_dims();
    }
    if (batch_dims != 0) {
        return std::nullopt;
    }
    const auto indices_type = gather->get_input_element_type(1);
    if (indices_type != ov::element::i32 && indices_type != ov::element::i64) {
        return std::nullopt;
    }

    const auto& data_shape = gather->get_input_shape(0);
    const auto& indices_shape = gather->get_input_shape(1);
    const auto& output_shape = gather->get_output_shape(0);
    if (data_shape.empty() ||
        ov::shape_size(indices_shape) == 0 ||
        ov::shape_size(output_shape) == 0) {
        return std::nullopt;
    }
    const auto axis_values = constant_i64_vector_input(node, 2);
    if (!axis_values || axis_values->size() != 1) {
        return std::nullopt;
    }
    const auto axis = normalize_axis(axis_values->front(), data_shape.size());
    if (!axis || data_shape[*axis] == 0) {
        return std::nullopt;
    }

    ov::Shape expected_output;
    expected_output.reserve(data_shape.size() + indices_shape.size() - 1);
    expected_output.insert(expected_output.end(), data_shape.begin(), data_shape.begin() + *axis);
    expected_output.insert(expected_output.end(), indices_shape.begin(), indices_shape.end());
    expected_output.insert(expected_output.end(), data_shape.begin() + *axis + 1, data_shape.end());
    if (expected_output != output_shape) {
        return std::nullopt;
    }

    uint32_t outer = 0;
    uint32_t inner = 0;
    uint32_t axis_dim = 0;
    uint32_t indices_count = 0;
    if (!checked_u32(shape_product_range(data_shape, 0, *axis), outer) ||
        !checked_u32(shape_product_range(data_shape, *axis + 1, data_shape.size()), inner) ||
        !checked_u32(data_shape[*axis], axis_dim) ||
        !checked_u32(ov::shape_size(indices_shape), indices_count)) {
        return std::nullopt;
    }
    return std::vector<uint32_t>{outer, inner, axis_dim, indices_count};
}

bool append_shape_u32(const ov::Shape& shape,
                      size_t max_rank,
                      std::vector<uint32_t>& values) {
    if (shape.size() > max_rank) {
        return false;
    }
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        uint32_t dim = 0;
        if (!checked_u32(shape[axis], dim)) {
            return false;
        }
        values.push_back(dim);
    }
    values.insert(values.end(), max_rank - shape.size(), 1u);
    return true;
}

bool append_strides_u32(const ov::Shape& shape,
                        size_t max_rank,
                        std::vector<uint32_t>& values) {
    if (shape.size() > max_rank) {
        return false;
    }
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        uint32_t stride = 0;
        if (!checked_u32(shape_product_range(shape, axis + 1, shape.size()), stride)) {
            return false;
        }
        values.push_back(stride);
    }
    values.insert(values.end(), max_rank - shape.size(), 1u);
    return true;
}

bool is_static_single_element_input(const std::shared_ptr<const ov::Node>& node,
                                    size_t input_idx) {
    return node &&
           input_idx < node->get_input_size() &&
           node->get_input_partial_shape(input_idx).is_static() &&
           ov::shape_size(node->get_input_shape(input_idx)) == 1;
}

bool range_has_baseline_source_artifact(const std::shared_ptr<const ov::Node>& node) {
    if (!node ||
        node->get_input_size() != 3 ||
        node->get_output_size() != 1 ||
        !node->get_output_partial_shape(0).is_static() ||
        !is_static_single_element_input(node, 0) ||
        !is_static_single_element_input(node, 1) ||
        !is_static_single_element_input(node, 2)) {
        return false;
    }
    const auto output_type = node->get_output_element_type(0);
    if (!is_opencl_range_tensor_type(output_type)) {
        return false;
    }
    if (node->get_input_element_type(0) != output_type ||
        node->get_input_element_type(1) != output_type ||
        node->get_input_element_type(2) != output_type) {
        return false;
    }
    const auto& output_shape = node->get_output_shape(0);
    return output_shape.size() == 1 && ov::shape_size(output_shape) > 0;
}

std::optional<ov::Shape> matmul_broadcast_batch_prefix(const ov::Shape& lhs,
                                                       const ov::Shape& rhs) {
    if (lhs.size() < 2 || rhs.size() < 2) {
        return std::nullopt;
    }
    const size_t lhs_batch_rank = lhs.size() - 2;
    const size_t rhs_batch_rank = rhs.size() - 2;
    const size_t out_batch_rank = std::max(lhs_batch_rank, rhs_batch_rank);
    ov::Shape out(out_batch_rank, 1);
    for (size_t i = 0; i < out_batch_rank; ++i) {
        const size_t lhs_dim = lhs_batch_rank > i ? lhs[lhs_batch_rank - 1 - i] : 1;
        const size_t rhs_dim = rhs_batch_rank > i ? rhs[rhs_batch_rank - 1 - i] : 1;
        if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
            return std::nullopt;
        }
        out[out_batch_rank - 1 - i] = std::max(lhs_dim, rhs_dim);
    }
    return out;
}

std::optional<std::vector<uint32_t>> matmul_static_u32_scalars(
    const std::shared_ptr<const ov::Node>& node) {
    auto matmul = ov::as_type_ptr<const ov::op::v0::MatMul>(node);
    if (!matmul ||
        matmul->get_input_size() != 2 ||
        matmul->get_output_size() != 1 ||
        !matmul->get_input_partial_shape(0).is_static() ||
        !matmul->get_input_partial_shape(1).is_static() ||
        !matmul->get_output_partial_shape(0).is_static() ||
        !is_f32_tensor_type(matmul->get_input_element_type(0)) ||
        !is_f32_tensor_type(matmul->get_input_element_type(1)) ||
        !is_f32_tensor_type(matmul->get_output_element_type(0))) {
        return std::nullopt;
    }

    const auto& lhs_raw = matmul->get_input_shape(0);
    const auto& rhs_raw = matmul->get_input_shape(1);
    const auto& output_shape = matmul->get_output_shape(0);
    if (lhs_raw.size() < 2 || lhs_raw.size() > 4 ||
        rhs_raw.size() < 2 || rhs_raw.size() > 4 ||
        ov::shape_size(output_shape) == 0) {
        return std::nullopt;
    }

    ov::Shape lhs_logical = lhs_raw;
    ov::Shape rhs_logical = rhs_raw;
    const size_t lhs_rank = lhs_logical.size();
    const size_t rhs_rank = rhs_logical.size();
    if (matmul->get_transpose_a()) {
        std::swap(lhs_logical[lhs_rank - 1], lhs_logical[lhs_rank - 2]);
    }
    if (matmul->get_transpose_b()) {
        std::swap(rhs_logical[rhs_rank - 1], rhs_logical[rhs_rank - 2]);
    }

    const size_t m = lhs_logical[lhs_rank - 2];
    const size_t k = lhs_logical[lhs_rank - 1];
    const size_t rhs_k = rhs_logical[rhs_rank - 2];
    const size_t n = rhs_logical[rhs_rank - 1];
    if (m == 0 || n == 0 || k == 0 || rhs_k != k) {
        return std::nullopt;
    }

    const auto batch_prefix = matmul_broadcast_batch_prefix(lhs_logical, rhs_logical);
    if (!batch_prefix) {
        return std::nullopt;
    }
    ov::Shape expected_output = *batch_prefix;
    expected_output.push_back(m);
    expected_output.push_back(n);
    if (expected_output != output_shape) {
        return std::nullopt;
    }

    const uint64_t output_batch = shape_product_range(*batch_prefix, 0, batch_prefix->size());
    const uint64_t lhs_batch = shape_product_range(lhs_logical, 0, lhs_rank - 2);
    const uint64_t rhs_batch = shape_product_range(rhs_logical, 0, rhs_rank - 2);
    if ((lhs_batch != 1 && lhs_batch != output_batch) ||
        (rhs_batch != 1 && rhs_batch != output_batch)) {
        return std::nullopt;
    }

    const uint64_t lhs_matrix_elements = static_cast<uint64_t>(m) * static_cast<uint64_t>(k);
    const uint64_t rhs_matrix_elements = static_cast<uint64_t>(k) * static_cast<uint64_t>(n);
    uint32_t m_u32 = 0;
    uint32_t n_u32 = 0;
    uint32_t k_u32 = 0;
    uint32_t lhs_batch_stride_u32 = 0;
    uint32_t rhs_batch_stride_u32 = 0;
    uint32_t lhs_row_stride_u32 = 0;
    uint32_t lhs_col_stride_u32 = 0;
    uint32_t rhs_row_stride_u32 = 0;
    uint32_t rhs_col_stride_u32 = 0;
    uint32_t output_count_u32 = 0;
    if (!checked_u32(m, m_u32) ||
        !checked_u32(n, n_u32) ||
        !checked_u32(k, k_u32) ||
        !checked_u32(lhs_batch == 1 ? 0 : lhs_matrix_elements, lhs_batch_stride_u32) ||
        !checked_u32(rhs_batch == 1 ? 0 : rhs_matrix_elements, rhs_batch_stride_u32) ||
        !checked_u32(matmul->get_transpose_a() ? 1 : k, lhs_row_stride_u32) ||
        !checked_u32(matmul->get_transpose_a() ? m : 1, lhs_col_stride_u32) ||
        !checked_u32(matmul->get_transpose_b() ? 1 : n, rhs_row_stride_u32) ||
        !checked_u32(matmul->get_transpose_b() ? k : 1, rhs_col_stride_u32) ||
        !checked_u32(ov::shape_size(output_shape), output_count_u32)) {
        return std::nullopt;
    }
    (void)output_count_u32;

    return std::vector<uint32_t>{
        m_u32,
        n_u32,
        k_u32,
        lhs_batch_stride_u32,
        rhs_batch_stride_u32,
        lhs_row_stride_u32,
        lhs_col_stride_u32,
        rhs_row_stride_u32,
        rhs_col_stride_u32,
    };
}

std::optional<std::vector<uint32_t>> softmax_static_u32_scalars(
    const std::shared_ptr<const ov::Node>& node) {
    int64_t raw_axis = 0;
    if (auto softmax_v8 = ov::as_type_ptr<const ov::op::v8::Softmax>(node)) {
        raw_axis = softmax_v8->get_axis();
    } else if (auto softmax_v1 = ov::as_type_ptr<const ov::op::v1::Softmax>(node)) {
        raw_axis = static_cast<int64_t>(softmax_v1->get_axis());
    } else {
        return std::nullopt;
    }

    if (node->get_input_size() != 1 ||
        node->get_output_size() != 1 ||
        !node->get_input_partial_shape(0).is_static() ||
        !node->get_output_partial_shape(0).is_static() ||
        !is_f32_tensor_type(node->get_input_element_type(0)) ||
        !is_f32_tensor_type(node->get_output_element_type(0))) {
        return std::nullopt;
    }

    const auto& input_shape = node->get_input_shape(0);
    const auto& output_shape = node->get_output_shape(0);
    const size_t rank = input_shape.size();
    if (rank == 0 ||
        output_shape != input_shape ||
        ov::shape_size(output_shape) == 0) {
        return std::nullopt;
    }
    const auto axis = normalize_axis(raw_axis, rank);
    if (!axis || input_shape[*axis] == 0) {
        return std::nullopt;
    }

    uint32_t outer = 0;
    uint32_t axis_dim = 0;
    uint32_t inner = 0;
    uint32_t output_count = 0;
    if (!checked_u32(shape_product_range(input_shape, 0, *axis), outer) ||
        !checked_u32(input_shape[*axis], axis_dim) ||
        !checked_u32(shape_product_range(input_shape, *axis + 1, rank), inner) ||
        !checked_u32(ov::shape_size(output_shape), output_count)) {
        return std::nullopt;
    }
    (void)output_count;

    return std::vector<uint32_t>{outer, axis_dim, inner};
}

std::optional<std::vector<uint32_t>> tile_static_u32_scalars(
    const std::shared_ptr<const ov::Node>& node) {
    auto tile = ov::as_type_ptr<const ov::op::v0::Tile>(node);
    if (!tile ||
        tile->get_input_size() != 2 ||
        tile->get_output_size() != 1 ||
        !tile->get_input_partial_shape(0).is_static() ||
        !tile->get_output_partial_shape(0).is_static() ||
        !is_f32_tensor_type(tile->get_input_element_type(0)) ||
        !is_f32_tensor_type(tile->get_output_element_type(0))) {
        return std::nullopt;
    }
    const auto& input_shape = tile->get_input_shape(0);
    const auto& output_shape = tile->get_output_shape(0);
    const size_t rank = input_shape.size();
    if (rank == 0 || rank > 4 ||
        output_shape.size() != rank ||
        ov::shape_size(input_shape) == 0 ||
        ov::shape_size(output_shape) == 0) {
        return std::nullopt;
    }
    const auto repeats = constant_i64_vector_input(node, 1);
    if (!repeats || repeats->size() != rank) {
        return std::nullopt;
    }
    for (size_t axis = 0; axis < rank; ++axis) {
        if (input_shape[axis] == 0 ||
            (*repeats)[axis] <= 0 ||
            static_cast<uint64_t>(input_shape[axis]) *
                    static_cast<uint64_t>((*repeats)[axis]) != output_shape[axis]) {
            return std::nullopt;
        }
    }
    uint32_t total = 0;
    if (!checked_u32(ov::shape_size(output_shape), total)) {
        return std::nullopt;
    }
    (void)total;

    std::vector<uint32_t> scalars;
    scalars.reserve(17);
    scalars.push_back(static_cast<uint32_t>(rank));
    if (!append_shape_u32(output_shape, 4, scalars) ||
        !append_shape_u32(input_shape, 4, scalars) ||
        !append_strides_u32(output_shape, 4, scalars) ||
        !append_strides_u32(input_shape, 4, scalars)) {
        return std::nullopt;
    }
    return scalars;
}

std::optional<std::vector<uint32_t>> gather_elements_static_u32_scalars(
    const std::shared_ptr<const ov::Node>& node) {
    auto gather = ov::as_type_ptr<const ov::op::v6::GatherElements>(node);
    if (!gather ||
        gather->get_input_size() != 2 ||
        !gather->get_input_partial_shape(0).is_static() ||
        !gather->get_input_partial_shape(1).is_static() ||
        !gather->get_output_partial_shape(0).is_static() ||
        !is_f32_tensor_type(gather->get_input_element_type(0)) ||
        !is_f32_tensor_type(gather->get_output_element_type(0))) {
        return std::nullopt;
    }
    const auto indices_type = gather->get_input_element_type(1);
    if (indices_type != ov::element::i32 && indices_type != ov::element::i64) {
        return std::nullopt;
    }

    const auto& data_shape = gather->get_input_shape(0);
    const auto& indices_shape = gather->get_input_shape(1);
    const auto& output_shape = gather->get_output_shape(0);
    const size_t rank = output_shape.size();
    if (rank == 0 || rank > 4 ||
        data_shape.size() != rank ||
        indices_shape != output_shape ||
        ov::shape_size(output_shape) == 0) {
        return std::nullopt;
    }
    const auto axis = normalize_axis(gather->get_axis(), rank);
    if (!axis || data_shape[*axis] == 0) {
        return std::nullopt;
    }
    for (size_t dim = 0; dim < rank; ++dim) {
        if (data_shape[dim] == 0 || output_shape[dim] == 0) {
            return std::nullopt;
        }
        if (dim != *axis && output_shape[dim] > data_shape[dim]) {
            return std::nullopt;
        }
    }
    uint32_t total = 0;
    if (!checked_u32(ov::shape_size(output_shape), total)) {
        return std::nullopt;
    }
    (void)total;

    std::vector<uint32_t> scalars;
    scalars.reserve(18);
    scalars.push_back(static_cast<uint32_t>(rank));
    scalars.push_back(static_cast<uint32_t>(*axis));
    if (!append_shape_u32(output_shape, 4, scalars) ||
        !append_strides_u32(output_shape, 4, scalars) ||
        !append_shape_u32(data_shape, 4, scalars) ||
        !append_strides_u32(data_shape, 4, scalars)) {
        return std::nullopt;
    }
    return scalars;
}

std::optional<std::vector<uint32_t>> gather_nd_static_u32_scalars(
    const std::shared_ptr<const ov::Node>& node) {
    auto gather = ov::as_type_ptr<const ov::op::util::GatherNDBase>(node);
    if (!gather ||
        gather->get_input_size() != 2 ||
        gather->get_batch_dims() != 0 ||
        !gather->get_input_partial_shape(0).is_static() ||
        !gather->get_input_partial_shape(1).is_static() ||
        !gather->get_output_partial_shape(0).is_static() ||
        !is_f32_tensor_type(gather->get_input_element_type(0)) ||
        !is_f32_tensor_type(gather->get_output_element_type(0))) {
        return std::nullopt;
    }
    const auto indices_type = gather->get_input_element_type(1);
    if (indices_type != ov::element::i32 && indices_type != ov::element::i64) {
        return std::nullopt;
    }

    const auto& data_shape = gather->get_input_shape(0);
    const auto& indices_shape = gather->get_input_shape(1);
    const auto& output_shape = gather->get_output_shape(0);
    const size_t data_rank = data_shape.size();
    const size_t indices_rank = indices_shape.size();
    if (data_rank == 0 || data_rank > 4 ||
        indices_rank == 0 ||
        indices_shape.back() == 0 ||
        indices_shape.back() > data_rank ||
        indices_shape.back() > 4 ||
        ov::shape_size(output_shape) == 0) {
        return std::nullopt;
    }
    const size_t index_depth = indices_shape.back();
    const size_t slice_rank = data_rank - index_depth;
    for (size_t axis = 0; axis < data_rank; ++axis) {
        if (data_shape[axis] == 0) {
            return std::nullopt;
        }
    }

    ov::Shape expected_output;
    expected_output.reserve(indices_rank - 1 + slice_rank);
    expected_output.insert(expected_output.end(), indices_shape.begin(), indices_shape.end() - 1);
    expected_output.insert(expected_output.end(), data_shape.begin() + index_depth, data_shape.end());
    if (expected_output != output_shape) {
        return std::nullopt;
    }

    uint32_t total = 0;
    uint32_t slice_size = 0;
    if (!checked_u32(ov::shape_size(output_shape), total) ||
        !checked_u32(shape_product_range(data_shape, index_depth, data_rank), slice_size) ||
        slice_size == 0) {
        return std::nullopt;
    }
    (void)total;

    std::vector<uint32_t> scalars;
    scalars.reserve(11);
    scalars.push_back(static_cast<uint32_t>(index_depth));
    scalars.push_back(static_cast<uint32_t>(slice_rank));
    scalars.push_back(slice_size);
    if (!append_shape_u32(data_shape, 4, scalars) ||
        !append_strides_u32(data_shape, 4, scalars)) {
        return std::nullopt;
    }
    return scalars;
}

std::optional<std::vector<uint32_t>> scatter_update_static_u32_scalars(
    const std::shared_ptr<const ov::Node>& node) {
    auto scatter = ov::as_type_ptr<const ov::op::v3::ScatterUpdate>(node);
    if (!scatter ||
        scatter->get_input_size() != 4 ||
        !scatter->get_input_partial_shape(0).is_static() ||
        !scatter->get_input_partial_shape(1).is_static() ||
        !scatter->get_input_partial_shape(2).is_static() ||
        !scatter->get_output_partial_shape(0).is_static() ||
        !is_f32_tensor_type(scatter->get_input_element_type(0)) ||
        !is_f32_tensor_type(scatter->get_input_element_type(2)) ||
        !is_f32_tensor_type(scatter->get_output_element_type(0))) {
        return std::nullopt;
    }
    const auto indices_type = scatter->get_input_element_type(1);
    if (indices_type != ov::element::i32 && indices_type != ov::element::i64) {
        return std::nullopt;
    }

    const auto& data_shape = scatter->get_input_shape(0);
    const auto& indices_shape = scatter->get_input_shape(1);
    const auto& updates_shape = scatter->get_input_shape(2);
    const auto& output_shape = scatter->get_output_shape(0);
    const size_t data_rank = data_shape.size();
    const size_t indices_rank = indices_shape.size();
    if (data_rank == 0 || data_rank > 4 ||
        indices_rank > 4 ||
        data_shape != output_shape ||
        ov::shape_size(output_shape) == 0 ||
        ov::shape_size(indices_shape) == 0) {
        return std::nullopt;
    }
    for (size_t axis = 0; axis < data_rank; ++axis) {
        if (data_shape[axis] == 0) {
            return std::nullopt;
        }
    }
    const auto axis_values = constant_i64_vector_input(node, 3);
    if (!axis_values || axis_values->size() != 1) {
        return std::nullopt;
    }
    const auto axis = normalize_axis(axis_values->front(), data_rank);
    if (!axis) {
        return std::nullopt;
    }

    ov::Shape expected_updates;
    expected_updates.reserve(data_rank + indices_rank - 1);
    expected_updates.insert(expected_updates.end(), data_shape.begin(), data_shape.begin() + *axis);
    expected_updates.insert(expected_updates.end(), indices_shape.begin(), indices_shape.end());
    expected_updates.insert(expected_updates.end(), data_shape.begin() + *axis + 1, data_shape.end());
    if (expected_updates != updates_shape || expected_updates.size() > 7 ||
        ov::shape_size(updates_shape) == 0) {
        return std::nullopt;
    }

    uint32_t indices_total = 0;
    if (!checked_u32(ov::shape_size(indices_shape), indices_total)) {
        return std::nullopt;
    }

    std::vector<uint32_t> scalars;
    scalars.reserve(28);
    scalars.push_back(static_cast<uint32_t>(data_rank));
    scalars.push_back(static_cast<uint32_t>(indices_rank));
    scalars.push_back(static_cast<uint32_t>(updates_shape.size()));
    scalars.push_back(static_cast<uint32_t>(*axis));
    scalars.push_back(indices_total);
    if (!append_shape_u32(data_shape, 4, scalars) ||
        !append_strides_u32(data_shape, 4, scalars) ||
        !append_shape_u32(indices_shape, 4, scalars) ||
        !append_strides_u32(indices_shape, 4, scalars) ||
        !append_strides_u32(updates_shape, 7, scalars)) {
        return std::nullopt;
    }
    return scalars;
}

bool scatter_elements_has_baseline_reduction(
    const std::shared_ptr<const ov::Node>& node) {
    if (auto scatter_v12 =
            ov::as_type_ptr<const ov::op::v12::ScatterElementsUpdate>(node)) {
        return scatter_v12->get_reduction() ==
               ov::op::v12::ScatterElementsUpdate::Reduction::NONE;
    }
    return ov::as_type_ptr<const ov::op::v3::ScatterElementsUpdate>(node) != nullptr;
}

std::optional<std::vector<uint32_t>> scatter_elements_static_u32_scalars(
    const std::shared_ptr<const ov::Node>& node) {
    auto scatter = ov::as_type_ptr<const ov::op::util::ScatterElementsUpdateBase>(node);
    if (!scatter ||
        !scatter_elements_has_baseline_reduction(node) ||
        scatter->get_input_size() != 4 ||
        !scatter->get_input_partial_shape(0).is_static() ||
        !scatter->get_input_partial_shape(1).is_static() ||
        !scatter->get_input_partial_shape(2).is_static() ||
        !scatter->get_output_partial_shape(0).is_static() ||
        !is_f32_tensor_type(scatter->get_input_element_type(0)) ||
        !is_f32_tensor_type(scatter->get_input_element_type(2)) ||
        !is_f32_tensor_type(scatter->get_output_element_type(0))) {
        return std::nullopt;
    }
    const auto indices_type = scatter->get_input_element_type(1);
    if (indices_type != ov::element::i32 && indices_type != ov::element::i64) {
        return std::nullopt;
    }

    const auto& data_shape = scatter->get_input_shape(0);
    const auto& indices_shape = scatter->get_input_shape(1);
    const auto& updates_shape = scatter->get_input_shape(2);
    const auto& output_shape = scatter->get_output_shape(0);
    const size_t rank = data_shape.size();
    if (rank == 0 || rank > 4 ||
        data_shape != output_shape ||
        indices_shape != updates_shape ||
        indices_shape.size() != rank ||
        ov::shape_size(output_shape) == 0 ||
        ov::shape_size(updates_shape) == 0) {
        return std::nullopt;
    }
    const auto axis_values = constant_i64_vector_input(node, 3);
    if (!axis_values || axis_values->size() != 1) {
        return std::nullopt;
    }
    const auto axis = normalize_axis(axis_values->front(), rank);
    if (!axis || data_shape[*axis] == 0) {
        return std::nullopt;
    }
    for (size_t dim = 0; dim < rank; ++dim) {
        if (data_shape[dim] == 0 || indices_shape[dim] == 0) {
            return std::nullopt;
        }
        if (dim != *axis && indices_shape[dim] > data_shape[dim]) {
            return std::nullopt;
        }
    }

    uint32_t update_count = 0;
    if (!checked_u32(ov::shape_size(updates_shape), update_count)) {
        return std::nullopt;
    }

    std::vector<uint32_t> scalars;
    scalars.reserve(19);
    scalars.push_back(static_cast<uint32_t>(rank));
    scalars.push_back(static_cast<uint32_t>(*axis));
    scalars.push_back(update_count);
    if (!append_shape_u32(updates_shape, 4, scalars) ||
        !append_strides_u32(updates_shape, 4, scalars) ||
        !append_shape_u32(data_shape, 4, scalars) ||
        !append_strides_u32(data_shape, 4, scalars)) {
        return std::nullopt;
    }
    return scalars;
}

bool scatter_nd_has_baseline_reduction(
    const std::shared_ptr<const ov::Node>& node) {
    if (auto scatter_v15 =
            ov::as_type_ptr<const ov::op::v15::ScatterNDUpdate>(node)) {
        return scatter_v15->get_reduction() ==
               ov::op::v15::ScatterNDUpdate::Reduction::NONE;
    }
    return ov::as_type_ptr<const ov::op::v3::ScatterNDUpdate>(node) != nullptr;
}

std::optional<std::vector<uint32_t>> scatter_nd_static_u32_scalars(
    const std::shared_ptr<const ov::Node>& node) {
    auto scatter = ov::as_type_ptr<const ov::op::util::ScatterNDBase>(node);
    if (!scatter ||
        !scatter_nd_has_baseline_reduction(node) ||
        scatter->get_input_size() != 3 ||
        !scatter->get_input_partial_shape(0).is_static() ||
        !scatter->get_input_partial_shape(1).is_static() ||
        !scatter->get_input_partial_shape(2).is_static() ||
        !scatter->get_output_partial_shape(0).is_static() ||
        !is_f32_tensor_type(scatter->get_input_element_type(0)) ||
        !is_f32_tensor_type(scatter->get_input_element_type(2)) ||
        !is_f32_tensor_type(scatter->get_output_element_type(0))) {
        return std::nullopt;
    }
    const auto indices_type = scatter->get_input_element_type(1);
    if (indices_type != ov::element::i32 && indices_type != ov::element::i64) {
        return std::nullopt;
    }

    const auto& data_shape = scatter->get_input_shape(0);
    const auto& indices_shape = scatter->get_input_shape(1);
    const auto& updates_shape = scatter->get_input_shape(2);
    const auto& output_shape = scatter->get_output_shape(0);
    const size_t data_rank = data_shape.size();
    const size_t indices_rank = indices_shape.size();
    if (data_rank == 0 || data_rank > 4 ||
        data_shape != output_shape ||
        indices_rank == 0 ||
        indices_shape.back() == 0 ||
        indices_shape.back() > data_rank ||
        indices_shape.back() > 4 ||
        ov::shape_size(output_shape) == 0 ||
        ov::shape_size(updates_shape) == 0) {
        return std::nullopt;
    }
    for (size_t axis = 0; axis < data_rank; ++axis) {
        if (data_shape[axis] == 0) {
            return std::nullopt;
        }
    }

    const size_t index_depth = indices_shape.back();
    ov::Shape expected_updates;
    expected_updates.reserve(indices_rank - 1 + data_rank - index_depth);
    expected_updates.insert(expected_updates.end(),
                            indices_shape.begin(),
                            indices_shape.end() - 1);
    expected_updates.insert(expected_updates.end(),
                            data_shape.begin() + index_depth,
                            data_shape.end());
    if (expected_updates != updates_shape) {
        return std::nullopt;
    }

    uint32_t slice_size = 0;
    uint32_t tuple_count = 0;
    if (!checked_u32(shape_product_range(data_shape, index_depth, data_rank), slice_size) ||
        !checked_u32(shape_product_range(indices_shape, 0, indices_rank - 1), tuple_count) ||
        slice_size == 0 ||
        tuple_count == 0) {
        return std::nullopt;
    }

    std::vector<uint32_t> scalars;
    scalars.reserve(11);
    scalars.push_back(static_cast<uint32_t>(index_depth));
    scalars.push_back(slice_size);
    scalars.push_back(tuple_count);
    if (!append_shape_u32(data_shape, 4, scalars) ||
        !append_strides_u32(data_shape, 4, scalars)) {
        return std::nullopt;
    }
    return scalars;
}

std::optional<size_t> shapeof_rank(const std::shared_ptr<const ov::Node>& node) {
    auto shape_of = ov::as_type_ptr<const ov::op::util::ShapeOfBase>(node);
    if (!shape_of ||
        shape_of->get_input_size() != 1 ||
        shape_of->get_output_size() != 1 ||
        !shape_of->get_input_partial_shape(0).rank().is_static() ||
        !shape_of->get_output_partial_shape(0).is_static()) {
        return std::nullopt;
    }
    const auto output_type = shape_of->get_output_element_type(0);
    if (!is_i32_tensor_type(output_type) && !is_i64_tensor_type(output_type)) {
        return std::nullopt;
    }

    const auto& output_shape = shape_of->get_output_shape(0);
    const size_t rank =
        static_cast<size_t>(shape_of->get_input_partial_shape(0).rank().get_length());
    if (rank == 0 || rank > 8 || output_shape.size() != 1 ||
        output_shape[0] != rank) {
        return std::nullopt;
    }

    return rank;
}

std::optional<size_t> normalize_axis(int64_t axis, size_t rank) {
    if (rank == 0) {
        return std::nullopt;
    }
    const int64_t signed_rank = static_cast<int64_t>(rank);
    const int64_t normalized = axis < 0 ? axis + signed_rank : axis;
    if (normalized < 0 || normalized >= signed_rank) {
        return std::nullopt;
    }
    return static_cast<size_t>(normalized);
}

std::optional<size_t> static_rank(const ov::PartialShape& shape) {
    if (shape.rank().is_dynamic()) {
        return std::nullopt;
    }
    return static_cast<size_t>(shape.rank().get_length());
}

bool partial_dim_compatible(const ov::Dimension& lhs, const ov::Dimension& rhs) {
    return lhs.is_dynamic() || rhs.is_dynamic() || lhs.get_length() == rhs.get_length();
}

std::optional<uint32_t> static_partial_dim_u32(const ov::PartialShape& shape, size_t axis) {
    if (axis >= shape.size() || shape[axis].is_dynamic()) {
        return std::nullopt;
    }
    const auto dim = shape[axis].get_length();
    if (dim <= 0 || dim > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        return std::nullopt;
    }
    return static_cast<uint32_t>(dim);
}

std::optional<uint32_t> static_partial_product_u32(const ov::PartialShape& shape,
                                                   size_t begin,
                                                   size_t end) {
    if (shape.rank().is_dynamic() || begin > end || end > shape.size()) {
        return std::nullopt;
    }
    uint64_t product = 1;
    for (size_t axis = begin; axis < end; ++axis) {
        if (shape[axis].is_dynamic()) {
            return std::nullopt;
        }
        const auto dim = shape[axis].get_length();
        if (dim <= 0) {
            return std::nullopt;
        }
        product *= static_cast<uint64_t>(dim);
        if (product > std::numeric_limits<uint32_t>::max()) {
            return std::nullopt;
        }
    }
    return static_cast<uint32_t>(product);
}

bool partial_same_rank_shapes_compatible(const std::shared_ptr<const ov::Node>& node,
                                         const std::vector<size_t>& inputs,
                                         size_t output_idx) {
    if (!node || output_idx >= node->get_output_size()) {
        return false;
    }
    const auto output_rank = static_rank(node->get_output_partial_shape(output_idx));
    if (!output_rank || *output_rank == 0 || *output_rank > 4) {
        return false;
    }
    const auto& output_shape = node->get_output_partial_shape(output_idx);
    for (const size_t input_idx : inputs) {
        if (input_idx >= node->get_input_size()) {
            return false;
        }
        const auto input_rank = static_rank(node->get_input_partial_shape(input_idx));
        if (!input_rank || *input_rank != *output_rank) {
            return false;
        }
        const auto& input_shape = node->get_input_partial_shape(input_idx);
        for (size_t axis = 0; axis < *output_rank; ++axis) {
            if (!partial_dim_compatible(input_shape[axis], output_shape[axis])) {
                return false;
            }
        }
    }
    return true;
}

GfxOpenClSourceScalarArg input_dim_scalar_arg(size_t input_idx, size_t axis) {
    return static_cast<GfxOpenClSourceScalarArg>(
        static_cast<uint32_t>(GfxOpenClSourceScalarArg::Input0Dim0) +
        static_cast<uint32_t>(input_idx * 8 + axis));
}

GfxOpenClSourceScalarArg output0_dim_scalar_arg(size_t axis) {
    return static_cast<GfxOpenClSourceScalarArg>(
        static_cast<uint32_t>(GfxOpenClSourceScalarArg::Output0Dim0) +
        static_cast<uint32_t>(axis));
}

struct ConcatStaticU32Scalars {
    uint32_t input_count = 0;
    std::vector<uint32_t> values;
};

std::optional<ConcatStaticU32Scalars> concat_static_u32_scalars(
    const std::shared_ptr<const ov::Node>& node) {
    auto concat = ov::as_type_ptr<const ov::op::v0::Concat>(node);
    if (!concat ||
        !concat->get_output_partial_shape(0).is_static()) {
        return std::nullopt;
    }
    const size_t input_count = concat->get_input_size();
    if (input_count < 2 || input_count > 4) {
        return std::nullopt;
    }
    for (size_t input_idx = 0; input_idx < input_count; ++input_idx) {
        if (!concat->get_input_partial_shape(input_idx).is_static() ||
            !is_f32_tensor_type(concat->get_input_element_type(input_idx))) {
            return std::nullopt;
        }
    }
    if (!is_f32_tensor_type(concat->get_output_element_type(0))) {
        return std::nullopt;
    }
    const auto& output_shape = concat->get_output_shape(0);
    const size_t rank = output_shape.size();
    if (rank == 0) {
        return std::nullopt;
    }
    const auto axis = normalize_axis(concat->get_axis(), rank);
    if (!axis) {
        return std::nullopt;
    }
    uint64_t inner = 1;
    for (size_t dim = *axis + 1; dim < rank; ++dim) {
        inner *= output_shape[dim];
    }
    if (inner == 0 ||
        inner > std::numeric_limits<uint32_t>::max() ||
        output_shape[*axis] == 0 ||
        output_shape[*axis] > std::numeric_limits<uint32_t>::max()) {
        return std::nullopt;
    }

    size_t axis_total = 0;
    std::vector<uint32_t> axis_lengths;
    axis_lengths.reserve(input_count);
    for (size_t input_idx = 0; input_idx < input_count; ++input_idx) {
        const auto& input_shape = concat->get_input_shape(input_idx);
        if (input_shape.size() != rank) {
            return std::nullopt;
        }
        for (size_t dim = 0; dim < rank; ++dim) {
            if (dim != *axis && input_shape[dim] != output_shape[dim]) {
                return std::nullopt;
            }
        }
        if (input_shape[*axis] == 0 ||
            input_shape[*axis] > std::numeric_limits<uint32_t>::max()) {
            return std::nullopt;
        }
        axis_total += input_shape[*axis];
        axis_lengths.push_back(static_cast<uint32_t>(input_shape[*axis]));
    }
    if (axis_total != output_shape[*axis]) {
        return std::nullopt;
    }

    ConcatStaticU32Scalars metadata;
    metadata.input_count = static_cast<uint32_t>(input_count);
    metadata.values.reserve(2 + input_count * 2);
    metadata.values.push_back(static_cast<uint32_t>(output_shape[*axis]));
    metadata.values.push_back(static_cast<uint32_t>(inner));
    uint32_t offset = 0;
    for (uint32_t len : axis_lengths) {
        metadata.values.push_back(offset);
        metadata.values.push_back(len);
        offset += len;
    }
    return metadata;
}

struct ConcatDynamicScalars {
    uint32_t input_count = 0;
    uint32_t axis = 0;
    uint32_t inner = 1;
};

std::optional<ConcatDynamicScalars> concat_dynamic_f16_scalars(
    const std::shared_ptr<const ov::Node>& node) {
    auto concat = ov::as_type_ptr<const ov::op::v0::Concat>(node);
    if (!concat || concat->get_output_size() != 1 ||
        concat->get_output_partial_shape(0).is_static() ||
        !is_f16_tensor_type(concat->get_output_element_type(0))) {
        return std::nullopt;
    }
    const size_t input_count = concat->get_input_size();
    if (input_count < 2 || input_count > 4) {
        return std::nullopt;
    }
    const auto rank = static_rank(concat->get_output_partial_shape(0));
    if (!rank || *rank == 0 || *rank > 4) {
        return std::nullopt;
    }
    const auto axis = normalize_axis(concat->get_axis(), *rank);
    if (!axis) {
        return std::nullopt;
    }
    const auto inner = static_partial_product_u32(
        concat->get_output_partial_shape(0),
        *axis + 1,
        *rank);
    if (!inner || *inner == 0) {
        return std::nullopt;
    }
    const auto& output_shape = concat->get_output_partial_shape(0);
    for (size_t input_idx = 0; input_idx < input_count; ++input_idx) {
        if (!is_f16_tensor_type(concat->get_input_element_type(input_idx))) {
            return std::nullopt;
        }
        const auto input_rank = static_rank(concat->get_input_partial_shape(input_idx));
        if (!input_rank || *input_rank != *rank) {
            return std::nullopt;
        }
        const auto& input_shape = concat->get_input_partial_shape(input_idx);
        for (size_t dim = 0; dim < *rank; ++dim) {
            if (dim == *axis) {
                continue;
            }
            if (!partial_dim_compatible(input_shape[dim], output_shape[dim])) {
                return std::nullopt;
            }
        }
    }
    ConcatDynamicScalars metadata;
    metadata.input_count = static_cast<uint32_t>(input_count);
    metadata.axis = static_cast<uint32_t>(*axis);
    metadata.inner = *inner;
    return metadata;
}

bool select_dynamic_f16_supported(const std::shared_ptr<const ov::Node>& node) {
    return node &&
           node->get_type_name() == std::string("Select") &&
           node->get_input_size() == 3 &&
           node->get_output_size() == 1 &&
           !node->get_output_partial_shape(0).is_static() &&
           is_bool_tensor_type(node->get_input_element_type(0)) &&
           is_f16_tensor_type(node->get_input_element_type(1)) &&
           is_f16_tensor_type(node->get_input_element_type(2)) &&
           is_f16_tensor_type(node->get_output_element_type(0)) &&
           partial_same_rank_shapes_compatible(node, {0, 1, 2}, 0);
}

struct BroadcastDynamicScalars {
    uint32_t rank = 0;
    uint32_t input_rank = 0;
};

std::optional<BroadcastDynamicScalars> broadcast_dynamic_f16_scalars(
    const std::shared_ptr<const ov::Node>& node) {
    auto broadcast = ov::as_type_ptr<const ov::op::v3::Broadcast>(node);
    if (!broadcast ||
        broadcast->get_input_size() != 2 ||
        broadcast->get_output_size() != 1 ||
        broadcast->get_output_partial_shape(0).is_static() ||
        !is_f16_tensor_type(broadcast->get_input_element_type(0)) ||
        !is_f16_tensor_type(broadcast->get_output_element_type(0))) {
        return std::nullopt;
    }
    const auto spec = broadcast->get_broadcast_spec();
    if (spec.m_type != ov::op::BroadcastType::BIDIRECTIONAL &&
        spec.m_type != ov::op::BroadcastType::NUMPY) {
        return std::nullopt;
    }
    const auto target_type = broadcast->get_input_element_type(1);
    if (target_type != ov::element::i64) {
        return std::nullopt;
    }
    const auto target_shape = broadcast->get_input_partial_shape(1);
    if (!target_shape.is_static() ||
        target_shape.size() != 1 ||
        target_shape[0].is_dynamic() ||
        target_shape[0].get_length() <= 0 ||
        target_shape[0].get_length() > 4) {
        return std::nullopt;
    }
    const auto output_rank = static_rank(broadcast->get_output_partial_shape(0));
    const auto input_rank = static_rank(broadcast->get_input_partial_shape(0));
    if (!output_rank || !input_rank ||
        *output_rank != static_cast<size_t>(target_shape[0].get_length()) ||
        *input_rank > *output_rank ||
        *output_rank == 0 ||
        *output_rank > 4) {
        return std::nullopt;
    }
    BroadcastDynamicScalars metadata;
    metadata.rank = static_cast<uint32_t>(*output_rank);
    metadata.input_rank = static_cast<uint32_t>(*input_rank);
    return metadata;
}

struct SliceDynamicScalars {
    uint32_t rank = 0;
    std::vector<uint32_t> begin;
    std::vector<uint32_t> steps;
};

std::optional<SliceDynamicScalars> slice_v8_dynamic_f16_scalars(
    const std::shared_ptr<const ov::Node>& node) {
    auto slice = ov::as_type_ptr<const ov::op::v8::Slice>(node);
    if (!slice ||
        slice->get_input_size() < 4 ||
        slice->get_input_size() > 5 ||
        slice->get_output_size() != 1 ||
        slice->get_output_partial_shape(0).is_static() ||
        !is_f16_tensor_type(slice->get_input_element_type(0)) ||
        !is_f16_tensor_type(slice->get_output_element_type(0)) ||
        slice->get_input_element_type(1) != ov::element::i64 ||
        slice->get_input_element_type(2) != ov::element::i64 ||
        slice->get_input_element_type(3) != ov::element::i64) {
        return std::nullopt;
    }
    const auto input_rank = static_rank(slice->get_input_partial_shape(0));
    const auto output_rank = static_rank(slice->get_output_partial_shape(0));
    if (!input_rank || !output_rank || *input_rank != *output_rank ||
        *input_rank == 0 || *input_rank > 4) {
        return std::nullopt;
    }
    if (slice->get_input_size() == 5) {
        const auto axes = constant_i64_vector_input(node, 4);
        if (!axes || axes->size() != *input_rank) {
            return std::nullopt;
        }
        for (size_t axis_idx = 0; axis_idx < axes->size(); ++axis_idx) {
            const auto axis = normalize_axis((*axes)[axis_idx], *input_rank);
            if (!axis || *axis != axis_idx) {
                return std::nullopt;
            }
        }
    }
    for (size_t input_idx = 1; input_idx <= 3; ++input_idx) {
        const auto shape = slice->get_input_partial_shape(input_idx);
        if (!shape.is_static() ||
            shape.size() != 1 ||
            shape[0].is_dynamic() ||
            static_cast<size_t>(shape[0].get_length()) != *input_rank) {
            return std::nullopt;
        }
    }

    SliceDynamicScalars metadata;
    metadata.rank = static_cast<uint32_t>(*input_rank);
    return metadata;
}

std::optional<SliceDynamicScalars> strided_slice_dynamic_f16_scalars(
    const std::shared_ptr<const ov::Node>& node) {
    auto slice = ov::as_type_ptr<const ov::op::v1::StridedSlice>(node);
    if (!slice ||
        slice->get_input_size() < 3 ||
        slice->get_input_size() > 4 ||
        slice->get_output_size() != 1 ||
        slice->get_output_partial_shape(0).is_static() ||
        !is_f16_tensor_type(slice->get_input_element_type(0)) ||
        !is_f16_tensor_type(slice->get_output_element_type(0))) {
        return std::nullopt;
    }
    for (const int64_t value : slice->get_new_axis_mask()) {
        if (value != 0) {
            return std::nullopt;
        }
    }
    for (const int64_t value : slice->get_shrink_axis_mask()) {
        if (value != 0) {
            return std::nullopt;
        }
    }
    for (const int64_t value : slice->get_ellipsis_mask()) {
        if (value != 0) {
            return std::nullopt;
        }
    }
    const auto input_rank = static_rank(slice->get_input_partial_shape(0));
    const auto output_rank = static_rank(slice->get_output_partial_shape(0));
    if (!input_rank || !output_rank || *input_rank != *output_rank ||
        *input_rank == 0 || *input_rank > 4 ||
        mask_has_non_zero_past_rank(slice->get_begin_mask(), *input_rank) ||
        mask_has_non_zero_past_rank(slice->get_end_mask(), *input_rank)) {
        return std::nullopt;
    }
    const auto end_shape = slice->get_input_partial_shape(2);
    if (!end_shape.is_static() ||
        end_shape.size() != 1 ||
        end_shape[0].is_dynamic() ||
        static_cast<size_t>(end_shape[0].get_length()) != *input_rank ||
        (slice->get_input_element_type(2) != ov::element::i64 &&
         slice->get_input_element_type(2) != ov::element::i32)) {
        return std::nullopt;
    }
    const auto begin_values = constant_i64_vector_input(node, 1);
    if (!begin_values || begin_values->size() > *input_rank) {
        return std::nullopt;
    }
    std::vector<int64_t> strides(*input_rank, 1);
    if (slice->get_input_size() == 4) {
        const auto stride_values = constant_i64_vector_input(node, 3);
        if (!stride_values || stride_values->size() > *input_rank) {
            return std::nullopt;
        }
        for (size_t axis = 0; axis < stride_values->size(); ++axis) {
            strides[axis] = (*stride_values)[axis];
        }
    }

    SliceDynamicScalars metadata;
    metadata.rank = static_cast<uint32_t>(*input_rank);
    metadata.begin.assign(4, 0u);
    metadata.steps.assign(4, 1u);
    for (size_t axis = 0; axis < *input_rank; ++axis) {
        const int64_t begin = axis < begin_values->size() ? (*begin_values)[axis] : 0;
        if (begin < 0 ||
            begin > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
            strides[axis] <= 0 ||
            strides[axis] > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
            return std::nullopt;
        }
        metadata.begin[axis] = static_cast<uint32_t>(begin);
        metadata.steps[axis] = static_cast<uint32_t>(strides[axis]);
    }
    return metadata;
}

bool range_dynamic_i64_unit_supported(const std::shared_ptr<const ov::Node>& node) {
    if (!node || node->get_type_name() != std::string("Range") ||
        node->get_input_size() != 3 ||
        node->get_output_size() != 1 ||
        node->get_output_partial_shape(0).is_static() ||
        node->get_output_element_type(0) != ov::element::i64 ||
        node->get_input_element_type(1) != ov::element::i64 ||
        !is_static_single_element_input(node, 1)) {
        return false;
    }
    const auto rank = static_rank(node->get_output_partial_shape(0));
    if (!rank || *rank != 1) {
        return false;
    }
    const auto start = constant_i64_vector_input(node, 0);
    const auto step = constant_i64_vector_input(node, 2);
    return start && step &&
           start->size() == 1 &&
           step->size() == 1 &&
           (*start)[0] == 0 &&
           (*step)[0] == 1;
}

struct SplitStaticU32Scalars {
    uint32_t output_count = 0;
    std::vector<uint32_t> values;
};

std::optional<SplitStaticU32Scalars> split_static_u32_scalars_from_outputs(
    const std::shared_ptr<const ov::Node>& node,
    size_t data_input_idx,
    size_t axis_input_idx,
    size_t output_count,
    bool require_equal_axis_lengths,
    const std::vector<int64_t>* explicit_axis_lengths) {
    if (!node ||
        data_input_idx >= node->get_input_size() ||
        axis_input_idx >= node->get_input_size() ||
        !node->get_input_partial_shape(data_input_idx).is_static()) {
        return std::nullopt;
    }
    if (output_count < 2 || output_count > 4 ||
        node->get_output_size() != output_count) {
        return std::nullopt;
    }
    if (explicit_axis_lengths && explicit_axis_lengths->size() != output_count) {
        return std::nullopt;
    }
    if (!is_f32_tensor_type(node->get_input_element_type(data_input_idx))) {
        return std::nullopt;
    }
    for (size_t output_idx = 0; output_idx < output_count; ++output_idx) {
        if (!node->get_output_partial_shape(output_idx).is_static() ||
            !is_f32_tensor_type(node->get_output_element_type(output_idx))) {
            return std::nullopt;
        }
    }

    const auto axis_i64 = constant_i64_vector_input(node, axis_input_idx);
    if (!axis_i64 || axis_i64->size() != 1) {
        return std::nullopt;
    }
    const auto& input_shape = node->get_input_shape(data_input_idx);
    const size_t rank = input_shape.size();
    if (rank == 0) {
        return std::nullopt;
    }
    const auto axis = normalize_axis(axis_i64->front(), rank);
    if (!axis) {
        return std::nullopt;
    }
    if (input_shape[*axis] == 0 ||
        input_shape[*axis] > std::numeric_limits<uint32_t>::max()) {
        return std::nullopt;
    }
    if (require_equal_axis_lengths && input_shape[*axis] % output_count != 0) {
        return std::nullopt;
    }

    uint64_t inner = 1;
    for (size_t dim = *axis + 1; dim < rank; ++dim) {
        inner *= input_shape[dim];
    }
    if (inner == 0 || inner > std::numeric_limits<uint32_t>::max()) {
        return std::nullopt;
    }

    std::vector<uint32_t> axis_lengths;
    axis_lengths.reserve(output_count);
    size_t axis_total = 0;
    const size_t equal_axis_length =
        require_equal_axis_lengths ? input_shape[*axis] / output_count : 0;
    for (size_t output_idx = 0; output_idx < output_count; ++output_idx) {
        const auto& output_shape = node->get_output_shape(output_idx);
        if (output_shape.size() != rank) {
            return std::nullopt;
        }
        for (size_t dim = 0; dim < rank; ++dim) {
            if (dim != *axis && output_shape[dim] != input_shape[dim]) {
                return std::nullopt;
            }
        }
        if (output_shape[*axis] == 0 ||
            output_shape[*axis] > std::numeric_limits<uint32_t>::max()) {
            return std::nullopt;
        }
        if (require_equal_axis_lengths &&
            output_shape[*axis] != equal_axis_length) {
            return std::nullopt;
        }
        if (explicit_axis_lengths) {
            const int64_t expected = (*explicit_axis_lengths)[output_idx];
            if (expected != -1 &&
                output_shape[*axis] != static_cast<size_t>(expected)) {
                return std::nullopt;
            }
        }
        axis_total += output_shape[*axis];
        axis_lengths.push_back(static_cast<uint32_t>(output_shape[*axis]));
    }
    if (axis_total != input_shape[*axis]) {
        return std::nullopt;
    }

    SplitStaticU32Scalars metadata;
    metadata.output_count = static_cast<uint32_t>(output_count);
    metadata.values.reserve(2 + output_count * 2);
    metadata.values.push_back(static_cast<uint32_t>(input_shape[*axis]));
    metadata.values.push_back(static_cast<uint32_t>(inner));
    uint32_t offset = 0;
    for (uint32_t len : axis_lengths) {
        metadata.values.push_back(offset);
        metadata.values.push_back(len);
        offset += len;
    }
    return metadata;
}

std::optional<SplitStaticU32Scalars> split_static_u32_scalars(
    const std::shared_ptr<const ov::Node>& node) {
    auto split = ov::as_type_ptr<const ov::op::v1::Split>(node);
    if (!split || split->get_input_size() != 2) {
        return std::nullopt;
    }
    const size_t output_count = split->get_output_size();
    if (split->get_num_splits() != output_count) {
        return std::nullopt;
    }
    return split_static_u32_scalars_from_outputs(
        node,
        /*data_input_idx=*/0,
        /*axis_input_idx=*/1,
        output_count,
        /*require_equal_axis_lengths=*/true,
        /*explicit_axis_lengths=*/nullptr);
}

std::optional<SplitStaticU32Scalars> variadic_split_static_u32_scalars(
    const std::shared_ptr<const ov::Node>& node) {
    auto split = ov::as_type_ptr<const ov::op::v1::VariadicSplit>(node);
    if (!split || split->get_input_size() != 3) {
        return std::nullopt;
    }
    const size_t output_count = split->get_output_size();
    const auto explicit_axis_lengths = constant_i64_vector_input(node, 2);
    if (!explicit_axis_lengths || explicit_axis_lengths->size() != output_count) {
        return std::nullopt;
    }
    bool has_inferred_axis_length = false;
    for (const int64_t axis_length : *explicit_axis_lengths) {
        if (axis_length == -1) {
            if (has_inferred_axis_length) {
                return std::nullopt;
            }
            has_inferred_axis_length = true;
            continue;
        }
        if (axis_length <= 0 ||
            axis_length > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
            return std::nullopt;
        }
    }
    return split_static_u32_scalars_from_outputs(
        node,
        /*data_input_idx=*/0,
        /*axis_input_idx=*/1,
        output_count,
        /*require_equal_axis_lengths=*/false,
        &*explicit_axis_lengths);
}

GfxKernelStageManifest make_opencl_baseline_manifest(
    GfxKernelStageFamily family,
    std::string specialization_key,
    std::string entry_point,
    uint32_t direct_inputs,
    uint32_t scalar_arg_count,
    uint32_t direct_outputs = 1) {
    GfxKernelExternalBufferAbiSpec abi{};
    abi.valid = true;
    abi.roles.insert(abi.roles.end(), direct_inputs, GfxKernelBufferRole::TensorInput);
    abi.roles.insert(abi.roles.end(), direct_outputs, GfxKernelBufferRole::TensorOutput);
    abi.roles.insert(abi.roles.end(), scalar_arg_count, GfxKernelBufferRole::ScalarParam);
    auto dispatch = make_gfx_kernel_linear_dispatch_policy(
        /*threads_per_threadgroup=*/64,
        /*precompiled_binary_required=*/false);
    auto custom = make_gfx_custom_kernel_manifest(
        gfx_kernel_family_name(GfxKernelFamily::EltwiseFusedBuffer),
        gfx_kernel_family_abi_id(GfxKernelFamily::EltwiseFusedBuffer),
        std::move(entry_point),
        std::move(abi),
        dispatch);
    return make_gfx_custom_kernel_stage_manifest(family,
                                                 GfxKernelBackendDomain::OpenCl,
                                                 GfxKernelStorageKind::Buffer,
                                                 std::move(specialization_key),
                                                 std::move(custom));
}

GfxOpenClSourceArtifact make_opencl_baseline_artifact(
    GfxKernelStageManifest manifest,
    std::string source_id,
    std::vector<GfxOpenClSourceScalarArg> scalar_args,
    std::vector<size_t> direct_input_indices,
    GfxOpenClBaselineOp op,
    GfxOpenClBaselineInputMode input_mode = GfxOpenClBaselineInputMode::Direct,
    float scalar_constant_f32 = 0.0f,
    std::vector<uint32_t> static_u32_scalars = {},
    GfxOpenClSourceElementCountSource element_count_source =
        GfxOpenClSourceElementCountSource::Output0) {
    GfxOpenClSourceArtifact artifact{};
    artifact.valid = manifest.valid;
    artifact.stage_manifest = std::move(manifest);
    artifact.artifact_ref = make_gfx_kernel_artifact_ref(artifact.stage_manifest);
    artifact.artifact_ref.source_id = std::move(source_id);
    artifact.artifact_ref.entry_point = artifact.stage_manifest.custom_kernel.entry_point;
    if (artifact.artifact_ref.entry_point == "gfx_opencl_baseline_matmul_f32") {
        artifact.source = kOpenClMatMulSource;
    } else if (artifact.artifact_ref.entry_point == "gfx_opencl_baseline_softmax_f32") {
        artifact.source = kOpenClSoftmaxSource;
    } else if (artifact.artifact_ref.entry_point ==
                   "gfx_opencl_baseline_shapeof_i32" ||
               artifact.artifact_ref.entry_point ==
                   "gfx_opencl_baseline_shapeof_i64") {
        artifact.source = kOpenClShapeOfSource;
    } else if (artifact.artifact_ref.entry_point ==
               "gfx_opencl_baseline_unary_f32") {
        artifact.source = kOpenClUnaryF32Source;
    } else if (artifact.artifact_ref.entry_point ==
               "gfx_opencl_baseline_binary_f32") {
        artifact.source = kOpenClBinaryF32Source;
    } else if (artifact.artifact_ref.entry_point ==
               "gfx_opencl_baseline_binary_broadcast_f32") {
        artifact.source = kOpenClBinaryBroadcastF32Source;
    } else if (artifact.artifact_ref.entry_point ==
               "gfx_opencl_baseline_binary_scalar_f32") {
        artifact.source = kOpenClBinaryScalarF32Source;
    } else if (artifact.artifact_ref.entry_point ==
               "gfx_opencl_baseline_binary_const_f32") {
        artifact.source = kOpenClBinaryConstF32Source;
    } else if (artifact.artifact_ref.entry_point ==
               "gfx_opencl_baseline_compare_f32") {
        artifact.source = kOpenClCompareF32Source;
    } else if (artifact.artifact_ref.entry_point ==
               "gfx_opencl_baseline_compare_broadcast_f32") {
        artifact.source = kOpenClCompareBroadcastF32Source;
    } else if (artifact.artifact_ref.entry_point ==
               "gfx_opencl_baseline_select_f32") {
        artifact.source = kOpenClSelectF32Source;
    } else if (artifact.artifact_ref.entry_point ==
               "gfx_opencl_baseline_select_broadcast_f32") {
        artifact.source = kOpenClSelectBroadcastF32Source;
    } else if (artifact.artifact_ref.entry_point == "gfx_opencl_baseline_select_f16" ||
               artifact.artifact_ref.entry_point == "gfx_opencl_baseline_concat2_f16" ||
               artifact.artifact_ref.entry_point == "gfx_opencl_baseline_concat3_f16" ||
               artifact.artifact_ref.entry_point == "gfx_opencl_baseline_concat4_f16" ||
               artifact.artifact_ref.entry_point == "gfx_opencl_baseline_broadcast_f16_i64shape" ||
               artifact.artifact_ref.entry_point == "gfx_opencl_baseline_slice_f16" ||
               artifact.artifact_ref.entry_point == "gfx_opencl_baseline_slice_v8_f16" ||
               artifact.artifact_ref.entry_point == "gfx_opencl_baseline_range_i64_unit") {
        artifact.source = kOpenClDynamicDataMovementF16Source;
    } else if (artifact.artifact_ref.entry_point ==
               "gfx_opencl_baseline_logical_unary_bool") {
        artifact.source = kOpenClLogicalUnaryBoolSource;
    } else if (artifact.artifact_ref.entry_point ==
               "gfx_opencl_baseline_logical_binary_bool") {
        artifact.source = kOpenClLogicalBinaryBoolSource;
    } else if (artifact.artifact_ref.entry_point ==
               "gfx_opencl_baseline_logical_binary_broadcast_bool") {
        artifact.source = kOpenClLogicalBinaryBroadcastBoolSource;
    } else if (artifact.artifact_ref.entry_point ==
               "gfx_opencl_baseline_reduce_logical_bool") {
        artifact.source = kOpenClReduceLogicalBoolSource;
    } else {
        artifact.source = kOpenClBaselineSource;
    }
    artifact.scalar_args = std::move(scalar_args);
    artifact.static_u32_scalars = std::move(static_u32_scalars);
    artifact.direct_input_indices = std::move(direct_input_indices);
    const auto roles = materialize_gfx_kernel_external_buffer_roles(
        artifact.stage_manifest.custom_kernel.external_buffer_abi);
    artifact.arg_count = static_cast<uint32_t>(roles.size());
    artifact.direct_input_count = static_cast<uint32_t>(artifact.direct_input_indices.size());
    artifact.direct_output_count = 0;
    for (const auto role : roles) {
        if (role == GfxKernelBufferRole::TensorOutput) {
            ++artifact.direct_output_count;
        }
    }
    artifact.element_count_source = element_count_source;
    artifact.op = op;
    artifact.input_mode = input_mode;
    artifact.scalar_constant_f32 = scalar_constant_f32;
    artifact.valid = artifact.valid &&
                     artifact.artifact_ref.valid &&
                     artifact.artifact_ref.kind == GfxKernelArtifactKind::OpenClSource;
    return artifact;
}

}  // namespace

std::optional<GfxOpenClSourceArtifact> resolve_gfx_opencl_source_artifact(
    const std::shared_ptr<const ov::Node>& node) {
    if (!node || node->get_output_size() == 0) {
        return std::nullopt;
    }

    const std::string type = node->get_type_name();
    if (type == "Concat") {
        auto dynamic_scalars = concat_dynamic_f16_scalars(node);
        if (dynamic_scalars) {
            std::vector<GfxOpenClSourceScalarArg> scalar_args = {
                GfxOpenClSourceScalarArg::ElementCount,
                GfxOpenClSourceScalarArg::StaticU32};
            for (size_t input_idx = 0; input_idx < dynamic_scalars->input_count; ++input_idx) {
                scalar_args.push_back(input_dim_scalar_arg(input_idx, dynamic_scalars->axis));
            }
            std::vector<size_t> direct_input_indices;
            direct_input_indices.reserve(dynamic_scalars->input_count);
            for (size_t input_idx = 0; input_idx < dynamic_scalars->input_count; ++input_idx) {
                direct_input_indices.push_back(input_idx);
            }
            const std::string entry_point =
                "gfx_opencl_baseline_concat" +
                std::to_string(dynamic_scalars->input_count) + "_f16";
            auto manifest = make_opencl_baseline_manifest(
                GfxKernelStageFamily::ConcatSplit,
                "opencl:baseline:Concat:f16:dynamic_static_rank:inputs" +
                    std::to_string(dynamic_scalars->input_count),
                entry_point,
                dynamic_scalars->input_count,
                static_cast<uint32_t>(scalar_args.size()));
            return make_opencl_baseline_artifact(
                std::move(manifest),
                "opencl/baseline/concat" +
                    std::to_string(dynamic_scalars->input_count) + "_f16_dynamic",
                std::move(scalar_args),
                std::move(direct_input_indices),
                GfxOpenClBaselineOp::Identity,
                GfxOpenClBaselineInputMode::Direct,
                0.0f,
                {dynamic_scalars->inner});
        }
    }

    if (type == "Broadcast") {
        auto dynamic_scalars = broadcast_dynamic_f16_scalars(node);
        if (dynamic_scalars) {
            std::vector<GfxOpenClSourceScalarArg> scalar_args = {
                GfxOpenClSourceScalarArg::ElementCount,
                GfxOpenClSourceScalarArg::StaticU32,
                GfxOpenClSourceScalarArg::StaticU32};
            for (uint32_t axis = 0; axis < 4; ++axis) {
                scalar_args.push_back(input_dim_scalar_arg(0, axis));
            }
            auto manifest = make_opencl_baseline_manifest(
                GfxKernelStageFamily::Layout,
                "opencl:baseline:Broadcast:f16:i64shape:dynamic_static_rank",
                "gfx_opencl_baseline_broadcast_f16_i64shape",
                /*direct_inputs=*/2,
                static_cast<uint32_t>(scalar_args.size()));
            return make_opencl_baseline_artifact(
                std::move(manifest),
                "opencl/baseline/broadcast_f16_i64shape_dynamic",
                std::move(scalar_args),
                {0, 1},
                GfxOpenClBaselineOp::Identity,
                GfxOpenClBaselineInputMode::Direct,
                0.0f,
                {dynamic_scalars->rank, dynamic_scalars->input_rank});
        }
    }

    if (type == "Select" && select_dynamic_f16_supported(node)) {
        auto manifest = make_opencl_baseline_manifest(
            GfxKernelStageFamily::Eltwise,
            "opencl:baseline:Select:bool_f16:dynamic_same_shape",
            "gfx_opencl_baseline_select_f16",
            /*direct_inputs=*/3,
            /*scalar_arg_count=*/1);
        return make_opencl_baseline_artifact(std::move(manifest),
                                             "opencl/baseline/select_f16_dynamic",
                                             {GfxOpenClSourceScalarArg::ElementCount},
                                             {0, 1, 2},
                                             GfxOpenClBaselineOp::Identity);
    }

    if (type == "Slice") {
        auto dynamic_scalars = slice_v8_dynamic_f16_scalars(node);
        if (dynamic_scalars) {
            std::vector<GfxOpenClSourceScalarArg> scalar_args = {
                GfxOpenClSourceScalarArg::ElementCount,
                GfxOpenClSourceScalarArg::StaticU32};
            for (uint32_t axis = 0; axis < 4; ++axis) {
                scalar_args.push_back(output0_dim_scalar_arg(axis));
            }
            for (uint32_t axis = 0; axis < 4; ++axis) {
                scalar_args.push_back(input_dim_scalar_arg(0, axis));
            }
            auto manifest = make_opencl_baseline_manifest(
                GfxKernelStageFamily::GatherScatter,
                "opencl:baseline:Slice:f16:dynamic_static_rank",
                "gfx_opencl_baseline_slice_v8_f16",
                /*direct_inputs=*/4,
                static_cast<uint32_t>(scalar_args.size()));
            return make_opencl_baseline_artifact(std::move(manifest),
                                                 "opencl/baseline/slice_v8_f16_dynamic",
                                                 std::move(scalar_args),
                                                 {0, 1, 2, 3},
                                                 GfxOpenClBaselineOp::Identity,
                                                 GfxOpenClBaselineInputMode::Direct,
                                                 0.0f,
                                                 {dynamic_scalars->rank});
        }
    }

    if (type == "StridedSlice") {
        auto dynamic_scalars = strided_slice_dynamic_f16_scalars(node);
        if (dynamic_scalars) {
            std::vector<GfxOpenClSourceScalarArg> scalar_args = {
                GfxOpenClSourceScalarArg::ElementCount,
                GfxOpenClSourceScalarArg::StaticU32};
            for (uint32_t axis = 0; axis < 4; ++axis) {
                scalar_args.push_back(output0_dim_scalar_arg(axis));
            }
            for (uint32_t axis = 0; axis < 4; ++axis) {
                scalar_args.push_back(input_dim_scalar_arg(0, axis));
            }
            scalar_args.insert(scalar_args.end(),
                               dynamic_scalars->begin.size() + dynamic_scalars->steps.size(),
                               GfxOpenClSourceScalarArg::StaticU32);
            std::vector<uint32_t> static_scalars = {dynamic_scalars->rank};
            static_scalars.insert(static_scalars.end(),
                                  dynamic_scalars->begin.begin(),
                                  dynamic_scalars->begin.end());
            static_scalars.insert(static_scalars.end(),
                                  dynamic_scalars->steps.begin(),
                                  dynamic_scalars->steps.end());
            auto manifest = make_opencl_baseline_manifest(
                GfxKernelStageFamily::GatherScatter,
                "opencl:baseline:StridedSlice:f16:dynamic_static_rank",
                "gfx_opencl_baseline_slice_f16",
                /*direct_inputs=*/2,
                static_cast<uint32_t>(scalar_args.size()));
            return make_opencl_baseline_artifact(std::move(manifest),
                                                 "opencl/baseline/slice_f16_dynamic",
                                                 std::move(scalar_args),
                                                 {0, 2},
                                                 GfxOpenClBaselineOp::Identity,
                                                 GfxOpenClBaselineInputMode::Direct,
                                                 0.0f,
                                                 std::move(static_scalars));
        }
    }

    if (type == "Range" && range_dynamic_i64_unit_supported(node)) {
        auto manifest = make_opencl_baseline_manifest(
            GfxKernelStageFamily::GatherScatter,
            "opencl:baseline:Range:i64:dynamic_unit",
            "gfx_opencl_baseline_range_i64_unit",
            /*direct_inputs=*/1,
            /*scalar_arg_count=*/1);
        return make_opencl_baseline_artifact(std::move(manifest),
                                             "opencl/baseline/range_i64_unit_dynamic",
                                             {GfxOpenClSourceScalarArg::ElementCount},
                                             {1},
                                             GfxOpenClBaselineOp::Identity);
    }

    if (!node->get_output_partial_shape(0).is_static()) {
        return std::nullopt;
    }

    if (is_linear_copy_op(type)) {
        if (node->get_input_size() < 1 ||
            !is_f32_tensor_type(node->get_output_element_type(0)) ||
            !is_f32_tensor_type(node->get_input_element_type(0)) ||
            !same_static_element_count_input_output(node, 0, 0)) {
            return std::nullopt;
        }
        auto manifest = make_opencl_baseline_manifest(
            type == "Convert" ? GfxKernelStageFamily::Convert : GfxKernelStageFamily::Layout,
            "opencl:baseline:" + type + ":f32:linear_copy",
            "gfx_opencl_baseline_unary_f32",
            /*direct_inputs=*/1,
            /*scalar_arg_count=*/2);
        return make_opencl_baseline_artifact(std::move(manifest),
                                             "opencl/baseline/linear_copy_f32",
                                             {GfxOpenClSourceScalarArg::ElementCount,
                                              GfxOpenClSourceScalarArg::OpCode},
                                             {0},
                                             GfxOpenClBaselineOp::Identity);
    }

    if (type == "Convert") {
        if (node->get_input_size() != 1 ||
            !same_static_element_count_input_output(node, 0, 0)) {
            return std::nullopt;
        }
        const auto input_type = node->get_input_element_type(0);
        const auto output_type = node->get_output_element_type(0);
        if (!is_opencl_convert_tensor_type(input_type) ||
            !is_opencl_convert_tensor_type(output_type)) {
            return std::nullopt;
        }
        const std::string input_suffix = opencl_scalar_type_suffix(input_type);
        const std::string output_suffix = opencl_scalar_type_suffix(output_type);
        const std::string suffix = input_suffix + "_to_" + output_suffix;
        const std::string entry_point =
            "gfx_opencl_baseline_convert_" + suffix;
        auto manifest = make_opencl_baseline_manifest(
            GfxKernelStageFamily::Convert,
            "opencl:baseline:Convert:" + suffix,
            entry_point,
            /*direct_inputs=*/1,
            /*scalar_arg_count=*/1);
        return make_opencl_baseline_artifact(std::move(manifest),
                                             "opencl/baseline/convert_" + suffix,
                                             {GfxOpenClSourceScalarArg::ElementCount},
                                             {0},
                                             GfxOpenClBaselineOp::Identity);
    }

    if (type == "MatMul") {
        auto static_u32_scalars = matmul_static_u32_scalars(node);
        if (!static_u32_scalars) {
            return std::nullopt;
        }
        std::vector<GfxOpenClSourceScalarArg> scalar_args = {
            GfxOpenClSourceScalarArg::ElementCount};
        scalar_args.insert(scalar_args.end(),
                           static_u32_scalars->size(),
                           GfxOpenClSourceScalarArg::StaticU32);
        auto manifest = make_opencl_baseline_manifest(
            GfxKernelStageFamily::Gemm,
            "opencl:baseline:MatMul:f32",
            "gfx_opencl_baseline_matmul_f32",
            /*direct_inputs=*/2,
            static_cast<uint32_t>(scalar_args.size()));
        return make_opencl_baseline_artifact(std::move(manifest),
                                             "opencl/baseline/matmul_f32",
                                             std::move(scalar_args),
                                             {0, 1},
                                             GfxOpenClBaselineOp::Identity,
                                             GfxOpenClBaselineInputMode::Direct,
                                             0.0f,
                                             std::move(*static_u32_scalars));
    }

    if (type == "Softmax") {
        auto static_u32_scalars = softmax_static_u32_scalars(node);
        if (!static_u32_scalars) {
            return std::nullopt;
        }
        std::vector<GfxOpenClSourceScalarArg> scalar_args = {
            GfxOpenClSourceScalarArg::ElementCount};
        scalar_args.insert(scalar_args.end(),
                           static_u32_scalars->size(),
                           GfxOpenClSourceScalarArg::StaticU32);
        auto manifest = make_opencl_baseline_manifest(
            GfxKernelStageFamily::Softmax,
            "opencl:baseline:Softmax:f32",
            "gfx_opencl_baseline_softmax_f32",
            /*direct_inputs=*/1,
            static_cast<uint32_t>(scalar_args.size()));
        return make_opencl_baseline_artifact(std::move(manifest),
                                             "opencl/baseline/softmax_f32",
                                             std::move(scalar_args),
                                             {0},
                                             GfxOpenClBaselineOp::Identity,
                                             GfxOpenClBaselineInputMode::Direct,
                                             0.0f,
                                             std::move(*static_u32_scalars));
    }

    if (type == "Transpose") {
        if (node->get_input_size() != 2 ||
            !is_f32_tensor_type(node->get_output_element_type(0)) ||
            !is_f32_tensor_type(node->get_input_element_type(0))) {
            return std::nullopt;
        }
        auto static_u32_scalars = transpose_static_u32_scalars(node);
        if (!static_u32_scalars) {
            return std::nullopt;
        }
        std::vector<GfxOpenClSourceScalarArg> scalar_args;
        scalar_args.reserve(1 + static_u32_scalars->size());
        scalar_args.push_back(GfxOpenClSourceScalarArg::ElementCount);
        scalar_args.insert(scalar_args.end(),
                           static_u32_scalars->size(),
                           GfxOpenClSourceScalarArg::StaticU32);
        auto manifest = make_opencl_baseline_manifest(
            GfxKernelStageFamily::Transpose,
            "opencl:baseline:Transpose:f32:rank" +
                std::to_string(static_u32_scalars->front()),
            "gfx_opencl_baseline_transpose_f32",
            /*direct_inputs=*/1,
            static_cast<uint32_t>(scalar_args.size()));
        return make_opencl_baseline_artifact(std::move(manifest),
                                             "opencl/baseline/transpose_f32",
                                             std::move(scalar_args),
                                             {0},
                                             GfxOpenClBaselineOp::Identity,
                                             GfxOpenClBaselineInputMode::Direct,
                                             0.0f,
                                             std::move(*static_u32_scalars));
    }

    if (type == "Slice" || type == "StridedSlice") {
        auto static_u32_scalars = type == "Slice"
                                      ? slice_static_u32_scalars(node)
                                      : strided_slice_static_u32_scalars(node);
        if (!static_u32_scalars) {
            return std::nullopt;
        }
        std::vector<GfxOpenClSourceScalarArg> scalar_args = {
            GfxOpenClSourceScalarArg::ElementCount};
        scalar_args.insert(scalar_args.end(),
                           static_u32_scalars->size(),
                           GfxOpenClSourceScalarArg::StaticU32);
        auto manifest = make_opencl_baseline_manifest(
            GfxKernelStageFamily::GatherScatter,
            "opencl:baseline:" + type + ":f32:rank" +
                std::to_string(static_u32_scalars->front()),
            "gfx_opencl_baseline_slice_f32",
            /*direct_inputs=*/1,
            static_cast<uint32_t>(scalar_args.size()));
        return make_opencl_baseline_artifact(std::move(manifest),
                                             "opencl/baseline/slice_f32",
                                             std::move(scalar_args),
                                             {0},
                                             GfxOpenClBaselineOp::Identity,
                                             GfxOpenClBaselineInputMode::Direct,
                                             0.0f,
                                             std::move(*static_u32_scalars));
    }

    if (type == "Range") {
        if (!range_has_baseline_source_artifact(node)) {
            return std::nullopt;
        }
        const auto output_type = node->get_output_element_type(0);
        const std::string type_suffix = opencl_range_type_suffix(output_type);
        const std::string entry_point = "gfx_opencl_baseline_range_" + type_suffix;
        auto manifest = make_opencl_baseline_manifest(
            GfxKernelStageFamily::GatherScatter,
            "opencl:baseline:Range:" + type_suffix,
            entry_point,
            /*direct_inputs=*/3,
            /*scalar_arg_count=*/1);
        return make_opencl_baseline_artifact(std::move(manifest),
                                             "opencl/baseline/range_" + type_suffix,
                                             {GfxOpenClSourceScalarArg::ElementCount},
                                             {0, 1, 2},
                                             GfxOpenClBaselineOp::Identity,
                                             GfxOpenClBaselineInputMode::Direct);
    }

    if (type == "Tile") {
        auto static_u32_scalars = tile_static_u32_scalars(node);
        if (!static_u32_scalars) {
            return std::nullopt;
        }
        std::vector<GfxOpenClSourceScalarArg> scalar_args = {
            GfxOpenClSourceScalarArg::ElementCount};
        scalar_args.insert(scalar_args.end(),
                           static_u32_scalars->size(),
                           GfxOpenClSourceScalarArg::StaticU32);
        auto manifest = make_opencl_baseline_manifest(
            GfxKernelStageFamily::Layout,
            "opencl:baseline:Tile:f32",
            "gfx_opencl_baseline_tile_f32",
            /*direct_inputs=*/1,
            static_cast<uint32_t>(scalar_args.size()));
        return make_opencl_baseline_artifact(std::move(manifest),
                                             "opencl/baseline/tile_f32",
                                             std::move(scalar_args),
                                             {0},
                                             GfxOpenClBaselineOp::Identity,
                                             GfxOpenClBaselineInputMode::Direct,
                                             0.0f,
                                             std::move(*static_u32_scalars));
    }

    if (type == "Gather") {
        auto static_u32_scalars = gather_static_u32_scalars(node);
        if (!static_u32_scalars) {
            return std::nullopt;
        }
        const auto indices_type = node->get_input_element_type(1);
        const bool indices_i64 = indices_type == ov::element::i64;
        std::vector<GfxOpenClSourceScalarArg> scalar_args = {
            GfxOpenClSourceScalarArg::ElementCount};
        scalar_args.insert(scalar_args.end(),
                           static_u32_scalars->size(),
                           GfxOpenClSourceScalarArg::StaticU32);
        const std::string entry_point =
            indices_i64 ? "gfx_opencl_baseline_gather_i64_f32"
                        : "gfx_opencl_baseline_gather_i32_f32";
        auto manifest = make_opencl_baseline_manifest(
            GfxKernelStageFamily::GatherScatter,
            "opencl:baseline:Gather:" +
                std::string(indices_i64 ? "i64" : "i32") + ":f32",
            entry_point,
            /*direct_inputs=*/2,
            static_cast<uint32_t>(scalar_args.size()));
        return make_opencl_baseline_artifact(
            std::move(manifest),
            indices_i64 ? "opencl/baseline/gather_i64_f32"
                        : "opencl/baseline/gather_i32_f32",
            std::move(scalar_args),
            {0, 1},
            GfxOpenClBaselineOp::Identity,
            GfxOpenClBaselineInputMode::Direct,
            0.0f,
            std::move(*static_u32_scalars));
    }

    if (type == "GatherElements") {
        auto static_u32_scalars = gather_elements_static_u32_scalars(node);
        if (!static_u32_scalars) {
            return std::nullopt;
        }
        const auto indices_type = node->get_input_element_type(1);
        const bool indices_i64 = indices_type == ov::element::i64;
        std::vector<GfxOpenClSourceScalarArg> scalar_args = {
            GfxOpenClSourceScalarArg::ElementCount};
        scalar_args.insert(scalar_args.end(),
                           static_u32_scalars->size(),
                           GfxOpenClSourceScalarArg::StaticU32);
        const std::string entry_point =
            indices_i64 ? "gfx_opencl_baseline_gather_elements_i64_f32"
                        : "gfx_opencl_baseline_gather_elements_i32_f32";
        auto manifest = make_opencl_baseline_manifest(
            GfxKernelStageFamily::GatherScatter,
            "opencl:baseline:GatherElements:" +
                std::string(indices_i64 ? "i64" : "i32") + ":f32",
            entry_point,
            /*direct_inputs=*/2,
            static_cast<uint32_t>(scalar_args.size()));
        return make_opencl_baseline_artifact(
            std::move(manifest),
            indices_i64 ? "opencl/baseline/gather_elements_i64_f32"
                        : "opencl/baseline/gather_elements_i32_f32",
            std::move(scalar_args),
            {0, 1},
            GfxOpenClBaselineOp::Identity,
            GfxOpenClBaselineInputMode::Direct,
            0.0f,
            std::move(*static_u32_scalars));
    }

    if (type == "GatherND") {
        auto static_u32_scalars = gather_nd_static_u32_scalars(node);
        if (!static_u32_scalars) {
            return std::nullopt;
        }
        const auto indices_type = node->get_input_element_type(1);
        const bool indices_i64 = indices_type == ov::element::i64;
        std::vector<GfxOpenClSourceScalarArg> scalar_args = {
            GfxOpenClSourceScalarArg::ElementCount};
        scalar_args.insert(scalar_args.end(),
                           static_u32_scalars->size(),
                           GfxOpenClSourceScalarArg::StaticU32);
        const std::string entry_point =
            indices_i64 ? "gfx_opencl_baseline_gather_nd_i64_f32"
                        : "gfx_opencl_baseline_gather_nd_i32_f32";
        auto manifest = make_opencl_baseline_manifest(
            GfxKernelStageFamily::GatherScatter,
            "opencl:baseline:GatherND:" +
                std::string(indices_i64 ? "i64" : "i32") + ":f32",
            entry_point,
            /*direct_inputs=*/2,
            static_cast<uint32_t>(scalar_args.size()));
        return make_opencl_baseline_artifact(
            std::move(manifest),
            indices_i64 ? "opencl/baseline/gather_nd_i64_f32"
                        : "opencl/baseline/gather_nd_i32_f32",
            std::move(scalar_args),
            {0, 1},
            GfxOpenClBaselineOp::Identity,
            GfxOpenClBaselineInputMode::Direct,
            0.0f,
            std::move(*static_u32_scalars));
    }

    if (type == "ScatterUpdate") {
        auto static_u32_scalars = scatter_update_static_u32_scalars(node);
        if (!static_u32_scalars) {
            return std::nullopt;
        }
        const auto indices_type = node->get_input_element_type(1);
        const bool indices_i64 = indices_type == ov::element::i64;
        std::vector<GfxOpenClSourceScalarArg> scalar_args = {
            GfxOpenClSourceScalarArg::ElementCount};
        scalar_args.insert(scalar_args.end(),
                           static_u32_scalars->size(),
                           GfxOpenClSourceScalarArg::StaticU32);
        const std::string entry_point =
            indices_i64 ? "gfx_opencl_baseline_scatter_update_i64_f32"
                        : "gfx_opencl_baseline_scatter_update_i32_f32";
        auto manifest = make_opencl_baseline_manifest(
            GfxKernelStageFamily::GatherScatter,
            "opencl:baseline:ScatterUpdate:" +
                std::string(indices_i64 ? "i64" : "i32") + ":f32",
            entry_point,
            /*direct_inputs=*/3,
            static_cast<uint32_t>(scalar_args.size()));
        return make_opencl_baseline_artifact(
            std::move(manifest),
            indices_i64 ? "opencl/baseline/scatter_update_i64_f32"
                        : "opencl/baseline/scatter_update_i32_f32",
            std::move(scalar_args),
            {0, 1, 2},
            GfxOpenClBaselineOp::Identity,
            GfxOpenClBaselineInputMode::Direct,
            0.0f,
            std::move(*static_u32_scalars));
    }

    if (type == "ScatterElementsUpdate") {
        auto static_u32_scalars = scatter_elements_static_u32_scalars(node);
        if (!static_u32_scalars) {
            return std::nullopt;
        }
        const auto indices_type = node->get_input_element_type(1);
        const bool indices_i64 = indices_type == ov::element::i64;
        std::vector<GfxOpenClSourceScalarArg> scalar_args = {
            GfxOpenClSourceScalarArg::ElementCount};
        scalar_args.insert(scalar_args.end(),
                           static_u32_scalars->size(),
                           GfxOpenClSourceScalarArg::StaticU32);
        const std::string entry_point =
            indices_i64 ? "gfx_opencl_baseline_scatter_elements_i64_f32"
                        : "gfx_opencl_baseline_scatter_elements_i32_f32";
        auto manifest = make_opencl_baseline_manifest(
            GfxKernelStageFamily::GatherScatter,
            "opencl:baseline:ScatterElementsUpdate:" +
                std::string(indices_i64 ? "i64" : "i32") + ":f32",
            entry_point,
            /*direct_inputs=*/3,
            static_cast<uint32_t>(scalar_args.size()));
        return make_opencl_baseline_artifact(
            std::move(manifest),
            indices_i64 ? "opencl/baseline/scatter_elements_i64_f32"
                        : "opencl/baseline/scatter_elements_i32_f32",
            std::move(scalar_args),
            {0, 1, 2},
            GfxOpenClBaselineOp::Identity,
            GfxOpenClBaselineInputMode::Direct,
            0.0f,
            std::move(*static_u32_scalars));
    }

    if (type == "ScatterNDUpdate") {
        auto static_u32_scalars = scatter_nd_static_u32_scalars(node);
        if (!static_u32_scalars) {
            return std::nullopt;
        }
        const auto indices_type = node->get_input_element_type(1);
        const bool indices_i64 = indices_type == ov::element::i64;
        std::vector<GfxOpenClSourceScalarArg> scalar_args = {
            GfxOpenClSourceScalarArg::ElementCount};
        scalar_args.insert(scalar_args.end(),
                           static_u32_scalars->size(),
                           GfxOpenClSourceScalarArg::StaticU32);
        const std::string entry_point =
            indices_i64 ? "gfx_opencl_baseline_scatter_nd_i64_f32"
                        : "gfx_opencl_baseline_scatter_nd_i32_f32";
        auto manifest = make_opencl_baseline_manifest(
            GfxKernelStageFamily::GatherScatter,
            "opencl:baseline:ScatterNDUpdate:" +
                std::string(indices_i64 ? "i64" : "i32") + ":f32",
            entry_point,
            /*direct_inputs=*/3,
            static_cast<uint32_t>(scalar_args.size()));
        return make_opencl_baseline_artifact(
            std::move(manifest),
            indices_i64 ? "opencl/baseline/scatter_nd_i64_f32"
                        : "opencl/baseline/scatter_nd_i32_f32",
            std::move(scalar_args),
            {0, 1, 2},
            GfxOpenClBaselineOp::Identity,
            GfxOpenClBaselineInputMode::Direct,
            0.0f,
            std::move(*static_u32_scalars));
    }

    if (type == "ShapeOf") {
        auto rank = shapeof_rank(node);
        if (!rank) {
            return std::nullopt;
        }
        const auto output_type = node->get_output_element_type(0);
        const bool output_i64 = is_i64_tensor_type(output_type);
        std::vector<GfxOpenClSourceScalarArg> scalar_args = {
            GfxOpenClSourceScalarArg::ElementCount};
        scalar_args.reserve(9);
        for (uint32_t axis = 0; axis < 8; ++axis) {
            scalar_args.push_back(static_cast<GfxOpenClSourceScalarArg>(
                static_cast<uint32_t>(GfxOpenClSourceScalarArg::Input0Dim0) + axis));
        }
        const std::string entry_point =
            output_i64 ? "gfx_opencl_baseline_shapeof_i64"
                       : "gfx_opencl_baseline_shapeof_i32";
        auto manifest = make_opencl_baseline_manifest(
            GfxKernelStageFamily::GatherScatter,
            "opencl:baseline:ShapeOf:" +
                std::string(output_i64 ? "i64" : "i32") + ":rank" +
                std::to_string(*rank),
            entry_point,
            /*direct_inputs=*/1,
            static_cast<uint32_t>(scalar_args.size()));
        return make_opencl_baseline_artifact(
            std::move(manifest),
            output_i64 ? "opencl/baseline/shapeof_i64"
                       : "opencl/baseline/shapeof_i32",
            std::move(scalar_args),
            {0},
            GfxOpenClBaselineOp::Identity,
            GfxOpenClBaselineInputMode::Direct);
    }

    if (type == "Concat") {
        auto static_u32_scalars = concat_static_u32_scalars(node);
        if (!static_u32_scalars) {
            return std::nullopt;
        }
        std::vector<GfxOpenClSourceScalarArg> scalar_args = {
            GfxOpenClSourceScalarArg::ElementCount};
        scalar_args.insert(scalar_args.end(),
                           static_u32_scalars->values.size(),
                           GfxOpenClSourceScalarArg::StaticU32);
        const std::string entry_point =
            "gfx_opencl_baseline_concat" +
            std::to_string(static_u32_scalars->input_count) + "_f32";
        std::vector<size_t> direct_input_indices;
        direct_input_indices.reserve(static_u32_scalars->input_count);
        for (size_t input_idx = 0;
             input_idx < static_u32_scalars->input_count;
             ++input_idx) {
            direct_input_indices.push_back(input_idx);
        }
        auto manifest = make_opencl_baseline_manifest(
            GfxKernelStageFamily::ConcatSplit,
            "opencl:baseline:Concat:f32:inputs" +
                std::to_string(static_u32_scalars->input_count),
            entry_point,
            static_u32_scalars->input_count,
            static_cast<uint32_t>(scalar_args.size()));
        return make_opencl_baseline_artifact(
            std::move(manifest),
            "opencl/baseline/concat" +
                std::to_string(static_u32_scalars->input_count) + "_f32",
            std::move(scalar_args),
            std::move(direct_input_indices),
            GfxOpenClBaselineOp::Identity,
            GfxOpenClBaselineInputMode::Direct,
            0.0f,
            std::move(static_u32_scalars->values));
    }

    if (type == "Split" || type == "VariadicSplit") {
        auto static_u32_scalars = type == "Split"
                                      ? split_static_u32_scalars(node)
                                      : variadic_split_static_u32_scalars(node);
        if (!static_u32_scalars) {
            return std::nullopt;
        }
        std::vector<GfxOpenClSourceScalarArg> scalar_args = {
            GfxOpenClSourceScalarArg::ElementCount};
        scalar_args.insert(scalar_args.end(),
                           static_u32_scalars->values.size(),
                           GfxOpenClSourceScalarArg::StaticU32);
        const std::string entry_point =
            "gfx_opencl_baseline_split" +
            std::to_string(static_u32_scalars->output_count) + "_f32";
        auto manifest = make_opencl_baseline_manifest(
            GfxKernelStageFamily::ConcatSplit,
            "opencl:baseline:" + type + ":f32:outputs" +
                std::to_string(static_u32_scalars->output_count),
            entry_point,
            /*direct_inputs=*/1,
            static_cast<uint32_t>(scalar_args.size()),
            static_u32_scalars->output_count);
        return make_opencl_baseline_artifact(
            std::move(manifest),
            "opencl/baseline/split" +
                std::to_string(static_u32_scalars->output_count) + "_f32",
            std::move(scalar_args),
            {0},
            GfxOpenClBaselineOp::Identity,
            GfxOpenClBaselineInputMode::Direct,
            0.0f,
            std::move(static_u32_scalars->values),
            GfxOpenClSourceElementCountSource::Input0);
    }

    if (auto op = unary_op_code(type)) {
        if (node->get_input_size() != 1 ||
            !is_f32_tensor_type(node->get_output_element_type(0)) ||
            !is_f32_tensor_type(node->get_input_element_type(0))) {
            return std::nullopt;
        }
        auto manifest = make_opencl_baseline_manifest(
            GfxKernelStageFamily::Eltwise,
            "opencl:baseline:" + type + ":f32",
            "gfx_opencl_baseline_unary_f32",
            /*direct_inputs=*/1,
            /*scalar_arg_count=*/2);
        return make_opencl_baseline_artifact(std::move(manifest),
                                             "opencl/baseline/eltwise_unary_f32",
                                             {GfxOpenClSourceScalarArg::ElementCount,
                                              GfxOpenClSourceScalarArg::OpCode},
                                             {0},
                                             *op);
    }

    if (auto op = binary_op_code(type)) {
        if (node->get_input_size() != 2 ||
            !is_f32_tensor_type(node->get_output_element_type(0)) ||
            !is_f32_tensor_type(node->get_input_element_type(0)) ||
            !is_f32_tensor_type(node->get_input_element_type(1))) {
            return std::nullopt;
        }
        const bool lhs_matches_output = input_static_element_count_matches_output(node, 0, 0);
        const bool rhs_matches_output = input_static_element_count_matches_output(node, 1, 0);
        const auto lhs_constant = scalar_f32_constant_input(node, 0);
        const auto rhs_constant = scalar_f32_constant_input(node, 1);
        if (same_static_shape(node, 0, 1) && lhs_matches_output && rhs_matches_output) {
            auto manifest = make_opencl_baseline_manifest(
                GfxKernelStageFamily::Eltwise,
                "opencl:baseline:" + type + ":f32:same_shape",
                "gfx_opencl_baseline_binary_f32",
                /*direct_inputs=*/2,
                /*scalar_arg_count=*/2);
            return make_opencl_baseline_artifact(std::move(manifest),
                                                 "opencl/baseline/eltwise_binary_f32",
                                                 {GfxOpenClSourceScalarArg::ElementCount,
                                                  GfxOpenClSourceScalarArg::OpCode},
                                                 {0, 1},
                                                 *op);
        }
        if (rhs_constant && lhs_matches_output) {
            auto manifest = make_opencl_baseline_manifest(
                GfxKernelStageFamily::Eltwise,
                "opencl:baseline:" + type + ":f32:rhs_scalar_const",
                "gfx_opencl_baseline_binary_const_f32",
                /*direct_inputs=*/1,
                /*scalar_arg_count=*/4);
            return make_opencl_baseline_artifact(std::move(manifest),
                                                 "opencl/baseline/eltwise_binary_const_f32",
                                                 {GfxOpenClSourceScalarArg::ElementCount,
                                                  GfxOpenClSourceScalarArg::OpCode,
                                                  GfxOpenClSourceScalarArg::InputMode,
                                                  GfxOpenClSourceScalarArg::ScalarConstantF32},
                                                 {0},
                                                 *op,
                                                 GfxOpenClBaselineInputMode::RhsScalarConstant,
                                                 *rhs_constant);
        }
        if (lhs_constant && rhs_matches_output) {
            auto manifest = make_opencl_baseline_manifest(
                GfxKernelStageFamily::Eltwise,
                "opencl:baseline:" + type + ":f32:lhs_scalar_const",
                "gfx_opencl_baseline_binary_const_f32",
                /*direct_inputs=*/1,
                /*scalar_arg_count=*/4);
            return make_opencl_baseline_artifact(std::move(manifest),
                                                 "opencl/baseline/eltwise_binary_const_f32",
                                                 {GfxOpenClSourceScalarArg::ElementCount,
                                                  GfxOpenClSourceScalarArg::OpCode,
                                                  GfxOpenClSourceScalarArg::InputMode,
                                                  GfxOpenClSourceScalarArg::ScalarConstantF32},
                                                 {1},
                                                 *op,
                                                 GfxOpenClBaselineInputMode::LhsScalarConstant,
                                                 *lhs_constant);
        }
        if (is_static_scalar_like_input(node, 1) && lhs_matches_output) {
            auto manifest = make_opencl_baseline_manifest(
                GfxKernelStageFamily::Eltwise,
                "opencl:baseline:" + type + ":f32:rhs_scalar",
                "gfx_opencl_baseline_binary_scalar_f32",
                /*direct_inputs=*/2,
                /*scalar_arg_count=*/3);
            return make_opencl_baseline_artifact(std::move(manifest),
                                                 "opencl/baseline/eltwise_binary_scalar_f32",
                                                 {GfxOpenClSourceScalarArg::ElementCount,
                                                  GfxOpenClSourceScalarArg::OpCode,
                                                  GfxOpenClSourceScalarArg::InputMode},
                                                 {0, 1},
                                                 *op,
                                                 GfxOpenClBaselineInputMode::RhsScalar);
        }
        if (is_static_scalar_like_input(node, 0) && rhs_matches_output) {
            auto manifest = make_opencl_baseline_manifest(
                GfxKernelStageFamily::Eltwise,
                "opencl:baseline:" + type + ":f32:lhs_scalar",
                "gfx_opencl_baseline_binary_scalar_f32",
                /*direct_inputs=*/2,
                /*scalar_arg_count=*/3);
            return make_opencl_baseline_artifact(std::move(manifest),
                                                 "opencl/baseline/eltwise_binary_scalar_f32",
                                                 {GfxOpenClSourceScalarArg::ElementCount,
                                                  GfxOpenClSourceScalarArg::OpCode,
                                                  GfxOpenClSourceScalarArg::InputMode},
                                                 {0, 1},
                                                 *op,
                                                 GfxOpenClBaselineInputMode::LhsScalar);
        }
        auto static_u32_scalars = binary_broadcast_static_u32_scalars(node);
        if (static_u32_scalars) {
            std::vector<GfxOpenClSourceScalarArg> scalar_args = {
                GfxOpenClSourceScalarArg::ElementCount,
                GfxOpenClSourceScalarArg::OpCode};
            scalar_args.insert(scalar_args.end(),
                               static_u32_scalars->size(),
                               GfxOpenClSourceScalarArg::StaticU32);
            auto manifest = make_opencl_baseline_manifest(
                GfxKernelStageFamily::Eltwise,
                "opencl:baseline:" + type + ":f32:broadcast",
                "gfx_opencl_baseline_binary_broadcast_f32",
                /*direct_inputs=*/2,
                static_cast<uint32_t>(scalar_args.size()));
            return make_opencl_baseline_artifact(std::move(manifest),
                                                 "opencl/baseline/eltwise_binary_broadcast_f32",
                                                 std::move(scalar_args),
                                                 {0, 1},
                                                 *op,
                                                 GfxOpenClBaselineInputMode::Direct,
                                                 0.0f,
                                                 std::move(*static_u32_scalars));
        }
    }

    if (auto op = compare_op_code(type)) {
        if (node->get_input_size() != 2 ||
            !is_bool_tensor_type(node->get_output_element_type(0)) ||
            !is_f32_tensor_type(node->get_input_element_type(0)) ||
            !is_f32_tensor_type(node->get_input_element_type(1))) {
            return std::nullopt;
        }
        if (same_static_shape(node, 0, 1) &&
            input_static_element_count_matches_output(node, 0, 0) &&
            input_static_element_count_matches_output(node, 1, 0)) {
            auto manifest = make_opencl_baseline_manifest(
                GfxKernelStageFamily::Eltwise,
                "opencl:baseline:" + type + ":f32:same_shape",
                "gfx_opencl_baseline_compare_f32",
                /*direct_inputs=*/2,
                /*scalar_arg_count=*/2);
            return make_opencl_baseline_artifact(std::move(manifest),
                                                 "opencl/baseline/compare_f32",
                                                 {GfxOpenClSourceScalarArg::ElementCount,
                                                  GfxOpenClSourceScalarArg::OpCode},
                                                 {0, 1},
                                                 *op);
        }
        auto static_u32_scalars = compare_broadcast_static_u32_scalars(node);
        if (static_u32_scalars) {
            std::vector<GfxOpenClSourceScalarArg> scalar_args = {
                GfxOpenClSourceScalarArg::ElementCount,
                GfxOpenClSourceScalarArg::OpCode};
            scalar_args.insert(scalar_args.end(),
                               static_u32_scalars->size(),
                               GfxOpenClSourceScalarArg::StaticU32);
            auto manifest = make_opencl_baseline_manifest(
                GfxKernelStageFamily::Eltwise,
                "opencl:baseline:" + type + ":f32:broadcast",
                "gfx_opencl_baseline_compare_broadcast_f32",
                /*direct_inputs=*/2,
                static_cast<uint32_t>(scalar_args.size()));
            return make_opencl_baseline_artifact(
                std::move(manifest),
                "opencl/baseline/compare_broadcast_f32",
                std::move(scalar_args),
                {0, 1},
                *op,
                GfxOpenClBaselineInputMode::Direct,
                0.0f,
                std::move(*static_u32_scalars));
        }
        return std::nullopt;
    }

    if (type == "Select") {
        if (node->get_input_size() != 3 ||
            !is_f32_tensor_type(node->get_output_element_type(0)) ||
            !is_bool_tensor_type(node->get_input_element_type(0)) ||
            !is_f32_tensor_type(node->get_input_element_type(1)) ||
            !is_f32_tensor_type(node->get_input_element_type(2))) {
            return std::nullopt;
        }
        if (input_static_element_count_matches_output(node, 0, 0) &&
            input_static_element_count_matches_output(node, 1, 0) &&
            input_static_element_count_matches_output(node, 2, 0)) {
            auto manifest = make_opencl_baseline_manifest(
                GfxKernelStageFamily::Eltwise,
                "opencl:baseline:Select:bool_f32:same_shape",
                "gfx_opencl_baseline_select_f32",
                /*direct_inputs=*/3,
                /*scalar_arg_count=*/1);
            return make_opencl_baseline_artifact(std::move(manifest),
                                                 "opencl/baseline/select_f32",
                                                 {GfxOpenClSourceScalarArg::ElementCount},
                                                 {0, 1, 2},
                                                 GfxOpenClBaselineOp::Identity);
        }
        auto static_u32_scalars = select_broadcast_static_u32_scalars(node);
        if (static_u32_scalars) {
            std::vector<GfxOpenClSourceScalarArg> scalar_args = {
                GfxOpenClSourceScalarArg::ElementCount};
            scalar_args.insert(scalar_args.end(),
                               static_u32_scalars->size(),
                               GfxOpenClSourceScalarArg::StaticU32);
            auto manifest = make_opencl_baseline_manifest(
                GfxKernelStageFamily::Eltwise,
                "opencl:baseline:Select:bool_f32:broadcast",
                "gfx_opencl_baseline_select_broadcast_f32",
                /*direct_inputs=*/3,
                static_cast<uint32_t>(scalar_args.size()));
            return make_opencl_baseline_artifact(
                std::move(manifest),
                "opencl/baseline/select_broadcast_f32",
                std::move(scalar_args),
                {0, 1, 2},
                GfxOpenClBaselineOp::Identity,
                GfxOpenClBaselineInputMode::Direct,
                0.0f,
                std::move(*static_u32_scalars));
        }
        return std::nullopt;
    }

    if (auto op = logical_unary_op_code(type)) {
        if (node->get_input_size() != 1 ||
            !is_bool_tensor_type(node->get_output_element_type(0)) ||
            !is_bool_tensor_type(node->get_input_element_type(0)) ||
            !input_static_element_count_matches_output(node, 0, 0)) {
            return std::nullopt;
        }
        auto manifest = make_opencl_baseline_manifest(
            GfxKernelStageFamily::Eltwise,
            "opencl:baseline:" + type + ":bool:same_shape",
            "gfx_opencl_baseline_logical_unary_bool",
            /*direct_inputs=*/1,
            /*scalar_arg_count=*/2);
        return make_opencl_baseline_artifact(std::move(manifest),
                                             "opencl/baseline/logical_unary_bool",
                                             {GfxOpenClSourceScalarArg::ElementCount,
                                              GfxOpenClSourceScalarArg::OpCode},
                                             {0},
                                             *op);
    }

    if (auto op = logical_binary_op_code(type)) {
        if (node->get_input_size() != 2 ||
            !is_bool_tensor_type(node->get_output_element_type(0)) ||
            !is_bool_tensor_type(node->get_input_element_type(0)) ||
            !is_bool_tensor_type(node->get_input_element_type(1))) {
            return std::nullopt;
        }
        if (same_static_shape(node, 0, 1) &&
            input_static_element_count_matches_output(node, 0, 0) &&
            input_static_element_count_matches_output(node, 1, 0)) {
            auto manifest = make_opencl_baseline_manifest(
                GfxKernelStageFamily::Eltwise,
                "opencl:baseline:" + type + ":bool:same_shape",
                "gfx_opencl_baseline_logical_binary_bool",
                /*direct_inputs=*/2,
                /*scalar_arg_count=*/2);
            return make_opencl_baseline_artifact(std::move(manifest),
                                                 "opencl/baseline/logical_binary_bool",
                                                 {GfxOpenClSourceScalarArg::ElementCount,
                                                  GfxOpenClSourceScalarArg::OpCode},
                                                 {0, 1},
                                                 *op);
        }
        auto static_u32_scalars = logical_binary_broadcast_static_u32_scalars(node);
        if (static_u32_scalars) {
            std::vector<GfxOpenClSourceScalarArg> scalar_args = {
                GfxOpenClSourceScalarArg::ElementCount,
                GfxOpenClSourceScalarArg::OpCode};
            scalar_args.insert(scalar_args.end(),
                               static_u32_scalars->size(),
                               GfxOpenClSourceScalarArg::StaticU32);
            auto manifest = make_opencl_baseline_manifest(
                GfxKernelStageFamily::Eltwise,
                "opencl:baseline:" + type + ":bool:broadcast",
                "gfx_opencl_baseline_logical_binary_broadcast_bool",
                /*direct_inputs=*/2,
                static_cast<uint32_t>(scalar_args.size()));
            return make_opencl_baseline_artifact(
                std::move(manifest),
                "opencl/baseline/logical_binary_broadcast_bool",
                std::move(scalar_args),
                {0, 1},
                *op,
                GfxOpenClBaselineInputMode::Direct,
                0.0f,
                std::move(*static_u32_scalars));
        }
        return std::nullopt;
    }

    if (auto op = reduce_logical_op_code(type)) {
        if (node->get_input_size() != 2 ||
            !is_bool_tensor_type(node->get_output_element_type(0)) ||
            !is_bool_tensor_type(node->get_input_element_type(0))) {
            return std::nullopt;
        }
        auto static_u32_scalars = reduce_logical_static_u32_scalars(node);
        if (!static_u32_scalars) {
            return std::nullopt;
        }
        std::vector<GfxOpenClSourceScalarArg> scalar_args = {
            GfxOpenClSourceScalarArg::ElementCount,
            GfxOpenClSourceScalarArg::OpCode};
        scalar_args.insert(scalar_args.end(),
                           static_u32_scalars->size(),
                           GfxOpenClSourceScalarArg::StaticU32);
        auto manifest = make_opencl_baseline_manifest(
            GfxKernelStageFamily::Reduction,
            "opencl:baseline:" + type + ":bool:static_axes",
            "gfx_opencl_baseline_reduce_logical_bool",
            /*direct_inputs=*/1,
            static_cast<uint32_t>(scalar_args.size()));
        return make_opencl_baseline_artifact(
            std::move(manifest),
            "opencl/baseline/reduce_logical_bool",
            std::move(scalar_args),
            {0},
            *op,
            GfxOpenClBaselineInputMode::Direct,
            0.0f,
            std::move(*static_u32_scalars));
    }

    return std::nullopt;
}

std::string gfx_opencl_source_artifact_build_options(
    const GfxOpenClSourceArtifact& artifact) {
    std::string joined;
    for (const auto& option : artifact.build_options) {
        if (!joined.empty()) {
            joined.push_back(' ');
        }
        joined += option;
    }
    return joined;
}

}  // namespace gfx_plugin
}  // namespace ov
