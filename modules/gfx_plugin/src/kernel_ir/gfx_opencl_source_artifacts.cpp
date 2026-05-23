// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/gfx_opencl_source_artifacts.hpp"

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
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/strided_slice.hpp"
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
    switch (op) {
    case 32u: return lhs == rhs;
    case 33u: return lhs != rhs;
    case 34u: return lhs > rhs;
    case 35u: return lhs >= rhs;
    case 36u: return lhs < rhs;
    case 37u: return lhs <= rhs;
    default: return 0;
    }
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

bool is_f32_tensor_type(const ov::element::Type& type) {
    return type == ov::element::f32;
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

bool is_linear_copy_op(std::string_view type) {
    return type == "Reshape" || type == "Squeeze" || type == "Unsqueeze" ||
           type == "Convert";
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

std::optional<std::vector<uint32_t>> shapeof_static_u32_scalars(
    const std::shared_ptr<const ov::Node>& node) {
    auto shape_of = ov::as_type_ptr<const ov::op::util::ShapeOfBase>(node);
    if (!shape_of ||
        shape_of->get_input_size() != 1 ||
        shape_of->get_output_size() != 1 ||
        !shape_of->get_input_partial_shape(0).is_static() ||
        !shape_of->get_output_partial_shape(0).is_static()) {
        return std::nullopt;
    }
    const auto output_type = shape_of->get_output_element_type(0);
    if (!is_i32_tensor_type(output_type) && !is_i64_tensor_type(output_type)) {
        return std::nullopt;
    }

    const auto& input_shape = shape_of->get_input_shape(0);
    const auto& output_shape = shape_of->get_output_shape(0);
    const size_t rank = input_shape.size();
    if (rank == 0 || rank > 8 || output_shape.size() != 1 ||
        output_shape[0] != rank) {
        return std::nullopt;
    }

    std::vector<uint32_t> dims(8, 0);
    for (size_t axis = 0; axis < rank; ++axis) {
        if (input_shape[axis] > std::numeric_limits<uint32_t>::max()) {
            return std::nullopt;
        }
        if (is_i32_tensor_type(output_type) &&
            input_shape[axis] >
                static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            return std::nullopt;
        }
        dims[axis] = static_cast<uint32_t>(input_shape[axis]);
    }
    return dims;
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
    artifact.source = kOpenClBaselineSource;
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
    if (!node || node->get_output_size() == 0 ||
        !node->get_output_partial_shape(0).is_static()) {
        return std::nullopt;
    }

    const std::string type = node->get_type_name();
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

    if (type == "ShapeOf") {
        auto static_u32_scalars = shapeof_static_u32_scalars(node);
        if (!static_u32_scalars) {
            return std::nullopt;
        }
        const auto output_type = node->get_output_element_type(0);
        const bool output_i64 = is_i64_tensor_type(output_type);
        std::vector<GfxOpenClSourceScalarArg> scalar_args = {
            GfxOpenClSourceScalarArg::ElementCount};
        scalar_args.insert(scalar_args.end(),
                           static_u32_scalars->size(),
                           GfxOpenClSourceScalarArg::StaticU32);
        const std::string entry_point =
            output_i64 ? "gfx_opencl_baseline_shapeof_i64"
                       : "gfx_opencl_baseline_shapeof_i32";
        auto manifest = make_opencl_baseline_manifest(
            GfxKernelStageFamily::GatherScatter,
            "opencl:baseline:ShapeOf:" +
                std::string(output_i64 ? "i64" : "i32"),
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
            GfxOpenClBaselineInputMode::Direct,
            0.0f,
            std::move(*static_u32_scalars));
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
    }

    if (auto op = compare_op_code(type)) {
        if (node->get_input_size() != 2 ||
            !is_bool_tensor_type(node->get_output_element_type(0)) ||
            !is_f32_tensor_type(node->get_input_element_type(0)) ||
            !is_f32_tensor_type(node->get_input_element_type(1)) ||
            !same_static_shape(node, 0, 1) ||
            !input_static_element_count_matches_output(node, 0, 0) ||
            !input_static_element_count_matches_output(node, 1, 0)) {
            return std::nullopt;
        }
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

    if (type == "Select") {
        if (node->get_input_size() != 3 ||
            !is_f32_tensor_type(node->get_output_element_type(0)) ||
            !is_bool_tensor_type(node->get_input_element_type(0)) ||
            !is_f32_tensor_type(node->get_input_element_type(1)) ||
            !is_f32_tensor_type(node->get_input_element_type(2)) ||
            !input_static_element_count_matches_output(node, 0, 0) ||
            !input_static_element_count_matches_output(node, 1, 0) ||
            !input_static_element_count_matches_output(node, 2, 0)) {
            return std::nullopt;
        }
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
