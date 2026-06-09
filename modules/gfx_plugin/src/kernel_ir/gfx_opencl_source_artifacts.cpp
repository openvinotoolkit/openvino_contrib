// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/gfx_opencl_source_artifacts.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kernel_ir/gfx_custom_kernel_families.hpp"
#include "kernel_ir/opencl_kernels/activation_kernel.hpp"
#include "kernel_ir/opencl_kernels/eltwise_compare_select_kernel.hpp"
#include "kernel_ir/opencl_kernels/eltwise_kernel.hpp"
#include "kernel_ir/opencl_kernels/eltwise_logical_bool_kernel.hpp"
#include "kernel_ir/opencl_kernels/interpolate_f16_kernel.hpp"
#include "kernel_ir/opencl_kernels/interpolate_f32_kernel.hpp"
#include "kernel_ir/opencl_kernels/matmul_f32_kernel.hpp"
#include "kernel_ir/opencl_kernels/range_kernel.hpp"
#include "kernel_ir/opencl_kernels/reduction_f32_kernel.hpp"
#include "kernel_ir/opencl_kernels/reduction_logical_bool_kernel.hpp"
#include "kernel_ir/opencl_kernels/softmax_kernel.hpp"
#include "kernel_ir/opencl_kernels/tile_kernel.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/acos.hpp"
#include "openvino/op/acosh.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/asin.hpp"
#include "openvino/op/asinh.hpp"
#include "openvino/op/atan.hpp"
#include "openvino/op/atanh.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/cosh.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/erf.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/hsigmoid.hpp"
#include "openvino/op/hswish.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/mish.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/sinh.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/op/softsign.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squared_difference.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/tan.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/scatter_elements_update_base.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

constexpr const char *kOpenClConvertSource = R"CLC(
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
)CLC";

constexpr const char *kOpenClTransposeF32Source = R"CLC(
__kernel void gfx_opencl_generated_transpose_f32(__global const float* src,
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
    const uint gid = (uint)get_global_id(0);
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
)CLC";

constexpr const char *kOpenClSliceF32Source = R"CLC(
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
    const uint gid = (uint)get_global_id(0);
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
)CLC";

constexpr const char *kOpenClGatherF32I32Source = R"CLC(
__kernel void gfx_opencl_baseline_gather_i32_f32(__global const float* data,
                                                 __global const int* indices,
                                                 __global float* dst,
                                                 uint count,
                                                 uint outer,
                                                 uint inner,
                                                 uint axis_dim,
                                                 uint indices_count) {
    const uint gid = (uint)get_global_id(0);
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
    const uint gid = (uint)get_global_id(0);
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
    const uint gid = (uint)get_global_id(0);
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
)CLC";

constexpr const char *kOpenClGatherF32I64Source = R"CLC(
__kernel void gfx_opencl_baseline_gather_i64_f32(__global const float* data,
                                                 __global const long* indices,
                                                 __global float* dst,
                                                 uint count,
                                                 uint outer,
                                                 uint inner,
                                                 uint axis_dim,
                                                 uint indices_count) {
    const uint gid = (uint)get_global_id(0);
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
    const uint gid = (uint)get_global_id(0);
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
    const uint gid = (uint)get_global_id(0);
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
)CLC";

constexpr const char *kOpenClScatterF32I32Source = R"CLC(
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
    const uint gid = (uint)get_global_id(0);
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
    const uint gid = (uint)get_global_id(0);
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
    const uint gid = (uint)get_global_id(0);
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
)CLC";

constexpr const char *kOpenClScatterF32I64Source = R"CLC(
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
    const uint gid = (uint)get_global_id(0);
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
    const uint gid = (uint)get_global_id(0);
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
    const uint gid = (uint)get_global_id(0);
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
)CLC";

constexpr const char *kOpenClConcatSplitF32Source = R"CLC(
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

__kernel void gfx_opencl_generated_concat2_f32(__global const float* src0,
                                              __global const float* src1,
                                              __global float* dst,
                                              uint count,
                                              uint out_axis,
                                              uint inner,
                                              uint src0_axis_offset,
                                              uint src0_axis_len,
                                              uint src1_axis_offset,
                                              uint src1_axis_len) {
    const uint gid = (uint)get_global_id(0);
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

__kernel void gfx_opencl_generated_concat3_f32(__global const float* src0,
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
    const uint gid = (uint)get_global_id(0);
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

__kernel void gfx_opencl_generated_concat4_f32(__global const float* src0,
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
    const uint gid = (uint)get_global_id(0);
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

)CLC";

constexpr const char *kOpenClUnaryF32Source = R"CLC(
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

constexpr const char *kOpenClBinaryScalarF32Source = R"CLC(
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
    case 9u: {
        const float rem = fmod(lhs, rhs);
        return fabs(rem) >= fabs(rhs) ? 0.0f : rem;
    }
    case 10u: {
        const float rem = lhs - floor(lhs / rhs) * rhs;
        return fabs(rem) >= fabs(rhs) ? 0.0f : rem;
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

constexpr const char *kOpenClBinaryConstF32Source = R"CLC(
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
    case 9u: {
        const float rem = fmod(lhs, rhs);
        return fabs(rem) >= fabs(rhs) ? 0.0f : rem;
    }
    case 10u: {
        const float rem = lhs - floor(lhs / rhs) * rhs;
        return fabs(rem) >= fabs(rhs) ? 0.0f : rem;
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

constexpr const char *kOpenClDynamicDataMovementF16Source = R"CLC(
#define GFX_LOW_U32_SHAPE_VALUE(words, idx) ((words)[(idx) * 2u])
#define GFX_LOAD_I32_SHAPE_VALUE(words, idx) ((int)GFX_LOW_U32_SHAPE_VALUE((words), (idx)))
#define GFX_LOAD_F16_BITS(src, idx) \
    (((idx) & 1u) == 0u ? ((src)[(idx) >> 1u] & 65535u) : (((src)[(idx) >> 1u] >> 16u) & 65535u))
#define GFX_STORE_F16_PAIR(dst, word_idx, lo, hi) \
    ((dst)[(word_idx)] = ((lo) & 65535u) | (((hi) & 65535u) << 16u))

static inline float gfx_f16_bits_to_f32(uint bits) {
    const uint sign = (bits & 32768u) << 16u;
    uint exp = (bits >> 10u) & 31u;
    uint mant = bits & 1023u;
    uint out = sign;
    if (exp == 0u) {
        if (mant == 0u) {
            return as_float(out);
        }
        int normalized_exp = -14;
        while ((mant & 1024u) == 0u) {
            mant <<= 1u;
            --normalized_exp;
        }
        mant &= 1023u;
        out |= (uint)(normalized_exp + 127) << 23u;
        out |= mant << 13u;
        return as_float(out);
    }
    if (exp == 31u) {
        out |= 2139095040u | (mant << 13u);
        return as_float(out);
    }
    out |= (exp + 112u) << 23u;
    out |= mant << 13u;
    return as_float(out);
}

static inline uint gfx_f32_to_f16_bits(float value) {
    const uint bits = as_uint(value);
    const uint sign = (bits >> 16u) & 32768u;
    const uint exp_bits = (bits >> 23u) & 255u;
    const uint mant = bits & 8388607u;
    if (exp_bits == 255u) {
        return sign | 31744u | (mant != 0u ? 512u : 0u);
    }
    int exp = (int)exp_bits - 127 + 15;
    if (exp <= 0) {
        if (exp < -10) {
            return sign;
        }
        uint sub = (mant | 8388608u) >> (uint)(1 - exp);
        return sign | ((sub + 4096u) >> 13u);
    }
    if (exp >= 31) {
        return sign | 31744u;
    }
    uint half_mant = (mant + 4096u) >> 13u;
    if (half_mant == 1024u) {
        half_mant = 0u;
        ++exp;
        if (exp >= 31) {
            return sign | 31744u;
        }
    }
    return sign | ((uint)exp << 10u) | half_mant;
}

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

__kernel void gfx_opencl_generated_concat2_f16(__global const uint* src0,
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

__kernel void gfx_opencl_generated_concat3_f16(__global const uint* src0,
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

__kernel void gfx_opencl_generated_concat4_f16(__global const uint* src0,
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
    for (int axis = (int)rank - 2; axis >= 0; --axis) {
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
    for (int axis = (int)rank - 2; axis >= 0; --axis) {
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

)CLC";

constexpr const char *kOpenClBinaryBroadcastF32Source = R"CLC(
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
    case 9u: {
        const float rem = fmod(lhs, rhs);
        return fabs(rem) >= fabs(rhs) ? 0.0f : rem;
    }
    case 10u: {
        const float rem = lhs - floor(lhs / rhs) * rhs;
        return fabs(rem) >= fabs(rhs) ? 0.0f : rem;
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

constexpr const char *kOpenClBinaryI32Source = R"CLC(
static inline int gfx_pow_i32_exact(int base, int exp) {
    if (exp < 0) {
        return (int)pow((float)base, (float)exp);
    }
    int result = 1;
    int factor = base;
    uint e = (uint)exp;
    while (e != 0u) {
        if ((e & 1u) != 0u) {
            result *= factor;
        }
        e >>= 1u;
        if (e != 0u) {
            factor *= factor;
        }
    }
    return result;
}

static inline int gfx_binary_i32(int lhs, int rhs, uint op) {
    switch (op) {
    case 1u: return lhs + rhs;
    case 2u: return lhs - rhs;
    case 3u: return lhs * rhs;
    case 4u: return lhs / rhs;
    case 5u: return lhs > rhs ? lhs : rhs;
    case 6u: return lhs < rhs ? lhs : rhs;
    case 7u: return gfx_pow_i32_exact(lhs, rhs);
    case 8u: {
        const int diff = lhs - rhs;
        return diff * diff;
    }
    case 9u: return lhs % rhs;
    case 10u: {
        const int rem = lhs % rhs;
        const int fix = ((rhs < 0) != (rem < 0)) ? rhs : 0;
        return rem + fix;
    }
    default: return lhs;
    }
}

static inline uint gfx_broadcast_offset_i32(uint idx,
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

__kernel void gfx_opencl_baseline_binary_i32(__global const int* lhs,
                                             __global const int* rhs,
                                             __global int* dst,
                                             uint count,
                                             uint op) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= count) {
        return;
    }
    dst[gid] = gfx_binary_i32(lhs[gid], rhs[gid], op);
}

__kernel void gfx_opencl_baseline_binary_scalar_i32(__global const int* lhs,
                                                    __global const int* rhs,
                                                    __global int* dst,
                                                    uint count,
                                                    uint op,
                                                    uint input_mode) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= count) {
        return;
    }
    const int l = input_mode == 2u ? lhs[0] : lhs[gid];
    const int r = input_mode == 1u ? rhs[0] : rhs[gid];
    dst[gid] = gfx_binary_i32(l, r, op);
}

__kernel void gfx_opencl_baseline_binary_broadcast_i32(__global const int* lhs,
                                                       __global const int* rhs,
                                                       __global int* dst,
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
    const uint lhs_offset = gfx_broadcast_offset_i32(gid, rank, out_dim1, out_dim2, out_dim3,
                                                     lhs_stride0, lhs_stride1, lhs_stride2, lhs_stride3);
    const uint rhs_offset = gfx_broadcast_offset_i32(gid, rank, out_dim1, out_dim2, out_dim3,
                                                     rhs_stride0, rhs_stride1, rhs_stride2, rhs_stride3);
    dst[gid] = gfx_binary_i32(lhs[lhs_offset], rhs[rhs_offset], op);
}
)CLC";

constexpr const char *kOpenClBinaryF16Source = R"CLC(
#define GFX_LOAD_F16_BITS(src, idx) \
    (((idx) & 1u) == 0u ? ((src)[(idx) >> 1u] & 65535u) : (((src)[(idx) >> 1u] >> 16u) & 65535u))
#define GFX_STORE_F16_PAIR(dst, word_idx, lo, hi) \
    ((dst)[(word_idx)] = ((lo) & 65535u) | (((hi) & 65535u) << 16u))

static inline float gfx_f16_bits_to_f32(uint bits) {
    const uint sign = (bits & 32768u) << 16u;
    uint exp = (bits >> 10u) & 31u;
    uint mant = bits & 1023u;
    uint out = sign;
    if (exp == 0u) {
        if (mant == 0u) {
            return as_float(out);
        }
        int normalized_exp = -14;
        while ((mant & 1024u) == 0u) {
            mant <<= 1u;
            --normalized_exp;
        }
        mant &= 1023u;
        out |= (uint)(normalized_exp + 127) << 23u;
        out |= mant << 13u;
        return as_float(out);
    }
    if (exp == 31u) {
        out |= 2139095040u | (mant << 13u);
        return as_float(out);
    }
    out |= (exp + 112u) << 23u;
    out |= mant << 13u;
    return as_float(out);
}

static inline uint gfx_f32_to_f16_bits(float value) {
    const uint bits = as_uint(value);
    const uint sign = (bits >> 16u) & 32768u;
    const uint exp_bits = (bits >> 23u) & 255u;
    const uint mant = bits & 8388607u;
    if (exp_bits == 255u) {
        return sign | 31744u | (mant != 0u ? 512u : 0u);
    }
    int exp = (int)exp_bits - 127 + 15;
    if (exp <= 0) {
        if (exp < -10) {
            return sign;
        }
        uint sub = (mant | 8388608u) >> (uint)(1 - exp);
        return sign | ((sub + 4096u) >> 13u);
    }
    if (exp >= 31) {
        return sign | 31744u;
    }
    uint half_mant = (mant + 4096u) >> 13u;
    if (half_mant == 1024u) {
        half_mant = 0u;
        ++exp;
        if (exp >= 31) {
            return sign | 31744u;
        }
    }
    return sign | ((uint)exp << 10u) | half_mant;
}

static inline uint gfx_binary_f16_bits(uint lhs_bits, uint rhs_bits, uint op) {
    const float lhs = gfx_f16_bits_to_f32(lhs_bits);
    const float rhs = gfx_f16_bits_to_f32(rhs_bits);
    float out = lhs;
    switch (op) {
    case 1u: out = lhs + rhs; break;
    case 2u: out = lhs - rhs; break;
    case 3u: out = lhs * rhs; break;
    case 4u: out = lhs / rhs; break;
    case 5u: out = fmax(lhs, rhs); break;
    case 6u: out = fmin(lhs, rhs); break;
    case 7u: out = pow(lhs, rhs); break;
    case 8u: {
        const float diff = lhs - rhs;
        out = diff * diff;
        break;
    }
    case 9u: {
        const float rem = fmod(lhs, rhs);
        out = fabs(rem) >= fabs(rhs) ? 0.0f : rem;
        break;
    }
    case 10u: {
        const float rem = lhs - floor(lhs / rhs) * rhs;
        out = fabs(rem) >= fabs(rhs) ? 0.0f : rem;
        break;
    }
    default: break;
    }
    return gfx_f32_to_f16_bits(out);
}

static inline uint gfx_broadcast_offset_f16(uint idx,
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

__kernel void gfx_opencl_baseline_binary_f16(__global const uint* lhs,
                                             __global const uint* rhs,
                                             __global uint* dst,
                                             uint count,
                                             uint op) {
    const uint word_idx = (uint)get_global_id(0);
    const uint elem0 = word_idx * 2u;
    if (elem0 >= count) {
        return;
    }
    const uint lo = gfx_binary_f16_bits(GFX_LOAD_F16_BITS(lhs, elem0),
                                        GFX_LOAD_F16_BITS(rhs, elem0),
                                        op);
    uint hi = 0u;
    if (elem0 + 1u < count) {
        hi = gfx_binary_f16_bits(GFX_LOAD_F16_BITS(lhs, elem0 + 1u),
                                 GFX_LOAD_F16_BITS(rhs, elem0 + 1u),
                                 op);
    }
    GFX_STORE_F16_PAIR(dst, word_idx, lo, hi);
}

__kernel void gfx_opencl_baseline_binary_scalar_f16(__global const uint* lhs,
                                                    __global const uint* rhs,
                                                    __global uint* dst,
                                                    uint count,
                                                    uint op,
                                                    uint input_mode) {
    const uint word_idx = (uint)get_global_id(0);
    const uint elem0 = word_idx * 2u;
    if (elem0 >= count) {
        return;
    }
    const uint lhs0 = input_mode == 2u ? GFX_LOAD_F16_BITS(lhs, 0u) : GFX_LOAD_F16_BITS(lhs, elem0);
    const uint rhs0 = input_mode == 1u ? GFX_LOAD_F16_BITS(rhs, 0u) : GFX_LOAD_F16_BITS(rhs, elem0);
    const uint lo = gfx_binary_f16_bits(lhs0, rhs0, op);
    uint hi = 0u;
    if (elem0 + 1u < count) {
        const uint lhs1 = input_mode == 2u ? GFX_LOAD_F16_BITS(lhs, 0u) : GFX_LOAD_F16_BITS(lhs, elem0 + 1u);
        const uint rhs1 = input_mode == 1u ? GFX_LOAD_F16_BITS(rhs, 0u) : GFX_LOAD_F16_BITS(rhs, elem0 + 1u);
        hi = gfx_binary_f16_bits(lhs1, rhs1, op);
    }
    GFX_STORE_F16_PAIR(dst, word_idx, lo, hi);
}

__kernel void gfx_opencl_baseline_binary_broadcast_f16(__global const uint* lhs,
                                                       __global const uint* rhs,
                                                       __global uint* dst,
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
    const uint elem0 = word_idx * 2u;
    if (elem0 >= count) {
        return;
    }
    (void)out_dim0;
    const uint lhs0 = gfx_broadcast_offset_f16(elem0, rank, out_dim1, out_dim2, out_dim3,
                                               lhs_stride0, lhs_stride1, lhs_stride2, lhs_stride3);
    const uint rhs0 = gfx_broadcast_offset_f16(elem0, rank, out_dim1, out_dim2, out_dim3,
                                               rhs_stride0, rhs_stride1, rhs_stride2, rhs_stride3);
    const uint lo = gfx_binary_f16_bits(GFX_LOAD_F16_BITS(lhs, lhs0),
                                        GFX_LOAD_F16_BITS(rhs, rhs0),
                                        op);
    uint hi = 0u;
    if (elem0 + 1u < count) {
        const uint elem1 = elem0 + 1u;
        const uint lhs1 = gfx_broadcast_offset_f16(elem1, rank, out_dim1, out_dim2, out_dim3,
                                                   lhs_stride0, lhs_stride1, lhs_stride2, lhs_stride3);
        const uint rhs1 = gfx_broadcast_offset_f16(elem1, rank, out_dim1, out_dim2, out_dim3,
                                                   rhs_stride0, rhs_stride1, rhs_stride2, rhs_stride3);
        hi = gfx_binary_f16_bits(GFX_LOAD_F16_BITS(lhs, lhs1),
                                 GFX_LOAD_F16_BITS(rhs, rhs1),
                                 op);
    }
    GFX_STORE_F16_PAIR(dst, word_idx, lo, hi);
}
)CLC";

bool is_f32_tensor_type(const ov::element::Type &type) {
  return type == ov::element::f32;
}

bool is_f16_tensor_type(const ov::element::Type &type) {
  return type == ov::element::f16;
}

bool is_bool_tensor_type(const ov::element::Type &type) {
  return type == ov::element::boolean;
}

bool is_i32_tensor_type(const ov::element::Type &type) {
  return type == ov::element::i32;
}

bool is_i64_tensor_type(const ov::element::Type &type) {
  return type == ov::element::i64;
}

bool is_opencl_convert_tensor_type(const ov::element::Type &type) {
  return is_f32_tensor_type(type) || is_i32_tensor_type(type) ||
         is_i64_tensor_type(type);
}

bool is_opencl_binary_tensor_type(const ov::element::Type &type) {
  return is_f16_tensor_type(type) || is_f32_tensor_type(type) ||
         is_i32_tensor_type(type);
}

const char *opencl_scalar_type_suffix(const ov::element::Type &type) {
  if (is_f16_tensor_type(type)) {
    return "f16";
  }
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

bool same_static_shape(const std::shared_ptr<const ov::Node> &node,
                       size_t input_a, size_t input_b) {
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

bool same_static_element_count_input_output(
    const std::shared_ptr<const ov::Node> &node, size_t input_idx,
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

std::optional<GfxOpenClArtifactOp>
activation_op_code(const std::shared_ptr<const ov::Node> &node) {
  if (ov::as_type_ptr<const ov::op::v0::Relu>(node)) {
    return GfxOpenClArtifactOp::Relu;
  }
  if (ov::as_type_ptr<const ov::op::v0::Sigmoid>(node)) {
    return GfxOpenClArtifactOp::Sigmoid;
  }
  if (ov::as_type_ptr<const ov::op::v0::Tanh>(node)) {
    return GfxOpenClArtifactOp::Tanh;
  }
  if (ov::as_type_ptr<const ov::op::v0::Gelu>(node)) {
    return GfxOpenClArtifactOp::GeluErf;
  }
  if (auto gelu = ov::as_type_ptr<const ov::op::v7::Gelu>(node)) {
    return gelu->get_approximation_mode() == ov::op::GeluApproximationMode::TANH
               ? GfxOpenClArtifactOp::GeluTanh
               : GfxOpenClArtifactOp::GeluErf;
  }
  if (ov::as_type_ptr<const ov::op::v4::HSwish>(node)) {
    return GfxOpenClArtifactOp::HSwish;
  }
  if (ov::as_type_ptr<const ov::op::v5::HSigmoid>(node)) {
    return GfxOpenClArtifactOp::HSigmoid;
  }
  if (ov::as_type_ptr<const ov::op::v4::SoftPlus>(node)) {
    return GfxOpenClArtifactOp::SoftPlus;
  }
  if (ov::as_type_ptr<const ov::op::v4::Swish>(node)) {
    return GfxOpenClArtifactOp::Swish;
  }
  if (ov::as_type_ptr<const ov::op::v4::Mish>(node)) {
    return GfxOpenClArtifactOp::Mish;
  }
  if (ov::as_type_ptr<const ov::op::v9::SoftSign>(node)) {
    return GfxOpenClArtifactOp::SoftSign;
  }
  if (ov::as_type_ptr<const ov::op::v0::Abs>(node)) {
    return GfxOpenClArtifactOp::Abs;
  }
  if (ov::as_type_ptr<const ov::op::v0::Sign>(node)) {
    return GfxOpenClArtifactOp::Sign;
  }
  if (ov::as_type_ptr<const ov::op::v0::Clamp>(node)) {
    return GfxOpenClArtifactOp::Clamp;
  }
  if (ov::as_type_ptr<const ov::op::v0::Negative>(node)) {
    return GfxOpenClArtifactOp::Negative;
  }
  if (ov::as_type_ptr<const ov::op::v0::Sin>(node)) {
    return GfxOpenClArtifactOp::Sin;
  }
  if (ov::as_type_ptr<const ov::op::v0::Cos>(node)) {
    return GfxOpenClArtifactOp::Cos;
  }
  if (ov::as_type_ptr<const ov::op::v0::Tan>(node)) {
    return GfxOpenClArtifactOp::Tan;
  }
  if (ov::as_type_ptr<const ov::op::v0::Erf>(node)) {
    return GfxOpenClArtifactOp::Erf;
  }
  if (ov::as_type_ptr<const ov::op::v0::Asin>(node)) {
    return GfxOpenClArtifactOp::Asin;
  }
  if (ov::as_type_ptr<const ov::op::v0::Acos>(node)) {
    return GfxOpenClArtifactOp::Acos;
  }
  if (ov::as_type_ptr<const ov::op::v0::Atan>(node)) {
    return GfxOpenClArtifactOp::Atan;
  }
  if (ov::as_type_ptr<const ov::op::v3::Asinh>(node)) {
    return GfxOpenClArtifactOp::Asinh;
  }
  if (ov::as_type_ptr<const ov::op::v3::Acosh>(node)) {
    return GfxOpenClArtifactOp::Acosh;
  }
  if (ov::as_type_ptr<const ov::op::v3::Atanh>(node)) {
    return GfxOpenClArtifactOp::Atanh;
  }
  if (ov::as_type_ptr<const ov::op::v0::Sinh>(node)) {
    return GfxOpenClArtifactOp::Sinh;
  }
  if (ov::as_type_ptr<const ov::op::v0::Cosh>(node)) {
    return GfxOpenClArtifactOp::Cosh;
  }
  if (auto round = ov::as_type_ptr<const ov::op::v5::Round>(node)) {
    return round->get_mode() == ov::op::v5::Round::RoundMode::HALF_TO_EVEN
               ? GfxOpenClArtifactOp::RoundEven
               : GfxOpenClArtifactOp::RoundAway;
  }
  if (ov::as_type_ptr<const ov::op::v0::Exp>(node)) {
    return GfxOpenClArtifactOp::Exp;
  }
  if (ov::as_type_ptr<const ov::op::v0::Log>(node)) {
    return GfxOpenClArtifactOp::Log;
  }
  if (ov::as_type_ptr<const ov::op::v0::Sqrt>(node)) {
    return GfxOpenClArtifactOp::Sqrt;
  }
  if (ov::as_type_ptr<const ov::op::v0::Floor>(node)) {
    return GfxOpenClArtifactOp::Floor;
  }
  if (ov::as_type_ptr<const ov::op::v0::Ceiling>(node)) {
    return GfxOpenClArtifactOp::Ceiling;
  }
  if (ov::as_type_ptr<const ov::op::v0::Elu>(node)) {
    return GfxOpenClArtifactOp::Elu;
  }
  return std::nullopt;
}

std::optional<GfxOpenClArtifactOp>
binary_op_code(const std::shared_ptr<const ov::Node> &node) {
  if (ov::as_type_ptr<const ov::op::v1::Add>(node)) {
    return GfxOpenClArtifactOp::Add;
  }
  if (ov::as_type_ptr<const ov::op::v1::Subtract>(node)) {
    return GfxOpenClArtifactOp::Subtract;
  }
  if (ov::as_type_ptr<const ov::op::v1::Multiply>(node)) {
    return GfxOpenClArtifactOp::Multiply;
  }
  if (ov::as_type_ptr<const ov::op::v1::Divide>(node)) {
    return GfxOpenClArtifactOp::Divide;
  }
  if (ov::as_type_ptr<const ov::op::v1::Maximum>(node)) {
    return GfxOpenClArtifactOp::Maximum;
  }
  if (ov::as_type_ptr<const ov::op::v1::Minimum>(node)) {
    return GfxOpenClArtifactOp::Minimum;
  }
  if (ov::as_type_ptr<const ov::op::v1::Power>(node)) {
    return GfxOpenClArtifactOp::Power;
  }
  if (ov::as_type_ptr<const ov::op::v0::SquaredDifference>(node)) {
    return GfxOpenClArtifactOp::SquaredDifference;
  }
  if (ov::as_type_ptr<const ov::op::v1::Mod>(node)) {
    return GfxOpenClArtifactOp::Mod;
  }
  if (ov::as_type_ptr<const ov::op::v1::FloorMod>(node)) {
    return GfxOpenClArtifactOp::FloorMod;
  }
  return std::nullopt;
}

std::optional<GfxOpenClArtifactOp> compare_op_code(std::string_view type) {
  if (type == "Equal")
    return GfxOpenClArtifactOp::Equal;
  if (type == "NotEqual")
    return GfxOpenClArtifactOp::NotEqual;
  if (type == "Greater")
    return GfxOpenClArtifactOp::Greater;
  if (type == "GreaterEqual")
    return GfxOpenClArtifactOp::GreaterEqual;
  if (type == "Less")
    return GfxOpenClArtifactOp::Less;
  if (type == "LessEqual")
    return GfxOpenClArtifactOp::LessEqual;
  return std::nullopt;
}

std::optional<GfxOpenClArtifactOp>
logical_unary_op_code(std::string_view type) {
  if (type == "LogicalNot")
    return GfxOpenClArtifactOp::LogicalNot;
  return std::nullopt;
}

std::optional<GfxOpenClArtifactOp>
logical_binary_op_code(std::string_view type) {
  if (type == "LogicalAnd")
    return GfxOpenClArtifactOp::LogicalAnd;
  if (type == "LogicalOr")
    return GfxOpenClArtifactOp::LogicalOr;
  if (type == "LogicalXor" || type == "Xor")
    return GfxOpenClArtifactOp::LogicalXor;
  return std::nullopt;
}

std::optional<GfxOpenClArtifactOp>
reduce_logical_op_code(std::string_view type) {
  if (type == "ReduceLogicalAnd")
    return GfxOpenClArtifactOp::ReduceLogicalAnd;
  if (type == "ReduceLogicalOr")
    return GfxOpenClArtifactOp::ReduceLogicalOr;
  return std::nullopt;
}

std::optional<GfxOpenClArtifactOp>
reduce_numeric_op_code(std::string_view type) {
  if (type == "ReduceSum")
    return GfxOpenClArtifactOp::ReduceSum;
  if (type == "ReduceMean")
    return GfxOpenClArtifactOp::ReduceMean;
  if (type == "ReduceMax")
    return GfxOpenClArtifactOp::ReduceMax;
  if (type == "ReduceMin")
    return GfxOpenClArtifactOp::ReduceMin;
  if (type == "ReduceProd")
    return GfxOpenClArtifactOp::ReduceProd;
  if (type == "ReduceL1")
    return GfxOpenClArtifactOp::ReduceL1;
  if (type == "ReduceL2")
    return GfxOpenClArtifactOp::ReduceL2;
  return std::nullopt;
}

bool is_linear_copy_op(std::string_view type) {
  return type == "Reshape" || type == "Squeeze" || type == "Unsqueeze";
}

bool is_static_scalar_like_input(const std::shared_ptr<const ov::Node> &node,
                                 size_t input_idx) {
  if (!node || input_idx >= node->get_input_size() ||
      !node->get_input_partial_shape(input_idx).is_static()) {
    return false;
  }
  return ov::shape_size(node->get_input_shape(input_idx)) == 1;
}

bool input_static_element_count_matches_output(
    const std::shared_ptr<const ov::Node> &node, size_t input_idx,
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

std::optional<float>
scalar_f32_constant_input(const std::shared_ptr<const ov::Node> &node,
                          size_t input_idx) {
  if (!node || input_idx >= node->get_input_size()) {
    return std::nullopt;
  }
  auto constant = ov::as_type_ptr<const ov::op::v0::Constant>(
      node->input_value(input_idx).get_node_shared_ptr());
  if (!constant || !is_f32_tensor_type(constant->get_output_element_type(0)) ||
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

std::optional<float>
scalar_float_constant_input(const std::shared_ptr<const ov::Node> &node,
                            size_t input_idx) {
  if (!node || input_idx >= node->get_input_size()) {
    return std::nullopt;
  }
  auto constant = ov::as_type_ptr<const ov::op::v0::Constant>(
      node->input_value(input_idx).get_node_shared_ptr());
  if (!constant ||
      constant->get_output_element_type(0) != node->get_input_element_type(0) ||
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

bool scalar_float_input(const std::shared_ptr<const ov::Node> &node,
                        size_t input_idx) {
  if (!node || input_idx >= node->get_input_size()) {
    return false;
  }
  return node->get_input_element_type(input_idx) ==
             node->get_input_element_type(0) &&
         node->get_input_partial_shape(input_idx).is_static() &&
         ov::shape_size(node->get_input_shape(input_idx)) == 1;
}

std::optional<std::vector<int64_t>>
constant_i64_vector_input(const std::shared_ptr<const ov::Node> &node,
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

bool checked_u32(uint64_t value, uint32_t &out) {
  if (value > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  out = static_cast<uint32_t>(value);
  return true;
}

uint64_t shape_product_range(const ov::Shape &shape, size_t begin, size_t end) {
  uint64_t product = 1;
  for (size_t axis = begin; axis < end; ++axis) {
    product *= shape[axis];
  }
  return product;
}

bool append_shape_u32(const ov::Shape &shape, size_t max_rank,
                      std::vector<uint32_t> &values);

std::optional<uint32_t> aligned_broadcast_stride(const ov::Shape &input_shape,
                                                 const ov::Shape &output_shape,
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
  if (!checked_u32(shape_product_range(input_shape, input_axis + 1, input_rank),
                   stride)) {
    return std::nullopt;
  }
  return stride;
}

bool append_aligned_broadcast_strides_u32(const ov::Shape &input_shape,
                                          const ov::Shape &output_shape,
                                          size_t max_rank,
                                          std::vector<uint32_t> &values) {
  if (output_shape.size() > max_rank ||
      input_shape.size() > output_shape.size()) {
    return false;
  }
  for (size_t axis = 0; axis < output_shape.size(); ++axis) {
    const auto stride =
        aligned_broadcast_stride(input_shape, output_shape, axis);
    if (!stride) {
      return false;
    }
    values.push_back(*stride);
  }
  values.insert(values.end(), max_rank - output_shape.size(), 0u);
  return true;
}

bool output_type_matches(const std::shared_ptr<const ov::Node> &node,
                         const ov::element::Type &output_type) {
  return node && node->get_output_size() == 1 &&
         node->get_output_element_type(0) == output_type;
}

std::optional<std::vector<uint32_t>>
broadcast_static_u32_scalars(const std::shared_ptr<const ov::Node> &node,
                             const std::vector<ov::element::Type> &input_types,
                             const ov::element::Type &output_type) {
  if (!node || node->get_input_size() != input_types.size() ||
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

  const auto &output_shape = node->get_output_shape(0);
  const size_t rank = output_shape.size();
  if (rank == 0 || rank > 4 || ov::shape_size(output_shape) == 0) {
    return std::nullopt;
  }

  for (size_t input_idx = 0; input_idx < input_types.size(); ++input_idx) {
    const auto &input_shape = node->get_input_shape(input_idx);
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
                                    output_shape, axis)) {
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
                                              output_shape, 4, scalars)) {
      return std::nullopt;
    }
  }
  return scalars;
}

std::optional<std::vector<uint32_t>> binary_broadcast_static_u32_scalars(
    const std::shared_ptr<const ov::Node> &node) {
  if (!node || node->get_input_size() != 2 || node->get_output_size() != 1) {
    return std::nullopt;
  }
  const auto element_type = node->get_output_element_type(0);
  if (!is_opencl_binary_tensor_type(element_type)) {
    return std::nullopt;
  }
  return broadcast_static_u32_scalars(node, {element_type, element_type},
                                      element_type);
}

GfxKernelStageManifest make_opencl_source_manifest(
    GfxKernelStageFamily family, std::string specialization_key,
    std::string entry_point, uint32_t direct_inputs, uint32_t scalar_arg_count,
    uint32_t direct_outputs = 1);

GfxOpenClSourceArtifact make_opencl_source_artifact(
    GfxKernelStageManifest manifest, std::string source_id,
    std::vector<GfxOpenClSourceScalarArg> scalar_args,
    std::vector<size_t> direct_input_indices, GfxOpenClArtifactOp op,
    GfxOpenClArtifactInputMode input_mode = GfxOpenClArtifactInputMode::Direct,
    float scalar_constant_f32 = 0.0f,
    std::vector<uint32_t> static_u32_scalars = {},
    GfxOpenClSourceElementCountSource element_count_source =
        GfxOpenClSourceElementCountSource::Output0,
    std::vector<float> static_f32_scalars = {});

bool is_numpy_aligned_binary_broadcast(
    const ov::op::util::BinaryElementwiseArithmetic &op) {
  return op.get_autob().m_type == ov::op::AutoBroadcastType::NUMPY;
}

std::optional<GfxOpenClSourceArtifact>
make_opencl_activation_artifact(const std::shared_ptr<const ov::Node> &node) {
  const auto op = activation_op_code(node);
  if (!op || !node ||
      (node->get_input_size() != 1 &&
       !(ov::as_type_ptr<const ov::op::v4::Swish>(node) &&
         node->get_input_size() == 2)) ||
      node->get_output_size() != 1 ||
      node->get_input_element_type(0) != node->get_output_element_type(0) ||
      (!is_f32_tensor_type(node->get_output_element_type(0)) &&
       !is_f16_tensor_type(node->get_output_element_type(0))) ||
      !input_static_element_count_matches_output(node, 0, 0)) {
    return std::nullopt;
  }

  const std::string type_suffix =
      opencl_scalar_type_suffix(node->get_output_element_type(0));
  const std::string entry_point =
      "gfx_opencl_generated_activation_" + type_suffix;
  float alpha = 0.0f;
  float beta = 0.0f;
  if (auto elu = ov::as_type_ptr<const ov::op::v0::Elu>(node)) {
    alpha = static_cast<float>(elu->get_alpha());
  }
  if (auto clamp = ov::as_type_ptr<const ov::op::v0::Clamp>(node)) {
    alpha = static_cast<float>(clamp->get_min());
    beta = static_cast<float>(clamp->get_max());
  }
  if (auto swish = ov::as_type_ptr<const ov::op::v4::Swish>(node)) {
    alpha = 1.0f;
    if (swish->get_input_size() == 2) {
      if (!scalar_float_input(node, 1)) {
        return std::nullopt;
      }
      const auto beta_value = scalar_float_constant_input(node, 1);
      if (!beta_value) {
        const auto runtime_entry_point =
            "gfx_opencl_generated_activation_runtime_beta_" + type_suffix;
        auto manifest = make_opencl_source_manifest(
            GfxKernelStageFamily::Activation,
            "opencl:generated:activation_runtime_beta:" +
                std::string(node->get_type_name()) + ":" + type_suffix,
            runtime_entry_point,
            /*direct_inputs=*/2,
            /*scalar_arg_count=*/2);
        return make_opencl_source_artifact(
            std::move(manifest),
            "opencl/generated/activation_runtime_beta_" + type_suffix,
            {GfxOpenClSourceScalarArg::ElementCount,
             GfxOpenClSourceScalarArg::OpCode},
            {0, 1}, *op);
      }
      alpha = *beta_value;
    }
  }
  auto manifest = make_opencl_source_manifest(
      GfxKernelStageFamily::Activation,
      "opencl:generated:activation:" + std::string(node->get_type_name()) +
          ":" + type_suffix,
      entry_point,
      /*direct_inputs=*/1,
      /*scalar_arg_count=*/4);
  return make_opencl_source_artifact(
      std::move(manifest), "opencl/generated/activation_" + type_suffix,
      {GfxOpenClSourceScalarArg::ElementCount, GfxOpenClSourceScalarArg::OpCode,
       GfxOpenClSourceScalarArg::StaticF32,
       GfxOpenClSourceScalarArg::StaticF32},
      {0}, *op, GfxOpenClArtifactInputMode::Direct, 0.0f, {},
      GfxOpenClSourceElementCountSource::Output0, {alpha, beta});
}

std::optional<GfxOpenClSourceArtifact>
make_opencl_eltwise_artifact(const std::shared_ptr<const ov::Node> &node) {
  auto eltwise =
      ov::as_type_ptr<const ov::op::util::BinaryElementwiseArithmetic>(node);
  const auto op = binary_op_code(node);
  if (!eltwise || !op || eltwise->get_input_size() != 2 ||
      eltwise->get_output_size() != 1 ||
      eltwise->get_output_element_type(0) !=
          eltwise->get_input_element_type(0) ||
      eltwise->get_output_element_type(0) !=
          eltwise->get_input_element_type(1) ||
      !is_opencl_binary_tensor_type(eltwise->get_output_element_type(0))) {
    return std::nullopt;
  }

  const auto element_type = eltwise->get_output_element_type(0);
  const bool is_f32 = is_f32_tensor_type(element_type);
  const std::string type_suffix = opencl_scalar_type_suffix(element_type);
  const bool lhs_matches_output =
      input_static_element_count_matches_output(node, 0, 0);
  const bool rhs_matches_output =
      input_static_element_count_matches_output(node, 1, 0);
  const auto lhs_constant = scalar_f32_constant_input(node, 0);
  const auto rhs_constant = scalar_f32_constant_input(node, 1);
  const std::string type = node->get_type_name();
  const auto source_id = [&type_suffix](std::string_view variant) {
    const std::string suffix =
        variant.empty() ? "binary_" + type_suffix : std::string(variant);
    return "opencl/generated/eltwise_" + suffix;
  };

  if (same_static_shape(node, 0, 1) && lhs_matches_output &&
      rhs_matches_output) {
    const std::string entry_point =
        "gfx_opencl_generated_eltwise_binary_" + type_suffix;
    auto manifest = make_opencl_source_manifest(
        GfxKernelStageFamily::Eltwise,
        "opencl:generated:eltwise:" + type + ":" + type_suffix + ":same_shape",
        entry_point,
        /*direct_inputs=*/2,
        /*scalar_arg_count=*/2);
    return make_opencl_source_artifact(std::move(manifest), source_id({}),
                                       {GfxOpenClSourceScalarArg::ElementCount,
                                        GfxOpenClSourceScalarArg::OpCode},
                                       {0, 1}, *op);
  }

  if (!is_numpy_aligned_binary_broadcast(*eltwise)) {
    return std::nullopt;
  }

  if (is_f32 && rhs_constant && lhs_matches_output) {
    auto manifest = make_opencl_source_manifest(
        GfxKernelStageFamily::Eltwise,
        "opencl:generated:eltwise:" + type + ":f32:rhs_scalar_const",
        "gfx_opencl_generated_eltwise_const_f32",
        /*direct_inputs=*/1,
        /*scalar_arg_count=*/4);
    return make_opencl_source_artifact(
        std::move(manifest), source_id("const_f32"),
        {GfxOpenClSourceScalarArg::ElementCount,
         GfxOpenClSourceScalarArg::OpCode, GfxOpenClSourceScalarArg::InputMode,
         GfxOpenClSourceScalarArg::ScalarConstantF32},
        {0}, *op, GfxOpenClArtifactInputMode::RhsScalarConstant, *rhs_constant);
  }
  if (is_f32 && lhs_constant && rhs_matches_output) {
    auto manifest = make_opencl_source_manifest(
        GfxKernelStageFamily::Eltwise,
        "opencl:generated:eltwise:" + type + ":f32:lhs_scalar_const",
        "gfx_opencl_generated_eltwise_const_f32",
        /*direct_inputs=*/1,
        /*scalar_arg_count=*/4);
    return make_opencl_source_artifact(
        std::move(manifest), source_id("const_f32"),
        {GfxOpenClSourceScalarArg::ElementCount,
         GfxOpenClSourceScalarArg::OpCode, GfxOpenClSourceScalarArg::InputMode,
         GfxOpenClSourceScalarArg::ScalarConstantF32},
        {1}, *op, GfxOpenClArtifactInputMode::LhsScalarConstant, *lhs_constant);
  }
  if (is_static_scalar_like_input(node, 1) && lhs_matches_output) {
    const std::string entry_point =
        "gfx_opencl_generated_eltwise_scalar_" + type_suffix;
    auto manifest = make_opencl_source_manifest(
        GfxKernelStageFamily::Eltwise,
        "opencl:generated:eltwise:" + type + ":" + type_suffix + ":rhs_scalar",
        entry_point,
        /*direct_inputs=*/2,
        /*scalar_arg_count=*/3);
    return make_opencl_source_artifact(
        std::move(manifest), source_id("scalar_" + type_suffix),
        {GfxOpenClSourceScalarArg::ElementCount,
         GfxOpenClSourceScalarArg::OpCode, GfxOpenClSourceScalarArg::InputMode},
        {0, 1}, *op, GfxOpenClArtifactInputMode::RhsScalar);
  }
  if (is_static_scalar_like_input(node, 0) && rhs_matches_output) {
    const std::string entry_point =
        "gfx_opencl_generated_eltwise_scalar_" + type_suffix;
    auto manifest = make_opencl_source_manifest(
        GfxKernelStageFamily::Eltwise,
        "opencl:generated:eltwise:" + type + ":" + type_suffix + ":lhs_scalar",
        entry_point,
        /*direct_inputs=*/2,
        /*scalar_arg_count=*/3);
    return make_opencl_source_artifact(
        std::move(manifest), source_id("scalar_" + type_suffix),
        {GfxOpenClSourceScalarArg::ElementCount,
         GfxOpenClSourceScalarArg::OpCode, GfxOpenClSourceScalarArg::InputMode},
        {0, 1}, *op, GfxOpenClArtifactInputMode::LhsScalar);
  }

  auto static_u32_scalars = binary_broadcast_static_u32_scalars(node);
  if (static_u32_scalars) {
    const std::string entry_point =
        "gfx_opencl_generated_eltwise_broadcast_" + type_suffix;
    std::vector<GfxOpenClSourceScalarArg> scalar_args = {
        GfxOpenClSourceScalarArg::ElementCount,
        GfxOpenClSourceScalarArg::OpCode};
    scalar_args.insert(scalar_args.end(), static_u32_scalars->size(),
                       GfxOpenClSourceScalarArg::StaticU32);
    auto manifest = make_opencl_source_manifest(
        GfxKernelStageFamily::Eltwise,
        "opencl:generated:eltwise:" + type + ":" + type_suffix + ":broadcast",
        entry_point,
        /*direct_inputs=*/2, static_cast<uint32_t>(scalar_args.size()));
    return make_opencl_source_artifact(
        std::move(manifest), source_id("broadcast_" + type_suffix),
        std::move(scalar_args), {0, 1}, *op, GfxOpenClArtifactInputMode::Direct,
        0.0f, std::move(*static_u32_scalars));
  }
  return std::nullopt;
}

std::optional<std::vector<uint32_t>> compare_broadcast_static_u32_scalars(
    const std::shared_ptr<const ov::Node> &node) {
  return broadcast_static_u32_scalars(
      node, {ov::element::f32, ov::element::f32}, ov::element::boolean);
}

std::optional<std::vector<uint32_t>> select_broadcast_static_u32_scalars(
    const std::shared_ptr<const ov::Node> &node) {
  return broadcast_static_u32_scalars(
      node, {ov::element::boolean, ov::element::f32, ov::element::f32},
      ov::element::f32);
}

std::optional<std::vector<uint32_t>>
logical_binary_broadcast_static_u32_scalars(
    const std::shared_ptr<const ov::Node> &node) {
  return broadcast_static_u32_scalars(
      node, {ov::element::boolean, ov::element::boolean}, ov::element::boolean);
}

std::optional<std::vector<uint32_t>>
reduce_static_u32_scalars(const std::shared_ptr<const ov::Node> &node,
                          bool keep_dims) {
  const auto &input_shape = node->get_input_shape(0);
  const auto &output_shape = node->get_output_shape(0);
  const size_t rank = input_shape.size();
  const size_t out_rank = output_shape.size();
  if (rank == 0 || rank > 4 || out_rank > 4 ||
      ov::shape_size(input_shape) == 0 || ov::shape_size(output_shape) == 0) {
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
      out_axis_to_input_axis.push_back(
          reduce_axes[axis] ? 4u : static_cast<uint32_t>(axis));
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
  scalars.insert(scalars.end(), out_axis_to_input_axis.begin(),
                 out_axis_to_input_axis.end());
  return scalars;
}

std::optional<bool>
reduce_logical_keep_dims(const std::shared_ptr<const ov::Node> &node) {
  if (!node || node->get_input_size() != 2 || node->get_output_size() != 1 ||
      !is_bool_tensor_type(node->get_input_element_type(0)) ||
      !is_bool_tensor_type(node->get_output_element_type(0)) ||
      !node->get_input_partial_shape(0).is_static() ||
      !node->get_output_partial_shape(0).is_static()) {
    return std::nullopt;
  }
  if (auto reduce = ov::as_type_ptr<const ov::op::v1::ReduceLogicalAnd>(node)) {
    return reduce->reduction_axes_constant()
               ? std::optional<bool>{reduce->get_keep_dims()}
               : std::nullopt;
  }
  if (auto reduce = ov::as_type_ptr<const ov::op::v1::ReduceLogicalOr>(node)) {
    return reduce->reduction_axes_constant()
               ? std::optional<bool>{reduce->get_keep_dims()}
               : std::nullopt;
  }
  return std::nullopt;
}

std::optional<bool>
reduce_numeric_keep_dims(const std::shared_ptr<const ov::Node> &node) {
  if (!node || node->get_input_size() != 2 || node->get_output_size() != 1 ||
      !is_f32_tensor_type(node->get_input_element_type(0)) ||
      !is_f32_tensor_type(node->get_output_element_type(0)) ||
      !node->get_input_partial_shape(0).is_static() ||
      !node->get_output_partial_shape(0).is_static()) {
    return std::nullopt;
  }
  if (auto reduce = ov::as_type_ptr<const ov::op::v1::ReduceSum>(node)) {
    return reduce->reduction_axes_constant()
               ? std::optional<bool>{reduce->get_keep_dims()}
               : std::nullopt;
  }
  if (auto reduce = ov::as_type_ptr<const ov::op::v1::ReduceMean>(node)) {
    return reduce->reduction_axes_constant()
               ? std::optional<bool>{reduce->get_keep_dims()}
               : std::nullopt;
  }
  if (auto reduce = ov::as_type_ptr<const ov::op::v1::ReduceMax>(node)) {
    return reduce->reduction_axes_constant()
               ? std::optional<bool>{reduce->get_keep_dims()}
               : std::nullopt;
  }
  if (auto reduce = ov::as_type_ptr<const ov::op::v1::ReduceMin>(node)) {
    return reduce->reduction_axes_constant()
               ? std::optional<bool>{reduce->get_keep_dims()}
               : std::nullopt;
  }
  if (auto reduce = ov::as_type_ptr<const ov::op::v1::ReduceProd>(node)) {
    return reduce->reduction_axes_constant()
               ? std::optional<bool>{reduce->get_keep_dims()}
               : std::nullopt;
  }
  if (auto reduce = ov::as_type_ptr<const ov::op::v4::ReduceL1>(node)) {
    return reduce->reduction_axes_constant()
               ? std::optional<bool>{reduce->get_keep_dims()}
               : std::nullopt;
  }
  if (auto reduce = ov::as_type_ptr<const ov::op::v4::ReduceL2>(node)) {
    return reduce->reduction_axes_constant()
               ? std::optional<bool>{reduce->get_keep_dims()}
               : std::nullopt;
  }
  return std::nullopt;
}

std::optional<std::vector<uint32_t>>
reduce_logical_static_u32_scalars(const std::shared_ptr<const ov::Node> &node) {
  const auto keep_dims = reduce_logical_keep_dims(node);
  if (!keep_dims) {
    return std::nullopt;
  }
  return reduce_static_u32_scalars(node, *keep_dims);
}

std::optional<std::vector<uint32_t>>
reduce_numeric_static_u32_scalars(const std::shared_ptr<const ov::Node> &node) {
  const auto keep_dims = reduce_numeric_keep_dims(node);
  if (!keep_dims) {
    return std::nullopt;
  }
  return reduce_static_u32_scalars(node, *keep_dims);
}

std::optional<std::vector<uint32_t>>
transpose_static_u32_scalars(const std::shared_ptr<const ov::Node> &node) {
  if (!node || node->get_input_size() != 2 ||
      !node->get_input_partial_shape(0).is_static() ||
      !node->get_output_partial_shape(0).is_static()) {
    return std::nullopt;
  }
  const auto &input_shape = node->get_input_shape(0);
  const auto &output_shape = node->get_output_shape(0);
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

std::optional<std::vector<uint32_t>>
slice_static_u32_scalars(const std::shared_ptr<const ov::Node> &node) {
  auto slice = ov::as_type_ptr<const ov::op::v8::Slice>(node);
  if (!slice || slice->get_input_size() < 4 || slice->get_input_size() > 5 ||
      !slice->get_input_partial_shape(0).is_static() ||
      !slice->get_output_partial_shape(0).is_static() ||
      !is_f32_tensor_type(slice->get_input_element_type(0)) ||
      !is_f32_tensor_type(slice->get_output_element_type(0))) {
    return std::nullopt;
  }

  const auto &input_shape = slice->get_input_shape(0);
  const auto &output_shape = slice->get_output_shape(0);
  const size_t rank = input_shape.size();
  if (rank == 0 || rank > 4 || output_shape.size() != rank ||
      ov::shape_size(output_shape) == 0) {
    return std::nullopt;
  }

  const auto starts = constant_i64_vector_input(node, 1);
  const auto ends = constant_i64_vector_input(node, 2);
  const auto steps = constant_i64_vector_input(node, 3);
  if (!starts || !ends || !steps || starts->size() != ends->size() ||
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
    const int64_t raw_start =
        (*starts)[i] < 0 ? (*starts)[i] + dim : (*starts)[i];
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

bool mask_has_non_zero_past_rank(const std::vector<int64_t> &mask,
                                 size_t rank) {
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

std::optional<std::vector<uint32_t>>
strided_slice_static_u32_scalars(const std::shared_ptr<const ov::Node> &node) {
  auto slice = ov::as_type_ptr<const ov::op::v1::StridedSlice>(node);
  if (!slice || slice->get_input_size() < 3 || slice->get_input_size() > 4 ||
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

  const auto &input_shape = slice->get_input_shape(0);
  const auto &output_shape = slice->get_output_shape(0);
  const size_t rank = input_shape.size();
  if (rank == 0 || rank > 4 || output_shape.size() != rank ||
      ov::shape_size(output_shape) == 0 ||
      mask_has_non_zero_past_rank(slice->get_begin_mask(), rank) ||
      mask_has_non_zero_past_rank(slice->get_end_mask(), rank)) {
    return std::nullopt;
  }

  const auto begin_values = constant_i64_vector_input(node, 1);
  const auto end_values = constant_i64_vector_input(node, 2);
  if (!begin_values || !end_values || begin_values->size() > rank ||
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

  const auto &begin_mask = slice->get_begin_mask();
  const auto &end_mask = slice->get_end_mask();
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
    raw_start =
        masked_begin ? 0 : (raw_start < 0 ? raw_start + dim : raw_start);
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

std::optional<std::vector<uint32_t>>
gather_static_u32_scalars(const std::shared_ptr<const ov::Node> &node) {
  auto gather = ov::as_type_ptr<const ov::op::util::GatherBase>(node);
  if (!gather || gather->get_input_size() != 3 ||
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

  const auto &data_shape = gather->get_input_shape(0);
  const auto &indices_shape = gather->get_input_shape(1);
  const auto &output_shape = gather->get_output_shape(0);
  if (data_shape.empty() || ov::shape_size(indices_shape) == 0 ||
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
  expected_output.insert(expected_output.end(), data_shape.begin(),
                         data_shape.begin() + *axis);
  expected_output.insert(expected_output.end(), indices_shape.begin(),
                         indices_shape.end());
  expected_output.insert(expected_output.end(), data_shape.begin() + *axis + 1,
                         data_shape.end());
  if (expected_output != output_shape) {
    return std::nullopt;
  }

  uint32_t outer = 0;
  uint32_t inner = 0;
  uint32_t axis_dim = 0;
  uint32_t indices_count = 0;
  if (!checked_u32(shape_product_range(data_shape, 0, *axis), outer) ||
      !checked_u32(
          shape_product_range(data_shape, *axis + 1, data_shape.size()),
          inner) ||
      !checked_u32(data_shape[*axis], axis_dim) ||
      !checked_u32(ov::shape_size(indices_shape), indices_count)) {
    return std::nullopt;
  }
  return std::vector<uint32_t>{outer, inner, axis_dim, indices_count};
}

bool append_shape_u32(const ov::Shape &shape, size_t max_rank,
                      std::vector<uint32_t> &values) {
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

bool append_strides_u32(const ov::Shape &shape, size_t max_rank,
                        std::vector<uint32_t> &values) {
  if (shape.size() > max_rank) {
    return false;
  }
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    uint32_t stride = 0;
    if (!checked_u32(shape_product_range(shape, axis + 1, shape.size()),
                     stride)) {
      return false;
    }
    values.push_back(stride);
  }
  values.insert(values.end(), max_rank - shape.size(), 1u);
  return true;
}

std::optional<ov::Shape> matmul_broadcast_batch_prefix(const ov::Shape &lhs,
                                                       const ov::Shape &rhs) {
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

std::optional<std::vector<uint32_t>>
matmul_static_u32_scalars(const std::shared_ptr<const ov::Node> &node) {
  auto matmul = ov::as_type_ptr<const ov::op::v0::MatMul>(node);
  if (!matmul || matmul->get_input_size() != 2 ||
      matmul->get_output_size() != 1 ||
      !matmul->get_input_partial_shape(0).is_static() ||
      !matmul->get_input_partial_shape(1).is_static() ||
      !matmul->get_output_partial_shape(0).is_static() ||
      !is_f32_tensor_type(matmul->get_input_element_type(0)) ||
      !is_f32_tensor_type(matmul->get_input_element_type(1)) ||
      !is_f32_tensor_type(matmul->get_output_element_type(0))) {
    return std::nullopt;
  }

  const auto &lhs_raw = matmul->get_input_shape(0);
  const auto &rhs_raw = matmul->get_input_shape(1);
  const auto &output_shape = matmul->get_output_shape(0);
  if (lhs_raw.size() < 2 || lhs_raw.size() > 4 || rhs_raw.size() < 2 ||
      rhs_raw.size() > 4 || ov::shape_size(output_shape) == 0) {
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

  const auto batch_prefix =
      matmul_broadcast_batch_prefix(lhs_logical, rhs_logical);
  if (!batch_prefix) {
    return std::nullopt;
  }
  ov::Shape expected_output = *batch_prefix;
  expected_output.push_back(m);
  expected_output.push_back(n);
  if (expected_output != output_shape) {
    return std::nullopt;
  }

  const uint64_t output_batch =
      shape_product_range(*batch_prefix, 0, batch_prefix->size());
  const uint64_t lhs_batch = shape_product_range(lhs_logical, 0, lhs_rank - 2);
  const uint64_t rhs_batch = shape_product_range(rhs_logical, 0, rhs_rank - 2);
  if ((lhs_batch != 1 && lhs_batch != output_batch) ||
      (rhs_batch != 1 && rhs_batch != output_batch)) {
    return std::nullopt;
  }

  const uint64_t lhs_matrix_elements =
      static_cast<uint64_t>(m) * static_cast<uint64_t>(k);
  const uint64_t rhs_matrix_elements =
      static_cast<uint64_t>(k) * static_cast<uint64_t>(n);
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
  if (!checked_u32(m, m_u32) || !checked_u32(n, n_u32) ||
      !checked_u32(k, k_u32) ||
      !checked_u32(lhs_batch == 1 ? 0 : lhs_matrix_elements,
                   lhs_batch_stride_u32) ||
      !checked_u32(rhs_batch == 1 ? 0 : rhs_matrix_elements,
                   rhs_batch_stride_u32) ||
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

std::optional<std::vector<uint32_t>> gather_elements_static_u32_scalars(
    const std::shared_ptr<const ov::Node> &node) {
  auto gather = ov::as_type_ptr<const ov::op::v6::GatherElements>(node);
  if (!gather || gather->get_input_size() != 2 ||
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

  const auto &data_shape = gather->get_input_shape(0);
  const auto &indices_shape = gather->get_input_shape(1);
  const auto &output_shape = gather->get_output_shape(0);
  const size_t rank = output_shape.size();
  if (rank == 0 || rank > 4 || data_shape.size() != rank ||
      indices_shape != output_shape || ov::shape_size(output_shape) == 0) {
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

std::optional<std::vector<uint32_t>>
gather_nd_static_u32_scalars(const std::shared_ptr<const ov::Node> &node) {
  auto gather = ov::as_type_ptr<const ov::op::util::GatherNDBase>(node);
  if (!gather || gather->get_input_size() != 2 ||
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

  const auto &data_shape = gather->get_input_shape(0);
  const auto &indices_shape = gather->get_input_shape(1);
  const auto &output_shape = gather->get_output_shape(0);
  const size_t data_rank = data_shape.size();
  const size_t indices_rank = indices_shape.size();
  if (data_rank == 0 || data_rank > 4 || indices_rank == 0 ||
      indices_shape.back() == 0 || indices_shape.back() > data_rank ||
      indices_shape.back() > 4 || ov::shape_size(output_shape) == 0) {
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
  expected_output.insert(expected_output.end(), indices_shape.begin(),
                         indices_shape.end() - 1);
  expected_output.insert(expected_output.end(),
                         data_shape.begin() + index_depth, data_shape.end());
  if (expected_output != output_shape) {
    return std::nullopt;
  }

  uint32_t total = 0;
  uint32_t slice_size = 0;
  if (!checked_u32(ov::shape_size(output_shape), total) ||
      !checked_u32(shape_product_range(data_shape, index_depth, data_rank),
                   slice_size) ||
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

std::optional<std::vector<uint32_t>>
scatter_update_static_u32_scalars(const std::shared_ptr<const ov::Node> &node) {
  auto scatter = ov::as_type_ptr<const ov::op::v3::ScatterUpdate>(node);
  if (!scatter || scatter->get_input_size() != 4 ||
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

  const auto &data_shape = scatter->get_input_shape(0);
  const auto &indices_shape = scatter->get_input_shape(1);
  const auto &updates_shape = scatter->get_input_shape(2);
  const auto &output_shape = scatter->get_output_shape(0);
  const size_t data_rank = data_shape.size();
  const size_t indices_rank = indices_shape.size();
  if (data_rank == 0 || data_rank > 4 || indices_rank > 4 ||
      data_shape != output_shape || ov::shape_size(output_shape) == 0 ||
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
  expected_updates.insert(expected_updates.end(), data_shape.begin(),
                          data_shape.begin() + *axis);
  expected_updates.insert(expected_updates.end(), indices_shape.begin(),
                          indices_shape.end());
  expected_updates.insert(expected_updates.end(),
                          data_shape.begin() + *axis + 1, data_shape.end());
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
    const std::shared_ptr<const ov::Node> &node) {
  if (auto scatter_v12 =
          ov::as_type_ptr<const ov::op::v12::ScatterElementsUpdate>(node)) {
    return scatter_v12->get_reduction() ==
           ov::op::v12::ScatterElementsUpdate::Reduction::NONE;
  }
  return ov::as_type_ptr<const ov::op::v3::ScatterElementsUpdate>(node) !=
         nullptr;
}

std::optional<std::vector<uint32_t>> scatter_elements_static_u32_scalars(
    const std::shared_ptr<const ov::Node> &node) {
  auto scatter =
      ov::as_type_ptr<const ov::op::util::ScatterElementsUpdateBase>(node);
  if (!scatter || !scatter_elements_has_baseline_reduction(node) ||
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

  const auto &data_shape = scatter->get_input_shape(0);
  const auto &indices_shape = scatter->get_input_shape(1);
  const auto &updates_shape = scatter->get_input_shape(2);
  const auto &output_shape = scatter->get_output_shape(0);
  const size_t rank = data_shape.size();
  if (rank == 0 || rank > 4 || data_shape != output_shape ||
      indices_shape != updates_shape || indices_shape.size() != rank ||
      ov::shape_size(output_shape) == 0 || ov::shape_size(updates_shape) == 0) {
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
    const std::shared_ptr<const ov::Node> &node) {
  if (auto scatter_v15 =
          ov::as_type_ptr<const ov::op::v15::ScatterNDUpdate>(node)) {
    return scatter_v15->get_reduction() ==
           ov::op::v15::ScatterNDUpdate::Reduction::NONE;
  }
  return ov::as_type_ptr<const ov::op::v3::ScatterNDUpdate>(node) != nullptr;
}

std::optional<std::vector<uint32_t>>
scatter_nd_static_u32_scalars(const std::shared_ptr<const ov::Node> &node) {
  auto scatter = ov::as_type_ptr<const ov::op::util::ScatterNDBase>(node);
  if (!scatter || !scatter_nd_has_baseline_reduction(node) ||
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

  const auto &data_shape = scatter->get_input_shape(0);
  const auto &indices_shape = scatter->get_input_shape(1);
  const auto &updates_shape = scatter->get_input_shape(2);
  const auto &output_shape = scatter->get_output_shape(0);
  const size_t data_rank = data_shape.size();
  const size_t indices_rank = indices_shape.size();
  if (data_rank == 0 || data_rank > 4 || data_shape != output_shape ||
      indices_rank == 0 || indices_shape.back() == 0 ||
      indices_shape.back() > data_rank || indices_shape.back() > 4 ||
      ov::shape_size(output_shape) == 0 || ov::shape_size(updates_shape) == 0) {
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
  expected_updates.insert(expected_updates.end(), indices_shape.begin(),
                          indices_shape.end() - 1);
  expected_updates.insert(expected_updates.end(),
                          data_shape.begin() + index_depth, data_shape.end());
  if (expected_updates != updates_shape) {
    return std::nullopt;
  }

  uint32_t slice_size = 0;
  uint32_t tuple_count = 0;
  if (!checked_u32(shape_product_range(data_shape, index_depth, data_rank),
                   slice_size) ||
      !checked_u32(shape_product_range(indices_shape, 0, indices_rank - 1),
                   tuple_count) ||
      slice_size == 0 || tuple_count == 0) {
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

std::optional<size_t> static_rank(const ov::PartialShape &shape) {
  if (shape.rank().is_dynamic()) {
    return std::nullopt;
  }
  return static_cast<size_t>(shape.rank().get_length());
}

bool all_zero_size_t(const std::vector<size_t> &values) {
  return std::all_of(values.begin(), values.end(),
                     [](size_t value) { return value == 0; });
}

bool axes_are_spatial_nchw(std::vector<int64_t> axes) {
  if (axes.size() != 2) {
    return false;
  }
  for (auto &axis : axes) {
    if (axis < 0) {
      axis += 4;
    }
  }
  std::sort(axes.begin(), axes.end());
  return axes == std::vector<int64_t>{2, 3};
}

bool axes_are_spatial_nchw(const ov::AxisSet &axes) {
  std::vector<int64_t> values;
  values.reserve(axes.size());
  for (const auto axis : axes) {
    values.push_back(static_cast<int64_t>(axis));
  }
  return axes_are_spatial_nchw(std::move(values));
}

bool constant_axes_input_is_spatial_nchw_or_absent(
    const std::shared_ptr<const ov::Node> &node) {
  if (!node || node->get_input_size() < 4) {
    return true;
  }
  const auto axes = constant_i64_vector_input(node, 3);
  return axes && axes_are_spatial_nchw(*axes);
}

bool static_nchw_spatial_resize_shape(
    const std::shared_ptr<const ov::Node> &node) {
  if (!node || !node->get_input_partial_shape(0).is_static() ||
      !node->get_output_partial_shape(0).is_static()) {
    return false;
  }
  const auto input_shape = node->get_input_shape(0);
  const auto output_shape = node->get_output_shape(0);
  return input_shape.size() == 4 && output_shape.size() == 4 &&
         input_shape[0] == output_shape[0] &&
         input_shape[1] == output_shape[1] && input_shape[2] != 0 &&
         input_shape[3] != 0 && output_shape[2] != 0 && output_shape[3] != 0;
}

struct OpenClInterpolateArtifactMetadata {
  std::string type_suffix;
  uint32_t nearest = 0;
  uint32_t align_corners = 0;
  uint32_t use_half_pixel = 1;
  uint32_t nearest_mode = 0;
};

std::optional<OpenClInterpolateArtifactMetadata>
opencl_interpolate_artifact_metadata(
    const std::shared_ptr<const ov::Node> &node) {
  if (!static_nchw_spatial_resize_shape(node) || node->get_input_size() < 1 ||
      node->get_output_element_type(0) != node->get_input_element_type(0) ||
      (!is_f32_tensor_type(node->get_output_element_type(0)) &&
       !is_f16_tensor_type(node->get_output_element_type(0)))) {
    return std::nullopt;
  }

  OpenClInterpolateArtifactMetadata metadata{};
  metadata.type_suffix =
      opencl_scalar_type_suffix(node->get_output_element_type(0));

  if (auto interp = ov::as_type_ptr<const ov::op::v0::Interpolate>(node)) {
    const auto mode = ov::util::to_lower(interp->get_attrs().mode);
    if (mode == "nearest") {
      metadata.nearest = 1;
    } else if (mode == "linear") {
      metadata.nearest = 0;
    } else {
      return std::nullopt;
    }
    if (interp->get_attrs().antialias ||
        !all_zero_size_t(interp->get_attrs().pads_begin) ||
        !all_zero_size_t(interp->get_attrs().pads_end) ||
        !axes_are_spatial_nchw(interp->get_attrs().axes)) {
      return std::nullopt;
    }
    metadata.align_corners = interp->get_attrs().align_corners ? 1u : 0u;
    metadata.use_half_pixel = metadata.align_corners == 0u ? 1u : 0u;
    metadata.nearest_mode = 0u;
    return metadata;
  }

  const auto configure_base_attrs =
      [&metadata](
          const ov::op::util::InterpolateBase::InterpolateAttrs &attrs) {
        using Base = ov::op::util::InterpolateBase;
        if (attrs.antialias || !all_zero_size_t(attrs.pads_begin) ||
            !all_zero_size_t(attrs.pads_end)) {
          return false;
        }
        switch (attrs.mode) {
        case Base::InterpolateMode::NEAREST:
          metadata.nearest = 1u;
          break;
        case Base::InterpolateMode::LINEAR:
        case Base::InterpolateMode::LINEAR_ONNX:
        case Base::InterpolateMode::BILINEAR_PILLOW:
          metadata.nearest = 0u;
          break;
        default:
          return false;
        }
        switch (attrs.coordinate_transformation_mode) {
        case Base::CoordinateTransformMode::HALF_PIXEL:
          metadata.align_corners = 0u;
          metadata.use_half_pixel = 1u;
          break;
        case Base::CoordinateTransformMode::ALIGN_CORNERS:
          metadata.align_corners = 1u;
          metadata.use_half_pixel = 1u;
          break;
        case Base::CoordinateTransformMode::ASYMMETRIC:
          metadata.align_corners = 0u;
          metadata.use_half_pixel = 0u;
          break;
        default:
          return false;
        }
        switch (attrs.nearest_mode) {
        case Base::NearestMode::FLOOR:
        case Base::NearestMode::ROUND_PREFER_FLOOR:
          metadata.nearest_mode = 1u;
          break;
        case Base::NearestMode::CEIL:
        case Base::NearestMode::ROUND_PREFER_CEIL:
          metadata.nearest_mode = 2u;
          break;
        case Base::NearestMode::SIMPLE:
        default:
          metadata.nearest_mode = 0u;
          break;
        }
        return true;
      };

  if (auto interp = ov::as_type_ptr<const ov::op::v4::Interpolate>(node)) {
    if (!constant_axes_input_is_spatial_nchw_or_absent(node) ||
        !configure_base_attrs(interp->get_attrs())) {
      return std::nullopt;
    }
    return metadata;
  }

  if (auto interp = ov::as_type_ptr<const ov::op::v11::Interpolate>(node)) {
    if (!constant_axes_input_is_spatial_nchw_or_absent(node) ||
        !configure_base_attrs(interp->get_attrs())) {
      return std::nullopt;
    }
    return metadata;
  }

  return std::nullopt;
}

bool partial_dim_compatible(const ov::Dimension &lhs,
                            const ov::Dimension &rhs) {
  return lhs.is_dynamic() || rhs.is_dynamic() ||
         lhs.get_length() == rhs.get_length();
}

std::optional<uint32_t> static_partial_dim_u32(const ov::PartialShape &shape,
                                               size_t axis) {
  if (axis >= shape.size() || shape[axis].is_dynamic()) {
    return std::nullopt;
  }
  const auto dim = shape[axis].get_length();
  if (dim <= 0 ||
      dim > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    return std::nullopt;
  }
  return static_cast<uint32_t>(dim);
}

std::optional<uint32_t>
static_partial_product_u32(const ov::PartialShape &shape, size_t begin,
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

bool partial_same_rank_shapes_compatible(
    const std::shared_ptr<const ov::Node> &node,
    const std::vector<size_t> &inputs, size_t output_idx) {
  if (!node || output_idx >= node->get_output_size()) {
    return false;
  }
  const auto output_rank =
      static_rank(node->get_output_partial_shape(output_idx));
  if (!output_rank || *output_rank == 0 || *output_rank > 4) {
    return false;
  }
  const auto &output_shape = node->get_output_partial_shape(output_idx);
  for (const size_t input_idx : inputs) {
    if (input_idx >= node->get_input_size()) {
      return false;
    }
    const auto input_rank =
        static_rank(node->get_input_partial_shape(input_idx));
    if (!input_rank || *input_rank != *output_rank) {
      return false;
    }
    const auto &input_shape = node->get_input_partial_shape(input_idx);
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
  std::string type_suffix;
  std::vector<uint32_t> values;
};

constexpr uint32_t kOpenClMaxStaticConcatInputs = 30;

std::optional<ConcatStaticU32Scalars>
concat_static_u32_scalars(const std::shared_ptr<const ov::Node> &node) {
  auto concat = ov::as_type_ptr<const ov::op::v0::Concat>(node);
  if (!concat || !concat->get_output_partial_shape(0).is_static()) {
    return std::nullopt;
  }
  const size_t input_count = concat->get_input_size();
  if (input_count < 2 || input_count > kOpenClMaxStaticConcatInputs) {
    return std::nullopt;
  }
  const bool is_f32 = is_f32_tensor_type(concat->get_output_element_type(0));
  const bool is_f16 = is_f16_tensor_type(concat->get_output_element_type(0));
  if (!is_f32 && !is_f16) {
    return std::nullopt;
  }
  for (size_t input_idx = 0; input_idx < input_count; ++input_idx) {
    if (!concat->get_input_partial_shape(input_idx).is_static() ||
        concat->get_input_element_type(input_idx) !=
            concat->get_output_element_type(0)) {
      return std::nullopt;
    }
  }
  const auto &output_shape = concat->get_output_shape(0);
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
  if (inner == 0 || inner > std::numeric_limits<uint32_t>::max() ||
      output_shape[*axis] == 0 ||
      output_shape[*axis] > std::numeric_limits<uint32_t>::max()) {
    return std::nullopt;
  }

  size_t axis_total = 0;
  std::vector<uint32_t> axis_lengths;
  axis_lengths.reserve(input_count);
  for (size_t input_idx = 0; input_idx < input_count; ++input_idx) {
    const auto &input_shape = concat->get_input_shape(input_idx);
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
  metadata.type_suffix = is_f16 ? "f16" : "f32";
  metadata.values.reserve(2 + input_count * 2);
  metadata.values.push_back(static_cast<uint32_t>(output_shape[*axis]));
  metadata.values.push_back(static_cast<uint32_t>(inner));
  uint32_t offset = 0;
  for (uint32_t len : axis_lengths) {
    if (is_f16 && (((static_cast<uint64_t>(offset) * inner) % 2u) != 0u ||
                   ((static_cast<uint64_t>(len) * inner) % 2u) != 0u)) {
      return std::nullopt;
    }
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

std::optional<ConcatDynamicScalars>
concat_dynamic_f16_scalars(const std::shared_ptr<const ov::Node> &node) {
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
      concat->get_output_partial_shape(0), *axis + 1, *rank);
  if (!inner || *inner == 0) {
    return std::nullopt;
  }
  const auto &output_shape = concat->get_output_partial_shape(0);
  for (size_t input_idx = 0; input_idx < input_count; ++input_idx) {
    if (!is_f16_tensor_type(concat->get_input_element_type(input_idx))) {
      return std::nullopt;
    }
    const auto input_rank =
        static_rank(concat->get_input_partial_shape(input_idx));
    if (!input_rank || *input_rank != *rank) {
      return std::nullopt;
    }
    const auto &input_shape = concat->get_input_partial_shape(input_idx);
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

bool select_dynamic_f16_supported(const std::shared_ptr<const ov::Node> &node) {
  return node && node->get_type_name() == std::string("Select") &&
         node->get_input_size() == 3 && node->get_output_size() == 1 &&
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

std::optional<BroadcastDynamicScalars>
broadcast_dynamic_f16_scalars(const std::shared_ptr<const ov::Node> &node) {
  auto broadcast = ov::as_type_ptr<const ov::op::v3::Broadcast>(node);
  if (!broadcast || broadcast->get_input_size() != 2 ||
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
  if (!target_shape.is_static() || target_shape.size() != 1 ||
      target_shape[0].is_dynamic() || target_shape[0].get_length() <= 0 ||
      target_shape[0].get_length() > 4) {
    return std::nullopt;
  }
  const auto output_rank = static_rank(broadcast->get_output_partial_shape(0));
  const auto input_rank = static_rank(broadcast->get_input_partial_shape(0));
  if (!output_rank || !input_rank ||
      *output_rank != static_cast<size_t>(target_shape[0].get_length()) ||
      *input_rank > *output_rank || *output_rank == 0 || *output_rank > 4) {
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

std::optional<SliceDynamicScalars>
slice_v8_dynamic_f16_scalars(const std::shared_ptr<const ov::Node> &node) {
  auto slice = ov::as_type_ptr<const ov::op::v8::Slice>(node);
  if (!slice || slice->get_input_size() < 4 || slice->get_input_size() > 5 ||
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
    if (!shape.is_static() || shape.size() != 1 || shape[0].is_dynamic() ||
        static_cast<size_t>(shape[0].get_length()) != *input_rank) {
      return std::nullopt;
    }
  }

  SliceDynamicScalars metadata;
  metadata.rank = static_cast<uint32_t>(*input_rank);
  return metadata;
}

std::optional<SliceDynamicScalars>
strided_slice_runtime_f16_scalars(const std::shared_ptr<const ov::Node> &node) {
  auto slice = ov::as_type_ptr<const ov::op::v1::StridedSlice>(node);
  if (!slice || slice->get_input_size() != 4 || slice->get_output_size() != 1 ||
      slice->get_output_partial_shape(0).is_static() ||
      !is_f16_tensor_type(slice->get_input_element_type(0)) ||
      !is_f16_tensor_type(slice->get_output_element_type(0)) ||
      slice->get_input_element_type(1) != ov::element::i64 ||
      slice->get_input_element_type(2) != ov::element::i64 ||
      slice->get_input_element_type(3) != ov::element::i64) {
    return std::nullopt;
  }
  for (const int64_t value : slice->get_begin_mask()) {
    if (value != 0) {
      return std::nullopt;
    }
  }
  for (const int64_t value : slice->get_end_mask()) {
    if (value != 0) {
      return std::nullopt;
    }
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
  if (constant_i64_vector_input(node, 1) &&
      constant_i64_vector_input(node, 3)) {
    return std::nullopt;
  }
  const auto input_rank = static_rank(slice->get_input_partial_shape(0));
  const auto output_rank = static_rank(slice->get_output_partial_shape(0));
  if (!input_rank || !output_rank || *input_rank != *output_rank ||
      *input_rank == 0 || *input_rank > 4) {
    return std::nullopt;
  }
  for (size_t input_idx = 1; input_idx <= 3; ++input_idx) {
    const auto shape = slice->get_input_partial_shape(input_idx);
    if (!shape.is_static() || shape.size() != 1 || shape[0].is_dynamic() ||
        static_cast<size_t>(shape[0].get_length()) != *input_rank) {
      return std::nullopt;
    }
  }

  SliceDynamicScalars metadata;
  metadata.rank = static_cast<uint32_t>(*input_rank);
  return metadata;
}

std::optional<SliceDynamicScalars>
strided_slice_dynamic_f16_scalars(const std::shared_ptr<const ov::Node> &node) {
  auto slice = ov::as_type_ptr<const ov::op::v1::StridedSlice>(node);
  if (!slice || slice->get_input_size() < 3 || slice->get_input_size() > 4 ||
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
  if (!end_shape.is_static() || end_shape.size() != 1 ||
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
    const int64_t begin =
        axis < begin_values->size() ? (*begin_values)[axis] : 0;
    if (begin < 0 ||
        begin > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
        strides[axis] <= 0 ||
        strides[axis] >
            static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
      return std::nullopt;
    }
    metadata.begin[axis] = static_cast<uint32_t>(begin);
    metadata.steps[axis] = static_cast<uint32_t>(strides[axis]);
  }
  return metadata;
}

struct SplitStaticU32Scalars {
  uint32_t output_count = 0;
  std::string type_suffix;
  std::vector<uint32_t> values;
};

constexpr uint32_t kOpenClMaxStaticSplitOutputs = 30;

std::optional<SplitStaticU32Scalars> split_static_u32_scalars_from_outputs(
    const std::shared_ptr<const ov::Node> &node, size_t data_input_idx,
    size_t axis_input_idx, size_t output_count, bool require_equal_axis_lengths,
    const std::vector<int64_t> *explicit_axis_lengths) {
  if (!node || data_input_idx >= node->get_input_size() ||
      axis_input_idx >= node->get_input_size() ||
      !node->get_input_partial_shape(data_input_idx).is_static()) {
    return std::nullopt;
  }
  if (output_count < 1 || output_count > kOpenClMaxStaticSplitOutputs ||
      node->get_output_size() != output_count) {
    return std::nullopt;
  }
  if (explicit_axis_lengths && explicit_axis_lengths->size() != output_count) {
    return std::nullopt;
  }
  const auto data_type = node->get_input_element_type(data_input_idx);
  if (!is_f32_tensor_type(data_type) && !is_f16_tensor_type(data_type)) {
    return std::nullopt;
  }
  for (size_t output_idx = 0; output_idx < output_count; ++output_idx) {
    if (!node->get_output_partial_shape(output_idx).is_static() ||
        node->get_output_element_type(output_idx) != data_type) {
      return std::nullopt;
    }
  }

  const auto axis_i64 = constant_i64_vector_input(node, axis_input_idx);
  if (!axis_i64 || axis_i64->size() != 1) {
    return std::nullopt;
  }
  const auto &input_shape = node->get_input_shape(data_input_idx);
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
    const auto &output_shape = node->get_output_shape(output_idx);
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
  metadata.type_suffix = opencl_scalar_type_suffix(data_type);
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

std::optional<SplitStaticU32Scalars>
split_static_u32_scalars(const std::shared_ptr<const ov::Node> &node) {
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
      /*axis_input_idx=*/1, output_count,
      /*require_equal_axis_lengths=*/true,
      /*explicit_axis_lengths=*/nullptr);
}

std::optional<SplitStaticU32Scalars>
variadic_split_static_u32_scalars(const std::shared_ptr<const ov::Node> &node) {
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
        axis_length >
            static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
      return std::nullopt;
    }
  }
  return split_static_u32_scalars_from_outputs(
      node,
      /*data_input_idx=*/0,
      /*axis_input_idx=*/1, output_count,
      /*require_equal_axis_lengths=*/false, &*explicit_axis_lengths);
}

GfxKernelStageManifest make_opencl_source_manifest(
    GfxKernelStageFamily family, std::string specialization_key,
    std::string entry_point, uint32_t direct_inputs, uint32_t scalar_arg_count,
    uint32_t direct_outputs) {
  GfxKernelExternalBufferAbiSpec abi{};
  abi.valid = true;
  abi.roles.insert(abi.roles.end(), direct_inputs,
                   GfxKernelBufferRole::TensorInput);
  abi.roles.insert(abi.roles.end(), direct_outputs,
                   GfxKernelBufferRole::TensorOutput);
  abi.roles.insert(abi.roles.end(), scalar_arg_count,
                   GfxKernelBufferRole::ScalarParam);
  GfxKernelFamily kernel_family = GfxKernelFamily::EltwiseFusedBuffer;
  if (family == GfxKernelStageFamily::Reduction) {
    kernel_family = GfxKernelFamily::ReductionBuffer;
  } else if (family == GfxKernelStageFamily::Softmax) {
    kernel_family = GfxKernelFamily::SoftmaxBuffer;
  } else if (family == GfxKernelStageFamily::Pooling) {
    kernel_family = GfxKernelFamily::Pool2DWindow;
  } else if (family == GfxKernelStageFamily::Gemm) {
    kernel_family = GfxKernelFamily::MatMulBuffer;
  } else if (family == GfxKernelStageFamily::Resize ||
             family == GfxKernelStageFamily::Layout ||
             family == GfxKernelStageFamily::Transpose) {
    kernel_family = GfxKernelFamily::TransposePackND;
  } else if (family == GfxKernelStageFamily::ConcatSplit) {
    kernel_family = GfxKernelFamily::ConcatSplitGeneric;
  } else if (family == GfxKernelStageFamily::GatherScatter) {
    kernel_family = GfxKernelFamily::GatherScatterIndexed;
  }
  auto dispatch = make_gfx_kernel_linear_dispatch_policy(
      /*threads_per_threadgroup=*/64,
      /*precompiled_binary_required=*/false);
  auto custom = make_gfx_custom_kernel_manifest(
      gfx_kernel_family_name(kernel_family),
      gfx_kernel_family_abi_id(kernel_family), std::move(entry_point),
      std::move(abi), dispatch);
  return make_gfx_custom_kernel_stage_manifest(
      family, GfxKernelBackendDomain::OpenCl, GfxKernelStorageKind::Buffer,
      std::move(specialization_key), std::move(custom));
}

std::optional<uint32_t>
split_output_count_from_entry_point(std::string_view entry_point,
                                    std::string_view type_suffix) {
  constexpr std::string_view generated_prefix = "gfx_opencl_generated_split";
  constexpr std::string_view baseline_prefix = "gfx_opencl_baseline_split";
  std::string_view prefix;
  if (entry_point.size() > generated_prefix.size() &&
      entry_point.substr(0, generated_prefix.size()) == generated_prefix) {
    prefix = generated_prefix;
  } else if (entry_point.size() > baseline_prefix.size() &&
             entry_point.substr(0, baseline_prefix.size()) == baseline_prefix) {
    prefix = baseline_prefix;
  } else {
    return std::nullopt;
  }
  if (entry_point.size() <= prefix.size() + type_suffix.size() + 1 ||
      entry_point.substr(entry_point.size() - type_suffix.size()) !=
          type_suffix ||
      entry_point[entry_point.size() - type_suffix.size() - 1] != '_') {
    return std::nullopt;
  }
  const auto digits =
      entry_point.substr(prefix.size(), entry_point.size() - prefix.size() -
                                            type_suffix.size() - 1);
  uint32_t output_count = 0;
  for (const char ch : digits) {
    if (ch < '0' || ch > '9') {
      return std::nullopt;
    }
    output_count = output_count * 10u + static_cast<uint32_t>(ch - '0');
  }
  if (output_count < 1 || output_count > kOpenClMaxStaticSplitOutputs) {
    return std::nullopt;
  }
  return output_count;
}

std::optional<uint32_t>
concat_input_count_from_entry_point(std::string_view entry_point,
                                    std::string_view type_suffix) {
  constexpr std::string_view generated_prefix = "gfx_opencl_generated_concat";
  constexpr std::string_view baseline_prefix = "gfx_opencl_baseline_concat";
  std::string_view prefix;
  if (entry_point.size() > generated_prefix.size() &&
      entry_point.substr(0, generated_prefix.size()) == generated_prefix) {
    prefix = generated_prefix;
  } else if (entry_point.size() > baseline_prefix.size() &&
             entry_point.substr(0, baseline_prefix.size()) == baseline_prefix) {
    prefix = baseline_prefix;
  } else {
    return std::nullopt;
  }
  if (entry_point.size() <= prefix.size() + type_suffix.size() + 1 ||
      entry_point.substr(entry_point.size() - type_suffix.size()) !=
          type_suffix ||
      entry_point[entry_point.size() - type_suffix.size() - 1] != '_') {
    return std::nullopt;
  }
  const auto digits =
      entry_point.substr(prefix.size(), entry_point.size() - prefix.size() -
                                            type_suffix.size() - 1);
  uint32_t input_count = 0;
  for (const char ch : digits) {
    if (ch < '0' || ch > '9') {
      return std::nullopt;
    }
    input_count = input_count * 10u + static_cast<uint32_t>(ch - '0');
  }
  if (input_count < 1 || input_count > kOpenClMaxStaticConcatInputs) {
    return std::nullopt;
  }
  return input_count;
}

std::string make_opencl_static_concat_f32_source(
    uint32_t input_count, std::string_view entry_point,
    const std::vector<uint32_t> &static_u32_scalars) {
  if (static_u32_scalars.size() != 2 + input_count * 2) {
    return {};
  }
  std::vector<uint32_t> local_axis_offsets;
  local_axis_offsets.reserve(input_count);
  uint32_t copy_axis_total = 0;
  for (uint32_t input_idx = 0; input_idx < input_count; ++input_idx) {
    const size_t value_idx = 2 + static_cast<size_t>(input_idx) * 2;
    local_axis_offsets.push_back(copy_axis_total);
    copy_axis_total += static_u32_scalars[value_idx + 1];
  }
  std::ostringstream cl;
  cl << "__kernel void " << entry_point << "(";
  for (uint32_t input_idx = 0; input_idx < input_count; ++input_idx) {
    if (input_idx != 0) {
      cl << ",\n                                             ";
    }
    cl << "__global const float* src" << input_idx;
  }
  cl << ",\n                                             __global float* dst,"
     << "\n                                             uint count) {\n";
  cl << "    const uint axis_total = " << static_u32_scalars[0] << "u;\n";
  cl << "    const uint inner = " << static_u32_scalars[1] << "u;\n";
  cl << "    const uint copy_axis_total = " << copy_axis_total << "u;\n";
  cl << R"CLC(
    const uint gid = (uint)get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint inner_idx = gid % inner;
    const uint chunk_axis_idx = (gid / inner) % copy_axis_total;
    const uint outer_idx = gid / (copy_axis_total * inner);
)CLC";
  for (uint32_t input_idx = 0; input_idx < input_count; ++input_idx) {
    const size_t value_idx = 2 + static_cast<size_t>(input_idx) * 2;
    const uint32_t axis_offset = static_u32_scalars[value_idx];
    const uint32_t axis_len = static_u32_scalars[value_idx + 1];
    const uint32_t local_axis_offset = local_axis_offsets[input_idx];
    cl << (input_idx == 0 ? "    if" : " else if")
       << " (chunk_axis_idx >= " << local_axis_offset
       << "u && chunk_axis_idx < " << (local_axis_offset + axis_len) << "u) {\n"
       << "        const uint src_axis_idx = chunk_axis_idx - "
       << local_axis_offset << "u;\n"
       << "        const uint dst_axis_idx = " << axis_offset
       << "u + src_axis_idx;\n"
       << "        const uint src_idx = (outer_idx * " << axis_len
       << "u + src_axis_idx) * inner + inner_idx;\n"
       << "        const uint dst_idx = (outer_idx * axis_total + "
          "dst_axis_idx) * inner + inner_idx;\n"
       << "        dst[dst_idx] = src" << input_idx << "[src_idx];\n"
       << "        return;\n"
       << "    }";
  }
  cl << "\n    dst[gid] = 0.0f;\n}\n";
  return cl.str();
}

std::string make_opencl_static_concat_f16_source(
    uint32_t input_count, std::string_view entry_point,
    const std::vector<uint32_t> &static_u32_scalars) {
  if (static_u32_scalars.size() != 2 + input_count * 2) {
    return {};
  }
  std::vector<uint32_t> local_axis_offsets;
  local_axis_offsets.reserve(input_count);
  uint32_t copy_axis_total = 0;
  for (uint32_t input_idx = 0; input_idx < input_count; ++input_idx) {
    const size_t value_idx = 2 + static_cast<size_t>(input_idx) * 2;
    local_axis_offsets.push_back(copy_axis_total);
    copy_axis_total += static_u32_scalars[value_idx + 1];
  }
  std::ostringstream cl;
  cl << R"CLC(
#define GFX_LOAD_F16_BITS(src, idx) \
    (((idx) & 1u) == 0u ? ((src)[(idx) >> 1u] & 65535u) : (((src)[(idx) >> 1u] >> 16u) & 65535u))
#define GFX_STORE_F16_PAIR(dst, word_idx, lo, hi) \
    ((dst)[(word_idx)] = ((lo) & 65535u) | (((hi) & 65535u) << 16u))

)CLC";
  cl << "__kernel void " << entry_point << "(";
  for (uint32_t input_idx = 0; input_idx < input_count; ++input_idx) {
    if (input_idx != 0) {
      cl << ",\n                                             ";
    }
    cl << "__global const uint* src" << input_idx;
  }
  cl << ",\n                                             __global uint* dst,"
     << "\n                                             uint count) {\n";
  cl << "    const uint axis_total = " << static_u32_scalars[0] << "u;\n";
  cl << "    const uint inner = " << static_u32_scalars[1] << "u;\n";
  cl << "    const uint copy_axis_total = " << copy_axis_total << "u;\n";
  cl << R"CLC(
    const uint word_idx = (uint)get_global_id(0);
    const uint elem0 = word_idx * 2u;
    if (elem0 >= count) {
        return;
    }
    const uint inner_idx0 = elem0 % inner;
    const uint chunk_axis_idx0 = (elem0 / inner) % copy_axis_total;
    const uint outer_idx0 = elem0 / (copy_axis_total * inner);
)CLC";
  for (uint32_t input_idx = 0; input_idx < input_count; ++input_idx) {
    const size_t value_idx = 2 + static_cast<size_t>(input_idx) * 2;
    const uint32_t axis_offset = static_u32_scalars[value_idx];
    const uint32_t axis_len = static_u32_scalars[value_idx + 1];
    const uint32_t local_axis_offset = local_axis_offsets[input_idx];
    cl << (input_idx == 0 ? "    if" : " else if")
       << " (chunk_axis_idx0 >= " << local_axis_offset
       << "u && chunk_axis_idx0 < " << (local_axis_offset + axis_len)
       << "u) {\n"
       << "        const uint src_axis_idx = chunk_axis_idx0 - "
       << local_axis_offset << "u;\n"
       << "        const uint dst_axis_idx = " << axis_offset
       << "u + src_axis_idx;\n"
       << "        const uint src_idx = (outer_idx0 * " << axis_len
       << "u + src_axis_idx) * inner + inner_idx0;\n"
       << "        const uint dst_elem = (outer_idx0 * axis_total + "
          "dst_axis_idx) * inner + inner_idx0;\n"
       << "        const uint lo = GFX_LOAD_F16_BITS(src" << input_idx
       << ", src_idx);\n"
       << "        const uint hi = elem0 + 1u < count ? GFX_LOAD_F16_BITS(src"
       << input_idx << ", src_idx + 1u) : 0u;\n"
       << "        GFX_STORE_F16_PAIR(dst, dst_elem >> 1u, lo, hi);\n"
       << "        return;\n"
       << "    }";
  }
  cl << R"CLC(
}
)CLC";
  return cl.str();
}

std::string
make_opencl_dynamic_concat_f16_source(std::string_view entry_point) {
  std::string source = kOpenClDynamicDataMovementF16Source;
  const auto input_count =
      concat_input_count_from_entry_point(entry_point, "f16");
  if (!input_count) {
    return source;
  }
  const std::string baseline_name =
      "gfx_opencl_generated_concat" + std::to_string(*input_count) + "_f16";
  const auto pos = source.find(baseline_name);
  if (pos == std::string::npos) {
    return {};
  }
  source.replace(pos, baseline_name.size(), std::string(entry_point));
  return source;
}

std::string make_opencl_static_split_f32_source(
    uint32_t output_count, std::string_view entry_point,
    const std::vector<uint32_t> &static_u32_scalars) {
  if (static_u32_scalars.size() != 2 + output_count * 2) {
    return {};
  }
  std::ostringstream cl;
  cl << R"CLC(
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

)CLC";
  cl << "__kernel void " << entry_point << "(__global const float* src";
  for (uint32_t output_idx = 0; output_idx < output_count; ++output_idx) {
    cl << ",\n                                             __global float* dst"
       << output_idx;
  }
  cl << ",\n                                             uint count) {\n";
  cl << "    const uint axis_total = " << static_u32_scalars[0] << "u;\n";
  cl << "    const uint inner = " << static_u32_scalars[1] << "u;\n";
  cl << R"CLC(
    const uint gid = (uint)get_global_id(0);
    if (gid >= count) {
        return;
    }
    const uint inner_idx = gid % inner;
    const uint axis_idx = (gid / inner) % axis_total;
    const uint outer_idx = gid / (axis_total * inner);
)CLC";
  for (uint32_t output_idx = 0; output_idx < output_count; ++output_idx) {
    const size_t value_idx = 2 + static_cast<size_t>(output_idx) * 2;
    cl << "    gfx_split_store_f32(src, dst" << output_idx
       << ", gid, axis_idx, outer_idx, inner_idx, inner, "
       << static_u32_scalars[value_idx] << "u, "
       << static_u32_scalars[value_idx + 1] << "u);\n";
  }
  cl << "}\n";
  return cl.str();
}

std::string make_opencl_static_split_f16_source(
    uint32_t output_count, std::string_view entry_point,
    const std::vector<uint32_t> &static_u32_scalars) {
  if (static_u32_scalars.size() != 2 + output_count * 2) {
    return {};
  }
  std::ostringstream cl;
  cl << R"CLC(
#define GFX_LOAD_F16_BITS(src, idx) \
    (((idx) & 1u) == 0u ? ((src)[(idx) >> 1u] & 65535u) : (((src)[(idx) >> 1u] >> 16u) & 65535u))
#define GFX_STORE_F16_PAIR(dst, word_idx, lo, hi) \
    ((dst)[(word_idx)] = ((lo) & 65535u) | (((hi) & 65535u) << 16u))

static inline void gfx_split_store_word_f16(__global const uint* src,
                                            __global uint* dst,
                                            uint word_idx,
                                            uint count,
                                            uint axis_total,
                                            uint inner,
                                            uint axis_offset,
                                            uint axis_len) {
    if (axis_total == 0u || inner == 0u || axis_len == 0u) {
        return;
    }
    const uint outer_times_inner = count / axis_total;
    const uint out_total = outer_times_inner * axis_len;
    const uint elem0 = word_idx * 2u;
    if (elem0 >= out_total) {
        return;
    }
    uint out_elem = elem0;
    uint inner_idx = out_elem % inner;
    uint axis_idx = (out_elem / inner) % axis_len;
    uint outer_idx = out_elem / (axis_len * inner);
    uint src_idx = (outer_idx * axis_total + axis_offset + axis_idx) * inner + inner_idx;
    const uint lo = GFX_LOAD_F16_BITS(src, src_idx);
    uint hi = 0u;
    if (elem0 + 1u < out_total) {
        out_elem = elem0 + 1u;
        inner_idx = out_elem % inner;
        axis_idx = (out_elem / inner) % axis_len;
        outer_idx = out_elem / (axis_len * inner);
        src_idx = (outer_idx * axis_total + axis_offset + axis_idx) * inner + inner_idx;
        hi = GFX_LOAD_F16_BITS(src, src_idx);
    }
    GFX_STORE_F16_PAIR(dst, word_idx, lo, hi);
}

)CLC";
  cl << "__kernel void " << entry_point << "(__global const uint* src";
  for (uint32_t output_idx = 0; output_idx < output_count; ++output_idx) {
    cl << ",\n                                             __global uint* dst"
       << output_idx;
  }
  cl << ",\n                                             uint count) {\n";
  cl << "    const uint axis_total = " << static_u32_scalars[0] << "u;\n";
  cl << "    const uint inner = " << static_u32_scalars[1] << "u;\n";
  cl << "    const uint word_idx = (uint)get_global_id(0);\n";
  for (uint32_t output_idx = 0; output_idx < output_count; ++output_idx) {
    const size_t value_idx = 2 + static_cast<size_t>(output_idx) * 2;
    cl << "    gfx_split_store_word_f16(src, dst" << output_idx
       << ", word_idx, count, axis_total, inner, "
       << static_u32_scalars[value_idx] << "u, "
       << static_u32_scalars[value_idx + 1] << "u);\n";
  }
  cl << "}\n";
  return cl.str();
}

GfxOpenClSourceArtifact make_opencl_source_artifact(
    GfxKernelStageManifest manifest, std::string source_id,
    std::vector<GfxOpenClSourceScalarArg> scalar_args,
    std::vector<size_t> direct_input_indices, GfxOpenClArtifactOp op,
    GfxOpenClArtifactInputMode input_mode, float scalar_constant_f32,
    std::vector<uint32_t> static_u32_scalars,
    GfxOpenClSourceElementCountSource element_count_source,
    std::vector<float> static_f32_scalars) {
  GfxOpenClSourceArtifact artifact{};
  artifact.valid = manifest.valid;
  artifact.stage_manifest = std::move(manifest);
  artifact.artifact_ref = make_gfx_kernel_artifact_ref(artifact.stage_manifest);
  artifact.artifact_ref.source_id = std::move(source_id);
  artifact.artifact_ref.entry_point =
      artifact.stage_manifest.custom_kernel.entry_point;
  bool source_inlines_static_u32_scalars = false;
  if (auto concat_inputs = concat_input_count_from_entry_point(
          artifact.artifact_ref.entry_point, "f32");
      concat_inputs && static_u32_scalars.size() ==
                           2 + static_cast<size_t>(*concat_inputs) * 2) {
    artifact.source = make_opencl_static_concat_f32_source(
        *concat_inputs, artifact.artifact_ref.entry_point, static_u32_scalars);
    source_inlines_static_u32_scalars = true;
  } else if (auto concat_inputs = concat_input_count_from_entry_point(
                 artifact.artifact_ref.entry_point, "f16");
             concat_inputs && static_u32_scalars.size() ==
                                  2 + static_cast<size_t>(*concat_inputs) * 2) {
    artifact.source = make_opencl_static_concat_f16_source(
        *concat_inputs, artifact.artifact_ref.entry_point, static_u32_scalars);
    source_inlines_static_u32_scalars = true;
  } else if (auto split_outputs = split_output_count_from_entry_point(
                 artifact.artifact_ref.entry_point, "f32")) {
    artifact.source = make_opencl_static_split_f32_source(
        *split_outputs, artifact.artifact_ref.entry_point, static_u32_scalars);
    source_inlines_static_u32_scalars = true;
  } else if (auto split_outputs = split_output_count_from_entry_point(
                 artifact.artifact_ref.entry_point, "f16")) {
    artifact.source = make_opencl_static_split_f16_source(
        *split_outputs, artifact.artifact_ref.entry_point, static_u32_scalars);
    source_inlines_static_u32_scalars = true;
  } else if (artifact.artifact_ref.entry_point ==
             "gfx_opencl_generated_eltwise_logical_unary_bool") {
    artifact.source =
        opencl_generated_eltwise_logical_unary_bool_kernel_source().source;
  } else if (artifact.artifact_ref.entry_point ==
             "gfx_opencl_generated_eltwise_logical_binary_bool") {
    artifact.source =
        opencl_generated_eltwise_logical_binary_bool_kernel_source().source;
  } else if (artifact.artifact_ref.entry_point ==
             "gfx_opencl_generated_eltwise_logical_binary_broadcast_bool") {
    artifact.source =
        opencl_generated_eltwise_logical_binary_broadcast_bool_kernel_source()
            .source;
  } else if (artifact.artifact_ref.entry_point ==
             "gfx_opencl_generated_eltwise_compare_f32") {
    artifact.source =
        opencl_generated_eltwise_compare_f32_kernel_source().source;
  } else if (artifact.artifact_ref.entry_point ==
             "gfx_opencl_generated_eltwise_compare_broadcast_f32") {
    artifact.source =
        opencl_generated_eltwise_compare_broadcast_f32_kernel_source().source;
  } else if (artifact.artifact_ref.entry_point ==
             "gfx_opencl_generated_eltwise_select_f32") {
    artifact.source =
        opencl_generated_eltwise_select_f32_kernel_source().source;
  } else if (artifact.artifact_ref.entry_point ==
             "gfx_opencl_generated_eltwise_select_broadcast_f32") {
    artifact.source =
        opencl_generated_eltwise_select_broadcast_f32_kernel_source().source;
  } else if (artifact.artifact_ref.entry_point ==
             "gfx_opencl_generated_eltwise_select_f16_dynamic") {
    artifact.source =
        opencl_generated_eltwise_select_f16_dynamic_kernel_source().source;
  } else if (std::string_view(artifact.artifact_ref.entry_point)
                 .substr(0, std::string_view("gfx_opencl_generated_eltwise_")
                                .size()) ==
             std::string_view("gfx_opencl_generated_eltwise_")) {
    artifact.source = opencl_generated_eltwise_kernel_source().source;
  } else if (std::string_view(artifact.artifact_ref.entry_point)
                 .substr(0, std::string_view("gfx_opencl_generated_activation_")
                                .size()) ==
             std::string_view("gfx_opencl_generated_activation_")) {
    artifact.source = opencl_generated_activation_kernel_source().source;
  } else if (std::string_view(artifact.artifact_ref.entry_point)
                 .substr(
                     0,
                     std::string_view("gfx_opencl_baseline_convert_").size()) ==
             std::string_view("gfx_opencl_baseline_convert_")) {
    artifact.source = kOpenClConvertSource;
  } else if (artifact.artifact_ref.entry_point ==
             "gfx_opencl_generated_matmul_f32") {
    artifact.source = opencl_generated_matmul_f32_kernel_source().source;
  } else if (artifact.artifact_ref.entry_point ==
             "gfx_opencl_generated_softmax_f32") {
    artifact.source = opencl_generated_softmax_f32_kernel_source().source;
  } else if (artifact.artifact_ref.entry_point ==
             "gfx_opencl_generated_softmax_dynamic_f32") {
    artifact.source =
        opencl_generated_softmax_f32_dynamic_kernel_source().source;
  } else if (artifact.artifact_ref.entry_point ==
             "gfx_opencl_generated_softmax_f16") {
    artifact.source = opencl_generated_softmax_f16_kernel_source().source;
  } else if (artifact.artifact_ref.entry_point ==
             "gfx_opencl_generated_softmax_dynamic_f16") {
    artifact.source =
        opencl_generated_softmax_f16_dynamic_kernel_source().source;
  } else if (artifact.artifact_ref.entry_point ==
             "gfx_opencl_generated_interpolate_f32") {
    artifact.source = opencl_generated_interpolate_f32_kernel_source().source;
  } else if (artifact.artifact_ref.entry_point ==
             "gfx_opencl_generated_interpolate_f16") {
    artifact.source = opencl_generated_interpolate_f16_kernel_source().source;
  } else if (artifact.artifact_ref.entry_point ==
             "gfx_opencl_generated_tile_f32") {
    artifact.source = opencl_generated_tile_f32_kernel_source().source;
  } else if (artifact.artifact_ref.entry_point ==
             "gfx_opencl_generated_tile_dynamic_f32") {
    artifact.source = opencl_generated_tile_dynamic_f32_kernel_source().source;
  } else if (artifact.artifact_ref.entry_point ==
             "gfx_opencl_generated_tile_f16") {
    artifact.source = opencl_generated_tile_f16_kernel_source().source;
  } else if (artifact.artifact_ref.entry_point ==
             "gfx_opencl_generated_tile_dynamic_f16") {
    artifact.source = opencl_generated_tile_dynamic_f16_kernel_source().source;
  } else if (artifact.artifact_ref.entry_point ==
             "gfx_opencl_baseline_unary_f32") {
    artifact.source = kOpenClUnaryF32Source;
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
                 "gfx_opencl_baseline_binary_i32" ||
             artifact.artifact_ref.entry_point ==
                 "gfx_opencl_baseline_binary_scalar_i32" ||
             artifact.artifact_ref.entry_point ==
                 "gfx_opencl_baseline_binary_broadcast_i32") {
    artifact.source = kOpenClBinaryI32Source;
  } else if (artifact.artifact_ref.entry_point ==
                 "gfx_opencl_baseline_binary_f16" ||
             artifact.artifact_ref.entry_point ==
                 "gfx_opencl_baseline_binary_scalar_f16" ||
             artifact.artifact_ref.entry_point ==
                 "gfx_opencl_baseline_binary_broadcast_f16") {
    artifact.source = kOpenClBinaryF16Source;
  } else if (artifact.artifact_ref.entry_point ==
             "gfx_opencl_generated_transpose_f32") {
    artifact.source = kOpenClTransposeF32Source;
  } else if (artifact.artifact_ref.entry_point ==
             "gfx_opencl_baseline_slice_f32") {
    artifact.source = kOpenClSliceF32Source;
  } else if (artifact.artifact_ref.entry_point ==
                 "gfx_opencl_baseline_gather_i32_f32" ||
             artifact.artifact_ref.entry_point ==
                 "gfx_opencl_baseline_gather_elements_i32_f32" ||
             artifact.artifact_ref.entry_point ==
                 "gfx_opencl_baseline_gather_nd_i32_f32") {
    artifact.source = kOpenClGatherF32I32Source;
  } else if (artifact.artifact_ref.entry_point ==
                 "gfx_opencl_baseline_gather_i64_f32" ||
             artifact.artifact_ref.entry_point ==
                 "gfx_opencl_baseline_gather_elements_i64_f32" ||
             artifact.artifact_ref.entry_point ==
                 "gfx_opencl_baseline_gather_nd_i64_f32") {
    artifact.source = kOpenClGatherF32I64Source;
  } else if (artifact.artifact_ref.entry_point ==
                 "gfx_opencl_baseline_scatter_update_i32_f32" ||
             artifact.artifact_ref.entry_point ==
                 "gfx_opencl_baseline_scatter_elements_i32_f32" ||
             artifact.artifact_ref.entry_point ==
                 "gfx_opencl_baseline_scatter_nd_i32_f32") {
    artifact.source = kOpenClScatterF32I32Source;
  } else if (artifact.artifact_ref.entry_point ==
                 "gfx_opencl_baseline_scatter_update_i64_f32" ||
             artifact.artifact_ref.entry_point ==
                 "gfx_opencl_baseline_scatter_elements_i64_f32" ||
             artifact.artifact_ref.entry_point ==
                 "gfx_opencl_baseline_scatter_nd_i64_f32") {
    artifact.source = kOpenClScatterF32I64Source;
  } else if (artifact.artifact_ref.entry_point ==
                 "gfx_opencl_generated_concat2_f32" ||
             artifact.artifact_ref.entry_point ==
                 "gfx_opencl_generated_concat3_f32" ||
             artifact.artifact_ref.entry_point ==
                 "gfx_opencl_generated_concat4_f32") {
    artifact.source = kOpenClConcatSplitF32Source;
  } else if (auto concat_inputs = concat_input_count_from_entry_point(
                 artifact.artifact_ref.entry_point, "f16");
             concat_inputs && *concat_inputs >= 2 && *concat_inputs <= 4 &&
             static_u32_scalars.size() == 1) {
    artifact.source = make_opencl_dynamic_concat_f16_source(
        artifact.artifact_ref.entry_point);
  } else if (artifact.artifact_ref.entry_point ==
                 "gfx_opencl_baseline_broadcast_f16_i64shape" ||
             artifact.artifact_ref.entry_point ==
                 "gfx_opencl_baseline_slice_f16" ||
             artifact.artifact_ref.entry_point ==
                 "gfx_opencl_baseline_slice_v8_f16") {
    artifact.source = kOpenClDynamicDataMovementF16Source;
  } else if (artifact.artifact_ref.entry_point ==
             "gfx_opencl_generated_reduction_f32") {
    artifact.source = opencl_generated_reduction_f32_kernel_source().source;
  } else if (artifact.artifact_ref.entry_point ==
             "gfx_opencl_generated_reduction_bool") {
    artifact.source = opencl_generated_reduction_bool_kernel_source().source;
  }
  if (source_inlines_static_u32_scalars) {
    artifact.source_static_u32_scalars = static_u32_scalars;
    static_u32_scalars.clear();
  }
  artifact.scalar_args = std::move(scalar_args);
  artifact.static_u32_scalars = std::move(static_u32_scalars);
  artifact.static_f32_scalars = std::move(static_f32_scalars);
  artifact.direct_input_indices = std::move(direct_input_indices);
  const auto roles = materialize_gfx_kernel_external_buffer_roles(
      artifact.stage_manifest.custom_kernel.external_buffer_abi);
  artifact.arg_count = static_cast<uint32_t>(roles.size());
  artifact.direct_input_count =
      static_cast<uint32_t>(artifact.direct_input_indices.size());
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
  artifact.valid =
      artifact.valid && artifact.artifact_ref.valid &&
      artifact.artifact_ref.kind == GfxKernelArtifactKind::OpenClSource &&
      !artifact.source.empty();
  return artifact;
}

void materialize_gfx_opencl_source_chunk_plan(
    GfxOpenClSourceArtifact &artifact) {
  artifact.planned_chunks.clear();
  if (!artifact.valid) {
    return;
  }

  if (artifact.input_chunk_size != 0) {
    for (uint32_t input_begin = 0; input_begin < artifact.direct_input_count;
         input_begin += artifact.input_chunk_size) {
      const uint32_t input_count = std::min<uint32_t>(
          artifact.input_chunk_size, artifact.direct_input_count - input_begin);
      auto chunk_artifact = make_gfx_opencl_concat_chunk_source_artifact(
          artifact, input_begin, input_count);
      if (!chunk_artifact || !chunk_artifact->valid) {
        artifact.valid = false;
        artifact.planned_chunks.clear();
        return;
      }
      const auto &chunk_static_u32 = chunk_artifact->source_static_u32_scalars;
      if (chunk_static_u32.size() != 2 + static_cast<size_t>(input_count) * 2 ||
          chunk_static_u32[0] == 0) {
        artifact.valid = false;
        artifact.planned_chunks.clear();
        return;
      }
      uint32_t chunk_axis_total = 0;
      for (uint32_t local_input = 0; local_input < input_count; ++local_input) {
        const size_t axis_extent_idx =
            2 + static_cast<size_t>(local_input) * 2 + 1;
        chunk_axis_total += chunk_static_u32[axis_extent_idx];
      }
      if (chunk_axis_total == 0) {
        artifact.valid = false;
        artifact.planned_chunks.clear();
        return;
      }
      artifact.planned_chunks.push_back(
          {input_begin, input_count,
           GfxOpenClSourceChunkBindingRole::DirectInputs, chunk_axis_total,
           chunk_static_u32[0],
           std::make_shared<const GfxOpenClSourceArtifact>(
               std::move(*chunk_artifact))});
    }
    artifact.valid = !artifact.planned_chunks.empty();
    return;
  }

  if (artifact.output_chunk_size != 0) {
    for (uint32_t output_begin = 0; output_begin < artifact.direct_output_count;
         output_begin += artifact.output_chunk_size) {
      const uint32_t output_count =
          std::min<uint32_t>(artifact.output_chunk_size,
                             artifact.direct_output_count - output_begin);
      auto chunk_artifact = make_gfx_opencl_split_chunk_source_artifact(
          artifact, output_begin, output_count);
      if (!chunk_artifact || !chunk_artifact->valid) {
        artifact.valid = false;
        artifact.planned_chunks.clear();
        return;
      }
      artifact.planned_chunks.push_back(
          {output_begin, output_count,
           GfxOpenClSourceChunkBindingRole::DirectOutputs,
           /*element_count_multiplier=*/1,
           /*element_count_divisor=*/1,
           std::make_shared<const GfxOpenClSourceArtifact>(
               std::move(*chunk_artifact))});
    }
    artifact.valid = !artifact.planned_chunks.empty();
  }
}

} // namespace

GfxOpenClSourceArtifactPayload::GfxOpenClSourceArtifactPayload(
    GfxOpenClSourceArtifact artifact)
    : m_artifact(std::move(artifact)) {}

KernelArtifactPayloadKind
GfxOpenClSourceArtifactPayload::payload_kind() const noexcept {
  return KernelArtifactPayloadKind::OpenClSource;
}

std::string_view
GfxOpenClSourceArtifactPayload::backend_domain() const noexcept {
  return "opencl";
}

std::string_view GfxOpenClSourceArtifactPayload::source_id() const noexcept {
  return m_artifact.artifact_ref.source_id;
}

std::string_view GfxOpenClSourceArtifactPayload::entry_point() const noexcept {
  return m_artifact.artifact_ref.entry_point;
}

bool GfxOpenClSourceArtifactPayload::valid() const noexcept {
  if (!m_artifact.valid || !m_artifact.artifact_ref.valid ||
      m_artifact.artifact_ref.kind != GfxKernelArtifactKind::OpenClSource ||
      m_artifact.artifact_ref.backend_domain !=
          GfxKernelBackendDomain::OpenCl ||
      m_artifact.source.empty()) {
    return false;
  }
  const bool expects_chunks =
      m_artifact.input_chunk_size != 0 || m_artifact.output_chunk_size != 0;
  if (!expects_chunks) {
    return m_artifact.planned_chunks.empty();
  }
  if (m_artifact.planned_chunks.empty()) {
    return false;
  }
  for (const auto &chunk : m_artifact.planned_chunks) {
    if (chunk.binding_count == 0 || chunk.element_count_multiplier == 0 ||
        chunk.element_count_divisor == 0 || !chunk.artifact ||
        !chunk.artifact->valid || !chunk.artifact->artifact_ref.valid ||
        chunk.artifact->artifact_ref.kind !=
            GfxKernelArtifactKind::OpenClSource ||
        chunk.artifact->artifact_ref.backend_domain !=
            GfxKernelBackendDomain::OpenCl ||
        chunk.artifact->source.empty() ||
        !chunk.artifact->planned_chunks.empty()) {
      return false;
    }
    if (chunk.binding_role == GfxOpenClSourceChunkBindingRole::DirectInputs) {
      if (chunk.binding_begin + chunk.binding_count >
              m_artifact.direct_input_count ||
          chunk.artifact->direct_input_count != chunk.binding_count ||
          chunk.artifact->direct_output_count !=
              m_artifact.direct_output_count) {
        return false;
      }
      continue;
    }
    if (chunk.binding_role == GfxOpenClSourceChunkBindingRole::DirectOutputs) {
      if (chunk.binding_begin + chunk.binding_count >
              m_artifact.direct_output_count ||
          chunk.artifact->direct_input_count != m_artifact.direct_input_count ||
          chunk.artifact->direct_output_count != chunk.binding_count) {
        return false;
      }
      continue;
    }
    return false;
  }
  return true;
}

std::optional<GfxOpenClSourceArtifact> make_opencl_eltwise_family_artifact(
    const std::shared_ptr<const ov::Node> &node) {
  if (!node) {
    return std::nullopt;
  }
  const std::string type = node->get_type_name();

  if (type == "Select" && select_dynamic_f16_supported(node)) {
    auto manifest = make_opencl_source_manifest(
        GfxKernelStageFamily::Eltwise,
        "opencl:generated:eltwise:Select:bool_f16:dynamic_same_shape",
        "gfx_opencl_generated_eltwise_select_f16_dynamic",
        /*direct_inputs=*/3,
        /*scalar_arg_count=*/1);
    return make_opencl_source_artifact(
        std::move(manifest), "opencl/generated/eltwise_select_f16_dynamic",
        {GfxOpenClSourceScalarArg::ElementCount}, {0, 1, 2},
        GfxOpenClArtifactOp::Identity);
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
      auto manifest = make_opencl_source_manifest(
          GfxKernelStageFamily::Eltwise,
          "opencl:generated:eltwise:" + type + ":f32:same_shape",
          "gfx_opencl_generated_eltwise_compare_f32",
          /*direct_inputs=*/2,
          /*scalar_arg_count=*/2);
      return make_opencl_source_artifact(
          std::move(manifest), "opencl/generated/eltwise_compare_f32",
          {GfxOpenClSourceScalarArg::ElementCount,
           GfxOpenClSourceScalarArg::OpCode},
          {0, 1}, *op);
    }
    auto static_u32_scalars = compare_broadcast_static_u32_scalars(node);
    if (static_u32_scalars) {
      std::vector<GfxOpenClSourceScalarArg> scalar_args = {
          GfxOpenClSourceScalarArg::ElementCount,
          GfxOpenClSourceScalarArg::OpCode};
      scalar_args.insert(scalar_args.end(), static_u32_scalars->size(),
                         GfxOpenClSourceScalarArg::StaticU32);
      auto manifest = make_opencl_source_manifest(
          GfxKernelStageFamily::Eltwise,
          "opencl:generated:eltwise:" + type + ":f32:broadcast",
          "gfx_opencl_generated_eltwise_compare_broadcast_f32",
          /*direct_inputs=*/2, static_cast<uint32_t>(scalar_args.size()));
      return make_opencl_source_artifact(
          std::move(manifest), "opencl/generated/eltwise_compare_broadcast_f32",
          std::move(scalar_args), {0, 1}, *op,
          GfxOpenClArtifactInputMode::Direct, 0.0f,
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
      auto manifest = make_opencl_source_manifest(
          GfxKernelStageFamily::Eltwise,
          "opencl:generated:eltwise:Select:bool_f32:same_shape",
          "gfx_opencl_generated_eltwise_select_f32",
          /*direct_inputs=*/3,
          /*scalar_arg_count=*/1);
      return make_opencl_source_artifact(
          std::move(manifest), "opencl/generated/eltwise_select_f32",
          {GfxOpenClSourceScalarArg::ElementCount}, {0, 1, 2},
          GfxOpenClArtifactOp::Identity);
    }
    auto static_u32_scalars = select_broadcast_static_u32_scalars(node);
    if (static_u32_scalars) {
      std::vector<GfxOpenClSourceScalarArg> scalar_args = {
          GfxOpenClSourceScalarArg::ElementCount};
      scalar_args.insert(scalar_args.end(), static_u32_scalars->size(),
                         GfxOpenClSourceScalarArg::StaticU32);
      auto manifest = make_opencl_source_manifest(
          GfxKernelStageFamily::Eltwise,
          "opencl:generated:eltwise:Select:bool_f32:broadcast",
          "gfx_opencl_generated_eltwise_select_broadcast_f32",
          /*direct_inputs=*/3, static_cast<uint32_t>(scalar_args.size()));
      return make_opencl_source_artifact(
          std::move(manifest), "opencl/generated/eltwise_select_broadcast_f32",
          std::move(scalar_args), {0, 1, 2}, GfxOpenClArtifactOp::Identity,
          GfxOpenClArtifactInputMode::Direct, 0.0f,
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
    auto manifest = make_opencl_source_manifest(
        GfxKernelStageFamily::Eltwise,
        "opencl:generated:eltwise:" + type + ":bool:same_shape",
        "gfx_opencl_generated_eltwise_logical_unary_bool",
        /*direct_inputs=*/1,
        /*scalar_arg_count=*/2);
    return make_opencl_source_artifact(
        std::move(manifest), "opencl/generated/eltwise_logical_unary_bool",
        {GfxOpenClSourceScalarArg::ElementCount,
         GfxOpenClSourceScalarArg::OpCode},
        {0}, *op);
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
      auto manifest = make_opencl_source_manifest(
          GfxKernelStageFamily::Eltwise,
          "opencl:generated:eltwise:" + type + ":bool:same_shape",
          "gfx_opencl_generated_eltwise_logical_binary_bool",
          /*direct_inputs=*/2,
          /*scalar_arg_count=*/2);
      return make_opencl_source_artifact(
          std::move(manifest), "opencl/generated/eltwise_logical_binary_bool",
          {GfxOpenClSourceScalarArg::ElementCount,
           GfxOpenClSourceScalarArg::OpCode},
          {0, 1}, *op);
    }
    auto static_u32_scalars = logical_binary_broadcast_static_u32_scalars(node);
    if (static_u32_scalars) {
      std::vector<GfxOpenClSourceScalarArg> scalar_args = {
          GfxOpenClSourceScalarArg::ElementCount,
          GfxOpenClSourceScalarArg::OpCode};
      scalar_args.insert(scalar_args.end(), static_u32_scalars->size(),
                         GfxOpenClSourceScalarArg::StaticU32);
      auto manifest = make_opencl_source_manifest(
          GfxKernelStageFamily::Eltwise,
          "opencl:generated:eltwise:" + type + ":bool:broadcast",
          "gfx_opencl_generated_eltwise_logical_binary_broadcast_bool",
          /*direct_inputs=*/2, static_cast<uint32_t>(scalar_args.size()));
      return make_opencl_source_artifact(
          std::move(manifest),
          "opencl/generated/eltwise_logical_binary_broadcast_bool",
          std::move(scalar_args), {0, 1}, *op,
          GfxOpenClArtifactInputMode::Direct, 0.0f,
          std::move(*static_u32_scalars));
    }
    return std::nullopt;
  }

  return std::nullopt;
}

std::optional<GfxOpenClSourceArtifact> make_opencl_activation_source_artifact(
    const std::shared_ptr<const ov::Node> &node,
    std::string_view expected_source_id) {
  auto artifact = make_opencl_activation_artifact(node);
  if (!artifact || !artifact->valid) {
    return std::nullopt;
  }
  if (!expected_source_id.empty() &&
      artifact->artifact_ref.source_id != expected_source_id) {
    return std::nullopt;
  }
  return artifact;
}

std::optional<GfxOpenClSourceArtifact>
make_opencl_eltwise_source_artifact(const std::shared_ptr<const ov::Node> &node,
                                    std::string_view expected_source_id) {
  auto artifact = make_opencl_eltwise_artifact(node);
  if (!artifact || !artifact->valid) {
    artifact = make_opencl_eltwise_family_artifact(node);
  }
  if (!artifact || !artifact->valid) {
    return std::nullopt;
  }
  if (!expected_source_id.empty() &&
      artifact->artifact_ref.source_id != expected_source_id) {
    return std::nullopt;
  }
  return artifact;
}

std::optional<GfxOpenClSourceArtifact>
make_gfx_opencl_concat_chunk_source_artifact(
    const GfxOpenClSourceArtifact &base_artifact, uint32_t input_begin,
    uint32_t input_count) {
  if (!base_artifact.valid || base_artifact.input_chunk_size == 0 ||
      input_count < 1 || input_count > base_artifact.input_chunk_size ||
      input_count > 4 || base_artifact.direct_output_count != 1 ||
      base_artifact.direct_input_indices.size() !=
          base_artifact.direct_input_count) {
    return std::nullopt;
  }

  std::string type_suffix;
  auto total_inputs = concat_input_count_from_entry_point(
      base_artifact.artifact_ref.entry_point, "f32");
  if (total_inputs) {
    type_suffix = "f32";
  } else {
    total_inputs = concat_input_count_from_entry_point(
        base_artifact.artifact_ref.entry_point, "f16");
    if (!total_inputs) {
      return std::nullopt;
    }
    type_suffix = "f16";
  }
  if (input_begin >= *total_inputs ||
      input_begin + input_count > *total_inputs ||
      base_artifact.direct_input_count != *total_inputs ||
      base_artifact.source_static_u32_scalars.size() !=
          2 + static_cast<size_t>(*total_inputs) * 2) {
    return std::nullopt;
  }

  std::vector<uint32_t> chunk_static_u32_scalars = {
      base_artifact.source_static_u32_scalars[0],
      base_artifact.source_static_u32_scalars[1],
  };
  chunk_static_u32_scalars.reserve(2 + static_cast<size_t>(input_count) * 2);
  for (uint32_t local_input = 0; local_input < input_count; ++local_input) {
    const size_t source_idx =
        2 + static_cast<size_t>(input_begin + local_input) * 2;
    chunk_static_u32_scalars.push_back(
        base_artifact.source_static_u32_scalars[source_idx]);
    chunk_static_u32_scalars.push_back(
        base_artifact.source_static_u32_scalars[source_idx + 1]);
  }

  std::vector<size_t> direct_input_indices;
  direct_input_indices.reserve(input_count);
  for (uint32_t local_input = 0; local_input < input_count; ++local_input) {
    direct_input_indices.push_back(
        base_artifact.direct_input_indices[input_begin + local_input]);
  }

  const std::string entry_point = "gfx_opencl_generated_concat" +
                                  std::to_string(input_count) + "_" +
                                  type_suffix;
  auto manifest = make_opencl_source_manifest(
      base_artifact.stage_manifest.stage_family,
      base_artifact.stage_manifest.specialization_key + ":chunk" +
          std::to_string(input_begin) + "x" + std::to_string(input_count),
      entry_point, input_count,
      /*scalar_arg_count=*/1);
  return make_opencl_source_artifact(
      std::move(manifest),
      base_artifact.artifact_ref.source_id + "/chunk" +
          std::to_string(input_begin) + "x" + std::to_string(input_count),
      {GfxOpenClSourceScalarArg::ElementCount}, std::move(direct_input_indices),
      base_artifact.op, base_artifact.input_mode,
      base_artifact.scalar_constant_f32, std::move(chunk_static_u32_scalars),
      base_artifact.element_count_source);
}

std::optional<GfxOpenClSourceArtifact>
make_gfx_opencl_split_chunk_source_artifact(
    const GfxOpenClSourceArtifact &base_artifact, uint32_t output_begin,
    uint32_t output_count) {
  if (!base_artifact.valid || base_artifact.output_chunk_size == 0 ||
      output_count < 1 || output_count > base_artifact.output_chunk_size ||
      output_count > 4 || base_artifact.direct_input_count != 1 ||
      base_artifact.direct_input_indices.empty()) {
    return std::nullopt;
  }

  std::string type_suffix;
  auto total_outputs = split_output_count_from_entry_point(
      base_artifact.artifact_ref.entry_point, "f32");
  if (total_outputs) {
    type_suffix = "f32";
  } else {
    total_outputs = split_output_count_from_entry_point(
        base_artifact.artifact_ref.entry_point, "f16");
    if (!total_outputs) {
      return std::nullopt;
    }
    type_suffix = "f16";
  }
  if (output_begin >= *total_outputs ||
      output_begin + output_count > *total_outputs ||
      base_artifact.source_static_u32_scalars.size() !=
          2 + static_cast<size_t>(*total_outputs) * 2) {
    return std::nullopt;
  }

  std::vector<uint32_t> chunk_static_u32_scalars = {
      base_artifact.source_static_u32_scalars[0],
      base_artifact.source_static_u32_scalars[1],
  };
  chunk_static_u32_scalars.reserve(2 + static_cast<size_t>(output_count) * 2);
  for (uint32_t local_output = 0; local_output < output_count; ++local_output) {
    const size_t source_idx =
        2 + static_cast<size_t>(output_begin + local_output) * 2;
    chunk_static_u32_scalars.push_back(
        base_artifact.source_static_u32_scalars[source_idx]);
    chunk_static_u32_scalars.push_back(
        base_artifact.source_static_u32_scalars[source_idx + 1]);
  }

  const std::string entry_point = "gfx_opencl_generated_split" +
                                  std::to_string(output_count) + "_" +
                                  type_suffix;
  auto manifest = make_opencl_source_manifest(
      base_artifact.stage_manifest.stage_family,
      base_artifact.stage_manifest.specialization_key + ":chunk" +
          std::to_string(output_begin) + "x" + std::to_string(output_count),
      entry_point,
      /*direct_inputs=*/1,
      /*scalar_arg_count=*/1, output_count);
  return make_opencl_source_artifact(
      std::move(manifest),
      base_artifact.artifact_ref.source_id + "/chunk" +
          std::to_string(output_begin) + "x" + std::to_string(output_count),
      {GfxOpenClSourceScalarArg::ElementCount},
      base_artifact.direct_input_indices, base_artifact.op,
      base_artifact.input_mode, base_artifact.scalar_constant_f32,
      std::move(chunk_static_u32_scalars), base_artifact.element_count_source);
}

std::string gfx_opencl_source_artifact_build_options(
    const GfxOpenClSourceArtifact &artifact) {
  std::string joined;
  for (const auto &option : artifact.build_options) {
    if (!joined.empty()) {
      joined.push_back(' ');
    }
    joined += option;
  }
  return joined;
}

} // namespace gfx_plugin
} // namespace ov
