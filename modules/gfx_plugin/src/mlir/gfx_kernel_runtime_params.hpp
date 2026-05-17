// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kernel_ir/gfx_codegen_desc.hpp"
#include "kernel_ir/gfx_kernel_cache.hpp"
#include "openvino/core/axis_set.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "runtime/gfx_shape_utils.hpp"
#include "runtime/gpu_buffer_manager.hpp"
#include "runtime/gpu_tensor.hpp"

namespace ov {
namespace gfx_plugin {

struct GfxKernelRuntimeParamPayload {
    std::vector<GpuTensor> extra_inputs;
    std::vector<int32_t> scalar_args;
};

inline std::string make_kernel_param_key(std::string_view stage_name, std::string_view suffix) {
    std::string key(stage_name);
    key.push_back('/');
    key.append(suffix.data(), suffix.size());
    return key;
}

inline GpuTensor make_kernel_i32_param_tensor(GpuBufferManager& buffer_manager,
                                              std::string_view stage_name,
                                              std::string_view suffix,
                                              const std::vector<int32_t>& values) {
    GpuBuffer buf = buffer_manager.wrap_const(make_kernel_param_key(stage_name, suffix),
                                              values.data(),
                                              values.size() * sizeof(int32_t),
                                              ov::element::i32);
    OPENVINO_ASSERT(buf.valid(), "GFX MLIR: failed to wrap i32 kernel metadata for stage ", stage_name);
    buf.owned = false;
    GpuTensor tensor;
    tensor.buf = std::move(buf);
    tensor.expected_type = ov::element::i32;
    tensor.shape = ov::Shape{values.size()};
    return tensor;
}

inline GpuTensor make_kernel_u32_param_tensor(GpuBufferManager& buffer_manager,
                                              std::string_view stage_name,
                                              std::string_view suffix,
                                              const std::vector<uint32_t>& values) {
    GpuBuffer buf = buffer_manager.wrap_const(make_kernel_param_key(stage_name, suffix),
                                              values.data(),
                                              values.size() * sizeof(uint32_t),
                                              ov::element::u32);
    OPENVINO_ASSERT(buf.valid(), "GFX MLIR: failed to wrap u32 kernel metadata for stage ", stage_name);
    buf.owned = false;
    GpuTensor tensor;
    tensor.buf = std::move(buf);
    tensor.expected_type = ov::element::u32;
    tensor.shape = ov::Shape{values.size()};
    return tensor;
}

inline GpuTensor make_kernel_bytes_param_tensor(GpuBufferManager& buffer_manager,
                                                std::string_view stage_name,
                                                std::string_view suffix,
                                                const void* data,
                                                size_t bytes,
                                                const ov::element::Type& type,
                                                ov::Shape shape) {
    GpuBuffer buf = buffer_manager.wrap_const(make_kernel_param_key(stage_name, suffix),
                                              data,
                                              bytes,
                                              type);
    OPENVINO_ASSERT(buf.valid(), "GFX MLIR: failed to wrap kernel metadata for stage ", stage_name);
    buf.owned = false;
    GpuTensor tensor;
    tensor.buf = std::move(buf);
    tensor.expected_type = type;
    tensor.shape = std::move(shape);
    return tensor;
}

inline GfxKernelRuntimeParamPayload make_single_bytes_runtime_param_payload(GpuBufferManager& buffer_manager,
                                                                            std::string_view stage_name,
                                                                            std::string_view suffix,
                                                                            const void* data,
                                                                            size_t bytes,
                                                                            const ov::element::Type& type,
                                                                            ov::Shape shape) {
    GfxKernelRuntimeParamPayload payload;
    payload.extra_inputs.push_back(make_kernel_bytes_param_tensor(buffer_manager,
                                                                  stage_name,
                                                                  suffix,
                                                                  data,
                                                                  bytes,
                                                                  type,
                                                                  std::move(shape)));
    return payload;
}

inline GpuTensor make_hashed_kernel_bytes_param_tensor(GpuBufferManager& buffer_manager,
                                                       std::string_view stage_name,
                                                       std::string_view suffix_prefix,
                                                       const void* data,
                                                       size_t bytes,
                                                       const ov::element::Type& type,
                                                       ov::Shape shape) {
    OPENVINO_ASSERT(data && bytes > 0,
                    "GFX MLIR: kernel metadata payload is empty for stage ",
                    stage_name);
    std::ostringstream suffix;
    suffix << suffix_prefix
           << "/"
           << type.get_type_name()
           << "/"
           << bytes
           << "/"
           << gfx_hash_bytes(data, bytes);
    return make_kernel_bytes_param_tensor(buffer_manager,
                                          stage_name,
                                          suffix.str(),
                                          data,
                                          bytes,
                                          type,
                                          std::move(shape));
}

inline std::vector<int32_t> gfx_shape_to_i32_vector(const ov::Shape& shape) {
    std::vector<int32_t> values(std::max<size_t>(shape.size(), 1), 1);
    for (size_t i = 0; i < shape.size(); ++i) {
        values[i] = static_cast<int32_t>(shape[i]);
    }
    return values;
}

inline std::vector<int32_t> gfx_shape_strides_i32(const ov::Shape& shape) {
    std::vector<int32_t> strides(std::max<size_t>(shape.size(), 1), 1);
    if (shape.empty()) {
        return strides;
    }
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        const size_t idx = static_cast<size_t>(i);
        strides[idx] = strides[idx + 1] * static_cast<int32_t>(shape[idx + 1]);
    }
    return strides;
}

inline std::vector<uint32_t> gfx_shape_to_u32_vector(const ov::Shape& shape) {
    std::vector<uint32_t> values(std::max<size_t>(shape.size(), 1), 1);
    for (size_t i = 0; i < shape.size(); ++i) {
        values[i] = static_cast<uint32_t>(shape[i]);
    }
    return values;
}

inline std::vector<uint32_t> gfx_shape_strides_u32(const ov::Shape& shape) {
    std::vector<uint32_t> strides(std::max<size_t>(shape.size(), 1), 1);
    if (shape.empty()) {
        return strides;
    }
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        const size_t idx = static_cast<size_t>(i);
        strides[idx] = strides[idx + 1] * static_cast<uint32_t>(shape[idx + 1]);
    }
    return strides;
}

template <typename DimsLike>
inline uint32_t gfx_dim_u32_at(const DimsLike& values, size_t index, std::string_view stage_name, std::string_view name) {
    OPENVINO_ASSERT(index < values.size(),
                    "GFX MLIR: missing ",
                    name,
                    " dimension ",
                    index,
                    " for stage ",
                    stage_name);
    return static_cast<uint32_t>(values.at(index));
}

template <typename KernelLike, typename StridesLike, typename PadsLike, typename DilationsLike>
inline GfxKernelRuntimeParamPayload make_pool_runtime_param_payload(GpuBufferManager& buffer_manager,
                                                                    std::string_view stage_name,
                                                                    const ov::Shape& input_shape,
                                                                    const ov::Shape& output_shape,
                                                                    const KernelLike& kernel,
                                                                    const StridesLike& strides,
                                                                    const PadsLike& pads_begin,
                                                                    const PadsLike& pads_end,
                                                                    const DilationsLike& dilations,
                                                                    bool is_avg,
                                                                    bool exclude_pad) {
    OPENVINO_ASSERT(input_shape.size() == 4 && output_shape.size() == 4,
                    "GFX MLIR: pool expects NCHW shapes for stage ",
                    stage_name);
    struct PoolParams {
        uint32_t N = 0;
        uint32_t C = 0;
        uint32_t H = 0;
        uint32_t W = 0;
        uint32_t kH = 0;
        uint32_t kW = 0;
        uint32_t strideH = 0;
        uint32_t strideW = 0;
        uint32_t dilationH = 0;
        uint32_t dilationW = 0;
        uint32_t padTop = 0;
        uint32_t padLeft = 0;
        uint32_t padBottom = 0;
        uint32_t padRight = 0;
        uint32_t outH = 0;
        uint32_t outW = 0;
        uint32_t is_avg = 0;
        uint32_t exclude_pad = 0;
    } params{};
    params.N = static_cast<uint32_t>(input_shape[0]);
    params.C = static_cast<uint32_t>(input_shape[1]);
    params.H = static_cast<uint32_t>(input_shape[2]);
    params.W = static_cast<uint32_t>(input_shape[3]);
    params.kH = gfx_dim_u32_at(kernel, 0, stage_name, "pool kernel");
    params.kW = gfx_dim_u32_at(kernel, 1, stage_name, "pool kernel");
    params.strideH = gfx_dim_u32_at(strides, 0, stage_name, "pool stride");
    params.strideW = gfx_dim_u32_at(strides, 1, stage_name, "pool stride");
    params.dilationH = gfx_dim_u32_at(dilations, 0, stage_name, "pool dilation");
    params.dilationW = gfx_dim_u32_at(dilations, 1, stage_name, "pool dilation");
    params.padTop = gfx_dim_u32_at(pads_begin, 0, stage_name, "pool pad begin");
    params.padLeft = gfx_dim_u32_at(pads_begin, 1, stage_name, "pool pad begin");
    params.padBottom = gfx_dim_u32_at(pads_end, 0, stage_name, "pool pad end");
    params.padRight = gfx_dim_u32_at(pads_end, 1, stage_name, "pool pad end");
    params.outH = static_cast<uint32_t>(output_shape[2]);
    params.outW = static_cast<uint32_t>(output_shape[3]);
    params.is_avg = is_avg ? 1u : 0u;
    params.exclude_pad = exclude_pad ? 1u : 0u;

    return make_single_bytes_runtime_param_payload(buffer_manager,
                                                   stage_name,
                                                   "pool_params",
                                                   &params,
                                                   sizeof(params),
                                                   ov::element::u8,
                                                   ov::Shape{sizeof(params)});
}

inline GfxKernelRuntimeParamPayload make_softmax_runtime_param_payload(GpuBufferManager& buffer_manager,
                                                                       std::string_view stage_name,
                                                                       uint64_t rows,
                                                                       uint64_t axis_len,
                                                                       uint64_t inner) {
    std::vector<int32_t> params = {
        static_cast<int32_t>(rows),
        static_cast<int32_t>(axis_len),
        static_cast<int32_t>(inner),
    };
    GfxKernelRuntimeParamPayload payload;
    payload.extra_inputs.push_back(make_kernel_i32_param_tensor(buffer_manager,
                                                               stage_name,
                                                               "softmax_params",
                                                               params));
    return payload;
}

template <typename StridesLike, typename PadsLike, typename DilationsLike>
inline GfxKernelRuntimeParamPayload make_conv3d_runtime_param_payload(GpuBufferManager& buffer_manager,
                                                                      std::string_view stage_name,
                                                                      const ov::Shape& input_shape,
                                                                      const ov::Shape& output_shape,
                                                                      const ov::Shape& weight_shape,
                                                                      const StridesLike& strides,
                                                                      const DilationsLike& dilations,
                                                                      const PadsLike& pads_begin,
                                                                      const PadsLike& pads_end) {
    OPENVINO_ASSERT(input_shape.size() == 5 && output_shape.size() == 5 && weight_shape.size() == 5,
                    "GFX MLIR: Conv3D expects NCDHW/OIDHW shapes for stage ",
                    stage_name);
    struct GfxConv3DRuntimeParams {
        uint32_t N = 0;
        uint32_t C_in = 0;
        uint32_t D = 0;
        uint32_t H = 0;
        uint32_t W = 0;
        uint32_t C_out = 0;
        uint32_t kD = 0;
        uint32_t kH = 0;
        uint32_t kW = 0;
        uint32_t strideD = 0;
        uint32_t strideH = 0;
        uint32_t strideW = 0;
        uint32_t dilationD = 0;
        uint32_t dilationH = 0;
        uint32_t dilationW = 0;
        uint32_t padFront = 0;
        uint32_t padTop = 0;
        uint32_t padLeft = 0;
        uint32_t padBack = 0;
        uint32_t padBottom = 0;
        uint32_t padRight = 0;
        uint32_t outD = 0;
        uint32_t outH = 0;
        uint32_t outW = 0;
    } params{};

    params.N = static_cast<uint32_t>(input_shape[0]);
    params.C_in = static_cast<uint32_t>(input_shape[1]);
    params.D = static_cast<uint32_t>(input_shape[2]);
    params.H = static_cast<uint32_t>(input_shape[3]);
    params.W = static_cast<uint32_t>(input_shape[4]);
    params.C_out = static_cast<uint32_t>(weight_shape[0]);
    params.kD = static_cast<uint32_t>(weight_shape[2]);
    params.kH = static_cast<uint32_t>(weight_shape[3]);
    params.kW = static_cast<uint32_t>(weight_shape[4]);
    params.strideD = gfx_dim_u32_at(strides, 0, stage_name, "conv3d stride");
    params.strideH = gfx_dim_u32_at(strides, 1, stage_name, "conv3d stride");
    params.strideW = gfx_dim_u32_at(strides, 2, stage_name, "conv3d stride");
    params.dilationD = gfx_dim_u32_at(dilations, 0, stage_name, "conv3d dilation");
    params.dilationH = gfx_dim_u32_at(dilations, 1, stage_name, "conv3d dilation");
    params.dilationW = gfx_dim_u32_at(dilations, 2, stage_name, "conv3d dilation");
    params.padFront = gfx_dim_u32_at(pads_begin, 0, stage_name, "conv3d pad begin");
    params.padTop = gfx_dim_u32_at(pads_begin, 1, stage_name, "conv3d pad begin");
    params.padLeft = gfx_dim_u32_at(pads_begin, 2, stage_name, "conv3d pad begin");
    params.padBack = gfx_dim_u32_at(pads_end, 0, stage_name, "conv3d pad end");
    params.padBottom = gfx_dim_u32_at(pads_end, 1, stage_name, "conv3d pad end");
    params.padRight = gfx_dim_u32_at(pads_end, 2, stage_name, "conv3d pad end");
    params.outD = static_cast<uint32_t>(output_shape[2]);
    params.outH = static_cast<uint32_t>(output_shape[3]);
    params.outW = static_cast<uint32_t>(output_shape[4]);

    return make_single_bytes_runtime_param_payload(buffer_manager,
                                                   stage_name,
                                                   "conv3d_params",
                                                   &params,
                                                   sizeof(params),
                                                   ov::element::u8,
                                                   ov::Shape{sizeof(params)});
}

inline GpuTensor make_kernel_float_vector_param_tensor(GpuBufferManager& buffer_manager,
                                                       std::string_view stage_name,
                                                       std::string_view suffix,
                                                       const std::vector<float>& values,
                                                       ov::Shape shape,
                                                       const ov::element::Type& preferred_type) {
    ov::element::Type buffer_type = preferred_type;
    const void* data_ptr = values.data();
    size_t bytes = values.size() * sizeof(float);
    std::vector<ov::float16> values_f16;
    if (preferred_type == ov::element::f16) {
        values_f16.resize(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            values_f16[i] = ov::float16(values[i]);
        }
        data_ptr = values_f16.data();
        bytes = values_f16.size() * sizeof(ov::float16);
    } else if (!preferred_type.is_real()) {
        buffer_type = ov::element::f32;
    }

    return make_kernel_bytes_param_tensor(buffer_manager,
                                          stage_name,
                                          suffix,
                                          data_ptr,
                                          bytes,
                                          buffer_type,
                                          std::move(shape));
}

inline GpuTensor make_kernel_float_vector_param_tensor(GpuBufferManager& buffer_manager,
                                                       std::string_view stage_name,
                                                       std::string_view suffix,
                                                       const std::vector<float>& values,
                                                       size_t tensor_channels,
                                                       const ov::element::Type& preferred_type) {
    return make_kernel_float_vector_param_tensor(buffer_manager,
                                                 stage_name,
                                                 suffix,
                                                 values,
                                                 ov::Shape{tensor_channels},
                                                 preferred_type);
}

inline GpuTensor make_bias_runtime_param_tensor(GpuBufferManager& buffer_manager,
                                                std::string_view stage_name,
                                                const std::vector<float>& values,
                                                const std::vector<int64_t>& source_shape,
                                                const ov::element::Type& source_element_type,
                                                const ov::element::Type& output_element_type,
                                                size_t output_rank,
                                                bool conv_like) {
    ov::element::Type bias_type =
        output_element_type == ov::element::dynamic ? source_element_type : output_element_type;
    if (bias_type == ov::element::dynamic) {
        bias_type = ov::element::f32;
    }

    ov::Shape shape;
    if (conv_like) {
        shape = ov::Shape{values.size()};
    } else {
        std::vector<int64_t> aligned_shape(output_rank, 1);
        if (output_rank >= source_shape.size()) {
            const size_t offset = output_rank - source_shape.size();
            for (size_t i = 0; i < source_shape.size(); ++i) {
                aligned_shape[offset + i] = source_shape[i];
            }
        }
        shape.reserve(aligned_shape.size());
        for (const auto dim : aligned_shape) {
            shape.push_back(static_cast<size_t>(dim));
        }
    }

    return make_kernel_float_vector_param_tensor(buffer_manager,
                                                 stage_name,
                                                 "bias",
                                                 values,
                                                 std::move(shape),
                                                 bias_type);
}

inline GfxKernelRuntimeParamPayload make_batchnorm_scale_bias_runtime_param_payload(
    GpuBufferManager& buffer_manager,
    std::string_view stage_name,
    const std::vector<float>& gamma,
    const std::vector<float>& beta,
    const std::vector<float>& mean,
    const std::vector<float>& variance,
    float epsilon,
    const ov::element::Type& output_element_type) {
    const size_t channels = gamma.size();
    OPENVINO_ASSERT(beta.size() == channels && mean.size() == channels && variance.size() == channels,
                    "GFX MLIR: BatchNorm parameter channel mismatch for stage ",
                    stage_name);
    ov::element::Type bn_type = output_element_type == ov::element::dynamic ? ov::element::f32 : output_element_type;
    if (bn_type == ov::element::dynamic) {
        bn_type = ov::element::f32;
    }

    std::vector<float> scale_values(channels);
    std::vector<float> bias_values(channels);
    for (size_t c = 0; c < channels; ++c) {
        const float inv_std = 1.0f / std::sqrt(variance[c] + epsilon);
        scale_values[c] = gamma[c] * inv_std;
        bias_values[c] = beta[c] - mean[c] * scale_values[c];
    }

    GfxKernelRuntimeParamPayload payload;
    payload.extra_inputs.push_back(make_kernel_float_vector_param_tensor(buffer_manager,
                                                                         stage_name,
                                                                         "bn_scale",
                                                                         scale_values,
                                                                         channels,
                                                                         bn_type));
    payload.extra_inputs.push_back(make_kernel_float_vector_param_tensor(buffer_manager,
                                                                         stage_name,
                                                                         "bn_bias",
                                                                         bias_values,
                                                                         channels,
                                                                         bn_type));
    return payload;
}

inline GfxKernelRuntimeParamPayload make_conv2d_runtime_param_payload(GpuBufferManager& buffer_manager,
                                                                      std::string_view stage_name,
                                                                      const ov::Shape& input_shape,
                                                                      const ov::Shape& output_shape,
                                                                      const ov::element::Type& param_element_type,
                                                                      uint32_t c_out,
                                                                      uint32_t groups,
                                                                      uint32_t c_in_per_group,
                                                                      uint32_t c_out_per_group,
                                                                      uint32_t kernel_h,
                                                                      uint32_t kernel_w,
                                                                      uint32_t stride_h,
                                                                      uint32_t stride_w,
                                                                      uint32_t dilation_h,
                                                                      uint32_t dilation_w,
                                                                      uint32_t pad_top,
                                                                      uint32_t pad_left,
                                                                      uint32_t pad_bottom,
                                                                      uint32_t pad_right,
                                                                      bool has_bias,
                                                                      bool has_batchnorm,
                                                                      uint32_t activation,
                                                                      float activation_alpha,
                                                                      float batchnorm_epsilon,
                                                                      const std::vector<float>& bias,
                                                                      const std::vector<float>& gamma,
                                                                      const std::vector<float>& beta,
                                                                      const std::vector<float>& mean,
                                                                      const std::vector<float>& variance) {
    OPENVINO_ASSERT(input_shape.size() == 4 && output_shape.size() == 4,
                    "GFX MLIR: Conv expects NCHW shapes for stage ",
                    stage_name);
    const size_t channels = static_cast<size_t>(c_out);
    GfxKernelRuntimeParamPayload payload;
    payload.extra_inputs.push_back(make_kernel_float_vector_param_tensor(buffer_manager,
                                                                         stage_name,
                                                                         "bias",
                                                                         bias,
                                                                         channels,
                                                                         param_element_type));
    payload.extra_inputs.push_back(make_kernel_float_vector_param_tensor(buffer_manager,
                                                                         stage_name,
                                                                         "gamma",
                                                                         gamma,
                                                                         channels,
                                                                         param_element_type));
    payload.extra_inputs.push_back(make_kernel_float_vector_param_tensor(buffer_manager,
                                                                         stage_name,
                                                                         "beta",
                                                                         beta,
                                                                         channels,
                                                                         param_element_type));
    payload.extra_inputs.push_back(make_kernel_float_vector_param_tensor(buffer_manager,
                                                                         stage_name,
                                                                         "mean",
                                                                         mean,
                                                                         channels,
                                                                         param_element_type));
    payload.extra_inputs.push_back(make_kernel_float_vector_param_tensor(buffer_manager,
                                                                         stage_name,
                                                                         "var",
                                                                         variance,
                                                                         channels,
                                                                         param_element_type));

    struct GfxConv2DRuntimeParams {
        uint32_t N = 0;
        uint32_t C_in = 0;
        uint32_t H = 0;
        uint32_t W = 0;
        uint32_t C_out = 0;
        uint32_t groups = 0;
        uint32_t C_in_pg = 0;
        uint32_t C_out_pg = 0;
        uint32_t kH = 0;
        uint32_t kW = 0;
        uint32_t strideH = 0;
        uint32_t strideW = 0;
        uint32_t dilationH = 0;
        uint32_t dilationW = 0;
        uint32_t padTop = 0;
        uint32_t padLeft = 0;
        uint32_t padBottom = 0;
        uint32_t padRight = 0;
        uint32_t outH = 0;
        uint32_t outW = 0;
        uint32_t has_bias = 0;
        uint32_t has_bn = 0;
        uint32_t activation = 0;
        float alpha = 0.0f;
        float epsilon = 0.0f;
        float clamp_min = 0.0f;
        float clamp_max = 0.0f;
    } params{};

    params.N = static_cast<uint32_t>(input_shape[0]);
    params.C_in = static_cast<uint32_t>(input_shape[1]);
    params.H = static_cast<uint32_t>(input_shape[2]);
    params.W = static_cast<uint32_t>(input_shape[3]);
    params.C_out = c_out;
    params.groups = groups;
    params.C_in_pg = c_in_per_group ? c_in_per_group : params.C_in;
    params.C_out_pg = c_out_per_group ? c_out_per_group : params.C_out;
    params.kH = kernel_h;
    params.kW = kernel_w;
    params.strideH = stride_h;
    params.strideW = stride_w;
    params.dilationH = dilation_h;
    params.dilationW = dilation_w;
    params.padTop = pad_top;
    params.padLeft = pad_left;
    params.padBottom = pad_bottom;
    params.padRight = pad_right;
    params.outH = static_cast<uint32_t>(output_shape[2]);
    params.outW = static_cast<uint32_t>(output_shape[3]);
    params.has_bias = has_bias ? 1u : 0u;
    params.has_bn = has_batchnorm ? 1u : 0u;
    params.activation = activation;
    params.alpha = activation_alpha;
    params.epsilon = batchnorm_epsilon;

    payload.extra_inputs.push_back(make_kernel_bytes_param_tensor(buffer_manager,
                                                                  stage_name,
                                                                  "conv_params",
                                                                  &params,
                                                                  sizeof(params),
                                                                  ov::element::u8,
                                                                  ov::Shape{sizeof(params)}));
    return payload;
}

inline GfxKernelRuntimeParamPayload make_shapeof_runtime_param_payload(GpuBufferManager& buffer_manager,
                                                                       std::string_view stage_name,
                                                                       const ov::Shape& runtime_shape,
                                                                       const ov::element::Type& output_type) {
    OPENVINO_ASSERT(output_type == ov::element::i32 || output_type == ov::element::i64,
                    "GFX MLIR: ShapeOf output must be i32/i64");
    std::ostringstream suffix;
    suffix << "shapeof_dims/";
    GfxKernelRuntimeParamPayload payload;
    if (output_type == ov::element::i32) {
        std::vector<int32_t> dims(runtime_shape.size(), 0);
        for (size_t i = 0; i < runtime_shape.size(); ++i) {
            dims[i] = static_cast<int32_t>(runtime_shape[i]);
            suffix << dims[i] << 'x';
        }
        payload.extra_inputs.push_back(
            make_kernel_i32_param_tensor(buffer_manager, stage_name, suffix.str(), dims));
    } else {
        std::vector<int64_t> dims(runtime_shape.size(), 0);
        for (size_t i = 0; i < runtime_shape.size(); ++i) {
            dims[i] = static_cast<int64_t>(runtime_shape[i]);
            suffix << dims[i] << 'x';
        }
        payload.extra_inputs.push_back(make_kernel_bytes_param_tensor(buffer_manager,
                                                                      stage_name,
                                                                      suffix.str(),
                                                                      dims.data(),
                                                                      dims.size() * sizeof(int64_t),
                                                                      ov::element::i64,
                                                                      ov::Shape{dims.size()}));
    }
    payload.scalar_args = {static_cast<int32_t>(runtime_shape.size())};
    return payload;
}

inline GfxKernelRuntimeParamPayload make_tile_runtime_param_payload(GpuBufferManager& buffer_manager,
                                                                    std::string_view stage_name,
                                                                    const ov::Shape& input_shape,
                                                                    const ov::Shape& output_shape) {
    GfxKernelRuntimeParamPayload payload;
    payload.extra_inputs.push_back(make_kernel_i32_param_tensor(buffer_manager,
                                                               stage_name,
                                                               "tile_out_dims",
                                                               gfx_shape_to_i32_vector(output_shape)));
    payload.extra_inputs.push_back(make_kernel_i32_param_tensor(buffer_manager,
                                                               stage_name,
                                                               "tile_in_dims",
                                                               gfx_shape_to_i32_vector(input_shape)));
    payload.extra_inputs.push_back(make_kernel_i32_param_tensor(buffer_manager,
                                                               stage_name,
                                                               "tile_out_strides",
                                                               gfx_shape_strides_i32(output_shape)));
    payload.extra_inputs.push_back(make_kernel_i32_param_tensor(buffer_manager,
                                                               stage_name,
                                                               "tile_in_strides",
                                                               gfx_shape_strides_i32(input_shape)));
    payload.scalar_args = {
        static_cast<int32_t>(ov::shape_size(output_shape)),
        static_cast<int32_t>(std::max<size_t>(output_shape.size(), 1))};
    return payload;
}

inline GfxKernelRuntimeParamPayload make_gather_elements_runtime_param_payload(GpuBufferManager& buffer_manager,
                                                                               std::string_view stage_name,
                                                                               const ov::Shape& data_shape,
                                                                               const ov::Shape& output_shape,
                                                                               uint32_t axis) {
    OPENVINO_ASSERT(data_shape.size() == output_shape.size(),
                    "GFX MLIR: GatherElements data/output rank mismatch for stage ",
                    stage_name);
    OPENVINO_ASSERT(output_shape.size() <= GatherElementsCodegenDesc::kMaxDims,
                    "GFX MLIR: GatherElements rank exceeds kernel metadata capacity for stage ",
                    stage_name);
    std::vector<uint32_t> params(GatherElementsCodegenDesc::kParamU32Count, 0);
    const auto data_strides_i64 = make_element_strides(data_shape);
    const auto out_strides_i64 = make_element_strides(output_shape);
    params[GatherElementsCodegenDesc::kRankOffset] = static_cast<uint32_t>(output_shape.size());
    params[GatherElementsCodegenDesc::kAxisOffset] = axis;
    params[GatherElementsCodegenDesc::kTotalOffset] = static_cast<uint32_t>(ov::shape_size(output_shape));
    for (size_t i = 0; i < output_shape.size(); ++i) {
        params[GatherElementsCodegenDesc::kOutDimsOffset + i] = static_cast<uint32_t>(output_shape[i]);
        params[GatherElementsCodegenDesc::kOutStridesOffset + i] = static_cast<uint32_t>(out_strides_i64[i]);
        params[GatherElementsCodegenDesc::kDataDimsOffset + i] = static_cast<uint32_t>(data_shape[i]);
        params[GatherElementsCodegenDesc::kDataStridesOffset + i] = static_cast<uint32_t>(data_strides_i64[i]);
    }

    GfxKernelRuntimeParamPayload payload;
    payload.extra_inputs.push_back(make_kernel_u32_param_tensor(buffer_manager,
                                                               stage_name,
                                                               "gather_elements_params",
                                                               params));
    return payload;
}

inline GfxKernelRuntimeParamPayload make_gather_runtime_param_payload(GpuBufferManager& buffer_manager,
                                                                      std::string_view stage_name,
                                                                      const ov::Shape& data_shape,
                                                                      const ov::Shape& indices_shape,
                                                                      uint32_t axis) {
    OPENVINO_ASSERT(axis < data_shape.size(), "GFX MLIR: Gather axis out of range for stage ", stage_name);
    struct GatherParams {
        uint32_t outer = 0;
        uint32_t inner = 0;
        uint32_t axis_dim = 0;
        uint32_t indices_count = 0;
    } params{};
    params.outer = static_cast<uint32_t>(shape_product(data_shape, 0, axis));
    params.inner = static_cast<uint32_t>(shape_product(data_shape,
                                                       static_cast<size_t>(axis) + 1,
                                                       data_shape.size()));
    params.axis_dim = static_cast<uint32_t>(data_shape[axis]);
    params.indices_count = static_cast<uint32_t>(ov::shape_size(indices_shape));

    std::ostringstream suffix;
    suffix << "gather_params/"
           << params.outer << 'x'
           << params.inner << 'x'
           << params.axis_dim << 'x'
           << params.indices_count;
    GfxKernelRuntimeParamPayload payload;
    payload.extra_inputs.push_back(make_kernel_bytes_param_tensor(buffer_manager,
                                                                  stage_name,
                                                                  suffix.str(),
                                                                  &params,
                                                                  sizeof(params),
                                                                  ov::element::u32,
                                                                  ov::Shape{4}));
    return payload;
}

inline GfxKernelRuntimeParamPayload make_transpose_runtime_param_payload(GpuBufferManager& buffer_manager,
                                                                         std::string_view stage_name,
                                                                         const ov::Shape& input_shape,
                                                                         const ov::Shape& output_shape,
                                                                         const std::vector<int64_t>& permutation) {
    OPENVINO_ASSERT(input_shape.size() == output_shape.size() && input_shape.size() == permutation.size(),
                    "GFX MLIR: Transpose rank mismatch for stage ",
                    stage_name);
    std::vector<uint32_t> perm_u32(permutation.size(), 0);
    for (size_t i = 0; i < permutation.size(); ++i) {
        const int64_t axis = permutation[i];
        OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < input_shape.size(),
                        "GFX MLIR: Transpose perm out of range for stage ",
                        stage_name);
        perm_u32[i] = static_cast<uint32_t>(axis);
    }

    GfxKernelRuntimeParamPayload payload;
    payload.extra_inputs.push_back(make_kernel_u32_param_tensor(buffer_manager,
                                                               stage_name,
                                                               "transpose_total",
                                                               {static_cast<uint32_t>(ov::shape_size(output_shape))}));
    payload.extra_inputs.push_back(make_kernel_u32_param_tensor(buffer_manager,
                                                               stage_name,
                                                               "transpose_rank",
                                                               {static_cast<uint32_t>(output_shape.size())}));
    payload.extra_inputs.push_back(make_kernel_u32_param_tensor(buffer_manager,
                                                               stage_name,
                                                               "transpose_out_shape",
                                                               gfx_shape_to_u32_vector(output_shape)));
    payload.extra_inputs.push_back(make_kernel_u32_param_tensor(buffer_manager,
                                                               stage_name,
                                                               "transpose_perm",
                                                               perm_u32));
    payload.extra_inputs.push_back(make_kernel_u32_param_tensor(buffer_manager,
                                                               stage_name,
                                                               "transpose_in_stride",
                                                               gfx_shape_strides_u32(input_shape)));
    return payload;
}

inline GfxKernelRuntimeParamPayload make_slice_runtime_param_payload(GpuBufferManager& buffer_manager,
                                                                     std::string_view stage_name,
                                                                     const ov::Shape& input_shape,
                                                                     const ov::Shape& output_shape,
                                                                     const std::vector<int32_t>& starts,
                                                                     const std::vector<int32_t>& steps) {
    const size_t rank = input_shape.size();
    OPENVINO_ASSERT(rank == output_shape.size() && rank == starts.size() && rank == steps.size(),
                    "GFX MLIR: Slice runtime metadata rank mismatch for stage ",
                    stage_name);
    GfxKernelRuntimeParamPayload payload;
    payload.extra_inputs.push_back(make_kernel_u32_param_tensor(buffer_manager,
                                                               stage_name,
                                                               "slice_total",
                                                               {static_cast<uint32_t>(ov::shape_size(output_shape))}));
    payload.extra_inputs.push_back(make_kernel_u32_param_tensor(buffer_manager,
                                                               stage_name,
                                                               "slice_rank",
                                                               {static_cast<uint32_t>(rank)}));
    payload.extra_inputs.push_back(make_kernel_u32_param_tensor(buffer_manager,
                                                               stage_name,
                                                               "slice_out_shape",
                                                               gfx_shape_to_u32_vector(output_shape)));
    payload.extra_inputs.push_back(make_kernel_u32_param_tensor(buffer_manager,
                                                               stage_name,
                                                               "slice_in_stride",
                                                               gfx_shape_strides_u32(input_shape)));
    payload.extra_inputs.push_back(make_kernel_i32_param_tensor(buffer_manager, stage_name, "slice_starts", starts));
    payload.extra_inputs.push_back(make_kernel_i32_param_tensor(buffer_manager, stage_name, "slice_steps", steps));
    return payload;
}

inline GfxKernelRuntimeParamPayload make_scatter_update_runtime_param_payload(GpuBufferManager& buffer_manager,
                                                                              std::string_view stage_name,
                                                                              const ov::Shape& data_shape,
                                                                              const ov::Shape& indices_shape,
                                                                              const ov::Shape& updates_shape,
                                                                              uint32_t axis) {
    OPENVINO_ASSERT(data_shape.size() <= 8 && indices_shape.size() <= 8 && updates_shape.size() <= 16,
                    "GFX MLIR: ScatterUpdate rank exceeds kernel metadata capacity for stage ",
                    stage_name);
    struct ScatterUpdateParams {
        uint32_t data_rank = 0;
        uint32_t idx_rank = 0;
        uint32_t update_rank = 0;
        uint32_t axis = 0;
        uint32_t total_data = 0;
        uint32_t idx_total = 0;
        uint32_t data_dims[8]{};
        uint32_t data_strides[8]{};
        uint32_t idx_dims[8]{};
        uint32_t idx_strides[8]{};
        uint32_t update_strides[16]{};
    } params{};

    params.data_rank = static_cast<uint32_t>(data_shape.size());
    params.idx_rank = static_cast<uint32_t>(indices_shape.size());
    params.update_rank = static_cast<uint32_t>(updates_shape.size());
    params.axis = axis;
    params.total_data = static_cast<uint32_t>(ov::shape_size(data_shape));
    params.idx_total = static_cast<uint32_t>(ov::shape_size(indices_shape));
    const auto data_strides = gfx_shape_strides_u32(data_shape);
    const auto idx_strides = gfx_shape_strides_u32(indices_shape);
    const auto update_strides = gfx_shape_strides_u32(updates_shape);
    for (size_t i = 0; i < data_shape.size(); ++i) {
        params.data_dims[i] = static_cast<uint32_t>(data_shape[i]);
        params.data_strides[i] = data_strides[i];
    }
    for (size_t i = 0; i < indices_shape.size(); ++i) {
        params.idx_dims[i] = static_cast<uint32_t>(indices_shape[i]);
        params.idx_strides[i] = idx_strides[i];
    }
    for (size_t i = 0; i < updates_shape.size(); ++i) {
        params.update_strides[i] = update_strides[i];
    }

    std::ostringstream suffix;
    suffix << "scatter_update_params/"
           << params.total_data << "x" << params.idx_total << "x" << params.axis;
    GfxKernelRuntimeParamPayload payload;
    payload.extra_inputs.push_back(make_kernel_bytes_param_tensor(buffer_manager,
                                                                  stage_name,
                                                                  suffix.str(),
                                                                  &params,
                                                                  sizeof(params),
                                                                  ov::element::u8,
                                                                  ov::Shape{sizeof(params)}));
    return payload;
}

inline GfxKernelRuntimeParamPayload make_binary_broadcast_runtime_param_payload(
    GpuBufferManager& buffer_manager,
    std::string_view stage_name,
    const ov::Shape& output_shape,
    std::vector<int32_t> lhs_strides,
    std::vector<int32_t> rhs_strides) {
    const ov::Shape meta_shape = output_shape.empty() ? ov::Shape{1} : output_shape;
    OPENVINO_ASSERT(lhs_strides.size() == meta_shape.size() && rhs_strides.size() == meta_shape.size(),
                    "GFX MLIR: binary broadcast metadata rank mismatch for stage ",
                    stage_name);

    GfxKernelRuntimeParamPayload payload;
    payload.extra_inputs.push_back(make_kernel_i32_param_tensor(buffer_manager,
                                                               stage_name,
                                                               "out_dims",
                                                               gfx_shape_to_i32_vector(meta_shape)));
    payload.extra_inputs.push_back(make_kernel_i32_param_tensor(buffer_manager,
                                                               stage_name,
                                                               "stride0",
                                                               lhs_strides));
    payload.extra_inputs.push_back(make_kernel_i32_param_tensor(buffer_manager,
                                                               stage_name,
                                                               "stride1",
                                                               rhs_strides));
    payload.scalar_args = {
        static_cast<int32_t>(ov::shape_size(output_shape)),
        static_cast<int32_t>(meta_shape.size())};
    return payload;
}

inline GfxKernelRuntimeParamPayload make_select_runtime_param_payload(GpuBufferManager& buffer_manager,
                                                                      std::string_view stage_name,
                                                                      const ov::Shape& condition_shape,
                                                                      const ov::Shape& true_shape,
                                                                      const ov::Shape& false_shape,
                                                                      const ov::Shape& output_shape) {
    const ov::Shape meta_shape = output_shape.empty() ? ov::Shape{1} : output_shape;
    GfxKernelRuntimeParamPayload payload;
    payload.extra_inputs.push_back(make_kernel_i32_param_tensor(buffer_manager,
                                                               stage_name,
                                                               "select_out_dims",
                                                               gfx_shape_to_i32_vector(meta_shape)));
    payload.extra_inputs.push_back(make_kernel_i32_param_tensor(buffer_manager,
                                                               stage_name,
                                                               "select_stride_cond",
                                                               compute_broadcast_element_strides(condition_shape, meta_shape)));
    payload.extra_inputs.push_back(make_kernel_i32_param_tensor(buffer_manager,
                                                               stage_name,
                                                               "select_stride_true",
                                                               compute_broadcast_element_strides(true_shape, meta_shape)));
    payload.extra_inputs.push_back(make_kernel_i32_param_tensor(buffer_manager,
                                                               stage_name,
                                                               "select_stride_false",
                                                               compute_broadcast_element_strides(false_shape, meta_shape)));
    payload.scalar_args = {
        static_cast<int32_t>(ov::shape_size(output_shape)),
        static_cast<int32_t>(meta_shape.size())};
    return payload;
}

inline GfxKernelRuntimeParamPayload make_reduce_runtime_param_payload(GpuBufferManager& buffer_manager,
                                                                      std::string_view stage_name,
                                                                      const ov::Shape& input_shape,
                                                                      const ov::AxisSet& axes,
                                                                      bool keep_dims,
                                                                      const ov::Shape& output_shape) {
    const size_t rank = input_shape.size();
    OPENVINO_ASSERT(rank <= 8, "GFX MLIR: Reduce rank exceeds kernel metadata capacity for stage ", stage_name);
    std::vector<int32_t> out_dims(rank, 1);
    std::vector<int32_t> in_dims = gfx_shape_to_i32_vector(input_shape);
    in_dims.resize(rank, 1);
    std::vector<int32_t> in_strides = gfx_shape_strides_i32(input_shape);
    in_strides.resize(rank, 1);
    std::vector<int32_t> axis_mask(rank, 0);
    std::vector<int32_t> reduce_dims(rank, 1);
    size_t output_axis = 0;
    for (size_t i = 0; i < rank; ++i) {
        const bool reduced = axes.count(i) != 0;
        axis_mask[i] = reduced ? 1 : 0;
        reduce_dims[i] = reduced ? static_cast<int32_t>(input_shape[i]) : 1;
        if (reduced) {
            out_dims[i] = 1;
            continue;
        }
        out_dims[i] = keep_dims ? static_cast<int32_t>(input_shape[i])
                                : static_cast<int32_t>(output_shape[output_axis++]);
    }

    GfxKernelRuntimeParamPayload payload;
    payload.extra_inputs.push_back(make_kernel_i32_param_tensor(buffer_manager, stage_name, "reduce_out_dims", out_dims));
    payload.extra_inputs.push_back(make_kernel_i32_param_tensor(buffer_manager, stage_name, "reduce_in_dims", in_dims));
    payload.extra_inputs.push_back(make_kernel_i32_param_tensor(buffer_manager, stage_name, "reduce_in_strides", in_strides));
    payload.extra_inputs.push_back(make_kernel_i32_param_tensor(buffer_manager, stage_name, "reduce_axis_mask", axis_mask));
    payload.extra_inputs.push_back(make_kernel_i32_param_tensor(buffer_manager, stage_name, "reduce_dims", reduce_dims));
    payload.scalar_args = {
        static_cast<int32_t>(ov::shape_size(output_shape)),
        static_cast<int32_t>(rank)};
    return payload;
}

inline GfxKernelRuntimeParamPayload make_broadcast_runtime_param_payload(GpuBufferManager& buffer_manager,
                                                                         std::string_view stage_name,
                                                                         const ov::Shape& input_shape,
                                                                         const ov::Shape& output_shape) {
    const size_t out_rank = output_shape.size();
    const size_t in_rank = input_shape.size();
    OPENVINO_ASSERT(out_rank <= 8, "GFX MLIR: Broadcast rank exceeds kernel metadata capacity for stage ", stage_name);
    OPENVINO_ASSERT(in_rank <= out_rank, "GFX MLIR: Broadcast input rank exceeds output rank for stage ", stage_name);
    std::vector<int32_t> axes(std::max<size_t>(in_rank, 1), 0);
    for (size_t i = 0; i < in_rank; ++i) {
        axes[i] = static_cast<int32_t>(out_rank - in_rank + i);
    }

    GfxKernelRuntimeParamPayload payload;
    payload.extra_inputs.push_back(make_kernel_i32_param_tensor(buffer_manager,
                                                               stage_name,
                                                               "broadcast_out_dims",
                                                               gfx_shape_to_i32_vector(output_shape)));
    payload.extra_inputs.push_back(make_kernel_i32_param_tensor(buffer_manager,
                                                               stage_name,
                                                               "broadcast_in_dims",
                                                               gfx_shape_to_i32_vector(input_shape)));
    payload.extra_inputs.push_back(make_kernel_i32_param_tensor(buffer_manager,
                                                               stage_name,
                                                               "broadcast_in_strides",
                                                               gfx_shape_strides_i32(input_shape)));
    payload.extra_inputs.push_back(make_kernel_i32_param_tensor(buffer_manager, stage_name, "broadcast_axes", axes));
    payload.scalar_args = {
        static_cast<int32_t>(ov::shape_size(output_shape)),
        static_cast<int32_t>(out_rank),
        static_cast<int32_t>(in_rank)};
    return payload;
}

inline GfxKernelRuntimeParamPayload make_interpolate_runtime_param_payload(GpuBufferManager& buffer_manager,
                                                                           std::string_view stage_name,
                                                                           const ov::Shape& input_shape,
                                                                           const ov::Shape& output_shape,
                                                                           bool align_corners,
                                                                           bool use_half_pixel,
                                                                           uint32_t nearest_mode) {
    OPENVINO_ASSERT(input_shape.size() == 4 && output_shape.size() == 4,
                    "GFX MLIR: Interpolate expects NCHW rank4 for stage ",
                    stage_name);
    struct GfxInterpolateRuntimeParams {
        uint32_t N = 0;
        uint32_t C = 0;
        uint32_t H_in = 0;
        uint32_t W_in = 0;
        uint32_t H_out = 0;
        uint32_t W_out = 0;
        float scale_h = 1.0f;
        float scale_w = 1.0f;
        uint32_t align_corners = 0;
        uint32_t use_half_pixel = 0;
        uint32_t nearest_mode = 0;
    } params{};
    params.N = static_cast<uint32_t>(input_shape[0]);
    params.C = static_cast<uint32_t>(input_shape[1]);
    params.H_in = static_cast<uint32_t>(input_shape[2]);
    params.W_in = static_cast<uint32_t>(input_shape[3]);
    params.H_out = static_cast<uint32_t>(output_shape[2]);
    params.W_out = static_cast<uint32_t>(output_shape[3]);
    params.scale_h = params.H_out ? static_cast<float>(params.H_in) / static_cast<float>(params.H_out) : 1.f;
    params.scale_w = params.W_out ? static_cast<float>(params.W_in) / static_cast<float>(params.W_out) : 1.f;
    params.align_corners = align_corners ? 1u : 0u;
    params.use_half_pixel = use_half_pixel ? 1u : 0u;
    params.nearest_mode = nearest_mode;

    return make_single_bytes_runtime_param_payload(buffer_manager,
                                                   stage_name,
                                                   "interpolate_params",
                                                   &params,
                                                   sizeof(params),
                                                   ov::element::u8,
                                                   ov::Shape{sizeof(params)});
}

}  // namespace gfx_plugin
}  // namespace ov
