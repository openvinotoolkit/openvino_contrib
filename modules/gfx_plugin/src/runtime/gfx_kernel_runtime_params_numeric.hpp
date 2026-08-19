// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cmath>

#include "openvino/core/type/float16.hpp"
#include "runtime/gfx_kernel_runtime_params_core.hpp"

namespace ov {
namespace gfx_plugin {

template <typename KernelLike, typename StridesLike, typename PadsLike,
          typename DilationsLike>
inline GfxKernelRuntimeParamPayload make_pool_runtime_param_payload(
    GpuBufferManager &buffer_manager, std::string_view stage_name,
    const ov::Shape &input_shape, const ov::Shape &output_shape,
    const KernelLike &kernel, const StridesLike &strides,
    const PadsLike &pads_begin, const PadsLike &pads_end,
    const DilationsLike &dilations, bool is_avg, bool exclude_pad) {
  OPENVINO_ASSERT(input_shape.size() == 4 && output_shape.size() == 4,
                  "GFX runtime: pool expects NCHW shapes for stage ", stage_name);
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

  return make_single_bytes_runtime_param_payload(
      buffer_manager, stage_name, "pool_params", &params, sizeof(params),
      ov::element::u8, ov::Shape{sizeof(params)});
}

inline GfxKernelRuntimeParamPayload
make_softmax_runtime_param_payload(GpuBufferManager &buffer_manager,
                                   std::string_view stage_name, uint64_t rows,
                                   uint64_t axis_len, uint64_t inner) {
  std::vector<int32_t> params = {
      static_cast<int32_t>(rows),
      static_cast<int32_t>(axis_len),
      static_cast<int32_t>(inner),
  };
  GfxKernelRuntimeParamPayload payload;
  payload.extra_inputs.push_back(make_kernel_i32_param_tensor(
      buffer_manager, stage_name, "softmax_params", params));
  return payload;
}

template <typename StridesLike, typename PadsLike, typename DilationsLike>
inline GfxKernelRuntimeParamPayload make_conv3d_runtime_param_payload(
    GpuBufferManager &buffer_manager, std::string_view stage_name,
    const ov::Shape &input_shape, const ov::Shape &output_shape,
    const ov::Shape &weight_shape, const StridesLike &strides,
    const DilationsLike &dilations, const PadsLike &pads_begin,
    const PadsLike &pads_end) {
  OPENVINO_ASSERT(input_shape.size() == 5 && output_shape.size() == 5 &&
                      weight_shape.size() == 5,
                  "GFX runtime: Conv3D expects NCDHW/OIDHW shapes for stage ",
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
  params.dilationD =
      gfx_dim_u32_at(dilations, 0, stage_name, "conv3d dilation");
  params.dilationH =
      gfx_dim_u32_at(dilations, 1, stage_name, "conv3d dilation");
  params.dilationW =
      gfx_dim_u32_at(dilations, 2, stage_name, "conv3d dilation");
  params.padFront =
      gfx_dim_u32_at(pads_begin, 0, stage_name, "conv3d pad begin");
  params.padTop = gfx_dim_u32_at(pads_begin, 1, stage_name, "conv3d pad begin");
  params.padLeft =
      gfx_dim_u32_at(pads_begin, 2, stage_name, "conv3d pad begin");
  params.padBack = gfx_dim_u32_at(pads_end, 0, stage_name, "conv3d pad end");
  params.padBottom = gfx_dim_u32_at(pads_end, 1, stage_name, "conv3d pad end");
  params.padRight = gfx_dim_u32_at(pads_end, 2, stage_name, "conv3d pad end");
  params.outD = static_cast<uint32_t>(output_shape[2]);
  params.outH = static_cast<uint32_t>(output_shape[3]);
  params.outW = static_cast<uint32_t>(output_shape[4]);

  return make_single_bytes_runtime_param_payload(
      buffer_manager, stage_name, "conv3d_params", &params, sizeof(params),
      ov::element::u8, ov::Shape{sizeof(params)});
}

inline GpuTensor make_kernel_float_vector_param_tensor(
    GpuBufferManager &buffer_manager, std::string_view stage_name,
    std::string_view suffix, const std::vector<float> &values, ov::Shape shape,
    const ov::element::Type &preferred_type) {
  ov::element::Type buffer_type = preferred_type;
  const void *data_ptr = values.data();
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

  return make_kernel_bytes_param_tensor(buffer_manager, stage_name, suffix,
                                        data_ptr, bytes, buffer_type,
                                        std::move(shape));
}

inline GpuTensor make_kernel_float_vector_param_tensor(
    GpuBufferManager &buffer_manager, std::string_view stage_name,
    std::string_view suffix, const std::vector<float> &values,
    size_t tensor_channels, const ov::element::Type &preferred_type) {
  return make_kernel_float_vector_param_tensor(
      buffer_manager, stage_name, suffix, values, ov::Shape{tensor_channels},
      preferred_type);
}

inline GpuTensor make_bias_runtime_param_tensor(
    GpuBufferManager &buffer_manager, std::string_view stage_name,
    const std::vector<float> &values, const std::vector<int64_t> &source_shape,
    const ov::element::Type &source_element_type,
    const ov::element::Type &output_element_type, size_t output_rank,
    bool conv_like) {
  ov::element::Type bias_type = output_element_type == ov::element::dynamic
                                    ? source_element_type
                                    : output_element_type;
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

  return make_kernel_float_vector_param_tensor(
      buffer_manager, stage_name, "bias", values, std::move(shape), bias_type);
}

inline GfxKernelRuntimeParamPayload
make_batchnorm_scale_bias_runtime_param_payload(
    GpuBufferManager &buffer_manager, std::string_view stage_name,
    const std::vector<float> &gamma, const std::vector<float> &beta,
    const std::vector<float> &mean, const std::vector<float> &variance,
    float epsilon, const ov::element::Type &output_element_type) {
  const size_t channels = gamma.size();
  OPENVINO_ASSERT(beta.size() == channels && mean.size() == channels &&
                      variance.size() == channels,
                  "GFX runtime: BatchNorm parameter channel mismatch for stage ",
                  stage_name);
  ov::element::Type bn_type = output_element_type == ov::element::dynamic
                                  ? ov::element::f32
                                  : output_element_type;
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
  payload.extra_inputs.push_back(make_kernel_float_vector_param_tensor(
      buffer_manager, stage_name, "bn_scale", scale_values, channels, bn_type));
  payload.extra_inputs.push_back(make_kernel_float_vector_param_tensor(
      buffer_manager, stage_name, "bn_bias", bias_values, channels, bn_type));
  return payload;
}

inline GfxKernelRuntimeParamPayload make_conv2d_runtime_param_payload(
    GpuBufferManager &buffer_manager, std::string_view stage_name,
    const ov::Shape &input_shape, const ov::Shape &output_shape,
    const ov::element::Type &param_element_type, uint32_t c_out,
    uint32_t groups, uint32_t c_in_per_group, uint32_t c_out_per_group,
    uint32_t kernel_h, uint32_t kernel_w, uint32_t stride_h, uint32_t stride_w,
    uint32_t dilation_h, uint32_t dilation_w, uint32_t pad_top,
    uint32_t pad_left, uint32_t pad_bottom, uint32_t pad_right, bool has_bias,
    bool has_batchnorm, uint32_t activation, float activation_alpha,
    float batchnorm_epsilon, const std::vector<float> &bias,
    const std::vector<float> &gamma, const std::vector<float> &beta,
    const std::vector<float> &mean, const std::vector<float> &variance) {
  OPENVINO_ASSERT(input_shape.size() == 4 && output_shape.size() == 4,
                  "GFX runtime: Conv expects NCHW shapes for stage ", stage_name);
  const size_t channels = static_cast<size_t>(c_out);
  GfxKernelRuntimeParamPayload payload;
  payload.extra_inputs.push_back(make_kernel_float_vector_param_tensor(
      buffer_manager, stage_name, "bias", bias, channels, param_element_type));
  payload.extra_inputs.push_back(make_kernel_float_vector_param_tensor(
      buffer_manager, stage_name, "gamma", gamma, channels,
      param_element_type));
  payload.extra_inputs.push_back(make_kernel_float_vector_param_tensor(
      buffer_manager, stage_name, "beta", beta, channels, param_element_type));
  payload.extra_inputs.push_back(make_kernel_float_vector_param_tensor(
      buffer_manager, stage_name, "mean", mean, channels, param_element_type));
  payload.extra_inputs.push_back(make_kernel_float_vector_param_tensor(
      buffer_manager, stage_name, "var", variance, channels,
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

  payload.extra_inputs.push_back(make_kernel_bytes_param_tensor(
      buffer_manager, stage_name, "conv_params", &params, sizeof(params),
      ov::element::u8, ov::Shape{sizeof(params)}));
  return payload;
}

inline GfxKernelRuntimeParamPayload make_interpolate_runtime_param_payload(
    GpuBufferManager &buffer_manager, std::string_view stage_name,
    const ov::Shape &input_shape, const ov::Shape &output_shape,
    bool align_corners, bool use_half_pixel, uint32_t nearest_mode) {
  OPENVINO_ASSERT(input_shape.size() == 4 && output_shape.size() == 4,
                  "GFX runtime: Interpolate expects NCHW rank4 for stage ",
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
  params.scale_h = params.H_out ? static_cast<float>(params.H_in) /
                                      static_cast<float>(params.H_out)
                                : 1.f;
  params.scale_w = params.W_out ? static_cast<float>(params.W_in) /
                                      static_cast<float>(params.W_out)
                                : 1.f;
  params.align_corners = align_corners ? 1u : 0u;
  params.use_half_pixel = use_half_pixel ? 1u : 0u;
  params.nearest_mode = nearest_mode;

  return make_single_bytes_runtime_param_payload(
      buffer_manager, stage_name, "interpolate_params", &params, sizeof(params),
      ov::element::u8, ov::Shape{sizeof(params)});
}

} // namespace gfx_plugin
} // namespace ov
