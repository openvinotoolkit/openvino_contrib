// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/axis_set.hpp"
#include "runtime/gfx_kernel_runtime_params_core.hpp"

namespace ov {
namespace gfx_plugin {

inline GfxKernelRuntimeParamPayload make_shapeof_runtime_param_payload(
    GpuBufferManager &buffer_manager, std::string_view stage_name,
    const ov::Shape &runtime_shape, const ov::element::Type &output_type) {
  OPENVINO_ASSERT(output_type == ov::element::i32 ||
                      output_type == ov::element::i64,
                  "GFX runtime: ShapeOf output must be i32/i64");
  std::ostringstream suffix;
  suffix << "shapeof_dims/";
  GfxKernelRuntimeParamPayload payload;
  if (output_type == ov::element::i32) {
    std::vector<int32_t> dims(runtime_shape.size(), 0);
    for (size_t i = 0; i < runtime_shape.size(); ++i) {
      dims[i] = static_cast<int32_t>(runtime_shape[i]);
      suffix << dims[i] << 'x';
    }
    payload.extra_inputs.push_back(make_kernel_i32_param_tensor(
        buffer_manager, stage_name, suffix.str(), dims));
  } else {
    std::vector<int64_t> dims(runtime_shape.size(), 0);
    for (size_t i = 0; i < runtime_shape.size(); ++i) {
      dims[i] = static_cast<int64_t>(runtime_shape[i]);
      suffix << dims[i] << 'x';
    }
    payload.extra_inputs.push_back(make_kernel_bytes_param_tensor(
        buffer_manager, stage_name, suffix.str(), dims.data(),
        dims.size() * sizeof(int64_t), ov::element::i64,
        ov::Shape{dims.size()}));
  }
  payload.scalar_args = {static_cast<int32_t>(runtime_shape.size())};
  return payload;
}

inline GfxKernelRuntimeParamPayload make_tile_runtime_param_payload(
    GpuBufferManager &buffer_manager, std::string_view stage_name,
    const ov::Shape &input_shape, const ov::Shape &output_shape) {
  GfxKernelRuntimeParamPayload payload;
  payload.extra_inputs.push_back(
      make_kernel_i32_param_tensor(buffer_manager, stage_name, "tile_out_dims",
                                   gfx_shape_to_i32_vector(output_shape)));
  payload.extra_inputs.push_back(
      make_kernel_i32_param_tensor(buffer_manager, stage_name, "tile_in_dims",
                                   gfx_shape_to_i32_vector(input_shape)));
  payload.extra_inputs.push_back(make_kernel_i32_param_tensor(
      buffer_manager, stage_name, "tile_out_strides",
      gfx_shape_strides_i32(output_shape)));
  payload.extra_inputs.push_back(make_kernel_i32_param_tensor(
      buffer_manager, stage_name, "tile_in_strides",
      gfx_shape_strides_i32(input_shape)));
  payload.scalar_args = {
      static_cast<int32_t>(ov::shape_size(output_shape)),
      static_cast<int32_t>(std::max<size_t>(output_shape.size(), 1))};
  return payload;
}

inline GfxKernelRuntimeParamPayload make_gather_elements_runtime_param_payload(
    GpuBufferManager &buffer_manager, std::string_view stage_name,
    const ov::Shape &data_shape, const ov::Shape &output_shape, uint32_t axis) {
  OPENVINO_ASSERT(
      data_shape.size() == output_shape.size(),
      "GFX runtime: GatherElements data/output rank mismatch for stage ",
      stage_name);
  OPENVINO_ASSERT(output_shape.size() <= GatherElementsCodegenDesc::kMaxDims,
                  "GFX runtime: GatherElements rank exceeds kernel metadata "
                  "capacity for stage ",
                  stage_name);
  std::vector<uint32_t> params(GatherElementsCodegenDesc::kParamU32Count, 0);
  const auto data_strides_i64 = make_element_strides(data_shape);
  const auto out_strides_i64 = make_element_strides(output_shape);
  params[GatherElementsCodegenDesc::kRankOffset] =
      static_cast<uint32_t>(output_shape.size());
  params[GatherElementsCodegenDesc::kAxisOffset] = axis;
  params[GatherElementsCodegenDesc::kTotalOffset] =
      static_cast<uint32_t>(ov::shape_size(output_shape));
  for (size_t i = 0; i < output_shape.size(); ++i) {
    params[GatherElementsCodegenDesc::kOutDimsOffset + i] =
        static_cast<uint32_t>(output_shape[i]);
    params[GatherElementsCodegenDesc::kOutStridesOffset + i] =
        static_cast<uint32_t>(out_strides_i64[i]);
    params[GatherElementsCodegenDesc::kDataDimsOffset + i] =
        static_cast<uint32_t>(data_shape[i]);
    params[GatherElementsCodegenDesc::kDataStridesOffset + i] =
        static_cast<uint32_t>(data_strides_i64[i]);
  }

  GfxKernelRuntimeParamPayload payload;
  payload.extra_inputs.push_back(make_kernel_u32_param_tensor(
      buffer_manager, stage_name, "gather_elements_params", params));
  return payload;
}

inline GfxKernelRuntimeParamPayload make_gather_runtime_param_payload(
    GpuBufferManager &buffer_manager, std::string_view stage_name,
    const ov::Shape &data_shape, const ov::Shape &indices_shape,
    uint32_t axis) {
  OPENVINO_ASSERT(axis < data_shape.size(),
                  "GFX runtime: Gather axis out of range for stage ", stage_name);
  struct GatherParams {
    uint32_t outer = 0;
    uint32_t inner = 0;
    uint32_t axis_dim = 0;
    uint32_t indices_count = 0;
  } params{};
  params.outer = static_cast<uint32_t>(shape_product(data_shape, 0, axis));
  params.inner = static_cast<uint32_t>(shape_product(
      data_shape, static_cast<size_t>(axis) + 1, data_shape.size()));
  params.axis_dim = static_cast<uint32_t>(data_shape[axis]);
  params.indices_count = static_cast<uint32_t>(ov::shape_size(indices_shape));

  std::ostringstream suffix;
  suffix << "gather_params/" << params.outer << 'x' << params.inner << 'x'
         << params.axis_dim << 'x' << params.indices_count;
  GfxKernelRuntimeParamPayload payload;
  payload.extra_inputs.push_back(make_kernel_bytes_param_tensor(
      buffer_manager, stage_name, suffix.str(), &params, sizeof(params),
      ov::element::u32, ov::Shape{4}));
  return payload;
}

inline GfxKernelRuntimeParamPayload make_gather_nd_runtime_param_payload(
    GpuBufferManager &buffer_manager, std::string_view stage_name,
    const ov::Shape &data_shape, const ov::Shape &indices_shape,
    const ov::Shape &output_shape) {
  OPENVINO_ASSERT(
      !data_shape.empty(),
      "GFX runtime: GatherND data rank must be static and non-zero for stage ",
      stage_name);
  OPENVINO_ASSERT(
      !indices_shape.empty(),
      "GFX runtime: GatherND indices rank must be static and non-zero for stage ",
      stage_name);
  OPENVINO_ASSERT(
      data_shape.size() <= GatherNDCodegenDesc::kMaxDims,
      "GFX runtime: GatherND rank exceeds kernel metadata capacity for stage ",
      stage_name);
  struct GatherNDParams {
    uint32_t inner = 0;
    uint32_t num_indices = 0;
    uint32_t k = 0;
    uint32_t total = 0;
    uint32_t strides[GatherNDCodegenDesc::kMaxDims]{};
    uint32_t dims[GatherNDCodegenDesc::kMaxDims]{};
  } params{};

  params.k = static_cast<uint32_t>(indices_shape.back());
  OPENVINO_ASSERT(params.k >= 1 && params.k <= data_shape.size(),
                  "GFX runtime: GatherND invalid index depth for stage ",
                  stage_name);
  params.inner = static_cast<uint32_t>(
      shape_product(data_shape, params.k, data_shape.size()));
  params.num_indices =
      static_cast<uint32_t>(ov::shape_size(indices_shape) / params.k);
  params.total = static_cast<uint32_t>(ov::shape_size(output_shape));
  OPENVINO_ASSERT(params.total == params.num_indices * params.inner,
                  "GFX runtime: GatherND output shape does not match index/inner "
                  "contract for stage ",
                  stage_name);

  const auto data_strides = gfx_shape_strides_u32(data_shape);
  for (size_t i = 0; i < data_shape.size(); ++i) {
    params.dims[i] = static_cast<uint32_t>(data_shape[i]);
    params.strides[i] = data_strides[i];
  }

  std::ostringstream suffix;
  suffix << "gather_nd_params/" << params.total << 'x' << params.num_indices
         << 'x' << params.k << 'x' << params.inner;
  GfxKernelRuntimeParamPayload payload;
  payload.extra_inputs.push_back(make_kernel_bytes_param_tensor(
      buffer_manager, stage_name, suffix.str(), &params, sizeof(params),
      ov::element::u8, ov::Shape{sizeof(params)}));
  return payload;
}

inline GfxKernelRuntimeParamPayload make_transpose_runtime_param_payload(
    GpuBufferManager &buffer_manager, std::string_view stage_name,
    const ov::Shape &input_shape, const ov::Shape &output_shape,
    const std::vector<int64_t> &permutation) {
  OPENVINO_ASSERT(input_shape.size() == output_shape.size() &&
                      input_shape.size() == permutation.size(),
                  "GFX runtime: Transpose rank mismatch for stage ", stage_name);
  std::vector<uint32_t> perm_u32(permutation.size(), 0);
  for (size_t i = 0; i < permutation.size(); ++i) {
    const int64_t axis = permutation[i];
    OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < input_shape.size(),
                    "GFX runtime: Transpose perm out of range for stage ",
                    stage_name);
    perm_u32[i] = static_cast<uint32_t>(axis);
  }

  GfxKernelRuntimeParamPayload payload;
  payload.extra_inputs.push_back(make_kernel_u32_param_tensor(
      buffer_manager, stage_name, "transpose_total",
      {static_cast<uint32_t>(ov::shape_size(output_shape))}));
  payload.extra_inputs.push_back(make_kernel_u32_param_tensor(
      buffer_manager, stage_name, "transpose_rank",
      {static_cast<uint32_t>(output_shape.size())}));
  payload.extra_inputs.push_back(make_kernel_u32_param_tensor(
      buffer_manager, stage_name, "transpose_out_shape",
      gfx_shape_to_u32_vector(output_shape)));
  payload.extra_inputs.push_back(make_kernel_u32_param_tensor(
      buffer_manager, stage_name, "transpose_perm", perm_u32));
  payload.extra_inputs.push_back(make_kernel_u32_param_tensor(
      buffer_manager, stage_name, "transpose_in_stride",
      gfx_shape_strides_u32(input_shape)));
  return payload;
}

inline GfxKernelRuntimeParamPayload make_slice_runtime_param_payload(
    GpuBufferManager &buffer_manager, std::string_view stage_name,
    const ov::Shape &input_shape, const ov::Shape &output_shape,
    const std::vector<int32_t> &starts, const std::vector<int32_t> &steps) {
  const size_t rank = input_shape.size();
  OPENVINO_ASSERT(rank == output_shape.size() && rank == starts.size() &&
                      rank == steps.size(),
                  "GFX runtime: Slice runtime metadata rank mismatch for stage ",
                  stage_name);
  GfxKernelRuntimeParamPayload payload;
  payload.extra_inputs.push_back(make_kernel_u32_param_tensor(
      buffer_manager, stage_name, "slice_total",
      {static_cast<uint32_t>(ov::shape_size(output_shape))}));
  payload.extra_inputs.push_back(make_kernel_u32_param_tensor(
      buffer_manager, stage_name, "slice_rank", {static_cast<uint32_t>(rank)}));
  payload.extra_inputs.push_back(make_kernel_u32_param_tensor(
      buffer_manager, stage_name, "slice_out_shape",
      gfx_shape_to_u32_vector(output_shape)));
  payload.extra_inputs.push_back(make_kernel_u32_param_tensor(
      buffer_manager, stage_name, "slice_in_stride",
      gfx_shape_strides_u32(input_shape)));
  payload.extra_inputs.push_back(make_kernel_i32_param_tensor(
      buffer_manager, stage_name, "slice_starts", starts));
  payload.extra_inputs.push_back(make_kernel_i32_param_tensor(
      buffer_manager, stage_name, "slice_steps", steps));
  return payload;
}

inline GfxKernelRuntimeParamPayload make_binary_broadcast_runtime_param_payload(
    GpuBufferManager &buffer_manager, std::string_view stage_name,
    const ov::Shape &output_shape, std::vector<int32_t> lhs_strides,
    std::vector<int32_t> rhs_strides) {
  const ov::Shape meta_shape =
      output_shape.empty() ? ov::Shape{1} : output_shape;
  OPENVINO_ASSERT(
      lhs_strides.size() == meta_shape.size() &&
          rhs_strides.size() == meta_shape.size(),
      "GFX runtime: binary broadcast metadata rank mismatch for stage ",
      stage_name);

  GfxKernelRuntimeParamPayload payload;
  payload.extra_inputs.push_back(
      make_kernel_i32_param_tensor(buffer_manager, stage_name, "out_dims",
                                   gfx_shape_to_i32_vector(meta_shape)));
  payload.extra_inputs.push_back(make_kernel_i32_param_tensor(
      buffer_manager, stage_name, "stride0", lhs_strides));
  payload.extra_inputs.push_back(make_kernel_i32_param_tensor(
      buffer_manager, stage_name, "stride1", rhs_strides));
  payload.scalar_args = {static_cast<int32_t>(ov::shape_size(output_shape)),
                         static_cast<int32_t>(meta_shape.size())};
  return payload;
}

inline GfxKernelRuntimeParamPayload make_select_runtime_param_payload(
    GpuBufferManager &buffer_manager, std::string_view stage_name,
    const ov::Shape &condition_shape, const ov::Shape &true_shape,
    const ov::Shape &false_shape, const ov::Shape &output_shape) {
  const ov::Shape meta_shape =
      output_shape.empty() ? ov::Shape{1} : output_shape;
  GfxKernelRuntimeParamPayload payload;
  payload.extra_inputs.push_back(make_kernel_i32_param_tensor(
      buffer_manager, stage_name, "select_out_dims",
      gfx_shape_to_i32_vector(meta_shape)));
  payload.extra_inputs.push_back(make_kernel_i32_param_tensor(
      buffer_manager, stage_name, "select_stride_cond",
      compute_broadcast_element_strides(condition_shape, meta_shape)));
  payload.extra_inputs.push_back(make_kernel_i32_param_tensor(
      buffer_manager, stage_name, "select_stride_true",
      compute_broadcast_element_strides(true_shape, meta_shape)));
  payload.extra_inputs.push_back(make_kernel_i32_param_tensor(
      buffer_manager, stage_name, "select_stride_false",
      compute_broadcast_element_strides(false_shape, meta_shape)));
  payload.scalar_args = {static_cast<int32_t>(ov::shape_size(output_shape)),
                         static_cast<int32_t>(meta_shape.size())};
  return payload;
}

inline GfxKernelRuntimeParamPayload make_reduce_runtime_param_payload(
    GpuBufferManager &buffer_manager, std::string_view stage_name,
    const ov::Shape &input_shape, const ov::AxisSet &axes, bool keep_dims,
    const ov::Shape &output_shape, uint32_t op_code) {
  const size_t rank = input_shape.size();
  OPENVINO_ASSERT(
      rank <= 8,
      "GFX runtime: Reduce rank exceeds kernel metadata capacity for stage ",
      stage_name);
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
  payload.extra_inputs.push_back(make_kernel_i32_param_tensor(
      buffer_manager, stage_name, "reduce_out_dims", out_dims));
  payload.extra_inputs.push_back(make_kernel_i32_param_tensor(
      buffer_manager, stage_name, "reduce_in_dims", in_dims));
  payload.extra_inputs.push_back(make_kernel_i32_param_tensor(
      buffer_manager, stage_name, "reduce_in_strides", in_strides));
  payload.extra_inputs.push_back(make_kernel_i32_param_tensor(
      buffer_manager, stage_name, "reduce_axis_mask", axis_mask));
  payload.extra_inputs.push_back(make_kernel_i32_param_tensor(
      buffer_manager, stage_name, "reduce_dims", reduce_dims));
  payload.scalar_args = {static_cast<int32_t>(ov::shape_size(output_shape)),
                         static_cast<int32_t>(rank),
                         static_cast<int32_t>(op_code)};
  return payload;
}

inline GfxKernelRuntimeParamPayload make_broadcast_runtime_param_payload(
    GpuBufferManager &buffer_manager, std::string_view stage_name,
    const ov::Shape &input_shape, const ov::Shape &output_shape) {
  const size_t out_rank = output_shape.size();
  const size_t in_rank = input_shape.size();
  OPENVINO_ASSERT(
      out_rank <= 8,
      "GFX runtime: Broadcast rank exceeds kernel metadata capacity for stage ",
      stage_name);
  OPENVINO_ASSERT(
      in_rank <= out_rank,
      "GFX runtime: Broadcast input rank exceeds output rank for stage ",
      stage_name);
  std::vector<int32_t> axes(std::max<size_t>(in_rank, 1), 0);
  for (size_t i = 0; i < in_rank; ++i) {
    axes[i] = static_cast<int32_t>(out_rank - in_rank + i);
  }

  GfxKernelRuntimeParamPayload payload;
  payload.extra_inputs.push_back(make_kernel_i32_param_tensor(
      buffer_manager, stage_name, "broadcast_out_dims",
      gfx_shape_to_i32_vector(output_shape)));
  payload.extra_inputs.push_back(make_kernel_i32_param_tensor(
      buffer_manager, stage_name, "broadcast_in_dims",
      gfx_shape_to_i32_vector(input_shape)));
  payload.extra_inputs.push_back(make_kernel_i32_param_tensor(
      buffer_manager, stage_name, "broadcast_in_strides",
      gfx_shape_strides_i32(input_shape)));
  payload.extra_inputs.push_back(make_kernel_i32_param_tensor(
      buffer_manager, stage_name, "broadcast_axes", axes));
  payload.scalar_args = {static_cast<int32_t>(ov::shape_size(output_shape)),
                         static_cast<int32_t>(out_rank),
                         static_cast<int32_t>(in_rank)};
  return payload;
}

} // namespace gfx_plugin
} // namespace ov
