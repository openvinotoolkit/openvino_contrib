// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "runtime/gfx_kernel_runtime_params_core.hpp"

namespace ov {
namespace gfx_plugin {

inline GfxKernelRuntimeParamPayload make_scatter_update_runtime_param_payload(
    GpuBufferManager &buffer_manager, std::string_view stage_name,
    const ov::Shape &data_shape, const ov::Shape &indices_shape,
    const ov::Shape &updates_shape, uint32_t axis) {
  OPENVINO_ASSERT(data_shape.size() <= 8 && indices_shape.size() <= 8 &&
                      updates_shape.size() <= 16,
                  "GFX runtime: ScatterUpdate rank exceeds kernel metadata "
                  "capacity for stage ",
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
  suffix << "scatter_update_params/" << params.total_data << "x"
         << params.idx_total << "x" << params.axis;
  GfxKernelRuntimeParamPayload payload;
  payload.extra_inputs.push_back(make_kernel_bytes_param_tensor(
      buffer_manager, stage_name, suffix.str(), &params, sizeof(params),
      ov::element::u8, ov::Shape{sizeof(params)}));
  return payload;
}

inline GfxKernelRuntimeParamPayload
make_scatter_elements_update_runtime_param_payload(
    GpuBufferManager &buffer_manager, std::string_view stage_name,
    const ov::Shape &data_shape, const ov::Shape &indices_shape,
    uint32_t axis) {
  OPENVINO_ASSERT(data_shape.size() <= 8 && indices_shape.size() <= 8,
                  "GFX runtime: ScatterElementsUpdate rank exceeds kernel "
                  "metadata capacity for stage ",
                  stage_name);
  struct ScatterElementsParams {
    uint32_t rank = 0;
    uint32_t axis = 0;
    uint32_t total_updates = 0;
    uint32_t total_data = 0;
    uint32_t update_dims[8]{};
    uint32_t update_strides[8]{};
    uint32_t data_dims[8]{};
    uint32_t data_strides[8]{};
  } params{};

  params.rank = static_cast<uint32_t>(data_shape.size());
  params.axis = axis;
  params.total_updates = static_cast<uint32_t>(ov::shape_size(indices_shape));
  params.total_data = static_cast<uint32_t>(ov::shape_size(data_shape));
  const auto update_strides = gfx_shape_strides_u32(indices_shape);
  const auto data_strides = gfx_shape_strides_u32(data_shape);
  for (size_t i = 0; i < indices_shape.size(); ++i) {
    params.update_dims[i] = static_cast<uint32_t>(indices_shape[i]);
    params.update_strides[i] = update_strides[i];
  }
  for (size_t i = 0; i < data_shape.size(); ++i) {
    params.data_dims[i] = static_cast<uint32_t>(data_shape[i]);
    params.data_strides[i] = data_strides[i];
  }

  std::ostringstream suffix;
  suffix << "scatter_elements_update_params/" << params.total_data << "x"
         << params.total_updates << "x" << params.axis;
  GfxKernelRuntimeParamPayload payload;
  payload.extra_inputs.push_back(make_kernel_bytes_param_tensor(
      buffer_manager, stage_name, suffix.str(), &params, sizeof(params),
      ov::element::u8, ov::Shape{sizeof(params)}));
  return payload;
}

inline GfxKernelRuntimeParamPayload
make_scatter_nd_update_runtime_param_payload(GpuBufferManager &buffer_manager,
                                             std::string_view stage_name,
                                             const ov::Shape &data_shape,
                                             const ov::Shape &indices_shape,
                                             const ov::Shape &updates_shape) {
  OPENVINO_ASSERT(data_shape.size() <= 8,
                  "GFX runtime: ScatterNDUpdate rank exceeds kernel metadata "
                  "capacity for stage ",
                  stage_name);
  OPENVINO_ASSERT(
      !indices_shape.empty(),
      "GFX runtime: ScatterNDUpdate indices rank must be positive for stage ",
      stage_name);
  const auto k = static_cast<uint32_t>(indices_shape.back());
  OPENVINO_ASSERT(
      k > 0,
      "GFX runtime: ScatterNDUpdate index depth must be positive for stage ",
      stage_name);
  OPENVINO_ASSERT(
      k <= data_shape.size(),
      "GFX runtime: ScatterNDUpdate index depth exceeds data rank for stage ",
      stage_name);
  struct ScatterNDParams {
    uint32_t inner = 0;
    uint32_t num_indices = 0;
    uint32_t k = 0;
    uint32_t total_updates = 0;
    uint32_t total_data = 0;
    uint32_t strides[8]{};
    uint32_t dims[8]{};
  } params{};

  params.k = k;
  params.num_indices = static_cast<uint32_t>(ov::shape_size(indices_shape) / k);
  params.total_updates = static_cast<uint32_t>(ov::shape_size(updates_shape));
  params.total_data = static_cast<uint32_t>(ov::shape_size(data_shape));
  const auto data_strides = gfx_shape_strides_u32(data_shape);
  params.inner = 1u;
  for (size_t dim = static_cast<size_t>(k); dim < data_shape.size(); ++dim) {
    params.inner *= static_cast<uint32_t>(data_shape[dim]);
  }
  for (size_t i = 0; i < data_shape.size(); ++i) {
    params.dims[i] = static_cast<uint32_t>(data_shape[i]);
    params.strides[i] = data_strides[i];
  }

  std::ostringstream suffix;
  suffix << "scatter_nd_update_params/" << params.total_data << "x"
         << params.num_indices << "x" << params.k << "x" << params.inner;
  GfxKernelRuntimeParamPayload payload;
  payload.extra_inputs.push_back(make_kernel_bytes_param_tensor(
      buffer_manager, stage_name, suffix.str(), &params, sizeof(params),
      ov::element::u8, ov::Shape{sizeof(params)}));
  return payload;
}

} // namespace gfx_plugin
} // namespace ov
