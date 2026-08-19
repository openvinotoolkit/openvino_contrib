// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kernel_ir/gfx_codegen_desc.hpp"
#include "kernel_ir/gfx_kernel_cache.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/type/element_type.hpp"
#include "runtime/gfx_shape_utils.hpp"
#include "runtime/gpu_buffer_manager.hpp"
#include "runtime/gpu_tensor.hpp"

namespace ov {
namespace gfx_plugin {

struct GfxKernelRuntimeParamPayload {
  std::vector<GpuTensor> extra_inputs;
  std::vector<int32_t> scalar_args;
};

inline std::string make_kernel_param_key(std::string_view stage_name,
                                         std::string_view suffix) {
  std::string key(stage_name);
  key.push_back('/');
  key.append(suffix.data(), suffix.size());
  return key;
}

inline GpuTensor make_kernel_i32_param_tensor(
    GpuBufferManager &buffer_manager, std::string_view stage_name,
    std::string_view suffix, const std::vector<int32_t> &values) {
  GpuBuffer buf = buffer_manager.wrap_const(
      make_kernel_param_key(stage_name, suffix), values.data(),
      values.size() * sizeof(int32_t), ov::element::i32);
  OPENVINO_ASSERT(buf.valid(),
                  "GFX runtime: failed to wrap i32 kernel metadata for stage ",
                  stage_name);
  buf.owned = false;
  GpuTensor tensor;
  tensor.buf = std::move(buf);
  tensor.expected_type = ov::element::i32;
  tensor.shape = ov::Shape{values.size()};
  return tensor;
}

inline GpuTensor make_kernel_u32_param_tensor(
    GpuBufferManager &buffer_manager, std::string_view stage_name,
    std::string_view suffix, const std::vector<uint32_t> &values) {
  GpuBuffer buf = buffer_manager.wrap_const(
      make_kernel_param_key(stage_name, suffix), values.data(),
      values.size() * sizeof(uint32_t), ov::element::u32);
  OPENVINO_ASSERT(buf.valid(),
                  "GFX runtime: failed to wrap u32 kernel metadata for stage ",
                  stage_name);
  buf.owned = false;
  GpuTensor tensor;
  tensor.buf = std::move(buf);
  tensor.expected_type = ov::element::u32;
  tensor.shape = ov::Shape{values.size()};
  return tensor;
}

inline GpuTensor make_kernel_bytes_param_tensor(
    GpuBufferManager &buffer_manager, std::string_view stage_name,
    std::string_view suffix, const void *data, size_t bytes,
    const ov::element::Type &type, ov::Shape shape) {
  GpuBuffer buf = buffer_manager.wrap_const(
      make_kernel_param_key(stage_name, suffix), data, bytes, type);
  OPENVINO_ASSERT(buf.valid(),
                  "GFX runtime: failed to wrap kernel metadata for stage ",
                  stage_name);
  buf.owned = false;
  GpuTensor tensor;
  tensor.buf = std::move(buf);
  tensor.expected_type = type;
  tensor.shape = std::move(shape);
  return tensor;
}

inline GfxKernelRuntimeParamPayload make_single_bytes_runtime_param_payload(
    GpuBufferManager &buffer_manager, std::string_view stage_name,
    std::string_view suffix, const void *data, size_t bytes,
    const ov::element::Type &type, ov::Shape shape) {
  GfxKernelRuntimeParamPayload payload;
  payload.extra_inputs.push_back(make_kernel_bytes_param_tensor(
      buffer_manager, stage_name, suffix, data, bytes, type, std::move(shape)));
  return payload;
}

inline GpuTensor make_hashed_kernel_bytes_param_tensor(
    GpuBufferManager &buffer_manager, std::string_view stage_name,
    std::string_view suffix_prefix, const void *data, size_t bytes,
    const ov::element::Type &type, ov::Shape shape) {
  OPENVINO_ASSERT(data && bytes > 0,
                  "GFX runtime: kernel metadata payload is empty for stage ",
                  stage_name);
  std::ostringstream suffix;
  suffix << suffix_prefix << "/" << type.get_type_name() << "/" << bytes << "/"
         << gfx_hash_bytes(data, bytes);
  return make_kernel_bytes_param_tensor(buffer_manager, stage_name,
                                        suffix.str(), data, bytes, type,
                                        std::move(shape));
}

inline std::vector<int32_t> gfx_shape_to_i32_vector(const ov::Shape &shape) {
  std::vector<int32_t> values(std::max<size_t>(shape.size(), 1), 1);
  for (size_t i = 0; i < shape.size(); ++i) {
    values[i] = static_cast<int32_t>(shape[i]);
  }
  return values;
}

inline std::vector<int32_t> gfx_shape_strides_i32(const ov::Shape &shape) {
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

inline std::vector<uint32_t> gfx_shape_to_u32_vector(const ov::Shape &shape) {
  std::vector<uint32_t> values(std::max<size_t>(shape.size(), 1), 1);
  for (size_t i = 0; i < shape.size(); ++i) {
    values[i] = static_cast<uint32_t>(shape[i]);
  }
  return values;
}

inline std::vector<uint32_t> gfx_shape_strides_u32(const ov::Shape &shape) {
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
inline uint32_t gfx_dim_u32_at(const DimsLike &values, size_t index,
                               std::string_view stage_name,
                               std::string_view name) {
  OPENVINO_ASSERT(index < values.size(), "GFX runtime: missing ", name,
                  " dimension ", index, " for stage ", stage_name);
  return static_cast<uint32_t>(values.at(index));
}

} // namespace gfx_plugin
} // namespace ov
