// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gfx_stage_runtime_values.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <numeric>
#include <sstream>

#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/validation_util.hpp"
#include "runtime/executable_descriptor.hpp"
#include "runtime/gfx_kernel_runtime_params.hpp"
#include "runtime/gfx_profiler.hpp"
#include "runtime/gfx_runtime_value_limits.hpp"
#include "runtime/gfx_shape_utils.hpp"
#include "runtime/gpu_buffer_manager.hpp"
#include "runtime/tensor_binding_contract.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

constexpr int64_t kRuntimeSliceMetadataVersion = 1;
constexpr int64_t kRuntimeSliceKindV8 = 1;
constexpr int64_t kRuntimeSliceKindStridedSliceV1 = 2;

struct RuntimeSliceDescriptor {
  int64_t kind = 0;
  size_t input_count = 0;
  std::vector<int64_t> begin_mask;
  std::vector<int64_t> end_mask;
  std::vector<int64_t> new_axis_mask;
  std::vector<int64_t> shrink_axis_mask;
  std::vector<int64_t> ellipsis_mask;

  bool is_slice_v8() const noexcept { return kind == kRuntimeSliceKindV8; }
  bool is_strided_slice_v1() const noexcept {
    return kind == kRuntimeSliceKindStridedSliceV1;
  }
};

std::vector<size_t> compute_row_major_strides(const ov::Shape &shape) {
  std::vector<size_t> strides(shape.size(), 1);
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] =
        strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
  }
  return strides;
}

size_t range_length_i64(int64_t start, int64_t stop, int64_t step,
                        std::string_view stage_name) {
  OPENVINO_ASSERT(step != 0, "GFX MLIR: Range step must be non-zero for stage ",
                  stage_name);
  const bool forward = step > 0;
  if ((forward && start >= stop) || (!forward && start <= stop)) {
    return 0;
  }
  const uint64_t distance = forward ? static_cast<uint64_t>(stop - start)
                                    : static_cast<uint64_t>(start - stop);
  const uint64_t stride = static_cast<uint64_t>(forward ? step : -step);
  return static_cast<size_t>((distance + stride - 1) / stride);
}

int64_t normalize_slice_index(int64_t index, int64_t dim, bool is_begin) {
  if (index < 0) {
    index += dim;
  }
  if (is_begin) {
    return std::clamp<int64_t>(index, 0, dim);
  }
  return std::clamp<int64_t>(index, -1, dim);
}

std::vector<int64_t>
read_counted_i64_vector(const std::vector<int64_t> &metadata, size_t &offset,
                        std::string_view field_name,
                        std::string_view stage_name) {
  OPENVINO_ASSERT(offset < metadata.size(),
                  "GFX runtime: Slice metadata field '", field_name,
                  "' is missing for stage ", stage_name);
  const int64_t count_i64 = metadata[offset++];
  OPENVINO_ASSERT(count_i64 >= 0, "GFX runtime: Slice metadata field '",
                  field_name, "' has negative count for stage ", stage_name);
  const size_t count = static_cast<size_t>(count_i64);
  OPENVINO_ASSERT(offset + count <= metadata.size(),
                  "GFX runtime: Slice metadata field '", field_name,
                  "' is truncated for stage ", stage_name);
  std::vector<int64_t> values(
      metadata.begin() + static_cast<std::ptrdiff_t>(offset),
      metadata.begin() + static_cast<std::ptrdiff_t>(offset + count));
  offset += count;
  return values;
}

RuntimeSliceDescriptor parse_runtime_slice_descriptor(
    const RuntimeStageExecutableDescriptor &descriptor,
    std::string_view stage_name) {
  const auto &metadata = descriptor.runtime_shape_i64_metadata;
  OPENVINO_ASSERT(metadata.size() >= 3,
                  "GFX runtime: Slice runtime shape metadata is missing for ",
                  stage_name);
  OPENVINO_ASSERT(metadata[0] == kRuntimeSliceMetadataVersion,
                  "GFX runtime: unsupported Slice metadata version for ",
                  stage_name);
  RuntimeSliceDescriptor slice;
  slice.kind = metadata[1];
  OPENVINO_ASSERT(slice.is_slice_v8() || slice.is_strided_slice_v1(),
                  "GFX runtime: unsupported Slice metadata kind for ",
                  stage_name);
  OPENVINO_ASSERT(metadata[2] >= 3,
                  "GFX runtime: Slice metadata input count is invalid for ",
                  stage_name);
  slice.input_count = static_cast<size_t>(metadata[2]);
  if (slice.is_slice_v8()) {
    OPENVINO_ASSERT(
        metadata.size() == 3,
        "GFX runtime: Slice v8 metadata has unexpected trailing fields for ",
        stage_name);
    return slice;
  }

  size_t offset = 3;
  slice.begin_mask =
      read_counted_i64_vector(metadata, offset, "begin_mask", stage_name);
  slice.end_mask =
      read_counted_i64_vector(metadata, offset, "end_mask", stage_name);
  slice.new_axis_mask =
      read_counted_i64_vector(metadata, offset, "new_axis_mask", stage_name);
  slice.shrink_axis_mask =
      read_counted_i64_vector(metadata, offset, "shrink_axis_mask", stage_name);
  slice.ellipsis_mask =
      read_counted_i64_vector(metadata, offset, "ellipsis_mask", stage_name);
  OPENVINO_ASSERT(offset == metadata.size(),
                  "GFX runtime: StridedSlice metadata has trailing fields for ",
                  stage_name);
  return slice;
}

void assert_zero_mask(const std::vector<int64_t> &mask,
                      std::string_view mask_name, std::string_view stage_name) {
  OPENVINO_ASSERT(
      std::all_of(mask.begin(), mask.end(), [](int64_t v) { return v == 0; }),
      "GFX MLIR: StridedSlice ", mask_name, " is not supported for stage ",
      stage_name);
}

bool descriptor_output_contract_shape(
    const RuntimeStageExecutableDescriptor *descriptor, size_t output_idx,
    ov::Shape &shape);

void build_slice_runtime_spec(const RuntimeInputResolver &inputs,
                              const RuntimeSliceDescriptor &slice,
                              const ov::Shape &in_shape,
                              const ov::Shape &out_shape,
                              std::vector<int32_t> &starts_full,
                              std::vector<int32_t> &steps_full,
                              std::string_view stage_name) {
  const size_t rank = in_shape.size();
  OPENVINO_ASSERT(
      rank == out_shape.size(),
      "GFX MLIR: rank-changing Slice/StridedSlice is not supported for stage ",
      stage_name);
  starts_full.assign(rank, 0);
  steps_full.assign(rank, 1);

  if (slice.is_slice_v8()) {
    auto starts = inputs.i64_values(1);
    auto ends = inputs.i64_values(2);
    auto steps = inputs.i64_values(3);
    OPENVINO_ASSERT(starts,
                    "GFX MLIR: Slice starts must be available for stage ",
                    stage_name);
    OPENVINO_ASSERT(steps, "GFX MLIR: Slice steps must be available for stage ",
                    stage_name);
    std::vector<int64_t> axes;
    if (slice.input_count > 4) {
      auto runtime_axes = inputs.i64_values(4);
      OPENVINO_ASSERT(runtime_axes,
                      "GFX MLIR: Slice axes must be available for stage ",
                      stage_name);
      axes = std::move(*runtime_axes);
    } else {
      axes.resize(starts->size());
      std::iota(axes.begin(), axes.end(), 0);
    }
    OPENVINO_ASSERT(
        starts->size() == steps->size() && starts->size() == axes.size(),
        "GFX MLIR: Slice starts/ends/steps/axes size mismatch for stage ",
        stage_name);
    if (ends.has_value()) {
      OPENVINO_ASSERT(
          starts->size() == ends->size(),
          "GFX MLIR: Slice starts/ends/steps/axes size mismatch for stage ",
          stage_name);
    }
    for (size_t i = 0; i < axes.size(); ++i) {
      int64_t axis = axes[i];
      if (axis < 0) {
        axis += static_cast<int64_t>(rank);
      }
      OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < rank,
                      "GFX MLIR: Slice axis out of range for stage ",
                      stage_name);
      OPENVINO_ASSERT((*steps)[i] != 0,
                      "GFX MLIR: Slice zero step is not supported for stage ",
                      stage_name);
      const auto dim =
          static_cast<int64_t>(in_shape[static_cast<size_t>(axis)]);
      starts_full[static_cast<size_t>(axis)] =
          static_cast<int32_t>(normalize_slice_index((*starts)[i], dim, true));
      steps_full[static_cast<size_t>(axis)] = static_cast<int32_t>((*steps)[i]);
    }
    return;
  }

  OPENVINO_ASSERT(slice.is_strided_slice_v1(),
                  "GFX runtime: expected Slice/StridedSlice descriptor for ",
                  stage_name);
  assert_zero_mask(slice.new_axis_mask, "new_axis_mask", stage_name);
  assert_zero_mask(slice.shrink_axis_mask, "shrink_axis_mask", stage_name);
  assert_zero_mask(slice.ellipsis_mask, "ellipsis_mask", stage_name);

  auto begin = inputs.i64_values(1);
  OPENVINO_ASSERT(begin,
                  "GFX MLIR: StridedSlice begin must be available for stage ",
                  stage_name);
  auto end = inputs.i64_values(2);
  std::vector<int64_t> strides(rank, 1);
  if (slice.input_count > 3) {
    auto values = inputs.i64_values(3);
    OPENVINO_ASSERT(
        values, "GFX MLIR: StridedSlice strides must be available for stage ",
        stage_name);
    OPENVINO_ASSERT(values->size() <= rank,
                    "GFX MLIR: StridedSlice strides rank mismatch for stage ",
                    stage_name);
    std::copy(values->begin(), values->end(), strides.begin());
  }
  OPENVINO_ASSERT(
      begin->size() <= rank && (!end.has_value() || end->size() <= rank),
      "GFX MLIR: StridedSlice begin/end rank mismatch for stage ", stage_name);
  for (size_t axis = 0; axis < rank; ++axis) {
    const auto dim = static_cast<int64_t>(in_shape[axis]);
    const bool masked_begin =
        axis < slice.begin_mask.size() && slice.begin_mask[axis] != 0;
    const bool masked_end =
        axis < slice.end_mask.size() && slice.end_mask[axis] != 0;
    const int64_t step = strides[axis];
    OPENVINO_ASSERT(
        step != 0,
        "GFX MLIR: StridedSlice zero step is not supported for stage ",
        stage_name);
    int64_t start = axis < begin->size() ? (*begin)[axis] : 0;
    int64_t finish = end.has_value() && axis < end->size() ? (*end)[axis] : dim;
    start = masked_begin ? (step < 0 ? dim - 1 : 0)
                         : normalize_slice_index(start, dim, true);
    finish = masked_end ? (step < 0 ? -1 : dim)
                        : normalize_slice_index(finish, dim, false);
    (void)finish;
    starts_full[axis] = static_cast<int32_t>(start);
    steps_full[axis] = static_cast<int32_t>(step);
  }
}

bool slice_requires_runtime_indexing(const RuntimeInputResolver &inputs,
                                     const RuntimeSliceDescriptor &slice) {
  if (slice.is_slice_v8()) {
    auto starts = inputs.i64_values(1);
    if (!starts) {
      return true;
    }
    auto steps = inputs.i64_values(3);
    if (!steps) {
      return true;
    }
    if (slice.input_count > 4 && !inputs.i64_values(4)) {
      return true;
    }
    return std::any_of(steps->begin(), steps->end(),
                       [](int64_t step) { return step < 0; });
  }
  if (slice.is_strided_slice_v1()) {
    auto begin = inputs.i64_values(1);
    if (!begin) {
      return true;
    }
    if (!inputs.i64_values(2)) {
      return true;
    }
    if (slice.input_count <= 3) {
      return false;
    }
    auto steps = inputs.i64_values(3);
    if (!steps) {
      return true;
    }
    return std::any_of(steps->begin(), steps->end(),
                       [](int64_t step) { return step < 0; });
  }
  return false;
}

ov::Shape infer_slice_output_shape(
    const RuntimeInputResolver &inputs,
    const RuntimeStageExecutableDescriptor &descriptor,
    const RuntimeSliceDescriptor &slice, const ov::Shape &in_shape,
    const std::vector<GpuTensor *> &outputs, std::string_view stage_name) {
  if (!outputs.empty() && outputs.front() && !outputs.front()->shape.empty()) {
    return outputs.front()->shape;
  }
  ov::Shape descriptor_shape;
  if (descriptor_output_contract_shape(&descriptor, 0, descriptor_shape)) {
    return descriptor_shape;
  }

  const size_t rank = in_shape.size();
  ov::Shape out_shape;
  if (slice.is_slice_v8()) {
    auto starts = inputs.i64_values(1);
    auto ends = inputs.i64_values(2);
    auto steps = inputs.i64_values(3);
    std::optional<std::vector<int64_t>> axes;
    if (slice.input_count > 4) {
      axes = inputs.i64_values(4);
    }
    if (starts && ends && steps) {
      out_shape = in_shape;
      std::vector<int64_t> axes_values;
      if (axes) {
        axes_values = *axes;
      } else {
        axes_values.resize(starts->size());
        std::iota(axes_values.begin(), axes_values.end(), 0);
      }
      OPENVINO_ASSERT(
          starts->size() == ends->size() && starts->size() == steps->size() &&
              starts->size() == axes_values.size(),
          "GFX MLIR: Slice starts/ends/steps/axes size mismatch for stage ",
          stage_name);
      for (size_t i = 0; i < axes_values.size(); ++i) {
        int64_t axis = axes_values[i];
        if (axis < 0) {
          axis += static_cast<int64_t>(rank);
        }
        OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < rank,
                        "GFX MLIR: Slice axis out of range for stage ",
                        stage_name);
        const auto dim =
            static_cast<int64_t>(in_shape[static_cast<size_t>(axis)]);
        const int64_t step = (*steps)[i];
        OPENVINO_ASSERT(step > 0,
                        "GFX MLIR: Slice only supports positive steps for "
                        "static shape inference for stage ",
                        stage_name);
        const int64_t start = normalize_slice_index((*starts)[i], dim, true);
        const int64_t finish = normalize_slice_index((*ends)[i], dim, false);
        const int64_t extent = std::max<int64_t>(0, finish - start);
        out_shape[static_cast<size_t>(axis)] =
            static_cast<size_t>((extent + step - 1) / step);
      }
    }
  } else if (slice.is_strided_slice_v1()) {
    assert_zero_mask(slice.new_axis_mask, "new_axis_mask", stage_name);
    assert_zero_mask(slice.shrink_axis_mask, "shrink_axis_mask", stage_name);
    assert_zero_mask(slice.ellipsis_mask, "ellipsis_mask", stage_name);

    auto begin = inputs.i64_values(1);
    auto end = inputs.i64_values(2);
    std::optional<std::vector<int64_t>> strides;
    if (slice.input_count > 3) {
      strides = inputs.i64_values(3);
    }
    if (begin && end) {
      OPENVINO_ASSERT(
          begin->size() <= rank && end->size() <= rank,
          "GFX MLIR: StridedSlice begin/end rank mismatch for stage ",
          stage_name);
      if (strides) {
        OPENVINO_ASSERT(
            strides->size() <= rank,
            "GFX MLIR: StridedSlice strides rank mismatch for stage ",
            stage_name);
      }
      out_shape.reserve(rank);
      for (size_t axis = 0; axis < rank; ++axis) {
        const auto dim = static_cast<int64_t>(in_shape[axis]);
        const int64_t step =
            strides && axis < strides->size() ? (*strides)[axis] : 1;
        OPENVINO_ASSERT(
            step != 0,
            "GFX MLIR: StridedSlice zero step is not supported for stage ",
            stage_name);
        const bool masked_begin =
            axis < slice.begin_mask.size() && slice.begin_mask[axis] != 0;
        const bool masked_end =
            axis < slice.end_mask.size() && slice.end_mask[axis] != 0;
        int64_t start = axis < begin->size() ? (*begin)[axis] : 0;
        int64_t finish = axis < end->size() ? (*end)[axis] : dim;
        start = masked_begin ? (step < 0 ? dim - 1 : 0)
                             : normalize_slice_index(start, dim, true);
        finish = masked_end ? (step < 0 ? -1 : dim)
                            : normalize_slice_index(finish, dim, false);
        const int64_t extent = step > 0 ? std::max<int64_t>(0, finish - start)
                                        : std::max<int64_t>(0, start - finish);
        const int64_t stride = step > 0 ? step : -step;
        out_shape.push_back(
            static_cast<size_t>((extent + stride - 1) / stride));
      }
    }
  }
  return out_shape;
}

bool slice_is_runtime_linear_view(const ov::Shape &in_shape,
                                  const ov::Shape &out_shape,
                                  const std::vector<int32_t> &starts_full,
                                  const std::vector<int32_t> &steps_full) {
  const size_t rank = in_shape.size();
  const auto in_stride = compute_row_major_strides(in_shape);
  std::vector<size_t> out_stride(rank, 1);
  for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
    out_stride[static_cast<size_t>(i)] =
        out_stride[static_cast<size_t>(i + 1)] *
        out_shape[static_cast<size_t>(i + 1)];
  }
  for (size_t i = 0; i < rank; ++i) {
    if (starts_full[i] != 0 || steps_full[i] != 1) {
      return false;
    }
    if (out_shape[i] <= 1) {
      continue;
    }
    if (out_stride[i] != in_stride[i]) {
      return false;
    }
  }
  return true;
}

bool transpose_is_runtime_linear_view(const ov::Shape &in_shape,
                                      const ov::Shape &out_shape,
                                      const std::vector<int64_t> &permutation,
                                      std::string_view stage_name) {
  const auto in_stride = compute_row_major_strides(in_shape);
  const auto out_stride = compute_row_major_strides(out_shape);
  for (size_t i = 0; i < out_shape.size(); ++i) {
    if (out_shape[i] <= 1) {
      continue;
    }
    const int64_t axis = permutation[i];
    OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < in_shape.size(),
                    "GFX MLIR: Transpose perm out of range for stage ",
                    stage_name);
    if (out_stride[i] != in_stride[static_cast<size_t>(axis)]) {
      return false;
    }
  }
  return true;
}

ov::Shape resolve_primary_output_shape(
    const RuntimeStageExecutableDescriptor &descriptor,
    const std::vector<GpuTensor *> &outputs) {
  if (!outputs.empty() && outputs.front() && !outputs.front()->shape.empty()) {
    return outputs.front()->shape;
  }
  ov::Shape descriptor_shape;
  if (descriptor_output_contract_shape(&descriptor, 0, descriptor_shape)) {
    return descriptor_shape;
  }
  return {};
}

bool descriptor_input_contract_shape(
    const RuntimeStageExecutableDescriptor *descriptor, size_t input_idx,
    ov::Shape &shape) {
  if (!descriptor || input_idx >= descriptor->input_bindings.size()) {
    return false;
  }
  return parse_static_shape_contract(
      descriptor->input_bindings[input_idx].partial_shape, shape);
}

bool descriptor_output_contract_shape(
    const RuntimeStageExecutableDescriptor *descriptor, size_t output_idx,
    ov::Shape &shape) {
  if (!descriptor || output_idx >= descriptor->output_bindings.size()) {
    return false;
  }
  return parse_static_shape_contract(
      descriptor->output_bindings[output_idx].partial_shape, shape);
}

ov::element::Type descriptor_output_contract_type(
    const RuntimeStageExecutableDescriptor &descriptor, size_t output_idx) {
  if (output_idx >= descriptor.output_bindings.size()) {
    return ov::element::dynamic;
  }
  return element_type_from_contract(
      descriptor.output_bindings[output_idx].element_type);
}

const RuntimeStageExecutableDescriptor &
require_descriptor_contract(const RuntimeInputResolver &inputs,
                            std::string_view rule,
                            std::string_view stage_name) {
  OPENVINO_ASSERT(inputs.descriptor,
                  "GFX runtime: descriptor-owned runtime shape rule '", rule,
                  "' requires a compiler runtime descriptor for ", stage_name);
  return *inputs.descriptor;
}

ov::element::Type descriptor_required_output_type(
    const RuntimeStageExecutableDescriptor &descriptor, size_t output_idx,
    std::string_view stage_name) {
  auto type = descriptor_output_contract_type(descriptor, output_idx);
  OPENVINO_ASSERT(type != ov::element::dynamic,
                  "GFX runtime: descriptor output type is missing for ",
                  stage_name);
  return type;
}

ov::Shape descriptor_required_output_shape(
    const RuntimeStageExecutableDescriptor &descriptor, size_t output_idx,
    std::string_view stage_name, std::string_view rule) {
  ov::Shape shape;
  OPENVINO_ASSERT(
      descriptor_output_contract_shape(&descriptor, output_idx, shape),
      "GFX runtime: descriptor-owned runtime shape rule '", rule,
      "' requires static output shape contract for ", stage_name);
  return shape;
}

size_t
descriptor_input_count(const RuntimeStageExecutableDescriptor &descriptor,
                       const RuntimeInputResolver &inputs) {
  if (!descriptor.input_bindings.empty()) {
    return descriptor.input_bindings.size();
  }
  return inputs.inputs ? inputs.inputs->size() : 0;
}

int64_t descriptor_runtime_shape_metadata_at(
    const RuntimeStageExecutableDescriptor &descriptor, size_t idx,
    std::string_view stage_name, std::string_view name) {
  OPENVINO_ASSERT(idx < descriptor.runtime_shape_i64_metadata.size(),
                  "GFX runtime: descriptor-owned runtime shape metadata '",
                  name, "' is missing for ", stage_name);
  return descriptor.runtime_shape_i64_metadata[idx];
}

uint64_t shape_size_u64(const ov::Shape &shape) {
  return static_cast<uint64_t>(ov::shape_size(shape));
}

int64_t descriptor_softmax_axis_from_metadata(const ov::Shape &shape,
                                              uint64_t rows,
                                              uint64_t axis_len,
                                              uint64_t inner) {
  uint64_t prefix = 1;
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    uint64_t suffix = 1;
    for (size_t i = axis + 1; i < shape.size(); ++i) {
      suffix *= static_cast<uint64_t>(shape[i]);
    }
    if (prefix * suffix == rows &&
        static_cast<uint64_t>(shape[axis]) == axis_len && suffix == inner) {
      return static_cast<int64_t>(axis);
    }
    prefix *= static_cast<uint64_t>(shape[axis]);
  }
  return -1;
}

ov::AxisSet
descriptor_reduce_axes(const RuntimeStageExecutableDescriptor &descriptor,
                       const ov::Shape &shape, std::string_view stage_name) {
  const auto rank = static_cast<int64_t>(shape.size());
  ov::AxisSet axes;
  for (auto axis : descriptor.runtime_param_i64_metadata) {
    if (axis < 0) {
      axis += rank;
    }
    OPENVINO_ASSERT(axis >= 0 && axis < rank,
                    "GFX runtime: Reduce RuntimeParams descriptor axis out of "
                    "range for ",
                    stage_name);
    axes.insert(static_cast<size_t>(axis));
  }
  OPENVINO_ASSERT(!axes.empty(),
                  "GFX runtime: Reduce RuntimeParams descriptor axes are "
                  "empty for ",
                  stage_name);
  return axes;
}

} // namespace

ov::Shape RuntimeInputResolver::shape(size_t idx) const {
  if (inputs && idx < inputs->size() && (*inputs)[idx] &&
      !(*inputs)[idx]->shape.empty()) {
    return (*inputs)[idx]->shape;
  }
  ov::Shape descriptor_shape;
  if (descriptor_input_contract_shape(descriptor, idx, descriptor_shape)) {
    return descriptor_shape;
  }
  return {};
}

bool RuntimeInputResolver::shape_known(size_t idx, ov::Shape &out_shape,
                                       RuntimeInputShapePolicy policy) const {
  if (inputs && idx < inputs->size() && (*inputs)[idx] &&
      !(*inputs)[idx]->shape.empty()) {
    out_shape = (*inputs)[idx]->shape;
    return true;
  }
  if (descriptor_input_contract_shape(descriptor, idx, out_shape)) {
    return true;
  }
  (void)policy;
  return false;
}

GpuTensor *RuntimeInputResolver::tensor(size_t input_idx) const {
  GpuTensor *t =
      inputs && input_idx < inputs->size() ? (*inputs)[input_idx] : nullptr;
  if (t && t->buf.valid()) {
    return t;
  }
  if (const_buffers && const_buffer_present &&
      input_idx < const_buffers->size() &&
      input_idx < const_buffer_present->size() &&
      (*const_buffer_present)[input_idx] &&
      (*const_buffers)[input_idx].buf.valid()) {
    return const_cast<GpuTensor *>(&(*const_buffers)[input_idx]);
  }
  return nullptr;
}

std::optional<std::vector<int64_t>>
RuntimeInputResolver::i64_values(size_t input_idx) const {
  if (inputs && input_idx < inputs->size() && (*inputs)[input_idx] &&
      !(*inputs)[input_idx]->i64_values.empty()) {
    return (*inputs)[input_idx]->i64_values;
  }
  if (const_buffers && const_buffer_present &&
      input_idx < const_buffers->size() &&
      input_idx < const_buffer_present->size() &&
      (*const_buffer_present)[input_idx] &&
      !(*const_buffers)[input_idx].i64_values.empty()) {
    return (*const_buffers)[input_idx].i64_values;
  }
  return std::nullopt;
}

void RuntimeInputResolver::ensure_output_shape(size_t output_idx,
                                               GpuTensor *out) const {
  if (!out || !out->shape.empty()) {
    return;
  }
  ov::Shape descriptor_shape;
  if (descriptor_output_contract_shape(descriptor, output_idx,
                                       descriptor_shape)) {
    out->shape = std::move(descriptor_shape);
  }
}

std::optional<RuntimeReduceInfo> runtime_reduce_info_from_descriptor(
    const RuntimeStageExecutableDescriptor &descriptor,
    const ov::Shape &input_shape, std::string_view stage_name) {
  if (descriptor.runtime_param_payload_kind !=
      RuntimeParamDescriptorPayloadKind::Reduce) {
    return std::nullopt;
  }
  OPENVINO_ASSERT(descriptor.runtime_param_reduce_keep_dims_valid,
                  "GFX runtime: Reduce descriptor keep_dims metadata is "
                  "missing for ",
                  stage_name);
  return RuntimeReduceInfo{
      descriptor_reduce_axes(descriptor, input_shape, stage_name),
      descriptor.runtime_param_reduce_keep_dims};
}

RuntimeReduceDispatchPlan runtime_reduce_dispatch_from_descriptor(
    const RuntimeStageExecutableDescriptor &descriptor,
    std::string_view stage_name) {
  RuntimeReduceDispatchPlan plan;
  if (descriptor.runtime_param_payload_kind !=
      RuntimeParamDescriptorPayloadKind::Reduce) {
    return plan;
  }
  OPENVINO_ASSERT(!descriptor.entry_point.empty(),
                  "GFX runtime: Reduce dispatch entry point is missing from "
                  "descriptor for ",
                  stage_name);
  OPENVINO_ASSERT(descriptor.launch_plan.valid,
                  "GFX runtime: Reduce dispatch launch plan is missing from "
                  "descriptor for ",
                  stage_name);
  OPENVINO_ASSERT(descriptor.launch_plan.scalar_args.size() >= 3,
                  "GFX runtime: Reduce dispatch scalar ABI is incomplete in "
                  "descriptor for ",
                  stage_name);
  OPENVINO_ASSERT(descriptor.launch_plan.scalar_args[2] >= 0,
                  "GFX runtime: Reduce dispatch op code is invalid in "
                  "descriptor for ",
                  stage_name);
  plan.entry_point = descriptor.entry_point;
  plan.op_code = static_cast<uint32_t>(descriptor.launch_plan.scalar_args[2]);
  plan.compiler_scalar_args = descriptor.launch_plan.scalar_args;
  plan.available = true;
  return plan;
}

ov::Shape compute_binary_broadcast_shape(const ov::Shape &lhs,
                                         const ov::Shape &rhs,
                                         std::string_view stage_name) {
  const size_t rank = std::max(lhs.size(), rhs.size());
  ov::Shape out(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    const size_t lhs_dim = lhs.size() > i ? lhs[lhs.size() - 1 - i] : 1;
    const size_t rhs_dim = rhs.size() > i ? rhs[rhs.size() - 1 - i] : 1;
    OPENVINO_ASSERT(lhs_dim == rhs_dim || lhs_dim == 1 || rhs_dim == 1,
                    "GFX MLIR: incompatible binary broadcast dims for stage ",
                    stage_name, " axis_from_back=", i, " lhs=", lhs,
                    " rhs=", rhs);
    out[rank - 1 - i] = std::max(lhs_dim, rhs_dim);
  }
  return out;
}

DescriptorOwnedRuntimeParamMaterialization
materialize_descriptor_owned_runtime_param_payload(
    GpuBufferManager &buffer_manager,
    const RuntimeStageExecutableDescriptor &descriptor,
    const RuntimeInputResolver &inputs, const std::vector<GpuTensor *> &outputs,
    const std::vector<int32_t> &compiler_scalar_args,
    std::string_view stage_name,
    const std::vector<size_t> *direct_input_indices) {
  const size_t runtime_param_count = descriptor.runtime_param_buffer_count;
  DescriptorOwnedRuntimeParamMaterialization materialization;
  materialization.scalar_args = compiler_scalar_args;
  materialization.descriptor_owned =
      descriptor_owns_runtime_param_payload(descriptor);

  const auto payload_kind = descriptor.runtime_param_payload_kind;
  if (payload_kind == RuntimeParamDescriptorPayloadKind::None) {
    return materialization;
  }
  OPENVINO_ASSERT(
      materialization.descriptor_owned,
      "GFX runtime: RuntimeParams payload for stage ", stage_name,
      " is not descriptor-owned; refusing request-time shape/source bridge");

  RuntimeInputResolver descriptor_inputs = inputs;

  auto direct_input_index = [&](size_t slot) -> size_t {
    return direct_input_indices && slot < direct_input_indices->size()
               ? (*direct_input_indices)[slot]
               : slot;
  };

  auto output_type = descriptor_output_contract_type(descriptor, 0);
  auto assign_output_contract = [&](const ov::Shape &output_shape) {
    RuntimeValuePlan output_plan;
    output_plan.output_shape = output_shape;
    output_plan.value_shape = output_shape;
    output_plan.output_type = output_type;
    assign_runtime_value_outputs(output_plan, outputs);
  };

  GfxKernelRuntimeParamPayload payload;
  switch (payload_kind) {
  case RuntimeParamDescriptorPayloadKind::BinaryBroadcast: {
    ov::Shape lhs_shape;
    ov::Shape rhs_shape;
    OPENVINO_ASSERT(
        descriptor_inputs.shape_known(direct_input_index(0), lhs_shape) &&
            descriptor_inputs.shape_known(direct_input_index(1), rhs_shape),
        "GFX runtime: binary RuntimeParams ABI requires descriptor-owned or "
        "tensor-owned input shapes for ",
        stage_name);
    const ov::Shape broadcast_shape =
        compute_binary_broadcast_shape(lhs_shape, rhs_shape, stage_name);
    ov::Shape output_shape;
    if (descriptor_output_contract_shape(&descriptor, 0, output_shape)) {
      OPENVINO_ASSERT(output_shape == broadcast_shape,
                      "GFX runtime: binary RuntimeParams descriptor output "
                      "shape drift for ",
                      stage_name);
    } else {
      output_shape = broadcast_shape;
    }
    const ov::Shape meta_shape =
        output_shape.empty() ? ov::Shape{1} : output_shape;
    payload = make_binary_broadcast_runtime_param_payload(
        buffer_manager, stage_name, output_shape,
        compute_broadcast_element_strides(lhs_shape, meta_shape),
        compute_broadcast_element_strides(rhs_shape, meta_shape));
    assign_output_contract(output_shape);
    break;
  }
  case RuntimeParamDescriptorPayloadKind::Broadcast: {
    ov::Shape input_shape;
    OPENVINO_ASSERT(
        descriptor_inputs.shape_known(direct_input_index(0), input_shape),
        "GFX runtime: Broadcast RuntimeParams ABI requires descriptor-owned "
        "or tensor-owned input shape for ",
        stage_name);
    const bool bidirectional_broadcast =
        !descriptor.runtime_shape_i64_metadata.empty() &&
        descriptor.runtime_shape_i64_metadata[0] != 0;
    const int64_t descriptor_rank =
        descriptor.runtime_shape_i64_metadata.size() > 1
            ? descriptor.runtime_shape_i64_metadata[1]
            : -1;
    ov::Shape output_shape;
    if (auto target = descriptor_inputs.i64_values(1)) {
      ov::Shape target_shape;
      target_shape.reserve(target->size());
      for (auto dim : *target) {
        target_shape.push_back(static_cast<size_t>(std::max<int64_t>(dim, 0)));
      }
      output_shape = bidirectional_broadcast
                         ? compute_binary_broadcast_shape(input_shape,
                                                          target_shape,
                                                          stage_name)
                         : target_shape;
    } else if (!descriptor_output_contract_shape(&descriptor, 0,
                                                 output_shape)) {
      OPENVINO_THROW(
          "GFX runtime: Broadcast RuntimeParams ABI requires descriptor-owned "
          "output shape or target-shape runtime values for ",
          stage_name);
    }
    if (descriptor_rank >= 0) {
      OPENVINO_ASSERT(static_cast<size_t>(descriptor_rank) ==
                          output_shape.size(),
                      "GFX runtime: Broadcast RuntimeParams descriptor output "
                      "rank drift for ",
                      stage_name);
    }
    ov::Shape descriptor_shape;
    if (descriptor_output_contract_shape(&descriptor, 0, descriptor_shape)) {
      OPENVINO_ASSERT(
          descriptor_shape == output_shape,
          "GFX runtime: Broadcast RuntimeParams descriptor output shape drift "
          "for ",
          stage_name);
      output_shape = descriptor_shape;
    }
    payload = make_broadcast_runtime_param_payload(buffer_manager, stage_name,
                                                   input_shape, output_shape);
    assign_output_contract(output_shape);
    break;
  }
  case RuntimeParamDescriptorPayloadKind::Select: {
    ov::Shape condition_shape;
    ov::Shape true_shape;
    ov::Shape false_shape;
    OPENVINO_ASSERT(
        descriptor_inputs.shape_known(direct_input_index(0), condition_shape) &&
            descriptor_inputs.shape_known(direct_input_index(1), true_shape) &&
            descriptor_inputs.shape_known(direct_input_index(2), false_shape),
        "GFX runtime: Select RuntimeParams ABI requires descriptor-owned or "
        "tensor-owned input shapes for ",
        stage_name);
    const auto values_shape =
        compute_binary_broadcast_shape(true_shape, false_shape, stage_name);
    const auto computed_output_shape = compute_binary_broadcast_shape(
        condition_shape, values_shape, stage_name);
    ov::Shape output_shape;
    if (descriptor_output_contract_shape(&descriptor, 0, output_shape)) {
      OPENVINO_ASSERT(output_shape == computed_output_shape,
                      "GFX runtime: Select RuntimeParams descriptor output "
                      "shape drift for ",
                      stage_name);
    } else {
      output_shape = computed_output_shape;
    }
    payload = make_select_runtime_param_payload(buffer_manager, stage_name,
                                                condition_shape, true_shape,
                                                false_shape, output_shape);
    assign_output_contract(output_shape);
    break;
  }
  case RuntimeParamDescriptorPayloadKind::Tile: {
    ov::Shape input_shape;
    OPENVINO_ASSERT(
        descriptor_inputs.shape_known(direct_input_index(0), input_shape),
        "GFX runtime: Tile RuntimeParams ABI requires descriptor-owned or "
        "tensor-owned input shape for ",
        stage_name);
    ov::Shape output_shape;
    if (!outputs.empty() && outputs.front() && !outputs.front()->shape.empty()) {
      output_shape = outputs.front()->shape;
    } else if (!descriptor_output_contract_shape(&descriptor, 0,
                                                 output_shape)) {
      auto repeats = descriptor_inputs.i64_values(1);
      OPENVINO_ASSERT(
          repeats,
          "GFX runtime: Tile RuntimeParams ABI requires descriptor-owned "
          "output shape or repeats runtime values for ",
          stage_name);
      OPENVINO_ASSERT(repeats->size() == input_shape.size(),
                      "GFX runtime: Tile RuntimeParams repeats rank mismatch "
                      "for ",
                      stage_name);
      output_shape.reserve(input_shape.size());
      for (size_t axis = 0; axis < input_shape.size(); ++axis) {
        OPENVINO_ASSERT((*repeats)[axis] > 0,
                        "GFX runtime: Tile RuntimeParams repeats must be "
                        "positive for ",
                        stage_name);
        output_shape.push_back(input_shape[axis] *
                               static_cast<size_t>((*repeats)[axis]));
      }
    }
    OPENVINO_ASSERT(input_shape.size() == output_shape.size(),
                    "GFX runtime: Tile RuntimeParams descriptor rank drift "
                    "for ",
                    stage_name);
    payload = make_tile_runtime_param_payload(buffer_manager, stage_name,
                                              input_shape, output_shape);
    assign_output_contract(output_shape);
    break;
  }
  case RuntimeParamDescriptorPayloadKind::Interpolate: {
    const auto interpolate_plan = plan_interpolate_runtime_values(
        descriptor_inputs, outputs, descriptor, stage_name);
    payload = make_interpolate_runtime_param_payload(
        buffer_manager, stage_name, interpolate_plan.input_shape,
        interpolate_plan.values.output_shape, interpolate_plan.align_corners,
        interpolate_plan.use_half_pixel, interpolate_plan.nearest_mode);
    assign_runtime_value_outputs(interpolate_plan.values, outputs);
    break;
  }
  case RuntimeParamDescriptorPayloadKind::Softmax: {
    OPENVINO_ASSERT(
        descriptor.runtime_param_i64_metadata.size() == 3,
        "GFX runtime: Softmax RuntimeParams descriptor metadata is incomplete "
        "for ",
        stage_name);
    ov::Shape input_shape;
    OPENVINO_ASSERT(
        descriptor_inputs.shape_known(direct_input_index(0), input_shape),
        "GFX runtime: Softmax RuntimeParams ABI requires descriptor-owned "
        "or tensor-owned input shape for ",
        stage_name);
    ov::Shape output_shape;
    if (descriptor_output_contract_shape(&descriptor, 0, output_shape)) {
      OPENVINO_ASSERT(input_shape == output_shape,
                      "GFX runtime: Softmax RuntimeParams descriptor output "
                      "shape drift for ",
                      stage_name);
    } else {
      output_shape = input_shape;
    }
    const auto rows =
        static_cast<uint64_t>(descriptor.runtime_param_i64_metadata[0]);
    const auto axis_len =
        static_cast<uint64_t>(descriptor.runtime_param_i64_metadata[1]);
    const auto inner =
        static_cast<uint64_t>(descriptor.runtime_param_i64_metadata[2]);
    const auto axis =
        descriptor_softmax_axis_from_metadata(input_shape, rows, axis_len, inner);
    OPENVINO_ASSERT(rows > 0 && axis_len > 0 && inner > 0 &&
                        rows * axis_len == ov::shape_size(input_shape) &&
                        axis >= 0,
                    "GFX runtime: Softmax RuntimeParams descriptor metadata "
                    "does not match tensor shape for ",
                    stage_name, ": rows=", rows, " axis_len=", axis_len,
                    " inner=", inner, " input_shape=", input_shape);
    payload = make_softmax_runtime_param_payload(buffer_manager, stage_name,
                                                 rows, axis_len, inner);
    assign_output_contract(output_shape);
    break;
  }
  case RuntimeParamDescriptorPayloadKind::Transpose: {
    ov::Shape input_shape;
    OPENVINO_ASSERT(
        descriptor_inputs.shape_known(direct_input_index(0), input_shape),
        "GFX runtime: Transpose RuntimeParams ABI requires descriptor-owned "
        "or tensor-owned input shape for ",
        stage_name);
    const auto &permutation = descriptor.runtime_param_i64_metadata;
    OPENVINO_ASSERT(
        permutation.size() == input_shape.size(),
        "GFX runtime: Transpose RuntimeParams descriptor rank mismatch for ",
        stage_name);
    ov::Shape output_shape;
    if (!descriptor_output_contract_shape(&descriptor, 0, output_shape)) {
      output_shape.assign(input_shape.size(), 0);
      for (size_t axis = 0; axis < permutation.size(); ++axis) {
        const auto input_axis = static_cast<size_t>(permutation[axis]);
        OPENVINO_ASSERT(
            permutation[axis] >= 0 && input_axis < input_shape.size(),
            "GFX runtime: Transpose RuntimeParams descriptor permutation out "
            "of range for ",
            stage_name);
        output_shape[axis] = input_shape[input_axis];
      }
    }
    OPENVINO_ASSERT(output_shape.size() == input_shape.size(),
                    "GFX runtime: Transpose RuntimeParams descriptor output "
                    "rank drift for ",
                    stage_name);
    for (size_t axis = 0; axis < permutation.size(); ++axis) {
      const auto input_axis = static_cast<size_t>(permutation[axis]);
      OPENVINO_ASSERT(
          permutation[axis] >= 0 && input_axis < input_shape.size(),
          "GFX runtime: Transpose RuntimeParams descriptor permutation out of "
          "range for ",
          stage_name);
      OPENVINO_ASSERT(
          output_shape[axis] == input_shape[input_axis],
          "GFX runtime: Transpose RuntimeParams descriptor output shape drift "
          "for ",
          stage_name);
    }
    payload = make_transpose_runtime_param_payload(
        buffer_manager, stage_name, input_shape, output_shape, permutation);
    assign_output_contract(output_shape);
    break;
  }
  case RuntimeParamDescriptorPayloadKind::Reduce: {
    ov::Shape input_shape;
    OPENVINO_ASSERT(
        descriptor_inputs.shape_known(direct_input_index(0), input_shape),
        "GFX runtime: Reduce RuntimeParams ABI requires descriptor-owned "
        "or tensor-owned input shape for ",
        stage_name);
    const auto reduce_info = runtime_reduce_info_from_descriptor(
        descriptor, input_shape, stage_name);
    OPENVINO_ASSERT(reduce_info,
                    "GFX runtime: Reduce RuntimeParams descriptor metadata is "
                    "not available for ",
                    stage_name);
    const auto reduce_plan = plan_reduce_runtime_values(
        descriptor_inputs, descriptor.op_family, *reduce_info, stage_name);
    ov::Shape output_shape = reduce_plan.values.output_shape;
    const uint32_t op_code =
        compiler_scalar_args.size() > 2
            ? static_cast<uint32_t>(compiler_scalar_args[2])
            : 0u;
    payload = make_reduce_runtime_param_payload(
        buffer_manager, stage_name, input_shape, reduce_info->axes,
        reduce_info->keep_dims, output_shape, op_code);
    assign_runtime_value_outputs(reduce_plan.values, outputs);
    break;
  }
  case RuntimeParamDescriptorPayloadKind::None:
    break;
  }

  OPENVINO_ASSERT(payload.extra_inputs.size() == runtime_param_count,
                  "GFX runtime: RuntimeParams ABI count does not match "
                  "materialized payload for ",
                  stage_name);
  materialization.available = true;
  materialization.extra_inputs = std::move(payload.extra_inputs);
  materialization.scalar_args = std::move(payload.scalar_args);
  return materialization;
}

RuntimeValuePlan plan_reshape_runtime_values(const RuntimeInputResolver &inputs,
                                             const RuntimeStageExecutableDescriptor &descriptor,
                                             std::string_view stage_name) {
  OPENVINO_ASSERT(descriptor.op_family == "Reshape",
                  "GFX runtime: Reshape runtime values require Reshape "
                  "descriptor for stage ",
                  stage_name);
  const bool special_zero =
      !descriptor.runtime_shape_i64_metadata.empty() &&
      descriptor.runtime_shape_i64_metadata[0] != 0;

  ov::Shape input_shape;
  bool input_shape_known = inputs.shape_known(
      0, input_shape, RuntimeInputShapePolicy::TensorOrStaticOrConstant);
  if (!input_shape_known) {
    input_shape_known = descriptor_input_contract_shape(&descriptor, 0, input_shape);
  }

  ov::Shape output_shape;
  if (auto pattern = inputs.i64_values(1)) {
    output_shape.reserve(pattern->size());
    int64_t infer_pos = -1;
    size_t known_product = 1;
    for (size_t i = 0; i < pattern->size(); ++i) {
      const int64_t dim = (*pattern)[i];
      if (dim == 0 && special_zero) {
        OPENVINO_ASSERT(input_shape_known,
                        "GFX runtime: Reshape special_zero requires input "
                        "shape for stage ",
                        stage_name);
        OPENVINO_ASSERT(
            i < input_shape.size(),
            "GFX MLIR: Reshape special_zero axis out of range for stage ",
            stage_name);
        output_shape.push_back(input_shape[i]);
        known_product *= input_shape[i];
      } else if (dim == -1) {
        OPENVINO_ASSERT(infer_pos < 0,
                        "GFX MLIR: Reshape has multiple -1 dims for stage ",
                        stage_name);
        infer_pos = static_cast<int64_t>(output_shape.size());
        output_shape.push_back(1);
      } else {
        OPENVINO_ASSERT(dim >= 0,
                        "GFX MLIR: Reshape target dim is invalid for stage ",
                        stage_name);
        output_shape.push_back(static_cast<size_t>(dim));
        known_product *= static_cast<size_t>(dim);
      }
    }
    if (infer_pos >= 0) {
      OPENVINO_ASSERT(input_shape_known,
                      "GFX runtime: Reshape -1 dimension requires input "
                      "shape for stage ",
                      stage_name);
      const size_t input_elems = ov::shape_size(input_shape);
      OPENVINO_ASSERT(known_product != 0 && input_elems % known_product == 0,
                      "GFX MLIR: Reshape cannot infer -1 dim for stage ",
                      stage_name);
      output_shape[static_cast<size_t>(infer_pos)] =
          input_elems / known_product;
    }
  }
  if (output_shape.empty()) {
    output_shape =
        descriptor_required_output_shape(descriptor, 0, stage_name, "reshape");
  }

  RuntimeValuePlan plan;
  plan.output_shape = std::move(output_shape);
  plan.value_shape = plan.output_shape;
  plan.output_type = descriptor_required_output_type(descriptor, 0, stage_name);
  plan.force_output_type = true;
  if (auto input_values = inputs.i64_values(0)) {
    plan.i64_values = std::move(*input_values);
    plan.has_i64_values = true;
  }
  return plan;
}

RuntimeValuePlan
plan_squeeze_unsqueeze_runtime_values(const RuntimeInputResolver &inputs,
                                      const RuntimeStageExecutableDescriptor &descriptor,
                                      std::string_view stage_name) {
  ov::Shape input_shape = inputs.shape(0);
  if (input_shape.empty()) {
    descriptor_input_contract_shape(&descriptor, 0, input_shape);
  }
  OPENVINO_ASSERT(!input_shape.empty(), "GFX MLIR: ", descriptor.op_family,
                  " input shape is unknown for stage ", stage_name);

  ov::Shape output_shape;
  if (descriptor.op_family == "Squeeze") {
    std::vector<int64_t> axes;
    if (descriptor_input_count(descriptor, inputs) > 1) {
      auto runtime_axes = inputs.i64_values(1);
      OPENVINO_ASSERT(runtime_axes,
                      "GFX runtime: Squeeze axes must be available for stage ",
                      stage_name);
      axes = std::move(*runtime_axes);
      ov::util::normalize_axes(axes, static_cast<int64_t>(input_shape.size()));
    }
    for (size_t i = 0; i < input_shape.size(); ++i) {
      const bool remove_axis =
          axes.empty() ? (input_shape[i] == 1)
                       : (std::find(axes.begin(), axes.end(),
                                    static_cast<int64_t>(i)) != axes.end());
      if (!remove_axis) {
        output_shape.push_back(input_shape[i]);
      }
    }
  } else if (descriptor.op_family == "Unsqueeze") {
    auto runtime_axes = inputs.i64_values(1);
    OPENVINO_ASSERT(runtime_axes,
                    "GFX runtime: Unsqueeze axes must be available for stage ",
                    stage_name);
    auto axes = std::move(*runtime_axes);
    ov::util::normalize_axes(
        axes, static_cast<int64_t>(input_shape.size() + axes.size()));
    std::sort(axes.begin(), axes.end());
    output_shape = input_shape;
    for (size_t i = 0; i < axes.size(); ++i) {
      output_shape.insert(
          output_shape.begin() + static_cast<std::ptrdiff_t>(axes[i]), 1);
    }
  } else {
    OPENVINO_THROW("GFX MLIR: expected Squeeze/Unsqueeze node for stage ",
                   stage_name);
  }
  if (output_shape.empty()) {
    output_shape = ov::Shape{1};
  }

  RuntimeValuePlan plan;
  plan.output_shape = std::move(output_shape);
  plan.value_shape = plan.output_shape;
  plan.output_type = descriptor_required_output_type(descriptor, 0, stage_name);
  plan.force_output_type = true;
  if (auto input_values = inputs.i64_values(0)) {
    plan.i64_values = std::move(*input_values);
    plan.has_i64_values = true;
  }
  return plan;
}

RuntimeValuePlan
plan_shape_preserving_runtime_values(const RuntimeInputResolver &inputs,
                                     const RuntimeStageExecutableDescriptor &descriptor,
                                     std::string_view stage_name) {
  ov::Shape output_shape;
  if (!inputs.shape_known(0, output_shape)) {
    output_shape = inputs.shape(0);
  }
  if (output_shape.empty()) {
    descriptor_output_contract_shape(&descriptor, 0, output_shape);
  }
  OPENVINO_ASSERT(
      !output_shape.empty(),
      "GFX MLIR: shape-preserving input shape is unknown for stage ",
      stage_name);

  RuntimeValuePlan plan;
  plan.output_shape = std::move(output_shape);
  plan.value_shape = plan.output_shape;
  plan.output_type = descriptor_required_output_type(descriptor, 0, stage_name);
  plan.force_output_type = true;
  return plan;
}

RuntimeValuePlan plan_shapeof_runtime_values(const RuntimeInputResolver &inputs,
                                             std::string_view stage_name) {
  const auto &descriptor =
      require_descriptor_contract(inputs, "shape_of", stage_name);
  ov::Shape in_shape = inputs.shape(0);
  if (in_shape.empty()) {
    OPENVINO_THROW("GFX runtime: ShapeOf input shape is unknown for stage ",
                   stage_name);
  }
  const auto output_type =
      descriptor_required_output_type(descriptor, 0, stage_name);
  OPENVINO_ASSERT(output_type == ov::element::i32 ||
                      output_type == ov::element::i64,
                  "GFX runtime: ShapeOf descriptor output must be i32/i64 for ",
                  stage_name);

  RuntimeValuePlan plan;
  plan.output_shape = ov::Shape{in_shape.size()};
  plan.value_shape = plan.output_shape;
  plan.output_type = output_type;
  plan.force_output_type = true;
  plan.i64_values.reserve(in_shape.size());
  for (auto dim : in_shape) {
    plan.i64_values.push_back(static_cast<int64_t>(dim));
  }
  plan.has_i64_values = true;
  return plan;
}

RuntimeValuePlan
plan_broadcast_runtime_values(const RuntimeInputResolver &inputs,
                              const ov::Shape &input_shape,
                              std::string_view stage_name) {
  const auto &descriptor =
      require_descriptor_contract(inputs, "broadcast", stage_name);
  const bool bidirectional_broadcast =
      !descriptor.runtime_shape_i64_metadata.empty() &&
      descriptor.runtime_shape_i64_metadata[0] != 0;
  const int64_t descriptor_rank =
      descriptor.runtime_shape_i64_metadata.size() > 1
          ? descriptor.runtime_shape_i64_metadata[1]
          : -1;

  ov::Shape out_shape;
  if (auto target = inputs.i64_values(1)) {
    ov::Shape target_shape;
    target_shape.reserve(target->size());
    for (auto dim : *target) {
      target_shape.push_back(static_cast<size_t>(std::max<int64_t>(dim, 0)));
    }
    out_shape = bidirectional_broadcast
                    ? compute_binary_broadcast_shape(input_shape, target_shape,
                                                     stage_name)
                    : target_shape;
  }
  if (out_shape.empty()) {
    out_shape = descriptor_required_output_shape(descriptor, 0, stage_name,
                                                 "broadcast");
  }
  if (descriptor_rank >= 0) {
    OPENVINO_ASSERT(static_cast<size_t>(descriptor_rank) == out_shape.size(),
                    "GFX runtime: Broadcast descriptor output rank drift for ",
                    stage_name);
  }
  OPENVINO_ASSERT(input_shape.size() <= out_shape.size(),
                  "GFX runtime: Broadcast input rank exceeds output rank for ",
                  stage_name);

  RuntimeValuePlan plan;
  plan.output_shape = std::move(out_shape);
  plan.value_shape = input_shape;
  plan.output_type = descriptor_required_output_type(descriptor, 0, stage_name);
  if (auto input_values = inputs.i64_values(0)) {
    plan.i64_values = std::move(*input_values);
    plan.has_i64_values = true;
  }
  return plan;
}

RuntimeValuePlan plan_convert_runtime_values(const RuntimeInputResolver &inputs,
                                             const RuntimeStageExecutableDescriptor &descriptor,
                                             std::string_view stage_name) {
  ov::Shape in_shape = inputs.shape(0);
  if (in_shape.empty()) {
    descriptor_input_contract_shape(&descriptor, 0, in_shape);
  }
  OPENVINO_ASSERT(!in_shape.empty(),
                  "GFX MLIR: Convert input shape is unknown for stage ",
                  stage_name);
  RuntimeValuePlan plan;
  plan.output_shape = in_shape;
  plan.value_shape = std::move(in_shape);
  plan.output_type = descriptor_required_output_type(descriptor, 0, stage_name);
  plan.force_output_type = true;
  if (auto input_values = inputs.i64_values(0)) {
    plan.i64_values = std::move(*input_values);
    plan.has_i64_values = true;
  }
  return plan;
}

RuntimeValuePlan plan_range_runtime_values(const RuntimeInputResolver &inputs,
                                           std::string_view stage_name) {
  const auto &descriptor =
      require_descriptor_contract(inputs, "range", stage_name);
  auto start_values = inputs.i64_values(0);
  auto stop_values = inputs.i64_values(1);
  auto step_values = inputs.i64_values(2);

  RuntimeValuePlan plan;
  if (start_values && stop_values && step_values && start_values->size() == 1 &&
      stop_values->size() == 1 && step_values->size() == 1) {
    const int64_t start = (*start_values)[0];
    const int64_t stop = (*stop_values)[0];
    const int64_t step = (*step_values)[0];
    const size_t len = range_length_i64(start, stop, step, stage_name);
    plan.output_shape = ov::Shape{len};
    plan.i64_values.reserve(len);
    for (size_t ri = 0; ri < len; ++ri) {
      plan.i64_values.push_back(start + static_cast<int64_t>(ri) * step);
    }
    plan.has_i64_values = true;
  } else {
    plan.output_shape =
        descriptor_required_output_shape(descriptor, 0, stage_name, "range");
  }

  plan.value_shape = plan.output_shape;
  plan.output_type = descriptor_required_output_type(descriptor, 0, stage_name);
  plan.force_output_type = true;
  return plan;
}

RuntimeSelectPlan plan_select_runtime_values(const RuntimeInputResolver &inputs,
                                             std::string_view stage_name) {
  const auto &descriptor =
      require_descriptor_contract(inputs, "select", stage_name);
  RuntimeSelectPlan plan;
  if (!inputs.shape_known(0, plan.condition_shape) ||
      !inputs.shape_known(1, plan.true_shape) ||
      !inputs.shape_known(2, plan.false_shape)) {
    return plan;
  }

  const ov::Shape data_shape = compute_binary_broadcast_shape(
      plan.true_shape, plan.false_shape, stage_name);
  plan.values.output_shape = compute_binary_broadcast_shape(
      plan.condition_shape, data_shape, stage_name);
  ov::Shape descriptor_shape;
  if (descriptor_output_contract_shape(&descriptor, 0, descriptor_shape)) {
    OPENVINO_ASSERT(descriptor_shape == plan.values.output_shape,
                    "GFX runtime: Select descriptor output shape drift for ",
                    stage_name);
    plan.values.output_shape = descriptor_shape;
  }
  plan.values.value_shape = plan.values.output_shape;
  plan.values.output_type =
      descriptor_required_output_type(descriptor, 0, stage_name);
  plan.available = true;
  return plan;
}

RuntimeReducePlan plan_reduce_runtime_values(
    const RuntimeInputResolver &inputs, std::string_view reduce_type,
    const RuntimeReduceInfo &reduce_info, std::string_view stage_name) {
  const auto &descriptor =
      require_descriptor_contract(inputs, "reduce", stage_name);
  RuntimeReducePlan plan;
  plan.input_shape = inputs.shape(0);
  OPENVINO_ASSERT(!plan.input_shape.empty(),
                  "GFX runtime: Reduce input shape is unknown for stage ",
                  stage_name);
  const size_t rank = plan.input_shape.size();
  OPENVINO_ASSERT(
      rank <= 8,
      "GFX runtime: Reduce rank exceeds kernel metadata capacity for stage ",
      stage_name);

  plan.values.output_shape.reserve(rank);
  for (size_t i = 0; i < rank; ++i) {
    const bool reduced = reduce_info.axes.count(i) != 0;
    if (reduced) {
      if (reduce_info.keep_dims) {
        plan.values.output_shape.push_back(1);
      }
      continue;
    }
    plan.values.output_shape.push_back(plan.input_shape[i]);
  }
  if (plan.values.output_shape.empty()) {
    plan.values.output_shape = ov::Shape{1};
  }

  ov::Shape descriptor_shape;
  if (descriptor_output_contract_shape(&descriptor, 0, descriptor_shape)) {
    OPENVINO_ASSERT(descriptor_shape == plan.values.output_shape,
                    "GFX runtime: Reduce descriptor output shape drift for ",
                    stage_name);
    plan.values.output_shape = descriptor_shape;
  }
  plan.values.value_shape = plan.values.output_shape;
  plan.values.output_type =
      descriptor_required_output_type(descriptor, 0, stage_name);
  plan.values.force_output_type = true;
  if (auto input_values = inputs.i64_values(0)) {
    if (auto reduced_values = compute_reduce_i64_values(
            reduce_type, *input_values, plan.input_shape, reduce_info,
            plan.values.output_shape)) {
      plan.values.i64_values = std::move(*reduced_values);
      plan.values.has_i64_values = true;
    }
  }
  plan.available = true;
  return plan;
}

RuntimeTilePlan
plan_tile_runtime_values(const RuntimeInputResolver &inputs,
                         const std::vector<GpuTensor *> &outputs,
                         std::string_view stage_name) {
  RuntimeTilePlan plan;
  if (outputs.empty() || !outputs.front()) {
    return plan;
  }
  plan.input_shape = inputs.shape(0);
  if (plan.input_shape.empty()) {
    return plan;
  }
  if (!outputs.front()->shape.empty()) {
    plan.output_shape = outputs.front()->shape;
  } else if (inputs.descriptor &&
             descriptor_output_contract_shape(inputs.descriptor, 0,
                                              plan.output_shape)) {
  } else if (auto repeats = inputs.i64_values(1)) {
    OPENVINO_ASSERT(repeats->size() == plan.input_shape.size(),
                    "GFX MLIR: Tile repeats rank mismatch for stage ",
                    stage_name);
    plan.output_shape.reserve(plan.input_shape.size());
    for (size_t axis = 0; axis < plan.input_shape.size(); ++axis) {
      OPENVINO_ASSERT((*repeats)[axis] > 0,
                      "GFX MLIR: Tile repeats must be positive for stage ",
                      stage_name);
      plan.output_shape.push_back(plan.input_shape[axis] *
                                  static_cast<size_t>((*repeats)[axis]));
    }
  }
  if (plan.output_shape.empty()) {
    return plan;
  }
  if (plan.input_shape.size() != plan.output_shape.size()) {
    plan.output_shape.clear();
    return plan;
  }
  plan.values.output_shape = plan.output_shape;
  plan.values.value_shape = plan.output_shape;
  if (inputs.descriptor) {
    plan.values.output_type =
        descriptor_required_output_type(*inputs.descriptor, 0, stage_name);
  } else {
    plan.values.output_type = ov::element::dynamic;
  }
  plan.scalar_args = {
      static_cast<int32_t>(ov::shape_size(plan.output_shape)),
      static_cast<int32_t>(std::max<size_t>(plan.output_shape.size(), 1))};
  plan.available = true;
  return plan;
}

RuntimeConcatPlan plan_concat_runtime_values(const RuntimeInputResolver &inputs,
                                             std::string_view stage_name) {
  const auto &descriptor =
      require_descriptor_contract(inputs, "concat", stage_name);

  RuntimeConcatPlan plan;
  const size_t input_count = descriptor_input_count(descriptor, inputs);
  plan.input_shapes.reserve(input_count);
  ov::Shape out_shape;
  for (size_t input_idx = 0; input_idx < input_count; ++input_idx) {
    const ov::Shape input_shape = inputs.shape(input_idx);
    if (input_shape.empty()) {
      OPENVINO_THROW("GFX runtime: Concat input shape is unknown for stage ",
                     stage_name);
    }
    if (out_shape.empty()) {
      out_shape = input_shape;
    } else {
      OPENVINO_ASSERT(input_shape.size() == out_shape.size(),
                      "GFX runtime: Concat rank mismatch for stage ",
                      stage_name);
    }
    plan.input_shapes.push_back(input_shape);
  }
  OPENVINO_ASSERT(!out_shape.empty(),
                  "GFX runtime: Concat has no resolved inputs for stage ",
                  stage_name);

  plan.axis_norm = normalize_axis(
      descriptor_runtime_shape_metadata_at(descriptor, 0, stage_name, "axis"),
      out_shape.size(), "GFX runtime: Concat");
  size_t axis_total = 0;
  for (const auto &input_shape : plan.input_shapes) {
    OPENVINO_ASSERT(input_shape.size() == out_shape.size(),
                    "GFX runtime: Concat rank mismatch for stage ", stage_name);
    for (size_t dim = 0; dim < out_shape.size(); ++dim) {
      if (static_cast<int64_t>(dim) == plan.axis_norm) {
        continue;
      }
      OPENVINO_ASSERT(input_shape[dim] == out_shape[dim],
                      "GFX runtime: Concat non-axis dim mismatch for stage ",
                      stage_name);
    }
    axis_total += input_shape[static_cast<size_t>(plan.axis_norm)];
  }
  out_shape[static_cast<size_t>(plan.axis_norm)] = axis_total;

  ov::Shape descriptor_shape;
  if (descriptor_output_contract_shape(&descriptor, 0, descriptor_shape)) {
    OPENVINO_ASSERT(descriptor_shape == out_shape,
                    "GFX runtime: Concat descriptor output shape drift for ",
                    stage_name);
    out_shape = descriptor_shape;
  }

  plan.values.output_shape = out_shape;
  plan.values.value_shape = out_shape;
  plan.values.output_type =
      descriptor_required_output_type(descriptor, 0, stage_name);

  std::vector<std::vector<int64_t>> concat_values;
  concat_values.reserve(input_count);
  bool all_values_resolved = true;
  for (size_t input_idx = 0; input_idx < input_count; ++input_idx) {
    auto values = inputs.i64_values(input_idx);
    if (!values.has_value()) {
      all_values_resolved = false;
      break;
    }
    concat_values.push_back(std::move(*values));
  }
  if (all_values_resolved) {
    const size_t axis = static_cast<size_t>(plan.axis_norm);
    const size_t inner = std::accumulate(
        out_shape.begin() + static_cast<std::ptrdiff_t>(axis + 1),
        out_shape.end(), size_t{1}, std::multiplies<size_t>());
    const size_t outer =
        std::accumulate(out_shape.begin(),
                        out_shape.begin() + static_cast<std::ptrdiff_t>(axis),
                        size_t{1}, std::multiplies<size_t>());
    std::vector<int64_t> values;
    values.reserve(ov::shape_size(out_shape));
    for (size_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
      for (size_t input_idx = 0; input_idx < plan.input_shapes.size();
           ++input_idx) {
        const auto &input_shape = plan.input_shapes[input_idx];
        const size_t chunk = input_shape[axis] * inner;
        const size_t offset = outer_idx * chunk;
        if (offset + chunk > concat_values[input_idx].size()) {
          all_values_resolved = false;
          break;
        }
        values.insert(values.end(),
                      concat_values[input_idx].begin() +
                          static_cast<std::ptrdiff_t>(offset),
                      concat_values[input_idx].begin() +
                          static_cast<std::ptrdiff_t>(offset + chunk));
      }
      if (!all_values_resolved) {
        break;
      }
    }
    if (all_values_resolved && values.size() == ov::shape_size(out_shape)) {
      plan.values.i64_values = std::move(values);
      plan.values.has_i64_values = true;
    }
  }
  plan.available = true;
  return plan;
}

RuntimeGatherPlan plan_gather_runtime_values(const RuntimeInputResolver &inputs,
                                             const RuntimeStageExecutableDescriptor &descriptor,
                                             std::string_view stage_name) {
  OPENVINO_ASSERT(descriptor.op_family == "Gather",
                  "GFX runtime: Gather runtime values require Gather "
                  "descriptor for stage ",
                  stage_name);
  if (!descriptor.runtime_shape_i64_metadata.empty()) {
    OPENVINO_ASSERT(descriptor.runtime_shape_i64_metadata[0] == 0,
                    "GFX MLIR: Gather batch_dims not supported for stage ",
                    stage_name);
  }

  RuntimeGatherPlan plan;
  const bool data_shape_known = inputs.shape_known(
      0, plan.data_shape, RuntimeInputShapePolicy::TensorOrStaticOrConstant);
  const bool idx_shape_known = inputs.shape_known(
      1, plan.indices_shape, RuntimeInputShapePolicy::TensorOrStaticOrConstant);
  if (!data_shape_known) {
    OPENVINO_THROW("GFX MLIR: Gather data shape is unknown for stage ",
                   stage_name);
  }
  if (!idx_shape_known) {
    OPENVINO_THROW("GFX MLIR: Gather indices shape is unknown for stage ",
                   stage_name);
  }
  const auto axis_values = inputs.i64_values(2);
  OPENVINO_ASSERT(axis_values,
                  "GFX runtime: Gather axis must be available for stage ",
                  stage_name);
  OPENVINO_ASSERT(axis_values->size() == 1,
                  "GFX MLIR: Gather axis must be scalar for stage ",
                  stage_name);
  plan.axis_norm = normalize_axis((*axis_values)[0], plan.data_shape.size(),
                                  "GFX MLIR: Gather");

  ov::Shape out_shape;
  out_shape.reserve(plan.data_shape.size() + plan.indices_shape.size());
  out_shape.insert(out_shape.end(), plan.data_shape.begin(),
                   plan.data_shape.begin() +
                       static_cast<std::ptrdiff_t>(plan.axis_norm));
  out_shape.insert(out_shape.end(), plan.indices_shape.begin(),
                   plan.indices_shape.end());
  out_shape.insert(out_shape.end(),
                   plan.data_shape.begin() +
                       static_cast<std::ptrdiff_t>(plan.axis_norm) + 1,
                   plan.data_shape.end());

  plan.values.output_shape = out_shape;
  plan.values.value_shape = out_shape;
  plan.values.output_type =
      descriptor_required_output_type(descriptor, 0, stage_name);
  plan.values.force_output_type = true;
  plan.axis_dim = static_cast<uint32_t>(
      plan.data_shape[static_cast<size_t>(plan.axis_norm)]);
  plan.indices_count =
      static_cast<uint32_t>(ov::shape_size(plan.indices_shape));
  plan.identity_view = plan.axis_dim == 1 && plan.indices_count == 1 &&
                       out_shape == plan.data_shape;

  if (auto data_values = inputs.i64_values(0)) {
    if (auto idx_values = inputs.i64_values(1)) {
      if (plan.axis_norm == 0 && plan.data_shape.size() == 1 &&
          !data_values->empty()) {
        std::vector<int64_t> gathered_values;
        gathered_values.reserve(idx_values->size());
        for (auto idx : *idx_values) {
          int64_t normalized =
              idx < 0 ? idx + static_cast<int64_t>(data_values->size()) : idx;
          normalized = std::clamp<int64_t>(
              normalized, 0,
              static_cast<int64_t>(
                  data_values->empty() ? 0 : data_values->size() - 1));
          gathered_values.push_back(
              (*data_values)[static_cast<size_t>(normalized)]);
        }
        plan.values.i64_values = std::move(gathered_values);
        plan.values.has_i64_values = true;
      }
    }
  }
  plan.available = true;
  return plan;
}

RuntimeScatterUpdatePlan
plan_scatter_update_runtime_values(const RuntimeInputResolver &inputs,
                                   const RuntimeStageExecutableDescriptor &descriptor,
                                   std::string_view stage_name) {
  OPENVINO_ASSERT(descriptor.op_family == "ScatterUpdate",
                  "GFX runtime: ScatterUpdate runtime values require "
                  "ScatterUpdate descriptor for stage ",
                  stage_name);
  RuntimeScatterUpdatePlan plan;
  if (!inputs.shape_known(0, plan.values.output_shape) ||
      !inputs.shape_known(1, plan.indices_shape) ||
      !inputs.shape_known(2, plan.updates_shape)) {
    return plan;
  }
  const auto axis_values = inputs.i64_values(3);
  OPENVINO_ASSERT(axis_values,
                  "GFX runtime: ScatterUpdate axis must be available for stage ",
                  stage_name);
  OPENVINO_ASSERT(axis_values->size() == 1,
                  "GFX MLIR: ScatterUpdate axis must be scalar for stage ",
                  stage_name);
  plan.axis_norm = (*axis_values)[0];
  if (plan.axis_norm < 0) {
    plan.axis_norm += static_cast<int64_t>(plan.values.output_shape.size());
  }
  OPENVINO_ASSERT(plan.axis_norm >= 0 && static_cast<size_t>(plan.axis_norm) <
                                             plan.values.output_shape.size(),
                  "GFX MLIR: ScatterUpdate axis out of range for stage ",
                  stage_name);
  OPENVINO_ASSERT(plan.values.output_shape.size() <= 8 &&
                      plan.indices_shape.size() <= 8 &&
                      plan.updates_shape.size() <= 16,
                  "GFX MLIR: ScatterUpdate rank exceeds Metal params capacity "
                  "for stage ",
                  stage_name);
  plan.values.value_shape = plan.values.output_shape;
  plan.values.output_type =
      descriptor_required_output_type(descriptor, 0, stage_name);
  plan.available = true;
  return plan;
}

RuntimeSplitPlan plan_split_runtime_values(const RuntimeInputResolver &inputs,
                                           const RuntimeStageExecutableDescriptor &descriptor,
                                           size_t output_count,
                                           std::string_view stage_name) {
  OPENVINO_ASSERT(descriptor.op_family == "Split" ||
                      descriptor.op_family == "VariadicSplit",
                  "GFX runtime: Split runtime values require Split/"
                  "VariadicSplit descriptor for stage ",
                  stage_name);
  RuntimeSplitPlan plan;
  plan.input_shape = inputs.shape(0);
  if (plan.input_shape.empty()) {
    descriptor_input_contract_shape(&descriptor, 0, plan.input_shape);
  }
  OPENVINO_ASSERT(!plan.input_shape.empty(),
                  "GFX MLIR: Split input shape is unknown for stage ",
                  stage_name);

  int64_t axis = 0;
  size_t parts = 0;
  bool is_split = false;
  if (descriptor.op_family == "Split") {
    const auto axis_values = inputs.i64_values(1);
    OPENVINO_ASSERT(axis_values,
                    "GFX runtime: Split axis must be available for stage ",
                    stage_name);
    OPENVINO_ASSERT(axis_values->size() == 1,
                    "GFX MLIR: Split axis must be scalar for stage ",
                    stage_name);
    axis = (*axis_values)[0];
    parts = output_count;
    is_split = true;
  } else if (descriptor.op_family == "VariadicSplit") {
    const auto axis_values = inputs.i64_values(1);
    OPENVINO_ASSERT(axis_values,
                    "GFX runtime: VariadicSplit axis must be available for "
                    "stage ",
                    stage_name);
    OPENVINO_ASSERT(axis_values->size() == 1,
                    "GFX MLIR: VariadicSplit axis must be scalar for stage ",
                    stage_name);
    axis = (*axis_values)[0];
    auto lengths = inputs.i64_values(2);
    OPENVINO_ASSERT(lengths,
                    "GFX runtime: VariadicSplit lengths must be available for "
                    "stage ",
                    stage_name);
    plan.split_sizes.reserve(lengths->size());
    int64_t infer_index = -1;
    size_t known_sum = 0;
    for (size_t i = 0; i < lengths->size(); ++i) {
      const int64_t length = (*lengths)[i];
      if (length < 0) {
        OPENVINO_ASSERT(length == -1,
                        "GFX MLIR: VariadicSplit only -1 negative length is "
                        "supported for stage ",
                        stage_name);
        OPENVINO_ASSERT(infer_index < 0,
                        "GFX MLIR: VariadicSplit supports only one inferred -1 "
                        "length for stage ",
                        stage_name);
        infer_index = static_cast<int64_t>(i);
        plan.split_sizes.push_back(0);
      } else {
        known_sum += static_cast<size_t>(length);
        plan.split_sizes.push_back(static_cast<size_t>(length));
      }
    }
    if (infer_index >= 0) {
      const int64_t axis_norm =
          normalize_axis(axis, plan.input_shape.size(), "GFX MLIR: VariadicSplit");
      const size_t axis_len = plan.input_shape[static_cast<size_t>(axis_norm)];
      OPENVINO_ASSERT(known_sum <= axis_len,
                      "GFX MLIR: VariadicSplit known lengths exceed axis "
                      "dimension for stage ",
                      stage_name);
      plan.split_sizes[static_cast<size_t>(infer_index)] = axis_len - known_sum;
    }
  }

  plan.axis_norm =
      normalize_axis(axis, plan.input_shape.size(), "GFX MLIR: Split");
  plan.axis_len =
      static_cast<uint32_t>(plan.input_shape[static_cast<size_t>(plan.axis_norm)]);

  if (is_split) {
    OPENVINO_ASSERT(parts > 0,
                    "GFX MLIR: Split number of splits is zero for stage ",
                    stage_name);
    OPENVINO_ASSERT(
        plan.axis_len % parts == 0,
        "GFX MLIR: Split dimension not divisible by parts for stage ",
        stage_name);
    plan.split_sizes.assign(parts, plan.axis_len / parts);
  }

  size_t sum = 0;
  for (auto split_size : plan.split_sizes) {
    sum += split_size;
  }
  OPENVINO_ASSERT(sum == plan.axis_len,
                  "GFX MLIR: Split sizes do not sum to axis length for stage ",
                  stage_name, " (", sum, " vs ", plan.axis_len, ")");
  OPENVINO_ASSERT(!plan.split_sizes.empty(),
                  "GFX MLIR: Split sizes are empty for stage ", stage_name);
  OPENVINO_ASSERT(output_count == plan.split_sizes.size(),
                  "GFX MLIR: Split output count mismatch for stage ",
                  stage_name, " (expected ", plan.split_sizes.size(), ", got ",
                  output_count, ")");

  plan.inner_stride = 1;
  for (size_t d = static_cast<size_t>(plan.axis_norm) + 1;
       d < plan.input_shape.size(); ++d) {
    plan.inner_stride *= plan.input_shape[d];
  }
  plan.available = true;
  return plan;
}

RuntimeSlicePlan plan_slice_runtime_values(
    const RuntimeInputResolver &inputs, const std::vector<GpuTensor *> &outputs,
    bool requires_runtime_shape_args, std::string_view stage_name) {
  const auto &descriptor =
      require_descriptor_contract(inputs, "slice", stage_name);
  OPENVINO_ASSERT(descriptor.runtime_shape_rule == "slice",
                  "GFX runtime: Slice planner received descriptor rule '",
                  descriptor.runtime_shape_rule, "' for ", stage_name);
  const auto slice = parse_runtime_slice_descriptor(descriptor, stage_name);
  RuntimeSlicePlan plan;
  ov::Shape static_input_shape;
  ov::Shape static_output_shape;
  const bool input_shape_static =
      descriptor_input_contract_shape(&descriptor, 0, static_input_shape);
  const bool output_shape_static =
      descriptor_output_contract_shape(&descriptor, 0, static_output_shape);
  plan.use_runtime_args = requires_runtime_shape_args || !input_shape_static ||
                          !output_shape_static ||
                          slice_requires_runtime_indexing(inputs, slice);
  plan.input_shape = inputs.shape(0);
  if (plan.input_shape.empty()) {
    OPENVINO_THROW(
        "GFX MLIR: Slice/StridedSlice input shape is unknown for stage ",
        stage_name);
  }

  plan.values.output_shape = infer_slice_output_shape(
      inputs, descriptor, slice, plan.input_shape, outputs, stage_name);
  OPENVINO_ASSERT(
      !plan.values.output_shape.empty(),
      "GFX MLIR: Slice/StridedSlice output shape is unknown for stage ",
      stage_name);
  OPENVINO_ASSERT(plan.input_shape.size() == plan.values.output_shape.size(),
                  "GFX MLIR: Slice/StridedSlice rank mismatch for stage ",
                  stage_name);

  build_slice_runtime_spec(inputs, slice, plan.input_shape,
                           plan.values.output_shape, plan.starts_full,
                           plan.steps_full, stage_name);

  plan.values.value_shape = plan.values.output_shape;
  plan.values.output_type =
      descriptor_required_output_type(descriptor, 0, stage_name);
  plan.values.force_output_type = true;
  plan.linear_view =
      slice_is_runtime_linear_view(plan.input_shape, plan.values.output_shape,
                                   plan.starts_full, plan.steps_full);
  plan.available = true;
  return plan;
}

RuntimeTransposePlan
plan_transpose_runtime_values(const RuntimeInputResolver &inputs,
                              const RuntimeStageExecutableDescriptor &descriptor,
                              std::string_view stage_name) {
  OPENVINO_ASSERT(descriptor.op_family == "Transpose",
                  "GFX runtime: Transpose runtime values require Transpose "
                  "descriptor for stage ",
                  stage_name);

  RuntimeTransposePlan plan;
  plan.input_shape = inputs.shape(0);
  if (plan.input_shape.empty()) {
    descriptor_input_contract_shape(&descriptor, 0, plan.input_shape);
  }
  if (plan.input_shape.empty()) {
    OPENVINO_THROW("GFX MLIR: Transpose input shape is unknown for stage ",
                   stage_name);
  }
  auto permutation = inputs.i64_values(1);
  OPENVINO_ASSERT(permutation,
                  "GFX runtime: Transpose permutation must be available for "
                  "stage ",
                  stage_name);
  plan.permutation = std::move(*permutation);
  OPENVINO_ASSERT(plan.permutation.size() == plan.input_shape.size(),
                  "GFX MLIR: Transpose perm rank mismatch for stage ",
                  stage_name);

  plan.values.output_shape.assign(plan.input_shape.size(), 0);
  for (size_t i = 0; i < plan.permutation.size(); ++i) {
    const int64_t axis = plan.permutation[i];
    OPENVINO_ASSERT(
        axis >= 0 && static_cast<size_t>(axis) < plan.input_shape.size(),
        "GFX MLIR: Transpose perm out of range for stage ", stage_name);
    plan.values.output_shape[i] = plan.input_shape[static_cast<size_t>(axis)];
  }
  plan.values.value_shape = plan.values.output_shape;
  plan.values.output_type =
      descriptor_required_output_type(descriptor, 0, stage_name);
  plan.values.force_output_type = true;
  plan.linear_view = transpose_is_runtime_linear_view(
      plan.input_shape, plan.values.output_shape, plan.permutation, stage_name);
  plan.available = true;
  return plan;
}

RuntimeInterpolatePlan plan_interpolate_runtime_values(
    const RuntimeInputResolver &inputs, const std::vector<GpuTensor *> &outputs,
    const RuntimeStageExecutableDescriptor &descriptor,
    std::string_view stage_name) {
  OPENVINO_ASSERT(descriptor.op_family == "Interpolate",
                  "GFX runtime: Interpolate runtime values require Interpolate "
                  "descriptor for stage ",
                  stage_name);
  RuntimeInterpolatePlan plan;
  plan.input_shape = inputs.shape(0);
  if (plan.input_shape.empty()) {
    descriptor_input_contract_shape(&descriptor, 0, plan.input_shape);
  }
  if (plan.input_shape.empty()) {
    OPENVINO_THROW("GFX MLIR: Interpolate input shape is unknown for stage ",
                   stage_name);
  }
  plan.values.output_shape = resolve_primary_output_shape(descriptor, outputs);
  OPENVINO_ASSERT(!plan.values.output_shape.empty(),
                  "GFX MLIR: Interpolate output shape is unknown for stage ",
                  stage_name);
  OPENVINO_ASSERT(
      plan.input_shape.size() == 4 && plan.values.output_shape.size() == 4,
      "GFX MLIR: Interpolate expects NCHW rank4 for stage ", stage_name);

  OPENVINO_ASSERT(
      descriptor.runtime_shape_i64_metadata.size() >= 3,
      "GFX runtime: Interpolate descriptor metadata must contain "
      "align_corners, half_pixel and nearest_mode for stage ",
      stage_name);
  plan.align_corners = descriptor.runtime_shape_i64_metadata[0] != 0;
  plan.use_half_pixel = descriptor.runtime_shape_i64_metadata[1] != 0;
  plan.nearest_mode = static_cast<uint32_t>(
      std::max<int64_t>(descriptor.runtime_shape_i64_metadata[2], 0));
  if (plan.nearest_mode > 2) {
    OPENVINO_THROW("GFX runtime: Interpolate descriptor nearest_mode is "
                   "unsupported for stage ",
                   stage_name);
  }
  plan.values.value_shape = plan.values.output_shape;
  plan.values.output_type =
      descriptor_required_output_type(descriptor, 0, stage_name);
  plan.values.force_output_type = true;
  plan.available = true;
  return plan;
}

RuntimeSoftmaxPlan
plan_softmax_runtime_values(const RuntimeInputResolver &inputs,
                            const RuntimeStageExecutableDescriptor &descriptor,
                            std::string_view stage_name) {
  RuntimeSoftmaxPlan plan;
  OPENVINO_ASSERT(
      descriptor.op_family == "Softmax" || descriptor.op_family == "LogSoftmax",
      "GFX runtime: Softmax runtime values require Softmax/LogSoftmax "
      "descriptor for stage ",
      stage_name);
  OPENVINO_ASSERT(
      descriptor.runtime_param_i64_metadata.size() == 3,
      "GFX runtime: Softmax descriptor runtime-param metadata must contain "
      "rows, axis_len and inner for stage ",
      stage_name);

  plan.values.output_shape = inputs.shape(0);
  if (plan.values.output_shape.empty()) {
    OPENVINO_THROW("GFX MLIR: Softmax input shape is unknown for stage ",
                   stage_name);
  }
  ov::Shape descriptor_output_shape;
  if (descriptor_output_contract_shape(&descriptor, 0, descriptor_output_shape)) {
    OPENVINO_ASSERT(descriptor_output_shape == plan.values.output_shape,
                    "GFX runtime: Softmax descriptor output shape drift for ",
                    stage_name);
  }

  OPENVINO_ASSERT(descriptor.runtime_param_i64_metadata[0] > 0 &&
                      descriptor.runtime_param_i64_metadata[1] > 0 &&
                      descriptor.runtime_param_i64_metadata[2] > 0,
                  "GFX runtime: Softmax descriptor metadata must be positive "
                  "for stage ",
                  stage_name);
  plan.rows = static_cast<uint64_t>(descriptor.runtime_param_i64_metadata[0]);
  plan.axis_len =
      static_cast<uint64_t>(descriptor.runtime_param_i64_metadata[1]);
  plan.inner = static_cast<uint64_t>(descriptor.runtime_param_i64_metadata[2]);
  plan.axis = descriptor_softmax_axis_from_metadata(
      plan.values.output_shape, plan.rows, plan.axis_len, plan.inner);
  OPENVINO_ASSERT(plan.rows * plan.axis_len ==
                          shape_size_u64(plan.values.output_shape) &&
                      plan.axis >= 0,
                  "GFX runtime: Softmax descriptor metadata does not map to an "
                  "axis for stage ",
                  stage_name);
  plan.log_softmax = descriptor.op_family == "LogSoftmax";
  plan.values.value_shape = plan.values.output_shape;
  plan.values.output_type = descriptor_required_output_type(descriptor, 0,
                                                            stage_name);
  plan.values.force_output_type = true;
  plan.available = true;
  return plan;
}

void assign_runtime_value_outputs(const RuntimeValuePlan &plan,
                                  const std::vector<GpuTensor *> &outputs) {
  for (auto *out : outputs) {
    if (!out) {
      continue;
    }
    out->shape = plan.output_shape;
    if (plan.force_output_type || out->expected_type == ov::element::dynamic) {
      out->expected_type = plan.output_type;
    }
    if (plan.has_i64_values) {
      assign_i64_values(out, plan.i64_values, plan.value_shape);
    }
  }
}

void assign_i64_values(GpuTensor *out, const std::vector<int64_t> &values,
                       const ov::Shape &shape) {
  if (!out || values.size() != ov::shape_size(shape)) {
    return;
  }
  out->i64_values = values;
}

std::optional<std::vector<int64_t>> compute_reduce_i64_values(
    std::string_view reduce_type, const std::vector<int64_t> &input_values,
    const ov::Shape &input_shape, const RuntimeReduceInfo &reduce_info,
    const ov::Shape &output_shape) {
  if (input_values.size() != ov::shape_size(input_shape)) {
    return std::nullopt;
  }

  const size_t output_count = ov::shape_size(output_shape);
  std::vector<int64_t> output_values(output_count, 0);
  std::vector<bool> initialized(output_count, false);
  auto in_strides = compute_row_major_strides(input_shape);
  auto out_strides = compute_row_major_strides(output_shape);
  std::vector<size_t> input_coord(input_shape.size(), 0);

  for (size_t linear = 0; linear < input_values.size(); ++linear) {
    size_t rem = linear;
    for (size_t axis = 0; axis < input_shape.size(); ++axis) {
      const size_t stride = in_strides[axis];
      input_coord[axis] = stride == 0 ? 0 : rem / stride;
      rem = stride == 0 ? 0 : rem - input_coord[axis] * stride;
    }

    size_t out_axis = 0;
    size_t out_linear = 0;
    for (size_t axis = 0; axis < input_shape.size(); ++axis) {
      const bool reduced = reduce_info.axes.count(axis) != 0;
      if (reduced && !reduce_info.keep_dims) {
        continue;
      }
      const size_t coord = reduced ? 0 : input_coord[axis];
      if (out_axis < out_strides.size()) {
        out_linear += coord * out_strides[out_axis];
      }
      ++out_axis;
    }
    if (output_shape.empty()) {
      out_linear = 0;
    }

    const int64_t value = input_values[linear];
    if (!initialized[out_linear]) {
      initialized[out_linear] = true;
      if (reduce_type == "ReduceSum" || reduce_type == "ReduceProd" ||
          reduce_type == "ReduceMax" || reduce_type == "ReduceMin") {
        output_values[out_linear] = value;
      } else {
        return std::nullopt;
      }
      continue;
    }
    if (reduce_type == "ReduceSum") {
      output_values[out_linear] += value;
    } else if (reduce_type == "ReduceProd") {
      output_values[out_linear] *= value;
    } else if (reduce_type == "ReduceMax") {
      output_values[out_linear] = std::max(output_values[out_linear], value);
    } else if (reduce_type == "ReduceMin") {
      output_values[out_linear] = std::min(output_values[out_linear], value);
    } else {
      return std::nullopt;
    }
  }

  return output_values;
}

bool bind_small_i64_const_stage_outputs(GpuBufferManager *buffer_manager,
                                        const std::vector<GpuTensor *> &outputs,
                                        std::vector<GpuTensor> &cache,
                                        GfxProfiler *profiler,
                                        bool profiling_enabled,
                                        std::string_view stage_name,
                                        std::string_view suffix) {
  if (!buffer_manager || outputs.empty()) {
    return false;
  }
  std::vector<ov::element::Type> resolved_types;
  resolved_types.reserve(outputs.size());
  for (auto *out : outputs) {
    if (!out || out->i64_values.empty() ||
        out->i64_values.size() != ov::shape_size(out->shape) ||
        out->i64_values.size() > kGfxInlineRuntimeI64ValueLimit) {
      return false;
    }
    const auto type = out->expected_type;
    if (type != ov::element::i32 && type != ov::element::i64) {
      return false;
    }
    resolved_types.push_back(type);
  }

  if (cache.size() == outputs.size()) {
    bool cache_hit = true;
    for (size_t oi = 0; oi < outputs.size(); ++oi) {
      const auto *out = outputs[oi];
      const auto &cached = cache[oi];
      if (!cached.buf.valid() || cached.expected_type != resolved_types[oi] ||
          cached.shape != out->shape || cached.i64_values != out->i64_values) {
        cache_hit = false;
        break;
      }
    }
    if (cache_hit) {
      for (size_t oi = 0; oi < outputs.size(); ++oi) {
        auto *out = outputs[oi];
        const auto &cached = cache[oi];
        out->buf = cached.buf;
        out->buf.external = true;
        out->expected_type = cached.expected_type;
        out->shape = cached.shape;
        out->i64_values = cached.i64_values;
      }
      if (profiler && profiling_enabled) {
        profiler->increment_counter("small_i64_const_cache_hit_count");
      }
      return true;
    }
  }

  cache.clear();
  cache.reserve(outputs.size());
  for (size_t oi = 0; oi < outputs.size(); ++oi) {
    auto *out = outputs[oi];
    const auto type = resolved_types[oi];
    std::ostringstream key;
    key << stage_name << "/" << suffix << "/" << type << "/";
    for (auto value : out->i64_values) {
      key << value << ",";
    }

    GpuBuffer buf;
    if (type == ov::element::i32) {
      std::vector<int32_t> values;
      values.reserve(out->i64_values.size());
      for (auto value : out->i64_values) {
        values.push_back(static_cast<int32_t>(value));
      }
      buf = buffer_manager->wrap_const(key.str(), values.data(),
                                       values.size() * sizeof(int32_t),
                                       ov::element::i32);
    } else {
      buf = buffer_manager->wrap_const(key.str(), out->i64_values.data(),
                                       out->i64_values.size() * sizeof(int64_t),
                                       ov::element::i64);
    }
    OPENVINO_ASSERT(buf.valid(),
                    "GFX MLIR: failed to bind small const output for stage ",
                    stage_name);
    buf.owned = false;
    out->buf = buf;
    out->buf.external = true;
    out->expected_type = type;

    GpuTensor cached;
    cached.buf = out->buf;
    cached.shape = out->shape;
    cached.i64_values = out->i64_values;
    cached.expected_type = type;
    cache.push_back(std::move(cached));
  }
  if (profiler && profiling_enabled) {
    profiler->increment_counter("small_i64_const_stage_count");
  }
  return true;
}

} // namespace gfx_plugin
} // namespace ov
