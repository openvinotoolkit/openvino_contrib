// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/gfx_stage_runtime_values.hpp"

#include <algorithm>
#include <functional>
#include <numeric>
#include <sstream>

#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "runtime/gfx_profiler.hpp"
#include "runtime/gfx_runtime_value_limits.hpp"
#include "runtime/gfx_shape_utils.hpp"
#include "runtime/gpu_buffer_manager.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

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

std::vector<int64_t> constant_source_i64(const ov::Output<ov::Node> &source,
                                         const char *what,
                                         std::string_view stage_name) {
  auto constant = ov::util::get_constant_from_source(source);
  OPENVINO_ASSERT(constant, "GFX MLIR: ", what, " must be constant for stage ",
                  stage_name);
  return constant->cast_vector<int64_t>();
}

std::optional<std::vector<int64_t>>
optional_constant_source_i64(const ov::Output<ov::Node> &source) {
  auto constant = ov::util::get_constant_from_source(source);
  if (!constant) {
    return std::nullopt;
  }
  return constant->cast_vector<int64_t>();
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

void build_slice_runtime_spec(const RuntimeInputResolver &inputs,
                              const ov::Node &node, const ov::Shape &in_shape,
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

  if (auto slice = dynamic_cast<const ov::op::v8::Slice *>(&node)) {
    auto starts = inputs.i64_values(1);
    auto ends = inputs.i64_values(2);
    auto steps = inputs.i64_values(3);
    OPENVINO_ASSERT(starts, "GFX MLIR: Slice starts must be available for stage ",
                    stage_name);
    OPENVINO_ASSERT(steps, "GFX MLIR: Slice steps must be available for stage ",
                    stage_name);
    std::vector<int64_t> axes;
    if (slice->get_input_size() > 4) {
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
      steps_full[static_cast<size_t>(axis)] =
          static_cast<int32_t>((*steps)[i]);
    }
    return;
  }

  auto slice = dynamic_cast<const ov::op::v1::StridedSlice *>(&node);
  OPENVINO_ASSERT(slice,
                  "GFX MLIR: expected Slice/StridedSlice node for stage ",
                  stage_name);
  OPENVINO_ASSERT(
      std::all_of(slice->get_new_axis_mask().begin(),
                  slice->get_new_axis_mask().end(),
                  [](int64_t v) { return v == 0; }),
      "GFX MLIR: StridedSlice new_axis_mask is not supported for stage ",
      stage_name);
  OPENVINO_ASSERT(
      std::all_of(slice->get_shrink_axis_mask().begin(),
                  slice->get_shrink_axis_mask().end(),
                  [](int64_t v) { return v == 0; }),
      "GFX MLIR: StridedSlice shrink_axis_mask is not supported for stage ",
      stage_name);
  OPENVINO_ASSERT(
      std::all_of(slice->get_ellipsis_mask().begin(),
                  slice->get_ellipsis_mask().end(),
                  [](int64_t v) { return v == 0; }),
      "GFX MLIR: StridedSlice ellipsis_mask is not supported for stage ",
      stage_name);

  auto begin = inputs.i64_values(1);
  OPENVINO_ASSERT(begin,
                  "GFX MLIR: StridedSlice begin must be available for stage ",
                  stage_name);
  auto end = inputs.i64_values(2);
  std::vector<int64_t> strides(rank, 1);
  if (slice->get_input_size() > 3) {
    auto values = inputs.i64_values(3);
    OPENVINO_ASSERT(values,
                    "GFX MLIR: StridedSlice strides must be available for stage ",
                    stage_name);
    OPENVINO_ASSERT(values->size() <= rank,
                    "GFX MLIR: StridedSlice strides rank mismatch for stage ",
                    stage_name);
    std::copy(values->begin(), values->end(), strides.begin());
  }
  OPENVINO_ASSERT(
      begin->size() <= rank && (!end.has_value() || end->size() <= rank),
      "GFX MLIR: StridedSlice begin/end rank mismatch for stage ", stage_name);
  const auto &begin_mask = slice->get_begin_mask();
  const auto &end_mask = slice->get_end_mask();
  for (size_t axis = 0; axis < rank; ++axis) {
    const auto dim = static_cast<int64_t>(in_shape[axis]);
    const bool masked_begin = axis < begin_mask.size() && begin_mask[axis] != 0;
    const bool masked_end = axis < end_mask.size() && end_mask[axis] != 0;
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

bool slice_requires_runtime_indexing(const ov::Node &node) {
  if (auto slice = dynamic_cast<const ov::op::v8::Slice *>(&node)) {
    auto starts = optional_constant_source_i64(slice->input_value(1));
    if (!starts) {
      return true;
    }
    auto steps = optional_constant_source_i64(slice->input_value(3));
    if (!steps) {
      return true;
    }
    if (slice->get_input_size() > 4 &&
        !optional_constant_source_i64(slice->input_value(4))) {
      return true;
    }
    return std::any_of(steps->begin(), steps->end(),
                       [](int64_t step) { return step < 0; });
  }
  if (auto slice = dynamic_cast<const ov::op::v1::StridedSlice *>(&node)) {
    auto begin = optional_constant_source_i64(slice->input_value(1));
    if (!begin) {
      return true;
    }
    if (!optional_constant_source_i64(slice->input_value(2))) {
      return true;
    }
    if (slice->get_input_size() <= 3) {
      return false;
    }
    auto steps = optional_constant_source_i64(slice->input_value(3));
    if (!steps) {
      return true;
    }
    return std::any_of(steps->begin(), steps->end(),
                       [](int64_t step) { return step < 0; });
  }
  return false;
}

ov::Shape infer_slice_output_shape(const RuntimeInputResolver &inputs,
                                   const ov::Node &node,
                                   const ov::Shape &in_shape,
                                   const std::vector<GpuTensor *> &outputs,
                                   std::string_view stage_name) {
  if (!outputs.empty() && outputs.front() && !outputs.front()->shape.empty()) {
    return outputs.front()->shape;
  }
  if (node.get_output_partial_shape(0).is_static()) {
    return node.get_output_shape(0);
  }

  const size_t rank = in_shape.size();
  ov::Shape out_shape;
  if (auto slice = dynamic_cast<const ov::op::v8::Slice *>(&node)) {
    auto starts = inputs.i64_values(1);
    auto ends = inputs.i64_values(2);
    auto steps = inputs.i64_values(3);
    std::optional<std::vector<int64_t>> axes;
    if (slice->get_input_size() > 4) {
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
  } else if (auto slice =
                 dynamic_cast<const ov::op::v1::StridedSlice *>(&node)) {
    OPENVINO_ASSERT(
        std::all_of(slice->get_new_axis_mask().begin(),
                    slice->get_new_axis_mask().end(),
                    [](int64_t v) { return v == 0; }),
        "GFX MLIR: StridedSlice new_axis_mask is not supported for stage ",
        stage_name);
    OPENVINO_ASSERT(
        std::all_of(slice->get_shrink_axis_mask().begin(),
                    slice->get_shrink_axis_mask().end(),
                    [](int64_t v) { return v == 0; }),
        "GFX MLIR: StridedSlice shrink_axis_mask is not supported for stage ",
        stage_name);
    OPENVINO_ASSERT(
        std::all_of(slice->get_ellipsis_mask().begin(),
                    slice->get_ellipsis_mask().end(),
                    [](int64_t v) { return v == 0; }),
        "GFX MLIR: StridedSlice ellipsis_mask is not supported for stage ",
        stage_name);

    auto begin = inputs.i64_values(1);
    auto end = inputs.i64_values(2);
    std::optional<std::vector<int64_t>> strides;
    if (slice->get_input_size() > 3) {
      strides = inputs.i64_values(3);
    }
    if (begin && end) {
      OPENVINO_ASSERT(begin->size() <= rank && end->size() <= rank,
                      "GFX MLIR: StridedSlice begin/end rank mismatch for stage ",
                      stage_name);
      if (strides) {
        OPENVINO_ASSERT(strides->size() <= rank,
                        "GFX MLIR: StridedSlice strides rank mismatch for stage ",
                        stage_name);
      }
      const auto &begin_mask = slice->get_begin_mask();
      const auto &end_mask = slice->get_end_mask();
      out_shape.reserve(rank);
      for (size_t axis = 0; axis < rank; ++axis) {
        const auto dim = static_cast<int64_t>(in_shape[axis]);
        const int64_t step =
            strides && axis < strides->size() ? (*strides)[axis] : 1;
        OPENVINO_ASSERT(step != 0,
                        "GFX MLIR: StridedSlice zero step is not supported for stage ",
                        stage_name);
        const bool masked_begin =
            axis < begin_mask.size() && begin_mask[axis] != 0;
        const bool masked_end = axis < end_mask.size() && end_mask[axis] != 0;
        int64_t start = axis < begin->size() ? (*begin)[axis] : 0;
        int64_t finish = axis < end->size() ? (*end)[axis] : dim;
        start = masked_begin ? (step < 0 ? dim - 1 : 0)
                             : normalize_slice_index(start, dim, true);
        finish = masked_end ? (step < 0 ? -1 : dim)
                            : normalize_slice_index(finish, dim, false);
        const int64_t extent =
            step > 0 ? std::max<int64_t>(0, finish - start)
                     : std::max<int64_t>(0, start - finish);
        const int64_t stride = step > 0 ? step : -step;
        out_shape.push_back(static_cast<size_t>((extent + stride - 1) / stride));
      }
    }
  }
  if (!out_shape.empty()) {
    return out_shape;
  }

  if (node.get_output_partial_shape(0).rank().is_static()) {
    const auto pshape = node.get_output_partial_shape(0);
    out_shape.reserve(static_cast<size_t>(pshape.rank().get_length()));
    for (size_t i = 0; i < static_cast<size_t>(pshape.rank().get_length());
         ++i) {
      out_shape.push_back(pshape[i].is_static()
                              ? static_cast<size_t>(pshape[i].get_length())
                              : (i < in_shape.size() ? in_shape[i] : 1));
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

ov::Shape
resolve_primary_output_shape(const ov::Node &node,
                             const std::vector<GpuTensor *> &outputs) {
  if (!outputs.empty() && outputs.front() && !outputs.front()->shape.empty()) {
    return outputs.front()->shape;
  }
  if (node.get_output_partial_shape(0).is_static()) {
    return node.get_output_shape(0);
  }
  return {};
}

ov::Shape partial_shape_with_dynamic_ones(const ov::PartialShape &pshape) {
  ov::Shape shape;
  if (!pshape.rank().is_static()) {
    return shape;
  }
  shape.reserve(static_cast<size_t>(pshape.rank().get_length()));
  for (size_t i = 0; i < static_cast<size_t>(pshape.rank().get_length()); ++i) {
    shape.push_back(pshape[i].is_static()
                        ? static_cast<size_t>(pshape[i].get_length())
                        : 1);
  }
  return shape;
}

ov::element::Type resolve_runtime_input_type(const RuntimeInputResolver &inputs,
                                             const ov::Node &node,
                                             size_t input_idx) {
  if (auto *tensor = inputs.tensor(input_idx)) {
    if (tensor->expected_type != ov::element::dynamic) {
      return tensor->expected_type;
    }
    if (tensor->buf.valid()) {
      return tensor->buf.type;
    }
  }
  if (input_idx < node.get_input_size()) {
    return node.get_input_element_type(input_idx);
  }
  return ov::element::dynamic;
}

} // namespace

ov::Shape RuntimeInputResolver::shape(size_t idx) const {
  if (inputs && idx < inputs->size() && (*inputs)[idx] &&
      !(*inputs)[idx]->shape.empty()) {
    return (*inputs)[idx]->shape;
  }
  if (node && idx < node->get_input_size() &&
      node->get_input_partial_shape(idx).is_static()) {
    return node->get_input_shape(idx);
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
  if (!node || idx >= node->get_input_size()) {
    return false;
  }
  if (node->get_input_partial_shape(idx).is_static()) {
    out_shape = node->get_input_shape(idx);
    return true;
  }
  if (policy == RuntimeInputShapePolicy::TensorOrStaticOrConstant) {
    if (auto input_const =
            ov::util::get_constant_from_source(node->input_value(idx))) {
      out_shape = input_const->get_shape();
      return true;
    }
  }
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
  if (!node || input_idx >= node->get_input_size()) {
    return std::nullopt;
  }
  if (auto input_const =
          ov::util::get_constant_from_source(node->input_value(input_idx))) {
    return input_const->cast_vector<int64_t>();
  }
  return std::nullopt;
}

void RuntimeInputResolver::ensure_output_shape(size_t output_idx,
                                               GpuTensor *out) const {
  if (!out || !out->shape.empty() || !node ||
      output_idx >= node->get_output_size() ||
      !node->get_output_partial_shape(output_idx).is_static()) {
    return;
  }
  out->shape = node->get_output_shape(output_idx);
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

RuntimeValuePlan plan_reshape_runtime_values(const RuntimeInputResolver &inputs,
                                             const ov::Node &node,
                                             std::string_view stage_name) {
  const auto *reshape = dynamic_cast<const ov::op::v1::Reshape *>(&node);
  OPENVINO_ASSERT(reshape, "GFX MLIR: expected Reshape node for stage ",
                  stage_name);

  ov::Shape input_shape;
  bool input_shape_known = inputs.shape_known(
      0, input_shape, RuntimeInputShapePolicy::TensorOrStaticOrConstant);
  if (!input_shape_known) {
    const auto input_pshape = node.get_input_partial_shape(0);
    if (input_pshape.rank().is_static()) {
      input_shape = partial_shape_with_dynamic_ones(input_pshape);
      input_shape_known = true;
    }
  }
  OPENVINO_ASSERT(input_shape_known,
                  "GFX MLIR: Reshape input shape is unknown for stage ",
                  stage_name);

  ov::Shape output_shape;
  if (auto pattern = inputs.i64_values(1)) {
    output_shape.reserve(pattern->size());
    int64_t infer_pos = -1;
    size_t known_product = 1;
    for (size_t i = 0; i < pattern->size(); ++i) {
      const int64_t dim = (*pattern)[i];
      if (dim == 0 && reshape->get_special_zero()) {
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
        partial_shape_with_dynamic_ones(node.get_output_partial_shape(0));
  }

  RuntimeValuePlan plan;
  plan.output_shape = std::move(output_shape);
  plan.value_shape = plan.output_shape;
  plan.output_type = resolve_runtime_input_type(inputs, node, 0);
  plan.force_output_type = true;
  if (auto input_values = inputs.i64_values(0)) {
    plan.i64_values = std::move(*input_values);
    plan.has_i64_values = true;
  }
  return plan;
}

RuntimeValuePlan
plan_squeeze_unsqueeze_runtime_values(const RuntimeInputResolver &inputs,
                                      const ov::Node &node,
                                      std::string_view stage_name) {
  ov::Shape input_shape = inputs.shape(0);
  OPENVINO_ASSERT(!input_shape.empty(), "GFX MLIR: ", node.get_type_name(),
                  " input shape is unknown for stage ", stage_name);

  ov::Shape output_shape;
  if (auto squeeze = dynamic_cast<const ov::op::v0::Squeeze *>(&node)) {
    std::vector<int64_t> axes;
    if (squeeze->get_input_size() > 1) {
      axes = constant_source_i64(squeeze->input_value(1), "Squeeze axes",
                                 stage_name);
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
  } else if (auto unsqueeze =
                 dynamic_cast<const ov::op::v0::Unsqueeze *>(&node)) {
    auto axes = constant_source_i64(unsqueeze->input_value(1), "Unsqueeze axes",
                                    stage_name);
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
  plan.output_type = resolve_runtime_input_type(inputs, node, 0);
  plan.force_output_type = true;
  if (auto input_values = inputs.i64_values(0)) {
    plan.i64_values = std::move(*input_values);
    plan.has_i64_values = true;
  }
  return plan;
}

RuntimeValuePlan
plan_shape_preserving_runtime_values(const RuntimeInputResolver &inputs,
                                     const ov::Node &node,
                                     std::string_view stage_name) {
  ov::Shape output_shape;
  if (!inputs.shape_known(0, output_shape)) {
    output_shape = inputs.shape(0);
  }
  OPENVINO_ASSERT(
      !output_shape.empty(),
      "GFX MLIR: shape-preserving input shape is unknown for stage ",
      stage_name);

  RuntimeValuePlan plan;
  plan.output_shape = std::move(output_shape);
  plan.value_shape = plan.output_shape;
  plan.output_type = node.get_output_element_type(0);
  plan.force_output_type = true;
  return plan;
}

RuntimeValuePlan plan_shapeof_runtime_values(const RuntimeInputResolver &inputs,
                                             const ov::Node *node,
                                             std::string_view stage_name) {
  ov::Shape in_shape = inputs.shape(0);
  if (in_shape.empty()) {
    OPENVINO_THROW("GFX MLIR: ShapeOf input shape is unknown for stage ",
                   stage_name);
  }
  const auto output_type =
      node ? node->get_output_element_type(0) : ov::element::i64;
  OPENVINO_ASSERT(output_type == ov::element::i32 ||
                      output_type == ov::element::i64,
                  "GFX MLIR: ShapeOf output must be i32/i64");

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

RuntimeValuePlan plan_broadcast_runtime_values(
    const RuntimeInputResolver &inputs, const ov::Node &node,
    const ov::Shape &input_shape, std::string_view stage_name) {
  bool bidirectional_broadcast = false;
  if (auto broadcast_v3 = dynamic_cast<const ov::op::v3::Broadcast *>(&node)) {
    bidirectional_broadcast = broadcast_v3->get_broadcast_spec().m_type ==
                              ov::op::BroadcastType::BIDIRECTIONAL;
  }

  const auto out_pshape = node.get_output_partial_shape(0);
  OPENVINO_ASSERT(out_pshape.rank().is_static(),
                  "GFX MLIR: Broadcast output rank is dynamic for stage ",
                  stage_name);
  const size_t out_rank = static_cast<size_t>(out_pshape.rank().get_length());
  OPENVINO_ASSERT(
      out_rank <= 8,
      "GFX MLIR: Broadcast rank exceeds kernel metadata capacity for stage ",
      stage_name);
  const size_t in_rank = input_shape.size();
  OPENVINO_ASSERT(
      in_rank <= out_rank,
      "GFX MLIR: Broadcast input rank exceeds output rank for stage ",
      stage_name);

  ov::Shape out_shape;
  if (node.get_input_size() > 1) {
    if (auto target = inputs.i64_values(1)) {
      ov::Shape target_shape;
      target_shape.reserve(target->size());
      out_shape.reserve(target->size());
      for (auto dim : *target) {
        target_shape.push_back(static_cast<size_t>(std::max<int64_t>(dim, 0)));
      }
      out_shape = bidirectional_broadcast
                      ? compute_binary_broadcast_shape(input_shape,
                                                       target_shape, stage_name)
                      : target_shape;
      if (bidirectional_broadcast) {
        OPENVINO_ASSERT(out_shape.size() == out_rank,
                        "GFX MLIR: Broadcast bidirectional output rank "
                        "mismatch for stage ",
                        stage_name, " input=", input_shape,
                        " target=", target_shape, " output=", out_shape);
      }
    }
  }
  if (out_shape.empty()) {
    out_shape.resize(out_rank, 1);
    for (size_t i = 0; i < out_rank; ++i) {
      const auto &dim = out_pshape[i];
      if (dim.is_static()) {
        out_shape[i] = static_cast<size_t>(dim.get_length());
        continue;
      }
      const size_t aligned_input_axis = out_rank - in_rank;
      if (i >= aligned_input_axis && !input_shape.empty()) {
        const size_t input_axis = i - aligned_input_axis;
        if (input_axis < input_shape.size() && input_shape[input_axis] != 1) {
          out_shape[i] = input_shape[input_axis];
        }
      }
    }
  }
  OPENVINO_ASSERT(out_shape.size() == out_rank,
                  "GFX MLIR: Broadcast target shape rank mismatch for stage ",
                  stage_name);

  RuntimeValuePlan plan;
  plan.output_shape = std::move(out_shape);
  plan.value_shape = input_shape;
  plan.output_type = node.get_output_element_type(0);
  if (auto input_values = inputs.i64_values(0)) {
    plan.i64_values = std::move(*input_values);
    plan.has_i64_values = true;
  }
  return plan;
}

RuntimeValuePlan plan_convert_runtime_values(const RuntimeInputResolver &inputs,
                                             const ov::Node *node,
                                             std::string_view stage_name) {
  ov::Shape in_shape = inputs.shape(0);
  OPENVINO_ASSERT(!in_shape.empty(),
                  "GFX MLIR: Convert input shape is unknown for stage ",
                  stage_name);
  RuntimeValuePlan plan;
  plan.output_shape = in_shape;
  plan.value_shape = std::move(in_shape);
  plan.output_type =
      node ? node->get_output_element_type(0) : ov::element::dynamic;
  if (auto input_values = inputs.i64_values(0)) {
    plan.i64_values = std::move(*input_values);
    plan.has_i64_values = true;
  }
  return plan;
}

RuntimeValuePlan plan_range_runtime_values(const RuntimeInputResolver &inputs,
                                           const ov::Node *node,
                                           std::string_view stage_name) {
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
  } else if (node && node->get_output_partial_shape(0).is_static()) {
    plan.output_shape = node->get_output_shape(0);
  }
  OPENVINO_ASSERT(!plan.output_shape.empty(),
                  "GFX MLIR: Range output shape is unknown for stage ",
                  stage_name);

  plan.value_shape = plan.output_shape;
  plan.output_type = node ? node->get_output_element_type(0) : ov::element::i64;
  plan.force_output_type = true;
  return plan;
}

RuntimeSelectPlan plan_select_runtime_values(const RuntimeInputResolver &inputs,
                                             const ov::Node &node,
                                             std::string_view stage_name) {
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
  plan.values.value_shape = plan.values.output_shape;
  plan.values.output_type = node.get_output_element_type(0);
  plan.available = true;
  return plan;
}

RuntimeReducePlan
plan_reduce_runtime_values(const RuntimeInputResolver &inputs,
                           const ov::Node *node, std::string_view reduce_type,
                           const RuntimeReduceInfo &reduce_info,
                           std::string_view stage_name) {
  RuntimeReducePlan plan;
  plan.input_shape = inputs.shape(0);
  OPENVINO_ASSERT(!plan.input_shape.empty(),
                  "GFX MLIR: Reduce input shape is unknown for stage ",
                  stage_name);
  const size_t rank = plan.input_shape.size();
  OPENVINO_ASSERT(
      rank <= 8,
      "GFX MLIR: Reduce rank exceeds kernel metadata capacity for stage ",
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

  plan.values.value_shape = plan.values.output_shape;
  plan.values.output_type =
      node ? node->get_output_element_type(0) : ov::element::dynamic;
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
  } else if (inputs.node &&
             inputs.node->get_output_size() > 0 &&
             inputs.node->get_output_partial_shape(0).is_static()) {
    plan.output_shape = inputs.node->get_output_shape(0);
  } else if (auto repeats = inputs.i64_values(1)) {
    OPENVINO_ASSERT(repeats->size() == plan.input_shape.size(),
                    "GFX MLIR: Tile repeats rank mismatch for stage ",
                    stage_name);
    plan.output_shape.reserve(plan.input_shape.size());
    for (size_t axis = 0; axis < plan.input_shape.size(); ++axis) {
      OPENVINO_ASSERT((*repeats)[axis] > 0,
                      "GFX MLIR: Tile repeats must be positive for stage ",
                      stage_name);
      plan.output_shape.push_back(
          plan.input_shape[axis] * static_cast<size_t>((*repeats)[axis]));
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
  plan.values.output_type =
      inputs.node ? inputs.node->get_output_element_type(0) : ov::element::dynamic;
  plan.scalar_args = {
      static_cast<int32_t>(ov::shape_size(plan.output_shape)),
      static_cast<int32_t>(std::max<size_t>(plan.output_shape.size(), 1))};
  plan.available = true;
  return plan;
}

RuntimeConcatPlan plan_concat_runtime_values(const RuntimeInputResolver &inputs,
                                             const ov::Node &node,
                                             std::string_view stage_name) {
  const auto *concat = dynamic_cast<const ov::op::v0::Concat *>(&node);
  OPENVINO_ASSERT(concat, "GFX MLIR: expected Concat node for stage ",
                  stage_name);

  RuntimeConcatPlan plan;
  plan.input_shapes.reserve(concat->get_input_size());
  ov::Shape out_shape;
  for (size_t input_idx = 0; input_idx < concat->get_input_size();
       ++input_idx) {
    const ov::Shape input_shape = inputs.shape(input_idx);
    if (input_shape.empty()) {
      OPENVINO_THROW("GFX MLIR: Concat input shape is unknown for stage ",
                     stage_name);
    }
    if (out_shape.empty()) {
      out_shape = input_shape;
    } else {
      OPENVINO_ASSERT(input_shape.size() == out_shape.size(),
                      "GFX MLIR: Concat rank mismatch for stage ", stage_name);
    }
    plan.input_shapes.push_back(input_shape);
  }
  OPENVINO_ASSERT(!out_shape.empty(),
                  "GFX MLIR: Concat has no resolved inputs for stage ",
                  stage_name);

  plan.axis_norm =
      normalize_axis(concat->get_axis(), out_shape.size(), "GFX MLIR: Concat");
  size_t axis_total = 0;
  for (const auto &input_shape : plan.input_shapes) {
    OPENVINO_ASSERT(input_shape.size() == out_shape.size(),
                    "GFX MLIR: Concat rank mismatch for stage ", stage_name);
    for (size_t dim = 0; dim < out_shape.size(); ++dim) {
      if (static_cast<int64_t>(dim) == plan.axis_norm) {
        continue;
      }
      OPENVINO_ASSERT(input_shape[dim] == out_shape[dim],
                      "GFX MLIR: Concat non-axis dim mismatch for stage ",
                      stage_name);
    }
    axis_total += input_shape[static_cast<size_t>(plan.axis_norm)];
  }
  out_shape[static_cast<size_t>(plan.axis_norm)] = axis_total;

  plan.values.output_shape = out_shape;
  plan.values.value_shape = out_shape;
  plan.values.output_type = node.get_output_element_type(0);

  std::vector<std::vector<int64_t>> concat_values;
  concat_values.reserve(concat->get_input_size());
  bool all_values_resolved = true;
  for (size_t input_idx = 0; input_idx < concat->get_input_size();
       ++input_idx) {
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
                                             const ov::Node &node,
                                             std::string_view stage_name) {
  if (auto gather_v7 = dynamic_cast<const ov::op::v7::Gather *>(&node)) {
    OPENVINO_ASSERT(gather_v7->get_batch_dims() == 0,
                    "GFX MLIR: Gather v7 batch_dims not supported for stage ",
                    stage_name);
  }
  if (auto gather_v8 = dynamic_cast<const ov::op::v8::Gather *>(&node)) {
    OPENVINO_ASSERT(gather_v8->get_batch_dims() == 0,
                    "GFX MLIR: Gather v8 batch_dims not supported for stage ",
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
  const auto axis_values =
      constant_source_i64(node.input_value(2), "Gather axis", stage_name);
  OPENVINO_ASSERT(axis_values.size() == 1,
                  "GFX MLIR: Gather axis must be scalar for stage ",
                  stage_name);
  plan.axis_norm = normalize_axis(axis_values[0], plan.data_shape.size(),
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
  plan.values.output_type = node.get_output_element_type(0);
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
                                   const ov::Node &node,
                                   std::string_view stage_name) {
  RuntimeScatterUpdatePlan plan;
  if (!inputs.shape_known(0, plan.values.output_shape) ||
      !inputs.shape_known(1, plan.indices_shape) ||
      !inputs.shape_known(2, plan.updates_shape)) {
    return plan;
  }
  const auto axis_values = constant_source_i64(
      node.input_value(3), "ScatterUpdate axis", stage_name);
  OPENVINO_ASSERT(axis_values.size() == 1,
                  "GFX MLIR: ScatterUpdate axis must be scalar for stage ",
                  stage_name);
  plan.axis_norm = axis_values[0];
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
  plan.values.output_type = node.get_output_element_type(0);
  plan.available = true;
  return plan;
}

RuntimeSplitPlan plan_split_runtime_values(const ov::Node *node,
                                           const ov::Shape &input_shape,
                                           size_t output_count,
                                           std::string_view stage_name) {
  RuntimeSplitPlan plan;
  plan.input_shape = input_shape;

  int64_t axis = 0;
  size_t parts = 0;
  bool is_split = false;
  if (auto split = dynamic_cast<const ov::op::v1::Split *>(node)) {
    const auto axis_values =
        constant_source_i64(split->input_value(1), "Split axis", stage_name);
    OPENVINO_ASSERT(axis_values.size() == 1,
                    "GFX MLIR: Split axis must be scalar for stage ",
                    stage_name);
    axis = axis_values[0];
    parts = split->get_num_splits();
    is_split = true;
  } else if (auto variadic =
                 dynamic_cast<const ov::op::v1::VariadicSplit *>(node)) {
    const auto axis_values = constant_source_i64(
        variadic->input_value(1), "VariadicSplit axis", stage_name);
    OPENVINO_ASSERT(axis_values.size() == 1,
                    "GFX MLIR: VariadicSplit axis must be scalar for stage ",
                    stage_name);
    axis = axis_values[0];
    auto lengths = constant_source_i64(variadic->input_value(2),
                                       "VariadicSplit lengths", stage_name);
    plan.split_sizes.reserve(lengths.size());
    int64_t infer_index = -1;
    size_t known_sum = 0;
    for (size_t i = 0; i < lengths.size(); ++i) {
      const int64_t length = lengths[i];
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
          normalize_axis(axis, input_shape.size(), "GFX MLIR: VariadicSplit");
      const size_t axis_len = input_shape[static_cast<size_t>(axis_norm)];
      OPENVINO_ASSERT(known_sum <= axis_len,
                      "GFX MLIR: VariadicSplit known lengths exceed axis "
                      "dimension for stage ",
                      stage_name);
      plan.split_sizes[static_cast<size_t>(infer_index)] = axis_len - known_sum;
    }
  } else {
    OPENVINO_THROW("GFX MLIR: expected Split/VariadicSplit node for stage ",
                   stage_name);
  }

  plan.axis_norm = normalize_axis(axis, input_shape.size(), "GFX MLIR: Split");
  plan.axis_len =
      static_cast<uint32_t>(input_shape[static_cast<size_t>(plan.axis_norm)]);

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
       d < input_shape.size(); ++d) {
    plan.inner_stride *= input_shape[d];
  }
  plan.available = true;
  return plan;
}

RuntimeSlicePlan
plan_slice_runtime_values(const RuntimeInputResolver &inputs,
                          const std::vector<GpuTensor *> &outputs,
                          bool requires_runtime_shape_args,
                          std::string_view stage_name) {
  OPENVINO_ASSERT(inputs.node,
                  "GFX MLIR: Slice/StridedSlice node is missing for stage ",
                  stage_name);
  const auto &node = *inputs.node;

  RuntimeSlicePlan plan;
  plan.use_runtime_args = requires_runtime_shape_args ||
                          !node.get_input_partial_shape(0).is_static() ||
                          !node.get_output_partial_shape(0).is_static() ||
                          slice_requires_runtime_indexing(node);
  plan.input_shape = inputs.shape(0);
  if (plan.input_shape.empty()) {
    OPENVINO_THROW(
        "GFX MLIR: Slice/StridedSlice input shape is unknown for stage ",
        stage_name);
  }

  plan.values.output_shape =
      infer_slice_output_shape(inputs, node, plan.input_shape, outputs, stage_name);
  OPENVINO_ASSERT(
      !plan.values.output_shape.empty(),
      "GFX MLIR: Slice/StridedSlice output shape is unknown for stage ",
      stage_name);
  OPENVINO_ASSERT(plan.input_shape.size() == plan.values.output_shape.size(),
                  "GFX MLIR: Slice/StridedSlice rank mismatch for stage ",
                  stage_name);

  build_slice_runtime_spec(inputs, node, plan.input_shape, plan.values.output_shape,
                           plan.starts_full, plan.steps_full, stage_name);

  plan.values.value_shape = plan.values.output_shape;
  plan.values.output_type = node.get_output_element_type(0);
  plan.values.force_output_type = true;
  plan.linear_view =
      slice_is_runtime_linear_view(plan.input_shape, plan.values.output_shape,
                                   plan.starts_full, plan.steps_full);
  plan.available = true;
  return plan;
}

RuntimeTransposePlan
plan_transpose_runtime_values(const RuntimeInputResolver &inputs,
                              const ov::Node &node,
                              std::string_view stage_name) {
  const auto *transpose = dynamic_cast<const ov::op::v1::Transpose *>(&node);
  OPENVINO_ASSERT(transpose, "GFX MLIR: expected Transpose node for stage ",
                  stage_name);

  RuntimeTransposePlan plan;
  plan.input_shape = inputs.shape(0);
  if (plan.input_shape.empty()) {
    OPENVINO_THROW("GFX MLIR: Transpose input shape is unknown for stage ",
                   stage_name);
  }
  plan.permutation = constant_source_i64(transpose->input_value(1),
                                         "Transpose perm", stage_name);
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
  plan.values.output_type = node.get_output_element_type(0);
  plan.values.force_output_type = true;
  plan.linear_view = transpose_is_runtime_linear_view(
      plan.input_shape, plan.values.output_shape, plan.permutation, stage_name);
  plan.available = true;
  return plan;
}

RuntimeInterpolatePlan plan_interpolate_runtime_values(
    const RuntimeInputResolver &inputs, const std::vector<GpuTensor *> &outputs,
    const ov::Node &node, bool is_opencl_backend, std::string_view stage_name) {
  RuntimeInterpolatePlan plan;
  plan.input_shape = inputs.shape(0);
  if (plan.input_shape.empty()) {
    OPENVINO_THROW("GFX MLIR: Interpolate input shape is unknown for stage ",
                   stage_name);
  }
  plan.values.output_shape = resolve_primary_output_shape(node, outputs);
  OPENVINO_ASSERT(!plan.values.output_shape.empty(),
                  "GFX MLIR: Interpolate output shape is unknown for stage ",
                  stage_name);
  OPENVINO_ASSERT(
      plan.input_shape.size() == 4 && plan.values.output_shape.size() == 4,
      "GFX MLIR: Interpolate expects NCHW rank4 for stage ", stage_name);

  if (auto v0 = dynamic_cast<const ov::op::v0::Interpolate *>(&node)) {
    plan.align_corners = v0->get_attrs().align_corners;
    plan.use_half_pixel = !plan.align_corners;
  } else if (auto v4 = dynamic_cast<const ov::op::v4::Interpolate *>(&node)) {
    using Base = ov::op::util::InterpolateBase;
    plan.align_corners = v4->get_attrs().coordinate_transformation_mode ==
                         Base::CoordinateTransformMode::ALIGN_CORNERS;
    plan.use_half_pixel = v4->get_attrs().coordinate_transformation_mode ==
                          Base::CoordinateTransformMode::HALF_PIXEL;
    switch (v4->get_attrs().nearest_mode) {
    case Base::NearestMode::FLOOR:
    case Base::NearestMode::ROUND_PREFER_FLOOR:
      plan.nearest_mode = 1;
      break;
    case Base::NearestMode::CEIL:
    case Base::NearestMode::ROUND_PREFER_CEIL:
      plan.nearest_mode = 2;
      break;
    case Base::NearestMode::SIMPLE:
    default:
      plan.nearest_mode = 0;
      break;
    }
  } else if (auto v11 = dynamic_cast<const ov::op::v11::Interpolate *>(&node)) {
    using Base = ov::op::util::InterpolateBase;
    plan.align_corners = v11->get_attrs().coordinate_transformation_mode ==
                         Base::CoordinateTransformMode::ALIGN_CORNERS;
    plan.use_half_pixel = v11->get_attrs().coordinate_transformation_mode ==
                          Base::CoordinateTransformMode::HALF_PIXEL;
    switch (v11->get_attrs().nearest_mode) {
    case Base::NearestMode::FLOOR:
    case Base::NearestMode::ROUND_PREFER_FLOOR:
      plan.nearest_mode = 1;
      break;
    case Base::NearestMode::CEIL:
    case Base::NearestMode::ROUND_PREFER_CEIL:
      plan.nearest_mode = 2;
      break;
    case Base::NearestMode::SIMPLE:
    default:
      plan.nearest_mode = 0;
      break;
    }
  } else {
    OPENVINO_THROW("GFX MLIR: unsupported Interpolate op kind for stage ",
                   stage_name);
  }
  plan.values.value_shape = plan.values.output_shape;
  plan.values.output_type = node.get_output_element_type(0);
  plan.values.force_output_type = true;
  plan.use_runtime_params = !is_opencl_backend;
  plan.available = true;
  return plan;
}

RuntimeSoftmaxPlan
plan_softmax_runtime_values(const RuntimeInputResolver &inputs,
                            const ov::Node &node, std::string_view stage_name) {
  RuntimeSoftmaxPlan plan;
  plan.values.output_shape = inputs.shape(0);
  if (plan.values.output_shape.empty()) {
    OPENVINO_THROW("GFX MLIR: Softmax input shape is unknown for stage ",
                   stage_name);
  }
  int64_t axis = -1;
  if (auto s1 = dynamic_cast<const ov::op::v1::Softmax *>(&node)) {
    axis = s1->get_axis();
  } else if (auto s8 = dynamic_cast<const ov::op::v8::Softmax *>(&node)) {
    axis = s8->get_axis();
  } else if (auto ls = dynamic_cast<const ov::op::v5::LogSoftmax *>(&node)) {
    axis = ls->get_axis();
    plan.log_softmax = true;
  } else {
    OPENVINO_THROW("GFX MLIR: unsupported softmax op kind for stage ",
                   stage_name);
  }
  const auto dims =
      compute_softmax_dims(plan.values.output_shape, axis, "GFX MLIR: Softmax");
  plan.axis = dims.axis;
  plan.rows = dims.rows;
  plan.axis_len = dims.axis_len;
  plan.inner = dims.inner;
  plan.values.value_shape = plan.values.output_shape;
  plan.values.output_type = node.get_output_element_type(0);
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

bool bind_small_i64_const_stage_outputs(
    GpuBufferManager *buffer_manager, const std::vector<GpuTensor *> &outputs,
    std::vector<GpuTensor> &cache, const std::shared_ptr<const ov::Node> &node,
    GfxProfiler *profiler, bool profiling_enabled, std::string_view stage_name,
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
    const auto type = out->expected_type == ov::element::dynamic && node &&
                              node->get_output_size() > 0
                          ? node->get_output_element_type(0)
                          : out->expected_type;
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
