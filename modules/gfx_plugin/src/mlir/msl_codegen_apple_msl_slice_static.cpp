// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen_apple_msl_slice_static.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <sstream>
#include <vector>

#include "mlir/codegen_common.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

std::vector<int64_t> get_slice_const_i64(const ov::Output<ov::Node> &source,
                                         const char *what) {
  auto c = ov::util::get_constant_from_source(source);
  OPENVINO_ASSERT(c, "GFX Metal Slice: ", what, " must be Constant");
  return c->cast_vector<int64_t>();
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

struct StaticSliceMeta {
  std::vector<uint32_t> out_shape;
  std::vector<uint32_t> in_stride;
  std::vector<int32_t> starts;
  std::vector<int32_t> steps;
  uint32_t total = 0;
};

StaticSliceMeta
build_static_slice_meta(const std::shared_ptr<const ov::Node> &node) {
  OPENVINO_ASSERT(node, "GFX Metal Slice: node is null");
  const auto in_shape = node->get_input_shape(0);
  const auto out_shape = node->get_output_shape(0);
  const size_t rank = in_shape.size();
  OPENVINO_ASSERT(
      rank == out_shape.size(),
      "GFX Metal Slice: rank-changing Slice/StridedSlice is not supported");

  StaticSliceMeta meta;
  meta.out_shape.reserve(rank);
  meta.starts.assign(rank, 0);
  meta.steps.assign(rank, 1);
  meta.in_stride.assign(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    meta.out_shape.push_back(static_cast<uint32_t>(out_shape[i]));
  }
  for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
    meta.in_stride[static_cast<size_t>(i)] =
        meta.in_stride[static_cast<size_t>(i + 1)] *
        static_cast<uint32_t>(in_shape[static_cast<size_t>(i + 1)]);
  }
  meta.total = static_cast<uint32_t>(ov::shape_size(out_shape));

  if (auto slice = ov::as_type_ptr<const ov::op::v8::Slice>(node)) {
    auto starts = get_slice_const_i64(slice->input_value(1), "Slice starts");
    auto ends = get_slice_const_i64(slice->input_value(2), "Slice ends");
    auto steps = get_slice_const_i64(slice->input_value(3), "Slice steps");
    std::vector<int64_t> axes;
    if (slice->get_input_size() > 4) {
      axes = get_slice_const_i64(slice->input_value(4), "Slice axes");
    } else {
      axes.resize(starts.size());
      std::iota(axes.begin(), axes.end(), 0);
    }
    OPENVINO_ASSERT(starts.size() == ends.size() &&
                        starts.size() == steps.size() &&
                        starts.size() == axes.size(),
                    "GFX Metal Slice: starts/ends/steps/axes size mismatch");
    for (size_t i = 0; i < axes.size(); ++i) {
      int64_t axis = axes[i];
      if (axis < 0) {
        axis += static_cast<int64_t>(rank);
      }
      OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < rank,
                      "GFX Metal Slice: axis out of range");
      OPENVINO_ASSERT(steps[i] != 0,
                      "GFX Metal Slice: zero step is not supported");
      const auto dim =
          static_cast<int64_t>(in_shape[static_cast<size_t>(axis)]);
      meta.starts[static_cast<size_t>(axis)] =
          static_cast<int32_t>(normalize_slice_index(starts[i], dim, true));
      meta.steps[static_cast<size_t>(axis)] = static_cast<int32_t>(steps[i]);
    }
    return meta;
  }

  auto slice = ov::as_type_ptr<const ov::op::v1::StridedSlice>(node);
  OPENVINO_ASSERT(slice, "GFX Metal Slice: expected Slice/StridedSlice node");
  OPENVINO_ASSERT(
      std::all_of(slice->get_new_axis_mask().begin(),
                  slice->get_new_axis_mask().end(),
                  [](int64_t v) { return v == 0; }),
      "GFX Metal Slice: StridedSlice new_axis_mask is not supported");
  OPENVINO_ASSERT(
      std::all_of(slice->get_shrink_axis_mask().begin(),
                  slice->get_shrink_axis_mask().end(),
                  [](int64_t v) { return v == 0; }),
      "GFX Metal Slice: StridedSlice shrink_axis_mask is not supported");
  OPENVINO_ASSERT(
      std::all_of(slice->get_ellipsis_mask().begin(),
                  slice->get_ellipsis_mask().end(),
                  [](int64_t v) { return v == 0; }),
      "GFX Metal Slice: StridedSlice ellipsis_mask is not supported");

  auto begin = get_slice_const_i64(slice->input_value(1), "StridedSlice begin");
  auto end = get_slice_const_i64(slice->input_value(2), "StridedSlice end");
  std::vector<int64_t> strides(rank, 1);
  if (slice->get_input_size() > 3) {
    auto values =
        get_slice_const_i64(slice->input_value(3), "StridedSlice strides");
    OPENVINO_ASSERT(values.size() <= rank,
                    "GFX Metal Slice: StridedSlice strides rank mismatch");
    std::copy(values.begin(), values.end(), strides.begin());
  }
  const auto &begin_mask = slice->get_begin_mask();
  const auto &end_mask = slice->get_end_mask();
  for (size_t axis = 0; axis < rank; ++axis) {
    const auto dim = static_cast<int64_t>(in_shape[axis]);
    const bool masked_begin = axis < begin_mask.size() && begin_mask[axis] != 0;
    const bool masked_end = axis < end_mask.size() && end_mask[axis] != 0;
    const int64_t step = strides[axis];
    OPENVINO_ASSERT(step != 0,
                    "GFX Metal Slice: StridedSlice zero step is not supported");
    int64_t start = axis < begin.size() ? begin[axis] : 0;
    int64_t finish = axis < end.size() ? end[axis] : dim;
    start = masked_begin ? (step < 0 ? dim - 1 : 0)
                         : normalize_slice_index(start, dim, true);
    finish = masked_end ? (step < 0 ? -1 : dim)
                        : normalize_slice_index(finish, dim, false);
    (void)finish;
    meta.starts[axis] = static_cast<int32_t>(start);
    meta.steps[axis] = static_cast<int32_t>(step);
  }
  return meta;
}

} // namespace

std::string
generate_static_msl_for_slice(const std::shared_ptr<const ov::Node> &node,
                              const ov::element::Type &storage_type) {
  const auto meta = build_static_slice_meta(node);
  const auto scalar_t = msl_type_from_element(
      storage_type == ov::element::dynamic ? ov::element::f32 : storage_type);
  const uint32_t rank = static_cast<uint32_t>(meta.out_shape.size());
  std::ostringstream ss;
  ss << "#include <metal_stdlib>\nusing namespace metal;\n";
  ss << "using scalar_t = " << scalar_t << ";\n";
  ss << "constant uint TOTAL_C = " << meta.total << ";\n";
  ss << "constant uint RANK_C = " << rank << ";\n";
  ss << "constant uint OUT_SHAPE_C[" << rank << "] = {";
  for (size_t i = 0; i < meta.out_shape.size(); ++i) {
    if (i) {
      ss << ", ";
    }
    ss << meta.out_shape[i];
  }
  ss << "};\n";
  ss << "constant uint IN_STRIDE_C[" << rank << "] = {";
  for (size_t i = 0; i < meta.in_stride.size(); ++i) {
    if (i) {
      ss << ", ";
    }
    ss << meta.in_stride[i];
  }
  ss << "};\n";
  ss << "constant int STARTS_C[" << rank << "] = {";
  for (size_t i = 0; i < meta.starts.size(); ++i) {
    if (i) {
      ss << ", ";
    }
    ss << meta.starts[i];
  }
  ss << "};\n";
  ss << "constant int STEPS_C[" << rank << "] = {";
  for (size_t i = 0; i < meta.steps.size(); ++i) {
    if (i) {
      ss << ", ";
    }
    ss << meta.steps[i];
  }
  ss << "};\n";
  ss << "kernel void slice_kernel(\n";
  ss << "  device const scalar_t* A [[buffer(0)]],\n";
  ss << "  device scalar_t* C [[buffer(1)]],\n";
  ss << "  uint gid [[thread_position_in_grid]]) {\n";
  ss << "    if (gid >= TOTAL_C) return;\n";
  ss << "    uint idx = gid;\n";
  ss << "    int in_off = 0;\n";
  ss << "    for (int d = (int)RANK_C - 1; d >= 0; --d) {\n";
  ss << "        uint coord = idx % OUT_SHAPE_C[d];\n";
  ss << "        idx /= OUT_SHAPE_C[d];\n";
  ss << "        in_off += (STARTS_C[d] + int(coord) * STEPS_C[d]) * "
        "int(IN_STRIDE_C[d]);\n";
  ss << "    }\n";
  ss << "    C[gid] = A[in_off];\n";
  ss << "}\n";
  return ss.str();
}

} // namespace gfx_plugin
} // namespace ov
