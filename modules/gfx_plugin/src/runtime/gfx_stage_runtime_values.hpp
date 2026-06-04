// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "kernel_ir/gfx_kernel_manifest.hpp"
#include "openvino/core/axis_set.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "runtime/gpu_tensor.hpp"

namespace ov {
namespace gfx_plugin {

class GfxProfiler;
class GpuBufferManager;

struct RuntimeReduceInfo {
  ov::AxisSet axes;
  bool keep_dims = false;
};

enum class RuntimeInputShapePolicy {
  TensorOrStatic,
  TensorOrStaticOrConstant,
};

struct RuntimeInputResolver {
  const std::vector<GpuTensor *> *inputs = nullptr;
  const std::vector<GpuTensor> *const_buffers = nullptr;
  const std::vector<bool> *const_buffer_present = nullptr;
  std::shared_ptr<const ov::Node> node;

  ov::Shape shape(size_t idx) const;
  bool shape_known(size_t idx, ov::Shape &shape,
                   RuntimeInputShapePolicy policy =
                       RuntimeInputShapePolicy::TensorOrStatic) const;
  GpuTensor *tensor(size_t input_idx) const;
  std::optional<std::vector<int64_t>> i64_values(size_t input_idx) const;
  void ensure_output_shape(size_t output_idx, GpuTensor *out) const;
};

struct RuntimeValuePlan {
  ov::Shape output_shape;
  ov::Shape value_shape;
  ov::element::Type output_type = ov::element::dynamic;
  bool force_output_type = false;
  std::vector<int64_t> i64_values;
  bool has_i64_values = false;
};

struct RuntimeSelectPlan {
  RuntimeValuePlan values;
  ov::Shape condition_shape;
  ov::Shape true_shape;
  ov::Shape false_shape;
  bool available = false;

  bool valid() const { return available; }
};

struct RuntimeReducePlan {
  RuntimeValuePlan values;
  ov::Shape input_shape;
  bool available = false;

  bool valid() const { return available; }
};

struct RuntimeTilePlan {
  RuntimeValuePlan values;
  ov::Shape input_shape;
  ov::Shape output_shape;
  std::vector<int32_t> scalar_args;
  bool available = false;

  bool valid() const { return available; }
};

struct RuntimeConcatPlan {
  RuntimeValuePlan values;
  std::vector<ov::Shape> input_shapes;
  int64_t axis_norm = 0;
  bool available = false;

  bool valid() const { return available; }
};

struct RuntimeGatherPlan {
  RuntimeValuePlan values;
  ov::Shape data_shape;
  ov::Shape indices_shape;
  int64_t axis_norm = 0;
  uint32_t axis_dim = 0;
  uint32_t indices_count = 0;
  bool identity_view = false;
  bool available = false;

  bool valid() const { return available; }
};

struct RuntimeScatterUpdatePlan {
  RuntimeValuePlan values;
  ov::Shape indices_shape;
  ov::Shape updates_shape;
  int64_t axis_norm = 0;
  bool available = false;

  bool valid() const { return available; }
};

struct RuntimeSplitPlan {
  ov::Shape input_shape;
  std::vector<size_t> split_sizes;
  int64_t axis_norm = 0;
  uint32_t axis_len = 0;
  uint32_t inner_stride = 1;
  bool available = false;

  bool valid() const { return available; }
};

struct RuntimeSlicePlan {
  RuntimeValuePlan values;
  ov::Shape input_shape;
  std::vector<int32_t> starts_full;
  std::vector<int32_t> steps_full;
  bool use_runtime_args = false;
  bool linear_view = false;
  bool available = false;

  bool valid() const { return available; }
};

struct RuntimeTransposePlan {
  RuntimeValuePlan values;
  ov::Shape input_shape;
  std::vector<int64_t> permutation;
  bool linear_view = false;
  bool available = false;

  bool valid() const { return available; }
};

struct RuntimeInterpolatePlan {
  RuntimeValuePlan values;
  ov::Shape input_shape;
  bool align_corners = false;
  bool use_half_pixel = true;
  uint32_t nearest_mode = 0;
  bool available = false;

  bool valid() const { return available; }
};

struct RuntimeSoftmaxPlan {
  RuntimeValuePlan values;
  int64_t axis = 0;
  uint64_t rows = 1;
  uint64_t axis_len = 0;
  uint64_t inner = 1;
  bool log_softmax = false;
  bool available = false;

  bool valid() const { return available; }
};

RuntimeValuePlan plan_reshape_runtime_values(const RuntimeInputResolver &inputs,
                                             const ov::Node &node,
                                             std::string_view stage_name);

RuntimeValuePlan
plan_squeeze_unsqueeze_runtime_values(const RuntimeInputResolver &inputs,
                                      const ov::Node &node,
                                      std::string_view stage_name);

RuntimeValuePlan
plan_shape_preserving_runtime_values(const RuntimeInputResolver &inputs,
                                     const ov::Node &node,
                                     std::string_view stage_name);

ov::Shape compute_binary_broadcast_shape(const ov::Shape &lhs,
                                         const ov::Shape &rhs,
                                         std::string_view stage_name);

RuntimeValuePlan plan_shapeof_runtime_values(const RuntimeInputResolver &inputs,
                                             const ov::Node *node,
                                             std::string_view stage_name);

RuntimeValuePlan plan_broadcast_runtime_values(
    const RuntimeInputResolver &inputs, const ov::Node &node,
    const ov::Shape &input_shape, std::string_view stage_name);

RuntimeValuePlan plan_convert_runtime_values(const RuntimeInputResolver &inputs,
                                             const ov::Node *node,
                                             std::string_view stage_name);

RuntimeValuePlan plan_range_runtime_values(const RuntimeInputResolver &inputs,
                                           const ov::Node *node,
                                           std::string_view stage_name);

RuntimeSelectPlan plan_select_runtime_values(const RuntimeInputResolver &inputs,
                                             const ov::Node &node,
                                             std::string_view stage_name);

RuntimeReducePlan
plan_reduce_runtime_values(const RuntimeInputResolver &inputs,
                           const ov::Node *node, std::string_view reduce_type,
                           const RuntimeReduceInfo &reduce_info,
                           std::string_view stage_name);

std::optional<RuntimeReduceInfo>
get_runtime_reduce_info(const std::shared_ptr<const ov::Node> &node);

RuntimeTilePlan
plan_tile_runtime_values(const RuntimeInputResolver &inputs,
                         const std::vector<GpuTensor *> &outputs,
                         std::string_view stage_name);

RuntimeConcatPlan plan_concat_runtime_values(const RuntimeInputResolver &inputs,
                                             const ov::Node &node,
                                             std::string_view stage_name);

RuntimeGatherPlan plan_gather_runtime_values(const RuntimeInputResolver &inputs,
                                             const ov::Node &node,
                                             std::string_view stage_name);

RuntimeScatterUpdatePlan
plan_scatter_update_runtime_values(const RuntimeInputResolver &inputs,
                                   const ov::Node &node,
                                   std::string_view stage_name);

RuntimeSplitPlan plan_split_runtime_values(const ov::Node *node,
                                           const ov::Shape &input_shape,
                                           size_t output_count,
                                           std::string_view stage_name);

RuntimeSlicePlan
plan_slice_runtime_values(const RuntimeInputResolver &inputs,
                          const std::vector<GpuTensor *> &outputs,
                          bool requires_runtime_shape_args,
                          std::string_view stage_name);

RuntimeTransposePlan
plan_transpose_runtime_values(const RuntimeInputResolver &inputs,
                              const ov::Node &node,
                              std::string_view stage_name);

RuntimeInterpolatePlan plan_interpolate_runtime_values(
    const RuntimeInputResolver &inputs, const std::vector<GpuTensor *> &outputs,
    const ov::Node &node, std::string_view stage_name);

RuntimeSoftmaxPlan
plan_softmax_runtime_values(const RuntimeInputResolver &inputs,
                            const ov::Node &node, std::string_view stage_name);

void assign_runtime_value_outputs(const RuntimeValuePlan &plan,
                                  const std::vector<GpuTensor *> &outputs);

void assign_i64_values(GpuTensor *out, const std::vector<int64_t> &values,
                       const ov::Shape &shape);

std::optional<std::vector<int64_t>> compute_reduce_i64_values(
    std::string_view reduce_type, const std::vector<int64_t> &input_values,
    const ov::Shape &input_shape, const RuntimeReduceInfo &reduce_info,
    const ov::Shape &output_shape);

bool bind_small_i64_const_stage_outputs(
    GpuBufferManager *buffer_manager, const std::vector<GpuTensor *> &outputs,
    std::vector<GpuTensor> &cache, const std::shared_ptr<const ov::Node> &node,
    GfxProfiler *profiler, bool profiling_enabled, std::string_view stage_name,
    std::string_view suffix);

} // namespace gfx_plugin
} // namespace ov
