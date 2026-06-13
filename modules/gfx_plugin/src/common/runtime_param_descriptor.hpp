// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <limits>
#include <string_view>

#include "common/artifact_payload.hpp"

namespace ov {
namespace gfx_plugin {

constexpr size_t kRuntimeParamDescriptorUnknownCount =
    std::numeric_limits<size_t>::max();

inline bool
runtime_param_descriptor_family_is_binary_broadcast(std::string_view op_family)
    noexcept {
  return op_family == "Add" || op_family == "Subtract" ||
         op_family == "Multiply" || op_family == "Divide" ||
         op_family == "Power" || op_family == "Mod" ||
         op_family == "FloorMod" || op_family == "Minimum" ||
         op_family == "Maximum" || op_family == "Equal" ||
         op_family == "NotEqual" || op_family == "Less" ||
         op_family == "Greater" || op_family == "LessEqual" ||
         op_family == "GreaterEqual" || op_family == "LogicalAnd" ||
         op_family == "LogicalOr" || op_family == "LogicalXor" ||
         op_family == "SquaredDifference" || op_family == "PRelu";
}

inline bool
runtime_param_descriptor_family_is_reduce(std::string_view op_family) noexcept {
  return op_family == "ReduceSum" || op_family == "ReduceMean" ||
         op_family == "ReduceMax" || op_family == "ReduceMin" ||
         op_family == "ReduceProd" || op_family == "ReduceL1" ||
         op_family == "ReduceL2" || op_family == "ReduceLogicalAnd" ||
         op_family == "ReduceLogicalOr";
}

inline RuntimeParamDescriptorPayloadKind
runtime_param_descriptor_payload_kind_for_stage(
    std::string_view op_family, size_t runtime_param_count) noexcept {
  if (runtime_param_count == 3 &&
      runtime_param_descriptor_family_is_binary_broadcast(op_family)) {
    return RuntimeParamDescriptorPayloadKind::BinaryBroadcast;
  }
  if (runtime_param_count == 4 && op_family == "Broadcast") {
    return RuntimeParamDescriptorPayloadKind::Broadcast;
  }
  if (runtime_param_count == 4 && op_family == "Select") {
    return RuntimeParamDescriptorPayloadKind::Select;
  }
  if (runtime_param_count == 4 && op_family == "Tile") {
    return RuntimeParamDescriptorPayloadKind::Tile;
  }
  if (runtime_param_count == 1 && op_family == "Interpolate") {
    return RuntimeParamDescriptorPayloadKind::Interpolate;
  }
  if (runtime_param_count == 1 &&
      (op_family == "Softmax" || op_family == "LogSoftmax")) {
    return RuntimeParamDescriptorPayloadKind::Softmax;
  }
  if (runtime_param_count == 5 && op_family == "Transpose") {
    return RuntimeParamDescriptorPayloadKind::Transpose;
  }
  if (runtime_param_count == 5 &&
      runtime_param_descriptor_family_is_reduce(op_family)) {
    return RuntimeParamDescriptorPayloadKind::Reduce;
  }
  return RuntimeParamDescriptorPayloadKind::None;
}

inline size_t runtime_param_descriptor_expected_buffer_count(
    RuntimeParamDescriptorPayloadKind kind) noexcept {
  switch (kind) {
  case RuntimeParamDescriptorPayloadKind::None:
    return 0;
  case RuntimeParamDescriptorPayloadKind::BinaryBroadcast:
    return 3;
  case RuntimeParamDescriptorPayloadKind::Broadcast:
  case RuntimeParamDescriptorPayloadKind::Select:
  case RuntimeParamDescriptorPayloadKind::Tile:
    return 4;
  case RuntimeParamDescriptorPayloadKind::Interpolate:
  case RuntimeParamDescriptorPayloadKind::Softmax:
    return 1;
  case RuntimeParamDescriptorPayloadKind::Transpose:
  case RuntimeParamDescriptorPayloadKind::Reduce:
    return 5;
  }
  return kRuntimeParamDescriptorUnknownCount;
}

inline bool runtime_param_descriptor_buffer_count_matches(
    RuntimeParamDescriptorPayloadKind kind, size_t count) noexcept {
  const auto expected = runtime_param_descriptor_expected_buffer_count(kind);
  return expected != kRuntimeParamDescriptorUnknownCount && expected == count;
}

inline bool runtime_param_descriptor_payload_kind_matches_stage(
    RuntimeParamDescriptorPayloadKind kind, std::string_view op_family,
    size_t runtime_param_count) noexcept {
  return kind ==
         runtime_param_descriptor_payload_kind_for_stage(op_family,
                                                        runtime_param_count);
}

} // namespace gfx_plugin
} // namespace ov
