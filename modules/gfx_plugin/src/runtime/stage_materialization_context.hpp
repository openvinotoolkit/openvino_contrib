// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>

#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "runtime/executable_descriptor.hpp"

namespace ov {
namespace gfx_plugin {

struct RuntimeStageMaterializationContext {
  const RuntimeStageExecutableDescriptor *descriptor = nullptr;
  std::shared_ptr<const ov::Node> source_node;

  RuntimeStageMaterializationContext() = default;

  RuntimeStageMaterializationContext(
      std::shared_ptr<const ov::Node> node,
      const RuntimeStageExecutableDescriptor &runtime_descriptor)
      : descriptor(&runtime_descriptor), source_node(std::move(node)) {}

  const RuntimeStageExecutableDescriptor &require_descriptor() const {
    OPENVINO_ASSERT(descriptor,
                    "GFX: stage materialization requires a compiler-owned "
                    "runtime executable descriptor for op ",
                    op_type_name());
    return *descriptor;
  }

  std::string op_type_name() const {
    if (descriptor && !descriptor->op_family.empty()) {
      return descriptor->op_family;
    }
    return source_node ? source_node->get_type_name() : std::string("<null>");
  }

  std::string op_friendly_name() const {
    if (descriptor && !descriptor->stage_name.empty()) {
      return descriptor->stage_name;
    }
    if (descriptor && !descriptor->manifest_ref.empty()) {
      return descriptor->manifest_ref;
    }
    if (descriptor && !descriptor->kernel_id.empty()) {
      return descriptor->kernel_id;
    }
    return source_node ? source_node->get_friendly_name()
                       : std::string("<null>");
  }
};

} // namespace gfx_plugin
} // namespace ov
