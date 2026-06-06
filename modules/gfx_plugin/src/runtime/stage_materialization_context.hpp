// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "runtime/executable_descriptor.hpp"

namespace ov {
namespace gfx_plugin {

struct RuntimeStageMaterializationContext {
  const RuntimeStageExecutableDescriptor *descriptor = nullptr;
  std::shared_ptr<const ov::Node> source_node;

  RuntimeStageMaterializationContext() = default;

  explicit RuntimeStageMaterializationContext(
      const RuntimeStageExecutableDescriptor &runtime_descriptor)
      : descriptor(&runtime_descriptor) {}

  RuntimeStageMaterializationContext(
      const RuntimeStageExecutableDescriptor &runtime_descriptor,
      std::shared_ptr<const ov::Node> node)
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
    if (descriptor && !descriptor->kernel_id.empty()) {
      return descriptor->kernel_id;
    }
    return descriptor ? std::string("<descriptor-missing-op-family>")
                      : std::string("<null>");
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
    return descriptor ? std::string("<descriptor-missing-stage-name>")
                      : std::string("<null>");
  }

  bool has_source_node() const noexcept {
    return static_cast<bool>(source_node);
  }

  const std::shared_ptr<const ov::Node> &
  require_source_node(std::string_view reason) const {
    (void)require_descriptor();
    OPENVINO_ASSERT(
        source_node,
        "GFX: backend runtime requested temporary ov::Node materialization "
        "source for ",
        op_friendly_name(), " (", op_type_name(),
        ") while materialization "
        "context is descriptor-only. Remaining migration reason: ",
        std::string(reason));
    return source_node;
  }
};

} // namespace gfx_plugin
} // namespace ov
