// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>

#include "runtime/executable_descriptor.hpp"

namespace ov {
namespace gfx_plugin {

struct RuntimeStageMaterializationContext {
  explicit RuntimeStageMaterializationContext(
      const RuntimeStageExecutableDescriptor &runtime_descriptor)
      : descriptor(runtime_descriptor) {}

  const RuntimeStageExecutableDescriptor &require_descriptor() const noexcept {
    return descriptor;
  }

  std::string op_type_name() const {
    if (!descriptor.op_family.empty()) {
      return descriptor.op_family;
    }
    if (!descriptor.kernel_id.empty()) {
      return descriptor.kernel_id;
    }
    return std::string("<descriptor-missing-op-family>");
  }

  std::string op_friendly_name() const {
    if (!descriptor.stage_name.empty()) {
      return descriptor.stage_name;
    }
    if (!descriptor.manifest_ref.empty()) {
      return descriptor.manifest_ref;
    }
    if (!descriptor.kernel_id.empty()) {
      return descriptor.kernel_id;
    }
    return std::string("<descriptor-missing-stage-name>");
  }

  const RuntimeStageExecutableDescriptor &descriptor;
};

} // namespace gfx_plugin
} // namespace ov
