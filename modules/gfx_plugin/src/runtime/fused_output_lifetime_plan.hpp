// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "runtime/executable_descriptor.hpp"
#include "runtime/output_lifetime.hpp"

#include <cstddef>
#include <vector>

namespace ov {
namespace gfx_plugin {

struct FusedOutputLifetimeInputRef {
  enum class Kind { None, External, Output };

  Kind kind = Kind::None;
  size_t index = 0;
};

struct FusedOutputLifetimeStage {
  std::vector<FusedOutputLifetimeInputRef> inputs;
  std::vector<size_t> output_indices;
  const RuntimeStageExecutableDescriptor *descriptor = nullptr;
};

std::vector<RuntimeOutputLifetime> build_fused_output_lifetime_plan(
    const std::vector<FusedOutputLifetimeStage> &stages,
    const RuntimeMemoryPlanDescriptor &memory_plan, size_t output_count);

}  // namespace gfx_plugin
}  // namespace ov
