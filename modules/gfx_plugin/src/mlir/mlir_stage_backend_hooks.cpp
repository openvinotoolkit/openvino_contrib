// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_stage_backend_hooks.hpp"

#include <array>

namespace ov {
namespace gfx_plugin {
namespace {

size_t hook_slot_index(GpuBackend backend) {
  switch (backend) {
  case GpuBackend::Metal:
    return 0;
  case GpuBackend::OpenCL:
    return 1;
  case GpuBackend::Unknown:
  default:
    return 2;
  }
}

std::array<const MlirStageBackendHooks *, 2> &hook_slots() {
  static std::array<const MlirStageBackendHooks *, 2> slots{};
  return slots;
}

} // namespace

bool register_mlir_stage_backend_hooks(GpuBackend backend,
                                       const MlirStageBackendHooks &hooks) {
  const size_t index = hook_slot_index(backend);
  if (index >= hook_slots().size()) {
    return false;
  }
  hook_slots()[index] = &hooks;
  return true;
}

const MlirStageBackendHooks *mlir_stage_backend_hooks_for(GpuBackend backend) {
  const size_t index = hook_slot_index(backend);
  if (index >= hook_slots().size()) {
    return nullptr;
  }
  return hook_slots()[index];
}

} // namespace gfx_plugin
} // namespace ov
