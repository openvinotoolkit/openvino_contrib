// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "common/gpu_backend.hpp"
#include "runtime/stage_materialization_context.hpp"

namespace ov {
namespace gfx_plugin {

class GpuStage;

class BackendStageFactory {
public:
  virtual ~BackendStageFactory() = default;

  virtual GpuBackend backend() const = 0;
  virtual std::unique_ptr<GpuStage>
  create_stage(const RuntimeStageMaterializationContext &context) const = 0;
};

} // namespace gfx_plugin
} // namespace ov
