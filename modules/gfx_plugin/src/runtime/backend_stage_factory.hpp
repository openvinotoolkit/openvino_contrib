// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "common/gpu_backend.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace gfx_plugin {

class GpuStage;
struct RuntimeStageExecutableDescriptor;

class BackendStageFactory {
public:
  virtual ~BackendStageFactory() = default;

  virtual GpuBackend backend() const = 0;
  virtual std::unique_ptr<GpuStage>
  create_stage(const std::shared_ptr<const ov::Node> &node,
               const RuntimeStageExecutableDescriptor *descriptor) const = 0;
};

} // namespace gfx_plugin
} // namespace ov
