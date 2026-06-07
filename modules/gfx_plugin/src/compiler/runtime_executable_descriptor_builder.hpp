// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>

#include "compiler/backend_target.hpp"
#include "compiler/executable_bundle.hpp"
#include "compiler/operation_support.hpp"
#include "runtime/executable_descriptor.hpp"

namespace ov {
class Model;

namespace gfx_plugin {
class GfxProfilingTrace;

namespace compiler {

class BackendRegistry;

RuntimeExecutableDescriptorVerificationResult
verify_runtime_executable_descriptor(
    const RuntimeExecutableDescriptor &descriptor,
    const ExecutableBundle &executable);

bool runtime_executable_descriptor_valid(
    const RuntimeExecutableDescriptor &descriptor,
    const ExecutableBundle &executable);

RuntimeExecutableDescriptorVerificationResult
verify_runtime_executable_descriptor_materialization(
    const RuntimeExecutableDescriptor &descriptor);

bool runtime_executable_descriptor_materialization_valid(
    const RuntimeExecutableDescriptor &descriptor);

struct RuntimeExecutableDescriptorBuildRequest {
  const ExecutableBundle *executable = nullptr;
  std::shared_ptr<const ov::Model> transformed_model;
  const BackendRegistry *backend_registry = nullptr;
  BackendTarget target;
  std::string backend_name;
  FusionCapabilities fusion_capabilities = {};
  bool enable_fusion = true;
  GfxProfilingTrace *compile_trace = nullptr;

  bool valid() const noexcept {
    return executable && transformed_model && backend_registry &&
           target.backend() != GpuBackend::Unknown;
  }
};

class RuntimeExecutableDescriptorBuilder final {
public:
  RuntimeExecutableDescriptor build(const ExecutableBundle &executable) const;
  RuntimeExecutableDescriptor
  build_finalized(const RuntimeExecutableDescriptorBuildRequest &request) const;
};

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
