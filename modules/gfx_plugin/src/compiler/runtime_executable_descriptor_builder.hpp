// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>

#include "compiler/backend_target.hpp"
#include "compiler/executable_bundle.hpp"
#include "runtime/executable_descriptor.hpp"

namespace ov {
namespace gfx_plugin {
class GfxProfilingTrace;

namespace compiler {

class BackendRegistry;

namespace detail {
struct PipelineStageGraphSnapshot;
}

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
  const detail::PipelineStageGraphSnapshot *stage_graph_snapshot = nullptr;
  const BackendRegistry *backend_registry = nullptr;
  BackendTarget target;
  std::string backend_name;
  GfxProfilingTrace *compile_trace = nullptr;

  bool valid() const noexcept {
    return executable && stage_graph_snapshot && backend_registry &&
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
