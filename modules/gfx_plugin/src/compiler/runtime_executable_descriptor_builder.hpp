// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>

#include "common/gpu_backend.hpp"
#include "compiler/executable_bundle.hpp"
#include "openvino/core/model.hpp"
#include "runtime/executable_descriptor.hpp"

namespace ov {
namespace gfx_plugin {

class GfxProfilingTrace;

namespace compiler {

struct RuntimeExecutableDescriptorBuildRequest {
  const ExecutableBundle *executable = nullptr;
  std::shared_ptr<const ov::Model> runtime_model;
  GpuBackend backend = GpuBackend::Unknown;
  std::string backend_name;
  bool enable_fusion = true;
  GfxProfilingTrace *compile_trace = nullptr;
};

RuntimeExecutableDescriptorVerificationResult
verify_runtime_executable_descriptor(
    const RuntimeExecutableDescriptor &descriptor,
    const ExecutableBundle &executable);

bool runtime_executable_descriptor_valid(
    const RuntimeExecutableDescriptor &descriptor,
    const ExecutableBundle &executable);

class RuntimeExecutableDescriptorBuilder final {
public:
  RuntimeExecutableDescriptor build(const ExecutableBundle &executable) const;
  RuntimeExecutableDescriptor
  build(const RuntimeExecutableDescriptorBuildRequest &request) const;
};

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
