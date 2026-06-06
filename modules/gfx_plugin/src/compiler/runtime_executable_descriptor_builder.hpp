// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "compiler/executable_bundle.hpp"
#include "runtime/executable_descriptor.hpp"

namespace ov {
namespace gfx_plugin {

namespace compiler {

RuntimeExecutableDescriptorVerificationResult
verify_runtime_executable_descriptor(
    const RuntimeExecutableDescriptor &descriptor,
    const ExecutableBundle &executable);

bool runtime_executable_descriptor_valid(
    const RuntimeExecutableDescriptor &descriptor,
    const ExecutableBundle &executable);

RuntimeExecutableDescriptorVerificationResult
verify_runtime_executable_descriptor_pipeline_plan(
    const RuntimeExecutableDescriptor &descriptor);

bool runtime_executable_descriptor_pipeline_plan_valid(
    const RuntimeExecutableDescriptor &descriptor);

void attach_runtime_public_output_descriptors(
    RuntimeExecutableDescriptor &descriptor,
    const ::ov::gfx_plugin::PipelineStageRuntimePlan &pipeline_plan);

class RuntimeExecutableDescriptorBuilder final {
public:
  RuntimeExecutableDescriptor build(const ExecutableBundle &executable) const;
};

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
