// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "kernel_ir/gfx_codegen_backend.hpp"
#include "runtime/executable_descriptor.hpp"

namespace ov {
namespace gfx_plugin {

class MetalRuntimeKernelLoader final {
public:
    static bool has_msl_source_payload(
        const RuntimeStageExecutableDescriptor& descriptor) noexcept;

    static KernelSource load_msl_source(
        const RuntimeStageExecutableDescriptor& descriptor);
};

}  // namespace gfx_plugin
}  // namespace ov
