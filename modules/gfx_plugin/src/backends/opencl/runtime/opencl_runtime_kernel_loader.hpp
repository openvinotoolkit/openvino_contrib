// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "backends/opencl/runtime/opencl_api.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "runtime/executable_descriptor.hpp"
#include "runtime/gpu_stage.hpp"

namespace ov {
namespace gfx_plugin {

class OpenClRuntimeKernelLoader final {
public:
    explicit OpenClRuntimeKernelLoader(std::shared_ptr<OpenClRuntimeContext> context);

    std::unique_ptr<GpuStage> load_source_stage(
        const std::shared_ptr<const ov::Node>& node,
        const RuntimeStageExecutableDescriptor& descriptor,
        GfxOpenClSourceArtifact artifact) const;

private:
    std::shared_ptr<OpenClRuntimeContext> m_context;
};

}  // namespace gfx_plugin
}  // namespace ov
