// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>

#include "compiler/backend_registry.hpp"
#include "compiler/backend_target.hpp"
#include "compiler/executable_bundle.hpp"
#include "compiler/manifest.hpp"
#include "compiler/operation_support.hpp"
#include "openvino/core/model.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

struct GfxCompileRequest {
    std::shared_ptr<const ov::Model> model;
    BackendTarget target = BackendTarget::from_backend(GpuBackend::Metal);
};

struct GfxCompileResult {
    BackendTarget target = BackendTarget::from_backend(GpuBackend::Metal);
    std::shared_ptr<const ov::Model> transformed_model;
    LoweringPlan lowering_plan;
    ManifestBundle manifest;
    ExecutableBundle executable;
    UnsupportedSummary unsupported;

    bool supported() const;
    std::string unsupported_message() const;
};

class GfxCompilerService {
public:
    GfxCompilerService();
    explicit GfxCompilerService(const BackendRegistry& registry);

    GfxCompileResult compile(const GfxCompileRequest& request) const;

private:
    const BackendRegistry* m_registry = nullptr;
};

}  // namespace compiler
}  // namespace gfx_plugin
}  // namespace ov
