// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/gfx_compiler_service.hpp"

#include <sstream>

#include "openvino/core/except.hpp"
#include "transforms/pipeline.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

GfxCompilerService::GfxCompilerService()
    : GfxCompilerService(BackendRegistry::default_registry()) {}

GfxCompilerService::GfxCompilerService(const BackendRegistry& registry)
    : m_registry(&registry) {}

bool GfxCompileResult::supported() const {
    return unsupported.type_counts.empty() &&
           lowering_plan.executable() &&
           manifest.valid() &&
           executable.valid();
}

std::string GfxCompileResult::unsupported_message() const {
    std::ostringstream oss;
    oss << "GFX: model contains unsupported ops for MLIR/GFX execution on "
        << target.debug_string() << ".";
    if (!unsupported.type_counts.empty()) {
        oss << " Types: ";
        size_t shown = 0;
        for (const auto& kv : unsupported.type_counts) {
            if (shown++) {
                oss << ", ";
            }
            oss << kv.first << " x" << kv.second;
        }
    }
    if (!unsupported.node_names.empty()) {
        oss << ". Nodes: ";
        for (size_t i = 0; i < unsupported.node_names.size(); ++i) {
            if (i) {
                oss << ", ";
            }
            oss << unsupported.node_names[i];
        }
    }
    return oss.str();
}

GfxCompileResult GfxCompilerService::compile(const GfxCompileRequest& request) const {
    OPENVINO_ASSERT(request.model, "Model is null");
    OPENVINO_ASSERT(m_registry, "GFX: compiler backend registry is null");

    const auto backend_module = m_registry->resolve(request.target);
    OPENVINO_ASSERT(backend_module,
                    "GFX: backend target is not registered: ",
                    request.target.debug_string());

    GfxCompileResult result;
    result.target = backend_module->target();
    result.transformed_model =
        ov::gfx_plugin::transforms::run_pipeline(request.model, result.target.backend());
    result.lowering_plan = backend_module->lowering_planner().plan(result.transformed_model,
                                                                   backend_module->legalizer());
    result.manifest = ManifestBuilder{}.build(result.lowering_plan);
    result.executable = ExecutableBundleBuilder(
        [backend_module](KernelArtifactDescriptor& descriptor,
                         const PlannedOperation& op) {
            return backend_module->materialize_artifact_payload(descriptor, op);
        }).build(result.manifest, result.lowering_plan);
    result.unsupported = result.lowering_plan.unsupported;
    return result;
}

}  // namespace compiler
}  // namespace gfx_plugin
}  // namespace ov
