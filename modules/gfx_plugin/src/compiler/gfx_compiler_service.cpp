// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/gfx_compiler_service.hpp"

#include <memory>
#include <sstream>
#include <utility>

#include "openvino/core/except.hpp"
#include "compiler/pipeline_stage_builder.hpp"
#include "compiler/runtime_executable_descriptor_builder.hpp"
#include "runtime/executable_descriptor.hpp"
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
           executable.valid() &&
           runtime_descriptor &&
           runtime_descriptor->stage_plan &&
           runtime_executable_descriptor_valid(*runtime_descriptor,
                                               executable) &&
           runtime_executable_stage_plan_valid(*runtime_descriptor) &&
           cache_envelope.valid(executable);
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
    OPENVINO_ASSERT(
        backend_module->target().is_compatible_with_fingerprint(
            request.target.fingerprint()),
        "GFX: compiler registry returned non-exact target ",
        backend_module->target().debug_string(),
        " for requested ",
        request.target.debug_string());

    GfxCompileResult result;
    result.target = backend_module->target();
    result.transformed_model =
        ov::gfx_plugin::transforms::run_pipeline(request.model,
                                                 backend_module->pipeline_options());
    result.lowering_plan = backend_module->lowering_planner().plan(result.transformed_model,
                                                                   backend_module->legalizer());
    result.manifest = ManifestBuilder{}.build(result.lowering_plan);
    result.executable = ExecutableBundleBuilder(
        [backend_module](KernelArtifactDescriptor& descriptor,
                         const PlannedOperation& op) {
            return backend_module->materialize_artifact_payload(descriptor, op);
        }).build(result.manifest, result.lowering_plan);
    CacheEnvelopeBuildOptions cache_options;
    cache_options.model_fingerprint = make_model_cache_fingerprint(*request.model);
    cache_options.backend_capabilities_fingerprint =
        make_backend_capabilities_fingerprint(backend_module->capabilities());
    cache_options.backend_compiler_revision =
        backend_module->target().compiler_id();
    cache_options.driver_identity = backend_module->target().driver_id();
    result.cache_envelope =
        CacheEnvelopeBuilder{}.build(result.executable, cache_options);
    result.unsupported = result.lowering_plan.unsupported;
    if (result.lowering_plan.executable() && result.manifest.valid() &&
        result.executable.valid()) {
        auto runtime_descriptor =
            RuntimeExecutableDescriptorBuilder{}.build(result.executable);
        PipelineStageBuildRequest stage_request;
        stage_request.runtime_model = result.transformed_model;
        stage_request.runtime_descriptor = &runtime_descriptor;
        stage_request.backend_registry = m_registry;
        stage_request.target = result.target;
        stage_request.backend_name = request.backend_name.empty()
                                         ? result.target.backend_id()
                                         : request.backend_name;
        stage_request.enable_fusion = request.enable_fusion;
        stage_request.compile_trace = request.compile_trace;
        runtime_descriptor.stage_plan =
            build_pipeline_stage_runtime_plan(stage_request);
        result.runtime_descriptor =
            std::make_shared<const RuntimeExecutableDescriptor>(
                std::move(runtime_descriptor));
    }
    return result;
}

}  // namespace compiler
}  // namespace gfx_plugin
}  // namespace ov
