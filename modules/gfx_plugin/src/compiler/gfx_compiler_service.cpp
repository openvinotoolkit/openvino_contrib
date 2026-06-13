// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/gfx_compiler_service.hpp"

#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "compiler/pipeline_stage_graph_snapshot.hpp"
#include "compiler/runtime_executable_descriptor_builder.hpp"
#include "runtime/executable_descriptor.hpp"
#include "runtime/gfx_logger.hpp"
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
           runtime_executable_descriptor_valid(*runtime_descriptor,
                                               executable) &&
           runtime_executable_descriptor_materialization_valid(
               *runtime_descriptor) &&
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
    if (unsupported.type_counts.empty()) {
        std::vector<std::string> contract_diagnostics;
        if (!lowering_plan.executable()) {
            contract_diagnostics.emplace_back("lowering plan is not executable");
        }
        if (!manifest.valid()) {
            contract_diagnostics.emplace_back("manifest is invalid");
        }
        if (!executable.valid()) {
            contract_diagnostics.emplace_back("executable bundle is invalid");
        }
        if (!runtime_descriptor) {
            contract_diagnostics.emplace_back("runtime descriptor is missing");
        } else {
            const auto descriptor_verification =
                verify_runtime_executable_descriptor(*runtime_descriptor,
                                                     executable);
            if (!descriptor_verification.valid()) {
                contract_diagnostics.insert(
                    contract_diagnostics.end(),
                    descriptor_verification.diagnostics.begin(),
                    descriptor_verification.diagnostics.end());
            }
            const auto materialization_verification =
                verify_runtime_executable_descriptor_materialization(
                    *runtime_descriptor);
            contract_diagnostics.insert(
                contract_diagnostics.end(),
                materialization_verification.diagnostics.begin(),
                materialization_verification.diagnostics.end());
        }
        if (!cache_envelope.valid(executable)) {
            contract_diagnostics.emplace_back(
                "cache envelope does not match executable bundle");
        }
        if (!contract_diagnostics.empty()) {
            oss << " Contract diagnostics: ";
            for (size_t i = 0; i < contract_diagnostics.size(); ++i) {
                if (i) {
                    oss << "; ";
                }
                oss << contract_diagnostics[i];
            }
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
    const auto compiler_stage_graph_snapshot =
        detail::make_pipeline_stage_graph_snapshot(
            result.transformed_model,
            detail::make_pipeline_stage_fusion_config(
                backend_module->capabilities().fusion(),
                request.enable_fusion,
                gfx_log_debug_enabled()));
    result.executable = ExecutableBundleBuilder(
        [backend_module](KernelArtifactDescriptor& descriptor,
                         const PlannedOperation& op) {
            return backend_module->finalize_artifact_descriptor(descriptor, op);
        },
        [backend_module](const KernelArtifactDescriptor& descriptor,
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
    cache_options.backend_payload_encoder =
        [backend_module](const KernelArtifactDescriptor &descriptor,
                         const KernelArtifactPayloadRecord &payload_record) {
          return backend_module->encode_cache_payload(descriptor,
                                                      payload_record);
        };
    result.unsupported = result.lowering_plan.unsupported;
    if (result.lowering_plan.executable() && result.manifest.valid() &&
        result.executable.valid()) {
        RuntimeExecutableDescriptorBuildRequest descriptor_request;
        descriptor_request.executable = &result.executable;
        descriptor_request.stage_graph_snapshot = &compiler_stage_graph_snapshot;
        descriptor_request.backend_registry = m_registry;
        descriptor_request.target = result.target;
        descriptor_request.backend_name = request.backend_name.empty()
                                              ? result.target.backend_id()
                                              : request.backend_name;
        descriptor_request.compile_trace = request.compile_trace;
        auto runtime_descriptor =
            RuntimeExecutableDescriptorBuilder{}.build_finalized(
                descriptor_request);
        result.runtime_descriptor =
            std::make_shared<const RuntimeExecutableDescriptor>(
                std::move(runtime_descriptor));
        result.cache_envelope =
            CacheEnvelopeBuilder{}.build(result.executable,
                                         *result.runtime_descriptor,
                                         cache_options);
    }
    return result;
}

}  // namespace compiler
}  // namespace gfx_plugin
}  // namespace ov
