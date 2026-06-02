// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unit/gfx_backend_contracts.hpp"

#include "compiler/cache_envelope.hpp"
#include "compiler/executable_bundle.hpp"
#include "compiler/manifest.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace gfx_plugin {
namespace test {
namespace {

bool contains(std::string_view haystack, std::string_view needle) {
    return haystack.find(needle) != std::string_view::npos;
}

}  // namespace

BackendTargetContract::BackendTargetContract(compiler::BackendTarget target)
    : m_target(std::move(target)) {}

const compiler::BackendTarget& BackendTargetContract::target() const noexcept {
    return m_target;
}

bool BackendTargetContract::has_concrete_oop_identity() const {
    return !m_target.backend_id().empty() &&
           !m_target.runtime_api().empty() &&
           !m_target.device_family().empty() &&
           !m_target.compiler_id().empty() &&
           m_target.is_compatible_with_fingerprint(m_target.fingerprint());
}

bool BackendTargetContract::avoids_inverse_apple_bucket() const {
    const auto debug = m_target.debug_string();
    return !contains(debug, "non_apple") &&
           !contains(debug, "non-apple") &&
           !contains(debug, "not_apple");
}

BackendModuleContract::BackendModuleContract(
    std::shared_ptr<const compiler::BackendModule> module)
    : m_module(std::move(module)) {}

const compiler::BackendModule& BackendModuleContract::module() const {
    OPENVINO_ASSERT(m_module, "GFX test backend module contract is empty");
    return *m_module;
}

const compiler::BackendTarget& BackendModuleContract::target() const {
    return module().target();
}

compiler::GfxCompileResult BackendModuleContract::compile_without_graph_pipeline(
    const std::shared_ptr<const ov::Model>& model) const {
    const auto& backend_module = module();
    compiler::GfxCompileResult compile_result;
    compile_result.target = backend_module.target();
    compile_result.transformed_model = model;
    compile_result.lowering_plan =
        backend_module.lowering_planner().plan(model,
                                               backend_module.legalizer());
    compile_result.manifest =
        compiler::ManifestBuilder{}.build(compile_result.lowering_plan);
    compile_result.executable = compiler::ExecutableBundleBuilder(
        [&backend_module](compiler::KernelArtifactDescriptor& descriptor,
                          const compiler::PlannedOperation& op) {
            return backend_module.materialize_artifact_payload(descriptor, op);
        }).build(compile_result.manifest, compile_result.lowering_plan);
    compiler::CacheEnvelopeBuildOptions cache_options;
    cache_options.model_fingerprint =
        compiler::make_model_cache_fingerprint(*model);
    cache_options.backend_capabilities_fingerprint =
        compiler::make_backend_capabilities_fingerprint(
            backend_module.capabilities());
    cache_options.backend_compiler_revision =
        backend_module.target().compiler_id();
    cache_options.driver_identity = backend_module.target().driver_id();
    compile_result.cache_envelope =
        compiler::CacheEnvelopeBuilder{}.build(compile_result.executable,
                                               cache_options);
    compile_result.unsupported = compile_result.lowering_plan.unsupported;
    return compile_result;
}

bool BackendModuleContract::compile_result_obeys_manifest_contract(
    const compiler::GfxCompileResult& compile_result) const {
    if (!compile_result.supported() ||
        compile_result.target.fingerprint() != target().fingerprint() ||
        !compile_result.manifest.verify().valid() ||
        !compile_result.executable.verify().valid() ||
        !compile_result.cache_envelope.verify(compile_result.executable)
             .valid()) {
        return false;
    }
    if (!compile_result.manifest.memory_plan.valid() ||
        !compile_result.executable.memory_plan.valid() ||
        compiler::make_memory_plan_fingerprint(
            compile_result.manifest.memory_plan) !=
            compiler::make_memory_plan_fingerprint(
                compile_result.executable.memory_plan)) {
        return false;
    }
    if (compile_result.manifest.route_count(compiler::LoweringRouteKind::Common) !=
        compile_result.lowering_plan.route_count(
            compiler::LoweringRouteKind::Common)) {
        return false;
    }
    for (const auto& stage : compile_result.manifest.stages) {
        if (stage.memory.hidden_host_copy_allowed ||
            stage.dispatch.backend_domain != stage.backend_domain ||
            stage.dispatch.kernel_unit_id != stage.kernel_unit_id ||
            stage.dispatch.kernel_unit_kind != stage.kernel_unit_kind ||
            !compile_result.manifest.memory_plan.has_alias_group(
                stage.memory.alias_group)) {
            return false;
        }
        for (const auto& input : stage.inputs) {
            if (!compile_result.manifest.memory_plan.has_region(
                    input.memory_region_id)) {
                return false;
            }
        }
        for (const auto& output : stage.outputs) {
            if (!compile_result.manifest.memory_plan.has_region(
                    output.memory_region_id)) {
                return false;
            }
        }
    }
    return true;
}

BackendContractCatalog::BackendContractCatalog()
    : BackendContractCatalog(compiler::BackendRegistry::default_registry()) {}

BackendContractCatalog::BackendContractCatalog(
    const compiler::BackendRegistry& registry)
    : m_registry(&registry) {}

std::vector<BackendTargetContract>
BackendContractCatalog::known_target_contracts() const {
    return {BackendTargetContract{
                compiler::BackendTarget::from_backend(GpuBackend::Metal)},
            BackendTargetContract{
                compiler::BackendTarget::from_backend(GpuBackend::OpenCL)}};
}

std::vector<BackendModuleContract>
BackendContractCatalog::compiled_module_contracts() const {
    OPENVINO_ASSERT(m_registry, "GFX test backend catalog has no registry");
    std::vector<BackendModuleContract> contracts;
    for (const auto& target : m_registry->available_targets()) {
        auto module = m_registry->resolve(target);
        if (module) {
            contracts.emplace_back(std::move(module));
        }
    }
    return contracts;
}

KernelRegistryContract KernelRegistryContract::for_opencl() {
    const auto target =
        compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
    return KernelRegistryContract{
        compiler::make_opencl_kernel_registry(target)};
}

KernelRegistryContract KernelRegistryContract::for_metal() {
    const auto target = compiler::BackendTarget::from_backend(GpuBackend::Metal);
    return KernelRegistryContract{compiler::make_metal_kernel_registry(target)};
}

KernelRegistryContract::KernelRegistryContract(compiler::KernelRegistry registry)
    : m_registry(std::move(registry)) {}

bool KernelRegistryContract::audit_is_valid() const {
    return m_registry.audit().valid();
}

bool KernelRegistryContract::rejects_unit(
    compiler::LoweringRouteKind route,
    std::string_view kernel_unit_id) const {
    return !m_registry.resolve(route, kernel_unit_id).valid();
}

compiler::KernelUnit KernelRegistryContract::resolve_unit(
    compiler::LoweringRouteKind route,
    std::string_view kernel_unit_id) const {
    return m_registry.resolve(route, kernel_unit_id);
}

std::shared_ptr<ov::Model> ModelContractFactory::passthrough(
    const ov::PartialShape& shape) const {
    auto parameter =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto result = std::make_shared<ov::op::v0::Result>(parameter);
    return std::make_shared<ov::Model>(ov::ResultVector{result},
                                       ov::ParameterVector{parameter});
}

}  // namespace test
}  // namespace gfx_plugin
}  // namespace ov
