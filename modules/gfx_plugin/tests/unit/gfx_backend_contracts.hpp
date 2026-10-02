// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "compiler/backend_registry.hpp"
#include "compiler/gfx_compiler_service.hpp"
#include "compiler/kernel_registry.hpp"
#include "openvino/core/model.hpp"
#include "runtime/gfx_backend_utils.hpp"

namespace ov {
namespace gfx_plugin {
namespace test {

class BackendTargetContract final {
public:
    explicit BackendTargetContract(compiler::BackendTarget target);

    const compiler::BackendTarget& target() const noexcept;
    bool has_concrete_oop_identity() const;
    bool has_profiled_cache_identity() const;
    bool avoids_inverse_apple_bucket() const;

private:
    compiler::BackendTarget m_target;
};

class BackendModuleContract final {
public:
    explicit BackendModuleContract(std::shared_ptr<const compiler::BackendModule> module);

    const compiler::BackendModule& module() const;
    const compiler::BackendTarget& target() const;
    compiler::GfxCompileResult compile_without_graph_pipeline(
        const std::shared_ptr<const ov::Model>& model) const;
    bool compile_result_obeys_manifest_contract(
        const compiler::GfxCompileResult& compile_result) const;

private:
    std::shared_ptr<const compiler::BackendModule> m_module;
};

class BackendContractCatalog final {
public:
    BackendContractCatalog();
    explicit BackendContractCatalog(const compiler::BackendRegistry& registry);

    std::vector<BackendTargetContract> known_target_contracts() const;
    std::vector<BackendModuleContract> compiled_module_contracts() const;

private:
    const compiler::BackendRegistry* m_registry = nullptr;
};

class KernelRegistryContract final {
public:
    static KernelRegistryContract for_opencl();
    static KernelRegistryContract for_metal();

    bool audit_is_valid() const;
    bool rejects_unit(compiler::LoweringRouteKind route,
                      std::string_view kernel_unit_id) const;
    compiler::KernelUnit resolve_unit(compiler::LoweringRouteKind route,
                                      std::string_view kernel_unit_id) const;

private:
    explicit KernelRegistryContract(compiler::KernelRegistry registry);

    compiler::KernelRegistry m_registry;
};

class ModelContractFactory final {
public:
    std::shared_ptr<ov::Model> passthrough(const ov::PartialShape& shape) const;
    std::shared_ptr<ov::Model> relu() const;
    std::shared_ptr<ov::Model> static_range() const;
};

}  // namespace test
}  // namespace gfx_plugin
}  // namespace ov
