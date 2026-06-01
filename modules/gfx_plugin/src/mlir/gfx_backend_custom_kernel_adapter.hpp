// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "kernel_ir/gfx_codegen_backend.hpp"
#include "kernel_ir/gfx_custom_kernel_families.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/gfx_mlir_kernel_metadata.hpp"

namespace ov {
namespace gfx_plugin {

GfxKernelRuntimeBindingPlan
make_backend_custom_kernel_binding_plan_from_stage_manifest(
    const GfxKernelStageManifest &manifest);

bool read_backend_custom_kernel_stage_manifest_from_module(
    mlir::ModuleOp module, GfxKernelBackendDomain backend_domain,
    GfxKernelStageManifest &manifest);

GfxKernelRuntimeBindingPlan make_backend_custom_kernel_binding_plan(
    std::string_view stage_type, std::string_view entry_point,
    GfxKernelBackendDomain backend_domain = GfxKernelBackendDomain::AppleMsl,
    GfxKernelStorageKind storage = GfxKernelStorageKind::Buffer,
    std::string_view specialization_prefix = "apple_msl:buffer:");

GfxKernelRuntimeBindingPlan make_backend_custom_kernel_binding_plan(
    std::string_view stage_type, std::string_view entry_point,
    std::vector<int32_t> scalar_args,
    GfxKernelBackendDomain backend_domain = GfxKernelBackendDomain::AppleMsl,
    GfxKernelStorageKind storage = GfxKernelStorageKind::Buffer,
    std::string_view specialization_prefix = "apple_msl:buffer:");

GfxKernelRuntimeBindingPlan make_backend_custom_kernel_binding_plan(
    bool is_opencl_backend, std::string_view stage_type,
    std::string_view entry_point);

GfxKernelRuntimeBindingPlan make_backend_custom_kernel_binding_plan(
    bool is_opencl_backend, std::string_view stage_type,
    std::string_view entry_point, std::vector<int32_t> scalar_args);

GfxCustomKernelStagePlan make_backend_custom_kernel_stage_plan_view(
    const GfxKernelRuntimeBindingPlan &binding);

GfxKernelRuntimeBindingPlan make_backend_custom_kernel_direct_io_binding_plan(
    std::string_view stage_type, std::string_view entry_point,
    size_t tensor_input_count, size_t output_count,
    GfxKernelBackendDomain backend_domain = GfxKernelBackendDomain::AppleMsl,
    GfxKernelStorageKind storage = GfxKernelStorageKind::Buffer,
    std::string_view specialization_prefix = "apple_msl:buffer:");

GfxKernelRuntimeBindingPlan make_backend_custom_kernel_roles_binding_plan(
    std::string_view stage_type, std::string_view entry_point,
    std::vector<GfxKernelBufferRole> roles,
    GfxKernelBackendDomain backend_domain = GfxKernelBackendDomain::AppleMsl,
    GfxKernelStorageKind storage = GfxKernelStorageKind::Buffer,
    std::string_view specialization_prefix = "apple_msl:buffer:");

GfxKernelRuntimeBindingPlan make_backend_custom_kernel_binding_plan_from_module_or_request(
    mlir::ModuleOp module, std::string_view stage_type,
    std::string_view entry_point, std::vector<int32_t> scalar_args = {},
    GfxKernelBackendDomain backend_domain = GfxKernelBackendDomain::AppleMsl,
    GfxKernelStorageKind storage = GfxKernelStorageKind::Buffer,
    std::string_view specialization_prefix = "apple_msl:buffer:");

GfxKernelRuntimeBindingPlan make_backend_custom_kernel_source_binding_plan(
    const KernelSource &source, bool is_opencl_backend,
    std::string_view stage_type, std::string_view entry_point = {},
    std::vector<int32_t> scalar_args = {});

bool annotate_backend_custom_kernel_module_with_binding_plan(
    mlir::ModuleOp module, const GfxKernelRuntimeBindingPlan &plan);

bool configure_backend_custom_kernel_source_from_binding_plan(
    KernelSource &source, const GfxKernelRuntimeBindingPlan &plan);

bool configure_backend_custom_kernel_source_binding(
    KernelSource &source, bool is_opencl_backend, std::string_view stage_type,
    std::string_view entry_point,
    std::vector<int32_t> scalar_args = {});

void require_backend_custom_kernel_source_binding(
    KernelSource &source, bool is_opencl_backend, std::string_view stage_type,
    std::string_view entry_point,
    std::vector<int32_t> scalar_args = {});

GfxKernelRuntimeBindingPlan require_backend_custom_kernel_binding_plan(
    bool is_opencl_backend, std::string_view stage_type,
    std::string_view entry_point, const std::vector<int32_t> &scalar_args,
    std::string_view stage_name);

KernelRuntimeBindingState require_backend_custom_kernel_runtime_binding(
    bool is_opencl_backend, std::string_view stage_type,
    std::string_view entry_point, const std::vector<int32_t> &scalar_args,
    std::string_view stage_name);

GfxKernelRuntimeBindingPlan annotate_required_backend_custom_kernel_binding(
    mlir::ModuleOp module, bool is_opencl_backend, std::string_view stage_type,
    std::string_view entry_point, const std::vector<int32_t> &scalar_args,
    std::string_view stage_name);

GfxKernelRuntimeBindingPlan annotate_required_backend_custom_kernel_abi_binding(
    mlir::ModuleOp module, bool is_opencl_backend, std::string_view stage_type,
    std::string_view entry_point, std::string_view stage_name);

GfxKernelRuntimeBindingPlan
annotate_required_backend_custom_kernel_direct_io_binding(
    mlir::ModuleOp module, bool is_opencl_backend, std::string_view stage_type,
    std::string_view entry_point, size_t tensor_input_count,
    size_t output_count, std::string_view stage_name);

bool configure_backend_custom_kernel_source_signature(
    KernelSource &source, const GfxKernelStageManifest &manifest);

bool configure_backend_custom_kernel_source_signature_from_module(
    KernelSource &source);

bool configure_backend_custom_kernel_source_signature(
    KernelSource &source, const GfxKernelRuntimeBindingPlan &plan);

size_t require_backend_manifest_arg_count(mlir::ModuleOp module,
                                          bool is_opencl_backend,
                                          std::string_view entry_point,
                                          std::string_view stage_name);

size_t require_backend_manifest_arg_count(
    mlir::ModuleOp module, GfxKernelBackendDomain backend_domain,
    std::string_view entry_point, std::string_view stage_name);

size_t infer_backend_custom_kernel_arg_count(
    mlir::ModuleOp module, GfxKernelBackendDomain backend_domain,
    size_t fallback, std::string_view entry_point = {});

} // namespace gfx_plugin
} // namespace ov
