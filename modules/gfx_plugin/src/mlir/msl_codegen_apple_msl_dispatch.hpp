// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "kernel_ir/gfx_codegen_backend.hpp"
#include "kernel_ir/gfx_custom_kernel_families.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/gfx_mlir_kernel_metadata.hpp"
#include "mlir/gfx_mpsrt_metadata.hpp"
#include "mlir/gfx_mpsrt_source_plan.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "backends/metal/runtime/mpsrt/gfx_mpsrt_program.hpp"
#include "runtime/gfx_stage_policy.hpp"

namespace ov {
namespace gfx_plugin {

class GpuBufferManager;

struct GfxMslGeneratedKernelSourcePlan {
  KernelSource source;
  GfxKernelRuntimeBindingPlan binding;

  bool valid() const {
    return (source.msl_generator || !source.msl_source.empty()) &&
           binding.valid;
  }
};

GfxMslGeneratedKernelSourcePlan make_msl_generated_custom_kernel_source_plan(
    KernelSource source, std::string_view stage_type);
GfxMslGeneratedKernelSourcePlan make_msl_generated_custom_kernel_source_plan(
    KernelSource source, const GfxKernelRuntimeBindingPlan &binding);

struct GfxAppleMslStageLoweringPlan {
  bool valid = false;
  GfxMpsrtModuleStagePlan stage_plan;
  GfxCustomKernelStagePlan custom_kernel_plan;
};

GfxAppleMslStageLoweringPlan materialize_apple_msl_stage_manifest(
    mlir::ModuleOp module, const GfxStageOptimizationPlan &plan,
    const std::string &stage_type, std::string_view kernel_entry_point = {});

bool materialize_apple_msl_typed_program(
    mlir::ModuleOp module, const GfxAppleMslStageLoweringPlan &lowering_plan,
    const GfxMpsrtExternalBufferAbiPlan &external_buffer_abi = {});

void force_apple_msl_buffer_placement(GfxStageOptimizationPlan &plan,
                                      std::string_view stage_type);

GfxMpsrtKernelSourcePlan
configure_msl_kernel_source_plan(KernelSource source,
                                 std::string_view stage_type);

void annotate_msl_module_with_stage_plan(
    mlir::ModuleOp module, const GfxStageOptimizationPlan &plan,
    const std::string &stage_type, std::string_view kernel_entry_point = {});

} // namespace gfx_plugin
} // namespace ov
