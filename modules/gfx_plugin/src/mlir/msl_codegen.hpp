// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

// Convenience include for MSL codegen helpers backed by MLIR analysis.
#include "kernel_ir/gfx_codegen_backend.hpp"
#include "kernel_ir/gfx_custom_kernel_families.hpp"
#include "kernel_ir/gfx_kernel_spec.hpp"
#include "mlir/IR/MLIRContext.h"
#include "mlir/codegen_common.hpp"
#include "mlir/gfx_mlir_kernel_metadata.hpp"
#include "mlir/gfx_mpsrt_source_plan.hpp"
#include "mlir/msl_codegen_matmul_metal.hpp"
#include "mlir/msl_codegen_matmul_mpsrt.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "runtime/gfx_parallelism.hpp"
#include "runtime/gfx_stage_policy.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ov {
namespace gfx_plugin {

std::string
normalize_msl_source_for_kernel_plan(std::string source,
                                     std::string_view current_entry_point,
                                     const GfxCustomKernelStagePlan &plan);

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

void configure_msl_kernel_source_for_plan(KernelSource &source,
                                          std::string_view stage_type);
GfxMpsrtKernelSourcePlan
configure_msl_kernel_source_plan(KernelSource source,
                                 std::string_view stage_type);
GfxMpsrtKernelSourcePlan configure_msl_kernel_source_plan_for_node(
    KernelSource source, const std::shared_ptr<const ov::Node> &node,
    const GpuBufferManager *buffer_manager, std::string_view stage_type,
    bool has_bias, bool has_activation, bool has_batchnorm);
GfxMpsrtKernelSourcePlan configure_apple_metal_kernel_source_plan_for_stage(
    KernelSource &source, const std::shared_ptr<const ov::Node> &node,
    const GpuBufferManager *buffer_manager, std::string_view stage_type,
    bool has_bias, bool has_activation, bool has_batchnorm,
    ActivationKind activation, const ov::element::Type &storage_type,
    bool has_runtime_slice_params,
    const std::optional<ov::Shape> &runtime_input_shape = std::nullopt);
GfxMpsrtKernelSourcePlan configure_msl_kernel_source_plan_for_spec(
    KernelSource source, const KernelSpec &spec,
    const GpuBufferManager *buffer_manager, std::string_view entry_point);

// Apple MSL two-phase lowering boundary: stage manifest first, then typed
// MPSRT program materialization.
void annotate_msl_module_with_stage_plan(
    mlir::ModuleOp module, const GfxStageOptimizationPlan &plan,
    const std::string &stage_type, std::string_view kernel_entry_point = {});

struct CompressedMatMulPart {
  std::shared_ptr<const ov::op::v0::Constant> weights;
  std::shared_ptr<const ov::op::v0::Constant> scale;
  size_t n = 0;
  size_t groups = 0;
  size_t group_size = 0;
  size_t k = 0;
};

struct CompressedMatMulInfo {
  std::vector<CompressedMatMulPart> parts;
  std::shared_ptr<const ov::op::v0::Constant> weights;
  std::shared_ptr<const ov::op::v0::Constant> scale;
  ov::element::Type input_type = ov::element::dynamic;
  ov::element::Type output_type = ov::element::dynamic;
  bool signed_weights = true;
  size_t n = 0;
  size_t k = 0;
  size_t groups = 0;
  size_t group_size = 0;
};

std::optional<CompressedMatMulInfo>
detect_compressed_matmul_weights(const std::shared_ptr<const ov::Node> &node);
uint32_t
compressed_matmul_parallel_reduction_threads(const CompressedMatMulInfo &info,
                                             const GfxParallelismCaps &caps);
uint32_t compressed_matmul_output_block(const CompressedMatMulInfo &info,
                                        const GfxParallelismCaps &caps,
                                        uint32_t reduction_threads);
std::vector<uint8_t> pack_compressed_matmul_weights_for_output_block(
    const CompressedMatMulInfo &info, uint32_t output_block);
std::vector<uint8_t>
pack_compressed_matmul_scales(const CompressedMatMulInfo &info);
std::string generate_msl_for_compressed_matmul(const CompressedMatMulInfo &info,
                                               uint32_t reduction_threads,
                                               uint32_t output_block);
KernelSource
make_compressed_matmul_msl_kernel_source(const CompressedMatMulInfo &info,
                                         uint32_t reduction_threads,
                                         uint32_t output_block);
std::string generate_msl_for_sdpa(ov::element::Type type);
std::string generate_msl_for_sdpa_with_causal_mask(ov::element::Type type);

struct GfxMslRuntimeBindingPlan {
  KernelRuntimeBindingState runtime_binding;
  size_t scalar_arg_count = 0;
  GfxKernelStageManifest stage_manifest;

  KernelRuntimeBindingState kernel_runtime_binding() const {
    return runtime_binding;
  }

  bool valid() const {
    return stage_manifest.valid && !runtime_binding.operand_kinds.empty() &&
           runtime_binding.operand_kinds.size() ==
               runtime_binding.operand_arg_indices.size() &&
           (runtime_binding.scalar_args.empty() ||
            runtime_binding.scalar_args.size() == scalar_arg_count);
  }
};

GfxMslRuntimeBindingPlan make_msl_runtime_binding_plan_from_stage_manifest(
    const GfxKernelStageManifest &manifest);
GfxMslRuntimeBindingPlan
make_msl_runtime_binding_plan_for_custom_kernel(std::string_view stage_type,
                                                std::string_view entry_point);
GfxMslRuntimeBindingPlan make_msl_runtime_binding_plan_for_custom_kernel(
    std::string_view stage_type, std::string_view entry_point,
    std::vector<int32_t> scalar_args);
GfxMslRuntimeBindingPlan
make_msl_runtime_binding_plan_for_direct_io_custom_kernel(
    std::string_view stage_type, std::string_view entry_point,
    size_t tensor_input_count, size_t output_count);
bool annotate_msl_module_with_runtime_binding_plan(
    mlir::ModuleOp module, const GfxMslRuntimeBindingPlan &plan);

struct GfxDirectSplitMslKernelSourcePlan {
  KernelSource source;
  GfxMslRuntimeBindingPlan binding;

  bool valid() const { return !source.msl_source.empty() && binding.valid(); }
};

GfxDirectSplitMslKernelSourcePlan make_direct_split_msl_kernel_source_plan(
    std::string_view stage_type, const ov::element::Type &element_type,
    const ov::Shape &input_shape, const std::vector<size_t> &split_sizes,
    uint32_t axis_len, uint32_t inner_stride, mlir::ModuleOp module = {});

struct GfxCompressedMatMulMslKernelSourcePlan {
  KernelSource source;
  GfxMslRuntimeBindingPlan binding;
  GfxCustomKernelStagePlan custom_kernel_plan;

  bool valid() const {
    return !source.msl_source.empty() && binding.valid() &&
           custom_kernel_plan.valid;
  }
};

GfxCompressedMatMulMslKernelSourcePlan
make_compressed_matmul_msl_kernel_source_plan(const CompressedMatMulInfo &info,
                                              uint32_t reduction_threads,
                                              uint32_t output_block);

struct GfxSdpaMslKernelSourcePlan {
  KernelSource source;
  GfxMslRuntimeBindingPlan binding;
  GfxCustomKernelStagePlan custom_kernel_plan;

  bool valid() const {
    return !source.msl_source.empty() && binding.valid() &&
           custom_kernel_plan.valid;
  }
};

GfxMslRuntimeBindingPlan make_causal_sdpa_msl_runtime_binding_plan();
GfxMslRuntimeBindingPlan make_sdpa_msl_runtime_binding_plan(bool has_mask);
GfxSdpaMslKernelSourcePlan
make_sdpa_msl_kernel_source_plan(ov::element::Type type, bool has_mask);
GfxSdpaMslKernelSourcePlan
make_causal_sdpa_msl_kernel_source_plan(ov::element::Type type);
KernelSource make_sdpa_msl_kernel_source(ov::element::Type type, bool has_mask);
KernelSource make_causal_sdpa_msl_kernel_source(ov::element::Type type);

struct GfxSdpaMslRuntimeParamsPlan {
  std::vector<int32_t> params;
  GfxMslRuntimeBindingPlan binding;

  bool valid() const { return !params.empty() && binding.valid(); }
};

GfxSdpaMslRuntimeParamsPlan make_causal_sdpa_msl_runtime_params_plan(
    const ov::Shape &q_shape, const ov::Shape &k_shape,
    const ov::Shape &v_shape, const ov::Shape &mask_shape, float scale,
    bool k_gqa, size_t k_heads, bool v_gqa, size_t v_heads);
GfxSdpaMslRuntimeParamsPlan make_sdpa_msl_runtime_params_plan(
    const ov::Shape &q_shape, const ov::Shape &k_shape,
    const ov::Shape &v_shape, const ov::Shape &mask_shape, bool has_mask,
    float scale, bool k_gqa, size_t k_heads, bool v_gqa, size_t v_heads);

} // namespace gfx_plugin
} // namespace ov
