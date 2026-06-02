// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "compiler/stage_compiler_policy.hpp"
#include "compiler/stage_placement.hpp"
#include "compiler/tensor_layout.hpp"
#include "compiler/operation_support.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "runtime/gfx_backend_utils.hpp"
#include "runtime/gpu_stage.hpp"

namespace ov {
namespace gfx_plugin {

struct GfxStageFusionPolicy {
  bool allow_bias = false;
  bool allow_activation = false;
  bool allow_batchnorm = false;
};

struct GfxStagePostOpSupport {
  bool bias = false;
  bool activation = false;
  bool batchnorm = false;
};

struct GfxStageExecutionPolicy {
  GfxStageFusionPolicy fusion{};
  GpuStageSubmitPolicy submit{};
};

struct GfxStagePrecisionPlan {
  bool keep_fp32 = false;
};

enum class GfxStageArchetype {
  Unknown,
  Convolution,
  GroupConvolution,
  MatMul,
  UnaryElementwise,
  BinaryElementwise,
  Reduction,
  Layout,
  Convert,
  SplitConcat,
};

enum class GfxConvRouteKind {
  None,
  Direct1x1,
  Direct3x3,
  Chunked,
};

enum class GfxConvFamily {
  Unknown,
  Pointwise1x1,
  Spatial3x3,
  Depthwise,
  Grouped,
  General,
};

enum class GfxConvAlgorithmKind {
  None,
  Direct1x1,
  Direct3x3Stride1,
  Direct3x3Stride2,
  DepthwiseDirect,
  ChunkedDirect,
  Indirect,
};

enum class GfxConvMultiKernelStageKind {
  Unknown,
  PartialReduce,
  ReducePartialSums,
  FinalizeOutput,
};

struct GfxConvMultiKernelStagePlan {
  GfxConvMultiKernelStageKind kind = GfxConvMultiKernelStageKind::Unknown;
  std::string name;
  uint64_t output_elements = 0;
  bool writes_intermediate = false;
  bool writes_final_output = false;
};

struct GfxConvMultiKernelManifestPlan {
  std::string family;
  std::vector<GfxConvMultiKernelStagePlan> stages;
  bool requires_owned_intermediates = false;
  bool requires_owned_launch_sequence = false;
  bool requires_output_reuse = false;
  bool requires_spatial_input_reuse = false;
  bool requires_coarse_output_tile_preservation = false;
  bool has_workgroup_local_reduction_plan = false;
  uint64_t coarse_spatial_tile_elements = 0;
  uint64_t coarse_output_channel_block = 0;
  uint64_t coarse_output_tile_elements = 0;
  uint64_t workgroup_output_tile_deficit = 0;
  uint64_t partial_sum_elements = 0;
  uint64_t reduced_accumulator_elements = 0;
  uint64_t owned_intermediate_elements = 0;
  uint64_t owned_intermediate_bytes = 0;
  uint64_t owned_intermediate_buffer_count = 0;
  uint64_t workgroup_local_accumulator_elements = 0;
  uint64_t workgroup_local_accumulator_bytes = 0;
  uint64_t launch_dispatch_count = 0;
};

struct GfxConvAlgorithmPlan {
  GfxConvAlgorithmKind kind = GfxConvAlgorithmKind::None;
  std::string variant;
  bool requires_multi_kernel_manifest = false;
  std::string multi_kernel_family;
  GfxConvMultiKernelManifestPlan multi_kernel_manifest{};
  uint64_t reduction_work = 0;
  uint64_t output_elements = 0;
  uint64_t intermediate_elements = 0;
  uint64_t reduction_chunk_count = 0;
  uint64_t reduction_chunk_size = 0;
  uint64_t workgroup_reduction_lanes = 0;
  uint64_t workgroup_output_lanes = 0;
  uint64_t output_channel_reuse_lanes = 1;
  uint64_t spatial_output_reuse_lanes = 1;
  uint64_t output_reuse_lanes = 1;
  uint64_t spatial_input_reuse_lanes = 1;
  uint64_t spatial_input_reuse_unique_width_loads = 0;
  uint64_t spatial_input_reuse_saved_width_loads = 0;
};

struct GfxConvRoutePlan {
  GfxConvRouteKind kind = GfxConvRouteKind::None;
  GfxConvFamily family = GfxConvFamily::Unknown;
  GfxConvAlgorithmPlan algorithm{};
};

struct GfxStageOptimizationPlan {
  GfxStageArchetype archetype = GfxStageArchetype::Unknown;
  GfxStagePlacementPlan placement{};
  GfxStagePrecisionPlan precision{};
  GfxTensorLayoutPlan layout{};
  GfxStagePostOpSupport post_ops{};
  GfxStageExecutionPolicy execution{};
  GfxConvRoutePlan conv{};
};

using GfxStageCompilerPolicy = compiler::StageCompilerPolicy;

GfxStageOptimizationPlan select_stage_optimization_plan(
    const GpuBufferManager *buffer_manager, GpuBackend backend,
    const std::string &stage_type, const std::shared_ptr<const ov::Node> &node,
    const ov::element::Type &element_type, bool has_bias, bool has_activation,
    bool has_batchnorm, const GfxStageRuntimeTraits &traits,
    const GfxStageCompilerPolicy *compiler_policy = nullptr);

GfxStageExecutionPolicy
select_stage_execution_policy(const GpuBufferManager *buffer_manager,
                              GpuBackend backend, const std::string &stage_type,
                              const GfxStageRuntimeTraits &traits,
                              const GfxStageCompilerPolicy *compiler_policy = nullptr);
GfxConvRoutePlan
select_conv_route_plan(const GpuBufferManager *buffer_manager,
                       const std::shared_ptr<const ov::Node> &node,
                       const ov::element::Type &element_type, bool has_bias,
                       bool has_activation, bool has_batchnorm,
                       const GfxStageCompilerPolicy *compiler_policy = nullptr);

const char *
gfx_conv_multi_kernel_stage_kind_name(GfxConvMultiKernelStageKind kind);
const char *gfx_stage_backend_domain_name(GfxStageBackendDomain domain);
const char *gfx_stage_storage_kind_name(GfxStageStorageKind storage);

} // namespace gfx_plugin
} // namespace ov
