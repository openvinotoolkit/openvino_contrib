// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen_apple_msl.hpp"

#include <memory>
#include <string_view>

#include "kernel_ir/gfx_custom_kernel_families.hpp"
#include "mlir/gfx_mpsrt_metadata.hpp"
#include "mlir/msl_codegen_apple_msl_ops.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

GfxKernelStageFamily
resolve_apple_metal_msl_stage_family(const KernelSource &source,
                                     std::string_view stage_type) {
  GfxKernelStageManifest manifest{};
  if (source.module &&
      detail::gfx_mpsrt_read_stage_manifest_attrs(source.module, manifest) &&
      manifest.valid &&
      manifest.backend_domain == GfxKernelBackendDomain::AppleMsl &&
      manifest.execution_kind == GfxKernelExecutionKind::CustomKernel) {
    return manifest.stage_family;
  }
  const auto kernel_family =
      classify_gfx_custom_kernel_family(stage_type, source.entry_point);
  return gfx_kernel_stage_family_from_kernel_family(kernel_family);
}

bool configure_apple_metal_msl_kernel_source_for_stage_type(
    KernelSource &source, const std::shared_ptr<const ov::Node> &node,
    std::string_view stage_type,
    const std::optional<ov::Shape> &runtime_input_shape) {
  switch (resolve_apple_metal_msl_stage_family(source, stage_type)) {
  case GfxKernelStageFamily::Convolution:
  case GfxKernelStageFamily::Conv3D:
  case GfxKernelStageFamily::Gemm:
  case GfxKernelStageFamily::RmsnormRope:
    return configure_apple_metal_compute_kernel_source(source, node);
  case GfxKernelStageFamily::Pooling:
    return configure_apple_metal_pool2d_kernel_source(source, node);
  case GfxKernelStageFamily::Softmax:
  case GfxKernelStageFamily::AttentionSoftmax:
    return configure_apple_metal_softmax_kernel_source(source, node,
                                                       runtime_input_shape);
  case GfxKernelStageFamily::Eltwise:
    return configure_apple_metal_elementwise_kernel_source(source, node) ||
           configure_apple_metal_unary_kernel_source(source, node) ||
           configure_apple_metal_structural_kernel_source(source, node);
  case GfxKernelStageFamily::Transpose:
    return configure_apple_metal_data_movement_kernel_source(source, node) ||
           configure_apple_metal_structural_kernel_source(source, node);
  case GfxKernelStageFamily::ConcatSplit:
  case GfxKernelStageFamily::Reduction:
  case GfxKernelStageFamily::TopK:
  case GfxKernelStageFamily::Convert:
  case GfxKernelStageFamily::Layout:
    return configure_apple_metal_structural_kernel_source(source, node);
  case GfxKernelStageFamily::GatherScatter:
    return configure_apple_metal_data_movement_kernel_source(source, node) ||
           configure_apple_metal_structural_kernel_source(source, node);
  case GfxKernelStageFamily::GroupConvolution:
  case GfxKernelStageFamily::Resize:
  case GfxKernelStageFamily::KvCache:
  case GfxKernelStageFamily::Unknown:
  default:
    return false;
  }
}

bool configure_apple_metal_msl_kernel_source_impl(
    KernelSource &source, const std::shared_ptr<const ov::Node> &node,
    std::string_view stage_type, const ov::element::Type &storage_type,
    bool has_runtime_slice_params,
    const std::optional<ov::Shape> &runtime_input_shape) {
  if (!source.msl_generator && source.msl_source.empty()) {
    (void)configure_apple_metal_msl_kernel_source_for_stage_type(
        source, node, stage_type, runtime_input_shape);
  }

  (void)configure_apple_metal_slice_kernel_source(source, node, storage_type,
                                                  has_runtime_slice_params);
  return source.msl_generator || !source.msl_source.empty();
}

} // namespace

bool configure_apple_metal_msl_kernel_source(
    KernelSource &source, const std::shared_ptr<const ov::Node> &node,
    std::string_view stage_type, const ov::element::Type &storage_type,
    bool has_runtime_slice_params,
    const std::optional<ov::Shape> &runtime_input_shape) {
  return configure_apple_metal_msl_kernel_source_impl(
      source, node, stage_type, storage_type, has_runtime_slice_params,
      runtime_input_shape);
}

} // namespace gfx_plugin
} // namespace ov
