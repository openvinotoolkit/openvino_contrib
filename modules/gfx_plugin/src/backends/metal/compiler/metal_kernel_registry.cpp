// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/kernel_registry.hpp"

#include <utility>

namespace ov {
namespace gfx_plugin {
namespace compiler {

namespace {

KernelUnit make_metal_lowering_unit(const BackendTarget &target) {
  return KernelUnit::describe(LoweringRouteKind::BackendLowering,
                              KernelUnitKind::BackendLowering, "metal_lowering",
                              target.backend_id(),
                              "apple_mps_mpsgraph_msl_transition");
}

KernelUnit make_metal_generated_unit(const BackendTarget &target,
                                     const char *unit_id,
                                     const char *op_family) {
  return KernelUnit::describe(LoweringRouteKind::GeneratedKernel,
                              KernelUnitKind::GeneratedKernel, unit_id,
                              target.backend_id(), op_family);
}

KernelUnit make_metal_vendor_unit(const BackendTarget &target,
                                  const char *unit_id, const char *op_family) {
  return KernelUnit::describe(LoweringRouteKind::VendorPrimitive,
                              KernelUnitKind::VendorPrimitive, unit_id,
                              target.backend_id(), op_family);
}

} // namespace

KernelRegistry make_metal_kernel_registry(const BackendTarget &target) {
  auto units = make_common_kernel_units(target);
  units.push_back(make_metal_lowering_unit(target));
  units.push_back(
      make_metal_generated_unit(target, "metal/generated/shapeof", "ShapeOf"));
  units.push_back(
      make_metal_generated_unit(target, "metal/generated/range", "Range"));
  units.push_back(
      make_metal_generated_unit(target, "metal/generated/tile", "Tile"));
  units.push_back(
      make_metal_generated_unit(target, "metal/generated/concat", "Concat"));
  units.push_back(
      make_metal_generated_unit(target, "metal/generated/split", "Split"));
  units.push_back(
      make_metal_generated_unit(target, "metal/generated/slice", "Slice"));
  units.push_back(make_metal_generated_unit(
      target, "metal/generated/sdpa_causal_mask", "GfxSDPAWithCausalMask"));
  units.push_back(make_metal_generated_unit(
      target, "metal/generated/activation", "Activation"));
  units.push_back(
      make_metal_generated_unit(target, "metal/generated/eltwise", "Eltwise"));
  units.push_back(make_metal_generated_unit(
      target, "metal/generated/reduction_f32", "Reduction"));
  units.push_back(make_metal_generated_unit(
      target, "metal/generated/reduction_logical_bool", "Reduction"));
  units.push_back(
      make_metal_vendor_unit(target, "metal/vendor/mps_gemm", "MatMul"));
  units.push_back(
      make_metal_vendor_unit(target, "metal/vendor/mps_softmax", "Softmax"));
  units.push_back(
      make_metal_vendor_unit(target, "metal/vendor/mps_pool2d", "MaxPool"));
  units.push_back(make_metal_vendor_unit(target, "metal/vendor/mps_resize2d",
                                         "Interpolate"));
  units.push_back(make_metal_vendor_unit(target, "metal/vendor/mpsgraph_sdpa",
                                         "ScaledDotProductAttention"));
  return KernelRegistry(target, std::move(units));
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
