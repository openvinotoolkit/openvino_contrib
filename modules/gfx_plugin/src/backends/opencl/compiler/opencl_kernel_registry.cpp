// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/kernel_registry.hpp"

#include <utility>

namespace ov {
namespace gfx_plugin {
namespace compiler {

namespace {

KernelUnit make_opencl_generated_kernel_unit(const BackendTarget &target,
                                             const char *unit_id,
                                             const char *op_family) {
  return KernelUnit::describe(LoweringRouteKind::GeneratedKernel,
                              KernelUnitKind::GeneratedKernel, unit_id,
                              target.backend_id(), op_family);
}

KernelUnit make_opencl_exception_unit(const BackendTarget &target,
                                      const char *unit_id,
                                      const char *op_family) {
  return KernelUnit::describe(LoweringRouteKind::HandwrittenKernelException,
                              KernelUnitKind::HandwrittenException, unit_id,
                              target.backend_id(), op_family);
}

} // namespace

KernelRegistry make_opencl_kernel_registry(const BackendTarget &target) {
  auto units = make_common_kernel_units(target);
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/interpolate_f32", "Interpolate"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/interpolate_f16", "Interpolate"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/matmul_f32", "MatMul"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/activation_f32", "Activation"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/activation_f16", "Activation"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/eltwise_binary_f32", "Eltwise"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/eltwise_scalar_f32", "Eltwise"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/eltwise_const_f32", "Eltwise"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/eltwise_broadcast_f32", "Eltwise"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/eltwise_binary_f16", "Eltwise"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/eltwise_scalar_f16", "Eltwise"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/eltwise_broadcast_f16", "Eltwise"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/eltwise_binary_i32", "Eltwise"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/eltwise_scalar_i32", "Eltwise"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/eltwise_broadcast_i32", "Eltwise"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/reduction_f32", "Reduction"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/pool2d_f32", "Pooling"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/pool2d_f16", "Pooling"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/softmax_f32", "Softmax"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/softmax_f16", "Softmax"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/softmax_f32_dynamic_static_rank", "Softmax"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/softmax_f16_dynamic_static_rank", "Softmax"));
  units.push_back(make_opencl_exception_unit(
      target, "opencl/baseline/reduce_logical_bool", "Reduction"));
  return KernelRegistry(target, std::move(units));
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
