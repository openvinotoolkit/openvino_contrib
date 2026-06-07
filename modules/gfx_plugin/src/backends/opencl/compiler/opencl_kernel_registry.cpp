// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/kernel_registry.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ov {
namespace gfx_plugin {
namespace compiler {

namespace {

KernelUnit make_opencl_generated_kernel_unit(const BackendTarget &target,
                                             std::string unit_id,
                                             const char *op_family) {
  return KernelUnit::describe(LoweringRouteKind::GeneratedKernel,
                              KernelUnitKind::GeneratedKernel,
                              std::move(unit_id),
                              target.backend_id(), op_family);
}

void append_opencl_split_kernel_units(const BackendTarget &target,
                                      std::vector<KernelUnit> &units) {
  constexpr uint32_t kMaxStaticSplitOutputs = 30;
  const char *type_suffixes[] = {"f32", "f16"};
  for (uint32_t output_count = 1; output_count <= kMaxStaticSplitOutputs;
       ++output_count) {
    for (const char *type_suffix : type_suffixes) {
      units.push_back(make_opencl_generated_kernel_unit(
          target,
          "opencl/generated/split" + std::to_string(output_count) + "_" +
              type_suffix,
          "Split"));
    }
  }
}

void append_opencl_concat_kernel_units(const BackendTarget &target,
                                       std::vector<KernelUnit> &units) {
  constexpr uint32_t kMaxStaticConcatInputs = 30;
  const char *type_suffixes[] = {"f32", "f16"};
  for (uint32_t input_count = 1; input_count <= kMaxStaticConcatInputs;
       ++input_count) {
    for (const char *type_suffix : type_suffixes) {
      units.push_back(make_opencl_generated_kernel_unit(
          target,
          "opencl/generated/concat" + std::to_string(input_count) + "_" +
              type_suffix,
          "Concat"));
    }
  }
  for (uint32_t input_count = 2; input_count <= 4; ++input_count) {
    units.push_back(make_opencl_generated_kernel_unit(
        target,
        "opencl/generated/concat" + std::to_string(input_count) +
            "_f16_dynamic",
        "Concat"));
  }
}

} // namespace

KernelRegistry make_opencl_kernel_registry(const BackendTarget &target) {
  auto units = make_common_kernel_units(target);
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/conv2d_f32", "Convolution"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/group_conv2d_f32", "GroupConvolution"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/interpolate_f32", "Interpolate"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/interpolate_f16", "Interpolate"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/matmul_f32", "MatMul"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/shapeof_i32", "ShapeOf"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/shapeof_i64", "ShapeOf"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/range_f32", "Range"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/range_f16", "Range"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/range_i64", "Range"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/range_i64_unit_dynamic", "Range"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/tile_f32", "Tile"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/tile_dynamic_f32", "Tile"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/tile_f16", "Tile"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/tile_dynamic_f16", "Tile"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/activation_f32", "Activation"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/activation_f16", "Activation"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/activation_runtime_beta_f32", "Activation"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/activation_runtime_beta_f16", "Activation"));
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
      target, "opencl/generated/eltwise_logical_unary_bool", "Eltwise"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/eltwise_logical_binary_bool", "Eltwise"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/eltwise_logical_binary_broadcast_bool",
      "Eltwise"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/eltwise_compare_f32", "Eltwise"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/eltwise_compare_broadcast_f32", "Eltwise"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/eltwise_select_f32", "Eltwise"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/eltwise_select_broadcast_f32", "Eltwise"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/eltwise_select_f16_dynamic", "Eltwise"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/reduction_f32", "Reduction"));
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/reduction_bool", "Reduction"));
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
  units.push_back(make_opencl_generated_kernel_unit(
      target, "opencl/generated/transpose_f32", "Transpose"));
  append_opencl_concat_kernel_units(target, units);
  append_opencl_split_kernel_units(target, units);
  return KernelRegistry(target, std::move(units));
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
