// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <vector>

#include "backends/metal/compiler/metal_operation_support.hpp"
#include "common/gpu_backend.hpp"
#include "compiler/backend_target.hpp"
#include "compiler/kernel_registry.hpp"
#include "compiler/operation_support.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "unit/gfx_backend_contracts.hpp"

using ov::gfx_plugin::compiler::KernelUnitKind;
using ov::gfx_plugin::compiler::LoweringRouteKind;

namespace ov {
namespace gfx_plugin {
namespace {

class GfxKernelRegistryContractTest : public ::testing::Test {};

struct ExpectedGeneratedUnit {
  std::string_view route;
  std::string_view backend_domain;
  std::string_view op_family;
};

void expect_generated_unit(const test::KernelRegistryContract &registry,
                           const ExpectedGeneratedUnit &expected) {
  const auto unit =
      registry.resolve_unit(LoweringRouteKind::GeneratedKernel, expected.route);
  ASSERT_TRUE(unit.valid()) << expected.route;
  EXPECT_EQ(unit.kind(), KernelUnitKind::GeneratedKernel) << expected.route;
  EXPECT_EQ(unit.backend_domain(), expected.backend_domain) << expected.route;
  EXPECT_EQ(unit.op_family(), expected.op_family) << expected.route;
  EXPECT_FALSE(unit.exception_contract().valid()) << expected.route;
}

void expect_rejected_generated_unit(
    const test::KernelRegistryContract &registry, std::string_view route) {
  EXPECT_TRUE(registry.rejects_unit(LoweringRouteKind::GeneratedKernel, route))
      << route;
}

TEST_F(GfxKernelRegistryContractTest,
       KernelRegistriesRequireExplicitOpOwnedUnits) {
  const auto opencl_registry = test::KernelRegistryContract::for_opencl();
  ASSERT_TRUE(opencl_registry.audit_is_valid());

  const auto opencl_common_unit =
      opencl_registry.resolve_unit(LoweringRouteKind::Common, "common");
  ASSERT_TRUE(opencl_common_unit.valid());
  EXPECT_EQ(opencl_common_unit.kind(), KernelUnitKind::Common);
  EXPECT_EQ(opencl_common_unit.op_family(), "common_io");
  const auto opencl_metadata_unit =
      opencl_registry.resolve_unit(LoweringRouteKind::Metadata, "metadata");
  ASSERT_TRUE(opencl_metadata_unit.valid());
  EXPECT_EQ(opencl_metadata_unit.kind(), KernelUnitKind::Metadata);
  EXPECT_EQ(opencl_metadata_unit.op_family(), "metadata");

  expect_rejected_generated_unit(opencl_registry, "");
  expect_rejected_generated_unit(opencl_registry, "opencl_generated_kernel");
  expect_rejected_generated_unit(opencl_registry,
                                 "opencl/generated/matmul_f32");

  const std::vector<ExpectedGeneratedUnit> opencl_generated_units = {
      {"opencl/generated/shapeof_i64", "opencl", "ShapeOf"},
      {"opencl/generated/range_f32", "opencl", "Range"},
      {"opencl/generated/range_i64_unit_dynamic", "opencl", "Range"},
      {"opencl/generated/tile_f32", "opencl", "Tile"},
      {"opencl/generated/eltwise_binary_f32", "opencl", "Eltwise"},
      {"opencl/generated/eltwise_logical_binary_bool", "opencl", "Eltwise"},
      {"opencl/generated/eltwise_compare_f32", "opencl", "Eltwise"},
      {"opencl/generated/eltwise_select_f32", "opencl", "Eltwise"},
      {"opencl/generated/activation_f32", "opencl", "Activation"},
      {"opencl/generated/activation_runtime_beta_f32", "opencl", "Activation"},
      {"opencl/generated/pool2d_f32", "opencl", "Pooling"},
      {"opencl/generated/interpolate_f32", "opencl", "Interpolate"},
      {"opencl/generated/interpolate_f16", "opencl", "Interpolate"},
      {"opencl/generated/reduction_f32", "opencl", "Reduction"},
      {"opencl/generated/reduction_bool", "opencl", "Reduction"},
  };
  for (const auto &expected : opencl_generated_units) {
    expect_generated_unit(opencl_registry, expected);
  }

  expect_rejected_generated_unit(opencl_registry,
                                 "opencl/generated/transpose_f32");
  expect_rejected_generated_unit(opencl_registry,
                                 "opencl/generated/split3_f32");
  expect_rejected_generated_unit(opencl_registry,
                                 "opencl/generated/concat2_f32");

  const auto metal_registry = test::KernelRegistryContract::for_metal();
  ASSERT_TRUE(metal_registry.audit_is_valid());
  const auto metal_common_unit =
      metal_registry.resolve_unit(LoweringRouteKind::Common, "common");
  ASSERT_TRUE(metal_common_unit.valid());
  EXPECT_EQ(metal_common_unit.kind(), KernelUnitKind::Common);
  EXPECT_EQ(metal_common_unit.op_family(), "common_io");
  const auto metal_metadata_unit =
      metal_registry.resolve_unit(LoweringRouteKind::Metadata, "metadata");
  ASSERT_TRUE(metal_metadata_unit.valid());
  EXPECT_EQ(metal_metadata_unit.kind(), KernelUnitKind::Metadata);
  EXPECT_EQ(metal_metadata_unit.op_family(), "metadata");
  EXPECT_TRUE(
      metal_registry.rejects_unit(LoweringRouteKind::GeneratedKernel, ""));
  EXPECT_TRUE(
      metal_registry.rejects_unit(LoweringRouteKind::VendorPrimitive, ""));
  EXPECT_TRUE(metal_registry.rejects_unit(LoweringRouteKind::GeneratedKernel,
                                          "metal_lowering"));
  EXPECT_TRUE(metal_registry.rejects_unit(LoweringRouteKind::VendorPrimitive,
                                          "metal_lowering"));
  EXPECT_TRUE(metal_registry
                  .resolve_unit(LoweringRouteKind::VendorPrimitive,
                                "metal/vendor/mps_gemm")
                  .valid());
  EXPECT_TRUE(metal_registry
                  .resolve_unit(LoweringRouteKind::GeneratedKernel,
                                "metal/generated/slice")
                  .valid());

  const std::vector<ExpectedGeneratedUnit> metal_generated_units = {
      {"metal/generated/eltwise", "metal", "Eltwise"},
      {"metal/generated/activation", "metal", "Activation"},
      {"metal/generated/transpose_f32", "metal", "Transpose"},
  };
  for (const auto &expected : metal_generated_units) {
    expect_generated_unit(metal_registry, expected);
  }

  const auto metal_pool_unit = metal_registry.resolve_unit(
      LoweringRouteKind::VendorPrimitive, "metal/vendor/mps_pool2d");
  ASSERT_TRUE(metal_pool_unit.valid());
  EXPECT_EQ(metal_pool_unit.op_family(), "Pooling");
}

TEST_F(GfxKernelRegistryContractTest,
       MetalUnsupportedCoverageNeverSelectsGenericKernelUnit) {
  const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                             ov::Shape{2, 3});
  const auto convert =
      std::make_shared<ov::op::v0::Convert>(input, ov::element::f16);
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::Metal);
  const compiler::BackendCapabilities capabilities(
      target, compiler::make_metal_operation_support_policy());

  const auto support = capabilities.query_operation({convert});
  EXPECT_FALSE(support.semantic_legal);
  EXPECT_EQ(support.preferred_route_kind, LoweringRouteKind::Unsupported);
  EXPECT_NE(support.preferred_route, "backend_lowering");
  EXPECT_TRUE(support.semantic_reason == "missing_metal_explicit_kernel_unit" ||
              support.semantic_reason == "unsupported_by_metal_capabilities");
}

TEST_F(GfxKernelRegistryContractTest,
       KernelRegistryAuditRejectsUndocumentedHandwrittenExceptions) {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  compiler::KernelRegistry registry{
      target,
      {compiler::KernelUnit::describe(
          LoweringRouteKind::HandwrittenKernelException,
          KernelUnitKind::HandwrittenException, "opencl/baseline/undocumented",
          target.backend_id(), "Eltwise")}};

  const auto audit = registry.audit();
  EXPECT_FALSE(audit.valid());
  EXPECT_EQ(audit.handwritten_exception_count, 1u);
  bool found_contract_diagnostic = false;
  for (const auto &diagnostic : audit.diagnostics) {
    if (diagnostic.find("missing exception contract") != std::string::npos) {
      found_contract_diagnostic = true;
    }
  }
  EXPECT_TRUE(found_contract_diagnostic);
}

} // namespace
} // namespace gfx_plugin
} // namespace ov
