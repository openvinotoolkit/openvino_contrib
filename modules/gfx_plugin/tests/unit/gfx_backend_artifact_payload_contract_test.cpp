// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unit/gfx_manifest_executable_contract_utils.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

TEST_F(GfxBackendArchitectureContractTest,
       ExecutablePayloadRoutesRequireCompilerMaterializedPayloads) {
  struct RouteCase {
    LoweringRouteKind route_kind;
    std::string backend_domain;
    std::string kernel_unit_id;
    std::string kernel_unit_kind;
  };

  const std::vector<RouteCase> routes = {
      {LoweringRouteKind::GeneratedKernel, "metal", "metal/generated/test",
       "generated_kernel"},
      {LoweringRouteKind::GeneratedKernel, "opencl", "opencl/generated/test",
       "generated_kernel"},
      {LoweringRouteKind::VendorPrimitive, "metal", "metal/vendor/test",
       "vendor_primitive"},
  };

  for (const auto &route : routes) {
    const auto manifest = make_single_payload_route_manifest(
        route.route_kind, route.backend_domain, route.kernel_unit_id,
        route.kernel_unit_kind);
    ASSERT_TRUE(manifest.valid()) << route.kernel_unit_id;

    const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
    const auto executable_result = executable.verify();
    EXPECT_FALSE(executable_result.valid()) << route.kernel_unit_id;
    EXPECT_TRUE(has_diagnostic_containing(executable_result.diagnostics,
                                          "requires a materialized payload"))
        << route.kernel_unit_id;

    const auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    const auto runtime_result = compiler::verify_runtime_executable_descriptor(
        runtime_descriptor, executable);
    EXPECT_FALSE(runtime_result.valid()) << route.kernel_unit_id;
    EXPECT_TRUE(has_diagnostic_containing(runtime_result.diagnostics,
                                          "requires a materialized payload"))
        << route.kernel_unit_id;
  }
}


TEST_F(GfxBackendArchitectureContractTest,
       OpenClPayloadMaterializationIsOwnedByBackendModule) {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  const compiler::BackendCapabilities capabilities(
      target, compiler::make_opencl_operation_support_policy());
  const compiler::OperationLegalizer legalizer(capabilities);
  const compiler::LoweringPlanner planner(
      target, compiler::make_opencl_kernel_registry(target));

  const auto lowering_plan = planner.plan(models.relu(), legalizer);
  ASSERT_TRUE(lowering_plan.executable());
  const auto manifest = compiler::ManifestBuilder{}.build(lowering_plan);
  ASSERT_TRUE(manifest.valid());

  const auto common_executable =
      compiler::ExecutableBundleBuilder{}.build(manifest, lowering_plan);
  EXPECT_TRUE(common_executable.artifact_payloads.empty());
  EXPECT_FALSE(common_executable.verify().valid());
  EXPECT_TRUE(has_diagnostic_containing(common_executable.verify().diagnostics,
                                        "requires a materialized payload"));

  const auto backend_executable =
      compiler::ExecutableBundleBuilder(
          compiler::make_opencl_kernel_artifact_descriptor_resolver(),
          compiler::make_opencl_kernel_artifact_payload_resolver())
          .build(manifest, lowering_plan);
  ASSERT_TRUE(backend_executable.verify().valid());
  ASSERT_EQ(backend_executable.artifact_payloads.size(), 1u);

  const auto descriptor_index =
      backend_executable.artifact_payloads.front().artifact_descriptor_index;
  ASSERT_LT(descriptor_index, backend_executable.artifact_descriptors.size());
  const auto &artifact =
      backend_executable.artifact_descriptors[descriptor_index];
  EXPECT_EQ(artifact.kernel.kernel_id, "opencl/generated/activation_f32");
  EXPECT_EQ(artifact.payload_kind,
            compiler::KernelArtifactPayloadKind::OpenClSource);
  EXPECT_EQ(artifact.entry_point, "gfx_opencl_generated_activation_f32");
}

TEST_F(GfxBackendArchitectureContractTest,
       PayloadResolverCannotOwnOrMutateArtifactAbi) {
  struct Case {
    compiler::BackendTarget target;
    std::shared_ptr<const compiler::OperationSupportPolicy> policy;
    compiler::KernelRegistry registry;
    compiler::KernelArtifactDescriptorResolver descriptor_resolver;
    compiler::KernelArtifactPayloadResolver payload_resolver;
  };

  const auto opencl_target =
      compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  const auto metal_target =
      compiler::BackendTarget::from_backend(GpuBackend::Metal);
  const std::vector<Case> cases = {
      {opencl_target, compiler::make_opencl_operation_support_policy(),
       compiler::make_opencl_kernel_registry(opencl_target),
       compiler::make_opencl_kernel_artifact_descriptor_resolver(),
       compiler::make_opencl_kernel_artifact_payload_resolver()},
      {metal_target, compiler::make_metal_operation_support_policy(),
       compiler::make_metal_kernel_registry(metal_target),
       compiler::make_metal_kernel_artifact_descriptor_resolver(),
       compiler::make_metal_kernel_artifact_payload_resolver()},
  };

  for (const auto &test_case : cases) {
    const compiler::BackendCapabilities capabilities(test_case.target,
                                                     test_case.policy);
    const compiler::OperationLegalizer legalizer(capabilities);
    const compiler::LoweringPlanner planner(test_case.target,
                                            test_case.registry);
    const auto lowering_plan = planner.plan(models.relu(), legalizer);
    ASSERT_TRUE(lowering_plan.executable()) << test_case.target.debug_string();
    const auto manifest = compiler::ManifestBuilder{}.build(lowering_plan);
    ASSERT_TRUE(manifest.valid()) << test_case.target.debug_string();

    const auto descriptor_only_executable =
        compiler::ExecutableBundleBuilder(test_case.descriptor_resolver, {})
            .build(manifest, lowering_plan);
    size_t checked_payload_descriptors = 0;
    const auto count = std::min(descriptor_only_executable.stages.size(),
                                lowering_plan.operations.size());
    for (size_t i = 0; i < count; ++i) {
      const auto descriptor_index =
          descriptor_only_executable.stages[i].artifact_descriptor_index;
      ASSERT_LT(descriptor_index,
                descriptor_only_executable.artifact_descriptors.size())
          << test_case.target.debug_string();
      const auto descriptor_before =
          descriptor_only_executable.artifact_descriptors[descriptor_index];
      if (descriptor_before.payload_kind ==
          compiler::KernelArtifactPayloadKind::None) {
        continue;
      }
      ASSERT_FALSE(descriptor_before.entry_point.empty())
          << test_case.target.debug_string();
      ASSERT_NE(descriptor_before.abi_arg_count, 0u)
          << test_case.target.debug_string();
      ASSERT_TRUE(descriptor_before.launch_plan.valid)
          << test_case.target.debug_string();
      ASSERT_FALSE(descriptor_before.artifact_key.empty())
          << test_case.target.debug_string();

      auto descriptor_after = descriptor_before;
      const auto payload =
          test_case.payload_resolver(descriptor_after, lowering_plan.operations[i]);
      ASSERT_TRUE(payload) << test_case.target.debug_string() << " "
                           << descriptor_before.kernel.kernel_id;
      expect_artifact_descriptor_abi_equal(descriptor_after, descriptor_before);
      ++checked_payload_descriptors;
    }
    EXPECT_GT(checked_payload_descriptors, 0u)
        << test_case.target.debug_string();
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       OpenClRangePayloadUsesRegisteredOpOwnedKernelUnit) {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  const compiler::BackendCapabilities capabilities(
      target, compiler::make_opencl_operation_support_policy());
  const compiler::OperationLegalizer legalizer(capabilities);
  const compiler::LoweringPlanner planner(
      target, compiler::make_opencl_kernel_registry(target));

  const auto lowering_plan = planner.plan(models.static_range(), legalizer);
  ASSERT_TRUE(lowering_plan.executable());

  bool found_range = false;
  for (const auto &op : lowering_plan.operations) {
    if (op.type_name != "Range") {
      continue;
    }
    found_range = true;
    EXPECT_EQ(op.kernel_unit.route_kind(), LoweringRouteKind::GeneratedKernel);
    EXPECT_EQ(op.kernel_unit.kind(), KernelUnitKind::GeneratedKernel);
    EXPECT_EQ(op.kernel_unit.backend_domain(), "opencl");
    EXPECT_EQ(op.kernel_unit.op_family(), "Range");
    EXPECT_EQ(op.kernel_unit.id(), "opencl/generated/range_f32");
  }
  ASSERT_TRUE(found_range);

  const auto manifest = compiler::ManifestBuilder{}.build(lowering_plan);
  ASSERT_TRUE(manifest.valid());
  const auto backend_executable =
      compiler::ExecutableBundleBuilder(
          compiler::make_opencl_kernel_artifact_descriptor_resolver(),
          compiler::make_opencl_kernel_artifact_payload_resolver())
          .build(manifest, lowering_plan);
  ASSERT_TRUE(backend_executable.verify().valid());
  ASSERT_EQ(backend_executable.artifact_payloads.size(), 1u);

  const auto descriptor_index =
      backend_executable.artifact_payloads.front().artifact_descriptor_index;
  ASSERT_LT(descriptor_index, backend_executable.artifact_descriptors.size());
  const auto &artifact =
      backend_executable.artifact_descriptors[descriptor_index];
  EXPECT_EQ(artifact.kernel.kernel_id, "opencl/generated/range_f32");
  EXPECT_EQ(artifact.payload_kind,
            compiler::KernelArtifactPayloadKind::OpenClSource);
  EXPECT_EQ(artifact.entry_point, "gfx_opencl_generated_range_f32");
  ASSERT_TRUE(backend_executable.artifact_payloads.front().payload);
  EXPECT_EQ(backend_executable.artifact_payloads.front().payload->source_id(),
            "opencl/generated/range_f32");
}

} // namespace
} // namespace gfx_plugin
} // namespace ov
