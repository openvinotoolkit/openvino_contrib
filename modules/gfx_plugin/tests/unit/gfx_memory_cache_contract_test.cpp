// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unit/gfx_manifest_executable_contract_utils.hpp"
#include "unit/gfx_cache_materialization_contract_utils.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

TEST_F(GfxBackendArchitectureContractTest,
       CompilerOwnsBackendNeutralTensorLayoutClassification) {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 2, 3});
  auto identity_perm =
      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 1, 2});
  auto identity_transpose =
      std::make_shared<ov::op::v1::Transpose>(input, identity_perm);
  const auto identity_plan = compiler::select_tensor_layout_plan(
      identity_transpose->get_type_name(), identity_transpose);
  EXPECT_TRUE(identity_plan.view_only);
  EXPECT_EQ(identity_plan.kind, GfxTensorLayoutKind::ViewOnly);

  auto materialized_perm =
      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
  auto materialized_transpose =
      std::make_shared<ov::op::v1::Transpose>(input, materialized_perm);
  const auto materialized_plan = compiler::select_tensor_layout_plan(
      materialized_transpose->get_type_name(), materialized_transpose);
  EXPECT_FALSE(materialized_plan.view_only);
  EXPECT_EQ(materialized_plan.kind, GfxTensorLayoutKind::Materialized);

  const auto reshape_plan =
      compiler::select_tensor_layout_plan("Reshape", nullptr);
  EXPECT_TRUE(reshape_plan.view_only);
  EXPECT_EQ(reshape_plan.kind, GfxTensorLayoutKind::ViewOnly);

  const auto read_value_plan =
      compiler::select_tensor_layout_plan("ReadValue", nullptr);
  EXPECT_FALSE(read_value_plan.view_only);
  EXPECT_EQ(read_value_plan.kind, GfxTensorLayoutKind::Materialized);

  const auto view_descriptor =
      make_runtime_descriptor_for_layout(identity_transpose, identity_plan);
  EXPECT_EQ(view_descriptor.layout_contract, "view_only");
  EXPECT_TRUE(view_descriptor.tensor_view_only);

  const auto materialized_descriptor = make_runtime_descriptor_for_layout(
      materialized_transpose, materialized_plan);
  EXPECT_EQ(materialized_descriptor.layout_contract, "materialized");
  EXPECT_FALSE(materialized_descriptor.tensor_view_only);
}

TEST_F(GfxBackendArchitectureContractTest,
       CompilerOwnsMemoryPlanInManifestExecutableAndCacheIdentity) {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 2, 3});
  const auto layout = compiler::select_tensor_layout_plan("Relu", input);
  compiler::LoweringPlan lowering_plan;
  lowering_plan.target =
      compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  lowering_plan.operations.push_back(
      make_metadata_planned_operation(input, layout));

  const auto manifest = compiler::ManifestBuilder{}.build(lowering_plan);
  ASSERT_TRUE(manifest.verify().valid());
  ASSERT_TRUE(manifest.memory_plan.valid());
  ASSERT_EQ(manifest.stages.size(), 1u);
  const auto &stage = manifest.stages.front();
  ASSERT_FALSE(stage.inputs.empty());
  ASSERT_FALSE(stage.outputs.empty());
  EXPECT_TRUE(
      manifest.memory_plan.has_region(stage.inputs.front().memory_region_id));
  EXPECT_TRUE(
      manifest.memory_plan.has_region(stage.outputs.front().memory_region_id));
  EXPECT_TRUE(manifest.memory_plan.has_alias_group(stage.memory.alias_group));
  EXPECT_FALSE(manifest.memory_plan.hidden_host_copies_allowed);

  const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
  ASSERT_TRUE(executable.verify().valid());
  EXPECT_EQ(compiler::make_memory_plan_fingerprint(executable.memory_plan),
            compiler::make_memory_plan_fingerprint(manifest.memory_plan));

  auto stale_manifest = manifest;
  ASSERT_FALSE(stale_manifest.memory_plan.regions.empty());
  stale_manifest.memory_plan.regions.front().layout = "stale_layout";
  EXPECT_NE(compiler::make_manifest_cache_hash(stale_manifest),
            compiler::make_manifest_cache_hash(manifest));
}

TEST_F(GfxBackendArchitectureContractTest,
       CompilerMemoryPlanUsesProducerConsumerLifetimesAcrossTargets) {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 2, 3});
  auto relu0 = std::make_shared<ov::op::v0::Relu>(input);
  auto relu1 = std::make_shared<ov::op::v0::Relu>(relu0);
  auto relu2 = std::make_shared<ov::op::v0::Relu>(relu1);
  auto result = std::make_shared<ov::op::v0::Result>(relu2);
  auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                           ov::ParameterVector{input});
  (void)model;

  struct TargetCase {
    std::string name;
    compiler::BackendTarget target;
  };
  const std::vector<TargetCase> targets = {
      {"macos",
       compiler::BackendTarget::from_backend(GpuBackend::Metal)},
      {"android",
       compiler::BackendTarget::from_backend_device_family(
           GpuBackend::OpenCL, GpuDeviceFamily::QualcommAdreno)},
      {"rpi4",
       compiler::BackendTarget::from_backend_device_family(
           GpuBackend::OpenCL, GpuDeviceFamily::BroadcomV3D)},
      {"rpi5",
       compiler::BackendTarget::from_backend_device_family(
           GpuBackend::OpenCL, GpuDeviceFamily::BroadcomV3D)}};

  for (const auto &target : targets) {
    SCOPED_TRACE(target.name);
    compiler::LoweringPlan lowering_plan;
    lowering_plan.target = target.target;
    lowering_plan.operations.push_back(make_metadata_planned_operation(
        relu0, compiler::select_tensor_layout_plan("Relu", relu0)));
    lowering_plan.operations.push_back(make_metadata_planned_operation(
        relu1, compiler::select_tensor_layout_plan("Relu", relu1)));
    lowering_plan.operations.push_back(make_metadata_planned_operation(
        relu2, compiler::select_tensor_layout_plan("Relu", relu2)));

    const auto manifest = compiler::ManifestBuilder{}.build(lowering_plan);
    ASSERT_TRUE(manifest.verify().valid());
    ASSERT_EQ(manifest.stages.size(), 3u);

    const auto &memory_plan = manifest.memory_plan;
    const auto &stage0_output = manifest.stages[0].outputs.front();
    const auto &stage1_input = manifest.stages[1].inputs.front();
    const auto &stage1_output = manifest.stages[1].outputs.front();
    const auto &stage2_output = manifest.stages[2].outputs.front();
    EXPECT_EQ(stage1_input.memory_region_id, stage0_output.memory_region_id);

    const auto *stage0_region =
        memory_plan.find_region(stage0_output.memory_region_id);
    const auto *stage1_region =
        memory_plan.find_region(stage1_output.memory_region_id);
    const auto *stage2_region =
        memory_plan.find_region(stage2_output.memory_region_id);
    ASSERT_NE(stage0_region, nullptr);
    ASSERT_NE(stage1_region, nullptr);
    ASSERT_NE(stage2_region, nullptr);
    EXPECT_EQ(stage0_region->kind, compiler::MemoryRegionKind::TransientTensor);
    EXPECT_EQ(stage1_region->kind, compiler::MemoryRegionKind::TransientTensor);
    EXPECT_EQ(stage2_region->kind, compiler::MemoryRegionKind::TransientTensor);
    EXPECT_EQ(stage0_region->lifetime.first_stage, 0u);
    EXPECT_EQ(stage0_region->lifetime.last_stage, 1u);
    EXPECT_EQ(stage1_region->lifetime.first_stage, 1u);
    EXPECT_EQ(stage1_region->lifetime.last_stage, 2u);
    EXPECT_EQ(stage2_region->lifetime.first_stage, 2u);
    EXPECT_EQ(stage2_region->lifetime.last_stage, 2u);
    EXPECT_EQ(stage0_region->alias_group, stage2_region->alias_group);
    EXPECT_NE(stage0_region->alias_group, stage1_region->alias_group);
    EXPECT_FALSE(memory_plan.hidden_host_copies_allowed);
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       CompilerMemoryPlanClassifiesExternalAndModelOwnedInputsAcrossTargets) {
  const auto model = make_add_constant_model(2.0f);
  std::shared_ptr<const ov::Node> add;
  for (const auto &node : model->get_ordered_ops()) {
    if (ov::as_type_ptr<const ov::op::v1::Add>(node)) {
      add = node;
      break;
    }
  }
  ASSERT_TRUE(add);

  struct TargetCase {
    std::string name;
    compiler::BackendTarget target;
  };
  const std::vector<TargetCase> targets = {
      {"macos",
       compiler::BackendTarget::from_backend(GpuBackend::Metal)},
      {"android",
       compiler::BackendTarget::from_backend_device_family(
           GpuBackend::OpenCL, GpuDeviceFamily::QualcommAdreno)},
      {"rpi4",
       compiler::BackendTarget::from_backend_device_family(
           GpuBackend::OpenCL, GpuDeviceFamily::BroadcomV3D)},
      {"rpi5",
       compiler::BackendTarget::from_backend_device_family(
           GpuBackend::OpenCL, GpuDeviceFamily::BroadcomV3D)}};

  for (const auto &target : targets) {
    SCOPED_TRACE(target.name);
    compiler::LoweringPlan lowering_plan;
    lowering_plan.target = target.target;
    auto add_operation = make_metadata_planned_operation(
        add, compiler::select_tensor_layout_plan("Add", add));
    add_operation.input_element_types = {"f32", "f32"};
    add_operation.input_shapes = {"{1,3}", "{1,3}"};
    lowering_plan.operations.push_back(std::move(add_operation));

    const auto manifest = compiler::ManifestBuilder{}.build(lowering_plan);
    ASSERT_TRUE(manifest.verify().valid());
    ASSERT_EQ(manifest.stages.size(), 1u);
    ASSERT_EQ(manifest.stages.front().inputs.size(), 2u);
    const auto &parameter_input = manifest.stages.front().inputs[0];
    const auto &constant_input = manifest.stages.front().inputs[1];
    const auto *parameter_region =
        manifest.memory_plan.find_region(parameter_input.memory_region_id);
    const auto *constant_region =
        manifest.memory_plan.find_region(constant_input.memory_region_id);
    ASSERT_NE(parameter_region, nullptr);
    ASSERT_NE(constant_region, nullptr);

    EXPECT_EQ(parameter_region->kind, compiler::MemoryRegionKind::ExternalTensor);
    EXPECT_TRUE(parameter_region->external_binding);
    EXPECT_EQ(parameter_input.lifetime_class, "external");
    EXPECT_EQ(constant_region->kind, compiler::MemoryRegionKind::ImmutableTensor);
    EXPECT_FALSE(constant_region->external_binding);
    EXPECT_EQ(constant_input.lifetime_class, "model_owned");
    EXPECT_FALSE(manifest.memory_plan.hidden_host_copies_allowed);
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       CompilerMemoryPlanRejectsOverlappingTransientAliasGroups) {
  compiler::MemoryPlan plan;
  plan.schema_version = 1;

  compiler::MemoryRegion first;
  first.region_id = "stage_0.output_0";
  first.logical_tensor_name = "relu0.output0";
  first.kind = compiler::MemoryRegionKind::TransientTensor;
  first.element_type = "f32";
  first.partial_shape = "{1,2,3}";
  first.layout = "logical";
  first.storage_kind = "device_buffer";
  first.alias_group = "transient_alias_0";
  first.lifetime = {0, 2};

  compiler::MemoryRegion second = first;
  second.region_id = "stage_1.output_0";
  second.logical_tensor_name = "relu1.output0";
  second.lifetime = {1, 3};

  compiler::AliasGroup alias_group;
  alias_group.group_id = "transient_alias_0";
  alias_group.region_ids = {first.region_id, second.region_id};

  compiler::TransientArena arena;
  arena.arena_id = "transient_device_buffer_arena";
  arena.storage_kind = "device_buffer";
  arena.region_ids = alias_group.region_ids;

  plan.regions = {std::move(first), std::move(second)};
  plan.alias_groups = {std::move(alias_group)};
  plan.transient_arenas = {std::move(arena)};

  const auto verification = plan.verify();
  EXPECT_FALSE(verification.valid());
  EXPECT_TRUE(has_diagnostic_containing(
      verification.diagnostics, "overlapping transient lifetimes"));
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeExecutableDescriptorCarriesCompilerOwnedMemoryPlan) {
  for (const auto *backend_domain : {"opencl", "metal"}) {
    SCOPED_TRACE(backend_domain);
    const auto manifest = make_single_payload_route_manifest(
        LoweringRouteKind::Metadata, backend_domain, "metadata", "metadata");
    ASSERT_TRUE(manifest.valid());

    const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
    ASSERT_TRUE(executable.valid());
    ASSERT_FALSE(executable.memory_plan.regions.empty());
    ASSERT_FALSE(executable.memory_plan.alias_groups.empty());
    ASSERT_FALSE(executable.memory_plan.transient_arenas.empty());

    const auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    const auto verification = compiler::verify_runtime_executable_descriptor(
        runtime_descriptor, executable);
    ASSERT_TRUE(verification.valid())
        << (verification.diagnostics.empty()
                ? std::string{}
                : verification.diagnostics.front());

    EXPECT_EQ(runtime_descriptor.memory_plan.schema_version,
              executable.memory_plan.schema_version);
    EXPECT_EQ(runtime_descriptor.memory_plan.fingerprint,
              compiler::make_memory_plan_fingerprint(executable.memory_plan));
    EXPECT_EQ(runtime_descriptor.memory_plan.regions.size(),
              executable.memory_plan.regions.size());
    EXPECT_EQ(runtime_descriptor.memory_plan.alias_groups.size(),
              executable.memory_plan.alias_groups.size());
    EXPECT_EQ(runtime_descriptor.memory_plan.transient_arenas.size(),
              executable.memory_plan.transient_arenas.size());
    EXPECT_FALSE(runtime_descriptor.memory_plan.hidden_host_copies_allowed);
    EXPECT_TRUE(runtime_descriptor.memory_plan.has_region(
        executable.memory_plan.regions.front().region_id));
    EXPECT_TRUE(runtime_descriptor.memory_plan.has_alias_group(
        executable.memory_plan.alias_groups.front().group_id));
    EXPECT_EQ(runtime_descriptor.memory_plan.regions.front().layout,
              executable.memory_plan.regions.front().layout);
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    const auto &runtime_stage = runtime_descriptor.stages.front();
    ASSERT_EQ(runtime_stage.input_bindings.size(), 1u);
    ASSERT_EQ(runtime_stage.output_bindings.size(), 1u);
    EXPECT_EQ(
        runtime_stage.input_bindings.front().memory_region_id,
        executable.manifest.stages.front().inputs.front().memory_region_id);
    EXPECT_EQ(
        runtime_stage.output_bindings.front().memory_region_id,
        executable.manifest.stages.front().outputs.front().memory_region_id);
    EXPECT_EQ(runtime_stage.input_bindings.front().alias_group,
              executable.memory_plan.regions.front().alias_group);

    std::vector<GpuTensor *> input_slots(runtime_stage.input_bindings.size(),
                                         nullptr);
    std::vector<GpuTensor *> output_slots(runtime_stage.output_bindings.size(),
                                          nullptr);
    const auto binding_table = ResourceBindingTable::for_stage(
        input_slots, output_slots, runtime_stage);
    EXPECT_TRUE(binding_table.compatible_with(runtime_stage));
    EXPECT_EQ(binding_table.input_region_ids().front(),
              runtime_stage.input_bindings.front().memory_region_id);
    EXPECT_EQ(binding_table.output_region_ids().front(),
              runtime_stage.output_bindings.front().memory_region_id);

    output_slots.clear();
    const auto incomplete_binding_table = ResourceBindingTable::for_stage(
        input_slots, output_slots, runtime_stage);
    EXPECT_FALSE(incomplete_binding_table.compatible_with(runtime_stage));

    auto stale_region_descriptor = runtime_descriptor;
    stale_region_descriptor.memory_plan.regions.front().layout =
        "stale_runtime_layout";
    const auto stale_region_verification =
        compiler::verify_runtime_executable_descriptor(stale_region_descriptor,
                                                       executable);
    EXPECT_FALSE(stale_region_verification.valid());
    EXPECT_TRUE(has_diagnostic_containing(stale_region_verification.diagnostics,
                                          "memory region drift"));

    auto stale_fingerprint_descriptor = runtime_descriptor;
    stale_fingerprint_descriptor.memory_plan.fingerprint =
        "stale-runtime-memory-plan";
    const auto stale_fingerprint_verification =
        compiler::verify_runtime_executable_descriptor(
            stale_fingerprint_descriptor, executable);
    EXPECT_FALSE(stale_fingerprint_verification.valid());
    EXPECT_TRUE(
        has_diagnostic_containing(stale_fingerprint_verification.diagnostics,
                                  "memory plan fingerprint drift"));

    auto stale_binding_descriptor = runtime_descriptor;
    stale_binding_descriptor.stages.front()
        .input_bindings.front()
        .memory_region_id = "stale-runtime-input-region";
    const auto stale_binding_verification =
        compiler::verify_runtime_executable_descriptor(stale_binding_descriptor,
                                                       executable);
    EXPECT_FALSE(stale_binding_verification.valid());
    EXPECT_TRUE(has_diagnostic_containing(
        stale_binding_verification.diagnostics, "input binding drift"));
  }
}


TEST_F(GfxBackendArchitectureContractTest,
       CacheEnvelopeContainsDocsRequiredCompilerOwnedIdentity) {
  const auto manifest = make_single_payload_route_manifest(
      LoweringRouteKind::Metadata, "opencl", "metadata", "metadata");
  const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
  ASSERT_TRUE(executable.verify().valid());

  const auto model = models.relu();
  const auto runtime_descriptor =
      make_finalized_cache_test_runtime_descriptor(executable);
  const auto envelope = compiler::CacheEnvelopeBuilder{}.build(
      executable, runtime_descriptor, make_test_cache_options(*model));

  const auto verification = envelope.verify(executable);
  EXPECT_TRUE(verification.valid())
      << (verification.diagnostics.empty() ? std::string{}
                                           : verification.diagnostics.front());
  EXPECT_FALSE(envelope.key.model_fingerprint.empty());
  EXPECT_EQ(envelope.key.manifest_hash,
            compiler::make_manifest_cache_hash(manifest));
  EXPECT_EQ(envelope.key.target_fingerprint, manifest.target_fingerprint);
  EXPECT_EQ(envelope.key.compile_options_hash,
            compiler::make_executable_compile_options_hash(executable));
  EXPECT_FALSE(envelope.key.backend_capabilities_fingerprint.empty());
  EXPECT_FALSE(envelope.key.compiler_revision.empty());
  EXPECT_FALSE(envelope.key.backend_compiler_revision.empty());
  EXPECT_FALSE(envelope.key.driver_identity.empty());
  EXPECT_FALSE(envelope.key.kernel_unit_versions.empty());
  EXPECT_FALSE(envelope.key.stable_key.empty());
  EXPECT_EQ(envelope.manifest.target_fingerprint, manifest.target_fingerprint);
  EXPECT_EQ(envelope.artifact_descriptors.size(),
            executable.artifact_descriptors.size());
}

TEST_F(GfxBackendArchitectureContractTest,
       ModelCacheFingerprintIncludesAttributesAndConstantPayloads) {
  const auto manifest = make_single_payload_route_manifest(
      LoweringRouteKind::Metadata, "opencl", "metadata", "metadata");
  const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
  ASSERT_TRUE(executable.verify().valid());

  const auto add_one = make_add_constant_model(1.0f);
  const auto add_two = make_add_constant_model(2.0f);
  const auto add_one_fingerprint =
      compiler::make_model_cache_fingerprint(*add_one);
  const auto add_two_fingerprint =
      compiler::make_model_cache_fingerprint(*add_two);
  EXPECT_NE(add_one_fingerprint, add_two_fingerprint);

  const auto reshape_plain = make_reshape_model(false);
  const auto reshape_special_zero = make_reshape_model(true);
  EXPECT_NE(compiler::make_model_cache_fingerprint(*reshape_plain),
            compiler::make_model_cache_fingerprint(*reshape_special_zero));

  const auto runtime_descriptor =
      make_finalized_cache_test_runtime_descriptor(executable);
  const auto envelope_one = compiler::CacheEnvelopeBuilder{}.build(
      executable, runtime_descriptor, make_test_cache_options(*add_one));
  const auto envelope_two = compiler::CacheEnvelopeBuilder{}.build(
      executable, runtime_descriptor, make_test_cache_options(*add_two));
  EXPECT_NE(envelope_one.key.model_fingerprint,
            envelope_two.key.model_fingerprint);
  EXPECT_NE(envelope_one.key.stable_key, envelope_two.key.stable_key);
}

TEST_F(GfxBackendArchitectureContractTest,
       CacheEnvelopeRejectsStaleManifestDriverAndKernelIdentity) {
  const auto manifest = make_single_payload_route_manifest(
      LoweringRouteKind::Metadata, "metal", "metadata", "metadata");
  const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
  ASSERT_TRUE(executable.verify().valid());

  const auto model = models.relu();
  const auto runtime_descriptor =
      make_finalized_cache_test_runtime_descriptor(executable);
  const auto envelope = compiler::CacheEnvelopeBuilder{}.build(
      executable, runtime_descriptor, make_test_cache_options(*model));
  ASSERT_TRUE(envelope.verify(executable).valid());

  auto stale_manifest = envelope;
  stale_manifest.key.manifest_hash = "stale-manifest";
  EXPECT_FALSE(stale_manifest.verify(executable).valid());
  EXPECT_TRUE(has_diagnostic_containing(
      stale_manifest.verify(executable).diagnostics, "manifest hash drift"));

  auto missing_driver = envelope;
  missing_driver.key.driver_identity.clear();
  EXPECT_FALSE(missing_driver.verify(executable).valid());
  EXPECT_TRUE(has_diagnostic_containing(
      missing_driver.verify(executable).diagnostics, "driver identity"));

  auto stale_kernel_unit = envelope;
  stale_kernel_unit.key.kernel_unit_versions.front() = "stale-kernel-unit";
  EXPECT_FALSE(stale_kernel_unit.verify(executable).valid());
  EXPECT_TRUE(has_diagnostic_containing(
      stale_kernel_unit.verify(executable).diagnostics,
      "kernel unit versions drift"));
}

TEST_F(GfxBackendArchitectureContractTest,
       UnknownBackendCannotMaterializeTargetOrStageFactory) {
  EXPECT_ANY_THROW(compiler::BackendTarget::from_backend(GpuBackend::Unknown));
  EXPECT_ANY_THROW(GpuStageFactory::factory_for_backend(GpuBackend::Unknown));
  EXPECT_ANY_THROW(memory_ops_for_backend(GpuBackend::Unknown));
}

} // namespace
} // namespace gfx_plugin
} // namespace ov
