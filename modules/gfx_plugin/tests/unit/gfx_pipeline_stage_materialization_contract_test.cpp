// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unit/gfx_manifest_executable_contract_utils.hpp"
#include "unit/gfx_runtime_materialization_contract_utils.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

TEST_F(GfxBackendArchitectureContractTest,
       PipelineStagePlanBuilderMarksOutputsAndDeduplicatesAliases) {
  auto parameter =
      std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
  auto relu = std::make_shared<ov::op::v0::Relu>(parameter);
  auto result = std::make_shared<ov::op::v0::Result>(relu);
  ov::Model model(ov::ResultVector{result}, ov::ParameterVector{parameter},
                  "pipeline_stage_plan_contract");

  const auto model_outputs = compiler::collect_model_output_ports(model);
  compiler::PipelineStagePlanBuilder builder(model_outputs);
  auto plan = builder.make_stage_plan(relu, 7);

  ASSERT_EQ(plan.outputs.size(), 1u);
  EXPECT_EQ(plan.runtime_stage_index, 7u);
  EXPECT_TRUE(plan.outputs[0].is_model_output);
  EXPECT_EQ(plan.outputs[0].source_node.get(), relu.get());
  EXPECT_EQ(plan.outputs[0].source_port, 0u);

  builder.append_output_alias(plan, relu, 0, 0);
  builder.append_output_alias(plan, relu, 0, 0);
  ASSERT_EQ(plan.output_aliases.size(), 1u);
  EXPECT_EQ(plan.output_aliases[0].node.get(), relu.get());

  compiler::PipelineOutputAliasMap aliases;
  builder.record_output_alias(aliases, relu.get(), 0, 3);
  const auto remapped = builder.remap_input_link(aliases, relu, 0);
  EXPECT_EQ(remapped.node.get(), relu.get());
  EXPECT_EQ(remapped.port, 3u);

  auto fused = builder.make_fused_stage_plan(relu, 2, 9);
  builder.describe_output(fused, 1, relu, 0);
  ASSERT_EQ(fused.outputs.size(), 2u);
  EXPECT_EQ(fused.runtime_stage_index, 9u);
  EXPECT_TRUE(fused.outputs[1].is_model_output);
  EXPECT_EQ(fused.outputs[1].source_node.get(), relu.get());
}

TEST_F(GfxBackendArchitectureContractTest,
       PipelineStagePlanBuilderOnlyDirectAliasesStatefulAssignWhenExclusive) {
  auto parameter =
      std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
  auto variable = std::make_shared<ov::op::util::Variable>(
      ov::op::util::VariableInfo{ov::Shape{1}, ov::element::f32, "variable0"});
  auto add = std::make_shared<ov::op::v1::Add>(parameter, parameter);
  auto assign = std::make_shared<ov::op::v6::Assign>(add, variable);
  auto unrelated_result = std::make_shared<ov::op::v0::Result>(parameter);
  ov::Model assign_only_model(
      ov::ResultVector{unrelated_result}, ov::SinkVector{assign},
      ov::ParameterVector{parameter}, "stateful_assign_direct_alias_safe");

  const auto assign_only_outputs =
      compiler::collect_model_output_ports(assign_only_model);
  compiler::PipelineStagePlanBuilder assign_only_builder(assign_only_outputs);
  const auto assign_only_plan = assign_only_builder.make_stage_plan(add, 0);
  ASSERT_EQ(assign_only_plan.outputs.size(), 1u);
  EXPECT_EQ(assign_only_plan.outputs[0].direct_stateful_assign_variable_id,
            "variable0");

  auto shared_parameter =
      std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
  auto shared_variable = std::make_shared<ov::op::util::Variable>(
      ov::op::util::VariableInfo{ov::Shape{1}, ov::element::f32, "variable0"});
  auto shared_add =
      std::make_shared<ov::op::v1::Add>(shared_parameter, shared_parameter);
  auto shared_assign =
      std::make_shared<ov::op::v6::Assign>(shared_add, shared_variable);
  auto shared_result = std::make_shared<ov::op::v0::Result>(shared_add);
  ov::Model shared_consumer_model(ov::ResultVector{shared_result},
                                  ov::SinkVector{shared_assign},
                                  ov::ParameterVector{shared_parameter},
                                  "stateful_assign_direct_alias_unsafe");

  const auto shared_outputs =
      compiler::collect_model_output_ports(shared_consumer_model);
  compiler::PipelineStagePlanBuilder shared_builder(shared_outputs);
  const auto shared_plan = shared_builder.make_stage_plan(shared_add, 0);
  ASSERT_EQ(shared_plan.outputs.size(), 1u);
  EXPECT_TRUE(
      shared_plan.outputs[0].direct_stateful_assign_variable_id.empty());
}

TEST_F(GfxBackendArchitectureContractTest,
       ManifestCarriesDirectStatefulAssignPrebindShapeContract) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f32, ov::PartialShape{ov::Dimension::dynamic(), 2});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f32, ov::PartialShape{ov::Dimension::dynamic(), 2});
  auto concat = std::make_shared<ov::op::v0::Concat>(
      ov::OutputVector{lhs->output(0), rhs->output(0)}, 0);
  auto variable = std::make_shared<ov::op::util::Variable>(
      ov::op::util::VariableInfo{ov::PartialShape{ov::Dimension::dynamic(), 2},
                                 ov::element::f32, "variable0"});
  auto assign = std::make_shared<ov::op::v6::Assign>(concat, variable);
  ASSERT_NE(assign, nullptr);
  ASSERT_EQ(concat->output(0).get_target_inputs().size(), 1u);

  compiler::PlannedOperation op;
  op.source_node = concat;
  op.node_name = concat->get_friendly_name();
  op.type_name = concat->get_type_name();
  op.kernel_unit = compiler::KernelUnit::describe(
      LoweringRouteKind::Metadata, KernelUnitKind::Metadata, "metadata",
      "opencl", "metadata", true);
  op.layout = compiler::select_tensor_layout_plan("Concat", concat);
  op.profitability_score = 1.0;
  op.input_element_types = {"f32", "f32"};
  op.input_shapes = {"{?,2}", "{?,2}"};
  op.output_element_types = {"f32"};
  op.output_shapes = {"{?,2}"};

  compiler::LoweringPlan plan;
  plan.target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  plan.operations.push_back(std::move(op));

  const auto manifest = compiler::ManifestBuilder{}.build(plan);
  ASSERT_TRUE(manifest.valid());
  ASSERT_EQ(manifest.stages.size(), 1u);
  ASSERT_EQ(manifest.stages.front().outputs.size(), 1u);
  const auto &output = manifest.stages.front().outputs.front();
  EXPECT_EQ(output.stateful_prebind_variable_id, "variable0");
  EXPECT_EQ(output.stateful_prebind_shape_rule, "sum_inputs_along_axis");
  EXPECT_EQ(output.stateful_prebind_shape_axis, 0);

  const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
  ASSERT_TRUE(executable.valid());
  const auto runtime_descriptor =
      compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
  ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
  ASSERT_EQ(runtime_descriptor.stages.front().output_bindings.size(), 1u);
  const auto &runtime_output =
      runtime_descriptor.stages.front().output_bindings.front();
  EXPECT_EQ(runtime_output.stateful_prebind_variable_id, "variable0");
  EXPECT_EQ(runtime_output.stateful_prebind_shape_rule,
            "sum_inputs_along_axis");
  EXPECT_EQ(runtime_output.stateful_prebind_shape_axis, 0);
  EXPECT_TRUE(compiler::runtime_executable_descriptor_valid(runtime_descriptor,
                                                            executable));

  auto stale_runtime_descriptor = runtime_descriptor;
  stale_runtime_descriptor.stages.front()
      .output_bindings.front()
      .stateful_prebind_shape_rule = "stale_runtime_rule";
  const auto stale_verification =
      compiler::verify_runtime_executable_descriptor(stale_runtime_descriptor,
                                                     executable);
  EXPECT_FALSE(stale_verification.valid());
  EXPECT_TRUE(has_diagnostic_containing(stale_verification.diagnostics,
                                        "output binding drift"));
}

TEST_F(GfxBackendArchitectureContractTest,
       SharedStatefulAndViewOnlyStagesMaterializeFromDescriptorOnly) {
  RuntimeStageExecutableDescriptor assign_descriptor;
  assign_descriptor.manifest_ref = "manifest://unit/assign";
  assign_descriptor.kernel_id = "metadata";
  assign_descriptor.op_family = "Assign";
  assign_descriptor.stage_name = "unit_assign";
  assign_descriptor.stateful_effect = "assign";
  assign_descriptor.output_bindings.push_back(
      make_runtime_binding("assign.output0", "assign_output", "TensorOutput"));

  const auto assign_stage = create_stateful_stage(assign_descriptor);
  ASSERT_NE(assign_stage, nullptr);
  EXPECT_EQ(assign_stage->type(), std::string("Assign"));
  EXPECT_EQ(assign_stage->name(), std::string("assign.output0"));

  RuntimeStageExecutableDescriptor read_descriptor;
  read_descriptor.manifest_ref = "manifest://unit/read_value";
  read_descriptor.kernel_id = "metadata";
  read_descriptor.op_family = "ReadValue";
  read_descriptor.stage_name = "unit_read_value";
  read_descriptor.stateful_effect = "read_value";
  read_descriptor.output_bindings.push_back(make_runtime_binding(
      "read_value.output0", "read_value_output", "TensorOutput"));

  const auto read_stage = create_stateful_stage(read_descriptor);
  ASSERT_NE(read_stage, nullptr);
  EXPECT_EQ(read_stage->type(), std::string("ReadValue"));
  EXPECT_EQ(read_stage->name(), std::string("read_value.output0"));

  RuntimeStageExecutableDescriptor view_descriptor;
  view_descriptor.manifest_ref = "manifest://unit/view_only";
  view_descriptor.kernel_id = "metadata";
  view_descriptor.op_family = "Reshape";
  view_descriptor.stage_name = "unit_view_only";
  view_descriptor.origin = KernelArtifactOrigin::Metadata;
  view_descriptor.payload_kind = KernelArtifactPayloadKind::None;
  view_descriptor.tensor_view_only = true;
  view_descriptor.layout_contract = "view_only";
  view_descriptor.output_bindings.push_back(make_runtime_binding(
      "reshape.output0", "reshape_output", "TensorOutput"));
  view_descriptor.output_bindings.front().partial_shape = "{2,3}";
  view_descriptor.output_bindings.front().element_type = "f32";

  auto view_stage = create_view_only_stage(view_descriptor);
  ASSERT_NE(view_stage, nullptr);
  EXPECT_EQ(view_stage->type(), std::string("Reshape"));
  EXPECT_EQ(view_stage->name(), std::string("reshape.output0"));

  GpuTensor input;
  input.buf.buffer = reinterpret_cast<GpuBufferHandle>(0x1234);
  input.buf.size = 24;
  input.buf.allocation_uid = 77;
  input.shape = ov::Shape{6};
  input.expected_type = ov::element::f32;
  GpuTensor output;

  view_stage->set_inputs({&input});
  view_stage->set_output(&output);
  view_stage->execute(nullptr);

  EXPECT_TRUE(same_gpu_allocation(input.buf, output.buf));
  EXPECT_FALSE(output.buf.owned);
  EXPECT_TRUE(output.buf.external);
  EXPECT_EQ(output.shape, (ov::Shape{2, 3}));
  EXPECT_EQ(output.expected_type, ov::element::f32);
}

TEST_F(GfxBackendArchitectureContractTest,
       PipelineStageMaterializerMaterializesDescriptorOnly) {
  auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                           ov::Shape{2, 3});
  auto shape =
      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {6});
  auto reshape = std::make_shared<ov::op::v1::Reshape>(parameter, shape, false);
  auto relu = std::make_shared<ov::op::v0::Relu>(parameter);

  auto view_descriptor = make_materializer_base_descriptor(reshape);
  view_descriptor.origin = KernelArtifactOrigin::Metadata;
  view_descriptor.payload_kind = KernelArtifactPayloadKind::None;
  view_descriptor.kernel_id = "metadata";
  view_descriptor.entry_point = "metadata";
  view_descriptor.layout_contract = "view_only";
  view_descriptor.tensor_view_only = true;
  view_descriptor.input_bindings = {make_runtime_binding(
      "reshape.input0", "compiler_view_input_region", "TensorInput")};
  view_descriptor.output_bindings = {make_runtime_binding(
      "reshape.output0", "compiler_view_output_region", "TensorOutput")};
  view_descriptor.output_bindings.front().partial_shape = "{6}";
  view_descriptor.abi_arg_count = 1;
  view_descriptor.abi_output_arg_count = 1;

  auto payload_descriptor = make_materializer_base_descriptor(relu);
  payload_descriptor.stage_index = 1;
  payload_descriptor.stage_record_key = 0x2234u;
  payload_descriptor.manifest_ref = "manifest://unit/relu_payload";
  payload_descriptor.abi_fingerprint = "abi://unit/relu_payload";
  payload_descriptor.artifact_key = "artifact://unit/relu_payload";

  RuntimeExecutableDescriptor runtime_descriptor;
  runtime_descriptor.target_fingerprint = "metal:unit";
  runtime_descriptor.stages = {view_descriptor, payload_descriptor};
  runtime_descriptor.materialization_finalized = true;

  auto view_plan = make_single_materialization_plan(reshape, view_descriptor);
  view_plan.io_plan.outputs.front().shape = ov::Shape{6};
  runtime_descriptor.materialization_stages.push_back(std::move(view_plan));
  runtime_descriptor.materialization_stages.push_back(
      make_single_materialization_plan(relu, payload_descriptor));

  CapturingBackendStageFactory stage_factory;
  PipelineStageRuntimeMaterializationRequest request;
  request.stage_factory = &stage_factory;
  request.runtime_descriptor = &runtime_descriptor;

  auto pipeline = materialize_pipeline_stage_descriptors(request);
  ASSERT_EQ(pipeline.size(), 2u);
  ASSERT_EQ(stage_factory.stage_names.size(), 2u);
  EXPECT_EQ(stage_factory.stage_names[0], reshape->get_friendly_name());
  EXPECT_EQ(stage_factory.stage_names[1], relu->get_friendly_name());
}

TEST_F(GfxBackendArchitectureContractTest,
       PipelineStageMaterializerProjectsDescriptorOutputContracts) {
  auto parameter =
      std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{8});
  auto relu = std::make_shared<ov::op::v0::Relu>(parameter);

  auto descriptor = make_materializer_base_descriptor(relu);
  descriptor.origin = KernelArtifactOrigin::Metadata;
  descriptor.payload_kind = KernelArtifactPayloadKind::None;
  descriptor.kernel_id = "metadata";
  descriptor.entry_point = "metadata";
  descriptor.output_bindings.front().element_type = "i32";
  descriptor.output_bindings.front().partial_shape = "{4,2}";

  PipelineStageMaterializationPlan plan;
  plan.kind = PipelineStageMaterializationKind::SingleStage;
  plan.io_plan.stage_name = relu->get_friendly_name();
  plan.io_plan.op_family = relu->get_type_name();
  plan.io_plan.runtime_stage_index = descriptor.stage_index;
  plan.descriptor_stage_index = descriptor.stage_index;

  PipelineStageOutputDesc output;
  output.type = ov::element::dynamic;
  output.source_port = 0;
  output.source_ref.kind = PipelineStageTensorRefKind::StageOutput;
  output.source_ref.index = descriptor.stage_index;
  output.source_ref.port = 0;
  plan.io_plan.outputs.push_back(std::move(output));
  plan.materialized_descriptor = descriptor;
  plan.materialized_descriptor_valid = true;

  RuntimeExecutableDescriptor runtime_descriptor;
  runtime_descriptor.target_fingerprint = "metal:unit";
  runtime_descriptor.stages = {descriptor};
  runtime_descriptor.materialization_finalized = true;

  runtime_descriptor.materialization_stages.push_back(std::move(plan));

  CapturingBackendStageFactory stage_factory;
  PipelineStageRuntimeMaterializationRequest request;
  request.stage_factory = &stage_factory;
  request.runtime_descriptor = &runtime_descriptor;

  auto pipeline = materialize_pipeline_stage_descriptors(request);
  ASSERT_EQ(pipeline.size(), 1u);
  ASSERT_EQ(pipeline.front().outputs.size(), 1u);
  EXPECT_EQ(pipeline.front().outputs.front().type, ov::element::i32);
  EXPECT_EQ(pipeline.front().outputs.front().shape, (ov::Shape{4, 2}));
  ASSERT_EQ(stage_factory.stage_names.size(), 1u);
  EXPECT_EQ(stage_factory.stage_names.front(), relu->get_friendly_name());
}

TEST_F(GfxBackendArchitectureContractTest,
       PipelineStageMaterializerPreservesCompilerOwnedMaterializedBindings) {
  auto parameter =
      std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
  auto relu = std::make_shared<ov::op::v0::Relu>(parameter);

  RuntimeExecutableDescriptor runtime_descriptor;
  runtime_descriptor.target_fingerprint = "metal:unit";
  runtime_descriptor.stages.push_back(make_materializer_base_descriptor(relu));

  auto vendor_descriptor = runtime_descriptor.stages.front();
  vendor_descriptor.input_bindings = {make_runtime_binding(
      "vendor.input0", "compiler_vendor_input_region", "TensorInput")};
  vendor_descriptor.output_bindings = {make_runtime_binding(
      "vendor.output0", "compiler_vendor_output_region", "TensorOutput")};
  vendor_descriptor.abi_arg_count = 1;
  vendor_descriptor.abi_output_arg_count = 1;

  const auto plan =
      make_vendor_materialization_plan(parameter, relu, vendor_descriptor);
  UnitBackendStageFactory stage_factory;
  PipelineStageMaterializer materializer(stage_factory, runtime_descriptor, {});

  const auto materialized = materializer.create_materialized_descriptor(plan);
  ASSERT_TRUE(materialized);
  ASSERT_EQ(materialized->input_bindings.size(), 1u);
  ASSERT_EQ(materialized->output_bindings.size(), 1u);
  EXPECT_EQ(materialized->input_bindings.front().memory_region_id,
            "compiler_vendor_input_region");
  EXPECT_EQ(materialized->output_bindings.front().memory_region_id,
            "compiler_vendor_output_region");
}

TEST_F(GfxBackendArchitectureContractTest,
       PipelineStageMaterializerRejectsIncompleteMaterializedBindings) {
  auto parameter =
      std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
  auto relu = std::make_shared<ov::op::v0::Relu>(parameter);

  RuntimeExecutableDescriptor runtime_descriptor;
  runtime_descriptor.target_fingerprint = "metal:unit";
  runtime_descriptor.stages.push_back(make_materializer_base_descriptor(relu));

  auto incomplete_vendor_descriptor = runtime_descriptor.stages.front();
  incomplete_vendor_descriptor.input_bindings.clear();
  incomplete_vendor_descriptor.output_bindings.clear();
  incomplete_vendor_descriptor.abi_arg_count = 0;
  incomplete_vendor_descriptor.abi_output_arg_count = 0;

  const auto plan = make_vendor_materialization_plan(
      parameter, relu, incomplete_vendor_descriptor);
  UnitBackendStageFactory stage_factory;
  PipelineStageMaterializer materializer(stage_factory, runtime_descriptor, {});

  EXPECT_THROW((void)materializer.create_materialized_descriptor(plan),
               ov::Exception);
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeExecutableDescriptorRejectsIncompleteMaterialization) {
  const auto manifest = make_single_payload_route_manifest(
      LoweringRouteKind::Metadata, "opencl", "metadata", "metadata");
  const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
  ASSERT_TRUE(executable.valid());

  auto runtime_descriptor =
      compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
  ASSERT_TRUE(compiler::runtime_executable_descriptor_valid(runtime_descriptor,
                                                            executable));

  auto parameter =
      std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
  auto relu = std::make_shared<ov::op::v0::Relu>(parameter);
  PipelineStageMaterializationPlan materialized_stage;
  materialized_stage.kind = PipelineStageMaterializationKind::SingleStage;
  materialized_stage.io_plan.stage_name = relu->get_friendly_name();
  materialized_stage.io_plan.op_family = relu->get_type_name();
  materialized_stage.io_plan.runtime_stage_index = 0;
  materialized_stage.descriptor_stage_index = 0;
  runtime_descriptor.materialization_finalized = true;
  runtime_descriptor.materialization_stages.push_back(
      std::move(materialized_stage));
  const auto unfrozen_verification =
      compiler::verify_runtime_executable_descriptor_materialization(
          runtime_descriptor);
  EXPECT_FALSE(unfrozen_verification.valid());
  EXPECT_TRUE(has_diagnostic_containing(unfrozen_verification.diagnostics,
                                        "materialized descriptor missing"));
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeExecutableDescriptorRejectsNonMaterializableFusedChild) {
  auto parameter =
      std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
  auto relu0 = std::make_shared<ov::op::v0::Relu>(parameter);
  auto relu1 = std::make_shared<ov::op::v0::Relu>(relu0);
  auto relu2 = std::make_shared<ov::op::v0::Relu>(relu1);

  auto child0 = make_materializer_base_descriptor(relu0);
  child0.origin = KernelArtifactOrigin::Metadata;
  child0.payload_kind = KernelArtifactPayloadKind::None;
  child0.kernel_id = "metadata";
  child0.entry_point = "metadata";
  child0.layout_contract = "view_only";
  child0.tensor_view_only = true;

  auto child1 = make_materializer_base_descriptor(relu1);
  child1.stage_index = 1;
  child1.stage_record_key = 0x2234u;
  child1.manifest_ref = "manifest://unit/fused_child1";
  child1.abi_fingerprint = "abi://unit/fused_child1";
  child1.artifact_key = "artifact://unit/fused_child1";
  child1.origin = KernelArtifactOrigin::Common;
  child1.payload_kind = KernelArtifactPayloadKind::None;
  child1.kernel_id = "materialized/fused_attention_sequence/child1";
  child1.entry_point.clear();
  child1.tensor_view_only = false;

  auto child2 = make_materializer_base_descriptor(relu2);
  child2.stage_index = 2;
  child2.stage_record_key = 0x3234u;
  child2.manifest_ref = "manifest://unit/fused_child2";
  child2.abi_fingerprint = "abi://unit/fused_child2";
  child2.artifact_key = "artifact://unit/fused_child2";
  child2.origin = KernelArtifactOrigin::Metadata;
  child2.payload_kind = KernelArtifactPayloadKind::None;
  child2.kernel_id = "metadata";
  child2.entry_point = "metadata";
  child2.layout_contract = "view_only";
  child2.tensor_view_only = true;

  PipelineStageMaterializationPlan plan;
  plan.kind = PipelineStageMaterializationKind::FusedAttentionSequence;
  plan.io_plan.stage_name = relu2->get_friendly_name();
  plan.io_plan.op_family = relu2->get_type_name();
  plan.io_plan.runtime_stage_index = child2.stage_index;
  PipelineStageInputLink input;
  input.port = 0;
  input.source_ref.kind = PipelineStageTensorRefKind::Parameter;
  input.source_ref.index = 0;
  input.source_ref.port = 0;
  plan.io_plan.inputs.push_back(std::move(input));
  PipelineStageOutputDesc output;
  output.shape = ov::Shape{1};
  output.type = ov::element::f32;
  output.source_port = 0;
  output.source_ref.kind = PipelineStageTensorRefKind::StageOutput;
  output.source_ref.index = child2.stage_index;
  output.source_ref.port = 0;
  plan.io_plan.outputs.push_back(std::move(output));
  plan.descriptor_stage_index = child2.stage_index;
  plan.fused_descriptor_stage_indices = {child0.stage_index, child1.stage_index,
                                         child2.stage_index};
  plan.fused_inner_stages.resize(plan.fused_descriptor_stage_indices.size());
  plan.materialized_descriptor = child2;
  plan.materialized_descriptor.origin = KernelArtifactOrigin::Common;
  plan.materialized_descriptor.payload_kind = KernelArtifactPayloadKind::None;
  plan.materialized_descriptor.payload.reset();
  plan.materialized_descriptor.kernel_id =
      "materialized/fused_attention_sequence/unit";
  plan.materialized_descriptor.op_family = "FusedAttentionSequence";
  plan.materialized_descriptor_valid = true;

  RuntimeExecutableDescriptor runtime_descriptor;
  runtime_descriptor.target_fingerprint = "metal:unit";
  runtime_descriptor.stages = {child0, child1, child2};
  runtime_descriptor.materialization_finalized = true;
  runtime_descriptor.materialization_stages.push_back(std::move(plan));

  const auto verification =
      compiler::verify_runtime_executable_descriptor_materialization(
          runtime_descriptor);
  EXPECT_FALSE(verification.valid());
  EXPECT_TRUE(has_diagnostic_containing(
      verification.diagnostics,
      "fused child descriptor is not materializable"));
}

} // namespace
} // namespace gfx_plugin
} // namespace ov
