// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unit/gfx_manifest_executable_contract_utils.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeSliceShapeArgsComeFromKernelUnitManifestAndDescriptor) {
  struct Case {
    GpuBackend backend;
    bool requires_runtime_shape_args;
  };

  for (const auto test_case :
       {Case{GpuBackend::OpenCL, true}, Case{GpuBackend::Metal, false}}) {
    SCOPED_TRACE(backend_to_string(test_case.backend));
    compiler::LoweringPlan plan;
    plan.target = compiler::BackendTarget::from_backend(test_case.backend);
    compiler::PlannedOperation op;
    op.node_name = "Slice";
    op.type_name = "Slice";
    op.kernel_unit = compiler::KernelUnit::describe(
        LoweringRouteKind::Metadata, KernelUnitKind::Metadata, "metadata",
        plan.target.backend_id(), "Slice",
        test_case.requires_runtime_shape_args);
    op.layout = compiler::TensorLayoutPlan{};
    op.profitability_score = 1.0;
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                         ov::Shape{1, 2, 3});
    auto starts =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto ends =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    auto steps =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto axes =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto slice =
        std::make_shared<ov::op::v8::Slice>(input, starts, ends, steps, axes);
    op.source_node = slice;
    op.node_name = slice->get_friendly_name();
    op.type_name = slice->get_type_name();
    op.input_element_types = {"f32", "i64", "i64", "i64", "i64"};
    op.input_shapes = {"{1,2,3}", "{1}", "{1}", "{1}", "{1}"};
    op.output_element_types = {"f32"};
    op.output_shapes = {"{1,2,3}"};
    plan.operations.push_back(std::move(op));

    const auto manifest = compiler::ManifestBuilder{}.build(plan);
    ASSERT_TRUE(manifest.valid());
    ASSERT_EQ(manifest.stages.size(), 1u);
    EXPECT_EQ(manifest.stages.front().requires_runtime_shape_args,
              test_case.requires_runtime_shape_args);

    const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
    ASSERT_TRUE(executable.valid());
    ASSERT_EQ(executable.artifact_descriptors.size(), 1u);
    EXPECT_EQ(executable.artifact_descriptors.front()
                  .kernel.requires_runtime_shape_args,
              test_case.requires_runtime_shape_args);

    const auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_TRUE(compiler::runtime_executable_descriptor_valid(
        runtime_descriptor, executable));
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    EXPECT_EQ(runtime_descriptor.stages.front().requires_runtime_shape_args,
              test_case.requires_runtime_shape_args);

    auto stale_descriptor = runtime_descriptor;
    stale_descriptor.stages.front().requires_runtime_shape_args =
        !test_case.requires_runtime_shape_args;
    const auto stale_result = compiler::verify_runtime_executable_descriptor(
        stale_descriptor, executable);
    EXPECT_FALSE(stale_result.valid());
    EXPECT_TRUE(
        has_diagnostic_containing(stale_result.diagnostics, "artifact drift"));
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeShapeRuleComesFromCompilerManifestAndDescriptor) {
  struct Case {
    const char *op_type;
    const char *expected_rule;
    std::vector<int64_t> expected_i64_metadata;
  };

  const std::vector<int64_t> slice_metadata = {1, 1, 5};
  const std::vector<int64_t> strided_slice_metadata = {
      1, 2, 4, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0};

  for (const auto test_case :
       {Case{"Concat", "concat", {1}}, Case{"Broadcast", "broadcast", {0, 2}},
        Case{"Select", "select", {}}, Case{"ShapeOf", "shape_of", {}},
        Case{"Slice", "slice", slice_metadata},
        Case{"StridedSlice", "slice", strided_slice_metadata},
        Case{"Range", "range", {}}, Case{"Tile", "tile", {}},
        Case{"Relu", "static_or_descriptor", {}}}) {
    SCOPED_TRACE(test_case.op_type);
    compiler::LoweringPlan plan;
    plan.target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
    compiler::PlannedOperation op;
    op.node_name = test_case.op_type;
    op.type_name = test_case.op_type;
    op.kernel_unit = compiler::KernelUnit::describe(
        LoweringRouteKind::Metadata, KernelUnitKind::Metadata, "metadata",
        plan.target.backend_id(), test_case.op_type);
    op.layout = compiler::TensorLayoutPlan{};
    op.profitability_score = 1.0;
    op.input_element_types = {"f32"};
    op.input_shapes = {"{1,2,3}"};
    op.output_element_types = {"f32"};
    op.output_shapes = {"{1,2,3}"};
    if (std::string_view(test_case.op_type) == "Concat") {
      auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                         ov::Shape{1, 2, 3});
      auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                         ov::Shape{1, 2, 3});
      auto concat =
          std::make_shared<ov::op::v0::Concat>(ov::OutputVector{lhs, rhs}, 1);
      op.source_node = concat;
      op.node_name = concat->get_friendly_name();
      op.type_name = concat->get_type_name();
      op.input_element_types = {"f32", "f32"};
      op.input_shapes = {"{1,2,3}", "{1,2,3}"};
      op.output_shapes = {"{1,4,3}"};
    } else if (std::string_view(test_case.op_type) == "Broadcast") {
      auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                           ov::Shape{1, 3});
      auto target =
          ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3});
      auto broadcast = std::make_shared<ov::op::v3::Broadcast>(input, target);
      op.source_node = broadcast;
      op.node_name = broadcast->get_friendly_name();
      op.type_name = broadcast->get_type_name();
      op.input_element_types = {"f32", "i64"};
      op.input_shapes = {"{1,3}", "{2}"};
      op.output_shapes = {"{2,3}"};
    } else if (std::string_view(test_case.op_type) == "Slice") {
      auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                           ov::Shape{1, 2, 3});
      auto starts =
          ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
      auto ends =
          ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
      auto steps =
          ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
      auto axes =
          ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
      auto slice =
          std::make_shared<ov::op::v8::Slice>(input, starts, ends, steps, axes);
      op.source_node = slice;
      op.node_name = slice->get_friendly_name();
      op.type_name = slice->get_type_name();
      op.input_element_types = {"f32", "i64", "i64", "i64", "i64"};
      op.input_shapes = {"{1,2,3}", "{1}", "{1}", "{1}", "{1}"};
      op.output_shapes = {"{1,2,3}"};
    } else if (std::string_view(test_case.op_type) == "StridedSlice") {
      auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                           ov::Shape{1, 2, 3});
      auto begin = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3},
                                                {0, 0, 0});
      auto end = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3},
                                              {1, 2, 3});
      auto strides = ov::op::v0::Constant::create(ov::element::i64,
                                                  ov::Shape{3}, {1, 1, 1});
      const std::vector<int64_t> zero_mask = {0, 0, 0};
      auto slice = std::make_shared<ov::op::v1::StridedSlice>(
          input, begin, end, strides, zero_mask, zero_mask, zero_mask,
          zero_mask, zero_mask);
      op.source_node = slice;
      op.node_name = slice->get_friendly_name();
      op.type_name = slice->get_type_name();
      op.input_element_types = {"f32", "i64", "i64", "i64"};
      op.input_shapes = {"{1,2,3}", "{3}", "{3}", "{3}"};
      op.output_shapes = {"{1,2,3}"};
    }
    plan.operations.push_back(std::move(op));

    const auto manifest = compiler::ManifestBuilder{}.build(plan);
    ASSERT_TRUE(manifest.valid());
    ASSERT_EQ(manifest.stages.size(), 1u);
    EXPECT_EQ(manifest.stages.front().runtime_shape.rule,
              test_case.expected_rule);
    EXPECT_EQ(manifest.stages.front().runtime_shape.i64_metadata,
              test_case.expected_i64_metadata);

    const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
    ASSERT_TRUE(executable.valid());
    ASSERT_EQ(executable.artifact_descriptors.size(), 1u);
    EXPECT_EQ(executable.artifact_descriptors.front().kernel.runtime_shape_rule,
              test_case.expected_rule);
    EXPECT_EQ(executable.artifact_descriptors.front()
                  .kernel.runtime_shape_i64_metadata,
              test_case.expected_i64_metadata);

    const auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_TRUE(compiler::runtime_executable_descriptor_valid(
        runtime_descriptor, executable));
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    EXPECT_EQ(runtime_descriptor.stages.front().runtime_shape_rule,
              test_case.expected_rule);
    EXPECT_EQ(runtime_descriptor.stages.front().runtime_shape_i64_metadata,
              test_case.expected_i64_metadata);

    auto stale_descriptor = runtime_descriptor;
    stale_descriptor.stages.front().runtime_shape_rule = "stale_runtime_rule";
    const auto stale_result = compiler::verify_runtime_executable_descriptor(
        stale_descriptor, executable);
    EXPECT_FALSE(stale_result.valid());
    EXPECT_TRUE(
        has_diagnostic_containing(stale_result.diagnostics, "artifact drift"));

    if (std::string_view(test_case.expected_rule) != "static_or_descriptor") {
      auto mismatched_descriptor = runtime_descriptor;
      mismatched_descriptor.stages.front().op_family = "Relu";
      const auto mismatched_result =
          compiler::verify_runtime_executable_descriptor(mismatched_descriptor,
                                                         executable);
      EXPECT_FALSE(mismatched_result.valid());
      EXPECT_TRUE(has_diagnostic_containing(
          mismatched_result.diagnostics,
          "runtime shape rule does not match op family"));
    }

    auto stale_metadata_descriptor = runtime_descriptor;
    stale_metadata_descriptor.stages.front().runtime_shape_i64_metadata = {42};
    const auto stale_metadata_result =
        compiler::verify_runtime_executable_descriptor(
            stale_metadata_descriptor, executable);
    EXPECT_FALSE(stale_metadata_result.valid());
    EXPECT_TRUE(has_diagnostic_containing(stale_metadata_result.diagnostics,
                                          "artifact drift"));
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeSubmissionContractComesFromCompilerManifestAndDescriptor) {
  struct Case {
    const char *op_type;
    bool expected_dependency_boundary;
  };

  for (const auto test_case :
       {Case{"Concat", true}, Case{"Softmax", true}, Case{"Relu", false}}) {
    SCOPED_TRACE(test_case.op_type);
    compiler::LoweringPlan plan;
    plan.target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
    compiler::PlannedOperation op;
    op.node_name = test_case.op_type;
    op.type_name = test_case.op_type;
    op.kernel_unit = compiler::KernelUnit::describe(
        LoweringRouteKind::Metadata, KernelUnitKind::Metadata, "metadata",
        plan.target.backend_id(), test_case.op_type);
    op.layout = compiler::TensorLayoutPlan{};
    op.profitability_score = 1.0;
    op.input_element_types = {"f32"};
    op.input_shapes = {"{1,2,3}"};
    op.output_element_types = {"f32"};
    op.output_shapes = {"{1,2,3}"};
    if (std::string_view(test_case.op_type) == "Concat") {
      auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                         ov::Shape{1, 2, 3});
      auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                         ov::Shape{1, 2, 3});
      auto concat =
          std::make_shared<ov::op::v0::Concat>(ov::OutputVector{lhs, rhs}, 1);
      op.source_node = concat;
      op.node_name = concat->get_friendly_name();
      op.type_name = concat->get_type_name();
      op.input_element_types = {"f32", "f32"};
      op.input_shapes = {"{1,2,3}", "{1,2,3}"};
      op.output_shapes = {"{1,4,3}"};
    }
    plan.operations.push_back(std::move(op));

    const auto manifest = compiler::ManifestBuilder{}.build(plan);
    ASSERT_TRUE(manifest.valid());
    ASSERT_EQ(manifest.stages.size(), 1u);
    EXPECT_EQ(manifest.stages.front().submission.stage_weight, 1u);
    EXPECT_EQ(manifest.stages.front().submission.dependency_extension_boundary,
              test_case.expected_dependency_boundary);

    const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
    ASSERT_TRUE(executable.valid());

    const auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_TRUE(compiler::runtime_executable_descriptor_valid(
        runtime_descriptor, executable));
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    EXPECT_EQ(runtime_descriptor.stages.front().submission_stage_weight,
              manifest.stages.front().submission.stage_weight);
    EXPECT_EQ(runtime_descriptor.stages.front().submission_macs_estimate,
              manifest.stages.front().submission.macs_estimate);
    EXPECT_EQ(runtime_descriptor.stages.front().submission_dependency_boundary,
              test_case.expected_dependency_boundary);

    auto stale_descriptor = runtime_descriptor;
    stale_descriptor.stages.front().submission_dependency_boundary =
        !test_case.expected_dependency_boundary;
    const auto stale_result = compiler::verify_runtime_executable_descriptor(
        stale_descriptor, executable);
    EXPECT_FALSE(stale_result.valid());
    EXPECT_TRUE(
        has_diagnostic_containing(stale_result.diagnostics, "artifact drift"));
  }
}


TEST_F(GfxBackendArchitectureContractTest,
       RuntimeShapeRuleOwnershipIsDescriptorContract) {
  struct Case {
    const char *op_family;
    const char *runtime_shape_rule;
  };

  const std::vector<Case> positive_cases = {
      {"Concat", "concat"},         {"Broadcast", "broadcast"},
      {"Select", "select"},         {"ShapeOf", "shape_of"},
      {"Slice", "slice"},           {"StridedSlice", "slice"},
      {"Range", "range"},           {"Tile", "tile"},
      {"MatMul", "static_or_descriptor"},
  };

  for (const auto &test_case : positive_cases) {
    SCOPED_TRACE(std::string(test_case.op_family) + ":" +
                 test_case.runtime_shape_rule);
    EXPECT_TRUE(descriptor_owns_runtime_shape_rule(
        test_case.op_family, test_case.runtime_shape_rule));
    const auto materialization_rule = runtime_shape_materialization_rule_for(
        test_case.runtime_shape_rule);
    ASSERT_TRUE(materialization_rule.has_value());
    EXPECT_EQ(materialization_rule->kind,
              runtime_shape_rule_kind_from_name(test_case.runtime_shape_rule));
    EXPECT_EQ(materialization_rule->runtime_shape_rule,
              std::string_view(test_case.runtime_shape_rule));
    EXPECT_EQ(materialization_rule->requires_descriptor,
              std::string_view(test_case.runtime_shape_rule) !=
                  std::string_view("static_or_descriptor"));
  }

  const std::vector<Case> negative_cases = {
      {"Concat", "broadcast"},      {"Broadcast", "concat"},
      {"Select", "slice"},          {"ShapeOf", "range"},
      {"Slice", "tile"},            {"StridedSlice", "concat"},
      {"Range", "shape_of"},        {"Tile", "select"},
      {"ReduceSum", "reduce"},      {"Broadcast", "unknown"},
  };

  for (const auto &test_case : negative_cases) {
    SCOPED_TRACE(std::string(test_case.op_family) + ":" +
                 test_case.runtime_shape_rule);
    EXPECT_FALSE(descriptor_owns_runtime_shape_rule(
        test_case.op_family, test_case.runtime_shape_rule));
    EXPECT_EQ(runtime_shape_materialization_rule_supported(
                  test_case.runtime_shape_rule),
              runtime_shape_rule_known(test_case.runtime_shape_rule));
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeShapeMaterializerUsesSameDescriptorContractForBackendDomains) {
  auto make_binding = [](std::string name, std::string role,
                         std::string element_type, std::string partial_shape) {
    auto binding =
        make_runtime_binding(std::move(name), "unit_region", std::move(role));
    binding.element_type = std::move(element_type);
    binding.partial_shape = std::move(partial_shape);
    return binding;
  };

  auto tensor = [](ov::Shape shape, ov::element::Type type) {
    GpuTensor value;
    value.shape = std::move(shape);
    value.expected_type = type;
    return value;
  };

  for (const auto backend_domain : {"metal", "opencl"}) {
    SCOPED_TRACE(backend_domain);

    RuntimeStageExecutableDescriptor descriptor;
    descriptor.stage_index = 0;
    descriptor.stage_record_key = 0x9002u;
    descriptor.artifact_descriptor_index = 0;
    descriptor.manifest_ref = "manifest://unit/runtime_shape_materializer";
    descriptor.abi_fingerprint = "abi://unit/runtime_shape_materializer";
    descriptor.artifact_key = "artifact://unit/runtime_shape_materializer";
    descriptor.backend_domain = backend_domain;
    descriptor.kernel_id = "unit/runtime_shape_materializer";
    descriptor.op_family = "Concat";
    descriptor.stage_name = "Concat_runtime_shape_materializer";
    descriptor.origin = KernelArtifactOrigin::Generated;
    descriptor.payload_kind = std::string_view(backend_domain) == "metal"
                                  ? KernelArtifactPayloadKind::MslSource
                                  : KernelArtifactPayloadKind::OpenClSource;
    descriptor.entry_point = "unit_runtime_shape_materializer";
    descriptor.runtime_shape_rule = "concat";
    descriptor.runtime_shape_i64_metadata = {1};
    descriptor.input_bindings = {
        make_binding("lhs", "TensorInput", "f32", "{1,2,2}"),
        make_binding("rhs", "TensorInput", "f32", "{1,3,2}")};
    descriptor.output_bindings = {
        make_binding("out", "TensorOutput", "f32", "{1,5,2}")};

    InferStage stage;
    stage.runtime_stage_descriptor =
        std::make_shared<RuntimeStageExecutableDescriptor>(descriptor);
    stage.outputs.emplace_back(std::make_unique<GpuTensor>());

    std::vector<GpuTensor> storage = {tensor({1, 2, 2}, ov::element::f32),
                                      tensor({1, 3, 2}, ov::element::f32)};
    std::vector<GpuTensor *> inputs = {&storage[0], &storage[1]};

    RuntimeInputResolver resolver;
    resolver.inputs = &inputs;
    resolver.descriptor = &descriptor;
    RuntimeShapeMaterializationRequest request;
    request.inputs = resolver;
    request.descriptor = &descriptor;
    request.outputs = {stage.outputs.front().get()};
    request.stage_name = descriptor.stage_name;
    request.error_prefix = "GFX test";
    const auto result = materialize_runtime_output_shapes(request);

    ASSERT_EQ(stage.outputs.size(), 1u);
    ASSERT_TRUE(stage.outputs.front());
    EXPECT_TRUE(result.materialized);
    EXPECT_EQ(result.kind, RuntimeShapeRuleKind::Concat);
    ASSERT_TRUE(result.concat.valid());
    EXPECT_EQ(result.concat.axis_norm, 1);
    EXPECT_EQ(stage.outputs.front()->shape, (ov::Shape{1, 5, 2}));
    EXPECT_EQ(stage.outputs.front()->expected_type, ov::element::f32);
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       DescriptorOwnedRuntimeShapeRulesDoNotRequireSourceNodeBridge) {
  auto make_binding = [](std::string name, std::string role,
                         std::string element_type, std::string partial_shape) {
    auto binding =
        make_runtime_binding(std::move(name), "unit_region", std::move(role));
    binding.element_type = std::move(element_type);
    binding.partial_shape = std::move(partial_shape);
    return binding;
  };

  auto make_descriptor = [&](std::string backend_domain, std::string op_family,
                             std::string rule,
                             std::vector<RuntimeTensorBindingContract> inputs,
                             std::vector<RuntimeTensorBindingContract> outputs,
                             std::vector<int64_t> metadata = {}) {
    RuntimeStageExecutableDescriptor descriptor;
    descriptor.stage_index = 0;
    descriptor.stage_record_key = 0x9001u;
    descriptor.artifact_descriptor_index = 0;
    descriptor.manifest_ref = "manifest://unit/runtime_shape";
    descriptor.abi_fingerprint = "abi://unit/runtime_shape";
    descriptor.artifact_key = "artifact://unit/runtime_shape";
    descriptor.backend_domain = std::move(backend_domain);
    descriptor.kernel_id = "unit/runtime_shape";
    descriptor.op_family = std::move(op_family);
    descriptor.stage_name = descriptor.op_family + "_runtime_shape";
    descriptor.origin = KernelArtifactOrigin::Generated;
    descriptor.payload_kind = KernelArtifactPayloadKind::MslSource;
    descriptor.entry_point = "unit_runtime_shape";
    descriptor.runtime_shape_rule = std::move(rule);
    descriptor.runtime_shape_i64_metadata = std::move(metadata);
    descriptor.input_bindings = std::move(inputs);
    descriptor.output_bindings = std::move(outputs);
    return descriptor;
  };

  auto tensor = [](ov::Shape shape, ov::element::Type type) {
    GpuTensor value;
    value.shape = std::move(shape);
    value.expected_type = type;
    return value;
  };
  auto i64_tensor = [](std::vector<int64_t> values) {
    GpuTensor value;
    value.shape = ov::Shape{values.size()};
    value.expected_type = ov::element::i64;
    value.i64_values = std::move(values);
    return value;
  };

  for (const auto backend_domain : {"metal", "opencl"}) {
    SCOPED_TRACE(backend_domain);

    {
      auto descriptor = make_descriptor(
          backend_domain, "Concat", "concat",
          {make_binding("lhs", "TensorInput", "f32", "{1,2,2}"),
           make_binding("rhs", "TensorInput", "f32", "{1,3,2}")},
          {make_binding("out", "TensorOutput", "f32", "{1,5,2}")}, {1});
      std::vector<GpuTensor> storage = {tensor({1, 2, 2}, ov::element::f32),
                                        tensor({1, 3, 2}, ov::element::f32)};
      std::vector<GpuTensor *> input_ptrs = {&storage[0], &storage[1]};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      ASSERT_EQ(inputs.descriptor, &descriptor);
      const auto plan =
          plan_concat_runtime_values(inputs, descriptor.stage_name);
      ASSERT_TRUE(plan.valid());
      EXPECT_EQ(plan.values.output_shape, (ov::Shape{1, 5, 2}));
      EXPECT_EQ(plan.axis_norm, 1);
    }

    {
      auto descriptor = make_descriptor(
          backend_domain, "Broadcast", "broadcast",
          {make_binding("input", "TensorInput", "f32", "{1,3}"),
           make_binding("target", "TensorInput", "i64", "{2}")},
          {make_binding("out", "TensorOutput", "f32", "{2,3}")}, {0, 2});
      std::vector<GpuTensor> storage = {tensor({1, 3}, ov::element::f32),
                                        i64_tensor({2, 3})};
      std::vector<GpuTensor *> input_ptrs = {&storage[0], &storage[1]};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      ASSERT_EQ(inputs.descriptor, &descriptor);
      const auto plan = plan_broadcast_runtime_values(inputs, storage[0].shape,
                                                      descriptor.stage_name);
      EXPECT_EQ(plan.output_shape, (ov::Shape{2, 3}));
      EXPECT_EQ(plan.output_type, ov::element::f32);
    }

    {
      auto descriptor = make_descriptor(
          backend_domain, "Select", "select",
          {make_binding("cond", "TensorInput", "boolean", "{1,3}"),
           make_binding("then", "TensorInput", "f32", "{1,3}"),
           make_binding("else", "TensorInput", "f32", "{1,3}")},
          {make_binding("out", "TensorOutput", "f32", "{1,3}")});
      std::vector<GpuTensor> storage = {tensor({1, 3}, ov::element::boolean),
                                        tensor({1, 3}, ov::element::f32),
                                        tensor({1, 3}, ov::element::f32)};
      std::vector<GpuTensor *> input_ptrs = {&storage[0], &storage[1],
                                             &storage[2]};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      ASSERT_EQ(inputs.descriptor, &descriptor);
      const auto plan =
          plan_select_runtime_values(inputs, descriptor.stage_name);
      ASSERT_TRUE(plan.valid());
      EXPECT_EQ(plan.values.output_shape, (ov::Shape{1, 3}));
      EXPECT_EQ(plan.values.output_type, ov::element::f32);
    }

    {
      auto descriptor = make_descriptor(
          backend_domain, "ShapeOf", "shape_of",
          {make_binding("input", "TensorInput", "f32", "{2,3,4}")},
          {make_binding("out", "TensorOutput", "i64", "{3}")});
      std::vector<GpuTensor> storage = {tensor({2, 3, 4}, ov::element::f32)};
      std::vector<GpuTensor *> input_ptrs = {&storage[0]};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      ASSERT_EQ(inputs.descriptor, &descriptor);
      const auto plan =
          plan_shapeof_runtime_values(inputs, descriptor.stage_name);
      EXPECT_EQ(plan.output_shape, (ov::Shape{3}));
      EXPECT_EQ(plan.i64_values, (std::vector<int64_t>{2, 3, 4}));
    }

    {
      auto descriptor =
          make_descriptor(backend_domain, "Range", "range",
                          {make_binding("start", "TensorInput", "i64", "{1}"),
                           make_binding("stop", "TensorInput", "i64", "{1}"),
                           make_binding("step", "TensorInput", "i64", "{1}")},
                          {make_binding("out", "TensorOutput", "i64", "{3}")});
      std::vector<GpuTensor> storage = {i64_tensor({0}), i64_tensor({3}),
                                        i64_tensor({1})};
      std::vector<GpuTensor *> input_ptrs = {&storage[0], &storage[1],
                                             &storage[2]};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      ASSERT_EQ(inputs.descriptor, &descriptor);
      const auto plan =
          plan_range_runtime_values(inputs, descriptor.stage_name);
      EXPECT_EQ(plan.output_shape, (ov::Shape{3}));
      EXPECT_EQ(plan.i64_values, (std::vector<int64_t>{0, 1, 2}));
    }

    {
      auto descriptor = make_descriptor(
          backend_domain, "Tile", "tile",
          {make_binding("input", "TensorInput", "f32", "{2,3}"),
           make_binding("repeats", "TensorInput", "i64", "{2}")},
          {make_binding("out", "TensorOutput", "f32", "{4,3}")});
      std::vector<GpuTensor> storage = {tensor({2, 3}, ov::element::f32),
                                        i64_tensor({2, 1})};
      std::vector<GpuTensor *> input_ptrs = {&storage[0], &storage[1]};
      GpuTensor output;
      std::vector<GpuTensor *> outputs = {&output};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      ASSERT_EQ(inputs.descriptor, &descriptor);
      const auto plan =
          plan_tile_runtime_values(inputs, outputs, descriptor.stage_name);
      ASSERT_TRUE(plan.valid());
      EXPECT_EQ(plan.output_shape, (ov::Shape{4, 3}));
      EXPECT_EQ(plan.values.output_type, ov::element::f32);
    }

    {
      auto descriptor = make_descriptor(
          backend_domain, "Reshape", "static_or_descriptor",
          {make_binding("input", "TensorInput", "f32", "{2,3}"),
           make_binding("pattern", "TensorInput", "i64", "{2}")},
          {make_binding("out", "TensorOutput", "f32", "{3,2}")}, {0});
      std::vector<GpuTensor> storage = {tensor({2, 3}, ov::element::f32),
                                        i64_tensor({3, 2})};
      std::vector<GpuTensor *> input_ptrs = {&storage[0], &storage[1]};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      const auto plan =
          plan_reshape_runtime_values(inputs, descriptor, descriptor.stage_name);
      EXPECT_EQ(plan.output_shape, (ov::Shape{3, 2}));
      EXPECT_EQ(plan.output_type, ov::element::f32);
    }

    {
      auto descriptor = make_descriptor(
          backend_domain, "Squeeze", "static_or_descriptor",
          {make_binding("input", "TensorInput", "f32", "{1,3,1}"),
           make_binding("axes", "TensorInput", "i64", "{2}")},
          {make_binding("out", "TensorOutput", "f32", "{3}")});
      std::vector<GpuTensor> storage = {tensor({1, 3, 1}, ov::element::f32),
                                        i64_tensor({0, 2})};
      std::vector<GpuTensor *> input_ptrs = {&storage[0], &storage[1]};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      const auto plan = plan_squeeze_unsqueeze_runtime_values(
          inputs, descriptor, descriptor.stage_name);
      EXPECT_EQ(plan.output_shape, (ov::Shape{3}));
      EXPECT_EQ(plan.output_type, ov::element::f32);
    }

    {
      auto descriptor = make_descriptor(
          backend_domain, "Unsqueeze", "static_or_descriptor",
          {make_binding("input", "TensorInput", "f32", "{3}"),
           make_binding("axes", "TensorInput", "i64", "{2}")},
          {make_binding("out", "TensorOutput", "f32", "{1,3,1}")});
      std::vector<GpuTensor> storage = {tensor({3}, ov::element::f32),
                                        i64_tensor({0, 2})};
      std::vector<GpuTensor *> input_ptrs = {&storage[0], &storage[1]};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      const auto plan = plan_squeeze_unsqueeze_runtime_values(
          inputs, descriptor, descriptor.stage_name);
      EXPECT_EQ(plan.output_shape, (ov::Shape{1, 3, 1}));
      EXPECT_EQ(plan.output_type, ov::element::f32);
    }

    {
      auto descriptor = make_descriptor(
          backend_domain, "Convert", "static_or_descriptor",
          {make_binding("input", "TensorInput", "i64", "{3}")},
          {make_binding("out", "TensorOutput", "f32", "{3}")});
      std::vector<GpuTensor> storage = {i64_tensor({1, 2, 3})};
      std::vector<GpuTensor *> input_ptrs = {&storage[0]};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      const auto plan =
          plan_convert_runtime_values(inputs, descriptor, descriptor.stage_name);
      EXPECT_EQ(plan.output_shape, (ov::Shape{3}));
      EXPECT_EQ(plan.output_type, ov::element::f32);
      EXPECT_EQ(plan.i64_values, (std::vector<int64_t>{1, 2, 3}));
    }

    {
      auto descriptor = make_descriptor(
          backend_domain, "Gather", "static_or_descriptor",
          {make_binding("data", "TensorInput", "i64", "{4}"),
           make_binding("indices", "TensorInput", "i64", "{2}"),
           make_binding("axis", "TensorInput", "i64", "{1}")},
          {make_binding("out", "TensorOutput", "i64", "{2}")}, {0});
      std::vector<GpuTensor> storage = {i64_tensor({10, 20, 30, 40}),
                                        i64_tensor({1, 3}), i64_tensor({0})};
      std::vector<GpuTensor *> input_ptrs = {&storage[0], &storage[1],
                                             &storage[2]};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      const auto plan =
          plan_gather_runtime_values(inputs, descriptor, descriptor.stage_name);
      ASSERT_TRUE(plan.valid());
      EXPECT_EQ(plan.values.output_shape, (ov::Shape{2}));
      EXPECT_EQ(plan.values.i64_values, (std::vector<int64_t>{20, 40}));
    }

    {
      auto descriptor = make_descriptor(
          backend_domain, "ScatterUpdate", "static_or_descriptor",
          {make_binding("data", "TensorInput", "f32", "{4}"),
           make_binding("indices", "TensorInput", "i64", "{2}"),
           make_binding("updates", "TensorInput", "f32", "{2}"),
           make_binding("axis", "TensorInput", "i64", "{1}")},
          {make_binding("out", "TensorOutput", "f32", "{4}")});
      std::vector<GpuTensor> storage = {tensor({4}, ov::element::f32),
                                        i64_tensor({1, 3}),
                                        tensor({2}, ov::element::f32),
                                        i64_tensor({0})};
      std::vector<GpuTensor *> input_ptrs = {&storage[0], &storage[1],
                                             &storage[2], &storage[3]};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      const auto plan = plan_scatter_update_runtime_values(
          inputs, descriptor, descriptor.stage_name);
      ASSERT_TRUE(plan.valid());
      EXPECT_EQ(plan.values.output_shape, (ov::Shape{4}));
      EXPECT_EQ(plan.axis_norm, 0);
    }

    {
      auto descriptor = make_descriptor(
          backend_domain, "Split", "static_or_descriptor",
          {make_binding("input", "TensorInput", "f32", "{2,4}"),
           make_binding("axis", "TensorInput", "i64", "{1}")},
          {make_binding("out0", "TensorOutput", "f32", "{2,2}"),
           make_binding("out1", "TensorOutput", "f32", "{2,2}")});
      std::vector<GpuTensor> storage = {tensor({2, 4}, ov::element::f32),
                                        i64_tensor({1})};
      std::vector<GpuTensor *> input_ptrs = {&storage[0], &storage[1]};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      const auto plan =
          plan_split_runtime_values(inputs, descriptor, 2, descriptor.stage_name);
      ASSERT_TRUE(plan.valid());
      EXPECT_EQ(plan.axis_norm, 1);
      EXPECT_EQ(plan.split_sizes, (std::vector<size_t>{2, 2}));
    }

    {
      auto descriptor = make_descriptor(
          backend_domain, "Transpose", "static_or_descriptor",
          {make_binding("input", "TensorInput", "f32", "{2,3}"),
           make_binding("perm", "TensorInput", "i64", "{2}")},
          {make_binding("out", "TensorOutput", "f32", "{3,2}")});
      std::vector<GpuTensor> storage = {tensor({2, 3}, ov::element::f32),
                                        i64_tensor({1, 0})};
      std::vector<GpuTensor *> input_ptrs = {&storage[0], &storage[1]};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      const auto plan = plan_transpose_runtime_values(inputs, descriptor,
                                                      descriptor.stage_name);
      ASSERT_TRUE(plan.valid());
      EXPECT_EQ(plan.values.output_shape, (ov::Shape{3, 2}));
      EXPECT_EQ(plan.permutation, (std::vector<int64_t>{1, 0}));
    }

    {
      auto descriptor = make_descriptor(
          backend_domain, "Interpolate", "static_or_descriptor",
          {make_binding("input", "TensorInput", "f32", "{1,1,2,2}")},
          {make_binding("out", "TensorOutput", "f32", "{1,1,4,4}")},
          {0, 1, 0});
      std::vector<GpuTensor> storage = {tensor({1, 1, 2, 2}, ov::element::f32)};
      std::vector<GpuTensor *> input_ptrs = {&storage[0]};
      GpuTensor output;
      std::vector<GpuTensor *> outputs = {&output};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      const auto plan = plan_interpolate_runtime_values(
          inputs, outputs, descriptor, descriptor.stage_name);
      ASSERT_TRUE(plan.valid());
      EXPECT_EQ(plan.values.output_shape, (ov::Shape{1, 1, 4, 4}));
      EXPECT_FALSE(plan.align_corners);
      EXPECT_TRUE(plan.use_half_pixel);
      EXPECT_EQ(plan.nearest_mode, 0u);
    }

    {
      auto descriptor = make_descriptor(
          backend_domain, "Slice", "slice",
          {make_binding("input", "TensorInput", "f32", "{2,5}"),
           make_binding("starts", "TensorInput", "i64", "{1}"),
           make_binding("ends", "TensorInput", "i64", "{1}"),
           make_binding("steps", "TensorInput", "i64", "{1}"),
           make_binding("axes", "TensorInput", "i64", "{1}")},
          {make_binding("out", "TensorOutput", "f32", "{2,3}")}, {1, 1, 5});
      std::vector<GpuTensor> storage = {tensor({2, 5}, ov::element::f32),
                                        i64_tensor({1}), i64_tensor({4}),
                                        i64_tensor({1}), i64_tensor({1})};
      std::vector<GpuTensor *> input_ptrs = {
          &storage[0], &storage[1], &storage[2], &storage[3], &storage[4]};
      GpuTensor output;
      std::vector<GpuTensor *> outputs = {&output};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      ASSERT_EQ(inputs.descriptor, &descriptor);
      const auto plan = plan_slice_runtime_values(inputs, outputs, false,
                                                  descriptor.stage_name);
      ASSERT_TRUE(plan.valid());
      EXPECT_EQ(plan.values.output_shape, (ov::Shape{2, 3}));
      EXPECT_EQ(plan.values.output_type, ov::element::f32);
      EXPECT_EQ(plan.starts_full, (std::vector<int32_t>{0, 1}));
      EXPECT_EQ(plan.steps_full, (std::vector<int32_t>{1, 1}));
    }

    {
      const std::vector<int64_t> strided_slice_metadata = {
          1, 2, 4, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0};
      auto descriptor = make_descriptor(
          backend_domain, "StridedSlice", "slice",
          {make_binding("input", "TensorInput", "f32", "{4,5}"),
           make_binding("begin", "TensorInput", "i64", "{2}"),
           make_binding("end", "TensorInput", "i64", "{2}"),
           make_binding("strides", "TensorInput", "i64", "{2}")},
          {make_binding("out", "TensorOutput", "f32", "{3,3}")},
          strided_slice_metadata);
      std::vector<GpuTensor> storage = {tensor({4, 5}, ov::element::f32),
                                        i64_tensor({1, 0}), i64_tensor({4, 5}),
                                        i64_tensor({1, 2})};
      std::vector<GpuTensor *> input_ptrs = {&storage[0], &storage[1],
                                             &storage[2], &storage[3]};
      GpuTensor output;
      std::vector<GpuTensor *> outputs = {&output};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      ASSERT_EQ(inputs.descriptor, &descriptor);
      const auto plan = plan_slice_runtime_values(inputs, outputs, false,
                                                  descriptor.stage_name);
      ASSERT_TRUE(plan.valid());
      EXPECT_EQ(plan.values.output_shape, (ov::Shape{3, 3}));
      EXPECT_EQ(plan.values.output_type, ov::element::f32);
      EXPECT_EQ(plan.starts_full, (std::vector<int32_t>{1, 0}));
      EXPECT_EQ(plan.steps_full, (std::vector<int32_t>{1, 2}));
    }
  }
}

} // namespace
} // namespace gfx_plugin
} // namespace ov
