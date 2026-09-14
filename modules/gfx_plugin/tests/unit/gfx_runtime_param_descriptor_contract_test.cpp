// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unit/gfx_runtime_param_contract_utils.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

void install_runtime_shape_metadata(compiler::ExecutableBundle &executable,
                                    std::vector<int64_t> metadata) {
  OPENVINO_ASSERT(executable.manifest.stages.size() == 1 &&
                      executable.artifact_descriptors.size() == 1 &&
                      executable.artifact_payloads.size() == 1,
                  "unit executable must have one stage and one artifact");
  executable.manifest.stages.front().runtime_shape.i64_metadata = metadata;
  auto &artifact = executable.artifact_descriptors.front();
  artifact.kernel.runtime_shape_i64_metadata = std::move(metadata);
  compiler::finalize_kernel_artifact_descriptor_identity(artifact);
  executable.artifact_payloads.front().artifact_key = artifact.artifact_key;
}

TEST_F(GfxBackendArchitectureContractTest,
       DescriptorOwnedRuntimeParamsStayDescriptorOnly) {
  struct Case {
    const char *backend_domain;
    KernelArtifactPayloadKind payload_kind;
  };

  const std::vector<GfxKernelBufferRole> binary_roles = {
      GfxKernelBufferRole::TensorInput,   GfxKernelBufferRole::TensorInput,
      GfxKernelBufferRole::TensorOutput,  GfxKernelBufferRole::ScalarParam,
      GfxKernelBufferRole::ScalarParam,   GfxKernelBufferRole::RuntimeParams,
      GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams};

  for (const auto test_case :
       {Case{"metal", KernelArtifactPayloadKind::MslSource},
        Case{"opencl", KernelArtifactPayloadKind::OpenClSource}}) {
    SCOPED_TRACE(test_case.backend_domain);
    auto executable = make_source_payload_executable(
        test_case.backend_domain, "Add", binary_roles, {"{1,3}", "{1,3}"},
        {"{1,3}"});
    ASSERT_TRUE(executable.verify().valid());
    ASSERT_EQ(executable.artifact_descriptors.size(), 1u);
    EXPECT_EQ(executable.artifact_descriptors.front().payload_kind,
              test_case.payload_kind);

    const auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_TRUE(compiler::runtime_executable_descriptor_valid(
        runtime_descriptor, executable));
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    EXPECT_EQ(runtime_descriptor.stages.front().payload_kind,
              test_case.payload_kind);
    EXPECT_EQ(runtime_descriptor.stages.front().runtime_param_buffer_count, 3u);
    EXPECT_EQ(runtime_descriptor.stages.front().runtime_param_payload_kind,
              RuntimeParamDescriptorPayloadKind::BinaryBroadcast);
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       SourceLaunchPlanAbiIsDescriptorOwnedAcrossSourceBackends) {
  const std::vector<GfxKernelBufferRole> roles = {
      GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
      GfxKernelBufferRole::ScalarParam, GfxKernelBufferRole::TensorOutput};

  for (const auto backend_domain : {"metal", "opencl"}) {
    SCOPED_TRACE(backend_domain);
    auto executable = make_source_payload_executable(
        backend_domain, "Add", roles, {"{1,3}", "{1,3}"}, {"{1,3}"});
    const auto executable_result = executable.verify();
    ASSERT_TRUE(executable_result.valid())
        << (executable_result.diagnostics.empty()
                ? std::string{}
                : executable_result.diagnostics.front());

    ASSERT_EQ(executable.artifact_descriptors.size(), 1u);
    const auto &artifact = executable.artifact_descriptors.front();
    ASSERT_TRUE(artifact.launch_plan.valid);
    EXPECT_EQ(artifact.launch_plan.buffer_roles,
              (std::vector<std::string>{"tensor_input", "tensor_input",
                                        "scalar_param", "tensor_output"}));
    EXPECT_EQ(artifact.launch_plan.direct_input_indices,
              (std::vector<size_t>{0, 1}));

    auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_TRUE(compiler::runtime_executable_descriptor_valid(
        runtime_descriptor, executable));
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    EXPECT_EQ(runtime_descriptor.stages.front().launch_plan.buffer_roles,
              artifact.launch_plan.buffer_roles);
    EXPECT_EQ(
        runtime_descriptor.stages.front().launch_plan.direct_input_indices,
        artifact.launch_plan.direct_input_indices);

    runtime_descriptor.stages.front().launch_plan.buffer_roles.pop_back();
    const auto stale_result = compiler::verify_runtime_executable_descriptor(
        runtime_descriptor, executable);
    ASSERT_FALSE(stale_result.valid());
    EXPECT_TRUE(has_diagnostic_containing(
        stale_result.diagnostics, "source launch-plan ABI count drift"));
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       SharedRuntimeParamDescriptorContractCoversGeneratedSourceFamilies) {
  struct FamilyCase {
    const char *op_family;
    size_t runtime_param_count;
    RuntimeParamDescriptorPayloadKind payload_kind;
    std::vector<GfxKernelBufferRole> roles;
    std::vector<std::string> input_shapes;
    std::vector<std::string> output_shapes;
    std::vector<int64_t> runtime_shape_i64_metadata;
  };

  const std::vector<FamilyCase> families = {
      {"Add",
       3,
       RuntimeParamDescriptorPayloadKind::BinaryBroadcast,
       {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
        GfxKernelBufferRole::TensorOutput, GfxKernelBufferRole::ScalarParam,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::RuntimeParams},
       {"{1,3}", "{1,3}"},
       {"{1,3}"}},
      {"Broadcast",
       4,
       RuntimeParamDescriptorPayloadKind::Broadcast,
       {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams},
       {"{1,3}"},
       {"{2,3}"}},
      {"Select",
       4,
       RuntimeParamDescriptorPayloadKind::Select,
       {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
        GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams},
       {"{1,3}", "{1,3}", "{1,3}"},
       {"{1,3}"}},
      {"Tile",
       4,
       RuntimeParamDescriptorPayloadKind::Tile,
       {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams},
       {"{1,3}"},
       {"{2,3}"}},
      {"Interpolate",
       1,
       RuntimeParamDescriptorPayloadKind::Interpolate,
       {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::TensorOutput},
       {"{1,3,2,2}"},
       {"{1,3,4,4}"},
       {0, 1, 0}},
  };

  for (const auto &family : families) {
    SCOPED_TRACE(family.op_family);
    for (const auto backend_domain : {"metal", "opencl"}) {
      SCOPED_TRACE(backend_domain);
      auto executable = make_source_payload_executable(
          backend_domain, family.op_family, family.roles, family.input_shapes,
          family.output_shapes);
      if (!family.runtime_shape_i64_metadata.empty()) {
        install_runtime_shape_metadata(executable,
                                       family.runtime_shape_i64_metadata);
      }
      ASSERT_TRUE(executable.verify().valid());

      const auto runtime_descriptor =
          compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
      ASSERT_TRUE(compiler::runtime_executable_descriptor_valid(
          runtime_descriptor, executable));
      ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
      const auto &stage = runtime_descriptor.stages.front();
      EXPECT_EQ(stage.runtime_param_buffer_count, family.runtime_param_count);
      EXPECT_EQ(stage.runtime_param_payload_kind, family.payload_kind);
      EXPECT_TRUE(descriptor_owns_runtime_param_payload(stage));
    }
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeParamPayloadKindIsDescriptorOwnedNotRediscoveredFromOpFamily) {
  const std::vector<GfxKernelBufferRole> binary_roles = {
      GfxKernelBufferRole::TensorInput,   GfxKernelBufferRole::TensorInput,
      GfxKernelBufferRole::TensorOutput,  GfxKernelBufferRole::ScalarParam,
      GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
      GfxKernelBufferRole::RuntimeParams};

  auto executable = make_source_payload_executable(
      "opencl", "Add", binary_roles, {"{1,3}", "{1,3}"}, {"{1,3}"});
  const auto runtime_descriptor =
      compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
  ASSERT_TRUE(compiler::runtime_executable_descriptor_valid(runtime_descriptor,
                                                            executable));
  ASSERT_EQ(runtime_descriptor.stages.size(), 1u);

  auto stage = runtime_descriptor.stages.front();
  ASSERT_EQ(stage.runtime_param_payload_kind,
            RuntimeParamDescriptorPayloadKind::BinaryBroadcast);
  stage.op_family = "Tile";
  EXPECT_TRUE(descriptor_owns_runtime_param_payload(stage));

  UnitMetadataBufferManager buffer_manager;
  RuntimeInputResolver inputs;
  inputs.descriptor = &stage;
  GpuTensor output;
  std::vector<GpuTensor *> outputs = {&output};
  const std::vector<int32_t> compiler_scalar_args;

  auto materialization = materialize_descriptor_owned_runtime_param_payload(
      buffer_manager, stage, inputs, outputs, compiler_scalar_args,
      "unit_payload_kind_not_op_family");
  ASSERT_TRUE(materialization.available);
  EXPECT_EQ(materialization.extra_inputs.size(), 3u);

  auto stale_descriptor = runtime_descriptor;
  stale_descriptor.stages.front().runtime_param_payload_kind =
      RuntimeParamDescriptorPayloadKind::Tile;
  const auto stale_result = compiler::verify_runtime_executable_descriptor(
      stale_descriptor, executable);
  ASSERT_FALSE(stale_result.valid());
  EXPECT_TRUE(has_diagnostic_containing(stale_result.diagnostics,
                                        "artifact drift"));
}

TEST_F(GfxBackendArchitectureContractTest,
       ArtifactFinalizerDoesNotRewriteExplicitRuntimeParamSchema) {
  compiler::KernelArtifactDescriptor descriptor;
  descriptor.stage_record_key = 42;
  descriptor.payload_kind = KernelArtifactPayloadKind::OpenClSource;
  descriptor.kernel.backend_domain = "opencl";
  descriptor.kernel.kernel_id = "unit/opencl/add";
  descriptor.kernel.op_family = "Add";
  descriptor.runtime_param_buffer_count = 3;
  descriptor.runtime_param_payload_kind =
      RuntimeParamDescriptorPayloadKind::Tile;

  EXPECT_THROW(
      compiler::finalize_kernel_artifact_descriptor_identity(descriptor),
      ov::Exception);

  descriptor.runtime_param_payload_kind =
      RuntimeParamDescriptorPayloadKind::None;
  EXPECT_NO_THROW(
      compiler::finalize_kernel_artifact_descriptor_identity(descriptor));
  EXPECT_EQ(descriptor.runtime_param_payload_kind,
            RuntimeParamDescriptorPayloadKind::BinaryBroadcast);
}

TEST_F(GfxBackendArchitectureContractTest,
       MslRuntimeParamDescriptorContractCoversGeneratedMetadataFamilies) {
  struct FamilyCase {
    const char *op_family;
    size_t runtime_param_count;
    RuntimeParamDescriptorPayloadKind payload_kind;
    std::vector<GfxKernelBufferRole> roles;
    std::vector<std::string> input_shapes;
    std::vector<std::string> output_shapes;
    std::vector<int64_t> i64_metadata;
    std::vector<int64_t> runtime_shape_i64_metadata;
    bool reduce_keep_dims = false;
    bool reduce_keep_dims_valid = false;
  };

  const std::vector<FamilyCase> families = {
      {"Softmax",
       1,
       RuntimeParamDescriptorPayloadKind::Softmax,
       {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
        GfxKernelBufferRole::RuntimeParams},
       {"{2,3}"},
       {"{2,3}"},
       {2, 3, 1}},
      {"Transpose",
       5,
       RuntimeParamDescriptorPayloadKind::Transpose,
       {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::RuntimeParams},
       {"{2,3}"},
       {"{3,2}"},
       {1, 0}},
      {"ReduceSum",
       5,
       RuntimeParamDescriptorPayloadKind::Reduce,
       {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
        GfxKernelBufferRole::ScalarParam, GfxKernelBufferRole::ScalarParam,
        GfxKernelBufferRole::ScalarParam, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams},
       {"{2,3}"},
       {"{2}"},
       {1},
       {},
       false,
       true},
      {"Interpolate",
       1,
       RuntimeParamDescriptorPayloadKind::Interpolate,
       {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::TensorOutput},
       {"{1,3,2,2}"},
       {"{1,3,4,4}"},
       {},
       {0, 1, 0}},
  };

  for (const auto &family : families) {
    SCOPED_TRACE(family.op_family);
    GfxKernelSourceRuntimeBinding runtime_binding;
    runtime_binding.runtime_param_i64_metadata = family.i64_metadata;
    runtime_binding.runtime_param_reduce_keep_dims = family.reduce_keep_dims;
    runtime_binding.runtime_param_reduce_keep_dims_valid =
        family.reduce_keep_dims_valid;

    auto executable = make_source_payload_executable(
        "metal", family.op_family, family.roles, family.input_shapes,
        family.output_shapes, runtime_binding);
    if (!family.runtime_shape_i64_metadata.empty()) {
      install_runtime_shape_metadata(executable,
                                     family.runtime_shape_i64_metadata);
    }
    ASSERT_TRUE(executable.verify().valid());

    const auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_TRUE(compiler::runtime_executable_descriptor_valid(
        runtime_descriptor, executable));
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    const auto &stage = runtime_descriptor.stages.front();
    EXPECT_EQ(stage.runtime_param_buffer_count, family.runtime_param_count);
    EXPECT_EQ(stage.runtime_param_payload_kind, family.payload_kind);
    EXPECT_TRUE(descriptor_owns_runtime_param_payload(stage));
    EXPECT_EQ(stage.runtime_param_i64_metadata, family.i64_metadata);
    EXPECT_EQ(stage.runtime_param_reduce_keep_dims, family.reduce_keep_dims);
    EXPECT_EQ(stage.runtime_param_reduce_keep_dims_valid,
              family.reduce_keep_dims_valid);
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeParamMetadataComesFromArtifactDescriptorNotSourcePayload) {
  const std::vector<GfxKernelBufferRole> roles = {
      GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
      GfxKernelBufferRole::RuntimeParams};

  GfxKernelSourceRuntimeBinding payload_binding;
  payload_binding.runtime_param_i64_metadata = {99, 99, 99};

  auto executable = make_source_payload_executable(
      "metal", "Softmax", roles, {"{2,3}"}, {"{2,3}"}, payload_binding);
  ASSERT_EQ(executable.artifact_descriptors.size(), 1u);
  ASSERT_EQ(executable.artifact_payloads.size(), 1u);

  auto &artifact = executable.artifact_descriptors.front();
  artifact.runtime_param_buffer_count = 1;
  artifact.runtime_param_payload_kind = RuntimeParamDescriptorPayloadKind::Softmax;
  artifact.runtime_param_i64_metadata = {2, 3, 1};
  artifact.runtime_param_reduce_keep_dims = false;
  artifact.runtime_param_reduce_keep_dims_valid = false;
  compiler::finalize_kernel_artifact_descriptor_identity(artifact);
  executable.artifact_payloads.front().artifact_key = artifact.artifact_key;

  const auto runtime_descriptor =
      compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
  ASSERT_TRUE(compiler::runtime_executable_descriptor_valid(runtime_descriptor,
                                                            executable));
  ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
  const auto &stage = runtime_descriptor.stages.front();
  EXPECT_EQ(stage.runtime_param_buffer_count, 1u);
  EXPECT_EQ(stage.runtime_param_payload_kind,
            RuntimeParamDescriptorPayloadKind::Softmax);
  EXPECT_EQ(stage.runtime_param_i64_metadata, (std::vector<int64_t>{2, 3, 1}));
  EXPECT_TRUE(descriptor_owns_runtime_param_payload(stage));
}

TEST_F(GfxBackendArchitectureContractTest,
       ReduceRuntimeMetadataComesFromDescriptorNotSourceNode) {
  RuntimeStageExecutableDescriptor descriptor;
  descriptor.op_family = "ReduceSum";
  descriptor.runtime_param_buffer_count = 5;
  descriptor.runtime_param_payload_kind = RuntimeParamDescriptorPayloadKind::Reduce;
  descriptor.stage_name = "unit_reduce_descriptor_metadata";
  descriptor.entry_point = "gfx_metal_generated_reduction_sum_f32";
  descriptor.launch_plan.valid = true;
  descriptor.launch_plan.scalar_args = {6, 2, 1};
  descriptor.runtime_param_i64_metadata = {-1};
  descriptor.runtime_param_reduce_keep_dims = false;
  descriptor.runtime_param_reduce_keep_dims_valid = true;

  const auto reduce_info = runtime_reduce_info_from_descriptor(
      descriptor, ov::Shape{2, 3}, descriptor.stage_name);

  ASSERT_TRUE(reduce_info);
  EXPECT_EQ(reduce_info->axes, ov::AxisSet{1});
  EXPECT_FALSE(reduce_info->keep_dims);

  const auto dispatch = runtime_reduce_dispatch_from_descriptor(
      descriptor, descriptor.stage_name);
  ASSERT_TRUE(dispatch.valid());
  EXPECT_EQ(dispatch.entry_point, descriptor.entry_point);
  EXPECT_EQ(dispatch.op_code, 1u);
  EXPECT_EQ(dispatch.compiler_scalar_args, (std::vector<int32_t>{6, 2, 1}));

  descriptor.launch_plan.scalar_args = {6, 2};
  EXPECT_THROW((void)runtime_reduce_dispatch_from_descriptor(
                   descriptor, descriptor.stage_name),
               ov::Exception);

  descriptor.op_family = "Softmax";
  descriptor.runtime_param_payload_kind = RuntimeParamDescriptorPayloadKind::Softmax;
  EXPECT_FALSE(runtime_reduce_info_from_descriptor(descriptor, ov::Shape{2, 3},
                                                   descriptor.stage_name));
  EXPECT_FALSE(
      runtime_reduce_dispatch_from_descriptor(descriptor, descriptor.stage_name)
          .valid());
}


TEST_F(GfxBackendArchitectureContractTest,
       DescriptorOwnedRuntimeParamsMaterializeFromDescriptorShapes) {
  const std::vector<GfxKernelBufferRole> binary_roles = {
      GfxKernelBufferRole::TensorInput,   GfxKernelBufferRole::TensorInput,
      GfxKernelBufferRole::TensorOutput,  GfxKernelBufferRole::ScalarParam,
      GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
      GfxKernelBufferRole::RuntimeParams};

  for (const auto backend_domain : {"metal", "opencl"}) {
    SCOPED_TRACE(backend_domain);
    auto executable = make_source_payload_executable(
        backend_domain, "Add", binary_roles, {"{1,3}", "{1,3}"}, {"{1,3}"});
    const auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    const auto &stage = runtime_descriptor.stages.front();
    EXPECT_EQ(stage.runtime_param_buffer_count, 3u);
    EXPECT_EQ(stage.runtime_param_payload_kind,
              RuntimeParamDescriptorPayloadKind::BinaryBroadcast);
    ASSERT_TRUE(descriptor_owns_runtime_param_payload(stage));

    UnitMetadataBufferManager buffer_manager;
    RuntimeInputResolver inputs;
    inputs.descriptor = &stage;
    GpuTensor output;
    std::vector<GpuTensor *> outputs = {&output};
    const std::vector<int32_t> compiler_scalar_args;

    auto materialization = materialize_descriptor_owned_runtime_param_payload(
        buffer_manager, stage, inputs, outputs, compiler_scalar_args,
        "unit_descriptor_owned_add");

    ASSERT_TRUE(materialization.available);
    EXPECT_TRUE(materialization.descriptor_owned);
    EXPECT_EQ(materialization.extra_inputs.size(), 3u);
    EXPECT_EQ(output.shape, (ov::Shape{1, 3}));
    EXPECT_EQ(materialization.scalar_args, (std::vector<int32_t>{3, 2}));
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       DescriptorOwnedRuntimeParamsMaterializeDynamicBroadcastAndTileValues) {
  struct Case {
    const char *op_family;
    size_t runtime_param_count;
    std::vector<GfxKernelBufferRole> roles;
    std::vector<std::string> input_shapes;
    std::vector<std::string> output_shapes;
    ov::Shape data_shape;
    std::vector<int64_t> shape_values;
    ov::Shape expected_output_shape;
    std::vector<int32_t> expected_scalar_args;
  };

  const std::vector<Case> cases = {
      {"Broadcast",
       4,
       {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams},
       {"{?,3}"},
       {"{?,3}"},
       ov::Shape{1, 3},
       {2, 3},
       ov::Shape{2, 3},
       {6, 2, 2}},
      {"Tile",
       4,
       {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams},
       {"{?,3}"},
       {"{?,3}"},
       ov::Shape{2, 3},
       {2, 1},
       ov::Shape{4, 3},
       {12, 2}},
  };

  for (const auto &test_case : cases) {
    SCOPED_TRACE(test_case.op_family);
    for (const auto backend_domain : {"metal", "opencl"}) {
      SCOPED_TRACE(backend_domain);
      auto executable = make_source_payload_executable(
          backend_domain, test_case.op_family, test_case.roles,
          test_case.input_shapes, test_case.output_shapes);
      ASSERT_TRUE(executable.verify().valid());

      const auto runtime_descriptor =
          compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
      ASSERT_TRUE(compiler::runtime_executable_descriptor_valid(
          runtime_descriptor, executable));
      ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
      const auto &stage = runtime_descriptor.stages.front();
      ASSERT_TRUE(descriptor_owns_runtime_param_payload(stage));

      GpuTensor data;
      data.shape = test_case.data_shape;
      data.expected_type = ov::element::f32;
      GpuTensor shape_values;
      shape_values.shape = ov::Shape{test_case.shape_values.size()};
      shape_values.expected_type = ov::element::i64;
      shape_values.i64_values = test_case.shape_values;
      std::vector<GpuTensor *> input_ptrs = {&data, &shape_values};

      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &stage;

      UnitMetadataBufferManager buffer_manager;
      GpuTensor output;
      std::vector<GpuTensor *> outputs = {&output};
      const std::vector<int32_t> compiler_scalar_args;

      auto materialization = materialize_descriptor_owned_runtime_param_payload(
          buffer_manager, stage, inputs, outputs, compiler_scalar_args,
          test_case.op_family);

      ASSERT_TRUE(materialization.available);
      EXPECT_TRUE(materialization.descriptor_owned);
      EXPECT_EQ(materialization.extra_inputs.size(),
                test_case.runtime_param_count);
      EXPECT_EQ(output.shape, test_case.expected_output_shape);
      EXPECT_EQ(materialization.scalar_args, test_case.expected_scalar_args);
    }
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       DescriptorOwnedRuntimeParamsMaterializeFromArtifactMetadata) {
  struct Case {
    const char *op_family;
    size_t runtime_param_count;
    std::vector<GfxKernelBufferRole> roles;
    std::vector<std::string> input_shapes;
    std::vector<std::string> output_shapes;
    std::vector<int64_t> i64_metadata;
    std::vector<int64_t> runtime_shape_i64_metadata;
    bool reduce_keep_dims = false;
    bool reduce_keep_dims_valid = false;
    std::vector<int32_t> compiler_scalar_args;
    size_t expected_extra_inputs = 0;
    ov::Shape expected_output_shape;
    std::vector<ov::Shape> runtime_input_shapes;
  };

  const std::vector<Case> cases = {
      {"Softmax",
       1,
       {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
        GfxKernelBufferRole::RuntimeParams},
       {"{?,3}"},
       {"{?,3}"},
       {2, 3, 1},
       {},
       false,
       false,
       {},
       1,
       ov::Shape{2, 3},
       {ov::Shape{2, 3}}},
      {"Transpose",
       5,
       {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::RuntimeParams},
       {"{?,3}"},
       {"{3,?}"},
       {1, 0},
       {},
       false,
       false,
       {},
       5,
       ov::Shape{3, 2},
       {ov::Shape{2, 3}}},
      {"ReduceSum",
       5,
       {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
        GfxKernelBufferRole::ScalarParam, GfxKernelBufferRole::ScalarParam,
        GfxKernelBufferRole::ScalarParam, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams},
       {"{?,3}"},
       {"{?}"},
       {1},
       {},
       false,
       true,
       {2, 2, 0},
       5,
       ov::Shape{2},
       {ov::Shape{2, 3}}},
      {"Interpolate",
       1,
       {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::TensorOutput},
       {"{?,3,?,?}"},
       {"{1,3,4,4}"},
       {},
       {0, 1, 0},
       false,
       false,
       {},
       1,
       ov::Shape{1, 3, 4, 4},
       {ov::Shape{1, 3, 2, 2}}},
  };

  for (const auto &test_case : cases) {
    SCOPED_TRACE(test_case.op_family);
    GfxKernelSourceRuntimeBinding runtime_binding;
    runtime_binding.runtime_param_i64_metadata = test_case.i64_metadata;
    runtime_binding.runtime_param_reduce_keep_dims = test_case.reduce_keep_dims;
    runtime_binding.runtime_param_reduce_keep_dims_valid =
        test_case.reduce_keep_dims_valid;

    auto executable = make_source_payload_executable(
        "metal", test_case.op_family, test_case.roles, test_case.input_shapes,
        test_case.output_shapes, runtime_binding);
    if (!test_case.runtime_shape_i64_metadata.empty()) {
      install_runtime_shape_metadata(
          executable, test_case.runtime_shape_i64_metadata);
    }
    const auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    const auto &stage = runtime_descriptor.stages.front();
    EXPECT_EQ(stage.runtime_param_buffer_count, test_case.runtime_param_count);
    ASSERT_TRUE(descriptor_owns_runtime_param_payload(stage));

    UnitMetadataBufferManager buffer_manager;
    std::vector<GpuTensor> input_storage;
    input_storage.reserve(test_case.runtime_input_shapes.size());
    for (const auto &shape : test_case.runtime_input_shapes) {
      GpuTensor input;
      input.shape = shape;
      input.expected_type = ov::element::f32;
      input_storage.push_back(std::move(input));
    }
    std::vector<GpuTensor *> input_ptrs;
    input_ptrs.reserve(input_storage.size());
    for (auto &input : input_storage) {
      input_ptrs.push_back(&input);
    }
    RuntimeInputResolver inputs;
    inputs.inputs = &input_ptrs;
    inputs.descriptor = &stage;
    GpuTensor output;
    std::vector<GpuTensor *> outputs = {&output};

    auto materialization = materialize_descriptor_owned_runtime_param_payload(
        buffer_manager, stage, inputs, outputs,
        test_case.compiler_scalar_args, test_case.op_family);

    ASSERT_TRUE(materialization.available);
    EXPECT_TRUE(materialization.descriptor_owned);
    EXPECT_EQ(materialization.extra_inputs.size(),
              test_case.expected_extra_inputs);
    EXPECT_EQ(output.shape, test_case.expected_output_shape);
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       DescriptorOwnedReduceRuntimeParamsPreserveI64Values) {
  const std::vector<GfxKernelBufferRole> roles = {
      GfxKernelBufferRole::TensorInput,   GfxKernelBufferRole::TensorOutput,
      GfxKernelBufferRole::ScalarParam,   GfxKernelBufferRole::ScalarParam,
      GfxKernelBufferRole::ScalarParam,   GfxKernelBufferRole::RuntimeParams,
      GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
      GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams};

  GfxKernelSourceRuntimeBinding runtime_binding;
  runtime_binding.runtime_param_i64_metadata = {1};
  runtime_binding.runtime_param_reduce_keep_dims = false;
  runtime_binding.runtime_param_reduce_keep_dims_valid = true;
  runtime_binding.scalar_args = {2, 2, 0};

  for (const auto backend_domain : {"metal", "opencl"}) {
    SCOPED_TRACE(backend_domain);
    auto executable = make_source_payload_executable(
        backend_domain, "ReduceSum", roles, {"{2,3}"}, {"{2}"},
        runtime_binding);
    const auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    const auto &stage = runtime_descriptor.stages.front();
    EXPECT_EQ(stage.runtime_param_payload_kind,
              RuntimeParamDescriptorPayloadKind::Reduce);
    ASSERT_TRUE(descriptor_owns_runtime_param_payload(stage));

    GpuTensor input;
    input.shape = {2, 3};
    input.expected_type = ov::element::i64;
    input.i64_values = {1, 2, 3, 4, 5, 6};
    std::vector<GpuTensor *> input_ptrs = {&input};

    RuntimeInputResolver inputs;
    inputs.inputs = &input_ptrs;
    inputs.descriptor = &stage;

    UnitMetadataBufferManager buffer_manager;
    GpuTensor output;
    std::vector<GpuTensor *> outputs = {&output};
    auto materialization = materialize_descriptor_owned_runtime_param_payload(
        buffer_manager, stage, inputs, outputs, runtime_binding.scalar_args,
        "unit_reduce_i64_descriptor");

    ASSERT_TRUE(materialization.available);
    EXPECT_TRUE(materialization.descriptor_owned);
    EXPECT_EQ(materialization.extra_inputs.size(), 5u);
    EXPECT_EQ(materialization.scalar_args,
              (std::vector<int32_t>{2, 2, 0}));
    EXPECT_EQ(output.shape, (ov::Shape{2}));
    EXPECT_EQ(output.i64_values, (std::vector<int64_t>{6, 15}));
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       DescriptorOwnedRuntimeParamsRejectSourceNodeShapeBridge) {
  const std::vector<GfxKernelBufferRole> binary_roles = {
      GfxKernelBufferRole::TensorInput,   GfxKernelBufferRole::TensorInput,
      GfxKernelBufferRole::TensorOutput,  GfxKernelBufferRole::ScalarParam,
      GfxKernelBufferRole::ScalarParam,   GfxKernelBufferRole::RuntimeParams,
      GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams};

  for (const auto backend_domain : {"metal", "opencl"}) {
    SCOPED_TRACE(backend_domain);
    auto executable = make_source_payload_executable(
        backend_domain, "Add", binary_roles, {"{?,3}", "{1,3}"}, {"{1,3}"});
    const auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    const auto &stage = runtime_descriptor.stages.front();
    ASSERT_TRUE(descriptor_owns_runtime_param_payload(stage));

    UnitMetadataBufferManager buffer_manager;
    RuntimeInputResolver inputs;
    inputs.descriptor = &stage;
    GpuTensor output;
    std::vector<GpuTensor *> outputs = {&output};
    const std::vector<int32_t> compiler_scalar_args;

    EXPECT_THROW((void)materialize_descriptor_owned_runtime_param_payload(
                     buffer_manager, stage, inputs, outputs,
                     compiler_scalar_args, "unit_no_source_bridge_add"),
                 ov::Exception);
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       DescriptorOwnedRuntimeParamsMaterializeFromRequestTensorShapes) {
  const std::vector<GfxKernelBufferRole> binary_roles = {
      GfxKernelBufferRole::TensorInput,   GfxKernelBufferRole::TensorInput,
      GfxKernelBufferRole::TensorOutput,  GfxKernelBufferRole::ScalarParam,
      GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
      GfxKernelBufferRole::RuntimeParams};

  for (const auto backend_domain : {"metal", "opencl"}) {
    SCOPED_TRACE(backend_domain);
    auto executable = make_source_payload_executable(
        backend_domain, "Add", binary_roles, {"{?,3}", "{1,3}"}, {"{?,3}"});
    const auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    const auto &stage = runtime_descriptor.stages.front();
    ASSERT_TRUE(descriptor_owns_runtime_param_payload(stage));

    GpuTensor lhs;
    lhs.shape = {1, 3};
    GpuTensor rhs;
    rhs.shape = {1, 3};
    std::vector<GpuTensor *> input_tensors = {&lhs, &rhs};
    RuntimeInputResolver inputs;
    inputs.inputs = &input_tensors;
    inputs.descriptor = &stage;

    UnitMetadataBufferManager buffer_manager;
    GpuTensor output;
    std::vector<GpuTensor *> outputs = {&output};
    const std::vector<int32_t> compiler_scalar_args;

    auto materialization = materialize_descriptor_owned_runtime_param_payload(
        buffer_manager, stage, inputs, outputs, compiler_scalar_args,
        "unit_request_shape_owned_add");
    ASSERT_TRUE(materialization.available);
    EXPECT_TRUE(materialization.descriptor_owned);
    EXPECT_EQ(materialization.extra_inputs.size(), 3u);
    EXPECT_EQ(output.shape, (ov::Shape{1, 3}));
    EXPECT_EQ(materialization.scalar_args, (std::vector<int32_t>{3, 2}));
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeParamsWithDynamicDescriptorShapePassDescriptorValidation) {
  const std::vector<GfxKernelBufferRole> binary_roles = {
      GfxKernelBufferRole::TensorInput,   GfxKernelBufferRole::TensorInput,
      GfxKernelBufferRole::TensorOutput,  GfxKernelBufferRole::ScalarParam,
      GfxKernelBufferRole::ScalarParam,   GfxKernelBufferRole::RuntimeParams,
      GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams};

  auto executable = make_source_payload_executable(
      "opencl", "Add", binary_roles, {"{?,3}", "{1,3}"}, {"{1,3}"});
  ASSERT_TRUE(executable.verify().valid());

  const auto runtime_descriptor =
      compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
  const auto verification = compiler::verify_runtime_executable_descriptor(
      runtime_descriptor, executable);
  ASSERT_TRUE(verification.valid())
      << (verification.diagnostics.empty() ? std::string{}
                                           : verification.diagnostics.front());
  ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
  const auto &stage = runtime_descriptor.stages.front();
  EXPECT_TRUE(descriptor_owns_runtime_param_payload(stage));
}

} // namespace
} // namespace gfx_plugin
} // namespace ov
