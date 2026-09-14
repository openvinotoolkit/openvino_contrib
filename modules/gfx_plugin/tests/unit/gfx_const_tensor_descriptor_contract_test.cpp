// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unit/gfx_const_tensor_contract_utils.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

TEST_F(GfxBackendArchitectureContractTest,
       ConstTensorSourcePayloadsAreDescriptorOwned) {
  struct Case {
    const char *backend_domain;
  };

  const std::vector<GfxKernelBufferRole> const_roles = {
      GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::ConstTensor,
      GfxKernelBufferRole::TensorOutput};

  for (const auto test_case : {Case{"metal"}, Case{"opencl"}}) {
    SCOPED_TRACE(test_case.backend_domain);
    KernelArtifactConstTensor const_tensor;
    const_tensor.source_input_index = 1;
    const_tensor.logical_name = "unit_const_input";
    const_tensor.element_type = "f32";
    const_tensor.shape = {1, 3};
    const_tensor.bytes = {0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 64};

    auto executable = make_source_payload_executable(
        test_case.backend_domain, "Multiply", const_roles, {"{1,3}", "{1,3}"},
        {"{1,3}"}, {}, {const_tensor});
    ASSERT_TRUE(executable.verify().valid());

    const auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_TRUE(compiler::runtime_executable_descriptor_valid(
        runtime_descriptor, executable));
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    const auto &stage = runtime_descriptor.stages.front();
    ASSERT_EQ(stage.const_tensors.size(), 1u);
    EXPECT_EQ(stage.const_tensors.front().source_input_index, 1u);
    EXPECT_EQ(stage.const_tensors.front().logical_name, "unit_const_input");
    EXPECT_EQ(stage.const_tensors.front().element_type, "f32");
    EXPECT_EQ(stage.const_tensors.front().shape, (std::vector<size_t>{1, 3}));
    EXPECT_EQ(stage.const_tensors.front().bytes, const_tensor.bytes);

    auto missing_const_executable = make_source_payload_executable(
        test_case.backend_domain, "Multiply", const_roles, {"{1,3}", "{1,3}"},
        {"{1,3}"});
    const auto missing_const_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(
            missing_const_executable);
    const auto missing_const_verification =
        compiler::verify_runtime_executable_descriptor(
            missing_const_descriptor, missing_const_executable);
    EXPECT_FALSE(missing_const_verification.valid());
    EXPECT_TRUE(has_diagnostic_containing(
        missing_const_verification.diagnostics, "ConstTensor ABI"));
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       DescriptorConstTensorMaterializerUsesSharedDescriptorSlots) {
  RuntimeStageExecutableDescriptor descriptor;
  descriptor.stage_name = "unit_descriptor_const_stage";
  descriptor.kernel_id = "unit/descriptor_const";
  descriptor.const_tensors = {
      KernelArtifactConstTensor{3, "rhs_const", "f32", {1}, {0, 0, 0, 64}},
      KernelArtifactConstTensor{1, "lhs_const", "f32", {1}, {0, 0, 128, 63}},
  };

  UnitDescriptorConstBufferManager buffer_manager;
  auto slots = materialize_descriptor_const_tensor_slots(
      buffer_manager, descriptor, "unit/const_tensor");

  ASSERT_EQ(slots.buffers.size(), 4u);
  ASSERT_EQ(slots.present.size(), 4u);
  EXPECT_FALSE(slots.present[0]);
  EXPECT_TRUE(slots.present[1]);
  EXPECT_FALSE(slots.present[2]);
  EXPECT_TRUE(slots.present[3]);
  EXPECT_EQ(slots.buffers[1].shape, (ov::Shape{1}));
  EXPECT_EQ(slots.buffers[3].shape, (ov::Shape{1}));
  EXPECT_EQ(slots.buffers[1].expected_type, ov::element::f32);
  EXPECT_EQ(slots.buffers[3].expected_type, ov::element::f32);
  EXPECT_TRUE(slots.buffers[1].buf.valid());
  EXPECT_TRUE(slots.buffers[3].buf.valid());

  auto args = descriptor_const_tensor_args(slots, 2);
  ASSERT_EQ(args.size(), 2u);
  EXPECT_EQ(args[0], &slots.buffers[1]);
  EXPECT_EQ(args[1], &slots.buffers[3]);
  EXPECT_NE(args[0]->buf.allocation_uid, args[1]->buf.allocation_uid);

  ASSERT_EQ(buffer_manager.uploads.size(), 2u);
  EXPECT_EQ(buffer_manager.uploads[0].bytes,
            (std::vector<uint8_t>{0, 0, 0, 64}));
  EXPECT_EQ(buffer_manager.uploads[1].bytes,
            (std::vector<uint8_t>{0, 0, 128, 63}));
  EXPECT_NE(buffer_manager.uploads[0].key, buffer_manager.uploads[1].key);
}

TEST_F(GfxBackendArchitectureContractTest,
       DescriptorConstTensorMaterializerRejectsInvalidSharedContracts) {
  RuntimeStageExecutableDescriptor duplicate_descriptor;
  duplicate_descriptor.stage_name = "unit_duplicate_const_stage";
  duplicate_descriptor.kernel_id = "unit/duplicate_const";
  duplicate_descriptor.const_tensors = {
      KernelArtifactConstTensor{1, "const_a", "f32", {1}, {0, 0, 128, 63}},
      KernelArtifactConstTensor{1, "const_b", "f32", {1}, {0, 0, 0, 64}},
  };

  UnitDescriptorConstBufferManager buffer_manager;
  EXPECT_THROW((void)materialize_descriptor_const_tensor_slots(
                   buffer_manager, duplicate_descriptor, "unit/const_tensor"),
               ov::Exception);

  RuntimeStageExecutableDescriptor cache_descriptor;
  cache_descriptor.stage_name = "unit_no_const_cache_stage";
  cache_descriptor.kernel_id = "unit/no_const_cache";
  cache_descriptor.const_tensors = {
      KernelArtifactConstTensor{0, "const_input", "f32", {1}, {0, 0, 128, 63}},
  };

  GpuBufferManager no_const_cache_manager;
  EXPECT_THROW(
      (void)materialize_descriptor_const_tensor_slots(
          no_const_cache_manager, cache_descriptor, "unit/const_tensor"),
      ov::Exception);
}

} // namespace
} // namespace gfx_plugin
} // namespace ov
