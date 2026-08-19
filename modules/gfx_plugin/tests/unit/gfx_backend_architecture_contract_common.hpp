// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "backends/metal/compiler/apple_vendor_descriptors.hpp"
#include "backends/metal/compiler/metal_kernel_artifacts.hpp"
#include "backends/metal/compiler/metal_operation_support.hpp"
#include "backends/opencl/compiler/opencl_kernel_artifacts.hpp"
#include "backends/opencl/compiler/opencl_operation_support.hpp"
#include "common/gpu_backend.hpp"
#include "common/gpu_device_profile.hpp"
#include "compiler/cache_envelope.hpp"
#include "compiler/executable_bundle.hpp"
#include "compiler/memory_plan.hpp"
#include "compiler/pipeline_stage_fusion.hpp"
#include "compiler/pipeline_stage_graph_snapshot.hpp"
#include "compiler/pipeline_stage_materialization_draft.hpp"
#include "compiler/pipeline_stage_plan.hpp"
#include "compiler/pipeline_stage_runtime_descriptor_builder_detail.hpp"
#include "compiler/runtime_executable_descriptor_builder.hpp"
#include "compiler/tensor_layout.hpp"
#include "kernel_ir/gfx_kernel_source.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "mlir/gfx_stage_kernel_binding.hpp"
#include "mlir/mlir_stage_runtime_value_bridge.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/transpose.hpp"
#include "runtime/backend_request_state.hpp"
#include "runtime/backend_runtime.hpp"
#include "runtime/backend_runtime_provider.hpp"
#include "runtime/backend_stage_factory.hpp"
#include "runtime/descriptor_const_tensor_materializer.hpp"
#include "runtime/executable_descriptor.hpp"
#include "runtime/execution_dispatcher.hpp"
#include "runtime/gfx_stage_runtime_values.hpp"
#include "runtime/gpu_buffer.hpp"
#include "runtime/gpu_buffer_manager.hpp"
#include "runtime/gpu_memory_ops.hpp"
#include "runtime/kernel_launch_plan.hpp"
#include "runtime/pipeline_stage_materializer.hpp"
#include "runtime/pipeline_stage_plan.hpp"
#include "runtime/runtime_session.hpp"
#include "runtime/runtime_shape_materializer.hpp"
#include "transforms/pipeline.hpp"

#include "unit/gfx_backend_contracts.hpp"

#include "runtime/stateful_stage.hpp"
#include "runtime/tensor_binding_contract.hpp"
#include "runtime/view_only_stage.hpp"

namespace ov {
namespace gfx_plugin {

class GfxBackendArchitectureContractTest : public ::testing::Test {
protected:
  test::ModelContractFactory models;
};

namespace {

using compiler::KernelUnitKind;
using compiler::LoweringRouteKind;

bool has_diagnostic_containing(const std::vector<std::string> &diagnostics,
                               std::string_view needle) {
  for (const auto &diagnostic : diagnostics) {
    if (diagnostic.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

bool launch_plan_contract_equal(const KernelLaunchPlanDescriptor &lhs,
                                const KernelLaunchPlanDescriptor &rhs) {
  return lhs.valid == rhs.valid && lhs.buffer_roles == rhs.buffer_roles &&
         lhs.direct_input_indices == rhs.direct_input_indices &&
         lhs.input_indices == rhs.input_indices &&
         lhs.input_arg_count == rhs.input_arg_count &&
         lhs.operand_kinds == rhs.operand_kinds &&
         lhs.operand_arg_indices == rhs.operand_arg_indices &&
         lhs.scalar_args == rhs.scalar_args &&
         lhs.scalar_arg_kinds == rhs.scalar_arg_kinds;
}

void expect_artifact_descriptor_abi_equal(
    const compiler::KernelArtifactDescriptor &actual,
    const compiler::KernelArtifactDescriptor &expected) {
  EXPECT_EQ(actual.manifest_ref, expected.manifest_ref);
  EXPECT_EQ(actual.abi_fingerprint, expected.abi_fingerprint);
  EXPECT_EQ(actual.artifact_key, expected.artifact_key);
  EXPECT_EQ(actual.entry_point, expected.entry_point);
  EXPECT_EQ(actual.compile_options_key, expected.compile_options_key);
  EXPECT_EQ(actual.abi_arg_count, expected.abi_arg_count);
  EXPECT_EQ(actual.abi_output_arg_count, expected.abi_output_arg_count);
  EXPECT_EQ(actual.runtime_param_buffer_count,
            expected.runtime_param_buffer_count);
  EXPECT_EQ(actual.runtime_param_payload_kind,
            expected.runtime_param_payload_kind);
  EXPECT_EQ(actual.runtime_param_i64_metadata,
            expected.runtime_param_i64_metadata);
  EXPECT_EQ(actual.runtime_param_reduce_keep_dims,
            expected.runtime_param_reduce_keep_dims);
  EXPECT_EQ(actual.runtime_param_reduce_keep_dims_valid,
            expected.runtime_param_reduce_keep_dims_valid);
  EXPECT_TRUE(
      launch_plan_contract_equal(actual.launch_plan, expected.launch_plan));
}

GpuTensor make_test_launch_plan_tensor(uint64_t allocation_uid) {
  GpuTensor tensor;
  tensor.buf.buffer = reinterpret_cast<GpuBufferHandle>(
      static_cast<uintptr_t>(0x1000u + allocation_uid));
  tensor.buf.size = 64;
  tensor.buf.allocation_uid = allocation_uid;
  tensor.shape = ov::Shape{16};
  tensor.expected_type = ov::element::f32;
  return tensor;
}

RuntimeTensorBindingContract make_runtime_binding(std::string logical_name,
                                                  std::string region_id,
                                                  std::string role) {
  RuntimeTensorBindingContract binding;
  binding.logical_name = std::move(logical_name);
  binding.memory_region_id = std::move(region_id);
  binding.role = std::move(role);
  binding.element_type = "f32";
  binding.partial_shape = "{1}";
  binding.layout = "logical";
  binding.storage_kind = "device_buffer";
  binding.lifetime_class = "unit_lifetime";
  binding.alias_group = binding.memory_region_id;
  return binding;
}

} // namespace
} // namespace gfx_plugin
} // namespace ov
