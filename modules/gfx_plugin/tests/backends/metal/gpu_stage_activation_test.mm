// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#import <Metal/Metal.h>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "backends/metal/compiler/msl_codegen_apple_msl_activation.hpp"
#include "backends/metal/runtime/metal_command_encoder.hpp"
#include "backends/metal/runtime/metal_memory.hpp"
#include "backends/metal/runtime/stage_factory.hpp"
#include "kernel_ir/gfx_codegen_backend.hpp"
#include "kernel_ir/gfx_kernel_source.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/tanh.hpp"
#include "runtime/executable_descriptor.hpp"
#include "runtime/execution_dispatcher.hpp"

using namespace ov::gfx_plugin;

namespace {

struct ActivationCase {
  std::vector<float> in;
  std::vector<float> expected;
};

KernelLaunchPlanDescriptor
make_test_launch_plan_descriptor(const GfxKernelRuntimeBindingPlan &plan) {
  KernelLaunchPlanDescriptor descriptor;
  if (!plan.valid || !plan.stage_manifest.valid ||
      !plan.stage_manifest.custom_kernel.valid ||
      !plan.stage_manifest.custom_kernel.external_buffer_abi.valid) {
    return descriptor;
  }

  const auto roles = materialize_gfx_kernel_external_buffer_roles(
      plan.stage_manifest.custom_kernel.external_buffer_abi);
  if (roles.empty()) {
    return descriptor;
  }

  descriptor.valid = true;
  descriptor.buffer_roles.reserve(roles.size());
  for (const auto role : roles) {
    descriptor.buffer_roles.emplace_back(kernel_buffer_role_descriptor_name(role));
  }
  descriptor.input_indices = plan.runtime_binding.inputs;
  descriptor.input_arg_count = plan.runtime_binding.input_arg_count;
  descriptor.operand_kinds = plan.runtime_binding.operand_kinds;
  descriptor.operand_arg_indices = plan.runtime_binding.operand_arg_indices;
  descriptor.scalar_args = plan.runtime_binding.scalar_args;
  return descriptor;
}

RuntimeStageExecutableDescriptor
make_metal_test_descriptor(const std::shared_ptr<const ov::Node> &node) {
  auto source_plan = make_activation_msl_kernel_source_plan(node);
  OPENVINO_ASSERT(source_plan.valid(),
                  "GFX Metal activation test requires a compiler-owned MSL "
                  "source payload");
  auto msl_source = resolve_msl_source(source_plan.source);
  OPENVINO_ASSERT(!msl_source.empty(),
                  "GFX Metal activation test source payload is empty");

  RuntimeStageExecutableDescriptor descriptor;
  descriptor.stage_index = 0;
  descriptor.stage_record_key = 1;
  descriptor.artifact_descriptor_index = 0;
  descriptor.manifest_ref = "test_manifest";
  descriptor.abi_fingerprint = "test_abi";
  descriptor.artifact_key = "test_artifact";
  descriptor.backend_domain = "metal";
  descriptor.kernel_id = "metal/generated/activation";
  descriptor.op_family = node ? node->get_type_name() : "null";
  descriptor.stage_name = node ? node->get_friendly_name() : "null";
  descriptor.origin = KernelArtifactOrigin::Generated;
  descriptor.payload_kind = KernelArtifactPayloadKind::MslSource;
  descriptor.entry_point = source_plan.source.entry_point;
  descriptor.abi_arg_count = source_plan.source.signature.arg_count;
  descriptor.abi_output_arg_count =
      source_plan.source.signature.output_arg_count;
  descriptor.launch_plan = make_test_launch_plan_descriptor(source_plan.binding);
  descriptor.dispatch_contract = "manifest";
  descriptor.payload = std::make_shared<GfxKernelSourcePayload>(
      descriptor.kernel_id, descriptor.backend_domain, descriptor.entry_point,
      GfxKernelSourceLanguage::MetalShadingLanguage, std::move(msl_source),
      source_plan.binding.stage_manifest);
  return descriptor;
}

RuntimeStageExecutableDescriptor make_descriptor_without_payload(
    const std::shared_ptr<const ov::Node> &node) {
  RuntimeStageExecutableDescriptor descriptor;
  descriptor.stage_index = 0;
  descriptor.stage_record_key = 1;
  descriptor.artifact_descriptor_index = 0;
  descriptor.manifest_ref = "test_manifest";
  descriptor.abi_fingerprint = "test_abi";
  descriptor.artifact_key = "test_artifact";
  descriptor.backend_domain = "metal";
  descriptor.kernel_id =
      std::string("metal/generated/") + (node ? node->get_type_name() : "null");
  descriptor.op_family = node ? node->get_type_name() : "null";
  descriptor.stage_name = node ? node->get_friendly_name() : "null";
  descriptor.origin = KernelArtifactOrigin::Generated;
  descriptor.payload_kind = KernelArtifactPayloadKind::None;
  descriptor.entry_point = descriptor.kernel_id;
  descriptor.dispatch_contract = "manifest";
  return descriptor;
}

template <typename OpFactory> void run_activation(const ActivationCase &tc) {
  ensure_metal_stage_factory_registered();
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  ASSERT_NE(device, nil);
  MetalCommandQueueHandle gfx_queue = metal_create_command_queue(device);
  ASSERT_NE(gfx_queue, nullptr);
  id<MTLCommandQueue> queue = static_cast<id<MTLCommandQueue>>(gfx_queue);

  MetalDeviceCaps caps = query_metal_device_caps(device);
  MetalAllocatorCore core(device, caps);
  MetalHeapPool heaps(core);
  MetalFreeList freelist;
  MetalStagingPool staging(core);
  MetalAllocator allocator(core, heaps, freelist, staging, caps);
  MetalConstCache const_cache(allocator, gfx_queue);
  MetalBufferManager mgr(core, &const_cache);
  MetalBufferManager::set_current_allocator(&allocator);

  // Prepare tensors
  GpuTensor input{};
  input.shape = {tc.in.size()};
  input.expected_type = ov::element::f32;
  input.buf = mgr.wrap_shared(tc.in.data(), tc.in.size() * sizeof(float),
                              ov::element::f32);

  GpuTensor output{};
  output.shape = {tc.in.size()};
  output.expected_type = ov::element::f32;
  output.prefer_private = false;
  const size_t out_bytes = tc.in.size() * sizeof(float);
  GpuBufferDesc out_desc{};
  out_desc.bytes = out_bytes;
  out_desc.type = ov::element::f32;
  out_desc.usage = BufferUsage::IO;
  out_desc.cpu_read = true;
  out_desc.prefer_device_local = false;
  output.buf = mgr.allocate(out_desc);
  ASSERT_TRUE(output.buf.valid());

  auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{tc.in.size()});
  auto node = OpFactory::make_node(param);
  const auto descriptor = make_metal_test_descriptor(node);
  auto stage = GpuStageFactory::create(
      RuntimeStageMaterializationContext{descriptor}, GpuBackend::Metal,
      device, queue);
  ASSERT_NE(stage, nullptr);
  stage->set_inputs({&input});
  stage->set_output(&output);
  stage->init(&mgr);
  stage->prepare_runtime_handle(&mgr);
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  stage->execute(cb);
  metal_end_compute_encoder((GpuCommandBufferHandle)cb);
  [cb commit];
  [cb waitUntilCompleted];

  MetalBufferManager::set_current_allocator(nullptr);
  metal_release_command_queue(gfx_queue);

  id<MTLBuffer> out_buf = static_cast<id<MTLBuffer>>(output.buf.buffer);
  ASSERT_NE(out_buf, nil);
  auto *data = static_cast<const float *>([out_buf contents]);
  ASSERT_NE(data, nullptr);
  ASSERT_EQ(output.shape.size(), 1u);
  ASSERT_EQ(output.shape[0], tc.expected.size());
  for (size_t i = 0; i < tc.expected.size(); ++i) {
    EXPECT_NEAR(data[i], tc.expected[i], 1e-4f) << "idx=" << i;
  }
}

struct ReluFactory {
  static std::shared_ptr<ov::Node>
  make_node(const std::shared_ptr<ov::Node> &p) {
    return std::make_shared<ov::op::v0::Relu>(p);
  }
};
struct SigmoidFactory {
  static std::shared_ptr<ov::Node>
  make_node(const std::shared_ptr<ov::Node> &p) {
    return std::make_shared<ov::op::v0::Sigmoid>(p);
  }
};
struct TanhFactory {
  static std::shared_ptr<ov::Node>
  make_node(const std::shared_ptr<ov::Node> &p) {
    return std::make_shared<ov::op::v0::Tanh>(p);
  }
};

} // namespace

TEST(GpuStageActivation, Relu) {
  ActivationCase tc{{-1.f, 0.f, 2.f}, {0.f, 0.f, 2.f}};
  run_activation<ReluFactory>(tc);
}

TEST(GpuStageActivation, Sigmoid) {
  ActivationCase tc{
      {0.f, 2.f, -2.f},
      {0.5f, 1.f / (1.f + std::exp(-2.f)), 1.f / (1.f + std::exp(2.f))}};
  run_activation<SigmoidFactory>(tc);
}

TEST(GpuStageActivation, Tanh) {
  ActivationCase tc{{0.f, 1.f, -1.f}, {0.f, std::tanh(1.f), std::tanh(-1.f)}};
  run_activation<TanhFactory>(tc);
}

TEST(GpuStageActivation, RejectsRuntimeSourcePlanWithoutCompilerPayload) {
  ensure_metal_stage_factory_registered();
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  ASSERT_NE(device, nil);
  MetalCommandQueueHandle gfx_queue = metal_create_command_queue(device);
  ASSERT_NE(gfx_queue, nullptr);
  id<MTLCommandQueue> queue = static_cast<id<MTLCommandQueue>>(gfx_queue);

  MetalDeviceCaps caps = query_metal_device_caps(device);
  MetalAllocatorCore core(device, caps);
  MetalHeapPool heaps(core);
  MetalFreeList freelist;
  MetalStagingPool staging(core);
  MetalAllocator allocator(core, heaps, freelist, staging, caps);
  MetalConstCache const_cache(allocator, gfx_queue);
  MetalBufferManager mgr(core, &const_cache);
  MetalBufferManager::set_current_allocator(&allocator);

  std::vector<float> values{1.0f};
  GpuTensor input{};
  input.shape = {values.size()};
  input.expected_type = ov::element::f32;
  input.buf = mgr.wrap_shared(values.data(), values.size() * sizeof(float),
                              ov::element::f32);

  GpuTensor output{};
  output.shape = {values.size()};
  output.expected_type = ov::element::f32;
  output.prefer_private = false;
  GpuBufferDesc out_desc{};
  out_desc.bytes = values.size() * sizeof(float);
  out_desc.type = ov::element::f32;
  out_desc.usage = BufferUsage::IO;
  out_desc.cpu_read = true;
  out_desc.prefer_device_local = false;
  output.buf = mgr.allocate(out_desc);
  ASSERT_TRUE(output.buf.valid());

  auto param = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f32, ov::Shape{values.size()});
  auto node = std::make_shared<ov::op::v0::Relu>(param);
  const auto descriptor = make_descriptor_without_payload(node);
  auto stage = GpuStageFactory::create(
      RuntimeStageMaterializationContext{descriptor}, GpuBackend::Metal,
      device, queue);
  ASSERT_NE(stage, nullptr);
  stage->set_inputs({&input});
  stage->set_output(&output);
  stage->init(&mgr);

  EXPECT_THROW(stage->prepare_runtime_handle(&mgr), ov::Exception);

  MetalBufferManager::set_current_allocator(nullptr);
  metal_release_command_queue(gfx_queue);
}
