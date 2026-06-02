// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/metal_executor.hpp"

#include <chrono>

#include "backends/metal/codegen/metal_codegen_backend.hpp"
#include "backends/metal/runtime/metal_memory.hpp"
#include "backends/metal/runtime/metal_runtime_kernel_loader.hpp"
#include "backends/metal/runtime/profiling/profiler.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {

MetalStage::MetalStage(const std::shared_ptr<const ov::Node> &node,
                       MetalDeviceHandle device, MetalCommandQueueHandle queue,
                       const RuntimeStageExecutableDescriptor *descriptor)
    : MlirStage(node, descriptor), m_device(device), m_queue(queue) {
  if (descriptor) {
    m_executable_descriptor =
        std::make_shared<RuntimeStageExecutableDescriptor>(*descriptor);
  }
}

namespace {

inline bool is_metal_descriptor_domain(std::string_view domain) {
  return domain == "metal" || domain == "apple_msl" ||
         domain == "apple_mps" || domain == "common";
}

} // namespace

void MetalStage::init(GpuBufferManager *buffer_manager) {
  MlirStage::init(buffer_manager);
  if (!m_device) {
    if (auto *metal_mgr = dynamic_cast<MetalBufferManager *>(buffer_manager)) {
      m_device = metal_mgr->device();
    }
  }
}

void MetalStage::compile(GpuBufferManager *buffer_manager) {
  MlirStage::compile(buffer_manager);
}

void MetalStage::execute(GpuCommandBufferHandle command_buffer) {
  MlirStage::execute(command_buffer);
}

void MetalStage::set_inputs(const std::vector<GpuTensor *> &inputs) {
  MlirStage::set_inputs(inputs);
}

void MetalStage::set_output(GpuTensor *output) {
  MlirStage::set_output(output);
}

void MetalStage::set_outputs(
    const std::vector<std::unique_ptr<GpuTensor>> &outputs) {
  MlirStage::set_outputs(outputs);
}

bool MetalStage::fuse_activation(ActivationKind kind, float alpha) {
  return MlirStage::fuse_activation(kind, alpha);
}

bool MetalStage::fuse_batchnorm(const BatchNormParams &params) {
  return MlirStage::fuse_batchnorm(params);
}

bool MetalStage::fuse_bias(const BiasParams &params) {
  return MlirStage::fuse_bias(params);
}

void MetalStage::enable_profiling(bool enable) {
  MlirStage::enable_profiling(enable);
}

void MetalStage::set_profiler(void *profiler, uint32_t node_id,
                              const std::string &node_name,
                              const std::string &node_type) {
  MlirStage::set_profiler(profiler, node_id, node_name, node_type);
}

std::unique_ptr<GpuStage> MetalStage::clone() const {
  auto stage = std::make_unique<MetalStage>(
      m_node, m_device, m_queue, m_executable_descriptor.get());
  clone_into(*stage);
  return stage;
}

std::shared_ptr<ICompiledKernel>
MetalStage::compile_kernel(const KernelSource &source, std::string *log) {
  OPENVINO_ASSERT(m_device, "MetalStage: Metal device handle is null");
  (void)source;
  OPENVINO_ASSERT(m_executable_descriptor,
                  "MetalStage: compiler-owned runtime descriptor is required");
  OPENVINO_ASSERT(is_metal_descriptor_domain(m_executable_descriptor->backend_domain),
                  "MetalStage: runtime descriptor backend domain mismatch: ",
                  m_executable_descriptor->backend_domain);
  OPENVINO_ASSERT(
      m_executable_descriptor->payload_kind !=
          compiler::KernelArtifactPayloadKind::OpenClSource,
      "MetalStage: OpenCL source payload cannot execute on Metal");
  OPENVINO_ASSERT(
      m_executable_descriptor->payload_kind ==
          compiler::KernelArtifactPayloadKind::MslSource,
      "MetalStage: compiler-owned MSL source payload is required for Metal "
      "custom kernel execution; runtime source-plan generation is forbidden");
  OPENVINO_ASSERT(m_executable_descriptor->payload,
                  "MetalStage: MSL source descriptor is missing payload");

  MetalCodegenBackend backend(m_device);
  auto descriptor_source =
      MetalRuntimeKernelLoader::load_msl_source(*m_executable_descriptor);
  m_last_compiled_kernel_entry_point = descriptor_source.entry_point;
  return backend.compile(descriptor_source, log);
}

KernelExecutionHooks *
MetalStage::prepare_profiling(ProfileState &state,
                              KernelExecutionHooks &hooks) {
  auto *profiler = static_cast<MetalProfiler *>(profiler_handle());
  if (!profiler) {
    return nullptr;
  }
  state.cpu_start = std::chrono::steady_clock::now();
  const char *node_name = profile_node_name().empty()
                              ? name().c_str()
                              : profile_node_name().c_str();
  const char *node_type = profile_node_type().empty()
                              ? type().c_str()
                              : profile_node_type().c_str();
  hooks.stage_name = node_name;
  hooks.stage_type = node_type;
  profiler->begin_node(profile_node_id(), node_name, node_type, "GFX");
  hooks.on_begin = [profiler, &state](GpuCommandEncoderHandle enc) {
    state.sample_begin =
        profiler->gpu_sample_begin(static_cast<MetalCommandEncoderHandle>(enc));
  };
  hooks.on_end = [profiler, &state](GpuCommandEncoderHandle enc) {
    state.sample_end =
        profiler->gpu_sample_end(static_cast<MetalCommandEncoderHandle>(enc));
  };
  hooks.on_counter = [profiler](std::string_view name, uint64_t delta) {
    profiler->increment_counter(name, delta);
  };
  hooks.on_segment =
      [profiler](std::string_view phase, std::string_view name,
                 std::chrono::microseconds cpu_us, uint64_t gpu_us,
                 uint32_t dispatches, uint64_t bytes_in, uint64_t bytes_out,
                 uint64_t macs_est, uint64_t flops_est, int64_t inflight_slot,
                 uint64_t queue_id, uint64_t cmd_buffer_id) {
        profiler->record_segment(phase, name, cpu_us, gpu_us, dispatches,
                                 bytes_in, bytes_out, macs_est, flops_est,
                                 inflight_slot, queue_id, cmd_buffer_id);
      };
  return &hooks;
}

void MetalStage::finalize_profiling(const ProfileState &state) {
  auto *profiler = static_cast<MetalProfiler *>(profiler_handle());
  if (!profiler) {
    return;
  }
  const auto cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - state.cpu_start);
  profiler->end_node(profile_node_id(), cpu_us, state.sample_begin,
                     state.sample_end);
}

} // namespace gfx_plugin
} // namespace ov
