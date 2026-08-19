// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/metal_executor.hpp"

#include <algorithm>
#include <chrono>
#include <sstream>
#include <utility>

#include "backends/metal/codegen/metal_codegen_backend.hpp"
#include "backends/metal/runtime/metal_memory.hpp"
#include "backends/metal/runtime/metal_runtime_kernel_loader.hpp"
#include "backends/metal/runtime/profiling/profiler.hpp"
#include "kernel_ir/gfx_kernel_args.hpp"
#include "kernel_ir/gfx_kernel_cache.hpp"
#include "kernel_ir/gfx_kernel_dispatch.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "runtime/descriptor_const_tensor_materializer.hpp"
#include "runtime/gfx_kernel_runtime_params.hpp"
#include "runtime/gfx_shape_utils.hpp"
#include "runtime/gfx_stage_runtime_values.hpp"
#include "runtime/kernel_launch_plan.hpp"
#include "runtime/tensor_binding_contract.hpp"

namespace ov {
namespace gfx_plugin {

MetalStage::MetalStage(const RuntimeStageExecutableDescriptor &descriptor,
                       MetalDeviceHandle device, MetalCommandQueueHandle queue)
    : m_device(device), m_queue(queue), m_descriptor(descriptor) {
  OPENVINO_ASSERT(!m_descriptor.stage_name.empty(),
                  "MetalStage: compiler-owned descriptor must provide stage "
                  "name for MSL source execution");
  OPENVINO_ASSERT(!m_descriptor.op_family.empty(),
                  "MetalStage: compiler-owned descriptor must provide op "
                  "family for MSL source execution");
  m_name = m_descriptor.stage_name;
  m_type = m_descriptor.op_family;
}

namespace {

inline bool is_metal_descriptor_domain(std::string_view domain) {
  return domain == "metal" || domain == "apple_msl" || domain == "apple_mps" ||
         domain == "common";
}

constexpr size_t kDefaultMetalSourceThreadsPerGroup = 64;

bool descriptor_output_shape(const RuntimeStageExecutableDescriptor &descriptor,
                             size_t output_idx, ov::Shape &shape) {
  if (output_idx >= descriptor.output_bindings.size()) {
    return false;
  }
  return parse_static_shape_contract(
      descriptor.output_bindings[output_idx].partial_shape, shape);
}

} // namespace

void MetalStage::init(GpuBufferManager *buffer_manager) {
  m_buffer_manager = buffer_manager;
  if (!m_device) {
    if (auto *metal_mgr = dynamic_cast<MetalBufferManager *>(buffer_manager)) {
      m_device = metal_mgr->device();
    }
  }
}

void MetalStage::prepare_runtime_handle(GpuBufferManager *buffer_manager) {
  if (buffer_manager) {
    init(buffer_manager);
  }
  ensure_prepared();
}

struct MetalStage::ProfileState {
  std::chrono::steady_clock::time_point cpu_start{};
  int32_t sample_begin = -1;
  int32_t sample_end = -1;
};

void MetalStage::execute(GpuCommandBufferHandle command_buffer) {
  OPENVINO_ASSERT(m_kernel, "MetalStage: runtime handle is not prepared for ",
                  m_name);
  const auto outputs = resolve_outputs();
  OPENVINO_ASSERT(!outputs.empty(),
                  "MetalStage: output tensor is not bound for ", m_name);
  auto args = materialize_source_args(outputs);
  const auto dispatch = make_source_dispatch(outputs);

  ProfileState profile_state{};
  KernelExecutionHooks hooks;
  KernelExecutionHooks *hooks_ptr = nullptr;
  if (m_profiling_enabled && m_profiler) {
    hooks_ptr = prepare_profiling(profile_state, hooks);
  }

  m_kernel->execute(command_buffer, dispatch, args, hooks_ptr);
  if (hooks_ptr) {
    finalize_profiling(profile_state);
  }
}

void MetalStage::set_inputs(const std::vector<GpuTensor *> &inputs) {
  m_inputs = inputs;
}

void MetalStage::set_output(GpuTensor *output) {
  m_output = output;
  m_outputs.clear();
  if (output) {
    m_outputs.push_back(output);
  }
}

void MetalStage::set_outputs(
    const std::vector<std::unique_ptr<GpuTensor>> &outputs) {
  std::vector<GpuTensor *> refs;
  refs.reserve(outputs.size());
  for (const auto &output : outputs) {
    refs.push_back(output.get());
  }
  set_output_refs(refs);
}

void MetalStage::set_output_refs(const std::vector<GpuTensor *> &outputs) {
  m_outputs = outputs;
  m_output = m_outputs.empty() ? nullptr : m_outputs.front();
}

bool MetalStage::fuse_activation(ActivationKind kind, float alpha) {
  (void)kind;
  (void)alpha;
  return false;
}

bool MetalStage::fuse_input_activation(size_t input_idx, ActivationKind kind,
                                       float alpha) {
  (void)input_idx;
  (void)kind;
  (void)alpha;
  return false;
}

bool MetalStage::fuse_residual_add() { return false; }

bool MetalStage::fuse_batchnorm(const BatchNormParams &params) {
  (void)params;
  return false;
}

bool MetalStage::fuse_bias(const BiasParams &params) {
  (void)params;
  return false;
}

void MetalStage::enable_profiling(bool enable) { m_profiling_enabled = enable; }

void MetalStage::set_profiler(void *profiler, uint32_t node_id,
                              const std::string &node_name,
                              const std::string &node_type) {
  m_profiler = profiler;
  m_profile_node_id = node_id;
  m_profile_node_name = node_name;
  m_profile_node_type = node_type;
}

std::unique_ptr<GpuStage> MetalStage::clone() const {
  auto stage = std::make_unique<MetalStage>(m_descriptor, m_device, m_queue);
  stage->m_name = m_name;
  stage->m_type = m_type;
  stage->m_profiling_enabled = m_profiling_enabled;
  stage->m_profiler = m_profiler;
  stage->m_profile_node_id = m_profile_node_id;
  stage->m_profile_node_name = m_profile_node_name;
  stage->m_profile_node_type = m_profile_node_type;
  stage->m_const_buffers = m_const_buffers;
  stage->m_kernel_extra_inputs = m_kernel_extra_inputs;
  if (m_kernel) {
    stage->m_kernel = m_kernel->fork();
  }
  return stage;
}

void MetalStage::ensure_prepared() {
  if (m_kernel) {
    return;
  }
  OPENVINO_ASSERT(m_device, "MetalStage: Metal device handle is null");
  OPENVINO_ASSERT(is_metal_descriptor_domain(m_descriptor.backend_domain),
                  "MetalStage: runtime descriptor backend domain mismatch: ",
                  m_descriptor.backend_domain);
  OPENVINO_ASSERT(m_descriptor.payload_kind !=
                      KernelArtifactPayloadKind::OpenClSource,
                  "MetalStage: OpenCL source payload cannot execute on Metal");
  OPENVINO_ASSERT(
      m_descriptor.payload_kind == KernelArtifactPayloadKind::MslSource,
      "MetalStage: compiler-owned MSL source payload is required for Metal "
      "custom kernel execution; runtime source-plan generation is forbidden");
  OPENVINO_ASSERT(m_descriptor.payload,
                  "MetalStage: MSL source descriptor is missing payload");

  MetalCodegenBackend backend(m_device);
  auto descriptor_source =
      MetalRuntimeKernelLoader::load_msl_source(m_descriptor);
  std::string log;
  m_kernel = backend.compile(descriptor_source, &log);
  OPENVINO_ASSERT(m_kernel,
                  "MetalStage: failed to compile compiler-owned MSL "
                  "payload for ",
                  m_name, ": ", log);
}

std::vector<GpuTensor *> MetalStage::resolve_outputs() const {
  if (!m_outputs.empty()) {
    return m_outputs;
  }
  if (m_output) {
    return {m_output};
  }
  return {};
}

void MetalStage::prepare_constant_input_buffers() {
  if (m_descriptor.const_tensors.empty()) {
    return;
  }
  OPENVINO_ASSERT(m_buffer_manager,
                  "MetalStage: const input buffer manager is required for ",
                  m_name);
  if (!m_const_buffers) {
    m_const_buffers = std::make_shared<ConstBufferSet>();
  }
  auto slots = materialize_descriptor_const_tensor_slots(
      *m_buffer_manager, m_descriptor, "metal/source");
  m_const_buffers->buffers = std::move(slots.buffers);
  m_const_buffers->present = std::move(slots.present);
}

GpuTensor *MetalStage::resolve_input_tensor(size_t input_idx) const {
  GpuTensor *tensor =
      input_idx < m_inputs.size() ? m_inputs[input_idx] : nullptr;
  if (tensor && tensor->buf.valid()) {
    return tensor;
  }
  if (m_const_buffers && input_idx < m_const_buffers->buffers.size() &&
      input_idx < m_const_buffers->present.size() &&
      m_const_buffers->present[input_idx] &&
      m_const_buffers->buffers[input_idx].buf.valid()) {
    return const_cast<GpuTensor *>(&m_const_buffers->buffers[input_idx]);
  }
  return nullptr;
}

std::vector<int32_t> MetalStage::refresh_runtime_param_buffers(
    const std::vector<GpuTensor *> &outputs,
    const std::vector<int32_t> &compiler_scalar_args) {
  std::vector<int32_t> scalar_args = compiler_scalar_args;
  const size_t runtime_param_count = m_descriptor.runtime_param_buffer_count;
  if (runtime_param_count == 0) {
    return scalar_args;
  }

  RuntimeInputResolver runtime_inputs;
  runtime_inputs.inputs = &m_inputs;
  runtime_inputs.descriptor = &m_descriptor;
  if (m_const_buffers) {
    runtime_inputs.const_buffers = &m_const_buffers->buffers;
    runtime_inputs.const_buffer_present = &m_const_buffers->present;
  }

  if (auto materialization = materialize_descriptor_owned_runtime_param_payload(
          *m_buffer_manager, m_descriptor, runtime_inputs, outputs,
          compiler_scalar_args, m_name);
      materialization.available) {
    m_kernel_extra_inputs = std::move(materialization.extra_inputs);
    scalar_args = std::move(materialization.scalar_args);
    return scalar_args;
  }

  OPENVINO_THROW(
      "MetalStage: RuntimeParams ABI is not descriptor-owned for ", m_name,
      "; compiler descriptor/artifact metadata must own runtime payload "
      "construction");
}

std::vector<KernelArg>
MetalStage::materialize_source_args(const std::vector<GpuTensor *> &outputs) {
  OPENVINO_ASSERT(m_buffer_manager,
                  "MetalStage: buffer manager is not initialized for ", m_name);
  OPENVINO_ASSERT(m_descriptor.launch_plan.valid &&
                      !m_descriptor.launch_plan.buffer_roles.empty(),
                  "MetalStage: source descriptor launch plan is missing for ",
                  m_name);
  OPENVINO_ASSERT(m_descriptor.launch_plan.buffer_roles.size() ==
                      m_descriptor.abi_arg_count,
                  "MetalStage: source descriptor launch plan ABI count drift "
                  "for ",
                  m_name);
  prepare_constant_input_buffers();
  const auto &launch_plan = m_descriptor.launch_plan;
  OPENVINO_ASSERT(launch_plan.operand_kinds.size() ==
                      launch_plan.operand_arg_indices.size(),
                  "MetalStage: source descriptor operand ABI metadata drift "
                  "for ",
                  m_name);
  const bool has_runtime_binding =
      (!launch_plan.input_indices.empty() ||
       !launch_plan.operand_kinds.empty() || !launch_plan.scalar_args.empty());
  if (has_runtime_binding) {
    const auto scalar_args =
        refresh_runtime_param_buffers(outputs, launch_plan.scalar_args);
    auto bundle = build_kernel_args_from_metadata(
        launch_plan.operand_kinds, launch_plan.operand_arg_indices, scalar_args,
        launch_plan.input_indices, launch_plan.input_arg_count,
        m_kernel_extra_inputs, outputs,
        [&](size_t input_idx) { return resolve_input_tensor(input_idx); },
        m_name.c_str(), nullptr);
    m_scalar_storage = std::move(bundle.scalar_storage);
    return materialize_kernel_bytes_args(bundle.args, *m_buffer_manager,
                                         m_name.c_str());
  }
  return materialize_role_ordered_source_args(outputs);
}

std::vector<KernelArg> MetalStage::materialize_role_ordered_source_args(
    const std::vector<GpuTensor *> &outputs) {
  const auto roles =
      materialize_descriptor_launch_roles(m_descriptor.launch_plan, m_name);

  const auto const_count = static_cast<size_t>(
      std::count(roles.begin(), roles.end(), GfxKernelBufferRole::ConstTensor));
  std::vector<GpuTensor *> const_tensors;
  if (const_count != 0) {
    prepare_constant_input_buffers();
    OPENVINO_ASSERT(m_const_buffers,
                    "MetalStage: descriptor-owned const tensor buffers are "
                    "missing for ",
                    m_name);
    const_tensors = descriptor_const_tensor_args(*m_const_buffers, const_count);
    OPENVINO_ASSERT(const_tensors.size() == const_count,
                    "MetalStage: ConstTensor ABI count does not match "
                    "descriptor-owned constants for ",
                    m_name);
  }

  m_kernel_extra_inputs.clear();
  const auto scalar_args = refresh_runtime_param_buffers(
      outputs, m_descriptor.launch_plan.scalar_args);
  auto launch_plan = build_role_ordered_kernel_launch_plan<int32_t>(
      roles, m_descriptor.launch_plan.direct_input_indices, scalar_args,
      outputs, const_tensors, m_kernel_extra_inputs,
      [&](size_t input_idx) { return resolve_input_tensor(input_idx); },
      m_name);
  auto args = materialize_kernel_bytes_args(launch_plan.args, *m_buffer_manager,
                                            m_name.c_str());
  m_scalar_storage = std::move(launch_plan.scalar_storage);
  return args;
}

KernelDispatch MetalStage::make_source_dispatch(
    const std::vector<GpuTensor *> &outputs) const {
  ov::Shape shape;
  if (!outputs.empty() && outputs.front() && !outputs.front()->shape.empty()) {
    shape = outputs.front()->shape;
  } else {
    ov::Shape descriptor_shape;
    if (descriptor_output_shape(m_descriptor, 0, descriptor_shape)) {
      shape = std::move(descriptor_shape);
    }
  }
  const size_t local =
      m_kernel
          ? m_kernel->clamp_threadgroup_size(kDefaultMetalSourceThreadsPerGroup)
          : kDefaultMetalSourceThreadsPerGroup;
  return make_default_dispatch(shape, local);
}

KernelExecutionHooks *
MetalStage::prepare_profiling(ProfileState &state,
                              KernelExecutionHooks &hooks) {
  auto *profiler = static_cast<MetalProfiler *>(m_profiler);
  if (!profiler) {
    return nullptr;
  }
  state.cpu_start = std::chrono::steady_clock::now();
  const char *node_name = m_profile_node_name.empty()
                              ? name().c_str()
                              : m_profile_node_name.c_str();
  const char *node_type = m_profile_node_type.empty()
                              ? type().c_str()
                              : m_profile_node_type.c_str();
  hooks.stage_name = node_name;
  hooks.stage_type = node_type;
  profiler->begin_node(m_profile_node_id, node_name, node_type, "GFX");
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
  auto *profiler = static_cast<MetalProfiler *>(m_profiler);
  if (!profiler) {
    return;
  }
  const auto cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - state.cpu_start);
  profiler->end_node(m_profile_node_id, cpu_us, state.sample_begin,
                     state.sample_end);
}

} // namespace gfx_plugin
} // namespace ov
