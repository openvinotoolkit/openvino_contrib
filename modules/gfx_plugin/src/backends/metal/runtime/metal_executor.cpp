// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/metal_executor.hpp"

#include <algorithm>
#include <chrono>
#include <sstream>

#include "backends/metal/codegen/metal_codegen_backend.hpp"
#include "backends/metal/runtime/metal_memory.hpp"
#include "backends/metal/runtime/metal_runtime_kernel_loader.hpp"
#include "backends/metal/runtime/profiling/profiler.hpp"
#include "common/constant_tensor_evaluator.hpp"
#include "kernel_ir/gfx_kernel_args.hpp"
#include "kernel_ir/gfx_kernel_cache.hpp"
#include "kernel_ir/gfx_kernel_dispatch.hpp"
#include "runtime/gfx_kernel_runtime_params.hpp"
#include "kernel_ir/gfx_kernel_source.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "runtime/gfx_shape_utils.hpp"
#include "runtime/gfx_stage_runtime_values.hpp"
#include "runtime/kernel_launch_plan.hpp"

namespace ov {
namespace gfx_plugin {

MetalStage::MetalStage(const std::shared_ptr<const ov::Node> &node,
                       MetalDeviceHandle device, MetalCommandQueueHandle queue,
                       const RuntimeStageExecutableDescriptor *descriptor)
    : m_device(device), m_queue(queue), m_node(node) {
  OPENVINO_ASSERT(descriptor,
                  "MetalStage: compiler-owned runtime descriptor is required");
  m_descriptor = *descriptor;
  if (m_node) {
    m_name = m_node->get_friendly_name();
    m_type = m_node->get_type_name();
  }
}

namespace {

inline bool is_metal_descriptor_domain(std::string_view domain) {
  return domain == "metal" || domain == "apple_msl" ||
         domain == "apple_mps" || domain == "common";
}

constexpr size_t kDefaultMetalSourceThreadsPerGroup = 64;

const GfxKernelSourcePayload *source_payload_or_null(
    const RuntimeStageExecutableDescriptor &descriptor) {
  if (!descriptor.payload) {
    return nullptr;
  }
  return dynamic_cast<const GfxKernelSourcePayload *>(descriptor.payload.get());
}

bool is_binary_runtime_param_stage(std::string_view type) {
  return type == "Add" || type == "Subtract" || type == "Multiply" ||
         type == "Divide" || type == "Power" || type == "Mod" ||
         type == "FloorMod" || type == "Minimum" || type == "Maximum" ||
         type == "Equal" || type == "NotEqual" || type == "Less" ||
         type == "Greater" || type == "LessEqual" ||
         type == "GreaterEqual" || type == "LogicalAnd" ||
         type == "LogicalOr" || type == "LogicalXor" ||
         type == "SquaredDifference" || type == "PRelu";
}

size_t count_runtime_param_roles(const GfxKernelSourcePayload &payload) {
  const auto roles = materialize_gfx_kernel_external_buffer_roles(
      payload.stage_manifest().custom_kernel.external_buffer_abi);
  return static_cast<size_t>(
      std::count(roles.begin(), roles.end(), GfxKernelBufferRole::RuntimeParams));
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
  OPENVINO_ASSERT(m_kernel,
                  "MetalStage: runtime handle is not prepared for ",
                  m_name);
  const auto outputs = resolve_outputs();
  OPENVINO_ASSERT(!outputs.empty(), "MetalStage: output tensor is not bound for ",
                  m_name);
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

bool MetalStage::fuse_residual_add() {
  return false;
}

bool MetalStage::fuse_batchnorm(const BatchNormParams &params) {
  (void)params;
  return false;
}

bool MetalStage::fuse_bias(const BiasParams &params) {
  (void)params;
  return false;
}

void MetalStage::enable_profiling(bool enable) {
  m_profiling_enabled = enable;
}

void MetalStage::set_profiler(void *profiler, uint32_t node_id,
                              const std::string &node_name,
                              const std::string &node_type) {
  m_profiler = profiler;
  m_profile_node_id = node_id;
  m_profile_node_name = node_name;
  m_profile_node_type = node_type;
}

std::unique_ptr<GpuStage> MetalStage::clone() const {
  auto stage = std::make_unique<MetalStage>(m_node, m_device, m_queue,
                                            &m_descriptor);
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
  OPENVINO_ASSERT(m_kernel, "MetalStage: failed to compile compiler-owned MSL "
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

struct MetalStage::ConstBufferSet {
  std::vector<GpuTensor> buffers;
  std::vector<bool> present;
};

void MetalStage::prepare_constant_input_buffers() {
  if (!m_node || !m_buffer_manager) {
    return;
  }
  const size_t input_count = m_node->get_input_size();
  if (!m_const_buffers) {
    m_const_buffers = std::make_shared<ConstBufferSet>();
  }
  if (m_const_buffers->buffers.size() < input_count) {
    m_const_buffers->buffers.resize(input_count);
    m_const_buffers->present.assign(input_count, false);
  }

  bool const_cache_checked = false;
  for (size_t input_idx = 0; input_idx < input_count; ++input_idx) {
    if (m_const_buffers->present[input_idx] &&
        m_const_buffers->buffers[input_idx].buf.valid()) {
      continue;
    }
    auto const_tensor =
        gfx_evaluate_constant_source_tensor(m_node->input_value(input_idx));
    if (!const_tensor.has_value()) {
      continue;
    }
    if (!const_cache_checked) {
      OPENVINO_ASSERT(
          m_buffer_manager->supports_const_cache(),
          "MetalStage: const cache is required for compiler-owned source "
          "artifact const inputs in ",
          m_name);
      const_cache_checked = true;
    }

    const void *data = const_tensor->data();
    const size_t bytes = const_tensor->get_byte_size();
    const auto element_type = const_tensor->get_element_type();
    if (bytes > 0) {
      std::ostringstream key;
      key << m_name << "/source-const/" << input_idx << "/"
          << element_type.get_type_name() << "/" << bytes << "/"
          << gfx_hash_bytes(data, bytes);
      GpuBuffer buffer =
          m_buffer_manager->wrap_const(key.str(), data, bytes, element_type);
      OPENVINO_ASSERT(buffer.valid(),
                      "MetalStage: failed to materialize const input ",
                      input_idx, " for ", m_name);
      buffer.owned = false;
      m_const_buffers->buffers[input_idx].buf = buffer;
    }
    m_const_buffers->buffers[input_idx].shape = const_tensor->get_shape();
    m_const_buffers->buffers[input_idx].expected_type = element_type;
    m_const_buffers->present[input_idx] = true;
  }
}

GpuTensor *MetalStage::resolve_input_tensor(size_t input_idx) const {
  GpuTensor *tensor = input_idx < m_inputs.size() ? m_inputs[input_idx]
                                                  : nullptr;
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
    const GfxKernelSourcePayload &payload,
    const std::vector<GpuTensor *> &outputs,
    const std::vector<int32_t> &compiler_scalar_args) {
  std::vector<int32_t> scalar_args = compiler_scalar_args;
  const size_t runtime_param_count = count_runtime_param_roles(payload);
  if (runtime_param_count == 0 || !m_node) {
    return scalar_args;
  }

  RuntimeInputResolver runtime_inputs;
  runtime_inputs.inputs = &m_inputs;
  runtime_inputs.node = m_node;
  if (m_const_buffers) {
    runtime_inputs.const_buffers = &m_const_buffers->buffers;
    runtime_inputs.const_buffer_present = &m_const_buffers->present;
  }

  if (runtime_param_count == 3 && is_binary_runtime_param_stage(m_type) &&
      m_node->get_input_size() >= 2) {
    ov::Shape lhs_shape;
    ov::Shape rhs_shape;
    if (runtime_inputs.shape_known(0, lhs_shape) &&
        runtime_inputs.shape_known(1, rhs_shape)) {
      const ov::Shape output_shape =
          compute_binary_broadcast_shape(lhs_shape, rhs_shape, m_name);
      const ov::Shape meta_shape =
          output_shape.empty() ? ov::Shape{1} : output_shape;
      auto broadcast_payload = make_binary_broadcast_runtime_param_payload(
          *m_buffer_manager, m_name, output_shape,
          compute_broadcast_element_strides(lhs_shape, meta_shape),
          compute_broadcast_element_strides(rhs_shape, meta_shape));
      m_kernel_extra_inputs = std::move(broadcast_payload.extra_inputs);
      scalar_args = std::move(broadcast_payload.scalar_args);
      for (auto *output : outputs) {
        if (!output) {
          continue;
        }
        if (output->shape.empty()) {
          output->shape = output_shape;
        }
        if (output->expected_type == ov::element::dynamic &&
            m_node->get_output_size() != 0) {
          output->expected_type = m_node->get_output_element_type(0);
        }
      }
    }
    return scalar_args;
  }

  if (runtime_param_count == 5 && m_type == "Transpose") {
    const auto transpose_plan =
        plan_transpose_runtime_values(runtime_inputs, *m_node, m_name);
    auto transpose_payload = make_transpose_runtime_param_payload(
        *m_buffer_manager, m_name, transpose_plan.input_shape,
        transpose_plan.values.output_shape, transpose_plan.permutation);
    m_kernel_extra_inputs = std::move(transpose_payload.extra_inputs);
    scalar_args = std::move(transpose_payload.scalar_args);
    assign_runtime_value_outputs(transpose_plan.values, outputs);
    return scalar_args;
  }

  if (runtime_param_count == 5) {
    if (auto reduce_info = get_runtime_reduce_info(m_node)) {
      const auto reduce_plan = plan_reduce_runtime_values(
          runtime_inputs, m_node.get(), m_type, *reduce_info, m_name);
      const uint32_t op_code = compiler_scalar_args.size() > 2
                                   ? static_cast<uint32_t>(compiler_scalar_args[2])
                                   : 0u;
      auto reduce_payload = make_reduce_runtime_param_payload(
          *m_buffer_manager, m_name, reduce_plan.input_shape, reduce_info->axes,
          reduce_info->keep_dims, reduce_plan.values.output_shape, op_code);
      m_kernel_extra_inputs = std::move(reduce_payload.extra_inputs);
      scalar_args = std::move(reduce_payload.scalar_args);
      for (auto *output : outputs) {
        if (!output) {
          continue;
        }
        if (output->shape.empty()) {
          output->shape = reduce_plan.values.output_shape;
        }
        if (output->expected_type == ov::element::dynamic &&
            m_node->get_output_size() != 0) {
          output->expected_type = m_node->get_output_element_type(0);
        }
      }
    }
  }

  return scalar_args;
}

std::vector<KernelArg>
MetalStage::materialize_source_args(const std::vector<GpuTensor *> &outputs) {
  OPENVINO_ASSERT(m_buffer_manager,
                  "MetalStage: buffer manager is not initialized for ",
                  m_name);
  const auto *payload = source_payload_or_null(m_descriptor);
  OPENVINO_ASSERT(payload && payload->has_stage_manifest(),
                  "MetalStage: MSL source payload is missing compiler-owned "
                  "kernel manifest ABI for ",
                  m_name);
  const auto &manifest = payload->stage_manifest();
  OPENVINO_ASSERT(manifest.backend_domain == GfxKernelBackendDomain::AppleMsl,
                  "MetalStage: MSL source payload backend domain drift for ",
                  m_name);
  prepare_constant_input_buffers();
  if (payload->has_runtime_binding()) {
    const auto &binding = payload->runtime_binding();
    const auto scalar_args =
        refresh_runtime_param_buffers(*payload, outputs, binding.scalar_args);
    auto bundle = build_kernel_args_from_metadata(
        binding.operand_kinds, binding.operand_arg_indices,
        scalar_args, binding.inputs, binding.input_arg_count,
        m_kernel_extra_inputs, outputs,
        [&](size_t input_idx) { return resolve_input_tensor(input_idx); },
        m_name.c_str(), nullptr);
    m_scalar_storage = std::move(bundle.scalar_storage);
    return materialize_kernel_bytes_args(bundle.args, *m_buffer_manager,
                                         m_name.c_str());
  }
  return materialize_role_ordered_source_args(*payload, outputs);
}

std::vector<KernelArg> MetalStage::materialize_role_ordered_source_args(
    const GfxKernelSourcePayload &payload,
    const std::vector<GpuTensor *> &outputs) {
  const auto &manifest = payload.stage_manifest();
  const auto roles = materialize_gfx_kernel_external_buffer_roles(
      manifest.custom_kernel.external_buffer_abi);
  OPENVINO_ASSERT(!roles.empty(), "MetalStage: source ABI roles are empty for ",
                  m_name);

  const auto const_count = static_cast<size_t>(
      std::count(roles.begin(), roles.end(), GfxKernelBufferRole::ConstTensor));
  std::vector<GpuTensor *> const_tensors;
  const_tensors.reserve(const_count);
  if (const_count != 0) {
    prepare_constant_input_buffers();
    if (m_const_buffers) {
      for (size_t input_idx = 0;
           input_idx < m_const_buffers->buffers.size() &&
           const_tensors.size() < const_count;
           ++input_idx) {
        if (input_idx < m_const_buffers->present.size() &&
            m_const_buffers->present[input_idx] &&
            m_const_buffers->buffers[input_idx].buf.valid()) {
          const_tensors.push_back(&m_const_buffers->buffers[input_idx]);
        }
      }
    }
    OPENVINO_ASSERT(const_tensors.size() == const_count,
                    "MetalStage: ConstTensor ABI count does not match "
                    "materialized constants for ",
                    m_name);
  }

  m_kernel_extra_inputs.clear();
  const auto scalar_args = refresh_runtime_param_buffers(
      payload, outputs, manifest.custom_kernel.scalar_args);
  auto launch_plan = build_role_ordered_kernel_launch_plan<int32_t>(
      roles, {}, scalar_args, outputs, const_tensors, m_kernel_extra_inputs,
      [&](size_t input_idx) { return resolve_input_tensor(input_idx); },
      m_name);
  auto args = materialize_kernel_bytes_args(launch_plan.args, *m_buffer_manager,
                                            m_name.c_str());
  m_scalar_storage = std::move(launch_plan.scalar_storage);
  return args;
}

KernelDispatch
MetalStage::make_source_dispatch(const std::vector<GpuTensor *> &outputs) const {
  ov::Shape shape;
  if (!outputs.empty() && outputs.front() && !outputs.front()->shape.empty()) {
    shape = outputs.front()->shape;
  } else if (m_node && m_node->get_output_size() > 0 &&
             m_node->get_output_partial_shape(0).is_static()) {
    shape = m_node->get_output_shape(0);
  }
  const size_t local = m_kernel
                           ? m_kernel->clamp_threadgroup_size(
                                 kDefaultMetalSourceThreadsPerGroup)
                           : kDefaultMetalSourceThreadsPerGroup;
  return make_default_dispatch(shape, local);
}

KernelExecutionHooks *MetalStage::prepare_profiling(ProfileState &state,
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
