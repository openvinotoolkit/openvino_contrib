// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/metal_executor.hpp"

#include <algorithm>
#include <chrono>
#include <optional>

#include "backends/metal/codegen/metal_codegen_backend.hpp"
#include "backends/metal/runtime/metal_memory.hpp"
#include "backends/metal/runtime/profiling/profiler.hpp"
#include "mlir/msl_codegen.hpp"
#include "openvino/core/except.hpp"
#include "runtime/gfx_shape_utils.hpp"

namespace ov {
namespace gfx_plugin {

MetalStage::MetalStage(const std::shared_ptr<const ov::Node>& node,
                       MetalDeviceHandle device,
                       MetalCommandQueueHandle queue)
    : MlirStage(node),
      m_device(device),
      m_queue(queue) {}

namespace {

// Compute row-major byte strides for a tensor shape. When broadcasting to a
// higher-rank output, extra leading dimensions are treated as length 1.
inline std::vector<int32_t> compute_broadcast_strides(const ov::Shape& in_shape,
                                                      const ov::Shape& out_shape,
                                                      size_t elem_size) {
    const size_t out_rank = out_shape.size();
    const size_t in_rank = in_shape.size();
    std::vector<int64_t> aligned(out_rank, 1);
    if (in_rank <= out_rank) {
        const size_t off = out_rank - in_rank;
        for (size_t i = 0; i < in_rank; ++i) {
            aligned[off + i] = static_cast<int64_t>(in_shape[i]);
        }
    }
    std::vector<int32_t> strides(out_rank, 0);
    int64_t stride = static_cast<int64_t>(elem_size);
    for (int64_t i = static_cast<int64_t>(out_rank) - 1; i >= 0; --i) {
        const int64_t dim = aligned[static_cast<size_t>(i)];
        // Broadcasted dim (size 1) uses zero stride.
        strides[static_cast<size_t>(i)] = (dim == 1) ? 0 : static_cast<int32_t>(stride);
        stride *= dim;
    }
    return strides;
}

inline std::vector<int32_t> to_i32_dims(const ov::Shape& shape) {
    std::vector<int32_t> dims(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        dims[i] = static_cast<int32_t>(shape[i]);
    }
    return dims;
}

inline ov::Shape resolve_shape_for_stage(const MlirStage& stage,
                                         const std::shared_ptr<const ov::Node>& node,
                                         GpuTensor* out_tensor) {
    if (out_tensor && !out_tensor->shape.empty()) {
        return out_tensor->shape;
    }
    if (node && node->get_output_partial_shape(0).is_static()) {
        return node->get_output_shape(0);
    }
    return {};
}

}  // namespace

void MetalStage::init(GpuBufferManager* buffer_manager) {
    MlirStage::init(buffer_manager);
    if (!m_device) {
        if (auto* metal_mgr = dynamic_cast<MetalBufferManager*>(buffer_manager)) {
            m_device = metal_mgr->device();
        }
    }
}

void MetalStage::compile(GpuBufferManager* buffer_manager) {
    MlirStage::compile(buffer_manager);
}

void MetalStage::execute(GpuCommandBufferHandle command_buffer) {
    MlirStage::execute(command_buffer);
}

void MetalStage::set_inputs(const std::vector<GpuTensor*>& inputs) {
    MlirStage::set_inputs(inputs);
}

void MetalStage::set_output(GpuTensor* output) {
    MlirStage::set_output(output);
}

void MetalStage::set_outputs(const std::vector<std::unique_ptr<GpuTensor>>& outputs) {
    MlirStage::set_outputs(outputs);
}

bool MetalStage::fuse_activation(ActivationKind kind, float alpha) {
    return MlirStage::fuse_activation(kind, alpha);
}

bool MetalStage::fuse_batchnorm(const BatchNormParams& params) {
    return MlirStage::fuse_batchnorm(params);
}

bool MetalStage::fuse_bias(const BiasParams& params) {
    return MlirStage::fuse_bias(params);
}

void MetalStage::enable_profiling(bool enable) {
    MlirStage::enable_profiling(enable);
}

void MetalStage::set_profiler(void* profiler,
                              uint32_t node_id,
                              const std::string& node_name,
                              const std::string& node_type) {
    MlirStage::set_profiler(profiler, node_id, node_name, node_type);
}

std::unique_ptr<GpuStage> MetalStage::clone() const {
    auto stage = std::make_unique<MetalStage>(m_node, m_device, m_queue);
    clone_into(*stage);
    return stage;
}

std::shared_ptr<ICompiledKernel> MetalStage::compile_kernel(const KernelSource& source,
                                                            std::string* log) {
    OPENVINO_ASSERT(m_device, "MetalStage: Metal device handle is null");
    KernelSource src = source;
    MetalCodegenBackend backend(m_device);
    ov::element::Type storage_type = ov::element::dynamic;
    if (!m_inputs.empty() && m_inputs.front()) {
        storage_type = m_inputs.front()->expected_type;
    }
    if (storage_type == ov::element::dynamic && !m_outputs.empty() && m_outputs.front()) {
        storage_type = m_outputs.front()->expected_type;
    }
    if (storage_type == ov::element::dynamic && m_node) {
        storage_type = m_node->get_output_element_type(0);
    }
    const std::optional<ov::Shape> runtime_input_shape =
        !m_inputs.empty() && m_inputs.front() && !m_inputs.front()->shape.empty()
            ? std::optional<ov::Shape>(m_inputs.front()->shape)
            : std::nullopt;
    auto source_plan = configure_apple_metal_kernel_source_plan_for_stage(src,
                                                                          m_node,
                                                                          m_buffer_manager,
                                                                          m_type,
                                                                          m_has_bias,
                                                                          m_has_activation,
                                                                          m_has_bn,
                                                                          m_activation,
                                                                          storage_type,
                                                                          !m_kernel_extra_inputs.empty(),
                                                                          runtime_input_shape);
    if (source_plan.valid()) {
        m_last_compiled_kernel_entry_point = source_plan.source.entry_point;
        auto kernel = backend.compile(source_plan.source, log);
        if (source_plan.has_runtime_binding) {
            apply_source_plan_kernel_runtime_binding_state(source_plan.runtime_binding);
        }
        return kernel;
    }
    OPENVINO_ASSERT(src.msl_generator || !src.msl_source.empty(),
                    "MetalStage: missing MSL source/generator for op ",
                    m_node ? m_node->get_type_name() : "");
    m_last_compiled_kernel_entry_point = src.entry_point;
    return backend.compile(src, log);
}

KernelExecutionHooks* MetalStage::prepare_profiling(ProfileState& state,
                                                    KernelExecutionHooks& hooks) {
    auto* profiler = static_cast<MetalProfiler*>(profiler_handle());
    if (!profiler) {
        return nullptr;
    }
    state.cpu_start = std::chrono::steady_clock::now();
    const char* node_name = profile_node_name().empty() ? name().c_str() : profile_node_name().c_str();
    const char* node_type = profile_node_type().empty() ? type().c_str() : profile_node_type().c_str();
    profiler->begin_node(profile_node_id(), node_name, node_type, "GFX");
    hooks.on_begin = [profiler, &state](GpuCommandEncoderHandle enc) {
        state.sample_begin = profiler->gpu_sample_begin(static_cast<MetalCommandEncoderHandle>(enc));
    };
    hooks.on_end = [profiler, &state](GpuCommandEncoderHandle enc) {
        state.sample_end = profiler->gpu_sample_end(static_cast<MetalCommandEncoderHandle>(enc));
    };
    hooks.on_counter = [profiler](std::string_view name, uint64_t delta) {
        profiler->increment_counter(name, delta);
    };
    hooks.on_segment = [profiler](std::string_view phase,
                                  std::string_view name,
                                  std::chrono::microseconds cpu_us,
                                  uint64_t gpu_us,
                                  uint32_t dispatches,
                                  uint64_t bytes_in,
                                  uint64_t bytes_out,
                                  uint64_t macs_est,
                                  uint64_t flops_est,
                                  int64_t inflight_slot,
                                  uint64_t queue_id,
                                  uint64_t cmd_buffer_id) {
        profiler->record_segment(phase,
                                 name,
                                 cpu_us,
                                 gpu_us,
                                 dispatches,
                                 bytes_in,
                                 bytes_out,
                                 macs_est,
                                 flops_est,
                                 inflight_slot,
                                 queue_id,
                                 cmd_buffer_id);
    };
    return &hooks;
}

void MetalStage::finalize_profiling(const ProfileState& state) {
    auto* profiler = static_cast<MetalProfiler*>(profiler_handle());
    if (!profiler) {
        return;
    }
    const auto cpu_us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - state.cpu_start);
    profiler->end_node(profile_node_id(), cpu_us, state.sample_begin, state.sample_end);
}

}  // namespace gfx_plugin
}  // namespace ov
