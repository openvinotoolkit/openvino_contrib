// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/metal_op.hpp"
#include "runtime/profiling/metal_profiler.hpp"

namespace ov {
namespace gfx_plugin {

MetalOp::MetalOp(std::string name,
                 std::string type,
                 const ov::Shape& output_shape,
                 void* device,
                 void* command_queue)
    : m_name(std::move(name)),
      m_type(std::move(type)),
      m_output_shape(output_shape),
      m_device(static_cast<MetalDeviceHandle>(device)),
      m_command_queue(static_cast<MetalCommandQueueHandle>(command_queue)) {}

void MetalOp::init(MetalBufferManager* buffer_manager) {
    m_buffer_manager = buffer_manager;
}

void MetalOp::compile(MetalBufferManager* buffer_manager) {
    if (!m_buffer_manager) {
        m_buffer_manager = buffer_manager;
    }
    m_compiled = true;
}

void MetalOp::set_inputs(const std::vector<MetalTensor*>& inputs) {
    m_inputs = inputs;
}

void MetalOp::set_output(MetalTensor* output) {
    m_output = output;
}

void MetalOp::set_outputs(const std::vector<std::unique_ptr<MetalTensor>>& outputs) {
    if (!outputs.empty()) {
        m_output = outputs.front().get();
    }
}

MetalTensor& MetalOp::require_output() const {
    OPENVINO_ASSERT(m_output, "Output tensor is not bound for op ", m_name);
    return *m_output;
}

MetalBuffer MetalOp::allocate_temp_buffer(size_t bytes,
                                          ov::element::Type type,
                                          bool persistent,
                                          bool storageModePrivate) {
    OPENVINO_ASSERT(m_buffer_manager, "Buffer manager is not set for op ", m_name);
    return m_buffer_manager->allocate(bytes, type, persistent, storageModePrivate);
}

void MetalOp::set_profiler(MetalProfiler* profiler,
                           uint32_t node_id,
                           const std::string& node_name,
                           const std::string& node_type) {
    m_profiler = profiler;
    m_profile_node_id = node_id;
    m_profile_node_name = node_name;
    m_profile_node_type = node_type;
}

void MetalOp::start_profiling(MetalCommandEncoderHandle encoder) {
    if (!m_profiling_enabled)
        return;
    m_profile_start = std::chrono::steady_clock::now();
    if (m_profiler) {
        const char* name = m_profile_node_name.empty() ? m_name.c_str() : m_profile_node_name.c_str();
        const char* type = m_profile_node_type.empty() ? m_type.c_str() : m_profile_node_type.c_str();
        m_profiler->begin_node(m_profile_node_id, name, type, "GFX");
        m_gpu_sample_begin = m_profiler->gpu_sample_begin(encoder);
    }
}

double MetalOp::stop_profiling_ms(MetalCommandEncoderHandle encoder) {
    if (!m_profiling_enabled)
        return m_last_duration_ms;
    const auto end = std::chrono::steady_clock::now();
    const auto delta = end - m_profile_start;
    m_last_duration_ms =
        std::chrono::duration<double, std::milli>(delta).count();
    if (m_profiler) {
        const auto cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(delta);
        const auto sample_end = m_profiler->gpu_sample_end(encoder);
        m_profiler->end_node(m_profile_node_id, cpu_us, m_gpu_sample_begin, sample_end);
        m_gpu_sample_begin = -1;
    }
    return m_last_duration_ms;
}

}  // namespace gfx_plugin
}  // namespace ov
