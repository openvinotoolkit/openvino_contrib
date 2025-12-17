// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/metal_op.hpp"

namespace ov {
namespace metal_plugin {

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

void MetalOp::start_profiling() {
    if (!m_profiling_enabled)
        return;
    m_profile_start = std::chrono::steady_clock::now();
}

double MetalOp::stop_profiling_ms() {
    if (!m_profiling_enabled)
        return m_last_duration_ms;
    const auto end = std::chrono::steady_clock::now();
    m_last_duration_ms =
        std::chrono::duration<double, std::milli>(end - m_profile_start).count();
    return m_last_duration_ms;
}

}  // namespace metal_plugin
}  // namespace ov
