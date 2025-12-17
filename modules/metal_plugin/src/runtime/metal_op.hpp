// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/core/except.hpp"
#include "runtime/metal_memory.hpp"

namespace ov {
namespace metal_plugin {

// Visibility helper: keep MetalOp symbols exported for tests and downstream components.
#if defined(__clang__) || defined(__GNUC__)
#    define METAL_OP_API __attribute__((visibility("default")))
#else
#    define METAL_OP_API
#endif

// Abstract base for all Metal GPU operations.
// Stores shared device context and lightweight metadata that concrete ops reuse.
class METAL_OP_API MetalOp {
public:
    MetalOp(std::string name,
            std::string type,
            const ov::Shape& output_shape,
            void* device = nullptr,
            void* command_queue = nullptr);

    virtual ~MetalOp() = default;

    // Optional initialization hook (e.g., compile kernels, cache layouts).
    virtual void init(MetalBufferManager* buffer_manager);

    // Execute op on device. Must be implemented by derived ops.
    virtual void execute() = 0;

    void set_inputs(const std::vector<MetalTensor*>& inputs);
    void set_output(MetalTensor* output);
    // Optional multi-output binding (used by ops like Split). Default binds first output.
    virtual void set_outputs(const std::vector<std::unique_ptr<MetalTensor>>& outputs);

    const std::vector<MetalTensor*>& inputs() const { return m_inputs; }
    MetalTensor* output() const { return m_output; }

    MetalDeviceHandle device() const { return m_device; }
    MetalCommandQueueHandle command_queue() const { return m_command_queue; }

    const std::string& name() const { return m_name; }
    const std::string& type() const { return m_type; }
    const ov::Shape& output_shape() const { return m_output_shape; }

    void enable_profiling(bool enable) { m_profiling_enabled = enable; }
    double last_exec_duration_ms() const { return m_last_duration_ms; }

protected:
    // Allocate temporary device memory via the shared buffer manager.
    MetalBuffer allocate_temp_buffer(size_t bytes,
                                     ov::element::Type type,
                                     bool persistent = false,
                                     bool storageModePrivate = true);

    MetalBufferManager* buffer_manager() const { return m_buffer_manager; }

    // Profiling helpers used by derived ops.
    void start_profiling();
    double stop_profiling_ms();

    MetalTensor& require_output() const;

private:
    std::string m_name;
    std::string m_type;
    ov::Shape m_output_shape;

    MetalDeviceHandle m_device;
    MetalCommandQueueHandle m_command_queue;
    MetalBufferManager* m_buffer_manager = nullptr;  // non-owning

    std::vector<MetalTensor*> m_inputs;  // non-owning
    MetalTensor* m_output = nullptr;     // non-owning

    bool m_profiling_enabled = false;
    std::chrono::steady_clock::time_point m_profile_start;
    double m_last_duration_ms = 0.0;
};

}  // namespace metal_plugin
}  // namespace ov
