// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/plugin/compiled_model_backend.hpp"

#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<MetalBackendState> create_metal_backend_state(const ov::AnyMap&,
                                                              const ov::SoPtr<ov::IRemoteContext>&) {
    OPENVINO_THROW("GFX Metal backend is not available in this build");
}

void release_metal_backend_state(MetalBackendState&) {}

MetalMemoryStats get_metal_memory_stats(const MetalBackendState* state) {
    return state ? state->last_stats : MetalMemoryStats{};
}

GpuDeviceHandle get_metal_device_handle(const MetalBackendState* state) {
    return state ? state->device : nullptr;
}

GpuCommandQueueHandle get_metal_command_queue(const MetalBackendState* state) {
    return state ? state->command_queue : nullptr;
}

GpuBufferManager* get_metal_const_buffer_manager(MetalBackendState* state) {
    return state ? static_cast<GpuBufferManager*>(state->const_manager.get()) : nullptr;
}

const GpuBufferManager* get_metal_const_buffer_manager(const MetalBackendState* state) {
    return state ? static_cast<const GpuBufferManager*>(state->const_manager.get()) : nullptr;
}

}  // namespace gfx_plugin
}  // namespace ov
