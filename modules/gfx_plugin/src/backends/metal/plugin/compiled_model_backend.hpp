// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "plugin/gfx_backend_config.hpp"
#include "backends/metal/plugin/compiled_model_state.hpp"
#include "runtime/gpu_buffer_manager.hpp"
#include "runtime/gpu_buffer.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<MetalBackendState> create_metal_backend_state(const ov::AnyMap& properties,
                                                              const ov::SoPtr<ov::IRemoteContext>& context);
void release_metal_backend_state(MetalBackendState& state);
MetalMemoryStats get_metal_memory_stats(const MetalBackendState* state);
GpuDeviceHandle get_metal_device_handle(const MetalBackendState* state);
GpuCommandQueueHandle get_metal_command_queue(const MetalBackendState* state);
GpuBufferManager* get_metal_const_buffer_manager(MetalBackendState* state);
const GpuBufferManager* get_metal_const_buffer_manager(const MetalBackendState* state);

}  // namespace gfx_plugin
}  // namespace ov
