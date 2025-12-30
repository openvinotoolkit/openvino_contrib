// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "plugin/compiled_model_backend_resources.hpp"

#include "backends/metal/plugin/compiled_model_backend.hpp"
#include "backends/vulkan/plugin/compiled_model_backend.hpp"
#include "plugin/backend_state.hpp"
#include "backends/metal/plugin/compiled_model_state.hpp"
#include "backends/vulkan/plugin/compiled_model_state.hpp"

namespace ov {
namespace gfx_plugin {

BackendResources get_backend_resources(GpuBackend backend, BackendState* state) {
    switch (backend) {
    case GpuBackend::Metal:
        return {get_metal_device_handle(dynamic_cast<MetalBackendState*>(state)),
                get_metal_command_queue(dynamic_cast<MetalBackendState*>(state)),
                get_metal_const_buffer_manager(dynamic_cast<MetalBackendState*>(state))};
    case GpuBackend::Vulkan:
        return {get_vulkan_device_handle(dynamic_cast<VulkanBackendState*>(state)),
                get_vulkan_command_queue(dynamic_cast<VulkanBackendState*>(state)),
                get_vulkan_const_buffer_manager(dynamic_cast<VulkanBackendState*>(state))};
    }
    return {};
}

bool backend_has_const_manager(GpuBackend backend, BackendState* state) {
    switch (backend) {
    case GpuBackend::Metal:
        return get_metal_const_buffer_manager(dynamic_cast<MetalBackendState*>(state)) != nullptr;
    case GpuBackend::Vulkan:
        return vulkan_has_const_manager(dynamic_cast<VulkanBackendState*>(state));
    }
    return true;
}

}  // namespace gfx_plugin
}  // namespace ov
