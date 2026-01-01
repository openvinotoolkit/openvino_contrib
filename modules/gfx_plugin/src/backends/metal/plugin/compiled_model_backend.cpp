// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/plugin/compiled_model_backend.hpp"

#include "openvino/core/except.hpp"
#include "runtime/gfx_remote_context.hpp"
#include "plugin/gfx_property_utils.hpp"
#include "runtime/gfx_logger.hpp"
#include "backends/metal/runtime/metal_memory.hpp"
#include "backends/metal/runtime/profiling/profiler.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<MetalBackendState> create_metal_backend_state(const ov::AnyMap& properties,
                                                              const ov::SoPtr<ov::IRemoteContext>& context) {
    auto state = std::make_unique<MetalBackendState>();

    MetalDeviceHandle dev = nullptr;
    if (context) {
        auto gfx_ctx = std::dynamic_pointer_cast<GfxRemoteContext>(context._ptr);
        OPENVINO_ASSERT(gfx_ctx, "GFX: remote context type mismatch");
        OPENVINO_ASSERT(gfx_ctx->backend() == GpuBackend::Metal,
                        "GFX: remote context backend mismatch (expected Metal)");
        dev = gfx_ctx->device_handle();
    }
    if (!dev) {
        int device_id = parse_device_id(properties);
        dev = metal_get_device_by_id(device_id);
    }

    state->device = dev;
    state->command_queue = metal_create_command_queue(state->device);
    OPENVINO_ASSERT(state->command_queue, "GFX: failed to create command queue");
    state->caps = query_metal_device_caps(state->device);
    state->alloc_core = std::make_unique<MetalAllocatorCore>(state->device, state->caps);
    state->persistent_heaps = std::make_unique<MetalHeapPool>(*state->alloc_core);
    state->persistent_freelist = std::make_unique<MetalFreeList>();
    state->persistent_staging = std::make_unique<MetalStagingPool>(*state->alloc_core);
    state->persistent_alloc = std::make_unique<MetalAllocator>(*state->alloc_core,
                                                               *state->persistent_heaps,
                                                               *state->persistent_freelist,
                                                               *state->persistent_staging,
                                                               state->caps);
    state->const_cache = std::make_unique<MetalConstCache>(*state->persistent_alloc, state->command_queue);
    state->const_manager = std::make_shared<MetalBufferManager>(*state->alloc_core, state->const_cache.get());

    return state;
}

std::unique_ptr<GfxProfiler> MetalBackendState::create_profiler(const GfxProfilerConfig& cfg) const {
    OPENVINO_ASSERT(device, "GFX: Metal device is null");
    return std::make_unique<MetalProfiler>(cfg, caps, device);
}

}  // namespace gfx_plugin
}  // namespace ov
