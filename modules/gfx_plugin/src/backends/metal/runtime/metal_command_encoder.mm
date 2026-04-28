// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/metal_command_encoder.hpp"

#import <Metal/Metal.h>

#include <unordered_map>
#include <vector>

namespace ov {
namespace gfx_plugin {
namespace {

struct MetalComputeEncoderState {
    id<MTLComputeCommandEncoder> encoder = nil;
    void* pipeline = nullptr;
    std::vector<void*> buffers;
    std::vector<size_t> offsets;
};

std::unordered_map<GpuCommandBufferHandle, MetalComputeEncoderState>& active_compute_encoders() {
    static thread_local std::unordered_map<GpuCommandBufferHandle, MetalComputeEncoderState> encoders;
    return encoders;
}

}  // namespace

GpuCommandEncoderHandle metal_get_or_create_compute_encoder(GpuCommandBufferHandle command_buffer,
                                                            bool* created) {
    if (created) {
        *created = false;
    }
    if (!command_buffer) {
        return nullptr;
    }

    auto& encoders = active_compute_encoders();
    auto it = encoders.find(command_buffer);
    if (it != encoders.end()) {
        return reinterpret_cast<GpuCommandEncoderHandle>(it->second.encoder);
    }

    id<MTLCommandBuffer> cb = static_cast<id<MTLCommandBuffer>>(command_buffer);
    id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
    MetalComputeEncoderState state;
    state.encoder = encoder;
    encoders.emplace(command_buffer, std::move(state));
    if (created) {
        *created = true;
    }
    return reinterpret_cast<GpuCommandEncoderHandle>(encoder);
}

bool metal_set_compute_pipeline_if_needed(GpuCommandBufferHandle command_buffer,
                                          GpuCommandEncoderHandle encoder,
                                          void* pipeline) {
    if (!command_buffer || !encoder || !pipeline) {
        return false;
    }
    auto& encoders = active_compute_encoders();
    auto it = encoders.find(command_buffer);
    if (it == encoders.end() || it->second.pipeline == pipeline) {
        return false;
    }
    id<MTLComputeCommandEncoder> enc = static_cast<id<MTLComputeCommandEncoder>>(encoder);
    [enc setComputePipelineState:static_cast<id<MTLComputePipelineState>>(pipeline)];
    it->second.pipeline = pipeline;
    return true;
}

size_t metal_bind_compute_buffers_if_needed(GpuCommandBufferHandle command_buffer,
                                            GpuCommandEncoderHandle encoder,
                                            const std::vector<void*>& buffers,
                                            const std::vector<size_t>& offsets) {
    if (!command_buffer || !encoder) {
        return 0;
    }
    auto& encoders = active_compute_encoders();
    auto it = encoders.find(command_buffer);
    if (it == encoders.end()) {
        return 0;
    }

    auto& state = it->second;
    if (state.buffers.size() < buffers.size()) {
        state.buffers.resize(buffers.size(), nullptr);
        state.offsets.resize(buffers.size(), 0);
    }

    id<MTLComputeCommandEncoder> enc = static_cast<id<MTLComputeCommandEncoder>>(encoder);
    size_t bound = 0;
    for (size_t index = 0; index < buffers.size(); ++index) {
        const size_t offset = index < offsets.size() ? offsets[index] : 0;
        if (state.buffers[index] == buffers[index] && state.offsets[index] == offset) {
            continue;
        }
        [enc setBuffer:static_cast<id<MTLBuffer>>(buffers[index]) offset:offset atIndex:index];
        state.buffers[index] = buffers[index];
        state.offsets[index] = offset;
        ++bound;
    }
    return bound;
}

void metal_end_compute_encoder(GpuCommandBufferHandle command_buffer) {
    if (!command_buffer) {
        return;
    }
    auto& encoders = active_compute_encoders();
    auto it = encoders.find(command_buffer);
    if (it == encoders.end()) {
        return;
    }
    [it->second.encoder endEncoding];
    encoders.erase(it);
}

}  // namespace gfx_plugin
}  // namespace ov
