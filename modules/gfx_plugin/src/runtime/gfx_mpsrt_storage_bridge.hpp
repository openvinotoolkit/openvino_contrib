// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <string_view>

#include "runtime/gfx_mpsrt_abi.hpp"

namespace ov {
namespace gfx_plugin {

enum class GfxMpsrtStorageBridgeDirection : uint32_t {
    Unknown = 0,
    BufferToImage = 1,
    ImageToBuffer = 2,
};

struct GfxMpsrtStorageBridgeDesc {
    GfxMpsrtValue value = 0;
    GfxMpsrtStorageBridgeDirection direction = GfxMpsrtStorageBridgeDirection::Unknown;
    GfxMpsrtStorage source_storage = GfxMpsrtStorage::Unknown;
    GfxMpsrtStorage target_storage = GfxMpsrtStorage::Unknown;
    GfxMpsrtTensorAbiDesc tensor{};
};

inline bool gfx_mpsrt_tensor_is_image(const GfxMpsrtTensorAbiDesc& desc) {
    return desc.storage == static_cast<uint32_t>(GfxMpsrtStorage::Image);
}

inline bool gfx_mpsrt_image_bridge_supported(const GfxMpsrtTensorAbiDesc& desc) {
    return desc.rank == 4 &&
           (desc.dtype == static_cast<uint32_t>(GfxMpsrtDType::F16) ||
            desc.dtype == static_cast<uint32_t>(GfxMpsrtDType::F32)) &&
           desc.image_width != 0 &&
           desc.image_height != 0 &&
           desc.image_feature_channels != 0 &&
           desc.image_batch != 0;
}

inline GfxMpsrtStorageBridgeDirection gfx_mpsrt_external_image_bridge_direction(bool external_output) {
    return external_output ? GfxMpsrtStorageBridgeDirection::ImageToBuffer
                           : GfxMpsrtStorageBridgeDirection::BufferToImage;
}

inline GfxMpsrtStorage gfx_mpsrt_storage_bridge_source_storage(GfxMpsrtStorageBridgeDirection direction) {
    switch (direction) {
        case GfxMpsrtStorageBridgeDirection::BufferToImage:
            return GfxMpsrtStorage::Buffer;
        case GfxMpsrtStorageBridgeDirection::ImageToBuffer:
            return GfxMpsrtStorage::Image;
        case GfxMpsrtStorageBridgeDirection::Unknown:
        default:
            return GfxMpsrtStorage::Unknown;
    }
}

inline GfxMpsrtStorage gfx_mpsrt_storage_bridge_target_storage(GfxMpsrtStorageBridgeDirection direction) {
    switch (direction) {
        case GfxMpsrtStorageBridgeDirection::BufferToImage:
            return GfxMpsrtStorage::Image;
        case GfxMpsrtStorageBridgeDirection::ImageToBuffer:
            return GfxMpsrtStorage::Buffer;
        case GfxMpsrtStorageBridgeDirection::Unknown:
        default:
            return GfxMpsrtStorage::Unknown;
    }
}

inline bool gfx_mpsrt_make_image_bridge_desc(GfxMpsrtValue value,
                                             const GfxMpsrtTensorAbiDesc& tensor,
                                             GfxMpsrtStorageBridgeDirection direction,
                                             GfxMpsrtStorageBridgeDesc& bridge) {
    if (!gfx_mpsrt_image_bridge_supported(tensor) || direction == GfxMpsrtStorageBridgeDirection::Unknown) {
        return false;
    }
    bridge.value = value;
    bridge.direction = direction;
    bridge.source_storage = gfx_mpsrt_storage_bridge_source_storage(direction);
    bridge.target_storage = gfx_mpsrt_storage_bridge_target_storage(direction);
    bridge.tensor = tensor;
    return bridge.source_storage != GfxMpsrtStorage::Unknown &&
           bridge.target_storage != GfxMpsrtStorage::Unknown;
}

inline const char* gfx_mpsrt_storage_bridge_direction_name(GfxMpsrtStorageBridgeDirection direction) {
    switch (direction) {
        case GfxMpsrtStorageBridgeDirection::BufferToImage:
            return "buffer_to_image";
        case GfxMpsrtStorageBridgeDirection::ImageToBuffer:
            return "image_to_buffer";
        case GfxMpsrtStorageBridgeDirection::Unknown:
        default:
            return "unknown";
    }
}

inline GfxMpsrtStorageBridgeDirection gfx_mpsrt_storage_bridge_direction_from_name(std::string_view name) {
    if (name == "buffer_to_image") {
        return GfxMpsrtStorageBridgeDirection::BufferToImage;
    }
    if (name == "image_to_buffer") {
        return GfxMpsrtStorageBridgeDirection::ImageToBuffer;
    }
    return GfxMpsrtStorageBridgeDirection::Unknown;
}

}  // namespace gfx_plugin
}  // namespace ov
