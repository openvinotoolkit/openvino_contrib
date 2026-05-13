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
    BufferToMatrix = 3,
    MatrixToBuffer = 4,
    BufferToNDArray = 5,
    NDArrayToBuffer = 6,
    Alias = 7,
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

inline bool gfx_mpsrt_matrix_bridge_supported(const GfxMpsrtTensorAbiDesc& desc) {
    return (desc.storage == static_cast<uint32_t>(GfxMpsrtStorage::Matrix)) &&
           (desc.dtype == static_cast<uint32_t>(GfxMpsrtDType::F16) ||
            desc.dtype == static_cast<uint32_t>(GfxMpsrtDType::F32) ||
            desc.dtype == static_cast<uint32_t>(GfxMpsrtDType::I8) ||
            desc.dtype == static_cast<uint32_t>(GfxMpsrtDType::U8) ||
            desc.dtype == static_cast<uint32_t>(GfxMpsrtDType::I32) ||
            desc.dtype == static_cast<uint32_t>(GfxMpsrtDType::U32)) &&
           desc.matrix_rows != 0 &&
           desc.matrix_columns != 0 &&
           desc.matrix_row_bytes != 0 &&
           desc.matrix_count != 0;
}

inline bool gfx_mpsrt_ndarray_bridge_supported(const GfxMpsrtTensorAbiDesc& desc) {
    return desc.storage == static_cast<uint32_t>(GfxMpsrtStorage::NDArray) &&
           desc.rank != 0 &&
           desc.byte_length != 0;
}

inline bool gfx_mpsrt_buffer_bridge_supported(const GfxMpsrtTensorAbiDesc& desc) {
    return desc.storage == static_cast<uint32_t>(GfxMpsrtStorage::Buffer) &&
           desc.rank != 0 &&
           desc.byte_length != 0;
}

inline GfxMpsrtStorageBridgeDirection gfx_mpsrt_external_image_bridge_direction(bool external_output) {
    return external_output ? GfxMpsrtStorageBridgeDirection::ImageToBuffer
                           : GfxMpsrtStorageBridgeDirection::BufferToImage;
}

inline GfxMpsrtStorageBridgeDirection gfx_mpsrt_external_bridge_direction_for_storage(
    GfxMpsrtStorage storage,
    bool external_output) {
    switch (storage) {
        case GfxMpsrtStorage::Image:
            return external_output ? GfxMpsrtStorageBridgeDirection::ImageToBuffer
                                   : GfxMpsrtStorageBridgeDirection::BufferToImage;
        case GfxMpsrtStorage::Matrix:
            return external_output ? GfxMpsrtStorageBridgeDirection::MatrixToBuffer
                                   : GfxMpsrtStorageBridgeDirection::BufferToMatrix;
        case GfxMpsrtStorage::NDArray:
            return external_output ? GfxMpsrtStorageBridgeDirection::NDArrayToBuffer
                                   : GfxMpsrtStorageBridgeDirection::BufferToNDArray;
        case GfxMpsrtStorage::Alias:
            return GfxMpsrtStorageBridgeDirection::Alias;
        case GfxMpsrtStorage::Buffer:
        case GfxMpsrtStorage::Unknown:
        default:
            return GfxMpsrtStorageBridgeDirection::Unknown;
    }
}

inline bool gfx_mpsrt_storage_bridge_reads_external_buffer(GfxMpsrtStorageBridgeDirection direction) {
    return direction == GfxMpsrtStorageBridgeDirection::BufferToImage ||
           direction == GfxMpsrtStorageBridgeDirection::BufferToMatrix ||
           direction == GfxMpsrtStorageBridgeDirection::BufferToNDArray;
}

inline bool gfx_mpsrt_storage_bridge_writes_external_buffer(GfxMpsrtStorageBridgeDirection direction) {
    return direction == GfxMpsrtStorageBridgeDirection::ImageToBuffer ||
           direction == GfxMpsrtStorageBridgeDirection::MatrixToBuffer ||
           direction == GfxMpsrtStorageBridgeDirection::NDArrayToBuffer;
}

inline GfxMpsrtStorage gfx_mpsrt_storage_bridge_source_storage(GfxMpsrtStorageBridgeDirection direction) {
    switch (direction) {
        case GfxMpsrtStorageBridgeDirection::BufferToImage:
        case GfxMpsrtStorageBridgeDirection::BufferToMatrix:
        case GfxMpsrtStorageBridgeDirection::BufferToNDArray:
            return GfxMpsrtStorage::Buffer;
        case GfxMpsrtStorageBridgeDirection::ImageToBuffer:
            return GfxMpsrtStorage::Image;
        case GfxMpsrtStorageBridgeDirection::MatrixToBuffer:
            return GfxMpsrtStorage::Matrix;
        case GfxMpsrtStorageBridgeDirection::NDArrayToBuffer:
            return GfxMpsrtStorage::NDArray;
        case GfxMpsrtStorageBridgeDirection::Alias:
            return GfxMpsrtStorage::Alias;
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
        case GfxMpsrtStorageBridgeDirection::MatrixToBuffer:
        case GfxMpsrtStorageBridgeDirection::NDArrayToBuffer:
            return GfxMpsrtStorage::Buffer;
        case GfxMpsrtStorageBridgeDirection::BufferToMatrix:
            return GfxMpsrtStorage::Matrix;
        case GfxMpsrtStorageBridgeDirection::BufferToNDArray:
            return GfxMpsrtStorage::NDArray;
        case GfxMpsrtStorageBridgeDirection::Alias:
            return GfxMpsrtStorage::Alias;
        case GfxMpsrtStorageBridgeDirection::Unknown:
        default:
            return GfxMpsrtStorage::Unknown;
    }
}

inline bool gfx_mpsrt_storage_bridge_target_supported(const GfxMpsrtTensorAbiDesc& tensor,
                                                      GfxMpsrtStorage target_storage) {
    switch (target_storage) {
        case GfxMpsrtStorage::Image:
            return gfx_mpsrt_image_bridge_supported(tensor);
        case GfxMpsrtStorage::Matrix:
            return gfx_mpsrt_matrix_bridge_supported(tensor);
        case GfxMpsrtStorage::NDArray:
            return gfx_mpsrt_ndarray_bridge_supported(tensor);
        case GfxMpsrtStorage::Buffer:
            return gfx_mpsrt_buffer_bridge_supported(tensor) ||
                   gfx_mpsrt_image_bridge_supported(tensor) ||
                   gfx_mpsrt_matrix_bridge_supported(tensor) ||
                   gfx_mpsrt_ndarray_bridge_supported(tensor);
        case GfxMpsrtStorage::Alias:
            return tensor.storage == static_cast<uint32_t>(GfxMpsrtStorage::Alias);
        case GfxMpsrtStorage::Unknown:
        default:
            return false;
    }
}

inline bool gfx_mpsrt_make_storage_bridge_desc(GfxMpsrtValue value,
                                               const GfxMpsrtTensorAbiDesc& tensor,
                                               GfxMpsrtStorageBridgeDirection direction,
                                               GfxMpsrtStorageBridgeDesc& bridge) {
    if (direction == GfxMpsrtStorageBridgeDirection::Unknown) {
        return false;
    }
    const auto source_storage = gfx_mpsrt_storage_bridge_source_storage(direction);
    const auto target_storage = gfx_mpsrt_storage_bridge_target_storage(direction);
    if (source_storage == GfxMpsrtStorage::Unknown ||
        target_storage == GfxMpsrtStorage::Unknown ||
        !gfx_mpsrt_storage_bridge_target_supported(tensor, target_storage)) {
        return false;
    }
    bridge.value = value;
    bridge.direction = direction;
    bridge.source_storage = source_storage;
    bridge.target_storage = target_storage;
    bridge.tensor = tensor;
    return true;
}

inline bool gfx_mpsrt_make_image_bridge_desc(GfxMpsrtValue value,
                                             const GfxMpsrtTensorAbiDesc& tensor,
                                             GfxMpsrtStorageBridgeDirection direction,
                                             GfxMpsrtStorageBridgeDesc& bridge) {
    if (direction != GfxMpsrtStorageBridgeDirection::BufferToImage &&
        direction != GfxMpsrtStorageBridgeDirection::ImageToBuffer) {
        return false;
    }
    return gfx_mpsrt_make_storage_bridge_desc(value, tensor, direction, bridge);
}

inline const char* gfx_mpsrt_storage_bridge_direction_name(GfxMpsrtStorageBridgeDirection direction) {
    switch (direction) {
        case GfxMpsrtStorageBridgeDirection::BufferToImage:
            return "buffer_to_image";
        case GfxMpsrtStorageBridgeDirection::ImageToBuffer:
            return "image_to_buffer";
        case GfxMpsrtStorageBridgeDirection::BufferToMatrix:
            return "buffer_to_matrix";
        case GfxMpsrtStorageBridgeDirection::MatrixToBuffer:
            return "matrix_to_buffer";
        case GfxMpsrtStorageBridgeDirection::BufferToNDArray:
            return "buffer_to_ndarray";
        case GfxMpsrtStorageBridgeDirection::NDArrayToBuffer:
            return "ndarray_to_buffer";
        case GfxMpsrtStorageBridgeDirection::Alias:
            return "alias";
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
    if (name == "buffer_to_matrix") {
        return GfxMpsrtStorageBridgeDirection::BufferToMatrix;
    }
    if (name == "matrix_to_buffer") {
        return GfxMpsrtStorageBridgeDirection::MatrixToBuffer;
    }
    if (name == "buffer_to_ndarray") {
        return GfxMpsrtStorageBridgeDirection::BufferToNDArray;
    }
    if (name == "ndarray_to_buffer") {
        return GfxMpsrtStorageBridgeDirection::NDArrayToBuffer;
    }
    if (name == "alias") {
        return GfxMpsrtStorageBridgeDirection::Alias;
    }
    return GfxMpsrtStorageBridgeDirection::Unknown;
}

}  // namespace gfx_plugin
}  // namespace ov
