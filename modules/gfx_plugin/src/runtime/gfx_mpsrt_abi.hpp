// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "runtime/gfx_stage_policy.hpp"

namespace ov {
namespace gfx_plugin {

enum class GfxMpsrtDType : uint32_t {
    Unknown = 0,
    F16 = 1,
    F32 = 2,
    I8 = 3,
    U8 = 4,
    I32 = 5,
    I64 = 6,
    Bool = 7,
    I16 = 8,
    U16 = 9,
    U32 = 10,
    U64 = 11,
};

enum class GfxMpsrtStorage : uint32_t {
    Unknown = 0,
    Buffer = 1,
    Image = 2,
    Matrix = 3,
    NDArray = 4,
    Alias = 5,
};

enum class GfxMpsrtLayout : uint32_t {
    Unknown = 0,
    Linear = 1,
    NCHW = 2,
    NHWC = 3,
    NHWC4 = 4,
    OHWI = 5,
    RowMajor = 6,
    KVCacheInterleaved = 7,
};

enum GfxMpsrtTensorFlags : uint32_t {
    GfxMpsrtTensorFlagNone = 0,
    GfxMpsrtTensorFlagConst = 1u << 0,
    GfxMpsrtTensorFlagExternalIo = 1u << 1,
    GfxMpsrtTensorFlagTransient = 1u << 2,
    GfxMpsrtTensorFlagRemote = 1u << 3,
    GfxMpsrtTensorFlagCpuVisible = 1u << 4,
    GfxMpsrtTensorFlagAllowAlias = 1u << 5,
    GfxMpsrtTensorFlagDynamicShape = 1u << 6,
};

using GfxMpsrtValue = uint32_t;

enum class GfxMpsrtExternalBufferRole : uint32_t {
    Unknown = 0,
    TensorInput = 1,
    TensorOutput = 2,
    ConstBuffer = 3,
    RuntimeParams = 4,
    Metadata = 5,
};

enum class GfxMpsrtStatus : uint32_t {
    Ok = 0,
    InvalidArgument = 1,
    Unsupported = 2,
    RuntimeError = 3,
};

struct GfxMpsrtTensorDesc {
    uint32_t rank = 0;
    std::array<uint64_t, 8> dims{};
    std::array<int64_t, 8> strides{};
    GfxMpsrtDType dtype = GfxMpsrtDType::Unknown;
    GfxMpsrtStorage storage = GfxMpsrtStorage::Unknown;
    GfxMpsrtLayout layout = GfxMpsrtLayout::Unknown;
    uint32_t flags = GfxMpsrtTensorFlagNone;
    uint64_t byte_offset = 0;
    uint64_t byte_length = 0;
    uint32_t image_width = 0;
    uint32_t image_height = 0;
    uint32_t image_feature_channels = 0;
    uint32_t image_batch = 0;
    uint32_t matrix_rows = 0;
    uint32_t matrix_columns = 0;
    uint32_t matrix_row_bytes = 0;
    uint32_t matrix_count = 0;
    uint32_t alias_of = 0;
};

struct GfxMpsrtTensorAbiDesc {
    uint32_t rank = 0;
    uint64_t dims[8] = {};
    int64_t strides[8] = {};
    uint32_t dtype = static_cast<uint32_t>(GfxMpsrtDType::Unknown);
    uint32_t storage = static_cast<uint32_t>(GfxMpsrtStorage::Unknown);
    uint32_t layout = static_cast<uint32_t>(GfxMpsrtLayout::Unknown);
    uint32_t flags = GfxMpsrtTensorFlagNone;
    uint64_t byte_offset = 0;
    uint64_t byte_length = 0;
    uint32_t image_width = 0;
    uint32_t image_height = 0;
    uint32_t image_feature_channels = 0;
    uint32_t image_batch = 0;
    uint32_t matrix_rows = 0;
    uint32_t matrix_columns = 0;
    uint32_t matrix_row_bytes = 0;
    uint32_t matrix_count = 0;
    uint32_t alias_of = 0;
};

struct GfxMpsrtConv2DAbiDesc {
    uint32_t groups = 1;
    uint32_t strides[2] = {1, 1};
    uint32_t dilations[2] = {1, 1};
    uint32_t pads[4] = {0, 0, 0, 0};
    uint32_t fused_activation = 0;
    uint32_t accumulate_fp32 = 1;
};

struct GfxMpsrtGemmAbiDesc {
    uint32_t transpose_lhs = 0;
    uint32_t transpose_rhs = 0;
    uint32_t accumulate_fp32 = 1;
    float alpha = 1.0f;
    float beta = 0.0f;
};

enum GfxMpsrtMslDispatchFlags : uint32_t {
    GfxMpsrtMslDispatchFlagNone = 0,
    GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired = 1u << 0,
};

struct GfxMpsrtMslDispatchAbiDesc {
    uint32_t kernel_family = 0;
    uint32_t storage = static_cast<uint32_t>(GfxMpsrtStorage::Buffer);
    uint32_t layout = static_cast<uint32_t>(GfxMpsrtLayout::Linear);
    uint32_t threads_per_threadgroup = 1;
    uint32_t input_count = 0;
    uint32_t output_count = 0;
    uint32_t flags = GfxMpsrtMslDispatchFlagNone;
    uint32_t reserved = 0;
};

inline GfxMpsrtDType gfx_mpsrt_dtype_from_element(const ov::element::Type& type) {
    if (type == ov::element::f16) {
        return GfxMpsrtDType::F16;
    }
    if (type == ov::element::f32) {
        return GfxMpsrtDType::F32;
    }
    if (type == ov::element::i8) {
        return GfxMpsrtDType::I8;
    }
    if (type == ov::element::u8) {
        return GfxMpsrtDType::U8;
    }
    if (type == ov::element::i16) {
        return GfxMpsrtDType::I16;
    }
    if (type == ov::element::u16) {
        return GfxMpsrtDType::U16;
    }
    if (type == ov::element::i32) {
        return GfxMpsrtDType::I32;
    }
    if (type == ov::element::u32) {
        return GfxMpsrtDType::U32;
    }
    if (type == ov::element::i64) {
        return GfxMpsrtDType::I64;
    }
    if (type == ov::element::u64) {
        return GfxMpsrtDType::U64;
    }
    if (type == ov::element::boolean) {
        return GfxMpsrtDType::Bool;
    }
    return GfxMpsrtDType::Unknown;
}

inline uint32_t gfx_mpsrt_element_size_bytes(GfxMpsrtDType dtype) {
    switch (dtype) {
        case GfxMpsrtDType::F16:
            return 2;
        case GfxMpsrtDType::F32:
            return 4;
        case GfxMpsrtDType::I8:
        case GfxMpsrtDType::U8:
        case GfxMpsrtDType::Bool:
            return 1;
        case GfxMpsrtDType::I16:
        case GfxMpsrtDType::U16:
            return 2;
        case GfxMpsrtDType::I32:
        case GfxMpsrtDType::U32:
            return 4;
        case GfxMpsrtDType::I64:
        case GfxMpsrtDType::U64:
            return 8;
        case GfxMpsrtDType::Unknown:
        default:
            return 0;
    }
}

inline GfxMpsrtStorage gfx_mpsrt_storage_from_stage_storage(GfxStageStorageKind storage) {
    switch (storage) {
        case GfxStageStorageKind::Buffer:
            return GfxMpsrtStorage::Buffer;
        case GfxStageStorageKind::Image:
            return GfxMpsrtStorage::Image;
        case GfxStageStorageKind::Matrix:
            return GfxMpsrtStorage::Matrix;
        case GfxStageStorageKind::NDArray:
            return GfxMpsrtStorage::NDArray;
        case GfxStageStorageKind::Alias:
            return GfxMpsrtStorage::Alias;
        case GfxStageStorageKind::Unknown:
        default:
            return GfxMpsrtStorage::Unknown;
    }
}

inline GfxMpsrtLayout gfx_mpsrt_default_layout(GfxMpsrtStorage storage, uint32_t rank) {
    switch (storage) {
        case GfxMpsrtStorage::Image:
            return rank == 4 ? GfxMpsrtLayout::NCHW : GfxMpsrtLayout::NHWC;
        case GfxMpsrtStorage::Matrix:
        case GfxMpsrtStorage::NDArray:
            return GfxMpsrtLayout::RowMajor;
        case GfxMpsrtStorage::Alias:
            return GfxMpsrtLayout::Linear;
        case GfxMpsrtStorage::Buffer:
            return GfxMpsrtLayout::Linear;
        case GfxMpsrtStorage::Unknown:
        default:
            return GfxMpsrtLayout::Unknown;
    }
}

inline std::vector<int64_t> gfx_mpsrt_dense_strides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    int64_t running = 1;
    for (size_t i = dims.size(); i > 0; --i) {
        const size_t idx = i - 1;
        strides[idx] = running;
        if (dims[idx] < 0 || running < 0) {
            running = -1;
        } else {
            running *= dims[idx];
        }
    }
    return strides;
}

inline GfxMpsrtTensorDesc gfx_mpsrt_make_tensor_desc(const std::vector<int64_t>& dims,
                                                     const ov::element::Type& element_type,
                                                     GfxStageStorageKind stage_storage,
                                                     uint32_t flags = GfxMpsrtTensorFlagNone) {
    GfxMpsrtTensorDesc desc{};
    desc.rank = static_cast<uint32_t>(std::min<size_t>(dims.size(), desc.dims.size()));
    desc.dtype = gfx_mpsrt_dtype_from_element(element_type);
    desc.storage = gfx_mpsrt_storage_from_stage_storage(stage_storage);
    desc.layout = gfx_mpsrt_default_layout(desc.storage, desc.rank);
    desc.flags = flags;

    const auto strides = gfx_mpsrt_dense_strides(dims);
    uint64_t element_count = 1;
    bool dynamic = false;
    for (uint32_t i = 0; i < desc.rank; ++i) {
        if (dims[i] < 0) {
            dynamic = true;
            desc.dims[i] = 0;
        } else {
            desc.dims[i] = static_cast<uint64_t>(dims[i]);
            element_count *= desc.dims[i];
        }
        desc.strides[i] = i < strides.size() ? strides[i] : 0;
    }
    if (dynamic) {
        desc.flags |= GfxMpsrtTensorFlagDynamicShape;
        element_count = 0;
    }
    desc.byte_length = element_count * gfx_mpsrt_element_size_bytes(desc.dtype);

    if (desc.storage == GfxMpsrtStorage::Image && desc.rank == 4) {
        desc.image_batch = static_cast<uint32_t>(desc.dims[0]);
        desc.image_feature_channels = static_cast<uint32_t>(desc.dims[1]);
        desc.image_height = static_cast<uint32_t>(desc.dims[2]);
        desc.image_width = static_cast<uint32_t>(desc.dims[3]);
    } else if (desc.storage == GfxMpsrtStorage::Matrix && desc.rank >= 2) {
        desc.matrix_rows = static_cast<uint32_t>(desc.dims[desc.rank - 2]);
        desc.matrix_columns = static_cast<uint32_t>(desc.dims[desc.rank - 1]);
        desc.matrix_row_bytes = desc.matrix_columns * gfx_mpsrt_element_size_bytes(desc.dtype);
        desc.matrix_count = 1;
        for (uint32_t i = 0; i + 2 < desc.rank; ++i) {
            desc.matrix_count *= static_cast<uint32_t>(desc.dims[i]);
        }
    }
    return desc;
}

inline GfxMpsrtTensorAbiDesc gfx_mpsrt_to_abi_desc(const GfxMpsrtTensorDesc& desc) {
    GfxMpsrtTensorAbiDesc abi{};
    abi.rank = desc.rank;
    for (uint32_t i = 0; i < abi.rank && i < desc.dims.size(); ++i) {
        abi.dims[i] = desc.dims[i];
        abi.strides[i] = desc.strides[i];
    }
    abi.dtype = static_cast<uint32_t>(desc.dtype);
    abi.storage = static_cast<uint32_t>(desc.storage);
    abi.layout = static_cast<uint32_t>(desc.layout);
    abi.flags = desc.flags;
    abi.byte_offset = desc.byte_offset;
    abi.byte_length = desc.byte_length;
    abi.image_width = desc.image_width;
    abi.image_height = desc.image_height;
    abi.image_feature_channels = desc.image_feature_channels;
    abi.image_batch = desc.image_batch;
    abi.matrix_rows = desc.matrix_rows;
    abi.matrix_columns = desc.matrix_columns;
    abi.matrix_row_bytes = desc.matrix_row_bytes;
    abi.matrix_count = desc.matrix_count;
    abi.alias_of = desc.alias_of;
    return abi;
}

inline GfxMpsrtTensorDesc gfx_mpsrt_from_abi_desc(const GfxMpsrtTensorAbiDesc& abi) {
    GfxMpsrtTensorDesc desc{};
    desc.rank = std::min<uint32_t>(abi.rank, static_cast<uint32_t>(desc.dims.size()));
    for (uint32_t i = 0; i < desc.rank; ++i) {
        desc.dims[i] = abi.dims[i];
        desc.strides[i] = abi.strides[i];
    }
    desc.dtype = static_cast<GfxMpsrtDType>(abi.dtype);
    desc.storage = static_cast<GfxMpsrtStorage>(abi.storage);
    desc.layout = static_cast<GfxMpsrtLayout>(abi.layout);
    desc.flags = abi.flags;
    desc.byte_offset = abi.byte_offset;
    desc.byte_length = abi.byte_length;
    desc.image_width = abi.image_width;
    desc.image_height = abi.image_height;
    desc.image_feature_channels = abi.image_feature_channels;
    desc.image_batch = abi.image_batch;
    desc.matrix_rows = abi.matrix_rows;
    desc.matrix_columns = abi.matrix_columns;
    desc.matrix_row_bytes = abi.matrix_row_bytes;
    desc.matrix_count = abi.matrix_count;
    desc.alias_of = abi.alias_of;
    return desc;
}

inline const char* gfx_mpsrt_dtype_name(GfxMpsrtDType dtype) {
    switch (dtype) {
        case GfxMpsrtDType::F16:
            return "f16";
        case GfxMpsrtDType::F32:
            return "f32";
        case GfxMpsrtDType::I8:
            return "i8";
        case GfxMpsrtDType::U8:
            return "u8";
        case GfxMpsrtDType::I16:
            return "i16";
        case GfxMpsrtDType::U16:
            return "u16";
        case GfxMpsrtDType::I32:
            return "i32";
        case GfxMpsrtDType::U32:
            return "u32";
        case GfxMpsrtDType::I64:
            return "i64";
        case GfxMpsrtDType::U64:
            return "u64";
        case GfxMpsrtDType::Bool:
            return "bool";
        case GfxMpsrtDType::Unknown:
        default:
            return "unknown";
    }
}

inline GfxMpsrtDType gfx_mpsrt_dtype_from_name(std::string_view name) {
    if (name == "f16") return GfxMpsrtDType::F16;
    if (name == "f32") return GfxMpsrtDType::F32;
    if (name == "i8") return GfxMpsrtDType::I8;
    if (name == "u8") return GfxMpsrtDType::U8;
    if (name == "i16") return GfxMpsrtDType::I16;
    if (name == "u16") return GfxMpsrtDType::U16;
    if (name == "i32") return GfxMpsrtDType::I32;
    if (name == "u32") return GfxMpsrtDType::U32;
    if (name == "i64") return GfxMpsrtDType::I64;
    if (name == "u64") return GfxMpsrtDType::U64;
    if (name == "bool") return GfxMpsrtDType::Bool;
    return GfxMpsrtDType::Unknown;
}

inline const char* gfx_mpsrt_storage_name(GfxMpsrtStorage storage) {
    switch (storage) {
        case GfxMpsrtStorage::Buffer:
            return "buffer";
        case GfxMpsrtStorage::Image:
            return "image";
        case GfxMpsrtStorage::Matrix:
            return "matrix";
        case GfxMpsrtStorage::NDArray:
            return "ndarray";
        case GfxMpsrtStorage::Alias:
            return "alias";
        case GfxMpsrtStorage::Unknown:
        default:
            return "unknown";
    }
}

inline GfxMpsrtStorage gfx_mpsrt_storage_from_name(std::string_view name) {
    if (name == "buffer") return GfxMpsrtStorage::Buffer;
    if (name == "image") return GfxMpsrtStorage::Image;
    if (name == "matrix") return GfxMpsrtStorage::Matrix;
    if (name == "ndarray") return GfxMpsrtStorage::NDArray;
    if (name == "alias") return GfxMpsrtStorage::Alias;
    return GfxMpsrtStorage::Unknown;
}

inline const char* gfx_mpsrt_layout_name(GfxMpsrtLayout layout) {
    switch (layout) {
        case GfxMpsrtLayout::Linear:
            return "linear";
        case GfxMpsrtLayout::NCHW:
            return "nchw";
        case GfxMpsrtLayout::NHWC:
            return "nhwc";
        case GfxMpsrtLayout::NHWC4:
            return "nhwc4";
        case GfxMpsrtLayout::OHWI:
            return "ohwi";
        case GfxMpsrtLayout::RowMajor:
            return "row_major";
        case GfxMpsrtLayout::KVCacheInterleaved:
            return "kv_cache_interleaved";
        case GfxMpsrtLayout::Unknown:
        default:
            return "unknown";
    }
}

inline GfxMpsrtLayout gfx_mpsrt_layout_from_name(std::string_view name) {
    if (name == "linear") return GfxMpsrtLayout::Linear;
    if (name == "nchw") return GfxMpsrtLayout::NCHW;
    if (name == "nhwc") return GfxMpsrtLayout::NHWC;
    if (name == "nhwc4") return GfxMpsrtLayout::NHWC4;
    if (name == "ohwi") return GfxMpsrtLayout::OHWI;
    if (name == "row_major") return GfxMpsrtLayout::RowMajor;
    if (name == "kv_cache_interleaved") return GfxMpsrtLayout::KVCacheInterleaved;
    return GfxMpsrtLayout::Unknown;
}

}  // namespace gfx_plugin
}  // namespace ov
