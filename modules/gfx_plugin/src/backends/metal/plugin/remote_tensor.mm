// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/plugin/remote_tensor.hpp"

#import <Metal/Metal.h>

#include "openvino/core/except.hpp"
#include "openvino/util/common_util.hpp"
#include "plugin/gfx_remote_utils.hpp"
#include "backends/metal/plugin/metal_properties.hpp"
#include "backends/metal/runtime/metal_memory.hpp"

namespace ov {
namespace gfx_plugin {
namespace {
uint32_t parse_storage_mode(const ov::AnyMap& params) {
    bool explicit_mode = false;
    uint32_t resolved = static_cast<uint32_t>(MTLStorageModeShared);
    auto it = params.find(kGfxStorageModeProperty);
    if (it == params.end())
        it = params.find(kStorageModeProperty);
    if (it != params.end()) {
        explicit_mode = true;
        if (it->second.is<int>()) {
            resolved = static_cast<uint32_t>(it->second.as<int>());
        } else if (it->second.is<uint32_t>()) {
            resolved = it->second.as<uint32_t>();
        } else if (it->second.is<std::string>()) {
            auto mode = ov::util::to_lower(it->second.as<std::string>());
            if (mode == "private")
                resolved = static_cast<uint32_t>(MTLStorageModePrivate);
            else if (mode == "managed")
                resolved = static_cast<uint32_t>(MTLStorageModeManaged);
            else
                resolved = static_cast<uint32_t>(MTLStorageModeShared);
        }
    }
    if (!explicit_mode) {
        const bool host_visible =
            find_any_bool(params, {kGfxHostVisibleProperty, "HOST_VISIBLE"}, /*fallback=*/false);
        resolved = host_visible ? static_cast<uint32_t>(MTLStorageModeShared)
                                : static_cast<uint32_t>(MTLStorageModePrivate);
    }
    return resolved;
}

MTLResourceOptions options_from_storage(uint32_t mode) {
    switch (static_cast<MTLStorageMode>(mode)) {
        case MTLStorageModePrivate:
            return MTLResourceStorageModePrivate;
        case MTLStorageModeManaged:
            return MTLResourceStorageModeManaged;
        case MTLStorageModeShared:
        default:
            return MTLResourceStorageModeShared;
    }
}

}  // namespace

static void release_metal_remote_tensor(GpuTensor& tensor) {
#ifdef __OBJC__
    if (!tensor.buf.owned || tensor.buf.backend != GpuBackend::Metal || !tensor.buf.buffer) {
        return;
    }
    [static_cast<id<MTLBuffer>>(tensor.buf.buffer) release];
    tensor.buf.buffer = nullptr;
#endif
}

RemoteTensorCreateResult create_metal_remote_tensor(const ov::element::Type& type,
                                                    const ov::Shape& shape,
                                                    const ov::AnyMap& params,
                                                    GpuDeviceHandle device,
                                                    size_t bytes) {
    RemoteTensorCreateResult result;
    GpuTensor tensor;
    tensor.shape = shape;
    tensor.expected_type = type;
    tensor.buf.type = type;
    tensor.buf.backend = GpuBackend::Metal;

    void* external = find_any_ptr(params, {kMtlBufferProperty, kGfxBufferProperty});
    const size_t declared_bytes = find_any_size(params, {kGfxBufferBytesProperty}, 0);

    if (external) {
        tensor.buf.buffer = external;
        tensor.buf.from_handle = true;
        tensor.buf.external = true;
        tensor.buf.owned = false;
#ifdef __OBJC__
        auto mb = static_cast<id<MTLBuffer>>(external);
        const size_t buf_len = static_cast<size_t>(mb.length);
        OPENVINO_ASSERT(buf_len >= bytes,
                        "GFX: remote MTLBuffer is smaller than required (",
                        buf_len,
                        " < ",
                        bytes,
                        ")");
        if (declared_bytes) {
            OPENVINO_ASSERT(declared_bytes == buf_len,
                            "GFX: remote MTLBuffer size mismatch (declared ",
                            declared_bytes,
                            ", actual ",
                            buf_len,
                            ")");
        }
        tensor.buf.size = buf_len;
        tensor.buf.storage_mode = static_cast<uint32_t>(mb.storageMode);
        tensor.buf.host_visible = (mb.storageMode != MTLStorageModePrivate);
#else
        OPENVINO_ASSERT(!declared_bytes || declared_bytes >= bytes,
                        "GFX: remote buffer bytes smaller than required (",
                        declared_bytes,
                        " < ",
                        bytes,
                        ")");
        tensor.buf.size = declared_bytes ? declared_bytes : bytes;
#endif
    } else {
        OPENVINO_ASSERT(device, "GFX: remote context device handle is null");
        if (declared_bytes) {
            OPENVINO_ASSERT(declared_bytes == bytes,
                            "GFX: remote tensor bytes mismatch (declared ",
                            declared_bytes,
                            ", required ",
                            bytes,
                            ")");
        }
#ifdef __OBJC__
        auto dev = static_cast<id<MTLDevice>>(device);
        const uint32_t storage_mode = parse_storage_mode(params);
        id<MTLBuffer> buf = [dev newBufferWithLength:bytes options:options_from_storage(storage_mode)];
        OPENVINO_ASSERT(buf, "GFX: failed to allocate remote buffer");
        tensor.buf.buffer = buf;
        tensor.buf.size = bytes;
        tensor.buf.storage_mode = static_cast<uint32_t>(buf.storageMode);
        tensor.buf.host_visible = (buf.storageMode != MTLStorageModePrivate);
        tensor.buf.owned = true;
#else
        OPENVINO_THROW("GFX remote tensor requires Objective-C++ (Metal)");
#endif
    }

    result.tensor = tensor;
    result.properties[kGfxBufferProperty] = tensor.buf.buffer;
    result.properties[kGfxBufferBytesProperty] = tensor.buf.size;
    result.properties[kGfxHostVisibleProperty] = tensor.buf.host_visible;
    result.properties[kGfxStorageModeProperty] = tensor.buf.storage_mode;
    result.properties[kMtlBufferProperty] = tensor.buf.buffer;
    result.properties[kStorageModeProperty] = tensor.buf.storage_mode;
    result.release_fn = &release_metal_remote_tensor;
    return result;
}

}  // namespace gfx_plugin
}  // namespace ov
