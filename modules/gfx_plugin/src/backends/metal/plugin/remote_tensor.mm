// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gfx_remote_context.hpp"

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
    auto it = params.find(kGfxStorageModeProperty);
    if (it == params.end())
        it = params.find(kStorageModeProperty);
    if (it == params.end())
        return static_cast<uint32_t>(MTLStorageModeShared);
    if (it->second.is<int>()) {
        return static_cast<uint32_t>(it->second.as<int>());
    }
    if (it->second.is<uint32_t>()) {
        return it->second.as<uint32_t>();
    }
    if (it->second.is<std::string>()) {
        auto mode = ov::util::to_lower(it->second.as<std::string>());
        if (mode == "private")
            return static_cast<uint32_t>(MTLStorageModePrivate);
        if (mode == "managed")
            return static_cast<uint32_t>(MTLStorageModeManaged);
        return static_cast<uint32_t>(MTLStorageModeShared);
    }
    return static_cast<uint32_t>(MTLStorageModeShared);
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

    bool owns_buffer = false;
    void* external = find_any_ptr(params, {kMtlBufferProperty, kGfxBufferProperty});

    if (external) {
        tensor.buf.buffer = external;
        tensor.buf.from_handle = true;
        tensor.buf.external = true;
#ifdef __OBJC__
        auto mb = static_cast<id<MTLBuffer>>(external);
        const size_t buf_len = static_cast<size_t>(mb.length);
        OPENVINO_ASSERT(buf_len >= bytes,
                        "GFX: remote MTLBuffer is smaller than required (",
                        buf_len,
                        " < ",
                        bytes,
                        ")");
        tensor.buf.size = buf_len;
        tensor.buf.storage_mode = static_cast<uint32_t>(mb.storageMode);
        tensor.buf.host_visible = (mb.storageMode != MTLStorageModePrivate);
#else
        tensor.buf.size = bytes;
#endif
    } else {
        OPENVINO_ASSERT(device, "GFX: remote context device handle is null");
#ifdef __OBJC__
        auto dev = static_cast<id<MTLDevice>>(device);
        const uint32_t storage_mode = parse_storage_mode(params);
        id<MTLBuffer> buf = [dev newBufferWithLength:bytes options:options_from_storage(storage_mode)];
        OPENVINO_ASSERT(buf, "GFX: failed to allocate remote buffer");
        tensor.buf.buffer = buf;
        tensor.buf.size = bytes;
        tensor.buf.storage_mode = static_cast<uint32_t>(buf.storageMode);
        tensor.buf.host_visible = (buf.storageMode != MTLStorageModePrivate);
        owns_buffer = true;
#else
        OPENVINO_THROW("GFX remote tensor requires Objective-C++ (Metal)");
#endif
    }

    result.tensor = tensor;
    result.owns_buffer = owns_buffer;
    return result;
}

void release_metal_remote_tensor(GpuTensor& tensor, bool owns_buffer) {
#ifdef __OBJC__
    if (!owns_buffer || tensor.buf.backend != GpuBackend::Metal || !tensor.buf.buffer) {
        return;
    }
    [static_cast<id<MTLBuffer>>(tensor.buf.buffer) release];
    tensor.buf.buffer = nullptr;
#endif
}

}  // namespace gfx_plugin
}  // namespace ov
