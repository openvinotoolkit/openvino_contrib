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
struct StorageModeResult {
    uint32_t mode = static_cast<uint32_t>(MTLStorageModeShared);
    bool explicit_mode = false;
    bool host_visible = false;
};

StorageModeResult parse_storage_mode(const ov::AnyMap& params) {
    StorageModeResult result;
    auto it = params.find(kGfxStorageModeProperty);
    if (it == params.end())
        it = params.find(kStorageModeProperty);
    if (it != params.end()) {
        result.explicit_mode = true;
        if (it->second.is<int>()) {
            result.mode = static_cast<uint32_t>(it->second.as<int>());
        } else if (it->second.is<uint32_t>()) {
            result.mode = it->second.as<uint32_t>();
        } else if (it->second.is<std::string>()) {
            auto mode = ov::util::to_lower(it->second.as<std::string>());
            if (mode == "private")
                result.mode = static_cast<uint32_t>(MTLStorageModePrivate);
            else if (mode == "managed")
                result.mode = static_cast<uint32_t>(MTLStorageModeManaged);
            else
                result.mode = static_cast<uint32_t>(MTLStorageModeShared);
        }
        result.host_visible = (result.mode != static_cast<uint32_t>(MTLStorageModePrivate));
        return result;
    }

    const bool host_visible =
        find_any_bool(params, {kGfxHostVisibleProperty, "HOST_VISIBLE"}, /*fallback=*/false);
    result.explicit_mode = host_visible;
    result.host_visible = host_visible;
    result.mode = host_visible ? static_cast<uint32_t>(MTLStorageModeShared)
                               : static_cast<uint32_t>(MTLStorageModePrivate);
    return result;
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

class MetalRemoteAllocator final : public IGpuAllocator {
public:
    MetalRemoteAllocator(MetalDeviceHandle device, uint32_t storage_mode, bool explicit_mode)
        : m_device(device), m_storage_mode(storage_mode), m_explicit_mode(explicit_mode) {}

    GpuBackend backend() const override { return GpuBackend::Metal; }

    GpuBuffer allocate(const GpuBufferDesc& desc) override {
#ifdef __OBJC__
        validate_gpu_buffer_desc(desc, "GFX Metal");
        auto dev = static_cast<id<MTLDevice>>(m_device);
        OPENVINO_ASSERT(dev, "GFX Metal: device is null");
        const uint32_t resolved_mode = resolve_storage_mode(desc);
        id<MTLBuffer> buf = [dev newBufferWithLength:desc.bytes
                                             options:options_from_storage(resolved_mode)];
        OPENVINO_ASSERT(buf, "GFX Metal: failed to allocate remote buffer");
        GpuBuffer out;
        out.buffer = buf;
        out.size = desc.bytes;
        out.type = desc.type;
        out.backend = GpuBackend::Metal;
        out.storage_mode = static_cast<uint32_t>(buf.storageMode);
        out.options_mask = static_cast<uint32_t>(options_from_storage(out.storage_mode));
        out.host_visible = (buf.storageMode != MTLStorageModePrivate);
        out.owned = true;
        return out;
#else
        (void)desc;
        OPENVINO_THROW("GFX remote tensor requires Objective-C++ (Metal)");
#endif
    }

    GpuBuffer wrap_shared(void* ptr, size_t bytes, ov::element::Type type) override {
#ifdef __OBJC__
        GpuBuffer out;
        if (!ptr || bytes == 0) {
            return out;
        }
        auto dev = static_cast<id<MTLDevice>>(m_device);
        OPENVINO_ASSERT(dev, "GFX Metal: device is null");
        id<MTLBuffer> buf = [dev newBufferWithBytesNoCopy:ptr
                                                   length:bytes
                                                  options:MTLResourceStorageModeShared
                                              deallocator:^(void*, NSUInteger) {
                                              }];
        OPENVINO_ASSERT(buf, "GFX Metal: failed to wrap shared memory");
        out.buffer = buf;
        out.size = bytes;
        out.type = type;
        out.backend = GpuBackend::Metal;
        out.storage_mode = static_cast<uint32_t>(MTLStorageModeShared);
        out.options_mask = static_cast<uint32_t>(MTLResourceStorageModeShared);
        out.external = true;
        out.from_handle = true;
        out.host_visible = true;
        out.owned = true;
        return out;
#else
        (void)ptr;
        (void)bytes;
        (void)type;
        OPENVINO_THROW("GFX remote tensor requires Objective-C++ (Metal)");
#endif
    }

    void release(GpuBuffer&& buf) override {
#ifdef __OBJC__
        if (!buf.owned || !buf.buffer) {
            return;
        }
        [static_cast<id<MTLBuffer>>(buf.buffer) release];
        buf.buffer = nullptr;
#else
        (void)buf;
#endif
    }

private:
    uint32_t resolve_storage_mode(const GpuBufferDesc& desc) const {
        if (m_explicit_mode) {
            return m_storage_mode;
        }
        const bool host_visible = desc.cpu_read || desc.cpu_write || !desc.prefer_device_local;
        return host_visible ? static_cast<uint32_t>(MTLStorageModeShared)
                            : static_cast<uint32_t>(MTLStorageModePrivate);
    }

    MetalDeviceHandle m_device = nullptr;
    uint32_t m_storage_mode = static_cast<uint32_t>(MTLStorageModeShared);
    bool m_explicit_mode = false;
};

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
        const StorageModeResult storage = parse_storage_mode(params);
        MetalRemoteAllocator allocator(device, storage.mode, storage.explicit_mode);
        GpuBufferDesc desc;
        desc.bytes = bytes;
        desc.type = type;
        desc.usage = BufferUsage::IO;
        desc.cpu_read = storage.host_visible;
        desc.cpu_write = storage.host_visible;
        desc.prefer_device_local = !storage.host_visible;
        tensor.buf = allocator.allocate(desc);
        tensor.buf.owned = true;
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
