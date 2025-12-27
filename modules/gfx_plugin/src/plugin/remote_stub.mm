// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remote_stub.hpp"

#import <Metal/Metal.h>

#include "openvino/core/except.hpp"
#include "openvino/util/common_util.hpp"
#include "plugin/gfx_remote_utils.hpp"
#include "plugin/gfx_remote_properties.hpp"
#include "backends/metal/plugin/properties.hpp"
#include "backends/metal/runtime/memory.hpp"
#include "plugin/gfx_backend_config.hpp"

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

GfxRemoteTensor::GfxRemoteTensor(const ov::element::Type& type,
                                     const ov::Shape& shape,
                                     const ov::AnyMap& params,
                                     const std::string& dev,
                                     const GpuTensor& tensor,
                                     bool owns_buffer)
    : m_type(type),
      m_shape(shape),
      m_params(params),
      m_device(dev),
      m_tensor(tensor),
      m_owns_buffer(owns_buffer) {
    recalc_strides();
}

GfxRemoteTensor::~GfxRemoteTensor() {
#ifdef __OBJC__
    if (!m_owns_buffer || !m_tensor.buf.buffer) {
        return;
    }
    if (m_tensor.buf.backend == GpuBackend::Metal) {
        [static_cast<id<MTLBuffer>>(m_tensor.buf.buffer) release];
        m_tensor.buf.buffer = nullptr;
    }
#endif
}

ov::SoPtr<ov::IRemoteTensor> GfxRemoteContext::create_tensor(const ov::element::Type& type,
                                                               const ov::Shape& shape,
                                                               const ov::AnyMap& params) {
    ov::AnyMap merged = m_params;
    merged.insert(params.begin(), params.end());
    merged[kGfxBackendProperty] = m_backend_name;

    if (m_backend != GpuBackend::Metal) {
        OPENVINO_THROW("GFX remote tensors are supported only with Metal backend");
    }

    const size_t bytes = std::max<size_t>(1, ov::shape_size(shape)) * type.size();
    GpuTensor tensor;
    tensor.shape = shape;
    tensor.expected_type = type;
    tensor.buf.type = type;
    tensor.buf.backend = GpuBackend::Metal;

    bool owns_buffer = false;
    void* external = find_any_ptr(merged, {kMtlBufferProperty, kGfxBufferProperty});

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
        OPENVINO_ASSERT(m_handle, "GFX: remote context device handle is null");
#ifdef __OBJC__
        auto dev = static_cast<id<MTLDevice>>(m_handle);
        const uint32_t storage_mode = parse_storage_mode(merged);
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

    auto t = std::make_shared<GfxRemoteTensor>(type, shape, merged, m_device, tensor, owns_buffer);
    return ov::SoPtr<ov::IRemoteTensor>{t, nullptr};
}

}  // namespace gfx_plugin
}  // namespace ov
