// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remote_stub.hpp"

#import <Metal/Metal.h>

#include "openvino/core/except.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

void* any_to_ptr(const ov::Any& value) {
    if (value.empty())
        return nullptr;
    if (value.is<void*>())
        return value.as<void*>();
    if (value.is<intptr_t>())
        return reinterpret_cast<void*>(value.as<intptr_t>());
    if (value.is<uintptr_t>())
        return reinterpret_cast<void*>(value.as<uintptr_t>());
    if (value.is<size_t>())
        return reinterpret_cast<void*>(value.as<size_t>());
    if (value.is<uint64_t>())
        return reinterpret_cast<void*>(value.as<uint64_t>());
    if (value.is<int64_t>())
        return reinterpret_cast<void*>(value.as<int64_t>());
    return nullptr;
}

uint32_t parse_storage_mode(const ov::AnyMap& params) {
    auto it = params.find("GFX_STORAGE_MODE");
    if (it == params.end())
        it = params.find("STORAGE_MODE");
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
                                     const MetalTensor& tensor,
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
    if (m_owns_buffer && m_tensor.buf.buffer) {
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

    const size_t bytes = std::max<size_t>(1, ov::shape_size(shape)) * type.size();
    MetalTensor tensor;
    tensor.shape = shape;
    tensor.expected_type = type;
    tensor.buf.type = type;

    bool owns_buffer = false;
    void* external = nullptr;
    if (auto it = merged.find("MTL_BUFFER"); it != merged.end())
        external = any_to_ptr(it->second);
    if (!external) {
        if (auto it = merged.find("GFX_BUFFER"); it != merged.end())
            external = any_to_ptr(it->second);
    }

    if (external) {
        tensor.buf.buffer = external;
        tensor.buf.size = bytes;
        tensor.buf.external = false;
#ifdef __OBJC__
        auto mb = static_cast<id<MTLBuffer>>(external);
        tensor.buf.storage_mode = static_cast<uint32_t>(mb.storageMode);
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
