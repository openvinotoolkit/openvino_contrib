// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gfx_remote_context.hpp"

#include <algorithm>

#include "openvino/core/except.hpp"
#include "openvino/gfx_plugin/properties.hpp"

namespace ov {
namespace gfx_plugin {

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
    if (!m_owns_buffer || !m_tensor.buf.buffer) {
        return;
    }
    switch (m_tensor.buf.backend) {
        case GpuBackend::Metal:
            release_metal_remote_tensor(m_tensor, m_owns_buffer);
            break;
        case GpuBackend::Vulkan:
            release_vulkan_remote_tensor(m_tensor, m_owns_buffer);
            break;
        default:
            break;
    }
}

ov::SoPtr<ov::IRemoteTensor> GfxRemoteContext::create_tensor(const ov::element::Type& type,
                                                             const ov::Shape& shape,
                                                             const ov::AnyMap& params) {
    ov::AnyMap merged = m_params;
    merged.insert(params.begin(), params.end());
    merged[kGfxBackendProperty] = m_backend_name;

    const size_t bytes = std::max<size_t>(1, ov::shape_size(shape)) * type.size();
    RemoteTensorCreateResult created;

    switch (m_backend) {
        case GpuBackend::Metal: {
            created = create_metal_remote_tensor(type, shape, merged, m_handle, bytes);
            break;
        }
        case GpuBackend::Vulkan: {
            created = create_vulkan_remote_tensor(type, shape, merged, m_handle, bytes);
            break;
        }
        default:
            OPENVINO_THROW("GFX: unsupported remote backend");
    }

    auto t = std::make_shared<GfxRemoteTensor>(type,
                                               shape,
                                               merged,
                                               m_device,
                                               created.tensor,
                                               created.owns_buffer);
    return ov::SoPtr<ov::IRemoteTensor>{t, nullptr};
}

}  // namespace gfx_plugin
}  // namespace ov
