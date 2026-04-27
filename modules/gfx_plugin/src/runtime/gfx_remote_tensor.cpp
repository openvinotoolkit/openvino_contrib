// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gfx_remote_tensor.hpp"

namespace ov {
namespace gfx_plugin {

GfxRemoteTensor::GfxRemoteTensor(const ov::element::Type& type,
                                 const ov::Shape& shape,
                                 const ov::AnyMap& params,
                                 const std::string& dev,
                                 const GpuTensor& tensor,
                                 RemoteTensorReleaseFn release_fn)
    : m_type(type),
      m_shape(shape),
      m_params(params),
      m_device(dev),
      m_tensor(tensor),
      m_release_fn(release_fn) {
    recalc_strides();
}

GfxRemoteTensor::~GfxRemoteTensor() {
    if (!m_tensor.buf.owned || !m_tensor.buf.buffer || !m_release_fn) {
        return;
    }
    m_release_fn(m_tensor);
}

}  // namespace gfx_plugin
}  // namespace ov
