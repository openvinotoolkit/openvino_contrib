// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gfx_remote_context.hpp"

#include <algorithm>

#include "openvino/gfx_plugin/properties.hpp"

namespace ov {
namespace gfx_plugin {

ov::SoPtr<ov::IRemoteTensor> GfxRemoteContext::create_tensor(const ov::element::Type& type,
                                                             const ov::Shape& shape,
                                                             const ov::AnyMap& params) {
    ov::AnyMap merged = m_params;
    merged.insert(params.begin(), params.end());
    merged[kGfxBackendProperty] = m_backend_name;

    const size_t bytes = std::max<size_t>(1, ov::shape_size(shape)) * type.size();
    auto created = create_remote_tensor(type, shape, merged, bytes);
    for (const auto& kv : created.properties) {
        merged[kv.first] = kv.second;
    }

    auto t = std::make_shared<GfxRemoteTensor>(type,
                                               shape,
                                               merged,
                                               m_device,
                                               created.tensor,
                                               created.release_fn);
    return ov::SoPtr<ov::IRemoteTensor>{t, nullptr};
}

}  // namespace gfx_plugin
}  // namespace ov
