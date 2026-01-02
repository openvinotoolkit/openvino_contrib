// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gfx_remote_context.hpp"

#include <algorithm>

#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/gfx_plugin/properties.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
bool parse_device_id_any(const ov::Any& value, int& out) {
    try {
        if (value.is<std::string>()) {
            const auto str = value.as<std::string>();
            size_t idx = 0;
            out = std::stoi(str, &idx);
            return idx == str.size();
        }
        if (value.is<int>()) {
            out = value.as<int>();
            return true;
        }
        if (value.is<int64_t>()) {
            out = static_cast<int>(value.as<int64_t>());
            return true;
        }
        if (value.is<size_t>()) {
            out = static_cast<int>(value.as<size_t>());
            return true;
        }
    } catch (...) {
        return false;
    }
    return false;
}
}  // namespace

ov::SoPtr<ov::IRemoteTensor> GfxRemoteContext::create_tensor(const ov::element::Type& type,
                                                             const ov::Shape& shape,
                                                             const ov::AnyMap& params) {
    ov::AnyMap merged = m_params;
    merged.insert(params.begin(), params.end());

    if (auto it = merged.find(kGfxBackendProperty); it != merged.end()) {
        if (!it->second.is<std::string>()) {
            OPENVINO_THROW("GFX: remote tensor backend must be a string");
        }
        const auto requested = ov::util::to_lower(it->second.as<std::string>());
        const auto current = ov::util::to_lower(m_backend_name);
        if (requested != current) {
            OPENVINO_THROW("GFX: remote tensor backend '",
                           requested,
                           "' does not match context backend '",
                           current,
                           "'");
        }
    }
    if (auto it = merged.find(ov::device::id.name()); it != merged.end()) {
        int requested_id = 0;
        if (!parse_device_id_any(it->second, requested_id)) {
            OPENVINO_THROW("GFX: remote tensor device id has invalid format");
        }
        if (requested_id != m_device_id) {
            OPENVINO_THROW("GFX: remote tensor device id ",
                           requested_id,
                           " does not match context device id ",
                           m_device_id);
        }
    }

    merged[kGfxBackendProperty] = m_backend_name;
    merged[ov::device::id.name()] = m_device_id;

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
