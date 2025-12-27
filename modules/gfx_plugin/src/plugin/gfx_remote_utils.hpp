// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include "openvino/core/any.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace gfx_plugin {

inline void* any_to_ptr(const ov::Any& value) {
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

inline bool any_to_bool(const ov::Any& value, bool fallback) {
    if (value.empty())
        return fallback;
    if (value.is<bool>())
        return value.as<bool>();
    if (value.is<int>())
        return value.as<int>() != 0;
    if (value.is<unsigned int>())
        return value.as<unsigned int>() != 0u;
    if (value.is<std::string>()) {
        auto s = ov::util::to_lower(value.as<std::string>());
        if (s == "1" || s == "true" || s == "yes" || s == "on")
            return true;
        if (s == "0" || s == "false" || s == "no" || s == "off")
            return false;
    }
    return fallback;
}

inline void* find_any_ptr(const ov::AnyMap& params, std::initializer_list<const char*> keys) {
    for (const auto* key : keys) {
        auto it = params.find(key);
        if (it == params.end()) {
            continue;
        }
        if (auto ptr = any_to_ptr(it->second)) {
            return ptr;
        }
    }
    return nullptr;
}

inline bool find_any_bool(const ov::AnyMap& params,
                          std::initializer_list<const char*> keys,
                          bool fallback) {
    for (const auto* key : keys) {
        auto it = params.find(key);
        if (it == params.end()) {
            continue;
        }
        return any_to_bool(it->second, fallback);
    }
    return fallback;
}

}  // namespace gfx_plugin
}  // namespace ov
