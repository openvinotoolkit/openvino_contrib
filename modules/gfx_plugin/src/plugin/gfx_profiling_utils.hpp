// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <cctype>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/gfx_plugin/profiling.hpp"

namespace ov {
namespace gfx_plugin {

inline ProfilingLevel parse_profiling_level(const ov::Any& value) {
    if (value.is<int>()) {
        const int v = value.as<int>();
        if (v <= 0)
            return ProfilingLevel::Off;
        if (v == 1)
            return ProfilingLevel::Standard;
        return ProfilingLevel::Detailed;
    }
    if (value.is<unsigned int>()) {
        const unsigned int v = value.as<unsigned int>();
        if (v == 0)
            return ProfilingLevel::Off;
        if (v == 1)
            return ProfilingLevel::Standard;
        return ProfilingLevel::Detailed;
    }
    if (value.is<bool>()) {
        return value.as<bool>() ? ProfilingLevel::Standard : ProfilingLevel::Off;
    }
    if (value.is<std::string>()) {
        auto s = value.as<std::string>();
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (s == "0" || s == "off" || s == "false") {
            return ProfilingLevel::Off;
        }
        if (s == "1" || s == "standard" || s == "on" || s == "true") {
            return ProfilingLevel::Standard;
        }
        if (s == "2" || s == "detailed" || s == "detail") {
            return ProfilingLevel::Detailed;
        }
    }
    OPENVINO_THROW("Unsupported profiling level type/value");
}

}  // namespace gfx_plugin
}  // namespace ov
