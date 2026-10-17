// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cctype>
#include <limits>
#include <optional>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace test {
namespace utils {

struct GfxAccuracyTolerance {
    double abs_threshold = 0.0;
    double rel_threshold = 0.0;
};

inline ov::element::Type gfx_parse_inference_precision_name(const std::string& value) {
    std::string lowered;
    lowered.reserve(value.size());
    for (const char ch : value) {
        lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    if (lowered == "f16" || lowered == "fp16" || lowered == "half") {
        return ov::element::f16;
    }
    if (lowered == "f32" || lowered == "fp32" || lowered == "float") {
        return ov::element::f32;
    }
    return ov::element::dynamic;
}

inline ov::element::Type gfx_inference_precision_from_config(const ov::AnyMap& config) {
    auto it = config.find(ov::hint::inference_precision.name());
    if (it == config.end()) {
        return ov::element::f16;
    }
    try {
        return it->second.as<ov::element::Type>();
    } catch (...) {
    }
    try {
        return gfx_parse_inference_precision_name(it->second.as<std::string>());
    } catch (...) {
    }
    return ov::element::f16;
}

inline double gfx_element_type_epsilon(const ov::element::Type& type) {
    if (type == ov::element::f16) {
        return static_cast<double>(std::numeric_limits<ov::float16>::epsilon());
    }
    if (type == ov::element::bf16) {
        return static_cast<double>(std::numeric_limits<ov::bfloat16>::epsilon());
    }
    if (type == ov::element::f32) {
        return static_cast<double>(std::numeric_limits<float>::epsilon());
    }
    if (type == ov::element::f64) {
        return static_cast<double>(std::numeric_limits<double>::epsilon());
    }
    return 0.0;
}

inline GfxAccuracyTolerance gfx_accuracy_tolerance(
    const ov::element::Type& expected_type,
    const ov::element::Type& actual_type,
    const ov::element::Type& inference_precision,
    double abs_floor,
    double rel_floor,
    std::optional<double> abs_override = std::nullopt,
    std::optional<double> rel_override = std::nullopt) {
    const double type_floor = std::max({gfx_element_type_epsilon(expected_type),
                                        gfx_element_type_epsilon(actual_type),
                                        gfx_element_type_epsilon(inference_precision)});
    GfxAccuracyTolerance tolerance;
    tolerance.abs_threshold = abs_override.value_or(std::max(abs_floor, type_floor));
    tolerance.rel_threshold = rel_override.value_or(std::max(rel_floor, type_floor));
    return tolerance;
}

}  // namespace utils
}  // namespace test
}  // namespace ov
