// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace gfx_plugin {

struct BiasParams {
    std::vector<float> values;
    std::vector<int64_t> shape;
    ov::element::Type element_type = ov::element::dynamic;

    bool empty() const {
        return values.empty() || element_type == ov::element::dynamic;
    }

    size_t element_count() const {
        return values.size();
    }
};

}  // namespace gfx_plugin
}  // namespace ov
