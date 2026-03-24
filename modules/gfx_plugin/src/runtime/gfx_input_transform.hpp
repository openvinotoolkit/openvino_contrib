// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <vector>

#include "openvino/core/shape.hpp"

namespace ov {
namespace gfx_plugin {

struct GfxInputTransform {
    ov::Shape source_shape;
    std::vector<int64_t> transpose_permutation;

    bool has_transpose() const {
        return !source_shape.empty() && !transpose_permutation.empty();
    }
};

}  // namespace gfx_plugin
}  // namespace ov
