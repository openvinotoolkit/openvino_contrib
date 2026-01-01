// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace gfx_plugin {

inline int64_t normalize_axis(int64_t axis, size_t rank, const char* error_prefix) {
    const int64_t rank_i64 = static_cast<int64_t>(rank);
    int64_t axis_norm = axis < 0 ? axis + rank_i64 : axis;
    OPENVINO_ASSERT(axis_norm >= 0 && axis_norm < rank_i64,
                    error_prefix ? error_prefix : "GFX",
                    ": axis out of range");
    return axis_norm;
}

inline uint64_t shape_product(const ov::Shape& shape, size_t start, size_t end) {
    uint64_t prod = 1;
    for (size_t i = start; i < end; ++i) {
        prod *= static_cast<uint64_t>(shape[i]);
    }
    return prod;
}

inline uint64_t shape_product(const std::vector<int64_t>& shape, size_t start, size_t end) {
    uint64_t prod = 1;
    for (size_t i = start; i < end; ++i) {
        prod *= static_cast<uint64_t>(shape[i]);
    }
    return prod;
}

inline size_t tensor_byte_size(const ov::Shape& shape, const ov::element::Type& type) {
    if (type == ov::element::dynamic) {
        return 0;
    }
    return type.size() * ov::shape_size(shape);
}

struct SoftmaxDims {
    int64_t axis = 0;
    uint64_t outer = 1;
    uint64_t axis_len = 0;
    uint64_t inner = 1;
    uint64_t rows = 1;  // outer * inner
};

inline SoftmaxDims compute_softmax_dims(const ov::Shape& shape,
                                        int64_t axis,
                                        const char* error_prefix) {
    OPENVINO_ASSERT(!shape.empty(),
                    error_prefix ? error_prefix : "GFX",
                    ": input shape is empty");
    SoftmaxDims dims;
    dims.axis = normalize_axis(axis, shape.size(), error_prefix);
    dims.outer = shape_product(shape, 0, static_cast<size_t>(dims.axis));
    dims.axis_len = static_cast<uint64_t>(shape[static_cast<size_t>(dims.axis)]);
    dims.inner = shape_product(shape, static_cast<size_t>(dims.axis) + 1, shape.size());
    dims.rows = dims.outer * dims.inner;
    return dims;
}

}  // namespace gfx_plugin
}  // namespace ov
