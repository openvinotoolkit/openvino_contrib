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

inline std::vector<int64_t> make_element_strides(const ov::Shape& shape) {
    const size_t rank = shape.size();
    std::vector<int64_t> strides(rank, 1);
    for (int64_t i = static_cast<int64_t>(rank) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i) + 1] *
                                          static_cast<int64_t>(shape[static_cast<size_t>(i) + 1]);
    }
    return strides;
}

inline std::vector<int32_t> compute_broadcast_element_strides(const ov::Shape& in_shape,
                                                              const ov::Shape& out_shape) {
    const size_t out_rank = out_shape.size();
    const size_t in_rank = in_shape.size();
    std::vector<int64_t> aligned(out_rank, 1);
    if (in_rank <= out_rank) {
        const size_t off = out_rank - in_rank;
        for (size_t i = 0; i < in_rank; ++i) {
            aligned[off + i] = static_cast<int64_t>(in_shape[i]);
        }
    }
    std::vector<int32_t> strides(out_rank, 0);
    int64_t stride = 1;
    for (int64_t i = static_cast<int64_t>(out_rank) - 1; i >= 0; --i) {
        const int64_t dim = aligned[static_cast<size_t>(i)];
        strides[static_cast<size_t>(i)] = dim == 1 ? 0 : static_cast<int32_t>(stride);
        stride *= dim;
    }
    return strides;
}

inline std::vector<int32_t> compute_permuted_broadcast_element_strides(
    const ov::Shape& source_shape,
    const ov::Shape& consumer_shape,
    const std::vector<int64_t>& permutation,
    const ov::Shape& out_shape,
    const char* error_prefix) {
    OPENVINO_ASSERT(source_shape.size() == consumer_shape.size(),
                    error_prefix ? error_prefix : "GFX",
                    ": transposed broadcast rank mismatch");
    OPENVINO_ASSERT(source_shape.size() == permutation.size(),
                    error_prefix ? error_prefix : "GFX",
                    ": transpose permutation rank mismatch");
    const size_t rank = source_shape.size();
    std::vector<int64_t> inverse(rank, -1);
    for (size_t axis = 0; axis < rank; ++axis) {
        const int64_t perm_axis = permutation[axis];
        OPENVINO_ASSERT(perm_axis >= 0 && perm_axis < static_cast<int64_t>(rank),
                        error_prefix ? error_prefix : "GFX",
                        ": transpose permutation axis is out of range");
        OPENVINO_ASSERT(inverse[static_cast<size_t>(perm_axis)] < 0,
                        error_prefix ? error_prefix : "GFX",
                        ": transpose permutation must be unique");
        inverse[static_cast<size_t>(perm_axis)] = static_cast<int64_t>(axis);
    }

    const auto source_strides = make_element_strides(source_shape);
    const size_t out_rank = out_shape.size();
    std::vector<int32_t> strides(out_rank, 0);
    OPENVINO_ASSERT(rank <= out_rank,
                    error_prefix ? error_prefix : "GFX",
                    ": transposed broadcast rank exceeds output rank");
    const size_t offset = out_rank - rank;
    for (size_t source_axis = 0; source_axis < rank; ++source_axis) {
        const size_t consumer_axis = static_cast<size_t>(inverse[source_axis]);
        const size_t out_axis = offset + consumer_axis;
        const auto consumer_dim = consumer_shape[consumer_axis];
        strides[out_axis] = consumer_dim == 1 ? 0 : static_cast<int32_t>(source_strides[source_axis]);
    }
    return strides;
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
