// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
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

inline std::array<int64_t, 3> flatten_matmul_to_3d(const ov::Shape& shape,
                                                   const char* error_prefix) {
    OPENVINO_ASSERT(shape.size() >= 2 && shape.size() <= 4,
                    error_prefix ? error_prefix : "GFX",
                    ": MatMul supports ranks 2-4");
    int64_t batch = 1;
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
        batch *= static_cast<int64_t>(shape[i]);
    }
    return {batch,
            static_cast<int64_t>(shape[shape.size() - 2]),
            static_cast<int64_t>(shape[shape.size() - 1])};
}

struct FlattenedMatMulShapes {
    std::array<int64_t, 3> lhs{{1, 1, 1}};
    std::array<int64_t, 3> rhs{{1, 1, 1}};
    ov::Shape batch_prefix{};
    int64_t batch = 1;
};

inline ov::Shape broadcast_batch_prefix(const ov::Shape& lhs,
                                        const ov::Shape& rhs,
                                        const char* error_prefix) {
    OPENVINO_ASSERT(lhs.size() >= 2 && rhs.size() >= 2,
                    error_prefix ? error_prefix : "GFX",
                    ": MatMul rank must be at least 2");
    const size_t lhs_batch_rank = lhs.size() - 2;
    const size_t rhs_batch_rank = rhs.size() - 2;
    const size_t out_batch_rank = std::max(lhs_batch_rank, rhs_batch_rank);
    ov::Shape out(out_batch_rank, 1);
    for (size_t i = 0; i < out_batch_rank; ++i) {
        const size_t lhs_dim = lhs_batch_rank > i ? lhs[lhs_batch_rank - 1 - i] : 1;
        const size_t rhs_dim = rhs_batch_rank > i ? rhs[rhs_batch_rank - 1 - i] : 1;
        OPENVINO_ASSERT(lhs_dim == rhs_dim || lhs_dim == 1 || rhs_dim == 1,
                        error_prefix ? error_prefix : "GFX",
                        ": incompatible MatMul batch broadcast dimensions");
        out[out_batch_rank - 1 - i] = std::max(lhs_dim, rhs_dim);
    }
    return out;
}

inline FlattenedMatMulShapes flatten_matmul_shapes_with_batch_broadcast(const ov::Shape& lhs,
                                                                        const ov::Shape& rhs,
                                                                        const char* error_prefix) {
    FlattenedMatMulShapes info;
    info.lhs = flatten_matmul_to_3d(lhs, error_prefix);
    info.rhs = flatten_matmul_to_3d(rhs, error_prefix);
    info.batch_prefix = broadcast_batch_prefix(lhs, rhs, error_prefix);
    info.batch = static_cast<int64_t>(shape_product(info.batch_prefix, 0, info.batch_prefix.size()));
    OPENVINO_ASSERT(info.lhs[0] == 1 || info.lhs[0] == info.batch,
                    error_prefix ? error_prefix : "GFX",
                    ": flattened MatMul route does not support mixed lhs batch-prefix broadcast");
    OPENVINO_ASSERT(info.rhs[0] == 1 || info.rhs[0] == info.batch,
                    error_prefix ? error_prefix : "GFX",
                    ": flattened MatMul route does not support mixed rhs batch-prefix broadcast");
    return info;
}

inline bool is_nchw_channel_bias_broadcast(const ov::Shape& bias_shape, const ov::Shape& out_shape) {
    return out_shape.size() == 4 && bias_shape.size() == 4 && bias_shape[0] == 1 &&
           bias_shape[1] == out_shape[1] && bias_shape[2] == 1 && bias_shape[3] == 1;
}

inline bool is_bias_broadcast_add(const std::shared_ptr<const ov::Node>& node) {
    if (!node || node->get_input_size() != 2 || node->get_output_size() != 1) {
        return false;
    }
    if (!node->get_input_partial_shape(0).is_static() || !node->get_input_partial_shape(1).is_static() ||
        !node->get_output_partial_shape(0).is_static()) {
        return false;
    }
    const ov::Shape out_shape = node->get_output_shape(0);
    const ov::Shape in0_shape = node->get_input_shape(0);
    const ov::Shape in1_shape = node->get_input_shape(1);
    return (in0_shape == out_shape && is_nchw_channel_bias_broadcast(in1_shape, out_shape)) ||
           (in1_shape == out_shape && is_nchw_channel_bias_broadcast(in0_shape, out_shape));
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

struct GatherLinearDims {
    int64_t axis = 0;
    uint64_t outer = 1;
    uint64_t axis_dim = 0;
    uint64_t inner = 1;
    uint64_t indices_count = 0;
};

inline GatherLinearDims compute_gather_linear_dims(const ov::Shape& data_shape,
                                                   const ov::Shape& indices_shape,
                                                   int64_t axis,
                                                   const char* error_prefix) {
    OPENVINO_ASSERT(!data_shape.empty(),
                    error_prefix ? error_prefix : "GFX",
                    ": Gather data shape is empty");
    GatherLinearDims dims;
    dims.axis = normalize_axis(axis, data_shape.size(), error_prefix);
    dims.outer = shape_product(data_shape, 0, static_cast<size_t>(dims.axis));
    dims.axis_dim = static_cast<uint64_t>(data_shape[static_cast<size_t>(dims.axis)]);
    dims.inner = shape_product(data_shape, static_cast<size_t>(dims.axis) + 1, data_shape.size());
    dims.indices_count = ov::shape_size(indices_shape);
    return dims;
}

inline ov::Shape compute_gather_output_shape(const ov::Shape& data_shape,
                                             const ov::Shape& indices_shape,
                                             int64_t axis,
                                             const char* error_prefix) {
    const int64_t axis_norm = normalize_axis(axis, data_shape.size(), error_prefix);
    ov::Shape out_shape;
    out_shape.reserve(data_shape.size() + indices_shape.size() - 1);
    out_shape.insert(out_shape.end(), data_shape.begin(), data_shape.begin() + axis_norm);
    out_shape.insert(out_shape.end(), indices_shape.begin(), indices_shape.end());
    out_shape.insert(out_shape.end(), data_shape.begin() + axis_norm + 1, data_shape.end());
    return out_shape;
}

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
