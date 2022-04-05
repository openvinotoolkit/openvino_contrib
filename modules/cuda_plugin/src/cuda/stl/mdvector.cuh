// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "atomic.cuh"
#include "mdspan.cuh"
#include "vector.cuh"

namespace CUDA {

/**
 * Multi-dimensional Span of Vector-s, that store Vector objects
 * @tparam T Type of objects in Vector
 * @tparam Dims Dimensions of MDSpan
 */
template <typename T, typename Extent, std::size_t Dims>
class MDSpan<Vector<T>, Extent, Dims> {
public:
    static_assert(std::is_same<DExtents<Dims>, Extent>::value, "Vector specialization supports only DExtents<>");

    using iterator = T*;
    using const_iterator = const T*;

    /**
     * Constructor of MDVector
     * @tparam TExtents
     * @param capacity Capacity of underlying Vector
     * @param data Buffer to store all needed information, please see @ref size_of
     * @param extents Dimensions of MDVector
     */
    template <typename... TExtents>
    __host__ explicit MDSpan(const CUDA::Stream& stream, size_t capacity, void* data, TExtents... extents)
        : capacity_{capacity}, sizes_{data, extents...}, data_{data + sizes_.size_bytes(), extents..., capacity} {
        stream.memset(CUDA::DevicePointer<void*>{sizes_.data()}, 0, sizes_.size_bytes());
    }

    /**
     * @brief Returns number of data that referred by Span class
     */
    __device__ std::size_t size() const { return data_.size(); }

    /**
     * @brief Returns size in bytes
     */
    __device__ std::size_t extent(size_t i) const { return data_.extent(i); }

    /**
     * @brief Returns size in bytes
     */
    __device__ std::size_t size_bytes() const { return data_.size_bytes(); }

    /**
     * @brief Returns item by idxs
     */
    template <typename... TIndex>
    __device__ auto operator()(TIndex... idxs) {
        return Vector<T>{&data_(idxs..., 0), &sizes_(idxs...), capacity_};
    }

    /**
     * Returns size_of MDVector in bytes
     * @tparam TExtents Types of extents
     * @param capacity Capacity of MDVector
     * @param extents Extents of MDVector
     * @return Returns size in bytes of MDVector
     */
    template <typename... TExtents>
    static size_t size_of(size_t capacity, TExtents... extents) {
        return MDSpan<size_t, DExtents<Dims>>::size_of(extents...) +
               MDSpan<T, DExtents<Dims + 1>>::size_of(extents..., capacity);
    }

private:
    size_t capacity_ = 0;
    MDSpan<size_t, DExtents<Dims>> sizes_;
    MDSpan<T, DExtents<Dims + 1>> data_;
};

template <typename T, std::size_t Dims>
using MDVector = MDSpan<Vector<T>, DExtents<Dims>, Dims>;

}  // namespace CUDA
