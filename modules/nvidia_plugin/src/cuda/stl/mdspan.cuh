// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

#include "span.cuh"

class MDSpanTest;

namespace CUDA {

/**
 * Dynamic Extent object to specify size of MDSpan
 */
template <std::size_t Dims>
struct DExtents {
    constexpr static std::size_t dims{Dims};
};

/**
 * Static Extent object to specify size of MDSpan
 */
template <std::size_t... Extent>
struct Extents {
    constexpr static std::size_t dims{sizeof...(Extent)};
    constexpr static std::size_t sizes[sizeof...(Extent)]{Extent...};
};

/**
 * @brief Stub implementation for get_template_pack
 */
template <size_t AgsIdx>
__host__ __device__ size_t get_template_pack(size_t idx) {
    return 0;
}

/**
 * @brief Gets template pack item by index
 */
template <size_t AgsIdx = 0, typename T, typename... TArgs>
__host__ __device__ T get_template_pack(size_t idx, T arg, TArgs... args) {
    if (AgsIdx == idx) {
        return arg;
    }
    return static_cast<T>(get_template_pack<AgsIdx + 1>(idx, args...));
}

/**
 * @brief Span class used for CUDA device
 * @tparam T Type of data
 * @tparam Extent Static size of data refered by Span class
 */
template <typename T, typename Extent, std::size_t Dims = Extent::dims>
class MDSpan {
public:
    /**
     * @brief Type used to iterate over this span (a raw pointer)
     */
    using iterator = T*;

    /**
     * @brief Constructor of Span class for dynamic Extent
     */
    template <typename TP,
              typename... TExtents,
              typename std::enable_if<std::is_same<DExtents<Dims>, Extent>::value, int*>::type = nullptr>
    __host__ __device__ MDSpan(TP* ptr, TExtents... extents)
        : ptr_{static_cast<T*>(ptr)}, sizes_{static_cast<size_t>(extents)...} {
        static_assert(sizeof...(TExtents) == Dims, "List of extents should be equal to dimensions");
        size_ = 1;
        for (size_t i = 0; i < Dims; ++i) {
            size_ *= sizes_[i];
        }
    }

    /**
     * @brief Constructor of Span class for static Extent
     */
    template <typename TP>
    __host__ __device__ MDSpan(TP* ptr) : ptr_{static_cast<T*>(ptr)}, sizes_{Extent::sizes} {
        size_ = 1;
        for (size_t i = 0; i < Dims; ++i) {
            size_ *= sizes_[i];
        }
    }

    /**
     * @brief Returns item by idxs
     */
    template <typename... TIndex>
    __device__ T& operator()(TIndex... idxs) {
        return *(ptr_ + get_offset<false>(idxs...));
    }

    /**
     * @brief Returns item by idxs
     */
    template <typename... TIndex>
    __device__ const T& operator()(TIndex... idxs) const {
        return *(ptr_ + get_offset<false>(idxs...));
    }

    /**
     * @brief Returns item by idxs if exists
     *        otherwise raise cuda error using __trap function
     */
    template <typename... TIndex>
    __device__ const T& at(TIndex... idxs) const {
        return *(ptr_ + get_offset<true>(idxs...));
    }

    /**
     * @brief Returns pointer to data
     */
    __host__ __device__ T* data() const { return ptr_; }

    /**
     * @brief Returns number of data that referred by Span class
     */
    __host__ __device__ std::size_t size() const { return size_; }

    /**
     * @brief Returns size of dimension @param i
     * @param i index of dimension
     * @return Size of dimension
     */
    __host__ __device__ std::size_t extent(size_t i) const { return sizes_[i]; }

    /**
     * @brief Returns size in bytes
     */
    __host__ __device__ std::size_t size_bytes() const { return sizeof(T) * size_; }

    /**
     * Returns size_of MDVector in bytes
     * @tparam TExtents Types of extents
     * @param capacity Capacity of MDVector
     * @param extents Extents of MDVector
     * @return Returns size in bytes of MDVector
     */
    template <typename... TExtents,
              typename std::enable_if<std::is_same<DExtents<Dims>, Extent>::value, int*>::type = nullptr>
    static size_t size_of(TExtents... extents) {
        static_assert(sizeof...(TExtents) == Dims, "List of extents should be equal to dimensions");
        std::size_t dims[Dims]{extents...};
        std::size_t size = sizeof(T);
        for (size_t i = 0; i < Dims; ++i) {
            size *= dims[i];
        }
        return size;
    }

    /**
     * @return Returns size in bytes of MDVector
     */
    static size_t size_of() {
        std::size_t size = sizeof(T);
        for (size_t i = 0; i < Dims; ++i) {
            size *= Extent::sizes[i];
        }
        return size;
    }

private:
    friend class ::MDSpanTest;

    /**
     * Returns offset to underlying data
     * @tparam BoundaryCheck Indicates if boundary check should be performed
     * @tparam TIndex Types of complex index
     * @param idxs Complex index to get offset to underlying data
     * @return Offset to data by @param idxs
     */
    template <bool BoundaryCheck, typename... TIndex>
    __host__ __device__ size_t get_offset(TIndex... idxs) const {
        static_assert(sizeof...(TIndex) == Dims, "Number of indexes should be equal to Dims");
        size_t offset = 0;
        size_t nextScale = 1;
        for (ptrdiff_t i = Dims - 1; i >= 0; --i) {
            const size_t idx = get_template_pack(i, idxs...);
            if (BoundaryCheck) {
                assert(idx < sizes_[i] && "idx is out of range");
            }
            offset += nextScale * idx;
            nextScale *= sizes_[i];
        }
        if (BoundaryCheck) {
            assert(offset < size_ && "offset is out of range");
        }
        return offset;
    }

    T* ptr_;
    std::size_t size_{};
    std::size_t sizes_[Dims]{};
};

}  // namespace CUDA
