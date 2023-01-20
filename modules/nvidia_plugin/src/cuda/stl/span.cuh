// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace CUDA {

/**
 * @brief Means number of elements determined at runtime
 */
constexpr size_t dynamic_extent = std::numeric_limits<std::size_t>::max();

/**
 * @brief Span class used for CUDA device
 * @tparam T Type of data
 * @tparam Extent Static size of data refered by Span class
 */
template <typename T, std::size_t Extent = dynamic_extent>
class Span {
public:
    /**
     * @brief Type used to iterate over this span (a raw pointer)
     */
    using iterator = T *;

    /**
     * @brief Constructor of Span class
     */
    __host__ __device__ Span() : ptr_{nullptr}, size_{0} {}

    /**
     * @brief Constructor of Span class
     */
    template <typename TP, typename... TArgs>
    __host__ __device__ Span(TP *t, std::size_t n) : ptr_{static_cast<T *>(t)}, size_{n} {}

    /**
     * @brief Constructor of Span class for containers
     */
    template <template <typename> class Container, typename TT = typename std::remove_const<T>::type>
    __host__ __device__ Span(Container<TT> &range) : ptr_{range.data()}, size_{range.size()} {}

    /**
     * @brief Constructor of Span class for containers
     */
    template <template <typename> class Container, typename TT = typename std::remove_const<T>::type>
    __host__ __device__ Span(const Container<TT> &range) : ptr_{range.data()}, size_{range.size()} {}

    /**
     * @brief Returns item by index
     */
    __device__ T &operator[](ptrdiff_t idx) { return ptr_[idx]; }

    /**
     * @brief Returns item by index
     */
    __device__ const T &operator[](ptrdiff_t idx) const { return ptr_[idx]; }

    /**
     * @brief Returns pointer to data
     */
    __device__ T *data() const { return ptr_; }

    /**
     * @brief Returns number of data that referred by Span class
     */
    __device__ std::size_t size() const { return size_; }

    /**
     * @brief Returns begin iterator
     */
    __device__ iterator begin() const { return ptr_; }

    /**
     * @brief Returns end iterator
     */
    __device__ iterator end() const { return ptr_ + size_; }

    /**
     * @brief Returns size in bytes
     */
    __device__ std::size_t size_bytes() const { return sizeof(T) * size_; }

    /**
     * Returns size_of Vector in bytes
     * @param size Size of Span
     * @return Returns size in bytes of Span
     */
    static size_t size_of(std::size_t size) { return sizeof(T) * size; }

private:
    T *ptr_;
    std::size_t size_;
};

}  // namespace CUDA
