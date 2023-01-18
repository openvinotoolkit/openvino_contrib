// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "atomic.cuh"

namespace CUDA {

/**
 * Vector implementation for CUDA
 * @tparam T Underlying data type to store in Vector
 */
template <typename T>
class Vector {
public:
    using iterator = T*;
    using const_iterator = const T*;

    /**
     * Constructor for creation of empty Vector
     */
    __device__ explicit Vector() : Vector(0) {}

    /**
     * Constructor for creation of empty Vector
     * @param data Buffer to store all needed information, please see @ref size_of
     * @param capacity Capacity of underlying Vector
     */
    __host__ __device__ explicit Vector(void* data, size_t capacity)
        : capacity_{capacity},
          size_{*static_cast<size_t*>(data)},
          data_{static_cast<T*>(static_cast<void*>(static_cast<char*>(data) + sizeof(size_t)))} {}

    /**
     * @brief Returns reference to first item in container
     */
    __device__ T& front() { return data_[0]; }

    /**
     * @brief Returns const reference to first item in container
     */
    __device__ const T& front() const { return data_[0]; }

    /**
     * @brief Returns iterator to begin
     */
    __device__ iterator begin() { return data_; }

    /**
     * @brief Returns const iterator to begin
     */
    __device__ const_iterator begin() const { return cbegin(); }

    /**
     * @brief Returns const iterator to begin
     */
    __device__ const_iterator cbegin() const { return data_; }

    /**
     * @brief Returns iterator to end
     */
    __device__ iterator end() { return data_ + size_; }

    /**
     * @brief Returns const iterator to end
     */
    __device__ const_iterator end() const { return cend(); }

    /**
     * @brief Returns const iterator to end
     */
    __device__ const_iterator cend() const { return data_ + size_; }

    /**
     * @brief Returns pointer to underlying data
     */
    __device__ T* data() { return data_; }

    /**
     * @brief Returns pointer to underlying data
     */
    __device__ const T* data() const { return data_; }

    /**
     * Adds new element in Vector
     * @param t New item to add
     */
    __device__ void push_back(T&& t) {
#ifndef NDEBUG
        if (size_ >= capacity_) {
            __trap();
        }
#endif
        data_[size_++] = std::move(t);
    }

    /**
     * Adds new element in Vector
     * @param t New item to add
     */
    __device__ void push_back(const T& t) {
#ifndef NDEBUG
        if (size_ >= capacity_) {
            __trap();
        }
#endif
        data_[size_++] = t;
    }

    /**
     * Resize Vector with new size @param newSize
     * @param newSize new size of Vector
     */
    __device__ void resize(size_t newSize) {
        if (newSize > size_) {
            size_t sizeDiff = newSize - size_;
            while ((sizeDiff--) > 0) {
                push_back(T{});
            }
        }
        size_ = newSize;
    }

    /**
     * @brief Clears Vector
     */
    __device__ void clear() { resize(0); }

    /**
     * Returns reference to item by index @param i
     * @param i Index of item to return
     * @return Reference to item of type T
     */
    __device__ T& operator[](size_t i) { return data_[i]; }

    /**
     * Returns const reference to item by index @param i
     * @param i Index of item to return
     * @return Reference to item of type T
     */
    __device__ const T& operator[](size_t i) const { return data_[i]; }

    /**
     * Returns element by index @param i
     * @param i Index of element to return
     * @return
     */
    __device__ const T& at(size_t i) const {
#ifndef NDEBUG
        if (i >= size_) {
            __trap();
        }
#endif
        return data_[i];
    }

    /**
     * @brief Returns size of Vector
     */
    __device__ size_t size() const { return size_; }

    /**
     * @brief Returns capacity of Vector
     */
    __device__ size_t capacity() const { return capacity_; }

    /**
     * Erase element by @param iter
     * @param iter Iterator to remove
     * @return Next iterator after erased element
     */
    __device__ auto erase(const_iterator iter) {
        const ptrdiff_t i = iter - begin();
        if (i >= 0 && (i + 1) < size_) {
            for (size_t k = i; (k + 1) < size_; ++k) {
                data_[k] = data_[k + 1];
            }
            size_ -= 1;
            return &data_[i + 1];
        } else {
            return end();
        }
    }

    /**
     * Returns size_of Vector in bytes
     * @param capacity Capacity of Vector
     * @return Returns size in bytes of Vector
     */
    static size_t size_of(size_t capacity) { return sizeof(size_t) + sizeof(T) * capacity; }

private:
    template <typename, typename, std::size_t>
    friend class MDSpan;

    /**
     * Private constructor that used in MDVector specialization of MDSpan container
     * @param data Buffer to store all needed information, please see @ref size_of
     * @param capacity Capacity of underlying Vector
     */
    __device__ explicit Vector(T* data, size_t* size, size_t capacity)
        : capacity_{capacity}, size_{*static_cast<size_t*>(size)}, data_{static_cast<T*>(data)} {}

    size_t capacity_{};
    size_t& size_;
    T* data_;
};

}  // namespace CUDA
