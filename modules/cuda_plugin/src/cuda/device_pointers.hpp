// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header that defines wrappers for pointers to memory, allocated on GPU
 *
 * @file device_pointers.hpp
 */
#pragma once
#include <gsl/pointers>
#include <gsl/span>
#include <limits>
#include <type_traits>

namespace CUDA {
/**
 * @brief This template implements a wrapper to a raw device pointer
 */

template <typename T>
class DevicePointer : private gsl::not_null<T> {
public:
    static_assert(std::is_pointer_v<T>, "T should be a pointer type");
    explicit DevicePointer(gsl::not_null<T> o) noexcept : gsl::not_null<T>{o} {}
    using gsl::not_null<T>::get;

    /**
     * @brief cast<U>() - Casts original type T* to U*, provided one of them is void*
     */
    template <typename U>
    auto cast() const noexcept {
        static_assert(std::is_pointer_v<U>, "U should be a pointer type");
        static_assert(std::is_void_v<std::remove_pointer_t<T>> || std::is_void_v<std::remove_pointer_t<U>>,
                      "cast requires one of U or T to be void");
        return DevicePointer<U>{static_cast<U>(this->get())};
    }
    auto as_mutable() const noexcept {
        return DevicePointer<std::remove_const_t<std::remove_pointer_t<T>>*>{
            const_cast<std::remove_const_t<std::remove_pointer_t<T>>*>(this->get())};
    }
};

template <typename T, std::enable_if_t<std::is_void_v<T>>* = nullptr>
auto operator+(DevicePointer<T*> p, std::ptrdiff_t d) noexcept {
    return DevicePointer<T*>{static_cast<T*>(const_cast<char*>(static_cast<const char*>(p.get()) + d))};
}

template <typename T, std::enable_if_t<std::is_void_v<T>>* = nullptr>
auto operator-(DevicePointer<T*> l, DevicePointer<T*> r) noexcept {
    return static_cast<const char*>(l.get()) - static_cast<const char*>(r);
}

template <typename T, std::size_t Extent = std::numeric_limits<std::size_t>::max()>
class DeviceBuffer : private gsl::span<T, Extent> {
public:
    explicit DeviceBuffer(gsl::span<T, Extent> o) noexcept : gsl::span<T, Extent>{o} {}
    using gsl::span<T, Extent>::data;
    using gsl::span<T, Extent>::size;

    auto as_mutable() const noexcept {
        return DeviceBuffer<std::remove_const_t<T>>{const_cast<std::remove_const_t<T>*>(this->data()), this->size()};
    }
};

}  // namespace CUDA
