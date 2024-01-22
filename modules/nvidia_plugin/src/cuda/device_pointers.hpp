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
    static_assert(std::is_pointer<T>::value, "T should be a pointer type");
    explicit DevicePointer(gsl::not_null<T> o) noexcept : gsl::not_null<T>{o} {}
    using gsl::not_null<T>::get;

    /**
     * @brief cast<U>() - Casts original type T* to U*, provided one of them is void*
     */
    template <typename U>
    auto cast() const noexcept {
        static_assert(std::is_pointer<U>::value, "U should be a pointer type");
        static_assert(std::is_void<typename std::remove_pointer<T>::type>::value ||
                          std::is_void<typename std::remove_pointer<U>::type>::value,
                      "cast requires one of U or T to be void");
        return DevicePointer<U>{static_cast<U>(this->get())};
    }
    auto as_mutable() const noexcept {
        return DevicePointer<typename std::remove_const<typename std::remove_pointer<T>::type>::type*>{
            const_cast<typename std::remove_const<typename std::remove_pointer<T>::type>::type*>(this->get())};
    }
};

template <typename T, typename std::enable_if<std::is_void<T>::value>::type* = nullptr>
auto operator+(DevicePointer<T*> p, std::ptrdiff_t d) noexcept {
    return DevicePointer<T*>{static_cast<T*>(const_cast<char*>(static_cast<const char*>(p.get()) + d))};
}

template <typename T, typename std::enable_if<std::is_void<T>::value>::type* = nullptr>
auto operator-(DevicePointer<T*> l, DevicePointer<T*> r) noexcept {
    return static_cast<const char*>(l.get()) - static_cast<const char*>(r);
}

template <typename T, typename U>
bool operator==(const DevicePointer<T*>& lhs, const DevicePointer<U*>& rhs) {
    return lhs.get() == rhs.get();
}

template <typename T, typename U>
bool operator!=(const DevicePointer<T*>& lhs, const DevicePointer<U*>& rhs) {
    return lhs.get() != rhs.get();
}

template <typename T, std::size_t Extent = gsl::dynamic_extent>
class DeviceBuffer : private gsl::span<T, Extent> {
public:
    static_assert(!std::is_void<T>::value, "T should not be a void type");
    DeviceBuffer() = default;
    explicit DeviceBuffer(gsl::span<T, Extent> o) noexcept : gsl::span<T, Extent>{o} {}
    DeviceBuffer(T* data, std::size_t size) noexcept : gsl::span<T, Extent>{data, size} {}
    using gsl::span<T, Extent>::data;
    using gsl::span<T, Extent>::size;
    using gsl::span<T, Extent>::size_bytes;
    using gsl::span<T, Extent>::empty;

    DeviceBuffer<typename std::remove_const<T>::type> as_mutable() const noexcept {
        using MT = typename std::remove_const<T>::type;
        return {const_cast<MT*>(data()), size()};
    }
};

}  // namespace CUDA
