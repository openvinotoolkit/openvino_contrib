// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header that defines wrappers for pointers to memory, allocated on GPU
 *
 * @file device_pointers.hpp
 */
#pragma once
#include <type_traits>
#include <limits>
#include <gsl/pointers>
#include <gsl/span>

namespace InferenceEngine {
namespace gpu {
/**
 * @brief This template implements a wrapper to a raw device pointer
 */

template <typename T>
class DevicePointer : private gsl::not_null<T> {
 public:
  static_assert(std::is_pointer<T>::value, "T should be a pointer type");
  using gsl::not_null<T>::not_null;
  using gsl::not_null<T>::get;

  /**
   * @brief cast<U>() - Casts original type T* to U*, provided one of them is void*
   */
  template<typename U>
  DevicePointer<U> cast() const noexcept {
    static_assert(std::is_pointer<U>::value, "U should be a pointer type");
    static_assert(std::is_void<typename std::remove_pointer<T>::type>::value ||
                  std::is_void<typename std::remove_pointer<U>::type>::value,
                  "cast requires one of U or T to be void");
    return DevicePointer<U>{static_cast<U>(this->get())};
  }
  DevicePointer<std::remove_const_t<std::remove_pointer_t<T>>*> as_mutable() const noexcept {
    return const_cast<std::remove_const_t<std::remove_pointer_t<T>>*>(this->get());
  }
};

template <typename T, std::size_t Extent = std::numeric_limits<std::size_t>::max()>
class DeviceBuffer : private gsl::span<T, Extent> {
 public:
  using gsl::span<T, Extent>::span;
  using gsl::span<T, Extent>::data;
  using gsl::span<T, Extent>::size;

  /**
   * @brief cast<U>() - Casts original type T* to U*, provided one of them is void*
   */
  template<typename U>
  DeviceBuffer<U, Extent> cast() const noexcept {
    static_assert(std::is_pointer<U>::value, "U should be a pointer type");
    static_assert(std::is_void<typename std::remove_pointer<T>::type>::value ||
                      std::is_void<typename std::remove_pointer<U>::type>::value,
                  "cast requires one of U or T to be void");
    return DeviceBuffer<U, Extent>{static_cast<U>(this->get()), this->size()};
  }
  DeviceBuffer<std::remove_const_t<T>> as_mutable() const noexcept {
    return {const_cast<std::remove_const_t<T>*>(this->data()), this->size()};
  }
};

} // namespace gpu

} // namespace InferenceEngine
