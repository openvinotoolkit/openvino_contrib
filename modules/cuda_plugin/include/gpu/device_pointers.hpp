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
#include <gsl/pointers>
#include <gsl/span>

namespace InferenceEngine {
namespace gpu {
/**
 * @brief This template implements a wrapper to a raw device pointer
 */

template <typename T>
class DevicePointer : public gsl::not_null<T> {
 public:
  static_assert(std::is_pointer<T>::value, "T should be a pointer type");
  using gsl::not_null<T>::not_null;
  using gsl::not_null<T>::get;
  /* dereferencing device pointer on the host is not allowed" */
  T operator->() const = delete;
  typename std::remove_pointer<T>::type operator*() const = delete;

  /**
   * @brief cast<U>() - Casts original type T* to U*, provided one of them is void*
   */
  template<typename U>
  DevicePointer<U> cast() const noexcept {
    static_assert(std::is_pointer<U>::value, "U should be a pointer type");
    static_assert(std::is_void<typename std::remove_pointer<T>::type>::value ||
                  std::is_void<typename std::remove_pointer<U>::type>::value,
                  "cast requires one of U or T to be void");
    return DevicePointer<U>{static_cast<U>(gsl::not_null<T>::get())};
  }
};

} // namespace gpu

} // namespace InferenceEngine
