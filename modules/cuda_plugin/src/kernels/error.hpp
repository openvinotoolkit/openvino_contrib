// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#if __has_include(<experimental/source_location>)
#include <experimental/source_location>
#else
namespace std {
namespace experimental {
struct source_location {
    constexpr std::uint_least32_t line() const noexcept { return 0; }
    constexpr std::uint_least32_t column() const noexcept { return 0; }
    constexpr const char* file_name() const noexcept { return "unknown"; }
    constexpr const char* function_name() const noexcept { return "unknown"; }
    static constexpr source_location current() noexcept { return {}; }
};
}  // namespace experimental
}  // namespace std
#endif
#include <cuda_runtime.h>

namespace CUDAPlugin {
namespace kernel {
[[gnu::cold]] void throwIfError(
    cudaError_t err,
    const std::experimental::source_location& location = std::experimental::source_location::current());

[[gnu::cold, noreturn]] void throwIEException(
    const std::string& msg,
    const std::experimental::source_location& location = std::experimental::source_location::current());

}  // namespace kernel
}  // namespace CUDAPlugin
