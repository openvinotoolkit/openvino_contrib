// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#if __has_include(<experimental/source_location>)
#include <experimental/source_location>
#else
namespace std::experimental {
struct source_location {
    constexpr std::uint_least32_t line() const noexcept { return 0; }
    constexpr std::uint_least32_t column() const noexcept { return 0; }
    constexpr const char* file_name() const noexcept { return "unknown"; }
    constexpr const char* function_name() const noexcept { return "unknown"; }
    static constexpr source_location current() noexcept { return {}; }
};
}  // namespace std::experimental
#endif

namespace CUDAPlugin {
[[gnu::cold, noreturn]] void throwIEException(
    const std::string& msg, const std::experimental::source_location& location =
                                std::experimental::source_location::current());
[[gnu::cold, noreturn]] void throwNotFound(const std::string& msg, const std::experimental::source_location& location =
                                                                       std::experimental::source_location::current());
[[gnu::cold, noreturn]] void throwInferCancelled(
    const std::string& msg = {},
    const std::experimental::source_location& location = std::experimental::source_location::current());
[[gnu::cold]] void logError(const std::string& msg, const std::experimental::source_location& location =
                                                        std::experimental::source_location::current());
}  // namespace CUDAPlugin
