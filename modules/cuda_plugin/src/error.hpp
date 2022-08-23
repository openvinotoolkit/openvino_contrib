// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>

#include "kernels/error.hpp"

namespace ov {
namespace nvidia_gpu {
[[gnu::cold, noreturn]] void throwIEException(
    const std::string& msg,
    const std::experimental::source_location& location = std::experimental::source_location::current());
[[gnu::cold, noreturn]] void throwNotFound(
    const std::string& msg,
    const std::experimental::source_location& location = std::experimental::source_location::current());
[[gnu::cold, noreturn]] void throwInferCancelled(
    const std::string& msg = {},
    const std::experimental::source_location& location = std::experimental::source_location::current());
[[gnu::cold]] void logError(
    const std::string& msg,
    const std::experimental::source_location& location = std::experimental::source_location::current());
}  // namespace nvidia_gpu
}  // namespace ov
