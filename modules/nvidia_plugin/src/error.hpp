// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>

#include "kernels/details/error.hpp"

namespace ov {
namespace nvidia_gpu {
[[gnu::cold, noreturn]] void throwIEException(
    const std::string& msg,
    const std::experimental::source_location& location = std::experimental::source_location::current());
[[gnu::cold]] void logError(
    const std::string& msg,
    const std::experimental::source_location& location = std::experimental::source_location::current());
}  // namespace nvidia_gpu
}  // namespace ov
