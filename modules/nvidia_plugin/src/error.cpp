// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "error.hpp"

#include <fmt/format.h>

#include "openvino/core/except.hpp"

namespace ov {
namespace nvidia_gpu {
namespace {
template <typename T>
[[gnu::cold, noreturn]] void throwException(const std::string& msg,
                                            const std::experimental::source_location& location) {
    throw T{fmt::format("{}:{}({}): {}", location.file_name(), location.line(), location.function_name(), msg)};
}
}  // namespace

[[gnu::cold, noreturn]] void throwIEException(const std::string& msg,
                                              const std::experimental::source_location& location) {
    throwException<ov::Exception>(msg, location);
}

[[gnu::cold]] void logError(const std::string& /*msg*/, const std::experimental::source_location& /*location*/) {
}  // TODO: log somewhere

}  // namespace nvidia_gpu
}  // namespace ov
