// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "error.hpp"

#include <fmt/format.h>

#include "openvino/core/except.hpp"

namespace ov {
namespace nvidia_gpu {
namespace {
class OVExceptionWrapper : public ov::Exception {
public:
    OVExceptionWrapper(const std::string& what) : ov::Exception(what) {}
};

template <typename T>
[[gnu::cold, noreturn]] void throw_exception(const std::string& msg,
                                             const std::experimental::source_location& location) {
    throw T{fmt::format("{}:{}({}): {}", location.file_name(), location.line(), location.function_name(), msg)};
}
}  // namespace

[[gnu::cold, noreturn]] void throw_ov_exception(const std::string& msg,
                                                const std::experimental::source_location& location) {
    throw_exception<OVExceptionWrapper>(msg, location);
}

[[gnu::cold]] void logError(const std::string& /*msg*/, const std::experimental::source_location& /*location*/) {
}  // TODO: log somewhere

}  // namespace nvidia_gpu
}  // namespace ov
