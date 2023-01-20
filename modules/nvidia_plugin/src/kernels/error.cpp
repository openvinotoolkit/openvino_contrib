// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include <details/ie_exception.hpp>
#include <exception>
#include <kernels/error.hpp>

namespace ov {
namespace nvidia_gpu {
namespace kernel {

class Error : public std::exception {
public:
    Error(const std::string& msg, const std::experimental::source_location& location)
        : msg_{fmt::format("{}:{}({}): {}", location.file_name(), location.line(), location.function_name(), msg)} {}

    [[nodiscard]] const char* what() const noexcept override { return msg_.c_str(); }

private:
    std::string msg_;
};

[[gnu::cold]] void throwIfError(cudaError_t err, const std::experimental::source_location& location) {
    if (err != cudaSuccess) {
        throw Error(cudaGetErrorString(err), location);
    }
}

[[gnu::cold, noreturn]] void throwIEException(const std::string& msg,
                                              const std::experimental::source_location& location) {
    throw Error(msg, location);
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
