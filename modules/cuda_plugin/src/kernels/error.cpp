// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include <exception>
#include <kernels/error.hpp>

namespace CUDAPlugin {
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

}  // namespace kernel
}  // namespace CUDAPlugin
