// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdexcept>
#include <string>

namespace ov {
namespace test {
void set_device_suffix(const std::string& suffix) {
    if (!suffix.empty()) {
        throw std::runtime_error("The suffix can't be used for LLAMA_CPP device!");
    }
}
}  // namespace test
}  // namespace ov
