// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "set_device_name.hpp"

#include <string>
#include <stdexcept>

namespace ov {
namespace test {

void set_device_suffix(const std::string& suffix) {
    if (!suffix.empty()) {
        throw std::runtime_error("The suffix can't be used for NVIDIA GPU device!");
    }
}

}  // namespace test
}  // namespace ov
