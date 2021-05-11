// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "props.hpp"

namespace CUDAPlugin {

std::unordered_map<std::string, size_t> cudaConcurrentKernels = {
    {"3.0", 32},
    {"3.5", 32},
    {"3.7", 32},
    {"5.0", 32},
    {"5.2", 32},

    {"5.3", 16},
    {"6.0", 128},
    {"6.1", 32},
    {"6.2", 16},
    {"7.0", 128},
    {"7.2", 16},

    {"7.5", 128},
    {"8.0", 128},
    {"8.6", 128},
};

} // namespace CUDAPlugin
