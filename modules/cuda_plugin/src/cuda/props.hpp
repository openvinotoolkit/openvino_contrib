// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <unordered_map>

namespace CUDAPlugin {

extern std::unordered_map<std::string, size_t> cudaConcurrentKernels;

} // namespace CUDAPlugin
