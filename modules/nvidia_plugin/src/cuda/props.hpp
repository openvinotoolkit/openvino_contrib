// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>

namespace CUDA {

extern std::unordered_map<std::string, size_t> cudaConcurrentKernels;
extern std::unordered_set<std::string> fp16SupportedArchitecture;
extern std::unordered_set<std::string> int8SupportedArchitecture;

}  // namespace CUDA
