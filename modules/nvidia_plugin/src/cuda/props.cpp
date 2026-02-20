// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "props.hpp"

namespace CUDA {

// NOTE: This map created based on data from the following table:
//       https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications__technical-specifications-per-compute-capability
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

// NOTE: This list created based on data from the following table:
//       https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#hardware-precision-matrix
std::unordered_set<std::string> fp16SupportedArchitecture = {
    "8.6",
    "8.0",
    "7.5",
    "7.2",
    "7.0",
    "6.2",
    "6.0",
    "5.3",
};

// NOTE: This list created based on data from the following table:
//       https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#hardware-precision-matrix
std::unordered_set<std::string> int8SupportedArchitecture = {
    "8.6",
    "8.0",
    "7.5",
    "7.2",
    "7.0",
    "6.1",
};

}  // namespace CUDA
