// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for advanced hardware related properties for NVIDIA plugin
 *        To use in set_property, compile_model, import_model, get_property methods
 *
 * @file nvidia/properties.hpp
 */
#pragma once

#include "openvino/runtime/properties.hpp"

namespace ov {

/**
 * @brief Namespace with NVIDIA GPU specific properties
 */
namespace nvidia_gpu {

/**
 * @brief Defines if benchmarks should be run to determine fastest algorithms for some operations (e.g. Convolution)
 */
static constexpr Property<bool, PropertyMutability::RW> operation_benchmark{"NVIDIA_OPERATION_BENCHMARK"};

/**
 * @brief Specifies if NVIDIA plugin attempts to use CUDA Graph feature to speed up sequential network inferences
 */
static constexpr ov::Property<bool, ov::PropertyMutability::RW> use_cuda_graph{"NVIDIA_USE_CUDA_GRAPH"};

}  // namespace nvidia_gpu
}  // namespace ov
