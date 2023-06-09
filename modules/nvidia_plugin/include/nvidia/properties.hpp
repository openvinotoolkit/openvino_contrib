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

}  // namespace nvidia_gpu
}  // namespace ov
