// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace ov {
namespace nvidia_gpu {

/**
 * Applies CUDA device specific allignment.
 *
 * @param [in] value A value to apply allignment to.
 * @returns a closest value which is greater or equal to input argument
 * and is a multiple of CUDA device specific allignment.
 */
size_t applyAllignment(size_t value);

}  // namespace nvidia_gpu
}  // namespace ov
