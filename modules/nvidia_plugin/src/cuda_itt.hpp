// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines openvino domains for tracing
 * @file template_itt.hpp
 */

#pragma once

#include <openvino/itt.hpp>

namespace ov {
namespace nvidia_gpu {
namespace itt {
namespace domains {
OV_ITT_DOMAIN(nvidia_gpu);
}
}  // namespace itt
}  // namespace nvidia_gpu
}  // namespace ov
