// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sub_msl.hpp"

#include "add_msl.hpp"  // reuse generator that now supports add/sub

namespace ov {
namespace metal_plugin {

std::string generate_msl_for_elementwise_sub(const KernelOp& op) {
    return generate_msl_for_elementwise_add(op);
}

}  // namespace metal_plugin
}  // namespace ov

