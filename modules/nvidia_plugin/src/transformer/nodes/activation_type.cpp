// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "activation_type.hpp"

namespace ov {
std::ostream& operator<<(std::ostream& s, const nvidia_gpu::nodes::ActivationMode& type) {
    return s << as_string(type);
}
template <>
EnumNames<nvidia_gpu::nodes::ActivationMode>&
EnumNames<nvidia_gpu::nodes::ActivationMode>::get() {
    static auto enum_names = EnumNames<nvidia_gpu::nodes::ActivationMode>(
        "nvidia_gpu::nodes::ActivationMode",
        {{"sigmoid", nvidia_gpu::nodes::ActivationMode::SIGMOID},
         {"relu", nvidia_gpu::nodes::ActivationMode::RELU},
         {"tanh", nvidia_gpu::nodes::ActivationMode::TANH},
         {"clipped_relu", nvidia_gpu::nodes::ActivationMode::CLIPPED_RELU},
         {"elu", nvidia_gpu::nodes::ActivationMode::ELU},
         {"swish", nvidia_gpu::nodes::ActivationMode::SWISH},
         {"no_activation", nvidia_gpu::nodes::ActivationMode::NO_ACTIVATION}});
    return enum_names;
}
} // namespace ov