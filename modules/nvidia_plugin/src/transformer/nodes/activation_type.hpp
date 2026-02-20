// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/core/enum_names.hpp"

namespace ov::nvidia_gpu::nodes {

/**
 * @brief Activation modes for fused convolutions.
 *
 * Mirrors the cuDNN cudnnActivationMode_t enum
 */
enum class ActivationMode { SIGMOID, RELU, TANH, CLIPPED_RELU, ELU, SWISH, NO_ACTIVATION };

}  // namespace ov::nvidia_gpu::nodes
namespace ov {
template <>
class AttributeAdapter<nvidia_gpu::nodes::ActivationMode>
    : public EnumAttributeAdapterBase<nvidia_gpu::nodes::ActivationMode> {
public:
    AttributeAdapter(nvidia_gpu::nodes::ActivationMode& value)
        : EnumAttributeAdapterBase<nvidia_gpu::nodes::ActivationMode>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ActivationMode>");
};
}  // namespace ov
