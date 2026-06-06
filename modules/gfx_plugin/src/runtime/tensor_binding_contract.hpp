// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <string_view>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace gfx_plugin {

struct RuntimeStageExecutableDescriptor;

enum class RuntimeParamDescriptorPayloadKind {
    None,
    BinaryBroadcast,
    Broadcast,
    Select,
    Tile,
    Softmax,
    Transpose,
    Reduce,
};

ov::element::Type element_type_from_contract(std::string_view name);

bool parse_static_shape_contract(std::string_view text, ov::Shape& shape);

bool is_binary_runtime_param_stage(std::string_view op_family) noexcept;

bool is_reduce_runtime_param_stage(std::string_view op_family) noexcept;

RuntimeParamDescriptorPayloadKind
descriptor_owned_runtime_param_payload_kind(std::string_view op_family,
                                            size_t runtime_param_count) noexcept;

bool descriptor_has_static_shape_contracts(
    const RuntimeStageExecutableDescriptor &descriptor, size_t input_count,
    size_t output_count = 1);

bool descriptor_owns_runtime_param_payload(
    const RuntimeStageExecutableDescriptor &descriptor,
    size_t runtime_param_count);

}  // namespace gfx_plugin
}  // namespace ov
