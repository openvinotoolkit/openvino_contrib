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

ov::element::Type element_type_from_contract(std::string_view name);

bool parse_static_shape_contract(std::string_view text, ov::Shape& shape);

bool descriptor_has_static_shape_contracts(
    const RuntimeStageExecutableDescriptor &descriptor, size_t input_count,
    size_t output_count = 1);

// Returns whether the compiler descriptor owns the RuntimeParams ABI schema:
// payload kind, role count, required bindings and metadata. Dynamic dimensions
// may still be supplied by request tensors inside this schema at runtime.
bool descriptor_owns_runtime_param_payload(
    const RuntimeStageExecutableDescriptor &descriptor);

}  // namespace gfx_plugin
}  // namespace ov
