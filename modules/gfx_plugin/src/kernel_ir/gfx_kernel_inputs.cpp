// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/gfx_kernel_inputs.hpp"

#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace gfx_plugin {

KernelInputMapping build_kernel_inputs(const std::shared_ptr<const ov::Node>& node,
                                       size_t func_inputs,
                                       const char* stage_name,
                                       size_t extra_inputs) {
    KernelInputMapping mapping;
    mapping.func_inputs = func_inputs;
    if (!node) {
        return mapping;
    }
    const size_t node_inputs = node->get_input_size();
    if (mapping.func_inputs == 0) {
        mapping.func_inputs = node_inputs + extra_inputs;
    }
    size_t mapped_inputs = mapping.func_inputs;
    if (extra_inputs <= mapped_inputs) {
        mapped_inputs -= extra_inputs;
    } else {
        mapped_inputs = 0;
    }
    if (mapped_inputs == 0) {
        mapped_inputs = node_inputs;
    }
    size_t nonconst_count = 0;
    for (size_t i = 0; i < node_inputs; ++i) {
        auto src = node->get_input_node_shared_ptr(i);
        if (!ov::as_type_ptr<const ov::op::v0::Constant>(src)) {
            ++nonconst_count;
        }
    }
    OPENVINO_ASSERT(mapped_inputs >= nonconst_count,
                    "GFX: MLIR expects fewer inputs than non-constant inputs for ",
                    stage_name);
    const size_t need_consts = mapped_inputs - nonconst_count;
    size_t const_added = 0;
    mapping.kernel_inputs.reserve(mapped_inputs);
    for (size_t i = 0; i < node_inputs; ++i) {
        auto src = node->get_input_node_shared_ptr(i);
        const bool is_const = ov::as_type_ptr<const ov::op::v0::Constant>(src) != nullptr;
        if (is_const) {
            if (const_added < need_consts) {
                mapping.kernel_inputs.push_back(i);
                ++const_added;
            }
        } else {
            mapping.kernel_inputs.push_back(i);
        }
    }
    OPENVINO_ASSERT(mapping.kernel_inputs.size() == mapped_inputs,
                    "GFX: MLIR input count mismatch for ",
                    stage_name,
                    " (expected ",
                    mapped_inputs,
                    ", got ",
                    mapping.kernel_inputs.size(),
                    ")");
    return mapping;
}

}  // namespace gfx_plugin
}  // namespace ov
