// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>

#include "mlir/IR/BuiltinOps.h"
#include "openvino/core/node.hpp"
#include "mlir/gfx_mlir_kernel_builder.hpp"
#include "mlir/gfx_mlir_kernel_metadata.hpp"

namespace ov {
namespace gfx_plugin {

struct MlirKernelPlanContext {
    explicit MlirKernelPlanContext(KernelPlanBuildInfo info)
        : build_info(std::move(info)) {}

    KernelPlanBuildInfo build_info;
    size_t scalar_inputs = 0;
    size_t output_args = 0;
    size_t buffer_inputs = 0;
    size_t kernel_inputs_size = 0;
    size_t extra_inputs_for_mapping = 0;
    size_t node_inputs = 0;
};

template <typename ArgCountFn>
inline MlirKernelPlanContext build_mlir_kernel_plan(mlir::ModuleOp module,
                                                    const std::string& entry_hint,
                                                    const std::shared_ptr<const ov::Node>& node,
                                                    size_t output_args_override,
                                                    size_t extra_inputs,
                                                    const char* stage_name,
                                                    std::string_view fallback_entry,
                                                    ArgCountFn&& arg_count_fn) {
    auto build_info = build_kernel_plan_from_module(module,
                                                    entry_hint,
                                                    node,
                                                    output_args_override,
                                                    extra_inputs,
                                                    stage_name,
                                                    fallback_entry,
                                                    std::forward<ArgCountFn>(arg_count_fn));
    const size_t scalar_inputs = build_info.mapping.scalar_inputs;
    const size_t output_args = build_info.mapping.output_args;
    const size_t buffer_inputs = build_info.mapping.buffer_inputs;
    const size_t kernel_inputs_size = build_info.mapping.mapping.kernel_inputs.size();
    const size_t node_inputs = node ? node->get_input_size() : 0;
    const size_t extra_inputs_for_mapping =
        infer_extra_inputs_for_mapping(buffer_inputs, node_inputs, extra_inputs);

    MlirKernelPlanContext ctx{std::move(build_info)};
    ctx.scalar_inputs = scalar_inputs;
    ctx.output_args = output_args;
    ctx.buffer_inputs = buffer_inputs;
    ctx.kernel_inputs_size = kernel_inputs_size;
    ctx.extra_inputs_for_mapping = extra_inputs_for_mapping;
    ctx.node_inputs = node_inputs;
    return ctx;
}

}  // namespace gfx_plugin
}  // namespace ov
