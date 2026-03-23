// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "openvino/core/node.hpp"
#include "kernel_ir/gfx_kernel_plan.hpp"
#include "kernel_ir/gfx_kernel_spec.hpp"

namespace ov {
namespace gfx_plugin {

mlir::ModuleOp build_mlir_for_node(const std::shared_ptr<const ov::Node>& node,
                                   mlir::MLIRContext& ctx);

std::string find_entry_point(mlir::ModuleOp module);
std::string resolve_entry_point(mlir::ModuleOp module,
                                const std::string& hint,
                                std::string_view fallback = "gfx_kernel");

struct KernelPlanBuildInfo {
    KernelPlan plan;
    KernelArgMappingInfo mapping;

    KernelRuntimeMetadata runtime_metadata(const std::shared_ptr<const ov::Node>& node,
                                           size_t outputs_hint = 0) const {
        return plan.runtime_metadata(mapping, node, outputs_hint);
    }
};

template <typename ArgCountFn>
inline KernelPlanBuildInfo build_kernel_plan_from_module(mlir::ModuleOp module,
                                                         const std::string& entry_hint,
                                                         const std::shared_ptr<const ov::Node>& node,
                                                         size_t output_args_override,
                                                         size_t extra_inputs,
                                                         const char* stage_name,
                                                         std::string_view fallback_entry,
                                                         ArgCountFn&& arg_count_fn) {
    std::string entry = resolve_entry_point(module, entry_hint, fallback_entry);
    auto mapping = build_kernel_arg_mapping(module,
                                            entry,
                                            node,
                                            output_args_override,
                                            extra_inputs,
                                            stage_name);
    const uint32_t arg_count =
        static_cast<uint32_t>(std::forward<ArgCountFn>(arg_count_fn)(mapping));
    KernelPlan plan(module, std::move(entry), arg_count);
    return KernelPlanBuildInfo{std::move(plan), std::move(mapping)};
}

class MlirKernelPlanBuilder {
public:
    KernelPlan build_plan(const std::shared_ptr<const ov::Node>& node,
                          mlir::MLIRContext& ctx,
                          uint32_t arg_count) const;
    KernelPlan build_plan(const KernelSpec& spec, mlir::MLIRContext& ctx) const;
};

}  // namespace gfx_plugin
}  // namespace ov
