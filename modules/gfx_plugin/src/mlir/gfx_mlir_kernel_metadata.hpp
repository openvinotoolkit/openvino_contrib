// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "kernel_ir/gfx_kernel_dispatch.hpp"
#include "kernel_ir/gfx_kernel_inputs.hpp"
#include "kernel_ir/gfx_kernel_signature.hpp"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Casting.h"

namespace ov {
namespace gfx_plugin {

struct KernelOperandMetadata {
    std::vector<int32_t> operand_kinds;
    std::vector<int32_t> operand_arg_indices;
    std::vector<int32_t> scalar_args;
};

struct KernelRuntimeMetadata {
    bool valid = false;
    ParallelDispatchConfig dispatch;
    bool force_single_dispatch = false;
    KernelOperandMetadata operands;
    size_t kernel_input_arg_count = 0;
};

struct KernelSignatureInfo {
    KernelFunctionSignature signature;
    mlir::func::FuncOp func;
    size_t scalar_inputs = 0;
};

struct KernelArgMappingInfo {
    KernelFunctionSignature signature;
    size_t scalar_inputs = 0;
    size_t func_inputs = 0;
    size_t func_results = 0;
    size_t output_args = 0;
    size_t buffer_inputs = 0;
    KernelInputMapping mapping;
};

inline mlir::func::FuncOp resolve_entry_func(mlir::ModuleOp module, const std::string& entry) {
    if (!module) {
        return {};
    }
    if (!entry.empty()) {
        if (auto func = module.lookupSymbol<mlir::func::FuncOp>(entry)) {
            return func;
        }
    }
    mlir::func::FuncOp func;
    module.walk([&](mlir::func::FuncOp f) {
        if (!func) {
            func = f;
        }
    });
    return func;
}

inline size_t count_scalar_inputs(mlir::func::FuncOp func) {
    if (!func) {
        return 0;
    }
    size_t scalar_inputs = 0;
    auto ftype = func.getFunctionType();
    for (auto type : ftype.getInputs()) {
        if (!mlir::isa<mlir::ShapedType>(type)) {
            ++scalar_inputs;
        }
    }
    return scalar_inputs;
}

inline size_t infer_extra_inputs_for_mapping(size_t buffer_inputs,
                                             size_t node_inputs,
                                             size_t extra_inputs) {
    if (buffer_inputs <= node_inputs) {
        return 0;
    }
    const size_t inferred = buffer_inputs - node_inputs;
    return std::min(inferred, extra_inputs);
}

inline ParallelDispatchConfig extract_kernel_dispatch_metadata(mlir::ModuleOp module) {
    ParallelDispatchConfig meta;
    if (!module) {
        return meta;
    }
    if (auto attr = module->getAttrOfType<mlir::BoolAttr>("gfx.parallel_dispatch")) {
        meta.enabled = attr.getValue();
    }
    if (auto attr = module->getAttrOfType<mlir::IntegerAttr>("gfx.parallel_loop_dims")) {
        meta.loop_dims = static_cast<size_t>(attr.getInt());
    }
    if (auto attr = module->getAttrOfType<mlir::IntegerAttr>("gfx.dispatch_tile_h")) {
        meta.tile_h = static_cast<uint32_t>(attr.getInt());
    }
    if (auto attr = module->getAttrOfType<mlir::IntegerAttr>("gfx.dispatch_tile_w")) {
        meta.tile_w = static_cast<uint32_t>(attr.getInt());
    }
    if (auto attr = module->getAttrOfType<mlir::IntegerAttr>("gfx.dispatch_threads_h")) {
        meta.threads_h = static_cast<uint32_t>(attr.getInt());
    }
    if (auto attr = module->getAttrOfType<mlir::IntegerAttr>("gfx.dispatch_threads_w")) {
        meta.threads_w = static_cast<uint32_t>(attr.getInt());
    }
    if (meta.threads_h == 1) {
        meta.threads_h = meta.tile_h;
    }
    if (meta.threads_w == 1) {
        meta.threads_w = meta.tile_w;
    }
    return meta;
}

inline bool extract_kernel_force_single_dispatch(mlir::ModuleOp module) {
    if (!module) {
        return false;
    }
    if (auto attr = module->getAttrOfType<mlir::BoolAttr>("gfx.force_single_dispatch")) {
        return attr.getValue();
    }
    return false;
}

inline std::vector<int32_t> extract_kernel_scalar_args(mlir::ModuleOp module) {
    std::vector<int32_t> scalars;
    if (!module) {
        return scalars;
    }
    if (auto attr = module->getAttr("gfx.kernel_scalar_args")) {
        if (auto attrs = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
            scalars.reserve(attrs.size());
            for (auto attr_val : attrs) {
                if (auto iattr = mlir::dyn_cast<mlir::IntegerAttr>(attr_val)) {
                    scalars.push_back(static_cast<int32_t>(iattr.getInt()));
                }
            }
            return scalars;
        }
        if (auto dense = mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr)) {
            auto vals = dense.asArrayRef();
            scalars.assign(vals.begin(), vals.end());
            return scalars;
        }
        if (auto dense = mlir::dyn_cast<mlir::DenseIntElementsAttr>(attr)) {
            scalars.reserve(dense.getNumElements());
            for (auto v : dense.getValues<int32_t>()) {
                scalars.push_back(v);
            }
            return scalars;
        }
    }
    return scalars;
}

inline std::vector<int32_t> extract_kernel_scalar_values(mlir::ModuleOp module) {
    if (!module) {
        return {};
    }
    if (auto attr = module->getAttr("gfx.kernel_scalar_values")) {
        if (auto attrs = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
            std::vector<int32_t> values;
            values.reserve(attrs.size());
            for (auto attr_val : attrs) {
                if (auto iattr = mlir::dyn_cast<mlir::IntegerAttr>(attr_val)) {
                    values.push_back(static_cast<int32_t>(iattr.getInt()));
                }
            }
            return values;
        }
        if (auto dense = mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr)) {
            auto vals = dense.asArrayRef();
            return std::vector<int32_t>(vals.begin(), vals.end());
        }
        if (auto dense = mlir::dyn_cast<mlir::DenseIntElementsAttr>(attr)) {
            std::vector<int32_t> values;
            values.reserve(dense.getNumElements());
            for (auto v : dense.getValues<int32_t>()) {
                values.push_back(v);
            }
            return values;
        }
    }
    return extract_kernel_scalar_args(module);
}

inline std::vector<int32_t> extract_kernel_operand_kinds(mlir::ModuleOp module) {
    std::vector<int32_t> kinds;
    if (!module) {
        return kinds;
    }
    if (auto attr = module->getAttr("gfx.kernel_operand_kinds")) {
        if (auto attrs = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
            kinds.reserve(attrs.size());
            for (auto attr_val : attrs) {
                if (auto iattr = mlir::dyn_cast<mlir::IntegerAttr>(attr_val)) {
                    kinds.push_back(static_cast<int32_t>(iattr.getInt()));
                }
            }
            return kinds;
        }
        if (auto dense = mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr)) {
            auto vals = dense.asArrayRef();
            kinds.assign(vals.begin(), vals.end());
            return kinds;
        }
        if (auto dense = mlir::dyn_cast<mlir::DenseIntElementsAttr>(attr)) {
            kinds.reserve(dense.getNumElements());
            for (auto v : dense.getValues<int32_t>()) {
                kinds.push_back(v);
            }
            return kinds;
        }
    }
    return kinds;
}

inline std::vector<int32_t> extract_kernel_operand_arg_indices(mlir::ModuleOp module) {
    std::vector<int32_t> indices;
    if (!module) {
        return indices;
    }
    if (auto attr = module->getAttr("gfx.kernel_operand_arg_indices")) {
        if (auto attrs = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
            indices.reserve(attrs.size());
            for (auto attr_val : attrs) {
                if (auto iattr = mlir::dyn_cast<mlir::IntegerAttr>(attr_val)) {
                    indices.push_back(static_cast<int32_t>(iattr.getInt()));
                }
            }
            return indices;
        }
        if (auto dense = mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr)) {
            auto vals = dense.asArrayRef();
            indices.assign(vals.begin(), vals.end());
            return indices;
        }
        if (auto dense = mlir::dyn_cast<mlir::DenseIntElementsAttr>(attr)) {
            indices.reserve(dense.getNumElements());
            for (auto v : dense.getValues<int32_t>()) {
                indices.push_back(v);
            }
            return indices;
        }
    }
    return indices;
}

inline KernelSignatureInfo extract_kernel_signature_info(mlir::ModuleOp module,
                                                         const std::string& entry) {
    KernelSignatureInfo info;
    info.signature = infer_kernel_signature(module, entry);
    info.func = resolve_entry_func(module, entry);
    info.scalar_inputs = count_scalar_inputs(info.func);
    return info;
}

inline KernelArgMappingInfo build_kernel_arg_mapping(mlir::ModuleOp module,
                                                     const std::string& entry,
                                                     const std::shared_ptr<const ov::Node>& node,
                                                     size_t output_args_override,
                                                     size_t extra_inputs,
                                                     const char* stage_name) {
    KernelArgMappingInfo info;
    const auto sig_info = extract_kernel_signature_info(module, entry);
    info.signature = sig_info.signature;
    info.scalar_inputs = sig_info.scalar_inputs;
    info.func_inputs = info.signature.inputs;
    info.func_results = info.signature.results;
    info.output_args = output_args_override;
    if (info.func_results == 0 && info.output_args == 0 && node) {
        info.output_args = node->get_output_size();
    }
    size_t buffer_inputs = info.func_inputs;
    if (info.scalar_inputs <= buffer_inputs) {
        buffer_inputs -= info.scalar_inputs;
    } else {
        buffer_inputs = 0;
    }
    // Only memref-style kernels pass outputs as trailing function arguments.
    // Tensor-returning kernels keep outputs in the result list and must not
    // lose real input operands here, otherwise constant OV inputs (for example
    // Split axis/lengths) get misclassified as runtime buffers.
    if (info.func_results == 0) {
        if (info.output_args <= buffer_inputs) {
            buffer_inputs -= info.output_args;
        } else {
            buffer_inputs = 0;
        }
    }
    info.buffer_inputs = buffer_inputs;
    const size_t node_inputs = node ? node->get_input_size() : 0;
    const size_t extra_inputs_for_mapping =
        infer_extra_inputs_for_mapping(buffer_inputs, node_inputs, extra_inputs);
    info.mapping = build_kernel_inputs(node, buffer_inputs, stage_name, extra_inputs_for_mapping);
    if (info.mapping.func_inputs != 0) {
        info.func_inputs = info.mapping.func_inputs;
    }
    return info;
}

inline KernelOperandMetadata extract_kernel_operand_metadata(mlir::ModuleOp module) {
    KernelOperandMetadata meta;
    meta.operand_kinds = extract_kernel_operand_kinds(module);
    meta.operand_arg_indices = extract_kernel_operand_arg_indices(module);
    meta.scalar_args = extract_kernel_scalar_values(module);
    return meta;
}

inline size_t resolve_kernel_runtime_output_args(const KernelArgMappingInfo& mapping,
                                                 const std::shared_ptr<const ov::Node>& node,
                                                 size_t outputs_hint = 0) {
    if (mapping.output_args != 0) {
        return mapping.output_args;
    }
    if (outputs_hint != 0) {
        return outputs_hint;
    }
    if (node) {
        return node->get_output_size();
    }
    return 0;
}

inline size_t infer_kernel_input_arg_count_from_operand_indices(const std::vector<int32_t>& indices,
                                                                size_t output_arg_count,
                                                                size_t fallback) {
    if (indices.empty()) {
        return fallback;
    }
    int32_t max_idx = -1;
    for (auto idx : indices) {
        if (idx > max_idx) {
            max_idx = idx;
        }
    }
    if (max_idx < 0) {
        return fallback;
    }
    const size_t total_buffer_args = static_cast<size_t>(max_idx) + 1;
    if (output_arg_count > total_buffer_args) {
        return fallback;
    }
    return total_buffer_args - output_arg_count;
}

inline KernelRuntimeMetadata extract_kernel_runtime_metadata(mlir::ModuleOp module,
                                                             size_t output_arg_count,
                                                             size_t fallback_input_arg_count) {
    KernelRuntimeMetadata meta;
    if (!module) {
        return meta;
    }
    meta.valid = true;
    meta.dispatch = extract_kernel_dispatch_metadata(module);
    meta.force_single_dispatch = extract_kernel_force_single_dispatch(module);
    meta.operands = extract_kernel_operand_metadata(module);
    meta.kernel_input_arg_count =
        infer_kernel_input_arg_count_from_operand_indices(meta.operands.operand_arg_indices,
                                                          output_arg_count,
                                                          fallback_input_arg_count);
    return meta;
}

inline KernelRuntimeMetadata extract_kernel_runtime_metadata(mlir::ModuleOp module,
                                                             const KernelArgMappingInfo& mapping,
                                                             const std::shared_ptr<const ov::Node>& node,
                                                             size_t outputs_hint = 0) {
    const size_t output_arg_count = resolve_kernel_runtime_output_args(mapping, node, outputs_hint);
    return extract_kernel_runtime_metadata(module, output_arg_count, mapping.buffer_inputs);
}

inline size_t infer_kernel_arg_count_from_module(mlir::ModuleOp module, size_t fallback) {
    if (!module) {
        return fallback;
    }
    if (auto attr = module->getAttrOfType<mlir::IntegerAttr>("gfx.fixed_arg_count")) {
        const auto value = attr.getInt();
        if (value > 0) {
            const auto scalar_values = extract_kernel_scalar_values(module);
            return static_cast<size_t>(value) + scalar_values.size();
        }
    }
    auto kinds = extract_kernel_operand_kinds(module);
    if (!kinds.empty()) {
        auto scalars = extract_kernel_scalar_args(module);
        return kinds.size() + scalars.size();
    }
    auto scalars = extract_kernel_scalar_args(module);
    if (!scalars.empty()) {
        return fallback + scalars.size();
    }
    size_t launch_operand_count = 0;
    module.walk([&](mlir::gpu::LaunchFuncOp launch) {
        if (launch_operand_count == 0) {
            launch_operand_count = launch.getKernelOperands().size();
        }
    });
    if (launch_operand_count != 0) {
        return launch_operand_count;
    }
    return fallback;
}

}  // namespace gfx_plugin
}  // namespace ov
