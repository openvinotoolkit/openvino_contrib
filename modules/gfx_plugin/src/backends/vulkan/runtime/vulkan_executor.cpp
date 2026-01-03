// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/vulkan/runtime/vulkan_executor.hpp"

#include <algorithm>
#include <cstring>
#include <sstream>

#include "mlir/mlir_builder.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "backends/vulkan/codegen/vulkan_codegen_backend.hpp"
#include "backends/vulkan/runtime/gpu_memory.hpp"
#include "backends/vulkan/runtime/vulkan_memory.hpp"
#include "runtime/memory_manager.hpp"
#include "runtime/gfx_logger.hpp"
#include "kernel_ir/gfx_kernel_cache.hpp"
#include "kernel_ir/gfx_kernel_dispatch.hpp"
#include "kernel_ir/gfx_kernel_args.hpp"
#include "kernel_ir/gfx_kernel_inputs.hpp"
#include "kernel_ir/gfx_kernel_plan.hpp"
#include "kernel_ir/gfx_kernel_spec.hpp"
#include "runtime/gfx_logger.hpp"
#include "runtime/gfx_shape_utils.hpp"
#include "mlir/gfx_mlir_kernel_builder.hpp"
#include "backends/vulkan/runtime/profiling/profiler.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/variadic_split.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

mlir::func::FuncOp resolve_entry_func(mlir::ModuleOp module, const std::string& entry) {
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

size_t count_scalar_inputs(mlir::func::FuncOp func) {
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

std::vector<int32_t> extract_kernel_scalar_args(mlir::ModuleOp module) {
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

std::vector<int32_t> extract_kernel_scalar_values(mlir::ModuleOp module) {
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

std::vector<int32_t> extract_kernel_operand_kinds(mlir::ModuleOp module) {
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

std::vector<int32_t> extract_kernel_operand_arg_indices(mlir::ModuleOp module) {
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

}  // namespace

VulkanStage::VulkanStage(const std::shared_ptr<const ov::Node>& node)
    : m_node(node),
      m_name(node ? node->get_friendly_name() : std::string("vulkan_stage")),
      m_type(node ? node->get_type_name() : std::string("Unknown")) {
    if (node && node->get_output_partial_shape(0).is_static()) {
        m_output_shape = node->get_output_shape(0);
    }
    if (m_type == "Reshape" || m_type == "Squeeze" || m_type == "Unsqueeze") {
        m_is_view_op = true;
    }
}

VulkanStage::~VulkanStage() = default;

VulkanStage::ConstBufferSet::~ConstBufferSet() {
    for (auto& tensor : buffers) {
        if (tensor.buf.valid() && tensor.buf.owned) {
            vulkan_free_buffer(tensor.buf);
        }
    }
}

void VulkanStage::init(GpuBufferManager* buffer_manager) {
    m_buffer_manager = buffer_manager;
}

void VulkanStage::compile(GpuBufferManager* buffer_manager) {
    mlir::MLIRContext ctx;
    if (m_is_view_op) {
        return;
    }
    if (m_kernel) {
        return;
    }
    if (!m_buffer_manager) {
        m_buffer_manager = buffer_manager;
    }
    if (m_node) {
        if (gfx_log_debug_enabled() && m_type == "MatMul") {
            if (auto mm = std::dynamic_pointer_cast<const ov::op::v0::MatMul>(m_node)) {
                std::ostringstream meta;
                meta << "MatMul ta=" << mm->get_transpose_a()
                     << " tb=" << mm->get_transpose_b()
                     << " A=" << mm->get_input_partial_shape(0)
                     << " B=" << mm->get_input_partial_shape(1);
                GFX_LOG_DEBUG("VulkanConst", meta.str());
            }
        }
        const size_t in_count = m_node->get_input_size();
        if (!m_const_buffers) {
            m_const_buffers = std::make_shared<ConstBufferSet>();
        }
        if (m_const_buffers->buffers.size() < in_count) {
            m_const_buffers->buffers.resize(in_count);
            m_const_buffers->present.assign(in_count, false);
        }
        OPENVINO_ASSERT(m_buffer_manager,
                        "GFX Vulkan: const buffer manager is required for constants (stage ",
                        m_name,
                        ")");
        OPENVINO_ASSERT(m_buffer_manager->supports_const_cache(),
                        "GFX Vulkan: const cache must be supported for stage ",
                        m_name);
        const bool use_const_cache = true;
        for (size_t i = 0; i < in_count; ++i) {
            auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(m_node->get_input_node_shared_ptr(i));
            if (!c) {
                continue;
            }
            if (m_const_buffers->present[i] && m_const_buffers->buffers[i].buf.valid()) {
                continue;
            }
            const size_t bytes = c->get_byte_size();
            const auto et = c->get_element_type();
            if (gfx_log_debug_enabled() && c->get_element_type() == ov::element::f32 && bytes >= sizeof(float)) {
                const float* vals = static_cast<const float*>(c->get_data_ptr());
                const size_t count = bytes / sizeof(float);
                std::ostringstream oss;
                oss << "const[" << i << "] ";
                const size_t dump_n = std::min<size_t>(count, 6);
                for (size_t vi = 0; vi < dump_n; ++vi) {
                    if (vi) {
                        oss << ", ";
                    }
                    oss << vals[vi];
                }
                GFX_LOG_DEBUG("VulkanConst", oss.str());
            }
            if (use_const_cache && bytes) {
                const uint64_t hash = gfx_hash_bytes(c->get_data_ptr(), bytes);
                std::ostringstream key;
                key << m_name
                    << "/const/"
                    << i
                    << "/"
                    << et.get_type_name()
                    << "/"
                    << bytes
                    << "/"
                    << hash;
                GpuBuffer buf = m_buffer_manager->wrap_const(key.str(), c->get_data_ptr(), bytes, et);
                OPENVINO_ASSERT(buf.valid(),
                                "GFX Vulkan: failed to wrap const buffer for stage ",
                                m_name);
                buf.owned = false;
                m_const_buffers->buffers[i].buf = buf;
            }
            m_const_buffers->buffers[i].shape = c->get_shape();
            m_const_buffers->buffers[i].expected_type = et;
            m_const_buffers->present[i] = true;
        }
    }
    if (m_type == "Softmax" || m_type == "LogSoftmax" ||
        m_type == "Split" || m_type == "VariadicSplit") {
        return;
    }
    MlirKernelPlanBuilder plan_builder;
    KernelSpec spec(m_node, 0);
    KernelPlan plan = plan_builder.build_plan(spec, ctx);
    auto module = plan.module();
    auto set_parallel_pref = [&](mlir::ModuleOp mod) {
        if (!mod) {
            return;
        }
        const bool prefer_parallel =
            (m_type == "Convolution" || m_type == "GroupConvolution" || m_type == "MatMul");
        mod->setAttr("gfx.prefer_parallel",
                     mlir::BoolAttr::get(mod.getContext(), prefer_parallel));
    };
    set_parallel_pref(module);
    std::string entry = plan.entry_point();
    if (module && entry.empty()) {
        entry = find_entry_point(module);
    }
    if (entry.empty()) {
        entry = "gfx_kernel";
    }
    const auto signature = infer_kernel_signature(module, entry);
    size_t func_inputs = signature.inputs;
    size_t func_results = signature.results;
    const auto func = resolve_entry_func(module, entry);
    const size_t scalar_inputs = count_scalar_inputs(func);
    size_t output_args = 0;
    if (signature.results == 0 && m_node) {
        output_args = m_node->get_output_size();
    }
    size_t buffer_inputs = func_inputs;
    if (scalar_inputs <= buffer_inputs) {
        buffer_inputs -= scalar_inputs;
    }
    if (output_args <= buffer_inputs) {
        buffer_inputs -= output_args;
    }
    auto mapping = build_kernel_inputs(m_node, buffer_inputs, m_name.c_str());
    func_inputs = mapping.func_inputs;
    m_kernel_inputs = std::move(mapping.kernel_inputs);
    if (m_node && func_results == 0) {
        func_results = m_node->get_output_size();
    }
    const uint32_t arg_count =
        signature.total() ? signature.total() : static_cast<uint32_t>(func_inputs + func_results);
    KernelPlan plan_with_count(module, entry, arg_count);
    KernelSource src = plan_with_count.to_source();

    VulkanCodegenBackend backend;
    std::string log;
    try {
        m_kernel = backend.compile(src, &log);
    } catch (const std::exception& e) {
        OPENVINO_THROW("GFX Vulkan: failed to compile stage ",
                       m_name,
                       " (",
                       m_type,
                       "): ",
                       e.what());
    }
    OPENVINO_ASSERT(m_kernel,
                    "GFX Vulkan: failed to compile stage ",
                    m_name,
                    " (",
                    m_type,
                    "): ",
                    log);
    if (module) {
        if (auto attr = module->getAttrOfType<mlir::BoolAttr>("gfx.parallel_dispatch")) {
            m_parallel_dispatch = attr.getValue();
        } else {
            m_parallel_dispatch = false;
        }
        m_kernel_operand_kinds = extract_kernel_operand_kinds(module);
        m_kernel_operand_arg_indices = extract_kernel_operand_arg_indices(module);
        m_kernel_scalar_args = extract_kernel_scalar_values(module);
        if (m_kernel_operand_kinds.empty() && scalar_inputs != 0) {
            OPENVINO_ASSERT(m_kernel_scalar_args.size() == scalar_inputs,
                            "GFX Vulkan: kernel scalar args mismatch for ",
                            m_name,
                            " (expected ",
                            scalar_inputs,
                            ", got ",
                            m_kernel_scalar_args.size(),
                            ")");
        }
        if (gfx_log_debug_enabled() && !m_kernel_scalar_args.empty()) {
            std::ostringstream oss;
            oss << "Kernel scalar args: ";
            const size_t dump_n = std::min<size_t>(m_kernel_scalar_args.size(), 8);
            for (size_t i = 0; i < dump_n; ++i) {
                if (i) {
                    oss << ", ";
                }
                oss << m_kernel_scalar_args[i];
            }
            if (m_kernel_scalar_args.size() > dump_n) {
                oss << ", ...";
            }
            GFX_LOG_DEBUG("VulkanExec", oss.str());
        }
        if (gfx_log_debug_enabled()) {
            const bool has_kinds = module->hasAttr("gfx.kernel_operand_kinds");
            const bool has_scalars = module->hasAttr("gfx.kernel_scalar_values");
            GFX_LOG_DEBUG("VulkanExec",
                          "Kernel attrs: operand_kinds=" << (has_kinds ? "yes" : "no")
                                                         << " scalar_values=" << (has_scalars ? "yes" : "no"));
            GFX_LOG_DEBUG("VulkanExec",
                          "Kernel operand kinds size=" << m_kernel_operand_kinds.size());
            if (!m_kernel_operand_arg_indices.empty()) {
                std::ostringstream idxs;
                idxs << "Kernel operand arg indices: ";
                const size_t dump_n = std::min<size_t>(m_kernel_operand_arg_indices.size(), 8);
                for (size_t i = 0; i < dump_n; ++i) {
                    if (i) {
                        idxs << ", ";
                    }
                    idxs << m_kernel_operand_arg_indices[i];
                }
                if (m_kernel_operand_arg_indices.size() > dump_n) {
                    idxs << ", ...";
                }
                GFX_LOG_DEBUG("VulkanExec", idxs.str());
            }
            if (auto attr = module->getAttr("gfx.kernel_operand_kinds")) {
                std::string text;
                llvm::raw_string_ostream os(text);
                attr.print(os);
                GFX_LOG_DEBUG("VulkanExec", "Kernel operand_kinds attr=" << os.str());
                GFX_LOG_DEBUG("VulkanExec",
                              "operand_kinds isa ArrayAttr="
                                  << (llvm::isa<mlir::ArrayAttr>(attr) ? "yes" : "no")
                                  << " DenseI32ArrayAttr="
                                  << (llvm::isa<mlir::DenseI32ArrayAttr>(attr) ? "yes" : "no")
                                  << " DenseIntElementsAttr="
                                  << (llvm::isa<mlir::DenseIntElementsAttr>(attr) ? "yes" : "no"));
            }
        }
    }
}

void VulkanStage::execute(GpuCommandBufferHandle /*command_buffer*/) {
    if (gfx_log_debug_enabled()) {
        GFX_LOG_DEBUG("VulkanExec", "Execute stage " << m_name << " (" << m_type << ")");
    }
    std::vector<GpuTensor*> outputs = m_outputs;
    if (outputs.empty() && m_output) {
        outputs.push_back(m_output);
    }
    if (outputs.empty()) {
        OPENVINO_THROW("GFX Vulkan: output tensor is not bound for stage ", m_name);
    }
    auto resolve_input_shape = [&](size_t idx) -> ov::Shape {
        if (idx < m_inputs.size() && m_inputs[idx] && !m_inputs[idx]->shape.empty()) {
            return m_inputs[idx]->shape;
        }
        if (m_node && m_node->get_input_partial_shape(idx).is_static()) {
            return m_node->get_input_shape(idx);
        }
        return {};
    };

    if (m_kernel_inputs.empty() && m_node) {
        const size_t in_count = m_node->get_input_size();
        m_kernel_inputs.reserve(in_count);
        for (size_t i = 0; i < in_count; ++i) {
            m_kernel_inputs.push_back(i);
        }
    }

    auto resolve_input_tensor = [&](size_t input_idx) -> GpuTensor* {
        GpuTensor* t = input_idx < m_inputs.size() ? m_inputs[input_idx] : nullptr;
        if (t && t->buf.valid()) {
            return t;
        }
        if (m_const_buffers &&
            input_idx < m_const_buffers->buffers.size() &&
            input_idx < m_const_buffers->present.size() &&
            m_const_buffers->present[input_idx] &&
            m_const_buffers->buffers[input_idx].buf.valid()) {
            return &m_const_buffers->buffers[input_idx];
        }
        return nullptr;
    };

    auto ensure_output_shape = [&](size_t oi, GpuTensor* out) {
        if (!out) {
            return;
        }
        if (out->shape.empty() && m_node && m_node->get_output_partial_shape(oi).is_static()) {
            out->shape = m_node->get_output_shape(oi);
        }
    };

    if (m_type == "Softmax" || m_type == "LogSoftmax") {
        ov::Shape in_shape = resolve_input_shape(0);
        if (in_shape.empty()) {
            OPENVINO_THROW("GFX Vulkan: Softmax input shape is unknown for stage ", m_name);
        }
        int64_t axis = -1;
        if (auto s1 = ov::as_type_ptr<const ov::op::v1::Softmax>(m_node)) axis = s1->get_axis();
        else if (auto s8 = ov::as_type_ptr<const ov::op::v8::Softmax>(m_node)) axis = s8->get_axis();
        else if (auto ls = ov::as_type_ptr<const ov::op::v5::LogSoftmax>(m_node)) axis = ls->get_axis();
        else OPENVINO_THROW("GFX Vulkan: unsupported softmax op kind");
        const auto dims = compute_softmax_dims(in_shape, axis, "GFX Vulkan: Softmax");
        const uint64_t total_work = dims.rows;
        if (gfx_log_debug_enabled()) {
            GFX_LOG_DEBUG("VulkanSoftmax",
                          "shape_rank=" << in_shape.size()
                                        << " axis=" << dims.axis
                                        << " rows*inner=" << total_work
                                        << " tiled=0");
        }
        for (auto* out : outputs) {
            if (out) {
                out->shape = in_shape;
            }
        }
        if (m_node && (m_last_input_shape != in_shape || !m_kernel)) {
            mlir::MLIRContext ctx;
            const bool log_softmax = ov::as_type_ptr<const ov::op::v5::LogSoftmax>(m_node) != nullptr;
            auto module = log_softmax
                              ? build_mlir_logsoftmax_from_node(m_node, ctx, in_shape)
                              : build_mlir_softmax_from_node(m_node, ctx, in_shape);
            if (module) {
                module->setAttr("gfx.prefer_parallel",
                                mlir::BoolAttr::get(module.getContext(), false));
            }
            std::string entry = find_entry_point(module);
            if (entry.empty()) {
                entry = "softmax_main";
            }
            const auto signature = infer_kernel_signature(module, entry);
            const auto func = resolve_entry_func(module, entry);
            const size_t scalar_inputs = count_scalar_inputs(func);
            size_t output_args = 0;
            if (signature.results == 0 && m_node) {
                output_args = outputs.size();
            }
            size_t buffer_inputs = signature.inputs;
            if (scalar_inputs <= buffer_inputs) {
                buffer_inputs -= scalar_inputs;
            }
            if (output_args <= buffer_inputs) {
                buffer_inputs -= output_args;
            }
            if (m_node) {
                auto mapping = build_kernel_inputs(m_node, buffer_inputs, m_name.c_str());
                m_kernel_inputs = std::move(mapping.kernel_inputs);
            }
            const uint32_t arg_count = signature.total()
                                           ? signature.total()
                                           : static_cast<uint32_t>(m_kernel_inputs.size() + outputs.size());
            KernelPlan plan(module, std::move(entry), arg_count);
            KernelSource src = plan.to_source();
            VulkanCodegenBackend backend;
            std::string log;
            try {
                m_kernel = backend.compile(src, &log);
            } catch (const std::exception& e) {
                OPENVINO_THROW("GFX Vulkan: failed to compile softmax stage ",
                               m_name,
                               " (",
                               m_type,
                               "): ",
                               e.what());
            }
            OPENVINO_ASSERT(m_kernel,
                            "GFX Vulkan: failed to compile softmax stage ",
                            m_name,
                            " (",
                            m_type,
                            "): ",
                            log);
            if (module) {
                if (auto attr = module->getAttrOfType<mlir::BoolAttr>("gfx.parallel_dispatch")) {
                    m_parallel_dispatch = attr.getValue();
                } else {
                    m_parallel_dispatch = false;
                }
                m_kernel_operand_kinds = extract_kernel_operand_kinds(module);
                m_kernel_operand_arg_indices = extract_kernel_operand_arg_indices(module);
                m_kernel_scalar_args = extract_kernel_scalar_values(module);
                if (m_kernel_operand_kinds.empty() && scalar_inputs != 0) {
                    OPENVINO_ASSERT(m_kernel_scalar_args.size() == scalar_inputs,
                                    "GFX Vulkan: kernel scalar args mismatch for ",
                                    m_name,
                                    " (expected ",
                                    scalar_inputs,
                                    ", got ",
                                    m_kernel_scalar_args.size(),
                                    ")");
                }
            }
            m_last_input_shape = in_shape;
        }
    } else if (m_type == "Split" || m_type == "VariadicSplit") {
        ov::Shape in_shape = resolve_input_shape(0);
        if (in_shape.empty()) {
            OPENVINO_THROW("GFX Vulkan: Split input shape is unknown for stage ", m_name);
        }
        int64_t axis = 0;
        std::vector<size_t> split_sizes;
        size_t parts = 0;
        bool is_split = false;
        if (auto s = ov::as_type_ptr<const ov::op::v1::Split>(m_node)) {
            auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(s->input_value(1).get_node_shared_ptr());
            OPENVINO_ASSERT(axis_const, "Split axis must be constant");
            axis = axis_const->cast_vector<int64_t>().at(0);
            parts = s->get_num_splits();
            is_split = true;
        } else if (auto vs = ov::as_type_ptr<const ov::op::v1::VariadicSplit>(m_node)) {
            auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(vs->input_value(1).get_node_shared_ptr());
            OPENVINO_ASSERT(axis_const, "VariadicSplit axis must be constant");
            axis = axis_const->cast_vector<int64_t>().at(0);
            auto lengths_const = ov::as_type_ptr<const ov::op::v0::Constant>(vs->input_value(2).get_node_shared_ptr());
            OPENVINO_ASSERT(lengths_const, "VariadicSplit lengths must be constant");
            auto lengths = lengths_const->cast_vector<int64_t>();
            split_sizes.reserve(lengths.size());
            for (auto v : lengths) {
                OPENVINO_ASSERT(v >= 0, "VariadicSplit negative length not supported");
                split_sizes.push_back(static_cast<size_t>(v));
            }
        }
        const int64_t axis_norm = normalize_axis(axis, in_shape.size(), "GFX Vulkan: Split");
        const size_t axis_len = in_shape[static_cast<size_t>(axis_norm)];
        if (is_split) {
            OPENVINO_ASSERT(parts > 0, "Split number of splits is zero");
            OPENVINO_ASSERT(axis_len % parts == 0, "Split dimension not divisible by parts");
            split_sizes.assign(parts, axis_len / parts);
        }
        size_t sum = 0;
        for (auto s : split_sizes) {
            sum += s;
        }
        OPENVINO_ASSERT(sum == axis_len,
                        "Split sizes do not sum to axis length (",
                        sum,
                        " vs ",
                        axis_len,
                        ")");
        OPENVINO_ASSERT(!split_sizes.empty(), "Split sizes are empty");
        OPENVINO_ASSERT(outputs.size() == split_sizes.size(),
                        "Split output count mismatch (expected ",
                        split_sizes.size(),
                        ", got ",
                        outputs.size(),
                        ")");
        for (size_t i = 0; i < outputs.size(); ++i) {
            if (!outputs[i]) {
                continue;
            }
            ov::Shape out_shape = in_shape;
            out_shape[static_cast<size_t>(axis_norm)] = split_sizes[i];
            outputs[i]->shape = out_shape;
        }
        if (m_node && (m_last_input_shape != in_shape || !m_kernel)) {
            mlir::MLIRContext ctx;
            auto module = build_mlir_split_from_node(m_node, ctx, in_shape);
            if (module) {
                module->setAttr("gfx.prefer_parallel",
                                mlir::BoolAttr::get(module.getContext(), false));
            }
            std::string entry = find_entry_point(module);
            if (entry.empty()) {
                entry = "split_main";
            }
            const auto signature = infer_kernel_signature(module, entry);
            size_t func_inputs = signature.inputs;
            size_t func_results = signature.results;
            const auto func = resolve_entry_func(module, entry);
            const size_t scalar_inputs = count_scalar_inputs(func);
            size_t output_args = 0;
            if (signature.results == 0) {
                output_args = outputs.size();
            }
            size_t buffer_inputs = func_inputs;
            if (scalar_inputs <= buffer_inputs) {
                buffer_inputs -= scalar_inputs;
            }
            if (output_args <= buffer_inputs) {
                buffer_inputs -= output_args;
            }
            auto mapping = build_kernel_inputs(m_node, buffer_inputs, "Split");
            func_inputs = mapping.func_inputs;
            m_kernel_inputs = std::move(mapping.kernel_inputs);
            if (func_results == 0) {
                func_results = outputs.size();
            }
            const uint32_t arg_count =
                signature.total() ? signature.total() : static_cast<uint32_t>(func_inputs + func_results);
            KernelPlan plan(module, std::move(entry), arg_count);
            KernelSource src = plan.to_source();
            VulkanCodegenBackend backend;
            std::string log;
            try {
                m_kernel = backend.compile(src, &log);
            } catch (const std::exception& e) {
                OPENVINO_THROW("GFX Vulkan: failed to compile split stage ",
                               m_name,
                               " (",
                               m_type,
                               "): ",
                               e.what());
            }
            OPENVINO_ASSERT(m_kernel,
                            "GFX Vulkan: failed to compile split stage ",
                            m_name,
                            " (",
                            m_type,
                            "): ",
                            log);
            if (module) {
                if (auto attr = module->getAttrOfType<mlir::BoolAttr>("gfx.parallel_dispatch")) {
                    m_parallel_dispatch = attr.getValue();
                } else {
                    m_parallel_dispatch = false;
                }
                m_kernel_operand_kinds = extract_kernel_operand_kinds(module);
                m_kernel_operand_arg_indices = extract_kernel_operand_arg_indices(module);
                m_kernel_scalar_args = extract_kernel_scalar_values(module);
                if (m_kernel_operand_kinds.empty() && scalar_inputs != 0) {
                    OPENVINO_ASSERT(m_kernel_scalar_args.size() == scalar_inputs,
                                    "GFX Vulkan: kernel scalar args mismatch for ",
                                    m_name,
                                    " (expected ",
                                    scalar_inputs,
                                    ", got ",
                                    m_kernel_scalar_args.size(),
                                    ")");
                }
            }
            m_last_input_shape = in_shape;
        }
    } else {
        for (size_t i = 0; i < outputs.size(); ++i) {
            ensure_output_shape(i, outputs[i]);
        }
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
        GpuTensor* out = outputs[i];
        if (!out) {
            continue;
        }
        if (out->shape.empty()) {
            OPENVINO_THROW("GFX Vulkan: output shape is not known for stage ", m_name);
        }
        ov::element::Type out_type = out->expected_type;
        if (out_type == ov::element::dynamic && m_node) {
            out_type = m_node->get_output_element_type(i);
        }
        const auto elem_size = out_type == ov::element::dynamic ? out->buf.type.size() : out_type.size();
        const size_t out_bytes = ov::shape_size(out->shape) * elem_size;
        if (m_is_view_op) {
            out->expected_type = out_type;
            continue;
        }
        if (!out->buf.valid()) {
            OPENVINO_THROW("GFX Vulkan: output buffer is not allocated for stage ", m_name);
        }
        if (out->buf.size < out_bytes) {
            OPENVINO_THROW("GFX Vulkan: output buffer too small for stage ",
                           m_name,
                           " (need ",
                           out_bytes,
                           ", have ",
                           out->buf.size,
                           ")");
        }
        out->expected_type = out_type;
    }

    if (m_is_view_op) {
        if (m_inputs.empty() || !m_inputs[0] || !m_inputs[0]->buf.valid()) {
            OPENVINO_THROW("GFX Vulkan: missing input buffer for view op ", m_name);
        }
        auto* in = m_inputs[0];
        auto* out = outputs.front();
        const auto in_type = in->expected_type == ov::element::dynamic ? in->buf.type : in->expected_type;
        const auto out_type = out->expected_type == ov::element::dynamic ? in_type : out->expected_type;
        const size_t in_bytes = ov::shape_size(in->shape) * in_type.size();
        const size_t out_bytes = ov::shape_size(out->shape) * out_type.size();
        OPENVINO_ASSERT(in_bytes == out_bytes,
                        "GFX Vulkan: view op byte size mismatch for ",
                        m_name,
                        " (",
                        in_bytes,
                        " vs ",
                        out_bytes,
                        ")");
        out->buf = in->buf;
        out->buf.external = true;
        out->buf.owned = false;
        out->expected_type = out_type;
        return;
    }

    if (!m_kernel) {
        OPENVINO_THROW("GFX Vulkan: kernel was not compiled for stage ", m_name);
    }

    VulkanProfiler* vk_profiler = nullptr;
    if (m_profiling_enabled && m_profiler) {
        vk_profiler = static_cast<VulkanProfiler*>(m_profiler);
        vk_profiler->begin_node(m_profile_node_id,
                                m_profile_node_name.c_str(),
                                m_profile_node_type.c_str(),
                                "GFX");
    }
    auto make_hooks = [&](KernelExecutionHooks& hooks) -> KernelExecutionHooks* {
        if (!vk_profiler) {
            return nullptr;
        }
        const auto sample = vk_profiler->reserve_samples();
        hooks.on_begin = [vk_profiler, sample](GpuCommandEncoderHandle enc) {
            vk_profiler->write_timestamp(reinterpret_cast<VkCommandBuffer>(enc), sample.begin);
        };
        hooks.on_end = [vk_profiler, sample](GpuCommandEncoderHandle enc) {
            vk_profiler->write_timestamp(reinterpret_cast<VkCommandBuffer>(enc), sample.end);
        };
        hooks.on_complete = [vk_profiler, sample, node_id = m_profile_node_id]() {
            vk_profiler->end_node(node_id, sample);
        };
        return &hooks;
    };

    std::vector<KernelArg> args;
    if (gfx_log_debug_enabled()) {
        GFX_LOG_DEBUG("VulkanExec", "Kernel args prep: scalars=" << m_kernel_scalar_args.size()
                                                                 << " inputs=" << m_kernel_inputs.size()
                                                                 << " outputs=" << outputs.size()
                                                                 << " kinds=" << m_kernel_operand_kinds.size());
    }
    uint32_t arg_index = 0;
    std::vector<int32_t> scalar_storage;
    if (!m_kernel_operand_kinds.empty()) {
        args.reserve(m_kernel_operand_kinds.size());
        std::ostringstream arg_map;
        if (gfx_log_debug_enabled()) {
            arg_map << "Kernel arg map: ";
        }
        size_t scalar_count = 0;
        for (auto kind : m_kernel_operand_kinds) {
            if (kind == 0) {
                ++scalar_count;
            }
        }
        scalar_storage.reserve(scalar_count);
        size_t scalar_idx = 0;
        size_t input_pos = 0;
        size_t output_pos = 0;
        const bool has_arg_indices =
            m_kernel_operand_arg_indices.size() == m_kernel_operand_kinds.size();
        const size_t input_arg_count = m_kernel_inputs.size();
        for (size_t op_idx = 0; op_idx < m_kernel_operand_kinds.size(); ++op_idx) {
            const auto kind = m_kernel_operand_kinds[op_idx];
            if (kind == 0) {
                int32_t value = 0;
                if (scalar_idx < m_kernel_scalar_args.size()) {
                    value = m_kernel_scalar_args[scalar_idx++];
                }
                scalar_storage.push_back(value);
                args.push_back(make_bytes_arg(arg_index++, &scalar_storage.back(), sizeof(value)));
                if (gfx_log_debug_enabled()) {
                    if (op_idx) {
                        arg_map << ", ";
                    }
                    arg_map << "arg" << op_idx << "=scalar(" << value << ")";
                }
                continue;
            }
            int32_t arg_idx = -1;
            if (has_arg_indices) {
                arg_idx = m_kernel_operand_arg_indices[op_idx];
            }
            if (arg_idx >= 0) {
                const size_t uarg = static_cast<size_t>(arg_idx);
                if (uarg < input_arg_count) {
                    const size_t input_idx = m_kernel_inputs[uarg];
                    GpuTensor* t = resolve_input_tensor(input_idx);
                    OPENVINO_ASSERT(t && t->buf.valid(),
                                    "GFX Vulkan: missing input buffer for stage ",
                                    m_name);
                    args.push_back(make_buffer_arg(arg_index++, t->buf));
                    if (gfx_log_debug_enabled()) {
                        if (op_idx) {
                            arg_map << ", ";
                        }
                        arg_map << "arg" << op_idx << "=input[" << input_idx << "]";
                    }
                    continue;
                }
                const size_t out_idx = uarg - input_arg_count;
                if (out_idx < outputs.size()) {
                    auto* out = outputs[out_idx];
                    OPENVINO_ASSERT(out && out->buf.valid(),
                                    "GFX Vulkan: missing output buffer for stage ",
                                    m_name);
                    args.push_back(make_buffer_arg(arg_index++, out->buf));
                    if (gfx_log_debug_enabled()) {
                        if (op_idx) {
                            arg_map << ", ";
                        }
                        arg_map << "arg" << op_idx << "=output[" << out_idx << "]";
                    }
                    continue;
                }
            }
            if (input_pos < m_kernel_inputs.size()) {
                const size_t input_idx = m_kernel_inputs[input_pos++];
                GpuTensor* t = resolve_input_tensor(input_idx);
                OPENVINO_ASSERT(t && t->buf.valid(),
                                "GFX Vulkan: missing input buffer for stage ",
                                m_name);
                args.push_back(make_buffer_arg(arg_index++, t->buf));
                if (gfx_log_debug_enabled()) {
                    if (op_idx) {
                        arg_map << ", ";
                    }
                    arg_map << "arg" << op_idx << "=input[" << input_idx << "]";
                }
                continue;
            }
            OPENVINO_ASSERT(output_pos < outputs.size(),
                            "GFX Vulkan: missing output buffer for stage ",
                            m_name);
            auto* out = outputs[output_pos++];
            OPENVINO_ASSERT(out && out->buf.valid(),
                            "GFX Vulkan: missing output buffer for stage ",
                            m_name);
            args.push_back(make_buffer_arg(arg_index++, out->buf));
            if (gfx_log_debug_enabled()) {
                if (op_idx) {
                    arg_map << ", ";
                }
                arg_map << "arg" << op_idx << "=output[" << (output_pos - 1) << "]";
            }
        }
        if (gfx_log_debug_enabled()) {
            GFX_LOG_DEBUG("VulkanExec", arg_map.str());
        }
    } else {
        args.reserve(m_kernel_scalar_args.size() + m_kernel_inputs.size() + outputs.size());
        scalar_storage = m_kernel_scalar_args;
        for (auto& v : scalar_storage) {
            args.push_back(make_bytes_arg(arg_index++, &v, sizeof(v)));
        }
        for (size_t ai = 0; ai < m_kernel_inputs.size(); ++ai) {
            const size_t input_idx = m_kernel_inputs[ai];
            GpuTensor* t = resolve_input_tensor(input_idx);
            OPENVINO_ASSERT(t && t->buf.valid(),
                            "GFX Vulkan: missing input buffer for stage ",
                            m_name);
            args.push_back(make_buffer_arg(arg_index++, t->buf));
        }
        for (auto* out : outputs) {
            OPENVINO_ASSERT(out && out->buf.valid(),
                            "GFX Vulkan: missing output buffer for stage ",
                            m_name);
            args.push_back(make_buffer_arg(arg_index++, out->buf));
        }
    }
    auto bound_args = materialize_kernel_bytes_args(args, *m_buffer_manager, m_name.c_str());

    KernelDispatch dispatch{};
    if (m_parallel_dispatch) {
        const ov::Shape& shape = outputs.front() && !outputs.front()->shape.empty()
                                     ? outputs.front()->shape
                                     : m_output_shape;
        const size_t rank = shape.size();
        if (rank == 1) {
            dispatch.grid[0] = shape[0];
        } else if (rank == 2) {
            dispatch.grid[0] = shape[0];
            dispatch.grid[1] = shape[1];
        } else if (rank >= 3) {
            dispatch.grid[0] = shape[rank - 3];
            dispatch.grid[1] = shape[rank - 2];
            dispatch.grid[2] = shape[rank - 1];
        }
        dispatch.threads_per_group[0] = m_kernel ? m_kernel->clamp_threadgroup_size(1) : 1;
        dispatch.threads_per_group[1] = 1;
        dispatch.threads_per_group[2] = 1;
    } else {
        dispatch.grid[0] = 1;
        dispatch.grid[1] = 1;
        dispatch.grid[2] = 1;
        dispatch.threads_per_group[0] = 1;
        dispatch.threads_per_group[1] = 1;
        dispatch.threads_per_group[2] = 1;
    }

    KernelExecutionHooks hooks;
    m_kernel->execute(nullptr, dispatch, bound_args, make_hooks(hooks));
}

void VulkanStage::set_inputs(const std::vector<GpuTensor*>& inputs) {
    m_inputs = inputs;
    if (!m_const_buffers) {
        m_const_buffers = std::make_shared<ConstBufferSet>();
        m_const_buffers->buffers.resize(inputs.size());
        m_const_buffers->present.assign(inputs.size(), false);
    }
}

void VulkanStage::set_output(GpuTensor* output) {
    m_output = output;
    m_outputs.clear();
    if (output) {
        m_outputs.push_back(output);
    }
}

void VulkanStage::set_outputs(const std::vector<std::unique_ptr<GpuTensor>>& outputs) {
    m_outputs.clear();
    m_outputs.reserve(outputs.size());
    for (const auto& o : outputs) {
        m_outputs.push_back(o.get());
    }
    if (!m_outputs.empty()) {
        m_output = m_outputs.front();
    }
}

void VulkanStage::enable_profiling(bool enable) {
    m_profiling_enabled = enable;
}

void VulkanStage::set_profiler(void* profiler,
                               uint32_t node_id,
                               const std::string& node_name,
                               const std::string& node_type) {
    m_profiler = profiler;
    m_profile_node_id = node_id;
    m_profile_node_name = node_name;
    m_profile_node_type = node_type;
}

std::unique_ptr<GpuStage> VulkanStage::clone() const {
    auto stage = std::make_unique<VulkanStage>(m_node);
    stage->m_kernel = m_kernel;
    stage->m_output_shape = m_output_shape;
    stage->m_last_input_shape = m_last_input_shape;
    stage->m_kernel_inputs = m_kernel_inputs;
    stage->m_const_buffers = m_const_buffers;
    stage->m_parallel_dispatch = m_parallel_dispatch;
    stage->m_kernel_scalar_args = m_kernel_scalar_args;
    stage->m_kernel_operand_kinds = m_kernel_operand_kinds;
    stage->m_kernel_operand_arg_indices = m_kernel_operand_arg_indices;
    return stage;
}

}  // namespace gfx_plugin
}  // namespace ov
