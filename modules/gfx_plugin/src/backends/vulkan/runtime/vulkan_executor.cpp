// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/vulkan/runtime/vulkan_executor.hpp"

#include <algorithm>
#include <cstring>
#include <sstream>

#include "mlir_builder.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "backends/vulkan/codegen/vulkan_compiler.hpp"
#include "backends/vulkan/runtime/vulkan_memory.hpp"
#include "runtime/memory_manager.hpp"
#include "runtime/gfx_kernel_dispatch.hpp"
#include "mlir/gfx_kernel_plan.hpp"
#include "mlir/gfx_kernel_spec.hpp"
#include "runtime/gfx_logger.hpp"
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

constexpr uint64_t kSoftmaxTileWork = 1024;

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

VulkanStage::~VulkanStage() {
    if (m_softmax_params.valid()) {
        vulkan_free_buffer(m_softmax_params);
    }
}

VulkanStage::ConstBufferSet::~ConstBufferSet() {
    for (auto& tensor : buffers) {
        if (tensor.buf.valid()) {
            vulkan_free_buffer(tensor.buf);
        }
    }
}

void VulkanStage::init(GpuBufferManager* /*buffer_manager*/) {
    // No-op: Vulkan backend does not allocate buffers at this stage yet.
}

void VulkanStage::compile(GpuBufferManager* /*buffer_manager*/) {
    mlir::MLIRContext ctx;
    if (m_is_view_op) {
        return;
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
        for (size_t i = 0; i < in_count; ++i) {
            auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(m_node->get_input_node_shared_ptr(i));
            if (!c) {
                continue;
            }
            if (m_const_buffers->present[i] && m_const_buffers->buffers[i].buf.valid()) {
                continue;
            }
            const size_t bytes = c->get_byte_size();
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
            GpuBuffer buf = vulkan_upload_device_buffer(c->get_data_ptr(),
                                                        bytes,
                                                        c->get_element_type(),
                                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            m_const_buffers->buffers[i].buf = buf;
            m_const_buffers->buffers[i].shape = c->get_shape();
            m_const_buffers->buffers[i].expected_type = c->get_element_type();
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
    std::string entry = plan.entry_point();
    if (module && entry.empty()) {
        entry = find_entry_point(module);
    }
    if (entry.empty()) {
        entry = "gfx_kernel";
    }
    size_t func_inputs = 0;
    size_t func_results = 0;
    if (module) {
        mlir::func::FuncOp func;
        if (!entry.empty()) {
            func = module.lookupSymbol<mlir::func::FuncOp>(entry);
        }
        if (!func) {
            module.walk([&](mlir::func::FuncOp f) {
                if (!func) {
                    func = f;
                }
            });
        }
        if (func) {
            auto ftype = func.getFunctionType();
            func_inputs = static_cast<size_t>(ftype.getNumInputs());
            func_results = static_cast<size_t>(ftype.getNumResults());
        }
    }
    m_kernel_inputs.clear();
    if (m_node) {
        const size_t node_inputs = m_node->get_input_size();
        if (func_inputs == 0) {
            func_inputs = node_inputs;
        }
        size_t nonconst_count = 0;
        for (size_t i = 0; i < node_inputs; ++i) {
            auto src = m_node->get_input_node_shared_ptr(i);
            if (!ov::as_type_ptr<const ov::op::v0::Constant>(src)) {
                ++nonconst_count;
            }
        }
        OPENVINO_ASSERT(func_inputs >= nonconst_count,
                        "GFX Vulkan: MLIR expects fewer inputs than non-constant inputs for ",
                        m_name);
        const size_t need_consts = func_inputs - nonconst_count;
        size_t const_added = 0;
        m_kernel_inputs.reserve(func_inputs);
        for (size_t i = 0; i < node_inputs; ++i) {
            auto src = m_node->get_input_node_shared_ptr(i);
            const bool is_const = ov::as_type_ptr<const ov::op::v0::Constant>(src) != nullptr;
            if (is_const) {
                if (const_added < need_consts) {
                    m_kernel_inputs.push_back(i);
                    ++const_added;
                }
            } else {
                m_kernel_inputs.push_back(i);
            }
        }
        OPENVINO_ASSERT(m_kernel_inputs.size() == func_inputs,
                        "GFX Vulkan: MLIR input count mismatch for ",
                        m_name,
                        " (expected ",
                        func_inputs,
                        ", got ",
                        m_kernel_inputs.size(),
                        ")");
        if (func_results == 0) {
            func_results = m_node->get_output_size();
        }
    }
    const uint32_t arg_count = static_cast<uint32_t>(func_inputs + func_results);
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
}

void VulkanStage::execute(GpuCommandBufferHandle /*command_buffer*/) {
    std::vector<GpuTensor*> outputs = m_outputs;
    if (outputs.empty() && m_output) {
        outputs.push_back(m_output);
    }
    if (outputs.empty()) {
        OPENVINO_THROW("GFX Vulkan: output tensor is not bound for stage ", m_name);
    }
    uint64_t softmax_total_work = 0;

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
        if (axis < 0) {
            axis += static_cast<int64_t>(in_shape.size());
        }
        OPENVINO_ASSERT(axis >= 0 && axis < static_cast<int64_t>(in_shape.size()),
                        "Softmax axis out of range");
        uint64_t rows = 1;
        uint64_t inner = 1;
        for (int64_t i = 0; i < axis; ++i) {
            rows *= in_shape[static_cast<size_t>(i)];
        }
        for (size_t i = static_cast<size_t>(axis) + 1; i < in_shape.size(); ++i) {
            inner *= in_shape[i];
        }
        const uint64_t total_work = rows * inner;
        const bool want_tiled = (axis == 0) || (total_work > kSoftmaxTileWork);
        softmax_total_work = total_work;
        if (gfx_log_debug_enabled()) {
            GFX_LOG_DEBUG("VulkanSoftmax",
                          "shape_rank=" << in_shape.size()
                                        << " axis=" << axis
                                        << " rows*inner=" << total_work
                                        << " tiled=" << (want_tiled ? "1" : "0"));
        }
        for (auto* out : outputs) {
            if (out) {
                out->shape = in_shape;
            }
        }
        if (m_node && (m_last_input_shape != in_shape || !m_kernel || want_tiled != m_softmax_tiled)) {
            mlir::MLIRContext ctx;
            const bool log_softmax = ov::as_type_ptr<const ov::op::v5::LogSoftmax>(m_node) != nullptr;
            auto module = log_softmax
                              ? (want_tiled ? build_mlir_logsoftmax_tiled_from_node(m_node, ctx, in_shape)
                                            : build_mlir_logsoftmax_from_node(m_node, ctx, in_shape))
                              : (want_tiled ? build_mlir_softmax_tiled_from_node(m_node, ctx, in_shape)
                                            : build_mlir_softmax_from_node(m_node, ctx, in_shape));
            std::string entry = find_entry_point(module);
            if (entry.empty()) {
                entry = "softmax_main";
            }
            const uint32_t arg_count =
                static_cast<uint32_t>(m_kernel_inputs.size() + outputs.size() + (want_tiled ? 1u : 0u));
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
            m_last_input_shape = in_shape;
            m_softmax_tiled = want_tiled;
        }
    } else if (m_type == "Split" || m_type == "VariadicSplit") {
        ov::Shape in_shape = resolve_input_shape(0);
        if (in_shape.empty()) {
            OPENVINO_THROW("GFX Vulkan: Split input shape is unknown for stage ", m_name);
        }
        int64_t axis = 0;
        std::vector<size_t> split_sizes;
        if (auto s = ov::as_type_ptr<const ov::op::v1::Split>(m_node)) {
            auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(s->input_value(1).get_node_shared_ptr());
            OPENVINO_ASSERT(axis_const, "Split axis must be constant");
            axis = axis_const->cast_vector<int64_t>().at(0);
            size_t parts = s->get_num_splits();
            int64_t axis_norm = axis < 0 ? axis + static_cast<int64_t>(in_shape.size()) : axis;
            OPENVINO_ASSERT(axis_norm >= 0 && axis_norm < static_cast<int64_t>(in_shape.size()),
                            "Split axis out of range");
            const size_t axis_len = in_shape[static_cast<size_t>(axis_norm)];
            OPENVINO_ASSERT(parts > 0, "Split number of splits is zero");
            OPENVINO_ASSERT(axis_len % parts == 0, "Split dimension not divisible by parts");
            split_sizes.assign(parts, axis_len / parts);
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
        int64_t axis_norm = axis < 0 ? axis + static_cast<int64_t>(in_shape.size()) : axis;
        OPENVINO_ASSERT(axis_norm >= 0 && axis_norm < static_cast<int64_t>(in_shape.size()),
                        "Split axis out of range");
        const size_t axis_len = in_shape[static_cast<size_t>(axis_norm)];
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
            std::string entry = find_entry_point(module);
            if (entry.empty()) {
                entry = "split_main";
            }
            if (m_node) {
                size_t func_inputs = 0;
                if (auto func = module.lookupSymbol<mlir::func::FuncOp>(entry)) {
                    func_inputs = static_cast<size_t>(func.getFunctionType().getNumInputs());
                }
                if (func_inputs == 0) {
                    func_inputs = m_node->get_input_size();
                }
                const size_t node_inputs = m_node->get_input_size();
                size_t nonconst_count = 0;
                for (size_t i = 0; i < node_inputs; ++i) {
                    auto src = m_node->get_input_node_shared_ptr(i);
                    if (!ov::as_type_ptr<const ov::op::v0::Constant>(src)) {
                        ++nonconst_count;
                    }
                }
                OPENVINO_ASSERT(func_inputs >= nonconst_count,
                                "GFX Vulkan: Split MLIR expects fewer inputs than non-constant inputs");
                const size_t need_consts = func_inputs - nonconst_count;
                size_t const_added = 0;
                m_kernel_inputs.clear();
                m_kernel_inputs.reserve(func_inputs);
                for (size_t i = 0; i < node_inputs; ++i) {
                    auto src = m_node->get_input_node_shared_ptr(i);
                    const bool is_const = ov::as_type_ptr<const ov::op::v0::Constant>(src) != nullptr;
                    if (is_const) {
                        if (const_added < need_consts) {
                            m_kernel_inputs.push_back(i);
                            ++const_added;
                        }
                    } else {
                        m_kernel_inputs.push_back(i);
                    }
                }
                OPENVINO_ASSERT(m_kernel_inputs.size() == func_inputs,
                                "GFX Vulkan: Split MLIR input count mismatch");
            }
            const uint32_t arg_count =
                static_cast<uint32_t>(m_kernel_inputs.size() + outputs.size());
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
        if (!out->buf.valid() || out->buf.size < out_bytes) {
            const bool host_visible = !out->prefer_private;
            const VkMemoryPropertyFlags props =
                host_visible ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                             : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            out->buf = vulkan_allocate_buffer(out_bytes,
                                              out_type == ov::element::dynamic ? out->buf.type : out_type,
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                              props);
            out->buf.host_visible = host_visible;
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
        out->expected_type = out_type;
        return;
    }

    if (!m_kernel) {
        OPENVINO_THROW("GFX Vulkan: kernel was not compiled for stage ", m_name);
    }

    VulkanProfiler::SamplePair sample{};
    KernelExecutionHooks hooks;
    KernelExecutionHooks* hooks_ptr = nullptr;
    if (m_profiling_enabled && m_profiler) {
        auto* profiler = static_cast<VulkanProfiler*>(m_profiler);
        profiler->begin_node(m_profile_node_id,
                             m_profile_node_name.c_str(),
                             m_profile_node_type.c_str(),
                             "GFX");
        sample = profiler->reserve_samples();
        hooks.on_begin = [profiler, sample](GpuCommandEncoderHandle enc) {
            profiler->write_timestamp(reinterpret_cast<VkCommandBuffer>(enc), sample.begin);
        };
        hooks.on_end = [profiler, sample](GpuCommandEncoderHandle enc) {
            profiler->write_timestamp(reinterpret_cast<VkCommandBuffer>(enc), sample.end);
        };
        hooks.on_complete = [profiler, sample, node_id = m_profile_node_id]() {
            profiler->end_node(node_id, sample);
        };
        hooks_ptr = &hooks;
    }

    if ((m_type == "Softmax" || m_type == "LogSoftmax") &&
        m_softmax_tiled && softmax_total_work > 0) {
        const size_t param_bytes = sizeof(uint32_t) * 2;
        if (!m_softmax_params.valid() || m_softmax_params.size < param_bytes) {
            m_softmax_params = vulkan_allocate_buffer(param_bytes,
                                                      ov::element::i32,
                                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            m_softmax_params.host_visible = true;
        }

        std::vector<KernelArg> args;
        args.reserve(m_kernel_inputs.size() + outputs.size() + 1);
        for (size_t ai = 0; ai < m_kernel_inputs.size(); ++ai) {
            const size_t input_idx = m_kernel_inputs[ai];
            GpuTensor* t = resolve_input_tensor(input_idx);
            OPENVINO_ASSERT(t && t->buf.valid(), "GFX Vulkan: missing input buffer for stage ", m_name);
            args.push_back(make_buffer_arg(static_cast<uint32_t>(ai), t->buf));
        }
        const uint32_t param_index = static_cast<uint32_t>(m_kernel_inputs.size());
        args.push_back(make_buffer_arg(param_index, m_softmax_params));
        for (size_t oi = 0; oi < outputs.size(); ++oi) {
            auto* out = outputs[oi];
            OPENVINO_ASSERT(out && out->buf.valid(), "GFX Vulkan: missing output buffer for stage ", m_name);
            args.push_back(make_buffer_arg(static_cast<uint32_t>(m_kernel_inputs.size() + 1 + oi), out->buf));
        }

        uint64_t offset = 0;
        while (offset < softmax_total_work) {
            const uint64_t count = std::min<uint64_t>(kSoftmaxTileWork, softmax_total_work - offset);
            if (m_softmax_params.host_visible) {
                auto* mapped = static_cast<uint32_t*>(gpu_map_buffer(m_softmax_params));
                OPENVINO_ASSERT(mapped, "GFX Vulkan: failed to map softmax params buffer");
                mapped[0] = static_cast<uint32_t>(offset);
                mapped[1] = static_cast<uint32_t>(count);
                gpu_unmap_buffer(m_softmax_params);
            }
            KernelDispatch dispatch = make_1d_dispatch(static_cast<size_t>(count), 1);
            m_kernel->execute(nullptr, dispatch, args, nullptr);
            offset += count;
        }
        return;
    }

    std::vector<KernelArg> args;
    args.reserve(m_kernel_inputs.size() + outputs.size());
    for (size_t ai = 0; ai < m_kernel_inputs.size(); ++ai) {
        const size_t input_idx = m_kernel_inputs[ai];
        GpuTensor* t = resolve_input_tensor(input_idx);
        OPENVINO_ASSERT(t && t->buf.valid(), "GFX Vulkan: missing input buffer for stage ", m_name);
        args.push_back(make_buffer_arg(static_cast<uint32_t>(ai), t->buf));
    }
    for (size_t oi = 0; oi < outputs.size(); ++oi) {
        auto* out = outputs[oi];
        OPENVINO_ASSERT(out && out->buf.valid(), "GFX Vulkan: missing output buffer for stage ", m_name);
        args.push_back(make_buffer_arg(static_cast<uint32_t>(m_kernel_inputs.size() + oi), out->buf));
    }

    KernelDispatch dispatch{};
    if (m_type == "Softmax" || m_type == "LogSoftmax" ||
        m_type == "Split" || m_type == "VariadicSplit") {
        dispatch.grid[0] = 1;
        dispatch.grid[1] = 1;
        dispatch.grid[2] = 1;
        dispatch.threads_per_group[0] = 1;
        dispatch.threads_per_group[1] = 1;
        dispatch.threads_per_group[2] = 1;
    } else {
        dispatch = KernelPlan::make_default_dispatch(outputs.front()->shape, *m_kernel);
    }

    m_kernel->execute(nullptr, dispatch, args, hooks_ptr);
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
    stage->m_softmax_tiled = m_softmax_tiled;
    stage->m_const_buffers = m_const_buffers;
    return stage;
}

}  // namespace gfx_plugin
}  // namespace ov
