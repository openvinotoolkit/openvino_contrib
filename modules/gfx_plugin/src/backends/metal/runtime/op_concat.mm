// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/op_concat.hpp"

#include <string>

#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "backends/metal/codegen/metal_codegen_backend.hpp"
#include "mlir/gfx_mlir_kernel_builder.hpp"
#include "mlir/codegen_common.hpp"
#include "runtime/gfx_logger.hpp"
#include "backends/metal/runtime/op_utils.hpp"
#include "kernel_ir/gfx_kernel_args.hpp"
#include "runtime/gfx_shape_utils.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

namespace ov {
namespace gfx_plugin {

namespace {
size_t element_size(const ov::element::Type& t) {
    return t.size();
}

struct ConcatParams {
    uint32_t outer = 0;
    uint32_t inner = 0;
    uint32_t axis_offset = 0;
    uint32_t axis_len = 0;
    uint32_t axis_total = 0;
};
}  // namespace

MetalConcatOp::MetalConcatOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "Concat",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_node(node),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    compute_layout(node);
}

void MetalConcatOp::compute_layout(const std::shared_ptr<const ov::Node>& node) {
    auto concat = ov::as_type_ptr<const ov::op::v0::Concat>(node);
    OPENVINO_ASSERT(concat, "MetalConcatOp expects v0::Concat");
    m_axis = concat->get_axis();
    m_element_type = concat->get_output_element_type(0);

    const auto& out_shape = concat->get_output_shape(0);
    OPENVINO_ASSERT(!out_shape.empty(), "Concat: static output shape required");
    const int64_t axis_norm = normalize_axis(m_axis, out_shape.size(), "Concat");

    m_axis_sizes.clear();
    m_axis_offsets.clear();
    m_axis_sizes.reserve(concat->get_input_size());
    m_axis_offsets.reserve(concat->get_input_size());

    m_outer = 1;
    for (size_t i = 0; i < static_cast<size_t>(axis_norm); ++i)
        m_outer *= out_shape[i];

    m_inner = 1;
    for (size_t i = static_cast<size_t>(axis_norm) + 1; i < out_shape.size(); ++i)
        m_inner *= out_shape[i];

    uint64_t offset = 0;
    for (size_t i = 0; i < concat->get_input_size(); ++i) {
        const auto& in_shape = concat->get_input_shape(i);
        OPENVINO_ASSERT(in_shape.size() == out_shape.size(),
                        "Concat: mismatched input rank");
        uint64_t axis_len = static_cast<uint64_t>(in_shape[static_cast<size_t>(axis_norm)]);
        m_axis_sizes.push_back(axis_len);
        m_axis_offsets.push_back(offset);
        offset += axis_len;
    }
}

void MetalConcatOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalConcatOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }
    MetalCodegenBackend backend(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    ConcatCodegenDesc desc{};
    desc.element_type = m_element_type == ov::element::dynamic ? ov::element::f32 : m_element_type;
    auto& ctx = gfx_mlir_context();
    auto module = build_mlir_for_node(m_node, ctx);
    auto msl_desc = desc;
    auto msl_generator = [msl_desc](mlir::ModuleOp mod) { return generate_msl_from_mlir(mod, msl_desc); };

    KernelSpec spec(m_node, 3u);
    m_kernel = compile_msl_kernel(backend, spec, module, "concat_kernel", msl_generator, &log);
    OPENVINO_ASSERT(m_kernel, "MetalConcatOp: failed to compile kernel: ", log);

    if (m_node) {
        m_const_inputs.resize(m_node->get_input_size());
        for (size_t i = 0; i < m_node->get_input_size(); ++i) {
            auto c = ov::as_type_ptr<const ov::op::v0::Constant>(m_node->get_input_node_shared_ptr(i));
            if (!c)
                continue;
            const size_t bytes = c->get_byte_size();
            if (bytes == 0)
                continue;
            MetalTensor t{};
            t.shape = c->get_shape();
            t.expected_type = c->get_element_type();
            const std::string key = m_node->get_friendly_name() + "/const_" + std::to_string(i);
            t.buf = buffer_manager->wrap_const(key, c->get_data_ptr(), bytes, t.expected_type);
            m_const_inputs[i] = std::move(t);
        }
    }

    MetalOp::compile(buffer_manager);
}

void MetalConcatOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(!inputs().empty(), "Concat: no inputs bound");
    MetalTensor& out = require_output();
    ov::Shape out_shape = !out.shape.empty() ? out.shape : output_shape();
    if (out_shape.empty() && m_node->get_output_partial_shape(0).is_static()) {
        out_shape = m_node->get_output_shape(0);
    }
    OPENVINO_ASSERT(!out_shape.empty(), "Concat: output shape unknown");
    out.expected_type = m_element_type;
    const size_t out_bytes = ov::shape_size(out_shape) * element_size(m_element_type);
    if (!out.buf.valid() || out.buf.size < out_bytes) {
        out.buf = allocate_temp_buffer(out_bytes, m_element_type, /*persistent=*/false, out.prefer_private);
    }
    out.shape = out_shape;

    const size_t elem_sz = element_size(m_element_type);
    const size_t rank = out_shape.size();
    const int64_t axis_norm = normalize_axis(m_axis, rank, "Concat");

    uint64_t outer = 1;
    for (size_t i = 0; i < static_cast<size_t>(axis_norm); ++i) {
        outer *= out_shape[i];
    }
    uint64_t inner = 1;
    for (size_t i = static_cast<size_t>(axis_norm) + 1; i < out_shape.size(); ++i) {
        inner *= out_shape[i];
    }
    OPENVINO_ASSERT(outer == m_outer && inner == m_inner,
                    "Concat: runtime shape mismatch (outer/inner)");
    OPENVINO_ASSERT(static_cast<uint64_t>(out_shape[static_cast<size_t>(axis_norm)]) ==
                        std::accumulate(m_axis_sizes.begin(), m_axis_sizes.end(), uint64_t{0}),
                    "Concat: runtime axis total mismatch");

    auto cap = [&](const MetalTensor* t) -> size_t {
        return t && t->buf.valid() ? t->buf.size : 0;
    };
    const size_t dst_cap = out.buf.size;
    const uint32_t axis_total = static_cast<uint32_t>(out_shape[static_cast<size_t>(axis_norm)]);
    const uint32_t outer_u32 = static_cast<uint32_t>(m_outer);
    const uint32_t inner_u32 = static_cast<uint32_t>(m_inner);

    for (size_t i = 0; i < inputs().size() && i < m_axis_sizes.size(); ++i) {
        MetalTensor* src = inputs()[i];
        if ((!src || !src->buf.valid()) && i < m_const_inputs.size() && m_const_inputs[i].buf.valid()) {
            src = &m_const_inputs[i];
        }
        OPENVINO_ASSERT(src && src->buf.valid(), "Concat: input ", i, " is null");
        ov::Shape src_shape = !src->shape.empty() ? src->shape : ov::Shape{};
        if (src_shape.empty() && m_node->get_input_partial_shape(i).is_static()) {
            src_shape = m_node->get_input_shape(i);
        }
        OPENVINO_ASSERT(!src_shape.empty(), "Concat: input ", i, " shape unknown");
        OPENVINO_ASSERT(src_shape.size() == out_shape.size(),
                        "Concat: runtime rank mismatch at input ", i);
        OPENVINO_ASSERT(static_cast<int64_t>(src_shape[static_cast<size_t>(axis_norm)]) ==
                            static_cast<int64_t>(m_axis_sizes[i]),
                        "Concat: runtime axis dim mismatch at input ", i);
        ConcatParams params{};
        params.outer = outer_u32;
        params.inner = inner_u32;
        params.axis_total = axis_total;
        params.axis_len = static_cast<uint32_t>(m_axis_sizes[i]);
        params.axis_offset = static_cast<uint32_t>(m_axis_offsets[i]);
        const uint64_t total =
            static_cast<uint64_t>(params.outer) * params.axis_len * params.inner;
        if (total == 0)
            continue;

        const size_t src_bytes = static_cast<size_t>(outer) * params.axis_len * static_cast<size_t>(inner) * elem_sz;
        const size_t dst_bytes = static_cast<size_t>(outer) * axis_total * static_cast<size_t>(inner) * elem_sz;
        OPENVINO_ASSERT(src_bytes <= cap(src),
                        "Concat: source buffer too small for input ", i,
                        " need=", src_bytes, " have=", cap(src));
        OPENVINO_ASSERT(dst_bytes <= dst_cap,
                        "Concat: destination buffer too small for input ", i,
                        " need=", dst_bytes, " have=", dst_cap);

        KernelDispatch dispatch = make_1d_dispatch(static_cast<size_t>(total), m_kernel->clamp_threadgroup_size(256));

        KernelArgsBuilder args_builder(name().c_str());
        append_kernel_input_args(args_builder, 1, [&](size_t) { return src; }, name().c_str());
        args_builder.add_output(&out);
        args_builder.add_bytes(&params, sizeof(params));

        const auto args = args_builder.finalize(buffer_manager(), m_kernel.get());
        execute_kernel(*m_kernel, cmd_buf_handle, dispatch, args);
    }
}

}  // namespace gfx_plugin
}  // namespace ov
