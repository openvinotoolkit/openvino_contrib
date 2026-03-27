// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/op_split.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/core/validation_util.hpp"
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
std::vector<size_t> extract_split_sizes(const std::shared_ptr<const ov::Node>& node,
                                        int64_t& axis_out,
                                        ov::Shape& input_shape) {
    if (auto s = ov::as_type_ptr<const ov::op::v1::Split>(node)) {
        auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(s->input_value(1).get_node_shared_ptr());
        OPENVINO_ASSERT(axis_const, "Split axis must be constant");
        axis_out = axis_const->cast_vector<int64_t>().at(0);
        input_shape = s->get_input_shape(0);
        const size_t parts = s->get_num_splits();
        const size_t axis_norm =
            static_cast<size_t>(normalize_axis(axis_out, input_shape.size(), "Split"));
        OPENVINO_ASSERT(input_shape.at(axis_norm) % parts == 0, "Split dimension not divisible by parts");
        size_t chunk = input_shape.at(axis_norm) / parts;
        return std::vector<size_t>(parts, chunk);
    } else if (auto vs = ov::as_type_ptr<const ov::op::v1::VariadicSplit>(node)) {
        auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(vs->input_value(1).get_node_shared_ptr());
        OPENVINO_ASSERT(axis_const, "VariadicSplit axis must be constant");
        axis_out = axis_const->cast_vector<int64_t>().at(0);
        input_shape = vs->get_input_shape(0);
        auto lengths_const = ov::as_type_ptr<const ov::op::v0::Constant>(vs->input_value(2).get_node_shared_ptr());
        OPENVINO_ASSERT(lengths_const, "VariadicSplit lengths must be constant");
        auto lengths = lengths_const->cast_vector<int64_t>();
        std::vector<size_t> res;
        res.reserve(lengths.size());
        for (auto v : lengths) {
            OPENVINO_ASSERT(v >= 0, "VariadicSplit negative length not supported");
            res.push_back(static_cast<size_t>(v));
        }
        return res;
    }
    OPENVINO_THROW("MetalSplitOp: unsupported node type");
}

}  // namespace

MetalSplitOp::MetalSplitOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "Split",
              {},  // output shapes are per-output; not used in base
              device,
              queue),
      m_node(node),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    parse_split(node);
}

void MetalSplitOp::parse_split(const std::shared_ptr<const ov::Node>& node) {
    m_split_sizes = extract_split_sizes(node, m_axis, m_input_shape);
    m_element_type = node->get_input_element_type(0);
    if (auto s = ov::as_type_ptr<const ov::op::v1::Split>(node)) {
        m_is_variadic = false;
        m_num_splits = s->get_num_splits();
    } else if (ov::as_type_ptr<const ov::op::v1::VariadicSplit>(node)) {
        m_is_variadic = true;
        m_num_splits = m_split_sizes.size();
    }
}

void MetalSplitOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalSplitOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }
    MetalCodegenBackend backend(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    SplitCodegenDesc desc{};
    desc.element_type = m_element_type == ov::element::dynamic ? ov::element::f32 : m_element_type;
    auto& ctx = gfx_mlir_context();
    auto module = build_mlir_for_node(m_node, ctx);
    auto msl_desc = desc;
    auto msl_generator = [msl_desc](mlir::ModuleOp mod) { return generate_msl_from_mlir(mod, msl_desc); };

    KernelSpec spec(m_node, 3u);
    m_kernel = compile_msl_kernel(backend, spec, module, "split_kernel", msl_generator, &log);
    OPENVINO_ASSERT(m_kernel, "MetalSplitOp: failed to compile kernel: ", log);

    MetalOp::compile(buffer_manager);
}

void MetalSplitOp::set_outputs(const std::vector<std::unique_ptr<MetalTensor>>& outputs) {
    m_outputs.clear();
    m_outputs.reserve(outputs.size());
    for (const auto& o : outputs) {
        m_outputs.push_back(o.get());
    }
}

void MetalSplitOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(!inputs().empty(), "Split: no inputs bound");
    MetalTensor* src = inputs().at(0);
    OPENVINO_ASSERT(src && src->buf.valid(), "Split: input buffer is null");
    // Some graphs (e.g. num_splits = 1) route through set_output(), so fill m_outputs lazily.
    if (m_outputs.empty() && output()) {
        m_outputs.push_back(output());
    }
    OPENVINO_ASSERT(!m_outputs.empty(), "Split: outputs not bound");

    const ov::Shape shape = !src->shape.empty() ? src->shape : m_input_shape;
    OPENVINO_ASSERT(!shape.empty(), "Split: input shape unknown");
    const size_t src_bytes = element_size() * ov::shape_size(shape);
    OPENVINO_ASSERT(src->buf.size >= src_bytes, "Split: input buffer too small");
    int64_t axis_norm = normalize_axis(m_axis, shape.size(), "Split");

    size_t axis_len = shape[static_cast<size_t>(axis_norm)];
    size_t outer = static_cast<size_t>(shape_product(shape, 0, static_cast<size_t>(axis_norm)));
    size_t inner = static_cast<size_t>(shape_product(shape,
                                                     static_cast<size_t>(axis_norm) + 1,
                                                     shape.size()));

    if (!m_is_variadic) {
        const size_t parts = m_num_splits ? m_num_splits : m_outputs.size();
        OPENVINO_ASSERT(parts > 0, "Split: number of splits is zero");
        OPENVINO_ASSERT(axis_len % parts == 0, "Split dimension not divisible by parts");
        const size_t chunk = axis_len / parts;
        m_split_sizes.assign(parts, chunk);
    } else {
        // If split sizes were not known statically (e.g., zeros), derive from output shapes if present.
        size_t total_requested = 0;
        for (auto s : m_split_sizes) total_requested += s;
        if (total_requested == 0 || total_requested != axis_len) {
            m_split_sizes.clear();
            m_split_sizes.reserve(m_outputs.size());
            for (auto* out : m_outputs) {
                size_t sz = out && out->shape.size() > static_cast<size_t>(axis_norm)
                                ? out->shape[static_cast<size_t>(axis_norm)]
                                : 0;
                m_split_sizes.push_back(sz);
            }
        }
    }
    size_t sum = 0;
    for (auto s : m_split_sizes) sum += s;
    OPENVINO_ASSERT(sum == axis_len, "Split: split sizes do not sum to axis length");

    // Allocate outputs and set shapes/types.
    for (size_t i = 0; i < m_outputs.size(); ++i) {
        MetalTensor* out = m_outputs[i];
        size_t split = m_split_sizes[i];
        if (out) {
            ov::Shape out_shape = shape;
            out_shape[static_cast<size_t>(axis_norm)] = split;
            out->shape = out_shape;
            out->expected_type = m_element_type;
            size_t bytes = element_size() * outer * split * inner;
            if (!out->buf.valid() || out->buf.size < bytes) {
                out->buf = allocate_temp_buffer(bytes,
                                                      m_element_type,
                                                      /*persistent=*/false,
                                                      out->prefer_private);
            }
        }
    }

    size_t axis_offset = 0;
    for (size_t out_idx = 0; out_idx < m_outputs.size(); ++out_idx) {
        MetalTensor* out = m_outputs[out_idx];
        if (!out || !out->buf.valid()) {
            axis_offset += m_split_sizes[out_idx];
            continue;
        }
        const size_t split = m_split_sizes[out_idx];

        const uint64_t total = static_cast<uint64_t>(outer) * split * inner;
        if (total > 0) {
            struct SplitParams {
                uint32_t outer;
                uint32_t inner;
                uint32_t axis_offset;
                uint32_t axis_len;
                uint32_t axis_total;
            } params{};
            params.outer = static_cast<uint32_t>(outer);
            params.inner = static_cast<uint32_t>(inner);
            params.axis_offset = static_cast<uint32_t>(axis_offset);
            params.axis_len = static_cast<uint32_t>(split);
            params.axis_total = static_cast<uint32_t>(axis_len);

            KernelDispatch dispatch = make_1d_dispatch(static_cast<size_t>(total), m_kernel->clamp_threadgroup_size(256));

            KernelArgsBuilder args_builder(name().c_str());
            append_kernel_input_args(args_builder, 1, [&](size_t) { return src; }, name().c_str());
            args_builder.add_output(out);
            args_builder.add_bytes(&params, sizeof(params));

            const auto args = args_builder.finalize(buffer_manager(), m_kernel.get());
            execute_kernel(*m_kernel, cmd_buf_handle, dispatch, args);
        }
        axis_offset += split;
    }
}

}  // namespace gfx_plugin
}  // namespace ov
