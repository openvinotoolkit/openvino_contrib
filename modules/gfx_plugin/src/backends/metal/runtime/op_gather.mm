// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/op_gather.hpp"

#include <string>

#include <numeric>

#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "backends/metal/runtime/metal_backend.hpp"
#include "mlir/mlir_builder.hpp"
#include "runtime/gfx_logger.hpp"
#include "backends/metal/runtime/op_utils.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/codegen/codegen_common.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
std::vector<int64_t> get_axis_const(const std::shared_ptr<const ov::Node>& node) {
    auto axis_c = ov::as_type_ptr<const ov::op::v0::Constant>(node->get_input_node_shared_ptr(2));
    OPENVINO_ASSERT(axis_c, "Gather: axis must be constant");
    return axis_c->cast_vector<int64_t>();
}

uint64_t product(const ov::Shape& s, size_t start, size_t end) {
    uint64_t prod = 1;
    for (size_t i = start; i < end; ++i) prod *= s[i];
    return prod;
}
}  // namespace

MetalGatherOp::MetalGatherOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "Gather",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_node(node),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    parse_gather(node);
}

void MetalGatherOp::parse_gather(const std::shared_ptr<const ov::Node>& node) {
    auto g1 = ov::as_type_ptr<const ov::op::v1::Gather>(node);
    auto g7 = ov::as_type_ptr<const ov::op::v7::Gather>(node);
    auto g8 = ov::as_type_ptr<const ov::op::v8::Gather>(node);
    OPENVINO_ASSERT(g1 || g7 || g8, "MetalGatherOp expects Gather v1/v7/v8");
    if (g7) {
        OPENVINO_ASSERT(g7->get_batch_dims() == 0, "Gather v7: batch_dims not supported");
    }
    if (g8) {
        OPENVINO_ASSERT(g8->get_batch_dims() == 0, "Gather v8: batch_dims not supported");
    }

    auto axis_v = get_axis_const(node);
    OPENVINO_ASSERT(axis_v.size() == 1, "Gather: axis must be scalar");
    int64_t axis = axis_v[0];

    const auto data_shape = node->get_input_shape(0);
    const auto idx_shape = node->get_input_shape(1);
    OPENVINO_ASSERT(!data_shape.empty(), "Gather: data shape must be static");
    OPENVINO_ASSERT(!idx_shape.empty(), "Gather: indices shape must be static");

    if (axis < 0)
        axis += static_cast<int64_t>(data_shape.size());
    OPENVINO_ASSERT(axis >= 0 && axis < static_cast<int64_t>(data_shape.size()), "Gather: axis out of range");

    m_axis = axis;
    m_axis_dim = static_cast<uint64_t>(data_shape[static_cast<size_t>(axis)]);
    m_outer = product(data_shape, 0, static_cast<size_t>(axis));
    m_inner = product(data_shape, static_cast<size_t>(axis) + 1, data_shape.size());
    m_indices_count = ov::shape_size(idx_shape);

    m_element_type = node->get_output_element_type(0);
    OPENVINO_ASSERT(m_element_type == ov::element::f16 || m_element_type == ov::element::f32 ||
                        m_element_type == ov::element::i32 || m_element_type == ov::element::i64,
                    "Gather: element type not supported");
    m_index_type = node->get_input_element_type(1);
    OPENVINO_ASSERT(m_index_type == ov::element::i32 || m_index_type == ov::element::i64,
                    "Gather: indices type must be i32/i64");

    m_desc.element_type = m_element_type == ov::element::dynamic ? ov::element::f32 : m_element_type;
    m_desc.index_type = m_index_type;
    m_desc.outer = m_outer;
    m_desc.inner = m_inner;
    m_desc.axis_dim = m_axis_dim;
    m_desc.indices_count = m_indices_count;
}

void MetalGatherOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalGatherOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }
    // Wrap constant indices if present.
    auto idx_const = ov::as_type_ptr<const ov::op::v0::Constant>(m_node->get_input_node_shared_ptr(1));
    if (idx_const) {
        const size_t bytes = idx_const->get_byte_size();
        const std::string key = m_node->get_friendly_name() + "/indices";
        m_const_indices.buf = buffer_manager->wrap_const(key, idx_const->get_data_ptr(), bytes, m_index_type);
        m_const_indices.shape = idx_const->get_shape();
        m_const_indices.expected_type = m_index_type;
    }

    MetalCodegenBackend backend(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    mlir::MLIRContext ctx;
    auto module = build_mlir_gather_from_model(make_single_op_model(m_node), ctx);
    auto source = generate_msl_from_mlir(module, m_desc);

    KernelSpec spec(m_node, 4u);
    m_kernel = compile_msl_kernel(backend, spec, module, "gather_kernel", source, &log);
    OPENVINO_ASSERT(m_kernel, "MetalGatherOp: failed to compile kernel: ", log);

    MetalOp::compile(buffer_manager);
}

void MetalGatherOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(inputs().size() >= 2, "Gather: missing inputs");
    MetalTensor* data = inputs()[0];
    MetalTensor* indices = inputs()[1];
    if ((!indices || !indices->buf.valid()) && m_const_indices.buf.valid()) {
        indices = &m_const_indices;
    }
    OPENVINO_ASSERT(data && data->buf.valid(), "Gather: data buffer null");
    OPENVINO_ASSERT(indices && indices->buf.valid(), "Gather: indices buffer null");

    ov::Shape data_shape = !data->shape.empty() ? data->shape : m_node->get_input_shape(0);
    OPENVINO_ASSERT(!data_shape.empty(), "Gather: data shape unknown");
    if (m_node->get_input_partial_shape(0).is_static()) {
        OPENVINO_ASSERT(data_shape == m_node->get_input_shape(0),
                        "Gather: runtime data shape mismatch");
    }
    const uint64_t axis_dim = data_shape[static_cast<size_t>(m_axis)];
    OPENVINO_ASSERT(axis_dim == m_axis_dim, "Gather: axis dim mismatch");
    const size_t data_bytes = ov::shape_size(data_shape) * m_element_type.size();
    OPENVINO_ASSERT(data->buf.size >= data_bytes, "Gather: data buffer too small");
    const size_t idx_bytes = static_cast<size_t>(m_indices_count) * m_index_type.size();
    OPENVINO_ASSERT(indices->buf.size >= idx_bytes, "Gather: indices buffer too small");

    MetalTensor& dst = require_output();
    ov::Shape out_shape = !output_shape().empty() ? output_shape() : ov::Shape{};
    if (out_shape.empty()) {
        out_shape = ov::Shape{static_cast<size_t>(m_outer), static_cast<size_t>(m_indices_count)};
    }

    const size_t bytes = ov::shape_size(out_shape) * m_element_type.size();
    if (!dst.buf.valid() || dst.buf.size < bytes) {
        dst.buf = buffer_manager()->allocate(bytes, m_element_type, /*persistent=*/false, dst.prefer_private);
    }
    dst.shape = out_shape;
    dst.expected_type = m_element_type;

    struct GatherParams {
        uint32_t outer;
        uint32_t inner;
        uint32_t axis_dim;
        uint32_t indices_count;
    } params{};
    params.outer = static_cast<uint32_t>(m_outer);
    params.inner = static_cast<uint32_t>(m_inner);
    params.axis_dim = static_cast<uint32_t>(m_axis_dim);
    params.indices_count = static_cast<uint32_t>(m_indices_count);

    const uint64_t total = m_outer * m_indices_count * m_inner;
    if (total == 0) {
        return;
    }
    KernelDispatch dispatch = make_1d_dispatch(static_cast<size_t>(total), m_kernel->clamp_threadgroup_size(256));

    std::vector<KernelArg> args;
    args.reserve(4);
    args.push_back(make_buffer_arg(0, data->buf));
    args.push_back(make_buffer_arg(1, indices->buf));
    args.push_back(make_buffer_arg(2, dst.buf));
    args.push_back(make_bytes_arg(3, &params, sizeof(params)));
    execute_kernel(*m_kernel, cmd_buf_handle, dispatch, args);
}

}  // namespace gfx_plugin
}  // namespace ov
