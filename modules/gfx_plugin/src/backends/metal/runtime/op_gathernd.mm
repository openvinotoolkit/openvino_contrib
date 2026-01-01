// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/op_gathernd.hpp"

#include <string>

#include "backends/metal/codegen/metal_codegen_backend.hpp"
#include "mlir/mlir_builder.hpp"
#include "mlir_codegen/codegen_common.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "runtime/gfx_logger.hpp"
#include "runtime/gfx_shape_utils.hpp"
#include "backends/metal/runtime/op_utils.hpp"
#include "kernel_ir/gfx_kernel_args.hpp"
#include "mlir/IR/MLIRContext.h"

namespace ov {
namespace gfx_plugin {

namespace {
constexpr size_t kMaxGatherNDDims = 8;
}  // namespace

MetalGatherNDOp::MetalGatherNDOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "GatherND",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_node(node),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    parse_gathernd(node);
}

void MetalGatherNDOp::parse_gathernd(const std::shared_ptr<const ov::Node>& node) {
    auto g5 = ov::as_type_ptr<const ov::op::v5::GatherND>(node);
    auto g8 = ov::as_type_ptr<const ov::op::v8::GatherND>(node);
    OPENVINO_ASSERT(g5 || g8, "MetalGatherNDOp expects GatherND v5/v8");
    size_t batch_dims = g5 ? g5->get_batch_dims() : g8->get_batch_dims();
    OPENVINO_ASSERT(batch_dims == 0, "GatherND: batch_dims not supported");

    const auto data_shape = node->get_input_shape(0);
    const auto idx_shape = node->get_input_shape(1);
    OPENVINO_ASSERT(!data_shape.empty(), "GatherND: data shape must be static");
    OPENVINO_ASSERT(!idx_shape.empty(), "GatherND: indices shape must be static");

    const size_t data_rank = data_shape.size();
    const size_t idx_rank = idx_shape.size();
    const size_t k = idx_shape.back();
    OPENVINO_ASSERT(k >= 1 && k <= data_rank, "GatherND: indices last dim out of range");
    OPENVINO_ASSERT(k <= kMaxGatherNDDims, "GatherND: indices last dim too large");

    m_k = k;
    m_num_indices = (idx_rank > 1) ? shape_product(idx_shape, 0, idx_rank - 1) : 1;
    m_inner = shape_product(data_shape, k, data_rank);
    m_total = m_num_indices * m_inner;

    for (size_t i = 0; i < k; ++i) {
        m_dims[i] = static_cast<uint32_t>(data_shape[i]);
        m_strides[i] = static_cast<uint32_t>(shape_product(data_shape, i + 1, data_rank));
    }

    m_element_type = node->get_output_element_type(0);
    OPENVINO_ASSERT(m_element_type == ov::element::f16 || m_element_type == ov::element::f32 ||
                        m_element_type == ov::element::i32 || m_element_type == ov::element::i64,
                    "GatherND: element type not supported");
    m_index_type = node->get_input_element_type(1);
    OPENVINO_ASSERT(m_index_type == ov::element::i32 || m_index_type == ov::element::i64,
                    "GatherND: indices type must be i32/i64");

    m_desc.element_type = m_element_type == ov::element::dynamic ? ov::element::f32 : m_element_type;
    m_desc.index_type = m_index_type;
    m_desc.inner = static_cast<uint32_t>(m_inner);
    m_desc.num_indices = static_cast<uint32_t>(m_num_indices);
    m_desc.k = static_cast<uint32_t>(m_k);
    m_desc.total = static_cast<uint32_t>(m_total);
    m_desc.strides = m_strides;
    m_desc.dims = m_dims;
}

void MetalGatherNDOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalGatherNDOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }
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
    auto module = build_mlir_gathernd_from_model(make_single_op_model(m_node), ctx);
    auto source = generate_msl_from_mlir(module, m_desc);

    KernelSpec spec(m_node, 4u);
    m_kernel = compile_msl_kernel(backend, spec, module, "gathernd_kernel", source, &log);
    OPENVINO_ASSERT(m_kernel, "MetalGatherNDOp: failed to compile kernel: ", log);

    MetalOp::compile(buffer_manager);
}

void MetalGatherNDOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(inputs().size() >= 2, "GatherND: missing inputs");
    MetalTensor* data = inputs()[0];
    MetalTensor* indices = inputs()[1];
    if ((!indices || !indices->buf.valid()) && m_const_indices.buf.valid()) {
        indices = &m_const_indices;
    }
    OPENVINO_ASSERT(data && data->buf.valid(), "GatherND: data buffer null");
    OPENVINO_ASSERT(indices && indices->buf.valid(), "GatherND: indices buffer null");

    ov::Shape data_shape = !data->shape.empty() ? data->shape : m_node->get_input_shape(0);
    OPENVINO_ASSERT(!data_shape.empty(), "GatherND: data shape unknown");
    if (m_node->get_input_partial_shape(0).is_static()) {
        OPENVINO_ASSERT(data_shape == m_node->get_input_shape(0),
                        "GatherND: runtime data shape mismatch");
    }
    const size_t data_bytes = ov::shape_size(data_shape) * m_element_type.size();
    OPENVINO_ASSERT(data->buf.size >= data_bytes, "GatherND: data buffer too small");
    const size_t idx_elems = static_cast<size_t>(m_num_indices) * static_cast<size_t>(m_k);
    const size_t idx_bytes = idx_elems * m_index_type.size();
    OPENVINO_ASSERT(indices->buf.size >= idx_bytes, "GatherND: indices buffer too small");

    MetalTensor& dst = require_output();
    ov::Shape out_shape = !output_shape().empty() ? output_shape() : ov::Shape{};
    if (out_shape.empty()) {
        out_shape = m_node->get_output_shape(0);
    }
    const size_t bytes = ov::shape_size(out_shape) * m_element_type.size();
    if (!dst.buf.valid() || dst.buf.size < bytes) {
        dst.buf = allocate_temp_buffer(bytes, m_element_type, /*persistent=*/false, dst.prefer_private);
    }
    dst.shape = out_shape;
    dst.expected_type = m_element_type;

    struct GatherNDParams {
        uint32_t inner;
        uint32_t num_indices;
        uint32_t k;
        uint32_t total;
        uint32_t strides[kMaxGatherNDDims];
        uint32_t dims[kMaxGatherNDDims];
    } params{};

    params.inner = static_cast<uint32_t>(m_inner);
    params.num_indices = static_cast<uint32_t>(m_num_indices);
    params.k = static_cast<uint32_t>(m_k);
    params.total = static_cast<uint32_t>(m_total);
    for (size_t i = 0; i < kMaxGatherNDDims; ++i) {
        params.strides[i] = m_strides[i];
        params.dims[i] = m_dims[i];
    }

    if (m_total == 0) {
        return;
    }
    KernelDispatch dispatch = make_1d_dispatch(static_cast<size_t>(m_total), m_kernel->clamp_threadgroup_size(256));

    std::vector<KernelArg> args;
    args.reserve(4);
    append_kernel_input_args(args,
                             2,
                             [&](size_t idx) { return idx == 0 ? data : indices; },
                             name().c_str());
    append_kernel_output_args(args, static_cast<uint32_t>(args.size()), &dst, name().c_str());
    args.push_back(make_bytes_arg(static_cast<uint32_t>(args.size()), &params, sizeof(params)));
    execute_kernel(*m_kernel, cmd_buf_handle, dispatch, args);
}

}  // namespace gfx_plugin
}  // namespace ov
