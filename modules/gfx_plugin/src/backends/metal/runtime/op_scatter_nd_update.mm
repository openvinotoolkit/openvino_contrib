// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/op_scatter_nd_update.hpp"

#include <string>

#include "backends/metal/codegen/metal_codegen_backend.hpp"
#include "mlir/mlir_builder.hpp"
#include "mlir/codegen_common.hpp"
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
constexpr size_t kMaxScatterNDDims = 8;
}  // namespace

MetalScatterNDUpdateOp::MetalScatterNDUpdateOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "ScatterNDUpdate",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_node(node),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    parse_scatter_nd_update(node);
}

void MetalScatterNDUpdateOp::parse_scatter_nd_update(const std::shared_ptr<const ov::Node>& node) {
    auto v3 = ov::as_type_ptr<const ov::op::v3::ScatterNDUpdate>(node);
    auto v15 = ov::as_type_ptr<const ov::op::v15::ScatterNDUpdate>(node);
    OPENVINO_ASSERT(v3 || v15, "MetalScatterNDUpdateOp expects ScatterNDUpdate v3/v15");
    if (v15) {
        OPENVINO_ASSERT(v15->get_reduction() == ov::op::v15::ScatterNDUpdate::Reduction::NONE,
                        "ScatterNDUpdate v15: only reduction NONE supported");
    }

    const auto data_shape = node->get_input_shape(0);
    const auto idx_shape = node->get_input_shape(1);
    const auto upd_shape = node->get_input_shape(2);
    OPENVINO_ASSERT(!data_shape.empty(), "ScatterNDUpdate: data shape must be static");
    OPENVINO_ASSERT(!idx_shape.empty(), "ScatterNDUpdate: indices shape must be static");
    OPENVINO_ASSERT(!upd_shape.empty(), "ScatterNDUpdate: updates shape must be static");

    const size_t data_rank = data_shape.size();
    const size_t idx_rank = idx_shape.size();
    OPENVINO_ASSERT(idx_rank >= 1, "ScatterNDUpdate: indices rank must be >=1");
    const size_t k = idx_shape.back();
    OPENVINO_ASSERT(k >= 1 && k <= data_rank, "ScatterNDUpdate: indices last dim out of range");
    OPENVINO_ASSERT(k <= kMaxScatterNDDims, "ScatterNDUpdate: indices last dim too large");

    const size_t expected_updates_prefix = idx_rank - 1;
    const size_t expected_updates_rank = expected_updates_prefix + (data_rank - k);
    OPENVINO_ASSERT(upd_shape.size() == expected_updates_rank, "ScatterNDUpdate: updates rank mismatch");

    for (size_t i = 0; i < expected_updates_prefix; ++i) {
        OPENVINO_ASSERT(upd_shape[i] == idx_shape[i], "ScatterNDUpdate: updates prefix shape mismatch");
    }
    for (size_t i = 0; i < data_rank - k; ++i) {
        OPENVINO_ASSERT(upd_shape[expected_updates_prefix + i] == data_shape[k + i],
                        "ScatterNDUpdate: updates suffix shape mismatch");
    }

    m_k = k;
    m_num_indices = (idx_rank > 1) ? shape_product(idx_shape, 0, idx_rank - 1) : 1;
    m_inner = shape_product(data_shape, k, data_rank);
    m_total_updates = m_num_indices * m_inner;
    m_total_data = ov::shape_size(data_shape);

    for (size_t i = 0; i < k; ++i) {
        m_dims[i] = static_cast<uint32_t>(data_shape[i]);
        m_strides[i] = static_cast<uint32_t>(shape_product(data_shape, i + 1, data_rank));
    }

    m_element_type = node->get_output_element_type(0);
    OPENVINO_ASSERT(m_element_type == ov::element::f16 || m_element_type == ov::element::f32 ||
                        m_element_type == ov::element::i32 || m_element_type == ov::element::i64,
                    "ScatterNDUpdate: element type not supported");
    m_index_type = node->get_input_element_type(1);
    OPENVINO_ASSERT(m_index_type == ov::element::i32 || m_index_type == ov::element::i64,
                    "ScatterNDUpdate: indices type must be i32/i64");

    m_desc.element_type = m_element_type == ov::element::dynamic ? ov::element::f32 : m_element_type;
    m_desc.index_type = m_index_type;
    m_desc.inner = static_cast<uint32_t>(m_inner);
    m_desc.num_indices = static_cast<uint32_t>(m_num_indices);
    m_desc.k = static_cast<uint32_t>(m_k);
    m_desc.total_updates = static_cast<uint32_t>(m_total_updates);
    m_desc.total_data = static_cast<uint32_t>(m_total_data);
    m_desc.strides = m_strides;
    m_desc.dims = m_dims;
}

void MetalScatterNDUpdateOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalScatterNDUpdateOp::compile(MetalBufferManager* buffer_manager) {
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
    auto upd_const = ov::as_type_ptr<const ov::op::v0::Constant>(m_node->get_input_node_shared_ptr(2));
    if (upd_const) {
        const size_t bytes = upd_const->get_byte_size();
        const std::string key = m_node->get_friendly_name() + "/updates";
        m_const_updates.buf = buffer_manager->wrap_const(key, upd_const->get_data_ptr(), bytes, m_element_type);
        m_const_updates.shape = upd_const->get_shape();
        m_const_updates.expected_type = m_element_type;
    }

    MetalCodegenBackend backend(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    mlir::MLIRContext ctx;
    auto module = build_mlir_scatter_nd_update_from_model(make_single_op_model(m_node), ctx);
    auto msl_desc = m_desc;
    auto msl_generator = [msl_desc](mlir::ModuleOp mod) { return generate_msl_from_mlir(mod, msl_desc); };

    KernelSpec spec_init(m_node, 3u);
    KernelSpec spec_update(m_node, 4u);
    m_kernel_init = compile_msl_kernel(backend, spec_init, module, "scatter_nd_init", msl_generator, &log);
    OPENVINO_ASSERT(m_kernel_init, "MetalScatterNDUpdateOp: init kernel compile failed: ", log);
    m_kernel_update = compile_msl_kernel(backend, spec_update, module, "scatter_nd_update", msl_generator, &log);
    OPENVINO_ASSERT(m_kernel_update, "MetalScatterNDUpdateOp: update kernel compile failed: ", log);

    MetalOp::compile(buffer_manager);
}

void MetalScatterNDUpdateOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(inputs().size() >= 3, "ScatterNDUpdate: missing inputs");
    MetalTensor* data = inputs()[0];
    MetalTensor* indices = inputs()[1];
    MetalTensor* updates = inputs()[2];

    if ((!indices || !indices->buf.valid()) && m_const_indices.buf.valid()) {
        indices = &m_const_indices;
    }
    if ((!updates || !updates->buf.valid()) && m_const_updates.buf.valid()) {
        updates = &m_const_updates;
    }

    OPENVINO_ASSERT(data && data->buf.valid(), "ScatterNDUpdate: data buffer null");
    OPENVINO_ASSERT(indices && indices->buf.valid(), "ScatterNDUpdate: indices buffer null");
    OPENVINO_ASSERT(updates && updates->buf.valid(), "ScatterNDUpdate: updates buffer null");

    ov::Shape data_shape = !data->shape.empty() ? data->shape : m_node->get_input_shape(0);
    OPENVINO_ASSERT(!data_shape.empty(), "ScatterNDUpdate: data shape unknown");
    if (m_node->get_input_partial_shape(0).is_static()) {
        OPENVINO_ASSERT(data_shape == m_node->get_input_shape(0),
                        "ScatterNDUpdate: runtime data shape mismatch");
    }
    const size_t data_bytes = ov::shape_size(data_shape) * m_element_type.size();
    OPENVINO_ASSERT(data->buf.size >= data_bytes, "ScatterNDUpdate: data buffer too small");
    const size_t idx_elems = static_cast<size_t>(m_num_indices) * static_cast<size_t>(m_k);
    const size_t idx_bytes = idx_elems * m_index_type.size();
    const size_t upd_bytes = static_cast<size_t>(m_total_updates) * m_element_type.size();
    OPENVINO_ASSERT(indices->buf.size >= idx_bytes, "ScatterNDUpdate: indices buffer too small");
    OPENVINO_ASSERT(updates->buf.size >= upd_bytes, "ScatterNDUpdate: updates buffer too small");

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

    struct ScatterNDParams {
        uint32_t inner;
        uint32_t num_indices;
        uint32_t k;
        uint32_t total_updates;
        uint32_t total_data;
        uint32_t strides[ScatterNDUpdateCodegenDesc::kMaxDims];
        uint32_t dims[ScatterNDUpdateCodegenDesc::kMaxDims];
    } params{};

    params.inner = static_cast<uint32_t>(m_inner);
    params.num_indices = static_cast<uint32_t>(m_num_indices);
    params.k = static_cast<uint32_t>(m_k);
    params.total_updates = static_cast<uint32_t>(m_total_updates);
    params.total_data = static_cast<uint32_t>(m_total_data);
    for (size_t i = 0; i < ScatterNDUpdateCodegenDesc::kMaxDims; ++i) {
        params.strides[i] = m_strides[i];
        params.dims[i] = m_dims[i];
    }

    // Init kernel: copy data to output
    if (m_total_data > 0) {
        KernelDispatch dispatch = make_1d_dispatch(static_cast<size_t>(m_total_data), m_kernel_init->clamp_threadgroup_size(256));
        KernelArgsBuilder args_builder(name().c_str());
        append_kernel_input_args(args_builder, 1, [&](size_t) { return data; }, name().c_str());
        args_builder.add_output(&dst);
        args_builder.add_bytes(&params, sizeof(params));

        const auto args = args_builder.finalize(buffer_manager(), m_kernel_init.get());
        execute_kernel(*m_kernel_init, cmd_buf_handle, dispatch, args);
    }

    // Update kernel: scatter updates
    if (m_total_updates > 0) {
        KernelDispatch dispatch = make_1d_dispatch(static_cast<size_t>(m_total_updates), m_kernel_update->clamp_threadgroup_size(256));
        KernelArgsBuilder args_builder(name().c_str());
        append_kernel_input_args(args_builder,
                                 2,
                                 [&](size_t idx) { return idx == 0 ? indices : updates; },
                                 name().c_str());
        args_builder.add_output(&dst);
        args_builder.add_bytes(&params, sizeof(params));

        const auto args = args_builder.finalize(buffer_manager(), m_kernel_update.get());
        execute_kernel(*m_kernel_update, cmd_buf_handle, dispatch, args);
    }
}

}  // namespace gfx_plugin
}  // namespace ov