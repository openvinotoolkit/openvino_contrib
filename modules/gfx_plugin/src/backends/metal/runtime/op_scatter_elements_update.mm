// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/op_scatter_elements_update.hpp"

#include <string>

#include "backends/metal/codegen/metal_codegen_backend.hpp"
#include "mlir/mlir_builder.hpp"
#include "mlir/codegen_common.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "runtime/gfx_logger.hpp"
#include "backends/metal/runtime/op_utils.hpp"
#include "kernel_ir/gfx_kernel_args.hpp"
#include "mlir/IR/MLIRContext.h"
#include "runtime/gfx_shape_utils.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
std::array<uint32_t, ScatterElementsUpdateCodegenDesc::kMaxDims> make_dims(const ov::Shape& shape) {
    std::array<uint32_t, ScatterElementsUpdateCodegenDesc::kMaxDims> dims{};
    for (size_t i = 0; i < shape.size() && i < ScatterElementsUpdateCodegenDesc::kMaxDims; ++i) {
        dims[i] = static_cast<uint32_t>(shape[i]);
    }
    return dims;
}

std::array<uint32_t, ScatterElementsUpdateCodegenDesc::kMaxDims> make_strides(const ov::Shape& shape) {
    std::array<uint32_t, ScatterElementsUpdateCodegenDesc::kMaxDims> strides{};
    if (shape.empty())
        return strides;
    uint64_t acc = 1;
    for (size_t i = shape.size(); i-- > 0;) {
        strides[i] = static_cast<uint32_t>(acc);
        acc *= shape[i];
    }
    return strides;
}
}  // namespace

MetalScatterElementsUpdateOp::MetalScatterElementsUpdateOp(const std::shared_ptr<const ov::Node>& node,
                                                           void* device,
                                                           void* queue)
    : MetalOp(node->get_friendly_name(),
              "ScatterElementsUpdate",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_node(node),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    parse_scatter_elements_update(node);
}

void MetalScatterElementsUpdateOp::parse_scatter_elements_update(const std::shared_ptr<const ov::Node>& node) {
    auto v3 = ov::as_type_ptr<const ov::op::v3::ScatterElementsUpdate>(node);
    auto v12 = ov::as_type_ptr<const ov::op::v12::ScatterElementsUpdate>(node);
    OPENVINO_ASSERT(v3 || v12, "MetalScatterElementsUpdateOp expects ScatterElementsUpdate v3/v12");
    if (v12) {
        OPENVINO_ASSERT(v12->get_reduction() == ov::op::v12::ScatterElementsUpdate::Reduction::NONE,
                        "ScatterElementsUpdate v12: only reduction NONE supported");
        OPENVINO_ASSERT(v12->get_use_init_val(), "ScatterElementsUpdate v12: use_init_val=false not supported");
    }

    const auto data_shape = node->get_input_shape(0);
    const auto idx_shape = node->get_input_shape(1);
    const auto upd_shape = node->get_input_shape(2);
    const auto out_shape = node->get_output_shape(0);

    OPENVINO_ASSERT(!data_shape.empty() && !idx_shape.empty() && !upd_shape.empty(),
                    "ScatterElementsUpdate: shapes must be static");
    OPENVINO_ASSERT(data_shape.size() == idx_shape.size(), "ScatterElementsUpdate: rank mismatch");
    OPENVINO_ASSERT(idx_shape == upd_shape, "ScatterElementsUpdate: indices/updates shape mismatch");
    OPENVINO_ASSERT(data_shape.size() <= ScatterElementsUpdateCodegenDesc::kMaxDims,
                    "ScatterElementsUpdate: rank too large");

    auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(node->get_input_node_shared_ptr(3));
    OPENVINO_ASSERT(axis_const, "ScatterElementsUpdate: axis must be constant");
    auto axis_v = axis_const->cast_vector<int64_t>();
    OPENVINO_ASSERT(axis_v.size() == 1, "ScatterElementsUpdate: axis must be scalar");
    const int64_t axis = normalize_axis(axis_v[0],
                                        data_shape.size(),
                                        "ScatterElementsUpdate");

    m_rank = static_cast<uint32_t>(data_shape.size());
    m_axis = static_cast<uint32_t>(axis);
    m_total_updates = ov::shape_size(idx_shape);
    m_total_data = ov::shape_size(data_shape);

    m_element_type = node->get_output_element_type(0);
    OPENVINO_ASSERT(m_element_type == ov::element::f16 || m_element_type == ov::element::f32 ||
                        m_element_type == ov::element::i32 || m_element_type == ov::element::i64,
                    "ScatterElementsUpdate: element type not supported");
    m_index_type = node->get_input_element_type(1);
    OPENVINO_ASSERT(m_index_type == ov::element::i32 || m_index_type == ov::element::i64,
                    "ScatterElementsUpdate: indices type must be i32/i64");

    m_update_dims = make_dims(idx_shape);
    m_update_strides = make_strides(idx_shape);
    m_data_dims = make_dims(data_shape);
    m_data_strides = make_strides(data_shape);

    m_desc.element_type = m_element_type == ov::element::dynamic ? ov::element::f32 : m_element_type;
    m_desc.index_type = m_index_type;
    m_desc.rank = m_rank;
    m_desc.axis = m_axis;
    m_desc.total_updates = static_cast<uint32_t>(m_total_updates);
    m_desc.total_data = static_cast<uint32_t>(m_total_data);
    m_desc.update_dims = m_update_dims;
    m_desc.update_strides = m_update_strides;
    m_desc.data_dims = m_data_dims;
    m_desc.data_strides = m_data_strides;
}

void MetalScatterElementsUpdateOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalScatterElementsUpdateOp::compile(MetalBufferManager* buffer_manager) {
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
    auto module = build_mlir_scatter_elements_update_from_model(make_single_op_model(m_node), ctx);
    auto msl_desc = m_desc;
    auto msl_generator = [msl_desc](mlir::ModuleOp mod) { return generate_msl_from_mlir(mod, msl_desc); };

    KernelSpec spec_init(m_node, 3u);
    KernelSpec spec_update(m_node, 4u);
    m_kernel_init = compile_msl_kernel(backend, spec_init, module, "scatter_elements_init", msl_generator, &log);
    OPENVINO_ASSERT(m_kernel_init, "MetalScatterElementsUpdateOp: init kernel compile failed: ", log);
    m_kernel_update = compile_msl_kernel(backend, spec_update, module, "scatter_elements_update", msl_generator, &log);
    OPENVINO_ASSERT(m_kernel_update, "MetalScatterElementsUpdateOp: update kernel compile failed: ", log);

    MetalOp::compile(buffer_manager);
}

void MetalScatterElementsUpdateOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(inputs().size() >= 3, "ScatterElementsUpdate: missing inputs");
    MetalTensor* data = inputs()[0];
    MetalTensor* indices = inputs()[1];
    MetalTensor* updates = inputs()[2];

    if ((!indices || !indices->buf.valid()) && m_const_indices.buf.valid()) {
        indices = &m_const_indices;
    }
    if ((!updates || !updates->buf.valid()) && m_const_updates.buf.valid()) {
        updates = &m_const_updates;
    }

    OPENVINO_ASSERT(data && data->buf.valid(), "ScatterElementsUpdate: data buffer null");
    OPENVINO_ASSERT(indices && indices->buf.valid(), "ScatterElementsUpdate: indices buffer null");
    OPENVINO_ASSERT(updates && updates->buf.valid(), "ScatterElementsUpdate: updates buffer null");

    ov::Shape data_shape = !data->shape.empty() ? data->shape : m_node->get_input_shape(0);
    OPENVINO_ASSERT(!data_shape.empty(), "ScatterElementsUpdate: data shape unknown");
    if (m_node->get_input_partial_shape(0).is_static()) {
        OPENVINO_ASSERT(data_shape == m_node->get_input_shape(0),
                        "ScatterElementsUpdate: runtime data shape mismatch");
    }
    const size_t data_bytes = ov::shape_size(data_shape) * m_element_type.size();
    OPENVINO_ASSERT(data->buf.size >= data_bytes, "ScatterElementsUpdate: data buffer too small");
    const size_t idx_bytes = static_cast<size_t>(m_total_updates) * m_index_type.size();
    const size_t upd_bytes = static_cast<size_t>(m_total_updates) * m_element_type.size();
    OPENVINO_ASSERT(indices->buf.size >= idx_bytes, "ScatterElementsUpdate: indices buffer too small");
    OPENVINO_ASSERT(updates->buf.size >= upd_bytes, "ScatterElementsUpdate: updates buffer too small");

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

    struct ScatterElementsParams {
        uint32_t rank;
        uint32_t axis;
        uint32_t total_updates;
        uint32_t total_data;
        uint32_t update_dims[ScatterElementsUpdateCodegenDesc::kMaxDims];
        uint32_t update_strides[ScatterElementsUpdateCodegenDesc::kMaxDims];
        uint32_t data_dims[ScatterElementsUpdateCodegenDesc::kMaxDims];
        uint32_t data_strides[ScatterElementsUpdateCodegenDesc::kMaxDims];
    } params{};

    params.rank = m_rank;
    params.axis = m_axis;
    params.total_updates = static_cast<uint32_t>(m_total_updates);
    params.total_data = static_cast<uint32_t>(m_total_data);
    for (size_t i = 0; i < ScatterElementsUpdateCodegenDesc::kMaxDims; ++i) {
        params.update_dims[i] = m_update_dims[i];
        params.update_strides[i] = m_update_strides[i];
        params.data_dims[i] = m_data_dims[i];
        params.data_strides[i] = m_data_strides[i];
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