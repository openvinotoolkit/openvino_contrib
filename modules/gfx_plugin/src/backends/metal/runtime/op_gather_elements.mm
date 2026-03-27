// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/op_gather_elements.hpp"

#include <string>

#include "backends/metal/codegen/metal_codegen_backend.hpp"
#include "mlir/gfx_mlir_kernel_builder.hpp"
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
std::array<uint32_t, GatherElementsCodegenDesc::kMaxDims> make_dims(const ov::Shape& shape) {
    std::array<uint32_t, GatherElementsCodegenDesc::kMaxDims> dims{};
    for (size_t i = 0; i < shape.size() && i < GatherElementsCodegenDesc::kMaxDims; ++i) {
        dims[i] = static_cast<uint32_t>(shape[i]);
    }
    return dims;
}

std::array<uint32_t, GatherElementsCodegenDesc::kMaxDims> make_strides(const ov::Shape& shape) {
    std::array<uint32_t, GatherElementsCodegenDesc::kMaxDims> strides{};
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

MetalGatherElementsOp::MetalGatherElementsOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "GatherElements",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_node(node),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    parse_gather_elements(node);
}

void MetalGatherElementsOp::parse_gather_elements(const std::shared_ptr<const ov::Node>& node) {
    auto g6 = ov::as_type_ptr<const ov::op::v6::GatherElements>(node);
    OPENVINO_ASSERT(g6, "MetalGatherElementsOp expects GatherElements v6");

    const auto data_shape = node->get_input_shape(0);
    const auto idx_shape = node->get_input_shape(1);
    const auto out_shape = node->get_output_shape(0);
    OPENVINO_ASSERT(!data_shape.empty() && !idx_shape.empty(), "GatherElements: shapes must be static");
    OPENVINO_ASSERT(data_shape.size() == idx_shape.size(), "GatherElements: data/indices rank mismatch");
    OPENVINO_ASSERT(data_shape.size() == out_shape.size(), "GatherElements: output rank mismatch");
    OPENVINO_ASSERT(data_shape.size() <= GatherElementsCodegenDesc::kMaxDims, "GatherElements: rank too large");

    const int64_t axis = normalize_axis(g6->get_axis(),
                                        data_shape.size(),
                                        "GatherElements");

    m_rank = static_cast<uint32_t>(data_shape.size());
    m_axis = static_cast<uint32_t>(axis);
    m_total = ov::shape_size(out_shape);

    m_element_type = node->get_output_element_type(0);
    OPENVINO_ASSERT(m_element_type == ov::element::f16 || m_element_type == ov::element::f32 ||
                        m_element_type == ov::element::i32 || m_element_type == ov::element::i64,
                    "GatherElements: element type not supported");
    m_index_type = node->get_input_element_type(1);
    OPENVINO_ASSERT(m_index_type == ov::element::i32 || m_index_type == ov::element::i64,
                    "GatherElements: indices type must be i32/i64");

    m_out_dims = make_dims(out_shape);
    m_out_strides = make_strides(out_shape);
    m_data_dims = make_dims(data_shape);
    m_data_strides = make_strides(data_shape);

    m_desc.element_type = m_element_type == ov::element::dynamic ? ov::element::f32 : m_element_type;
    m_desc.index_type = m_index_type;
    m_desc.rank = m_rank;
    m_desc.axis = m_axis;
    m_desc.total = static_cast<uint32_t>(m_total);
    m_desc.out_dims = m_out_dims;
    m_desc.out_strides = m_out_strides;
    m_desc.data_dims = m_data_dims;
    m_desc.data_strides = m_data_strides;
}

void MetalGatherElementsOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalGatherElementsOp::compile(MetalBufferManager* buffer_manager) {
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
    auto& ctx = gfx_mlir_context();
    auto module = build_mlir_for_node(m_node, ctx);
    auto msl_desc = m_desc;
    auto msl_generator = [msl_desc](mlir::ModuleOp mod) { return generate_msl_from_mlir(mod, msl_desc); };

    KernelSpec spec(m_node, 4u);
    m_kernel = compile_msl_kernel(backend, spec, module, "gather_elements_kernel", msl_generator, &log);
    OPENVINO_ASSERT(m_kernel, "MetalGatherElementsOp: failed to compile kernel: ", log);

    MetalOp::compile(buffer_manager);
}

void MetalGatherElementsOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(inputs().size() >= 2, "GatherElements: missing inputs");
    MetalTensor* data = inputs()[0];
    MetalTensor* indices = inputs()[1];
    if ((!indices || !indices->buf.valid()) && m_const_indices.buf.valid()) {
        indices = &m_const_indices;
    }
    OPENVINO_ASSERT(data && data->buf.valid(), "GatherElements: data buffer null");
    OPENVINO_ASSERT(indices && indices->buf.valid(), "GatherElements: indices buffer null");

    ov::Shape data_shape = !data->shape.empty() ? data->shape : m_node->get_input_shape(0);
    OPENVINO_ASSERT(!data_shape.empty(), "GatherElements: data shape unknown");
    if (m_node->get_input_partial_shape(0).is_static()) {
        OPENVINO_ASSERT(data_shape == m_node->get_input_shape(0),
                        "GatherElements: runtime data shape mismatch");
    }
    const size_t data_bytes = ov::shape_size(data_shape) * m_element_type.size();
    OPENVINO_ASSERT(data->buf.size >= data_bytes, "GatherElements: data buffer too small");
    const size_t idx_bytes = static_cast<size_t>(m_total) * m_index_type.size();
    OPENVINO_ASSERT(indices->buf.size >= idx_bytes, "GatherElements: indices buffer too small");

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

    struct GatherElementsParams {
        uint32_t rank;
        uint32_t axis;
        uint32_t total;
        uint32_t out_dims[GatherElementsCodegenDesc::kMaxDims];
        uint32_t out_strides[GatherElementsCodegenDesc::kMaxDims];
        uint32_t data_dims[GatherElementsCodegenDesc::kMaxDims];
        uint32_t data_strides[GatherElementsCodegenDesc::kMaxDims];
    } params{};

    params.rank = m_rank;
    params.axis = m_axis;
    params.total = static_cast<uint32_t>(m_total);
    for (size_t i = 0; i < GatherElementsCodegenDesc::kMaxDims; ++i) {
        params.out_dims[i] = m_out_dims[i];
        params.out_strides[i] = m_out_strides[i];
        params.data_dims[i] = m_data_dims[i];
        params.data_strides[i] = m_data_strides[i];
    }

    if (m_total == 0) {
        return;
    }
    KernelDispatch dispatch = make_1d_dispatch(static_cast<size_t>(m_total), m_kernel->clamp_threadgroup_size(256));

    KernelArgsBuilder args_builder(name().c_str());
    append_kernel_input_args(args_builder,
                             2,
                             [&](size_t idx) { return idx == 0 ? data : indices; },
                             name().c_str());
    args_builder.add_output(&dst);
    args_builder.add_bytes(&params, sizeof(params));

    const auto args = args_builder.finalize(buffer_manager(), m_kernel.get());
    execute_kernel(*m_kernel, cmd_buf_handle, dispatch, args);
}

}  // namespace gfx_plugin
}  // namespace ov
