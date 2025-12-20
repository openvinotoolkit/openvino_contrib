// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "runtime/metal_op_topk.hpp"

#include <limits>

#include "openvino/core/validation_util.hpp"
#include "openvino/op/util/topk_base.hpp"
#include "openvino/op/topk.hpp"
#include "kernel_codegen/metal_kernel_compiler.hpp"
#include "runtime/metal_logger.hpp"
#include "runtime/metal_op_utils.hpp"
#include "mlir_builder.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir_codegen/codegen_common.hpp"

namespace ov {
namespace metal_plugin {

namespace {
inline id<MTLBuffer> to_mtl(const MetalBuffer& buf) {
    return (__bridge id<MTLBuffer>)buf.buffer;
}
}  // namespace

MetalTopKOp::MetalTopKOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "TopK",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_node(node),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    parse_topk(node);
}

void MetalTopKOp::parse_topk(const std::shared_ptr<const ov::Node>& node) {
    auto topk = std::dynamic_pointer_cast<const ov::op::util::TopKBase>(node);
    OPENVINO_ASSERT(topk, "MetalTopKOp: TopKBase cast failed");
    m_input_shape = node->get_input_shape(0);
    m_output_shape = node->get_output_shape(0);
    m_element_type = node->get_output_element_type(0);
    m_index_type = node->get_output_element_type(1);
    m_axis = static_cast<uint32_t>(topk->get_axis());
    m_k = static_cast<uint32_t>(topk->get_k());
    m_mode_max = topk->get_mode() == ov::op::TopKMode::MAX;
    switch (topk->get_sort_type()) {
        case ov::op::TopKSortType::SORT_INDICES:
            m_sort_type = TopKSortType::SortIndices;
            break;
        case ov::op::TopKSortType::NONE:
        case ov::op::TopKSortType::SORT_VALUES:
        default:
            m_sort_type = TopKSortType::SortValues;
            break;
    }

    OPENVINO_ASSERT(!m_input_shape.empty(), "TopK: input shape is empty");
    OPENVINO_ASSERT(m_axis < m_input_shape.size(), "TopK: axis out of range");
    m_axis_len = static_cast<uint32_t>(m_input_shape[m_axis]);
    OPENVINO_ASSERT(m_k > 0 && m_k <= m_axis_len, "TopK: invalid k");
    uint64_t outer = 1;
    for (size_t i = 0; i < m_axis; ++i)
        outer *= m_input_shape[i];
    uint64_t inner = 1;
    for (size_t i = m_axis + 1; i < m_input_shape.size(); ++i)
        inner *= m_input_shape[i];
    m_outer = static_cast<uint32_t>(outer);
    m_inner = static_cast<uint32_t>(inner);
}

void MetalTopKOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalTopKOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }
    if (!m_device && buffer_manager) {
        m_device = (id<MTLDevice>)buffer_manager->device();
    }
    MetalKernelCompiler compiler(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    mlir::MLIRContext ctx;
    auto module = build_mlir_topk_from_model(make_single_op_model_all_outputs(m_node), ctx);

    TopKCodegenDesc desc{};
    desc.element_type = m_element_type == ov::element::dynamic ? ov::element::f32 : m_element_type;
    desc.index_type = m_index_type == ov::element::dynamic ? ov::element::i32 : m_index_type;
    desc.axis_len = m_axis_len;
    desc.k = m_k;
    desc.outer = m_outer == 0 ? 1u : m_outer;
    desc.inner = m_inner == 0 ? 1u : m_inner;
    desc.mode_max = m_mode_max;
    desc.sort_type = m_sort_type;

    auto source = generate_msl_from_mlir(module, desc);
    m_pipeline = compiler.compile_msl_from_source(source, "topk_kernel", log);
    OPENVINO_ASSERT(m_pipeline, "MetalTopKOp: failed to compile kernel: ", log);

    MetalOp::compile(buffer_manager);
}

void MetalTopKOp::set_outputs(const std::vector<std::unique_ptr<MetalTensor>>& outputs) {
    m_outputs.clear();
    m_outputs.reserve(outputs.size());
    for (const auto& o : outputs) {
        m_outputs.push_back(o.get());
    }
}

void MetalTopKOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(!inputs().empty(), "TopK: no inputs bound");
    MetalTensor* src = inputs().at(0);
    OPENVINO_ASSERT(src && src->buf.valid(), "TopK: input buffer is null");

    if (m_outputs.empty() && output()) {
        m_outputs.push_back(output());
    }
    OPENVINO_ASSERT(m_outputs.size() >= 2, "TopK: outputs not bound");
    MetalTensor* out_vals = m_outputs[0];
    MetalTensor* out_idx = m_outputs[1];
    OPENVINO_ASSERT(out_vals && out_idx, "TopK: output tensors are null");

    ov::Shape in_shape = !src->shape.empty() ? src->shape : m_input_shape;
    OPENVINO_ASSERT(!in_shape.empty(), "TopK: input shape unknown");
    OPENVINO_ASSERT(in_shape == m_input_shape, "TopK: runtime input shape mismatch");
    OPENVINO_ASSERT(m_axis < in_shape.size(), "TopK: axis out of range");
    const uint32_t axis_len = static_cast<uint32_t>(in_shape[m_axis]);
    OPENVINO_ASSERT(axis_len == m_axis_len, "TopK: runtime axis length mismatch");
    OPENVINO_ASSERT(m_k > 0 && m_k <= axis_len, "TopK: invalid k at runtime");
    const size_t in_bytes = m_element_type.size() * ov::shape_size(in_shape);
    OPENVINO_ASSERT(src->buf.size >= in_bytes, "TopK: input buffer too small");

    ov::Shape out_shape = !m_output_shape.empty() ? m_output_shape : ov::Shape{};
    OPENVINO_ASSERT(!out_shape.empty(), "TopK: output shape unknown");
    OPENVINO_ASSERT(out_shape == m_output_shape, "TopK: runtime output shape mismatch");

    const size_t values_bytes = m_element_type.size() * ov::shape_size(out_shape);
    const size_t index_bytes = m_index_type.size() * ov::shape_size(out_shape);
    if (!out_vals->buf.valid() || out_vals->buf.size < values_bytes) {
        out_vals->buf = buffer_manager()->allocate(values_bytes,
                                                   m_element_type,
                                                   /*persistent=*/false,
                                                   out_vals->prefer_private);
    }
    if (!out_idx->buf.valid() || out_idx->buf.size < index_bytes) {
        out_idx->buf = buffer_manager()->allocate(index_bytes,
                                                  m_index_type,
                                                  /*persistent=*/false,
                                                  out_idx->prefer_private);
    }
    out_vals->shape = out_shape;
    out_vals->expected_type = m_element_type;
    out_idx->shape = out_shape;
    out_idx->expected_type = m_index_type;

    id<MTLCommandBuffer> cb = static_cast<id<MTLCommandBuffer>>(cmd_buf_handle);
    OPENVINO_ASSERT(cb, "MetalTopKOp: command buffer is null");
    OPENVINO_ASSERT(m_pipeline, "MetalTopKOp: pipeline is null");
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m_pipeline];
    [enc setBuffer:to_mtl(src->buf) offset:0 atIndex:0];
    [enc setBuffer:to_mtl(out_vals->buf) offset:0 atIndex:1];
    [enc setBuffer:to_mtl(out_idx->buf) offset:0 atIndex:2];

    const uint32_t rows = (m_outer == 0 ? 1u : m_outer) * (m_inner == 0 ? 1u : m_inner);
    if (rows == 0) {
        [enc endEncoding];
        return;
    }
    MTLSize grid = MTLSizeMake(rows, 1, 1);
    const NSUInteger tg_size = static_cast<NSUInteger>(metal_clamp_tg_size((void*)m_pipeline, 64));
    MTLSize tg = MTLSizeMake(tg_size, 1, 1);
    start_profiling(enc);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    stop_profiling_ms(enc);
    [enc endEncoding];
}

}  // namespace metal_plugin
}  // namespace ov
