// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "runtime/metal_op_reshape.hpp"

#include <numeric>

#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "kernel_codegen/metal_kernel_compiler.hpp"
#include "runtime/metal_logger.hpp"

namespace ov {
namespace metal_plugin {

namespace {
inline id<MTLBuffer> to_mtl(const MetalBuffer& buf) {
    return (__bridge id<MTLBuffer>)buf.buffer;
}
}  // namespace

MetalReshapeOp::MetalReshapeOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "Reshape",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue) {
    m_element_type = node->get_output_element_type(0);
    if (node->get_output_partial_shape(0).is_static()) {
        m_target_shape = node->get_output_shape(0);
    }
}

void MetalReshapeOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
    // No kernel needed: reshape is treated as a view; copy avoided.
}

void MetalReshapeOp::execute() {
    OPENVINO_ASSERT(!inputs().empty(), "Reshape: missing input");
    MetalTensor* src = inputs()[0];
    OPENVINO_ASSERT(src && src->buf.valid(), "Reshape: input buffer is null");
    MetalTensor& dst = require_output();

    ov::Shape out_shape = !m_target_shape.empty() ? m_target_shape : output_shape();
    if (out_shape.empty() && !src->shape.empty()) {
        out_shape = src->shape;  // fallback
    }
    const size_t in_elems = ov::shape_size(src->shape.empty() ? out_shape : src->shape);
    const size_t out_elems = ov::shape_size(out_shape);
    OPENVINO_ASSERT(in_elems == out_elems, "Reshape: element count mismatch");

    dst.shape = out_shape;
    dst.expected_type = m_element_type.is_dynamic() ? src->expected_type : m_element_type;

    // View semantics: reuse underlying buffer, no copy.
    dst.buf = src->buf;

    start_profiling();
    stop_profiling_ms();
}

MetalTransposeOp::MetalTransposeOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "Transpose",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    build_desc(node);
}

void MetalTransposeOp::build_desc(const std::shared_ptr<const ov::Node>& node) {
    auto tr = ov::as_type_ptr<const ov::op::v1::Transpose>(node);
    OPENVINO_ASSERT(tr, "MetalTransposeOp expects v1::Transpose");
    m_element_type = tr->get_output_element_type(0);

    if (tr->get_input_partial_shape(0).is_static()) {
        m_desc.transpose.in_shape = std::vector<int64_t>(tr->get_input_shape(0).begin(),
                                                         tr->get_input_shape(0).end());
    }
    if (tr->get_output_partial_shape(0).is_static()) {
        m_desc.transpose.out_shape = std::vector<int64_t>(tr->get_output_shape(0).begin(),
                                                          tr->get_output_shape(0).end());
    }

    auto perm_const = ov::as_type_ptr<const ov::op::v0::Constant>(tr->input_value(1).get_node_shared_ptr());
    OPENVINO_ASSERT(perm_const, "Transpose: perm must be constant");
    auto perm = perm_const->cast_vector<int64_t>();
    m_desc.transpose.perm = perm;

    m_desc.kind = KernelOpKind::Transpose;
    m_desc.dtype = resolve_metal_dtype(m_element_type);
    m_desc.element_type = static_cast<uint32_t>(static_cast<ov::element::Type_t>(m_element_type));
}

void MetalTransposeOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
    MetalKernelCompiler compiler(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    m_pipeline = compiler.compile_transpose_kernel(m_desc, log);
    OPENVINO_ASSERT(m_pipeline, "MetalTransposeOp: failed to compile kernel: ", log);
}

void MetalTransposeOp::execute() {
    OPENVINO_ASSERT(!inputs().empty(), "Transpose: missing input");
    MetalTensor* src = inputs()[0];
    OPENVINO_ASSERT(src && src->buf.valid(), "Transpose: input buffer null");
    MetalTensor& dst = require_output();

    ov::Shape in_shape = !src->shape.empty()
                             ? src->shape
                             : ov::Shape(m_desc.transpose.in_shape.begin(), m_desc.transpose.in_shape.end());
    ov::Shape out_shape = !dst.shape.empty()
                              ? dst.shape
                              : ov::Shape(m_desc.transpose.out_shape.begin(), m_desc.transpose.out_shape.end());
    OPENVINO_ASSERT(!in_shape.empty() && !out_shape.empty(), "Transpose: shapes unknown");

    if (!dst.buf.valid()) {
        size_t bytes = ov::shape_size(out_shape) * m_element_type.size();
        dst.buf = buffer_manager()->allocate(bytes, m_element_type, /*persistent=*/false, /*storageModePrivate=*/true);
    }
    dst.shape = out_shape;
    dst.expected_type = m_element_type;

    // Precompute strides for input.
    std::vector<uint32_t> in_stride(in_shape.size(), 1);
    for (int i = static_cast<int>(in_shape.size()) - 2; i >= 0; --i) {
        in_stride[static_cast<size_t>(i)] =
            in_stride[static_cast<size_t>(i + 1)] * static_cast<uint32_t>(in_shape[static_cast<size_t>(i + 1)]);
    }
    std::vector<uint32_t> out_shape_u(out_shape.begin(), out_shape.end());
    std::vector<uint32_t> perm_u;
    perm_u.reserve(m_desc.transpose.perm.size());
    for (auto p : m_desc.transpose.perm) perm_u.push_back(static_cast<uint32_t>(p));

    const uint32_t num_elems = static_cast<uint32_t>(ov::shape_size(out_shape));
    const uint32_t rank = static_cast<uint32_t>(out_shape.size());

    if (!m_queue) {
        m_queue = [m_device newCommandQueue];
    }
    id<MTLCommandBuffer> cb = [m_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m_pipeline];
    [enc setBuffer:to_mtl(src->buf) offset:0 atIndex:0];
    [enc setBuffer:to_mtl(dst.buf) offset:0 atIndex:1];
    const size_t elem_sz = m_element_type.size();
    const size_t need_src = ov::shape_size(in_shape) * elem_sz;
    const size_t need_dst = ov::shape_size(out_shape) * elem_sz;
    OPENVINO_ASSERT(src->buf.size >= need_src, "Transpose: src buffer too small");
    OPENVINO_ASSERT(dst.buf.size >= need_dst, "Transpose: dst buffer too small");
    [enc setBytes:&num_elems length:sizeof(num_elems) atIndex:2];
    [enc setBytes:&rank length:sizeof(rank) atIndex:3];
    [enc setBytes:out_shape_u.data() length:out_shape_u.size() * sizeof(uint32_t) atIndex:4];
    [enc setBytes:perm_u.data() length:perm_u.size() * sizeof(uint32_t) atIndex:5];
    [enc setBytes:in_stride.data() length:in_stride.size() * sizeof(uint32_t) atIndex:6];

    MTLSize grid = MTLSizeMake(num_elems, 1, 1);
    MTLSize tg = MTLSizeMake(64, 1, 1);

    start_profiling();
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    stop_profiling_ms();
}

}  // namespace metal_plugin
}  // namespace ov
