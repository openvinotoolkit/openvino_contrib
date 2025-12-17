// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "runtime/metal_op_convert.hpp"

#include "kernel_codegen/metal_kernel_compiler.hpp"
#include "runtime/metal_dtype.hpp"
#include "runtime/metal_logger.hpp"

namespace ov {
namespace metal_plugin {

namespace {
inline id<MTLBuffer> to_mtl(const MetalBuffer& buf) {
    return (__bridge id<MTLBuffer>)buf.buffer;
}

}  // namespace

MetalConvertOp::MetalConvertOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "Convert",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
    m_device((id<MTLDevice>)device),
    m_queue((id<MTLCommandQueue>)queue) {
    auto cvt = ov::as_type_ptr<const ov::op::v0::Convert>(node);
    OPENVINO_ASSERT(cvt, "MetalConvertOp expects v0::Convert");
    m_src_type = cvt->get_input_element_type(0);
    m_dst_type = cvt->get_output_element_type(0);
    m_desc.kind = KernelOpKind::Convert;
    m_desc.convert.src_dtype = resolve_metal_dtype(m_src_type);
    m_desc.convert.dst_dtype = resolve_metal_dtype(m_dst_type);
    m_desc.dtype = m_desc.convert.dst_dtype;
}

void MetalConvertOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
    MetalKernelCompiler compiler(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    m_pipeline = compiler.compile_convert_kernel(m_desc, log);
    OPENVINO_ASSERT(m_pipeline, "MetalConvertOp: failed to compile kernel: ", log);
}

void MetalConvertOp::execute() {
    OPENVINO_ASSERT(!inputs().empty(), "Convert: missing input");
    MetalTensor* src = inputs()[0];
    OPENVINO_ASSERT(src && src->buf.valid(), "Convert: input buffer null");
    MetalTensor& dst = require_output();

    ov::Shape out_shape = !dst.shape.empty() ? dst.shape : output_shape();
    if (out_shape.empty())
        out_shape = src->shape;
    OPENVINO_ASSERT(!out_shape.empty(), "Convert: unknown shape");

    const size_t num_elems = ov::shape_size(out_shape);
    if (!dst.buf.valid()) {
        size_t bytes = num_elems * m_dst_type.size();
        dst.buf = buffer_manager()->allocate(bytes, m_dst_type, /*persistent=*/false, /*storageModePrivate=*/true);
    }
    dst.shape = out_shape;
    dst.expected_type = m_dst_type;

    if (m_src_type == m_dst_type) {
        // Trivial case: reuse buffer.
        dst.buf = src->buf;
        start_profiling();
        stop_profiling_ms();
        return;
    }

    if (!m_queue) {
        m_queue = [m_device newCommandQueue];
    }
    id<MTLCommandBuffer> cb = [m_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m_pipeline];
    [enc setBuffer:to_mtl(src->buf) offset:0 atIndex:0];
    [enc setBuffer:to_mtl(dst.buf) offset:0 atIndex:1];
    uint32_t n = static_cast<uint32_t>(num_elems);
    [enc setBytes:&n length:sizeof(n) atIndex:2];

    MTLSize grid = MTLSizeMake(n, 1, 1);
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
