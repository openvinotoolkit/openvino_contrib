// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "runtime/metal_op_activations.hpp"

#include <cmath>
#include <limits>

#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/parameter.hpp"
#include "runtime/metal_logger.hpp"

namespace ov {
namespace metal_plugin {

namespace {
inline id<MTLBuffer> to_mtl(const MetalBuffer& buf) {
    return (__bridge id<MTLBuffer>)buf.buffer;
}
}  // namespace

MetalActivationOp::MetalActivationOp(const std::shared_ptr<const ov::Node>& node,
                                     ActivationKind kind,
                                     float alpha,
                                     void* device,
                                     void* queue)
    : MetalOp(node->get_friendly_name(),
              "Activation",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_kind(kind),
      m_alpha(alpha),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    OPENVINO_ASSERT(node->get_input_size() == 1, "Activation expects single input");
    OPENVINO_ASSERT(node->get_output_element_type(0) == ov::element::f32, "Activation supports only f32 for now");
}

void MetalActivationOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);

    MetalKernelCompiler compiler(m_device);
    KernelOp op{};
    op.kind = KernelOpKind::Unary;
    op.activation = m_kind;
    op.alpha = m_alpha;

    std::string log;
    m_pipeline = compiler.compile_unary_kernel(op, log);
    OPENVINO_ASSERT(m_pipeline, "MetalActivationOp: failed to compile unary kernel: ", log);
}

void MetalActivationOp::execute() {
    OPENVINO_ASSERT(!inputs().empty(), "Activation: no inputs bound");
    MetalTensor* src = inputs().at(0);
    OPENVINO_ASSERT(src && src->buf.valid(), "Activation: input buffer is null");
    MetalTensor& dst = require_output();

    if (!dst.buf.valid()) {
        auto et = ov::element::f32;
        size_t bytes = et.size();
        for (auto d : output_shape()) bytes *= d;
        // Keep output host-visible to avoid CPU copies in tests.
        dst.buf = buffer_manager()->allocate(bytes, et, /*persistent=*/false, /*storageModePrivate=*/false);
        dst.expected_type = et;
    }

    if (!m_queue) {
        m_queue = [m_device newCommandQueue];
    }
    id<MTLCommandBuffer> cb = [m_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m_pipeline];

    [enc setBuffer:to_mtl(src->buf) offset:0 atIndex:0];
    [enc setBuffer:to_mtl(dst.buf) offset:0 atIndex:1];

    const NSUInteger elems = static_cast<NSUInteger>(shape_size(output_shape()));
    OPENVINO_ASSERT(elems <= std::numeric_limits<uint32_t>::max(),
                    "Activation: element count exceeds uint32 range");
    const uint32_t num_elems = static_cast<uint32_t>(elems);
    [enc setBytes:&num_elems length:sizeof(num_elems) atIndex:2];
    MTLSize grid = MTLSizeMake(elems, 1, 1);
    MTLSize tg = MTLSizeMake(64, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];

    start_profiling();
    [cb commit];
    [cb waitUntilCompleted];
    stop_profiling_ms();

    dst.shape = output_shape();
    dst.expected_type = ov::element::f32;
}

}  // namespace metal_plugin
}  // namespace ov
