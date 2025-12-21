// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "runtime/metal_op_shapeof.hpp"

#include "openvino/core/shape_util.hpp"
#include "kernel_codegen/metal_kernel_compiler.hpp"
#include "mlir/mlir_builder.hpp"
#include "runtime/metal_logger.hpp"
#include "runtime/metal_op_utils.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir_codegen/codegen_common.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
inline id<MTLBuffer> to_mtl(const MetalBuffer& buf) {
    return (__bridge id<MTLBuffer>)buf.buffer;
}
}  // namespace

MetalShapeOfOp::MetalShapeOfOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "ShapeOf",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_node(node),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    m_element_type = node->get_output_element_type(0);
    OPENVINO_ASSERT(m_element_type == ov::element::i32 || m_element_type == ov::element::i64,
                    "ShapeOf: output must be i32/i64");
    const auto in_pshape = node->get_input_partial_shape(0);
    OPENVINO_ASSERT(in_pshape.rank().is_static(), "ShapeOf: input rank must be static");
    const size_t rank = static_cast<size_t>(in_pshape.rank().get_length());
    m_shape_vals.assign(rank, 0);
    if (in_pshape.is_static()) {
        const auto in_shape = node->get_input_shape(0);
        m_shape_vals.assign(in_shape.begin(), in_shape.end());
    }
    m_desc.rank = static_cast<uint32_t>(rank);
    m_desc.element_type = m_element_type;
}

void MetalShapeOfOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalShapeOfOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }
    MetalKernelCompiler compiler(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    mlir::MLIRContext ctx;
    auto module = build_mlir_shapeof_from_model(make_single_op_model(m_node), ctx);
    auto source = generate_msl_from_mlir(module, m_desc);
    m_pipeline = compiler.compile_msl_from_source(source, "shapeof_kernel", log);
    OPENVINO_ASSERT(m_pipeline, "MetalShapeOfOp: failed to compile kernel: ", log);

    MetalOp::compile(buffer_manager);
}

void MetalShapeOfOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(!inputs().empty(), "ShapeOf: missing input");
    MetalTensor* src = inputs()[0];
    OPENVINO_ASSERT(src && src->buf.valid(), "ShapeOf: input buffer null");

    std::vector<int64_t> shape_vals;
    if (!src->shape.empty()) {
        shape_vals.assign(src->shape.begin(), src->shape.end());
    } else {
        shape_vals = m_shape_vals;
    }
    OPENVINO_ASSERT(!shape_vals.empty(), "ShapeOf: input shape is unknown");

    MetalTensor& dst = require_output();
    ov::Shape out_shape = ov::Shape{shape_vals.size()};
    const size_t bytes = ov::shape_size(out_shape) * m_element_type.size();
    if (!dst.buf.valid() || dst.buf.size < bytes) {
        dst.buf = buffer_manager()->allocate(bytes, m_element_type, /*persistent=*/false, dst.prefer_private);
    }
    dst.shape = out_shape;
    dst.expected_type = m_element_type;

    const uint32_t rank = static_cast<uint32_t>(shape_vals.size());

    id<MTLCommandBuffer> cb = static_cast<id<MTLCommandBuffer>>(cmd_buf_handle);
    OPENVINO_ASSERT(cb, "MetalShapeOfOp: command buffer is null");
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m_pipeline];
    [enc setBuffer:to_mtl(src->buf) offset:0 atIndex:0];
    [enc setBuffer:to_mtl(dst.buf) offset:0 atIndex:1];
    [enc setBytes:&rank length:sizeof(rank) atIndex:2];

    if (m_element_type == ov::element::i32) {
        std::vector<int32_t> tmp(rank);
        for (size_t i = 0; i < shape_vals.size(); ++i) tmp[i] = static_cast<int32_t>(shape_vals[i]);
        [enc setBytes:tmp.data() length:tmp.size() * sizeof(int32_t) atIndex:3];
    } else {
        [enc setBytes:shape_vals.data() length:shape_vals.size() * sizeof(int64_t) atIndex:3];
    }

    if (rank == 0) {
        [enc endEncoding];
        return;
    }
    MTLSize grid = MTLSizeMake(rank, 1, 1);
    const NSUInteger tg_size = static_cast<NSUInteger>(metal_clamp_tg_size((void*)m_pipeline, 64));
    MTLSize tg = MTLSizeMake(tg_size, 1, 1);

    start_profiling(enc);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    stop_profiling_ms(enc);
    [enc endEncoding];
}

}  // namespace gfx_plugin
}  // namespace ov
