// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "runtime/metal_op_reverse.hpp"

#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "kernel_codegen/metal_kernel_compiler.hpp"
#include "mlir/mlir_builder.hpp"
#include "runtime/metal_logger.hpp"
#include "runtime/metal_op_utils.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

namespace ov {
namespace metal_plugin {

namespace {
inline id<MTLBuffer> to_mtl(const MetalBuffer& buf) {
    return (__bridge id<MTLBuffer>)buf.buffer;
}

std::vector<int64_t> get_axes_from_const(const std::shared_ptr<const ov::op::v1::Reverse>& node,
                                        const ov::Shape& in_shape) {
    auto axes_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(
        node->input_value(1).get_node_shared_ptr());
    OPENVINO_ASSERT(axes_const, "Reverse: axes input must be constant");

    std::vector<int64_t> axes;
    if (node->get_mode() == ov::op::v1::Reverse::Mode::MASK) {
        auto mask = axes_const->cast_vector<int64_t>();
        OPENVINO_ASSERT(mask.size() == in_shape.size(), "Reverse: mask rank mismatch");
        for (size_t i = 0; i < mask.size(); ++i) {
            if (mask[i])
                axes.push_back(static_cast<int64_t>(i));
        }
    } else {
        axes = axes_const->cast_vector<int64_t>();
    }
    return axes;
}

}  // namespace

MetalReverseOp::MetalReverseOp(const std::shared_ptr<const ov::op::v1::Reverse>& node,
                               MetalDeviceHandle device,
                               MetalCommandQueueHandle queue)
    : MetalOp(node->get_friendly_name(),
              "Reverse",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_node(node),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    OPENVINO_ASSERT(m_node, "MetalReverseOp: node is null");
    parse_axes();
}

void MetalReverseOp::parse_axes() {
    const auto& in_shape = m_node->get_input_shape(0);
    m_element_type = m_node->get_output_element_type(0);
    OPENVINO_ASSERT(!in_shape.empty(), "Reverse: input shape is empty");
    OPENVINO_ASSERT(in_shape.size() <= ReverseCodegenDesc::kMaxDims,
                    "Reverse: rank exceeds supported maximum");

    m_desc.rank = static_cast<uint32_t>(in_shape.size());
    m_desc.total = static_cast<uint32_t>(ov::shape_size(in_shape));
    m_desc.axes_mask = 0;

    std::vector<int64_t> axes = get_axes_from_const(m_node, in_shape);
    for (auto& a : axes) {
        if (a < 0)
            a += static_cast<int64_t>(in_shape.size());
        OPENVINO_ASSERT(a >= 0 && a < static_cast<int64_t>(in_shape.size()), "Reverse: axis out of range");
        m_desc.axes_mask |= (1u << static_cast<uint32_t>(a));
    }

    uint32_t stride = 1;
    for (size_t i = in_shape.size(); i-- > 0;) {
        m_desc.dims[i] = static_cast<uint32_t>(in_shape[i]);
        m_desc.strides[i] = stride;
        stride *= static_cast<uint32_t>(in_shape[i]);
    }
}

void MetalReverseOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalReverseOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }
    MetalKernelCompiler compiler(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    mlir::MLIRContext ctx;
    auto module = build_mlir_reverse_from_model(make_single_op_model(m_node), ctx);
    ReverseCodegenDesc desc = m_desc;
    desc.element_type = m_element_type == ov::element::dynamic ? ov::element::f32 : m_element_type;
    auto source = generate_msl_from_mlir(module, desc);
    m_pipeline = compiler.compile_msl_from_source(source, "reverse_kernel", log);
    OPENVINO_ASSERT(m_pipeline, "MetalReverseOp: failed to compile kernel: ", log);

    MetalOp::compile(buffer_manager);
}

void MetalReverseOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(!inputs().empty(), "Reverse: missing input");
    MetalTensor* src = inputs()[0];
    OPENVINO_ASSERT(src && src->buf.valid(), "Reverse: input buffer is null");
    MetalTensor& dst = require_output();

    ov::Shape in_shape = !src->shape.empty() ? src->shape : m_node->get_input_shape(0);
    OPENVINO_ASSERT(!in_shape.empty(), "Reverse: input shape unknown");
    if (m_node->get_input_partial_shape(0).is_static()) {
        OPENVINO_ASSERT(in_shape == m_node->get_input_shape(0),
                        "Reverse: runtime input shape mismatch");
    }
    const ov::Shape out_shape = !output_shape().empty() ? output_shape() : m_node->get_output_shape(0);
    const size_t num_elems = ov::shape_size(out_shape);

    const size_t in_bytes = ov::shape_size(in_shape) * m_element_type.size();
    OPENVINO_ASSERT(src->buf.size >= in_bytes, "Reverse: input buffer too small");
    if (!dst.buf.valid() || dst.buf.size < num_elems * m_element_type.size()) {
        size_t bytes = num_elems * m_element_type.size();
        dst.buf = buffer_manager()->allocate(bytes,
                                             m_element_type,
                                             /*persistent=*/false,
                                             /*storageModePrivate=*/dst.prefer_private);
    }
    dst.shape = out_shape;
    dst.expected_type = m_element_type;

    id<MTLCommandBuffer> cb = static_cast<id<MTLCommandBuffer>>(cmd_buf_handle);
    OPENVINO_ASSERT(cb, "MetalReverseOp: command buffer is null");
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m_pipeline];
    [enc setBuffer:to_mtl(src->buf) offset:0 atIndex:0];
    [enc setBuffer:to_mtl(dst.buf) offset:0 atIndex:1];
    [enc setBytes:&m_desc length:sizeof(m_desc) atIndex:2];

    if (m_desc.total == 0) {
        [enc endEncoding];
        return;
    }
    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(m_desc.total), 1, 1);
    const NSUInteger tg_size = static_cast<NSUInteger>(metal_clamp_tg_size((void*)m_pipeline, 256));
    MTLSize tg = MTLSizeMake(tg_size, 1, 1);

    start_profiling(enc);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    stop_profiling_ms(enc);
    [enc endEncoding];
}

}  // namespace metal_plugin
}  // namespace ov
