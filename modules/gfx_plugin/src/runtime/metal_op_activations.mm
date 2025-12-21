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
#include "runtime/metal_op_utils.hpp"
#include "runtime/metal_logger.hpp"
#include "mlir_builder.hpp"
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

MetalActivationOp::MetalActivationOp(const std::shared_ptr<const ov::Node>& node,
                                     ActivationKind kind,
                                     float alpha,
                                     double clamp_min,
                                     double clamp_max,
                                     void* device,
                                     void* queue)
    : MetalOp(node->get_friendly_name(),
              "Activation",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_kind(kind),
      m_alpha(alpha),
      m_clamp_min(clamp_min),
      m_clamp_max(clamp_max),
      m_node(node),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    OPENVINO_ASSERT(node->get_input_size() == 1, "Activation expects single input");
    m_element_type = node->get_output_element_type(0);
    const bool is_float = (m_element_type == ov::element::f32 || m_element_type == ov::element::f16);
    const bool is_int = m_element_type.is_integral_number();
    const bool is_bool = (m_element_type == ov::element::boolean);
    if (m_kind == ActivationKind::Sign || m_kind == ActivationKind::Abs) {
        OPENVINO_ASSERT(is_float || is_int, "Activation supports only f16/f32 or integer for Sign/Abs");
    } else if (m_kind == ActivationKind::LogicalNot) {
        OPENVINO_ASSERT(is_int || is_bool, "LogicalNot supports only integer/bool types");
    } else {
        OPENVINO_ASSERT(is_float, "Activation supports only f16/f32");
    }
}

void MetalActivationOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalActivationOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }

    MetalKernelCompiler compiler(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    mlir::MLIRContext ctx;
    std::optional<std::pair<double, double>> clamp_range;
    if (m_kind == ActivationKind::Clamp) {
        clamp_range = std::make_pair(m_clamp_min, m_clamp_max);
    }
    auto module = build_mlir_unary_from_node(m_node, ctx, m_kind, m_alpha, clamp_range);
    UnaryCodegenDesc desc;
    desc.activation = m_kind;
    desc.alpha = m_alpha;
    desc.element_type = m_element_type == ov::element::dynamic ? ov::element::f32 : m_element_type;
    desc.clamp_min = m_clamp_min;
    desc.clamp_max = m_clamp_max;
    auto source = generate_msl_from_mlir(module, desc);
    m_pipeline = compiler.compile_msl_from_source(source, "unary_kernel", log);
    OPENVINO_ASSERT(m_pipeline, "MetalActivationOp: failed to compile unary kernel: ", log);

    MetalOp::compile(buffer_manager);
}

void MetalActivationOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    if (!is_compiled()) {
        compile(buffer_manager());
    }
    OPENVINO_ASSERT(!inputs().empty(), "Activation: no inputs bound");
    MetalTensor* src = inputs().at(0);
    OPENVINO_ASSERT(src && src->buf.valid(), "Activation: input buffer is null");
    MetalTensor& dst = require_output();

    ov::Shape out_shape = !output_shape().empty() ? output_shape() : src->shape;
    OPENVINO_ASSERT(!out_shape.empty(), "Activation: output shape unknown");
    if (!src->shape.empty() && m_node->get_input_partial_shape(0).is_static()) {
        OPENVINO_ASSERT(src->shape == m_node->get_input_shape(0),
                        "Activation: runtime input shape mismatch");
    }
    const size_t bytes = m_element_type.size() * ov::shape_size(out_shape);
    const size_t src_bytes = m_element_type.size() * ov::shape_size(src->shape.empty() ? out_shape : src->shape);
    OPENVINO_ASSERT(src->buf.size >= src_bytes, "Activation: input buffer too small");
    if (!dst.buf.valid() || dst.buf.size < bytes) {
        // Keep output device-only; CPU-visible outputs are not supported.
        dst.buf = buffer_manager()->allocate(bytes, m_element_type, /*persistent=*/false, dst.prefer_private);
        dst.expected_type = m_element_type;
    }

    id<MTLCommandBuffer> cb = static_cast<id<MTLCommandBuffer>>(cmd_buf_handle);
    OPENVINO_ASSERT(cb, "MetalActivationOp: command buffer is null");
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m_pipeline];

    [enc setBuffer:to_mtl(src->buf) offset:0 atIndex:0];
    [enc setBuffer:to_mtl(dst.buf) offset:0 atIndex:1];

    const NSUInteger elems = static_cast<NSUInteger>(shape_size(out_shape));
    OPENVINO_ASSERT(elems <= std::numeric_limits<uint32_t>::max(),
                    "Activation: element count exceeds uint32 range");
    const uint32_t num_elems = static_cast<uint32_t>(elems);
    [enc setBytes:&num_elems length:sizeof(num_elems) atIndex:2];
    if (num_elems == 0) {
        [enc endEncoding];
        return;
    }
    MTLSize grid = MTLSizeMake(elems, 1, 1);
    const NSUInteger tg_size = static_cast<NSUInteger>(metal_clamp_tg_size((void*)m_pipeline, 64));
    MTLSize tg = MTLSizeMake(tg_size, 1, 1);
    start_profiling(enc);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    stop_profiling_ms(enc);
    [enc endEncoding];

    dst.shape = out_shape;
    dst.expected_type = m_element_type;
}

}  // namespace gfx_plugin
}  // namespace ov
