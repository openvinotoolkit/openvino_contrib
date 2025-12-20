// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "runtime/metal_op_range.hpp"

#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/range.hpp"
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

template <typename T>
T get_scalar_from_const(const ov::Output<ov::Node>& input) {
    auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(input.get_node_shared_ptr());
    OPENVINO_ASSERT(c, "Range expects Constant inputs for start/stop/step");
    auto v = c->cast_vector<T>();
    OPENVINO_ASSERT(!v.empty(), "Range constant is empty");
    return v[0];
}
}  // namespace

MetalRangeOp::MetalRangeOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "Range",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_node(node),
      m_element_type(node->get_output_element_type(0)),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {}

void MetalRangeOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalRangeOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }

    MetalKernelCompiler compiler(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    mlir::MLIRContext ctx;
    auto module = build_mlir_range_from_model(make_single_op_model(m_node), ctx);
    RangeCodegenDesc desc{};
    desc.element_type = m_element_type;
    auto source = generate_msl_from_mlir(module, desc);
    m_pipeline = compiler.compile_msl_from_source(source, "range_kernel", log);
    OPENVINO_ASSERT(m_pipeline, "MetalRangeOp: failed to compile range kernel: ", log);

    if (m_node->get_output_partial_shape(0).is_static()) {
        m_num_elems = static_cast<uint32_t>(ov::shape_size(m_node->get_output_shape(0)));
    }

    // Read constant start/step.
    if (m_element_type == ov::element::i32) {
        m_start = static_cast<double>(get_scalar_from_const<int32_t>(m_node->input_value(0)));
        m_step = static_cast<double>(get_scalar_from_const<int32_t>(m_node->input_value(2)));
    } else if (m_element_type == ov::element::i64) {
        m_start = static_cast<double>(get_scalar_from_const<int64_t>(m_node->input_value(0)));
        m_step = static_cast<double>(get_scalar_from_const<int64_t>(m_node->input_value(2)));
    } else if (m_element_type == ov::element::f16) {
        m_start = static_cast<double>(get_scalar_from_const<ov::float16>(m_node->input_value(0)));
        m_step = static_cast<double>(get_scalar_from_const<ov::float16>(m_node->input_value(2)));
    } else {
        m_start = static_cast<double>(get_scalar_from_const<float>(m_node->input_value(0)));
        m_step = static_cast<double>(get_scalar_from_const<float>(m_node->input_value(2)));
    }

    MetalOp::compile(buffer_manager);
}

void MetalRangeOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    MetalTensor& dst = require_output();
    ov::Shape out_shape = !dst.shape.empty() ? dst.shape : output_shape();
    if (out_shape.empty() && m_node->get_output_partial_shape(0).is_static()) {
        out_shape = m_node->get_output_shape(0);
    }
    if (m_node->get_output_partial_shape(0).is_static()) {
        OPENVINO_ASSERT(out_shape == m_node->get_output_shape(0),
                        "Range: runtime output shape mismatch");
    }
    OPENVINO_ASSERT(!out_shape.empty(), "Range: output shape unknown");
    m_num_elems = static_cast<uint32_t>(ov::shape_size(out_shape));
    const size_t bytes = m_element_type.size() * static_cast<size_t>(m_num_elems);
    if (!dst.buf.valid() || dst.buf.size < bytes) {
        dst.buf = buffer_manager()->allocate(bytes, m_element_type, /*persistent=*/false, dst.prefer_private);
    }
    dst.expected_type = m_element_type;
    dst.shape = out_shape;

    id<MTLCommandBuffer> cb = static_cast<id<MTLCommandBuffer>>(cmd_buf_handle);
    OPENVINO_ASSERT(cb, "MetalRangeOp: command buffer is null");
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m_pipeline];

    [enc setBuffer:to_mtl(dst.buf) offset:0 atIndex:0];
    uint32_t num = m_num_elems;
    [enc setBytes:&num length:sizeof(num) atIndex:1];

    if (m_element_type == ov::element::i32) {
        int32_t start = static_cast<int32_t>(m_start);
        int32_t step = static_cast<int32_t>(m_step);
        [enc setBytes:&start length:sizeof(start) atIndex:2];
        [enc setBytes:&step length:sizeof(step) atIndex:3];
    } else if (m_element_type == ov::element::i64) {
        int64_t start = static_cast<int64_t>(m_start);
        int64_t step = static_cast<int64_t>(m_step);
        [enc setBytes:&start length:sizeof(start) atIndex:2];
        [enc setBytes:&step length:sizeof(step) atIndex:3];
    } else if (m_element_type == ov::element::f16) {
        ov::float16 start = static_cast<ov::float16>(m_start);
        ov::float16 step = static_cast<ov::float16>(m_step);
        [enc setBytes:&start length:sizeof(start) atIndex:2];
        [enc setBytes:&step length:sizeof(step) atIndex:3];
    } else {
        float start = static_cast<float>(m_start);
        float step = static_cast<float>(m_step);
        [enc setBytes:&start length:sizeof(start) atIndex:2];
        [enc setBytes:&step length:sizeof(step) atIndex:3];
    }

    if (num == 0) {
        [enc endEncoding];
        return;
    }
    const NSUInteger threads = 64;
    MTLSize grid = MTLSizeMake(num, 1, 1);
    const NSUInteger tg_size = static_cast<NSUInteger>(metal_clamp_tg_size((void*)m_pipeline, threads));
    MTLSize tg = MTLSizeMake(tg_size, 1, 1);
    start_profiling(enc);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    stop_profiling_ms(enc);
    [enc endEncoding];
    dst.expected_type = m_element_type;
}

}  // namespace metal_plugin
}  // namespace ov
