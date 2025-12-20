// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "runtime/metal_op_pad.hpp"

#include <limits>

#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/pad.hpp"
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

std::vector<int> make_strides(const ov::Shape& shp) {
    if (shp.empty()) return {1};
    std::vector<int> st(shp.size(), 1);
    for (int i = static_cast<int>(shp.size()) - 2; i >= 0; --i) {
        st[i] = st[i + 1] * static_cast<int>(shp[i + 1]);
    }
    return st;
}
}  // namespace

MetalPadOp::MetalPadOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "Pad",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_node(node),
      m_element_type(node->get_output_element_type(0)),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {}

void MetalPadOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalPadOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }

    MetalKernelCompiler compiler(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    mlir::MLIRContext ctx;
    auto module = build_mlir_pad_from_model(make_single_op_model(m_node), ctx);
    PadCodegenDesc desc{};
    desc.element_type = m_element_type;
    auto source = generate_msl_from_mlir(module, desc);
    m_pipeline = compiler.compile_msl_from_source(source, "pad_kernel", log);
    OPENVINO_ASSERT(m_pipeline, "MetalPadOp: failed to compile pad kernel: ", log);

    auto in_shape = m_node->get_input_shape(0);
    auto out_shape = output_shape();
    m_num_elems = static_cast<uint32_t>(ov::shape_size(out_shape));

    m_in_dims.assign(in_shape.begin(), in_shape.end());
    if (m_in_dims.empty()) m_in_dims.push_back(1);
    m_out_dims.assign(out_shape.begin(), out_shape.end());
    if (m_out_dims.empty()) m_out_dims.push_back(1);

    m_in_strides = make_strides(in_shape);
    if (m_in_strides.empty()) m_in_strides.push_back(1);
    m_out_strides = make_strides(out_shape);
    if (m_out_strides.empty()) m_out_strides.push_back(1);

    auto pads_begin_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(
        m_node->input_value(1).get_node_shared_ptr());
    auto pads_end_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(
        m_node->input_value(2).get_node_shared_ptr());
    OPENVINO_ASSERT(pads_begin_const && pads_end_const, "Pad requires constant pads");
    auto pads_begin = pads_begin_const->cast_vector<int64_t>();
    auto pads_end = pads_end_const->cast_vector<int64_t>();
    OPENVINO_ASSERT(pads_begin.size() == m_in_dims.size(), "Pad: pads_begin rank mismatch");
    OPENVINO_ASSERT(pads_end.size() == m_in_dims.size(), "Pad: pads_end rank mismatch");
    m_pads_begin.resize(m_in_dims.size());
    for (size_t i = 0; i < pads_begin.size(); ++i) {
        OPENVINO_ASSERT(pads_begin[i] >= 0 && pads_end[i] >= 0, "Pad: negative pads not supported");
        m_pads_begin[i] = static_cast<int>(pads_begin[i]);
    }

    m_pad_value = 0.0f;
    if (m_node->get_input_size() >= 4) {
        auto pad_val_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(
            m_node->input_value(3).get_node_shared_ptr());
        OPENVINO_ASSERT(pad_val_const, "Pad: pad_value must be constant");
        if (pad_val_const->get_element_type() == ov::element::f16 ||
            pad_val_const->get_element_type() == ov::element::f32) {
            auto vals = pad_val_const->cast_vector<float>();
            OPENVINO_ASSERT(!vals.empty(), "Pad: pad_value constant is empty");
            m_pad_value = vals[0];
        } else {
            auto vals = pad_val_const->cast_vector<int64_t>();
            OPENVINO_ASSERT(!vals.empty(), "Pad: pad_value constant is empty");
            m_pad_value = static_cast<float>(vals[0]);
        }
    }

    MetalOp::compile(buffer_manager);
}

void MetalPadOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(inputs().size() >= 1, "Pad: missing input");
    MetalTensor* src = inputs()[0];
    OPENVINO_ASSERT(src && src->buf.valid(), "Pad: input buffer null");

    MetalTensor& dst = require_output();
    ov::Shape in_shape = !src->shape.empty() ? src->shape : m_node->get_input_shape(0);
    OPENVINO_ASSERT(!in_shape.empty(), "Pad: input shape unknown");
    if (m_node->get_input_partial_shape(0).is_static()) {
        OPENVINO_ASSERT(in_shape == m_node->get_input_shape(0),
                        "Pad: runtime input shape mismatch");
    }
    ov::Shape out_shape = !dst.shape.empty() ? dst.shape : output_shape();
    if (out_shape.empty() && m_node->get_output_partial_shape(0).is_static()) {
        out_shape = m_node->get_output_shape(0);
    }
    OPENVINO_ASSERT(!out_shape.empty(), "Pad: output shape unknown");

    m_num_elems = static_cast<uint32_t>(ov::shape_size(out_shape));
    m_in_dims.assign(in_shape.begin(), in_shape.end());
    if (m_in_dims.empty()) m_in_dims.push_back(1);
    m_out_dims.assign(out_shape.begin(), out_shape.end());
    if (m_out_dims.empty()) m_out_dims.push_back(1);
    m_in_strides = make_strides(in_shape);
    if (m_in_strides.empty()) m_in_strides.push_back(1);
    m_out_strides = make_strides(out_shape);
    if (m_out_strides.empty()) m_out_strides.push_back(1);

    const size_t in_bytes = m_element_type.size() * ov::shape_size(in_shape);
    OPENVINO_ASSERT(src->buf.size >= in_bytes, "Pad: input buffer too small");
    const size_t bytes = m_element_type.size() * static_cast<size_t>(m_num_elems);
    if (!dst.buf.valid() || dst.buf.size < bytes) {
        dst.buf = buffer_manager()->allocate(bytes, m_element_type, /*persistent=*/false, dst.prefer_private);
    }
    dst.expected_type = m_element_type;
    dst.shape = out_shape;

    id<MTLCommandBuffer> cb = static_cast<id<MTLCommandBuffer>>(cmd_buf_handle);
    OPENVINO_ASSERT(cb, "MetalPadOp: command buffer is null");
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m_pipeline];

    [enc setBuffer:to_mtl(src->buf) offset:0 atIndex:0];
    [enc setBuffer:to_mtl(dst.buf) offset:0 atIndex:1];
    uint32_t num = m_num_elems;
    uint32_t rank = static_cast<uint32_t>(m_out_dims.size());
    [enc setBytes:&num length:sizeof(num) atIndex:2];
    [enc setBytes:&rank length:sizeof(rank) atIndex:3];
    [enc setBytes:m_out_dims.data() length:m_out_dims.size() * sizeof(int) atIndex:4];
    [enc setBytes:m_in_dims.data() length:m_in_dims.size() * sizeof(int) atIndex:5];
    [enc setBytes:m_out_strides.data() length:m_out_strides.size() * sizeof(int) atIndex:6];
    [enc setBytes:m_in_strides.data() length:m_in_strides.size() * sizeof(int) atIndex:7];
    [enc setBytes:m_pads_begin.data() length:m_pads_begin.size() * sizeof(int) atIndex:8];
    float pad_val = m_pad_value;
    [enc setBytes:&pad_val length:sizeof(pad_val) atIndex:9];

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
