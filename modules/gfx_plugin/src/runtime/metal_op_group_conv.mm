// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "runtime/metal_op_group_conv.hpp"

#include <numeric>
#include <cstdint>
#include <string>

#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "runtime/metal_logger.hpp"
#include "runtime/metal_memory.hpp"
#include "runtime/metal_op_utils.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir_codegen/codegen_common.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

inline id<MTLBuffer> to_mtl(const MetalBuffer& buf) {
    return (__bridge id<MTLBuffer>)buf.buffer;
}

inline size_t element_size(const ov::element::Type& t) {
    return t.size();
}

}  // namespace

MetalGroupConvOp::MetalGroupConvOp(const std::shared_ptr<const ov::op::v1::GroupConvolution>& node,
                                   MetalDeviceHandle device,
                                   MetalCommandQueueHandle queue)
    : MetalOp(node->get_friendly_name(),
              "GroupConv2D",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_node(node),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    OPENVINO_ASSERT(m_node, "MetalGroupConvOp: node is null");

    const auto strides = m_node->get_strides();
    const auto dilations = m_node->get_dilations();
    const auto pads_begin = m_node->get_pads_begin();
    const auto pads_end = m_node->get_pads_end();

    const auto& in_shape = m_node->get_input_shape(0);   // NCHW
    const auto& w_shape = m_node->get_input_shape(1);    // GOIHW

    OPENVINO_ASSERT(in_shape.size() == 4, "MetalGroupConvOp: expected NCHW input");
    OPENVINO_ASSERT(w_shape.size() == 5, "MetalGroupConvOp: expected GOIHW weights");

    m_element_type = m_node->get_output_element_type(0);
    m_desc.element_type = m_element_type;
    m_desc.N = static_cast<uint32_t>(in_shape.at(0));
    m_desc.C_in = static_cast<uint32_t>(in_shape.at(1));
    m_desc.H = static_cast<uint32_t>(in_shape.at(2));
    m_desc.W = static_cast<uint32_t>(in_shape.at(3));

    const uint32_t groups = static_cast<uint32_t>(w_shape.at(0));
    const uint32_t c_out_pg = static_cast<uint32_t>(w_shape.at(1));
    const uint32_t c_in_pg = static_cast<uint32_t>(w_shape.at(2));
    m_desc.groups = groups;
    m_desc.C_out_pg = c_out_pg;
    m_desc.C_in_pg = c_in_pg;
    m_desc.C_out = groups * c_out_pg;
    OPENVINO_ASSERT(m_desc.C_in == groups * c_in_pg,
                    "MetalGroupConvOp: input channels mismatch for groups");

    m_desc.kH = static_cast<uint32_t>(w_shape.at(3));
    m_desc.kW = static_cast<uint32_t>(w_shape.at(4));
    m_desc.strideH = static_cast<uint32_t>(strides.at(0));
    m_desc.strideW = static_cast<uint32_t>(strides.at(1));
    m_desc.dilationH = static_cast<uint32_t>(dilations.at(0));
    m_desc.dilationW = static_cast<uint32_t>(dilations.at(1));
    m_desc.padTop = static_cast<uint32_t>(pads_begin.at(0));
    m_desc.padLeft = static_cast<uint32_t>(pads_begin.at(1));
    m_desc.padBottom = static_cast<uint32_t>(pads_end.at(0));
    m_desc.padRight = static_cast<uint32_t>(pads_end.at(1));
    m_desc.outH = 0;
    m_desc.outW = 0;
    m_desc.has_bias = false;
    m_desc.has_bn = false;
    m_desc.has_activation = false;
}

void MetalGroupConvOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

bool MetalGroupConvOp::fuse_activation(ActivationKind kind, float alpha) {
    OPENVINO_ASSERT(!is_compiled(), "MetalGroupConvOp: cannot fuse activation after compilation");
    m_has_activation = true;
    m_activation = kind;
    m_activation_alpha = alpha;
    return true;
}

void MetalGroupConvOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }
    prepare_weights();
    OPENVINO_ASSERT(m_device, "MetalGroupConvOp: Metal device is null");
    MetalKernelCompiler compiler(m_device);
    std::string log;
    Conv2DCodegenDesc desc = m_desc;
    desc.element_type = m_element_type == ov::element::dynamic ? ov::element::f32 : m_element_type;
    if (m_has_activation) {
        desc.has_activation = true;
        desc.activation = m_activation;
        desc.alpha = m_activation_alpha;
    }
    mlir::ModuleOp module;
    auto source = generate_msl_from_mlir(module, desc);
    m_pipeline = compiler.compile_msl_from_source(source, "conv2d_kernel", log);
    OPENVINO_ASSERT(m_pipeline, "MetalGroupConvOp: failed to compile conv2d kernel: ", log);
    MetalOp::compile(buffer_manager);
}

void MetalGroupConvOp::prepare_weights() {
    OPENVINO_ASSERT(buffer_manager(), "MetalGroupConvOp: buffer manager is null");
    auto weights_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(m_node->get_input_node_shared_ptr(1));
    OPENVINO_ASSERT(weights_const, "MetalGroupConvOp requires constant weights");

    const auto& et = weights_const->get_element_type();
    const size_t bytes = element_size(et) * shape_size(weights_const->get_shape());
    const std::string key = m_node->get_friendly_name() + "/weights";
    m_weights = buffer_manager()->wrap_const(key, weights_const->get_data_ptr(), bytes, et);
    OPENVINO_ASSERT(m_weights.valid(), "MetalGroupConvOp: failed to wrap weights buffer");
}

void MetalGroupConvOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(!inputs().empty(), "MetalGroupConvOp: no inputs bound");
    MetalTensor* src = inputs().at(0);
    OPENVINO_ASSERT(src && src->buf.valid(), "MetalGroupConvOp: input buffer is null");
    MetalTensor& dst_tensor = require_output();

    const ov::Shape in_shape = !src->shape.empty() ? src->shape : m_node->get_input_shape(0);
    OPENVINO_ASSERT(in_shape.size() == 4, "MetalGroupConvOp: expected NCHW input");
    OPENVINO_ASSERT(in_shape[0] == m_desc.N &&
                        in_shape[1] == m_desc.C_in &&
                        in_shape[2] == m_desc.H &&
                        in_shape[3] == m_desc.W,
                    "MetalGroupConvOp: runtime input shape differs from compiled shape");
    const size_t in_bytes = ov::shape_size(in_shape) * element_size(m_element_type);
    OPENVINO_ASSERT(src->buf.size >= in_bytes, "MetalGroupConvOp: input buffer too small");

    const ov::Shape out_shape = !output_shape().empty() ? output_shape() : m_node->get_output_shape(0);
    OPENVINO_ASSERT(out_shape.size() == 4, "MetalGroupConvOp: expected NCHW output");
    const size_t out_bytes = ov::shape_size(out_shape) * element_size(m_element_type);
    if (!dst_tensor.buf.valid() || dst_tensor.buf.size < out_bytes) {
        const auto& et = m_node->get_output_element_type(0);
        size_t bytes = element_size(et);
        for (auto d : out_shape) bytes *= d;
        dst_tensor.buf = buffer_manager()->allocate(bytes,
                                                    et,
                                                    /*persistent=*/false,
                                                    dst_tensor.prefer_private);
        dst_tensor.expected_type = et;
    }
    dst_tensor.shape = out_shape;
    dst_tensor.expected_type = m_node->get_output_element_type(0);

    id<MTLCommandBuffer> cb = static_cast<id<MTLCommandBuffer>>(cmd_buf_handle);
    OPENVINO_ASSERT(cb, "MetalGroupConvOp: command buffer is null");
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m_pipeline];

    id<MTLBuffer> in0 = to_mtl(src->buf);
    id<MTLBuffer> w = to_mtl(m_weights);
    id<MTLBuffer> bias = nil;
    id<MTLBuffer> gamma = nil;
    id<MTLBuffer> beta = nil;
    id<MTLBuffer> mean = nil;
    id<MTLBuffer> var = nil;
    id<MTLBuffer> out = to_mtl(dst_tensor.buf);
    if (w) {
        const ov::Shape w_shape = m_node->get_input_shape(1);
        const size_t w_bytes = ov::shape_size(w_shape) * element_size(m_element_type);
        OPENVINO_ASSERT(m_weights.size >= w_bytes, "MetalGroupConvOp: weights buffer too small");
    }

    [enc setBuffer:in0 offset:0 atIndex:0];
    [enc setBuffer:w offset:0 atIndex:1];
    [enc setBuffer:bias offset:0 atIndex:2];
    [enc setBuffer:gamma offset:0 atIndex:3];
    [enc setBuffer:beta offset:0 atIndex:4];
    [enc setBuffer:mean offset:0 atIndex:5];
    [enc setBuffer:var offset:0 atIndex:6];
    [enc setBuffer:out offset:0 atIndex:7];
    struct ConvParams {
        uint32_t N, C_in, H, W;
        uint32_t C_out;
        uint32_t groups;
        uint32_t C_in_pg;
        uint32_t C_out_pg;
        uint32_t kH, kW;
        uint32_t strideH, strideW;
        uint32_t dilationH, dilationW;
        uint32_t padTop, padLeft;
        uint32_t padBottom, padRight;
        uint32_t outH, outW;
        uint32_t has_bias;
        uint32_t has_bn;
        uint32_t activation;
        float alpha;
        float epsilon;
        float clamp_min;
        float clamp_max;
    } params{};
    params.N = m_desc.N;
    params.C_in = m_desc.C_in;
    params.H = m_desc.H;
    params.W = m_desc.W;
    params.C_out = m_desc.C_out;
    params.groups = m_desc.groups;
    params.C_in_pg = m_desc.C_in_pg;
    params.C_out_pg = m_desc.C_out_pg;
    params.kH = m_desc.kH;
    params.kW = m_desc.kW;
    params.strideH = m_desc.strideH;
    params.strideW = m_desc.strideW;
    params.dilationH = m_desc.dilationH;
    params.dilationW = m_desc.dilationW;
    params.padTop = m_desc.padTop;
    params.padLeft = m_desc.padLeft;
    params.padBottom = m_desc.padBottom;
    params.padRight = m_desc.padRight;
    params.outH = static_cast<uint32_t>(out_shape[2]);
    params.outW = static_cast<uint32_t>(out_shape[3]);
    OPENVINO_ASSERT(params.outH > 0 && params.outW > 0, "MetalGroupConvOp: output spatial dims must be positive");
    params.has_bias = 0;
    params.has_bn = 0;
    params.activation = m_has_activation ? static_cast<uint32_t>(m_activation) : 0;
    params.alpha = m_activation_alpha;
    params.epsilon = 0.0f;
    params.clamp_min = 0.0f;
    params.clamp_max = 0.0f;

    [enc setBytes:&params length:sizeof(params) atIndex:8];

    const uint64_t total_u64 = static_cast<uint64_t>(params.N) *
                               static_cast<uint64_t>(params.outH) *
                               static_cast<uint64_t>(params.outW) *
                               static_cast<uint64_t>(params.C_out);
    OPENVINO_ASSERT(total_u64 > 0, "MetalGroupConvOp: computed zero elements");
    constexpr uint64_t kMaxGrid = 1ULL << 31;
    OPENVINO_ASSERT(total_u64 < kMaxGrid, "MetalGroupConvOp: computed grid too large: ", total_u64);
    const NSUInteger total = static_cast<NSUInteger>(total_u64);
    if (total == 0) {
        [enc endEncoding];
        return;
    }
    MTLSize grid = MTLSizeMake(total, 1, 1);
    const NSUInteger tg_size = static_cast<NSUInteger>(metal_clamp_tg_size((void*)m_pipeline, 64));
    MTLSize tg = MTLSizeMake(tg_size, 1, 1);
    start_profiling(enc);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    stop_profiling_ms(enc);
    [enc endEncoding];

    dst_tensor.shape = output_shape();
    dst_tensor.expected_type = m_node->get_output_element_type(0);
}

}  // namespace gfx_plugin
}  // namespace ov
