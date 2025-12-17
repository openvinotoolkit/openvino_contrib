// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "runtime/metal_op_conv.hpp"

#include <numeric>
#include <cstdint>

#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "runtime/metal_dtype.hpp"
#include "runtime/metal_logger.hpp"
#include "runtime/metal_memory.hpp"

namespace ov {
namespace metal_plugin {

namespace {

inline id<MTLBuffer> to_mtl(const MetalBuffer& buf) {
    return (__bridge id<MTLBuffer>)buf.buffer;
}

inline size_t element_size(const ov::element::Type& t) {
    return t.size();
}

}  // namespace

MetalConvOp::MetalConvOp(const std::shared_ptr<const ov::op::v1::Convolution>& node,
                         MetalDeviceHandle device,
                         MetalCommandQueueHandle queue)
    : MetalOp(node->get_friendly_name(),
              "Conv2D",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_node(node),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    OPENVINO_ASSERT(m_node, "MetalConvOp: node is null");

    // Fill kernel descriptor from node attributes.
    const auto strides = m_node->get_strides();
    const auto dilations = m_node->get_dilations();
    const auto pads_begin = m_node->get_pads_begin();
    const auto pads_end = m_node->get_pads_end();

    const auto& in_shape = m_node->get_input_shape(0);   // NCHW
    const auto& w_shape = m_node->get_input_shape(1);    // OIHW (groups folded)

    m_desc.kind = KernelOpKind::Conv2D;
    m_desc.dtype = resolve_metal_dtype(m_node->get_output_element_type(0));
    m_desc.conv2d.dtype = m_desc.dtype;
    m_desc.conv2d.N = static_cast<uint32_t>(in_shape.at(0));
    m_desc.conv2d.C_in = static_cast<uint32_t>(in_shape.at(1));
    m_desc.conv2d.H = static_cast<uint32_t>(in_shape.at(2));
    m_desc.conv2d.W = static_cast<uint32_t>(in_shape.at(3));
    m_desc.conv2d.C_out = static_cast<uint32_t>(w_shape.at(0));
    // Derive groups from weight shape: O x (I/groups) x kH x kW
    const uint32_t cin_per_group = static_cast<uint32_t>(w_shape.at(1));
    m_desc.conv2d.groups = (cin_per_group > 0 && (m_desc.conv2d.C_in % cin_per_group) == 0)
                               ? m_desc.conv2d.C_in / cin_per_group
                               : 1;
    m_desc.conv2d.C_in_per_group = cin_per_group;
    m_desc.conv2d.C_out_per_group = m_desc.conv2d.groups ? m_desc.conv2d.C_out / m_desc.conv2d.groups
                                                         : m_desc.conv2d.C_out;
    m_desc.conv2d.kernelH = static_cast<uint32_t>(w_shape.at(2));
    m_desc.conv2d.kernelW = static_cast<uint32_t>(w_shape.at(3));
    m_desc.conv2d.strideH = static_cast<uint32_t>(strides.at(0));
    m_desc.conv2d.strideW = static_cast<uint32_t>(strides.at(1));
    m_desc.conv2d.dilationH = static_cast<uint32_t>(dilations.at(0));
    m_desc.conv2d.dilationW = static_cast<uint32_t>(dilations.at(1));
    m_desc.conv2d.padTop = static_cast<uint32_t>(pads_begin.at(0));
    m_desc.conv2d.padLeft = static_cast<uint32_t>(pads_begin.at(1));
    m_desc.conv2d.padBottom = static_cast<uint32_t>(pads_end.at(0));
    m_desc.conv2d.padRight = static_cast<uint32_t>(pads_end.at(1));
    m_desc.conv2d.outH = 0;  // will be derived by codegen if zero
    m_desc.conv2d.outW = 0;
    m_desc.conv2d.has_bias = false;
    m_desc.conv2d.has_bn = false;
    m_desc.conv2d.has_activation = false;
    m_desc.output = nullptr;
}

void MetalConvOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
    prepare_weights();
    compile_pipeline();
}

void MetalConvOp::prepare_weights() {
    OPENVINO_ASSERT(buffer_manager(), "MetalConvOp: buffer manager is null");
    auto weights_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(m_node->get_input_node_shared_ptr(1));
    OPENVINO_ASSERT(weights_const, "MetalConvOp requires constant weights");

    const auto& et = weights_const->get_element_type();
    const size_t bytes = element_size(et) * shape_size(weights_const->get_shape());

    m_weights = buffer_manager()->allocate(bytes,
                                           et,
                                           /*persistent=*/true,
                                           /*storageModePrivate=*/true);
    OPENVINO_ASSERT(m_weights.valid(), "MetalConvOp: failed to allocate weights buffer");
    buffer_manager()->upload(m_weights, weights_const->get_data_ptr(), bytes);
}

void MetalConvOp::compile_pipeline() {
    OPENVINO_ASSERT(m_device, "MetalConvOp: Metal device is null");
    MetalKernelCompiler compiler(m_device);
    std::string log;
    m_pipeline = compiler.compile_conv2d_kernel(m_desc, log);
    OPENVINO_ASSERT(m_pipeline, "MetalConvOp: failed to compile conv2d kernel: ", log);
}

void MetalConvOp::execute() {
    OPENVINO_ASSERT(!inputs().empty(), "MetalConvOp: no inputs bound");
    MetalTensor* src = inputs().at(0);
    OPENVINO_ASSERT(src && src->buf.valid(), "MetalConvOp: input buffer is null");
    MetalTensor& dst_tensor = require_output();

    // Sanity‑check runtime shapes: the compiled pipeline is specialized to the
    // static shapes captured in m_desc. If the actual tensor differs (e.g.
    // dynamic reshape slipped through), abort early instead of dispatching an
    // out‑of‑bounds kernel that can hang or reset the GPU.
    const ov::Shape in_shape = !src->shape.empty() ? src->shape : m_node->get_input_shape(0);
    OPENVINO_ASSERT(in_shape.size() == 4, "MetalConvOp: expected NCHW input, got rank ", in_shape.size());
    OPENVINO_ASSERT(in_shape[0] == m_desc.conv2d.N &&
                        in_shape[1] == m_desc.conv2d.C_in &&
                        in_shape[2] == m_desc.conv2d.H &&
                        in_shape[3] == m_desc.conv2d.W,
                    "MetalConvOp: runtime input shape differs from compiled shape");

    const ov::Shape out_shape = !output_shape().empty() ? output_shape() : m_node->get_output_shape(0);
    OPENVINO_ASSERT(out_shape.size() == 4, "MetalConvOp: expected NCHW output, got rank ", out_shape.size());

    if (!dst_tensor.buf.valid()) {
        // Allocate output on first run; reuse across inferences.
        const auto& et = m_node->get_output_element_type(0);
        size_t bytes = element_size(et);
        for (auto d : out_shape) bytes *= d;
        dst_tensor.buf = buffer_manager()->allocate(bytes,
                                                    et,
                                                    /*persistent=*/false,
                                                    /*storageModePrivate=*/true);
        dst_tensor.expected_type = et;
    }
    dst_tensor.shape = out_shape;
    dst_tensor.expected_type = m_node->get_output_element_type(0);

    if (!m_queue) {
        m_queue = [m_device newCommandQueue];
    }
    id<MTLCommandBuffer> cb = [m_queue commandBuffer];
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
    params.N = m_desc.conv2d.N;
    params.C_in = m_desc.conv2d.C_in;
    params.H = m_desc.conv2d.H;
    params.W = m_desc.conv2d.W;
    params.C_out = m_desc.conv2d.C_out;
    params.groups = m_desc.conv2d.groups;
    params.C_in_pg = m_desc.conv2d.C_in_per_group;
    params.C_out_pg = m_desc.conv2d.C_out_per_group;
    params.kH = m_desc.conv2d.kernelH;
    params.kW = m_desc.conv2d.kernelW;
    params.strideH = m_desc.conv2d.strideH;
    params.strideW = m_desc.conv2d.strideW;
    params.dilationH = m_desc.conv2d.dilationH;
    params.dilationW = m_desc.conv2d.dilationW;
    params.padTop = m_desc.conv2d.padTop;
    params.padLeft = m_desc.conv2d.padLeft;
    params.padBottom = m_desc.conv2d.padBottom;
    params.padRight = m_desc.conv2d.padRight;
    // Trust the model's output shape to avoid negative/overflowed dims.
    params.outH = static_cast<uint32_t>(out_shape[2]);
    params.outW = static_cast<uint32_t>(out_shape[3]);
    OPENVINO_ASSERT(params.outH > 0 && params.outW > 0, "MetalConvOp: output spatial dims must be positive");
    params.has_bias = m_desc.conv2d.has_bias ? 1u : 0u;
    params.has_bn = m_desc.conv2d.has_bn ? 1u : 0u;
    params.activation = static_cast<uint32_t>(m_desc.conv2d.activation);
    params.alpha = m_desc.conv2d.alpha;
    params.epsilon = m_desc.conv2d.epsilon;
    params.clamp_min = m_desc.conv2d.clamp_min;
    params.clamp_max = m_desc.conv2d.clamp_max;
    [enc setBytes:&params length:sizeof(params) atIndex:8];

    const uint64_t total_u64 = static_cast<uint64_t>(params.N) *
                               static_cast<uint64_t>(params.outH) *
                               static_cast<uint64_t>(params.outW) *
                               static_cast<uint64_t>(params.C_out);
    OPENVINO_ASSERT(total_u64 > 0, "MetalConvOp: computed zero elements");
    // Guard against runaway grids that can wedge the GPU in case of bad shapes.
    constexpr uint64_t kMaxGrid = 1ULL << 31;  // ~2 billion threads
    OPENVINO_ASSERT(total_u64 < kMaxGrid, "MetalConvOp: computed grid too large: ", total_u64);
    const NSUInteger total = static_cast<NSUInteger>(total_u64);
    MTLSize grid = MTLSizeMake(total, 1, 1);
    MTLSize tg = MTLSizeMake(64, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];

    start_profiling();
    [cb commit];
    [cb waitUntilCompleted];
    stop_profiling_ms();

    // Update output shape/type metadata in tensor.
    dst_tensor.shape = output_shape();
    dst_tensor.expected_type = m_node->get_output_element_type(0);
}

}  // namespace metal_plugin
}  // namespace ov
