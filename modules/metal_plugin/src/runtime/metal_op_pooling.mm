// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "runtime/metal_op_pooling.hpp"

#include <cmath>
#include <numeric>

#include "openvino/core/shape_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "runtime/metal_dtype.hpp"
#include "runtime/metal_logger.hpp"
#include "kernel_codegen/metal_kernel_compiler.hpp"

namespace ov {
namespace metal_plugin {

namespace {
inline id<MTLBuffer> to_mtl(const MetalBuffer& buf) {
    return (__bridge id<MTLBuffer>)buf.buffer;
}

uint32_t compute_out_dim(uint32_t in,
                         uint32_t k,
                         uint32_t stride,
                         uint32_t dilation,
                         uint32_t pad_begin,
                         uint32_t pad_end,
                         ov::op::RoundingType rounding) {
    const int64_t eff = static_cast<int64_t>(dilation) * (static_cast<int64_t>(k) - 1) + 1;
    const int64_t numer = static_cast<int64_t>(in) + pad_begin + pad_end - eff;
    const double raw = static_cast<double>(numer) / static_cast<double>(stride) + 1.0;
    return static_cast<uint32_t>(rounding == ov::op::RoundingType::CEIL ? std::ceil(raw) : std::floor(raw));
}

void fill_pool_desc_from_node(const ov::op::v1::MaxPool* node,
                              KernelOp& desc,
                              ov::element::Type& et,
                              ov::op::RoundingType& rounding) {
    OPENVINO_ASSERT(node, "MetalPoolOp: MaxPool node is null");
    OPENVINO_ASSERT(node->get_input_size() == 1, "MetalPoolOp expects single input");
    OPENVINO_ASSERT(node->get_input_partial_shape(0).is_static(), "MetalPoolOp requires static input shape");
    OPENVINO_ASSERT(node->get_output_partial_shape(0).is_static(), "MetalPoolOp requires static output shape");
    const auto& in_shape = node->get_input_shape(0);
    const auto& out_shape = node->get_output_shape(0);
    OPENVINO_ASSERT(in_shape.size() == 4, "MetalPoolOp supports only NCHW rank-4 inputs");
    OPENVINO_ASSERT(out_shape.size() == 4, "MetalPoolOp supports only NCHW rank-4 outputs");
    et = node->get_output_element_type(0);
    rounding = node->get_rounding_type();

    const auto& k = node->get_kernel();
    const auto& s = node->get_strides();
    const auto& pb = node->get_pads_begin();
    const auto& pe = node->get_pads_end();
    desc.kind = KernelOpKind::MaxPool2D;
    desc.dtype = resolve_metal_dtype(et);
    desc.pool.N = static_cast<uint32_t>(in_shape[0]);
    desc.pool.C = static_cast<uint32_t>(in_shape[1]);
    desc.pool.H = static_cast<uint32_t>(in_shape[2]);
    desc.pool.W = static_cast<uint32_t>(in_shape[3]);
    desc.pool.outH = static_cast<uint32_t>(out_shape[2]);
    desc.pool.outW = static_cast<uint32_t>(out_shape[3]);
    desc.pool.kernelH = static_cast<uint32_t>(k.at(0));
    desc.pool.kernelW = static_cast<uint32_t>(k.at(1));
    desc.pool.strideH = static_cast<uint32_t>(s.at(0));
    desc.pool.strideW = static_cast<uint32_t>(s.at(1));
    desc.pool.padTop = static_cast<uint32_t>(pb.at(0));
    desc.pool.padLeft = static_cast<uint32_t>(pb.at(1));
    desc.pool.padBottom = static_cast<uint32_t>(pe.at(0));
    desc.pool.padRight = static_cast<uint32_t>(pe.at(1));
    desc.pool.dilationH = 1;
    desc.pool.dilationW = 1;
    desc.pool.exclude_pad = false;
}

void fill_pool_desc_from_node(const ov::op::v1::AvgPool* node,
                              KernelOp& desc,
                              ov::element::Type& et,
                              ov::op::RoundingType& rounding) {
    OPENVINO_ASSERT(node, "MetalPoolOp: AvgPool node is null");
    OPENVINO_ASSERT(node->get_input_size() == 1, "MetalPoolOp expects single input");
    OPENVINO_ASSERT(node->get_input_partial_shape(0).is_static(), "MetalPoolOp requires static input shape");
    OPENVINO_ASSERT(node->get_output_partial_shape(0).is_static(), "MetalPoolOp requires static output shape");
    const auto& in_shape = node->get_input_shape(0);
    const auto& out_shape = node->get_output_shape(0);
    OPENVINO_ASSERT(in_shape.size() == 4, "MetalPoolOp supports only NCHW rank-4 inputs");
    OPENVINO_ASSERT(out_shape.size() == 4, "MetalPoolOp supports only NCHW rank-4 outputs");
    et = node->get_output_element_type(0);
    rounding = node->get_rounding_type();

    const auto& k = node->get_kernel();
    const auto& s = node->get_strides();
    const auto& pb = node->get_pads_begin();
    const auto& pe = node->get_pads_end();
    desc.kind = KernelOpKind::AvgPool2D;
    desc.dtype = resolve_metal_dtype(et);
    desc.pool.N = static_cast<uint32_t>(in_shape[0]);
    desc.pool.C = static_cast<uint32_t>(in_shape[1]);
    desc.pool.H = static_cast<uint32_t>(in_shape[2]);
    desc.pool.W = static_cast<uint32_t>(in_shape[3]);
    desc.pool.outH = static_cast<uint32_t>(out_shape[2]);
    desc.pool.outW = static_cast<uint32_t>(out_shape[3]);
    desc.pool.kernelH = static_cast<uint32_t>(k.at(0));
    desc.pool.kernelW = static_cast<uint32_t>(k.at(1));
    desc.pool.strideH = static_cast<uint32_t>(s.at(0));
    desc.pool.strideW = static_cast<uint32_t>(s.at(1));
    desc.pool.padTop = static_cast<uint32_t>(pb.at(0));
    desc.pool.padLeft = static_cast<uint32_t>(pb.at(1));
    desc.pool.padBottom = static_cast<uint32_t>(pe.at(0));
    desc.pool.padRight = static_cast<uint32_t>(pe.at(1));
    desc.pool.dilationH = 1;
    desc.pool.dilationW = 1;
    desc.pool.exclude_pad = node->get_exclude_pad();
}

}  // namespace

MetalPoolOp::MetalPoolOp(const std::shared_ptr<const ov::Node>& node,
                         KernelOpKind kind,
                         bool exclude_pad,
                         void* device,
                         void* queue)
    : MetalOp(node->get_friendly_name(),
              kind == KernelOpKind::MaxPool2D ? "MaxPool2D" : "AvgPool2D",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_kind(kind),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    if (m_kind == KernelOpKind::MaxPool2D) {
        auto mp = std::dynamic_pointer_cast<const ov::op::v1::MaxPool>(node);
        fill_pool_desc_from_node(mp.get(), m_desc, m_element_type, m_rounding);
    } else {
        auto ap = std::dynamic_pointer_cast<const ov::op::v1::AvgPool>(node);
        fill_pool_desc_from_node(ap.get(), m_desc, m_element_type, m_rounding);
        // Override exclude_pad from ctor argument in case it differs.
        m_desc.pool.exclude_pad = exclude_pad;
    }
    OPENVINO_ASSERT(m_element_type == ov::element::f32,
                    "MetalPoolOp currently supports only f32, got ",
                    m_element_type.to_string());
}

void MetalPoolOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
    if (m_compiled)
        return;
    if (!m_device && buffer_manager) {
        m_device = (id<MTLDevice>)buffer_manager->device();
    }

    MetalKernelCompiler compiler(m_device);
    std::string log;
    if (m_kind == KernelOpKind::MaxPool2D) {
        m_pipeline = compiler.compile_maxpool2d_kernel(m_desc, log);
    } else {
        m_pipeline = compiler.compile_avgpool2d_kernel(m_desc, log);
    }
    OPENVINO_ASSERT(m_pipeline, "MetalPoolOp: failed to compile pool2d kernel: ", log);
    m_compiled = true;
}

void MetalPoolOp::execute() {
    OPENVINO_ASSERT(!inputs().empty(), "MetalPoolOp: no inputs bound");
    MetalTensor* src = inputs().at(0);
    OPENVINO_ASSERT(src && src->buf.valid(), "MetalPoolOp: input buffer is null");
    MetalTensor& dst = require_output();

    KernelOp op = m_desc;  // copy to allow runtime shape tweaks without mutating descriptor
    if (!src->shape.empty()) {
        OPENVINO_ASSERT(src->shape.size() == 4, "MetalPoolOp: expected rank-4 input tensor");
        op.pool.N = static_cast<uint32_t>(src->shape[0]);
        op.pool.C = static_cast<uint32_t>(src->shape[1]);
        op.pool.H = static_cast<uint32_t>(src->shape[2]);
        op.pool.W = static_cast<uint32_t>(src->shape[3]);
    }

    if (op.pool.outH == 0 || op.pool.outW == 0) {
        op.pool.outH = compute_out_dim(op.pool.H,
                                       op.pool.kernelH,
                                       op.pool.strideH,
                                       op.pool.dilationH,
                                       op.pool.padTop,
                                       op.pool.padBottom,
                                       m_rounding);
        op.pool.outW = compute_out_dim(op.pool.W,
                                       op.pool.kernelW,
                                       op.pool.strideW,
                                       op.pool.dilationW,
                                       op.pool.padLeft,
                                       op.pool.padRight,
                                       m_rounding);
    }

    // Ensure destination metadata and buffer
    const ov::Shape out_shape{op.pool.N, op.pool.C, op.pool.outH, op.pool.outW};
    if (dst.shape != out_shape) {
        dst.shape = out_shape;
    }
    const size_t bytes = m_element_type.size() * ov::shape_size(dst.shape);
    if (!dst.buf.valid()) {
        dst.buf = buffer_manager()->allocate(bytes, m_element_type, /*persistent=*/false, /*storageModePrivate=*/true);
    }
    dst.expected_type = m_element_type;

    if (!m_queue) {
        m_queue = [m_device newCommandQueue];
    }
    id<MTLCommandBuffer> cb = [m_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m_pipeline];

    struct PoolParams {
        uint32_t N, C, H, W;
        uint32_t kH, kW;
        uint32_t strideH, strideW;
        uint32_t dilationH, dilationW;
        uint32_t padTop, padLeft, padBottom, padRight;
        uint32_t outH, outW;
        uint32_t is_avg;
        uint32_t exclude_pad;
    } params{};
    params.N = op.pool.N;
    params.C = op.pool.C;
    params.H = op.pool.H;
    params.W = op.pool.W;
    params.kH = op.pool.kernelH;
    params.kW = op.pool.kernelW;
    params.strideH = op.pool.strideH;
    params.strideW = op.pool.strideW;
    params.dilationH = op.pool.dilationH ? op.pool.dilationH : 1;
    params.dilationW = op.pool.dilationW ? op.pool.dilationW : 1;
    params.padTop = op.pool.padTop;
    params.padLeft = op.pool.padLeft;
    params.padBottom = op.pool.padBottom;
    params.padRight = op.pool.padRight;
    params.outH = op.pool.outH;
    params.outW = op.pool.outW;
    params.is_avg = (m_kind == KernelOpKind::AvgPool2D) ? 1u : 0u;
    params.exclude_pad = (m_kind == KernelOpKind::AvgPool2D && op.pool.exclude_pad) ? 1u : 0u;

    [enc setBuffer:to_mtl(src->buf) offset:0 atIndex:0];
    [enc setBuffer:to_mtl(dst.buf) offset:0 atIndex:1];
    [enc setBytes:&params length:sizeof(params) atIndex:2];

    // One thread per (n, c) pair; kernel iterates over spatial dims.
    const NSUInteger total = static_cast<NSUInteger>(params.N) * static_cast<NSUInteger>(params.C);
    const NSUInteger threads_per_tg = 64;
    MTLSize grid = MTLSizeMake(total, 1, 1);
    MTLSize tg = MTLSizeMake(threads_per_tg, 1, 1);

    start_profiling();
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    stop_profiling_ms();
}

MetalMaxPoolOp::MetalMaxPoolOp(const std::shared_ptr<const ov::op::v1::MaxPool>& node,
                               void* device,
                               void* queue)
    : MetalPoolOp(node, KernelOpKind::MaxPool2D, /*exclude_pad*/ false, device, queue) {}

MetalAvgPoolOp::MetalAvgPoolOp(const std::shared_ptr<const ov::op::v1::AvgPool>& node,
                               void* device,
                               void* queue)
    : MetalPoolOp(node, KernelOpKind::AvgPool2D, node ? node->get_exclude_pad() : true, device, queue) {}

}  // namespace metal_plugin
}  // namespace ov
