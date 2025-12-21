// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "runtime/metal_op_pooling.hpp"

#include <cmath>
#include <numeric>

#include "openvino/core/shape_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "runtime/metal_logger.hpp"
#include "kernel_codegen/metal_kernel_compiler.hpp"
#include "runtime/metal_op_utils.hpp"
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
                              Pool2DCodegenDesc& desc,
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
    desc.N = static_cast<uint32_t>(in_shape[0]);
    desc.C = static_cast<uint32_t>(in_shape[1]);
    desc.H = static_cast<uint32_t>(in_shape[2]);
    desc.W = static_cast<uint32_t>(in_shape[3]);
    desc.outH = static_cast<uint32_t>(out_shape[2]);
    desc.outW = static_cast<uint32_t>(out_shape[3]);
    desc.kH = static_cast<uint32_t>(k.at(0));
    desc.kW = static_cast<uint32_t>(k.at(1));
    desc.strideH = static_cast<uint32_t>(s.at(0));
    desc.strideW = static_cast<uint32_t>(s.at(1));
    desc.padTop = static_cast<uint32_t>(pb.at(0));
    desc.padLeft = static_cast<uint32_t>(pb.at(1));
    desc.padBottom = static_cast<uint32_t>(pe.at(0));
    desc.padRight = static_cast<uint32_t>(pe.at(1));
    desc.exclude_pad = false;
}

void fill_pool_desc_from_node(const ov::op::v1::AvgPool* node,
                              Pool2DCodegenDesc& desc,
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
    desc.N = static_cast<uint32_t>(in_shape[0]);
    desc.C = static_cast<uint32_t>(in_shape[1]);
    desc.H = static_cast<uint32_t>(in_shape[2]);
    desc.W = static_cast<uint32_t>(in_shape[3]);
    desc.outH = static_cast<uint32_t>(out_shape[2]);
    desc.outW = static_cast<uint32_t>(out_shape[3]);
    desc.kH = static_cast<uint32_t>(k.at(0));
    desc.kW = static_cast<uint32_t>(k.at(1));
    desc.strideH = static_cast<uint32_t>(s.at(0));
    desc.strideW = static_cast<uint32_t>(s.at(1));
    desc.padTop = static_cast<uint32_t>(pb.at(0));
    desc.padLeft = static_cast<uint32_t>(pb.at(1));
    desc.padBottom = static_cast<uint32_t>(pe.at(0));
    desc.padRight = static_cast<uint32_t>(pe.at(1));
    desc.exclude_pad = node->get_exclude_pad();
}

}  // namespace

MetalPoolOp::MetalPoolOp(const std::shared_ptr<const ov::Node>& node,
                         bool is_avg,
                         bool exclude_pad,
                         void* device,
                         void* queue)
    : MetalOp(node->get_friendly_name(),
              is_avg ? "AvgPool2D" : "MaxPool2D",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_is_avg(is_avg),
      m_node(node),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    if (!m_is_avg) {
        auto mp = std::dynamic_pointer_cast<const ov::op::v1::MaxPool>(node);
        fill_pool_desc_from_node(mp.get(), m_desc, m_element_type, m_rounding);
    } else {
        auto ap = std::dynamic_pointer_cast<const ov::op::v1::AvgPool>(node);
        fill_pool_desc_from_node(ap.get(), m_desc, m_element_type, m_rounding);
        // Override exclude_pad from ctor argument in case it differs.
        m_desc.exclude_pad = exclude_pad;
    }
    OPENVINO_ASSERT(m_element_type == ov::element::f32,
                    "MetalPoolOp currently supports only f32, got ",
                    m_element_type.to_string());
}

void MetalPoolOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalPoolOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }
    if (!m_device && buffer_manager) {
        m_device = (id<MTLDevice>)buffer_manager->device();
    }

    MetalKernelCompiler compiler(m_device);
    std::string log;
    Pool2DCodegenDesc desc = m_desc;
    desc.is_avg = m_is_avg;
    desc.element_type = m_element_type == ov::element::dynamic ? ov::element::f32 : m_element_type;
    mlir::MLIRContext ctx;
    auto model = make_single_op_model(m_node);
    mlir::ModuleOp module = m_is_avg ? build_mlir_avgpool_from_model(model, ctx)
                                     : build_mlir_maxpool_from_model(model, ctx);
    auto source = generate_msl_from_mlir(module, desc);
    m_pipeline = compiler.compile_msl_from_source(source, "pool2d_kernel", log);
    OPENVINO_ASSERT(m_pipeline, "MetalPoolOp: failed to compile pool2d kernel: ", log);

    MetalOp::compile(buffer_manager);
}

void MetalPoolOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(!inputs().empty(), "MetalPoolOp: no inputs bound");
    MetalTensor* src = inputs().at(0);
    OPENVINO_ASSERT(src && src->buf.valid(), "MetalPoolOp: input buffer is null");
    MetalTensor& dst = require_output();

    Pool2DCodegenDesc op = m_desc;  // copy to allow runtime shape tweaks without mutating descriptor
    ov::Shape in_shape = !src->shape.empty() ? src->shape : ov::Shape{};
    if (in_shape.empty() && m_node->get_input_partial_shape(0).is_static()) {
        in_shape = m_node->get_input_shape(0);
    }
    OPENVINO_ASSERT(!in_shape.empty(), "MetalPoolOp: input shape unknown");
    OPENVINO_ASSERT(in_shape.size() == 4, "MetalPoolOp: expected rank-4 input tensor");
    op.N = static_cast<uint32_t>(in_shape[0]);
    op.C = static_cast<uint32_t>(in_shape[1]);
    op.H = static_cast<uint32_t>(in_shape[2]);
    op.W = static_cast<uint32_t>(in_shape[3]);
    const size_t in_bytes = m_element_type.size() * ov::shape_size(in_shape);
    OPENVINO_ASSERT(src->buf.size >= in_bytes, "MetalPoolOp: input buffer too small");

    if (op.outH == 0 || op.outW == 0) {
        op.outH = compute_out_dim(op.H,
                                       op.kH,
                                       op.strideH,
                                       1,
                                       op.padTop,
                                       op.padBottom,
                                       m_rounding);
        op.outW = compute_out_dim(op.W,
                                       op.kW,
                                       op.strideW,
                                       1,
                                       op.padLeft,
                                       op.padRight,
                                       m_rounding);
    }
    OPENVINO_ASSERT(op.outH > 0 && op.outW > 0, "MetalPoolOp: invalid output spatial dims");

    // Ensure destination metadata and buffer
    const ov::Shape out_shape{op.N, op.C, op.outH, op.outW};
    if (dst.shape != out_shape) {
        dst.shape = out_shape;
    }
    const size_t bytes = m_element_type.size() * ov::shape_size(dst.shape);
    if (!dst.buf.valid() || dst.buf.size < bytes) {
        dst.buf = buffer_manager()->allocate(bytes, m_element_type, /*persistent=*/false, dst.prefer_private);
    }
    dst.expected_type = m_element_type;

    id<MTLCommandBuffer> cb = static_cast<id<MTLCommandBuffer>>(cmd_buf_handle);
    OPENVINO_ASSERT(cb, "MetalPoolOp: command buffer is null");
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
    params.N = op.N;
    params.C = op.C;
    params.H = op.H;
    params.W = op.W;
    params.kH = op.kH;
    params.kW = op.kW;
    params.strideH = op.strideH;
    params.strideW = op.strideW;
    params.dilationH = 1;
    params.dilationW = 1;
    params.padTop = op.padTop;
    params.padLeft = op.padLeft;
    params.padBottom = op.padBottom;
    params.padRight = op.padRight;
    params.outH = op.outH;
    params.outW = op.outW;
    params.is_avg = m_is_avg ? 1u : 0u;
    params.exclude_pad = (m_is_avg && op.exclude_pad) ? 1u : 0u;

    [enc setBuffer:to_mtl(src->buf) offset:0 atIndex:0];
    [enc setBuffer:to_mtl(dst.buf) offset:0 atIndex:1];
    [enc setBytes:&params length:sizeof(params) atIndex:2];

    // One thread per (n, c) pair; kernel iterates over spatial dims.
    const NSUInteger total = static_cast<NSUInteger>(params.N) * static_cast<NSUInteger>(params.C);
    if (total == 0) {
        [enc endEncoding];
        return;
    }
    const NSUInteger threads_per_tg = 64;
    MTLSize grid = MTLSizeMake(total, 1, 1);
    const NSUInteger tg_size = static_cast<NSUInteger>(metal_clamp_tg_size((void*)m_pipeline, threads_per_tg));
    MTLSize tg = MTLSizeMake(tg_size, 1, 1);

    start_profiling(enc);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    stop_profiling_ms(enc);
    [enc endEncoding];
}

MetalMaxPoolOp::MetalMaxPoolOp(const std::shared_ptr<const ov::op::v1::MaxPool>& node,
                               void* device,
                               void* queue)
    : MetalPoolOp(node, /*is_avg*/ false, /*exclude_pad*/ false, device, queue) {}

MetalAvgPoolOp::MetalAvgPoolOp(const std::shared_ptr<const ov::op::v1::AvgPool>& node,
                               void* device,
                               void* queue)
    : MetalPoolOp(node, /*is_avg*/ true, node ? node->get_exclude_pad() : true, device, queue) {}

}  // namespace gfx_plugin
}  // namespace ov
