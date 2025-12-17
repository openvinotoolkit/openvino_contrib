// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "runtime/metal_op_batchnorm.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/batch_norm.hpp"
#include "kernel_codegen/metal_kernel_compiler.hpp"
#include "runtime/metal_logger.hpp"

namespace ov {
namespace metal_plugin {

namespace {
inline id<MTLBuffer> to_mtl(const MetalBuffer& buf) {
    return (__bridge id<MTLBuffer>)buf.buffer;
}
}  // namespace

MetalBatchNormOp::MetalBatchNormOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "BatchNorm",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    parse_bn(node);
}

void MetalBatchNormOp::parse_bn(const std::shared_ptr<const ov::Node>& node) {
    auto bn0 = ov::as_type_ptr<const ov::op::v0::BatchNormInference>(node);
    auto bn5 = ov::as_type_ptr<const ov::op::v5::BatchNormInference>(node);
    OPENVINO_ASSERT(bn0 || bn5, "MetalBatchNormOp expects BatchNormInference v0/v5");
    const auto& in_shape = node->get_input_shape(0);  // NCHW
    OPENVINO_ASSERT(in_shape.size() == 4, "BatchNorm: expects NCHW rank4");

    const size_t C = in_shape[1];
    m_element_type = node->get_output_element_type(0);
    OPENVINO_ASSERT(m_element_type == ov::element::f32 || m_element_type == ov::element::f16,
                    "MetalBatchNormOp supports f16/f32 only");

    auto get_const = [](const ov::Input<const ov::Node>& in) {
        return ov::as_type_ptr<const ov::op::v0::Constant>(in.get_source_output().get_node_shared_ptr());
    };

    auto gamma = get_const(node->input(1));
    auto beta = get_const(node->input(2));
    auto mean = get_const(node->input(3));
    auto var = get_const(node->input(4));
    OPENVINO_ASSERT(gamma && beta && mean && var, "BatchNorm: parameters must be constants");
    auto gamma_v = gamma->cast_vector<float>();
    auto beta_v = beta->cast_vector<float>();
    auto mean_v = mean->cast_vector<float>();
    auto var_v = var->cast_vector<float>();
    OPENVINO_ASSERT(gamma_v.size() == C && beta_v.size() == C && mean_v.size() == C && var_v.size() == C,
                    "BatchNorm: parameter size mismatch with channels");

    float eps = bn0 ? bn0->get_eps_value() : bn5->get_eps_value();
    m_params.resize(4 * C + 1);
    std::copy(gamma_v.begin(), gamma_v.end(), m_params.begin());
    std::copy(beta_v.begin(), beta_v.end(), m_params.begin() + C);
    std::copy(mean_v.begin(), mean_v.end(), m_params.begin() + 2 * C);
    std::copy(var_v.begin(), var_v.end(), m_params.begin() + 3 * C);
    m_params[4 * C] = eps;

    m_desc.kind = KernelOpKind::BatchNorm2D;
    m_desc.dtype = resolve_metal_dtype(m_element_type);
    m_desc.batchnorm.N = static_cast<uint32_t>(in_shape[0]);
    m_desc.batchnorm.C = static_cast<uint32_t>(in_shape[1]);
    m_desc.batchnorm.H = static_cast<uint32_t>(in_shape[2]);
    m_desc.batchnorm.W = static_cast<uint32_t>(in_shape[3]);
    m_desc.batchnorm.eps = eps;
    m_desc.bn_params = m_params;
    m_desc.element_type = static_cast<uint32_t>(static_cast<ov::element::Type_t>(m_element_type));
}

void MetalBatchNormOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
    const size_t bytes = m_params.size() * sizeof(float);
    m_params_buf = buffer_manager->allocate(bytes,
                                            ov::element::f32,
                                            /*persistent=*/true,
                                            /*storageModePrivate=*/true);
    buffer_manager->upload(m_params_buf, m_params.data(), bytes);

    MetalKernelCompiler compiler(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    m_pipeline = compiler.compile_batchnorm2d_kernel(m_desc, log);
    OPENVINO_ASSERT(m_pipeline, "MetalBatchNormOp: failed to compile kernel: ", log);
}

void MetalBatchNormOp::execute() {
    OPENVINO_ASSERT(!inputs().empty(), "BatchNorm: missing input");
    MetalTensor* src = inputs()[0];
    OPENVINO_ASSERT(src && src->buf.valid(), "BatchNorm: input buffer null");
    MetalTensor& dst = require_output();

    ov::Shape out_shape = !dst.shape.empty() ? dst.shape : output_shape();
    if (out_shape.empty())
        out_shape = src->shape;
    OPENVINO_ASSERT(!out_shape.empty(), "BatchNorm: output shape unknown");

    if (!dst.buf.valid()) {
        size_t bytes = ov::shape_size(out_shape) * m_element_type.size();
        dst.buf = buffer_manager()->allocate(bytes, m_element_type, /*persistent=*/false, /*storageModePrivate=*/true);
    }
    dst.shape = out_shape;
    dst.expected_type = m_element_type;
    const size_t elem_sz = m_element_type.size();
    const size_t need_src = ov::shape_size(src->shape.empty() ? out_shape : src->shape) * elem_sz;
    const size_t need_dst = ov::shape_size(out_shape) * elem_sz;
    OPENVINO_ASSERT(src->buf.size >= need_src, "BatchNorm: src buffer too small");
    OPENVINO_ASSERT(dst.buf.size >= need_dst, "BatchNorm: dst buffer too small");

    if (!m_queue) {
        m_queue = [m_device newCommandQueue];
    }
    id<MTLCommandBuffer> cb = [m_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m_pipeline];
    [enc setBuffer:to_mtl(src->buf) offset:0 atIndex:0];
    [enc setBuffer:to_mtl(m_params_buf) offset:0 atIndex:1];
    [enc setBuffer:to_mtl(dst.buf) offset:0 atIndex:2];

    struct BNParamsGPU {
        uint32_t N;
        uint32_t C;
        uint32_t H;
        uint32_t W;
    } gpu_params{m_desc.batchnorm.N, m_desc.batchnorm.C, m_desc.batchnorm.H, m_desc.batchnorm.W};
    [enc setBytes:&gpu_params length:sizeof(gpu_params) atIndex:3];

    const uint32_t total = m_desc.batchnorm.N * m_desc.batchnorm.C * m_desc.batchnorm.H * m_desc.batchnorm.W;
    MTLSize grid = MTLSizeMake(total, 1, 1);
    MTLSize tg = MTLSizeMake(64, 1, 1);

    start_profiling();
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    stop_profiling_ms();
}

}  // namespace metal_plugin
}  // namespace ov
