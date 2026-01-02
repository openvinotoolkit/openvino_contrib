// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/op_batchnorm.hpp"

#include <string>

#include "openvino/op/constant.hpp"
#include "openvino/op/batch_norm.hpp"
#include "backends/metal/codegen/metal_codegen_backend.hpp"
#include "runtime/gfx_logger.hpp"
#include "backends/metal/runtime/op_utils.hpp"
#include "kernel_ir/gfx_kernel_args.hpp"
#include "mlir/mlir_builder.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir_codegen/codegen_common.hpp"

namespace ov {
namespace gfx_plugin {

MetalBatchNormOp::MetalBatchNormOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "BatchNorm",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue),
      m_node(node) {
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

    m_desc.N = static_cast<uint32_t>(in_shape[0]);
    m_desc.C = static_cast<uint32_t>(in_shape[1]);
    m_desc.H = static_cast<uint32_t>(in_shape[2]);
    m_desc.W = static_cast<uint32_t>(in_shape[3]);
    m_desc.element_type = m_element_type;
}

void MetalBatchNormOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalBatchNormOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }
    const size_t bytes = m_params.size() * sizeof(float);
    const std::string key = m_node->get_friendly_name() + "/params";
    m_params_buf = buffer_manager->wrap_const(key, m_params.data(), bytes, ov::element::f32);

    MetalCodegenBackend backend(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    BatchNorm2DCodegenDesc desc{};
    desc.N = m_desc.N;
    desc.C = m_desc.C;
    desc.H = m_desc.H;
    desc.W = m_desc.W;
    desc.element_type = m_element_type == ov::element::dynamic ? ov::element::f32 : m_element_type;
    mlir::MLIRContext ctx;
    auto model = make_single_op_model(m_node);
    auto module = build_mlir_batchnorm_from_model(model, ctx);
    auto msl_desc = desc;
    auto msl_generator = [msl_desc](mlir::ModuleOp mod) { return generate_msl_from_mlir(mod, msl_desc); };

    KernelSpec spec(m_node, 4u);
    m_kernel = compile_msl_kernel(backend, spec, module, "batchnorm2d_kernel", msl_generator, &log);
    OPENVINO_ASSERT(m_kernel, "MetalBatchNormOp: failed to compile kernel: ", log);

    MetalOp::compile(buffer_manager);
}

void MetalBatchNormOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(!inputs().empty(), "BatchNorm: missing input");
    MetalTensor* src = inputs()[0];
    OPENVINO_ASSERT(src && src->buf.valid(), "BatchNorm: input buffer null");
    MetalTensor& dst = require_output();

    ov::Shape out_shape = !dst.shape.empty() ? dst.shape : output_shape();
    if (out_shape.empty())
        out_shape = src->shape;
    OPENVINO_ASSERT(!out_shape.empty(), "BatchNorm: output shape unknown");

    if (!src->shape.empty() && m_node->get_input_partial_shape(0).is_static()) {
        OPENVINO_ASSERT(src->shape == m_node->get_input_shape(0),
                        "BatchNorm: runtime input shape mismatch");
    }
    const size_t bytes = ov::shape_size(out_shape) * m_element_type.size();
    if (!dst.buf.valid() || dst.buf.size < bytes) {
        dst.buf = allocate_temp_buffer(bytes, m_element_type, /*persistent=*/false, dst.prefer_private);
    }
    dst.shape = out_shape;
    dst.expected_type = m_element_type;
    const size_t elem_sz = m_element_type.size();
    const ov::Shape src_shape = !src->shape.empty() ? src->shape : out_shape;
    const size_t need_src = ov::shape_size(src_shape) * elem_sz;
    const size_t need_dst = ov::shape_size(out_shape) * elem_sz;
    OPENVINO_ASSERT(src->buf.size >= need_src, "BatchNorm: src buffer too small");
    OPENVINO_ASSERT(dst.buf.size >= need_dst, "BatchNorm: dst buffer too small");

    struct BNParamsGPU {
        uint32_t N;
        uint32_t C;
        uint32_t H;
        uint32_t W;
    } gpu_params{m_desc.N, m_desc.C, m_desc.H, m_desc.W};

    const uint32_t total = m_desc.N * m_desc.C * m_desc.H * m_desc.W;
    if (total == 0) {
        return;
    }
    KernelDispatch dispatch = make_1d_dispatch(total, m_kernel->clamp_threadgroup_size(64));

    KernelArgsBuilder args_builder(name().c_str());
    append_kernel_input_args(args_builder, 1, [&](size_t) { return src; }, name().c_str());
    args_builder.add_input_buffer(m_params_buf, "params");
    args_builder.add_output(&dst);
    args_builder.add_bytes(&gpu_params, sizeof(gpu_params));

    const auto args = args_builder.finalize(buffer_manager(), m_kernel.get());
    execute_kernel(*m_kernel, cmd_buf_handle, dispatch, args);
}

}  // namespace gfx_plugin
}  // namespace ov