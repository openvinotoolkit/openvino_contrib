// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/op_depth_to_space.hpp"

#include "backends/metal/codegen/metal_codegen_backend.hpp"
#include "mlir/mlir_builder.hpp"
#include "mlir_codegen/codegen_common.hpp"
#include "openvino/core/shape_util.hpp"
#include "runtime/gfx_logger.hpp"
#include "backends/metal/runtime/op_utils.hpp"
#include "kernel_ir/gfx_kernel_args.hpp"
#include "mlir/IR/MLIRContext.h"

namespace ov {
namespace gfx_plugin {

namespace {
}  // namespace

MetalDepthToSpaceOp::MetalDepthToSpaceOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "DepthToSpace",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_node(node),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    parse_depth_to_space(node);
}

void MetalDepthToSpaceOp::parse_depth_to_space(const std::shared_ptr<const ov::Node>& node) {
    auto d2s = ov::as_type_ptr<const ov::op::v0::DepthToSpace>(node);
    OPENVINO_ASSERT(d2s, "MetalDepthToSpaceOp expects DepthToSpace v0");

    const auto in_shape = node->get_input_shape(0);
    const auto out_shape = node->get_output_shape(0);
    OPENVINO_ASSERT(in_shape.size() == 4, "DepthToSpace: only 4D NCHW supported");
    OPENVINO_ASSERT(out_shape.size() == 4, "DepthToSpace: output must be 4D");

    m_block = static_cast<uint32_t>(d2s->get_block_size());
    m_mode = (d2s->get_mode() == ov::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST) ? 1u : 0u;

    const uint32_t N = static_cast<uint32_t>(in_shape[0]);
    const uint32_t C = static_cast<uint32_t>(in_shape[1]);
    const uint32_t H = static_cast<uint32_t>(in_shape[2]);
    const uint32_t W = static_cast<uint32_t>(in_shape[3]);
    const uint32_t C_out = static_cast<uint32_t>(out_shape[1]);
    const uint32_t H_out = static_cast<uint32_t>(out_shape[2]);
    const uint32_t W_out = static_cast<uint32_t>(out_shape[3]);

    OPENVINO_ASSERT(m_block > 0, "DepthToSpace: block size must be > 0");
    OPENVINO_ASSERT(C == C_out * m_block * m_block, "DepthToSpace: invalid channel dimension");
    OPENVINO_ASSERT(H_out == H * m_block && W_out == W * m_block, "DepthToSpace: invalid spatial dimensions");

    m_element_type = node->get_output_element_type(0);
    OPENVINO_ASSERT(m_element_type == ov::element::f16 || m_element_type == ov::element::f32 ||
                        m_element_type == ov::element::i32 || m_element_type == ov::element::i64,
                    "DepthToSpace: element type not supported");

    m_desc.element_type = m_element_type == ov::element::dynamic ? ov::element::f32 : m_element_type;
    m_desc.N = N;
    m_desc.C = C;
    m_desc.H = H;
    m_desc.W = W;
    m_desc.C_out = C_out;
    m_desc.H_out = H_out;
    m_desc.W_out = W_out;
    m_desc.block = m_block;
    m_desc.mode = m_mode;
    m_desc.total = static_cast<uint32_t>(ov::shape_size(out_shape));
}

void MetalDepthToSpaceOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalDepthToSpaceOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }

    MetalCodegenBackend backend(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    mlir::MLIRContext ctx;
    auto module = build_mlir_depth_to_space_from_model(make_single_op_model(m_node), ctx);
    auto msl_desc = m_desc;
    auto msl_generator = [msl_desc](mlir::ModuleOp mod) { return generate_msl_from_mlir(mod, msl_desc); };

    KernelSpec spec(m_node, 3u);
    m_kernel = compile_msl_kernel(backend, spec, module, "depth_to_space_kernel", msl_generator, &log);
    OPENVINO_ASSERT(m_kernel, "MetalDepthToSpaceOp: failed to compile kernel: ", log);

    MetalOp::compile(buffer_manager);
}

void MetalDepthToSpaceOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(inputs().size() >= 1, "DepthToSpace: missing input");
    MetalTensor* src = inputs()[0];
    OPENVINO_ASSERT(src && src->buf.valid(), "DepthToSpace: input buffer null");

    MetalTensor& dst = require_output();
    ov::Shape out_shape = !output_shape().empty() ? output_shape() : ov::Shape{};
    if (out_shape.empty()) {
        out_shape = m_node->get_output_shape(0);
    }

    ov::Shape in_shape = !src->shape.empty() ? src->shape : m_node->get_input_shape(0);
    OPENVINO_ASSERT(!in_shape.empty(), "DepthToSpace: input shape unknown");
    if (m_node->get_input_partial_shape(0).is_static()) {
        OPENVINO_ASSERT(in_shape == m_node->get_input_shape(0),
                        "DepthToSpace: runtime input shape mismatch");
    }
    const size_t in_bytes = ov::shape_size(in_shape) * m_element_type.size();
    OPENVINO_ASSERT(src->buf.size >= in_bytes, "DepthToSpace: input buffer too small");
    const size_t bytes = ov::shape_size(out_shape) * m_element_type.size();
    if (!dst.buf.valid() || dst.buf.size < bytes) {
        dst.buf = allocate_temp_buffer(bytes, m_element_type, /*persistent=*/false, dst.prefer_private);
    }
    dst.shape = out_shape;
    dst.expected_type = m_element_type;

    struct DepthToSpaceParams {
        uint32_t N; uint32_t C; uint32_t H; uint32_t W;
        uint32_t C_out; uint32_t H_out; uint32_t W_out;
        uint32_t block; uint32_t mode; uint32_t total;
    } params{};

    params.N = m_desc.N;
    params.C = m_desc.C;
    params.H = m_desc.H;
    params.W = m_desc.W;
    params.C_out = m_desc.C_out;
    params.H_out = m_desc.H_out;
    params.W_out = m_desc.W_out;
    params.block = m_desc.block;
    params.mode = m_desc.mode;
    params.total = m_desc.total;

    if (params.total == 0) {
        return;
    }
    KernelDispatch dispatch = make_1d_dispatch(static_cast<size_t>(params.total), m_kernel->clamp_threadgroup_size(256));

    KernelArgsBuilder args_builder(name().c_str());
    append_kernel_input_args(args_builder, 1, [&](size_t) { return src; }, name().c_str());
    args_builder.add_output(&dst);
    args_builder.add_bytes(&params, sizeof(params));

    const auto args = args_builder.finalize(buffer_manager(), m_kernel.get());
    execute_kernel(*m_kernel, cmd_buf_handle, dispatch, args);
}

}  // namespace gfx_plugin
}  // namespace ov