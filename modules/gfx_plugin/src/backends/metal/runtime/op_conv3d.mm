// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/op_conv3d.hpp"

#include <string>

#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "runtime/gfx_logger.hpp"
#include "backends/metal/runtime/op_utils.hpp"
#include "kernel_ir/gfx_kernel_args.hpp"
#include "mlir/mlir_builder.hpp"
#include "mlir/codegen_common.hpp"
#include "backends/metal/codegen/metal_codegen_backend.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
}  // namespace

MetalConv3DOp::MetalConv3DOp(const std::shared_ptr<const ov::op::v1::Convolution>& node,
                             MetalDeviceHandle device,
                             MetalCommandQueueHandle queue)
    : MetalOp(node->get_friendly_name(),
              "Conv3D",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_node(node),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    OPENVINO_ASSERT(m_node, "MetalConv3DOp: node is null");

    const auto& in_shape = m_node->get_input_shape(0);   // NCDHW
    const auto& w_shape = m_node->get_input_shape(1);    // OIDHW
    OPENVINO_ASSERT(in_shape.size() == 5, "MetalConv3DOp: expected NCDHW input");
    OPENVINO_ASSERT(w_shape.size() == 5, "MetalConv3DOp: expected OIDHW weights");

    const auto strides = m_node->get_strides();
    const auto dilations = m_node->get_dilations();
    const auto pads_begin = m_node->get_pads_begin();
    const auto pads_end = m_node->get_pads_end();

    m_element_type = m_node->get_output_element_type(0);
    OPENVINO_ASSERT(m_element_type == ov::element::f16 || m_element_type == ov::element::f32,
                    "MetalConv3DOp supports only f16/f32");
    m_desc.element_type = m_element_type;

    m_desc.N = static_cast<uint32_t>(in_shape.at(0));
    m_desc.C_in = static_cast<uint32_t>(in_shape.at(1));
    m_desc.D = static_cast<uint32_t>(in_shape.at(2));
    m_desc.H = static_cast<uint32_t>(in_shape.at(3));
    m_desc.W = static_cast<uint32_t>(in_shape.at(4));
    m_desc.C_out = static_cast<uint32_t>(w_shape.at(0));
    m_desc.kD = static_cast<uint32_t>(w_shape.at(2));
    m_desc.kH = static_cast<uint32_t>(w_shape.at(3));
    m_desc.kW = static_cast<uint32_t>(w_shape.at(4));
    m_desc.strideD = static_cast<uint32_t>(strides.at(0));
    m_desc.strideH = static_cast<uint32_t>(strides.at(1));
    m_desc.strideW = static_cast<uint32_t>(strides.at(2));
    m_desc.dilationD = static_cast<uint32_t>(dilations.at(0));
    m_desc.dilationH = static_cast<uint32_t>(dilations.at(1));
    m_desc.dilationW = static_cast<uint32_t>(dilations.at(2));
    m_desc.padFront = static_cast<uint32_t>(pads_begin.at(0));
    m_desc.padTop = static_cast<uint32_t>(pads_begin.at(1));
    m_desc.padLeft = static_cast<uint32_t>(pads_begin.at(2));
    m_desc.padBack = static_cast<uint32_t>(pads_end.at(0));
    m_desc.padBottom = static_cast<uint32_t>(pads_end.at(1));
    m_desc.padRight = static_cast<uint32_t>(pads_end.at(2));

    const auto out_shape = m_node->get_output_shape(0);
    OPENVINO_ASSERT(out_shape.size() == 5, "MetalConv3DOp: expected NCDHW output");
    m_desc.outD = static_cast<uint32_t>(out_shape.at(2));
    m_desc.outH = static_cast<uint32_t>(out_shape.at(3));
    m_desc.outW = static_cast<uint32_t>(out_shape.at(4));
}

void MetalConv3DOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalConv3DOp::prepare_weights() {
    OPENVINO_ASSERT(buffer_manager(), "MetalConv3DOp: buffer manager is null");
    auto weights_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(m_node->get_input_node_shared_ptr(1));
    OPENVINO_ASSERT(weights_const, "MetalConv3DOp requires constant weights");

    const auto& et = weights_const->get_element_type();
    const size_t bytes = et.size() * shape_size(weights_const->get_shape());
    const std::string key = m_node->get_friendly_name() + "/weights";
    m_weights = buffer_manager()->wrap_const(key, weights_const->get_data_ptr(), bytes, et);
    OPENVINO_ASSERT(m_weights.valid(), "MetalConv3DOp: failed to wrap weights buffer");
}

void MetalConv3DOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }
    prepare_weights();
    OPENVINO_ASSERT(m_device, "MetalConv3DOp: Metal device is null");

    MetalCodegenBackend backend(m_device);
    std::string log;
    mlir::MLIRContext ctx;
    auto model = make_single_op_model(m_node);
    auto module = build_mlir_conv3d_from_model(model, ctx);
    Conv3DCodegenDesc desc = m_desc;
    desc.element_type = m_element_type == ov::element::dynamic ? ov::element::f32 : m_element_type;
    auto msl_desc = desc;
    auto msl_generator = [msl_desc](mlir::ModuleOp mod) { return generate_msl_from_mlir(mod, msl_desc); };

    KernelSpec spec(m_node, 4u);
    m_kernel = compile_msl_kernel(backend, spec, module, "conv3d_kernel", msl_generator, &log);
    OPENVINO_ASSERT(m_kernel, "MetalConv3DOp: failed to compile conv3d kernel: ", log);

    MetalOp::compile(buffer_manager);
}

void MetalConv3DOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(!inputs().empty(), "MetalConv3DOp: no inputs bound");
    MetalTensor* src = inputs().at(0);
    OPENVINO_ASSERT(src && src->buf.valid(), "MetalConv3DOp: input buffer is null");
    MetalTensor& dst_tensor = require_output();

    const ov::Shape in_shape = !src->shape.empty() ? src->shape : m_node->get_input_shape(0);
    OPENVINO_ASSERT(in_shape.size() == 5, "MetalConv3DOp: expected NCDHW input");
    OPENVINO_ASSERT(in_shape[0] == m_desc.N &&
                        in_shape[1] == m_desc.C_in &&
                        in_shape[2] == m_desc.D &&
                        in_shape[3] == m_desc.H &&
                        in_shape[4] == m_desc.W,
                    "MetalConv3DOp: runtime input shape differs from compiled shape");
    const size_t in_bytes = ov::shape_size(in_shape) * m_element_type.size();
    OPENVINO_ASSERT(src->buf.size >= in_bytes, "MetalConv3DOp: input buffer too small");

    const ov::Shape out_shape = !output_shape().empty() ? output_shape() : m_node->get_output_shape(0);
    OPENVINO_ASSERT(out_shape.size() == 5, "MetalConv3DOp: expected NCDHW output");

    const size_t out_bytes = ov::shape_size(out_shape) * m_element_type.size();
    if (!dst_tensor.buf.valid() || dst_tensor.buf.size < out_bytes) {
        const auto& et = m_node->get_output_element_type(0);
        size_t bytes = element_size();
        for (auto d : out_shape) bytes *= d;
        dst_tensor.buf = allocate_temp_buffer(bytes, et, /*persistent=*/false, dst_tensor.prefer_private);
        dst_tensor.expected_type = et;
    }
    dst_tensor.shape = out_shape;
    dst_tensor.expected_type = m_node->get_output_element_type(0);

    struct Conv3DParams {
        uint32_t N, C_in, D, H, W;
        uint32_t C_out;
        uint32_t kD, kH, kW;
        uint32_t strideD, strideH, strideW;
        uint32_t dilationD, dilationH, dilationW;
        uint32_t padFront, padTop, padLeft, padBack, padBottom, padRight;
        uint32_t outD, outH, outW;
    } params{};
    params.N = m_desc.N;
    params.C_in = m_desc.C_in;
    params.D = m_desc.D;
    params.H = m_desc.H;
    params.W = m_desc.W;
    params.C_out = m_desc.C_out;
    params.kD = m_desc.kD;
    params.kH = m_desc.kH;
    params.kW = m_desc.kW;
    params.strideD = m_desc.strideD;
    params.strideH = m_desc.strideH;
    params.strideW = m_desc.strideW;
    params.dilationD = m_desc.dilationD;
    params.dilationH = m_desc.dilationH;
    params.dilationW = m_desc.dilationW;
    params.padFront = m_desc.padFront;
    params.padTop = m_desc.padTop;
    params.padLeft = m_desc.padLeft;
    params.padBack = m_desc.padBack;
    params.padBottom = m_desc.padBottom;
    params.padRight = m_desc.padRight;
    params.outD = static_cast<uint32_t>(out_shape[2]);
    params.outH = static_cast<uint32_t>(out_shape[3]);
    params.outW = static_cast<uint32_t>(out_shape[4]);
    if (m_desc.C_out == 0 || m_desc.N == 0) {
        return;
    }
    KernelDispatch dispatch =
        make_2d_dispatch(m_desc.C_out, m_desc.N, m_kernel->clamp_threadgroup_size(8));

    KernelArgsBuilder args_builder(name().c_str());
    append_kernel_input_args(args_builder, 1, [&](size_t) { return src; }, name().c_str());
    args_builder.add_input_buffer(m_weights, "weights");
    args_builder.add_output(&dst_tensor);
    args_builder.add_bytes(&params, sizeof(params));

    const auto args = args_builder.finalize(buffer_manager(), m_kernel.get());
    execute_kernel(*m_kernel, cmd_buf_handle, dispatch, args);
}

}  // namespace gfx_plugin
}  // namespace ov