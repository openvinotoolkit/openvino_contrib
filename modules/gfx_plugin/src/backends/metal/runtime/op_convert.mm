// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/op_convert.hpp"

#include "backends/metal/codegen/metal_codegen_backend.hpp"
#include "mlir/mlir_builder.hpp"
#include "runtime/gfx_logger.hpp"
#include "backends/metal/runtime/op_utils.hpp"
#include "kernel_ir/gfx_kernel_args.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir_codegen/codegen_common.hpp"

namespace ov {
namespace gfx_plugin {

MetalConvertOp::MetalConvertOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "Convert",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
    m_node(node),
    m_device((id<MTLDevice>)device),
    m_queue((id<MTLCommandQueue>)queue) {
    auto cvt = ov::as_type_ptr<const ov::op::v0::Convert>(node);
    OPENVINO_ASSERT(cvt, "MetalConvertOp expects v0::Convert");
    m_src_type = cvt->get_input_element_type(0);
    m_dst_type = cvt->get_output_element_type(0);
}

void MetalConvertOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalConvertOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }
    MetalCodegenBackend backend(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    ConvertCodegenDesc desc{};
    desc.src_type = m_src_type;
    desc.dst_type = m_dst_type;
    desc.element_type = m_dst_type == ov::element::dynamic ? ov::element::f32 : m_dst_type;
    mlir::MLIRContext ctx;
    auto module = build_mlir_convert_from_model(make_single_op_model(m_node), ctx);
    auto source = generate_msl_from_mlir(module, desc);

    KernelSpec spec(m_node, 3u);
    m_kernel = compile_msl_kernel(backend, spec, module, "convert_kernel", source, &log);
    OPENVINO_ASSERT(m_kernel, "MetalConvertOp: failed to compile kernel: ", log);

    MetalOp::compile(buffer_manager);
}

void MetalConvertOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(!inputs().empty(), "Convert: missing input");
    MetalTensor* src = inputs()[0];
    OPENVINO_ASSERT(src && src->buf.valid(), "Convert: input buffer null");
    MetalTensor& dst = require_output();

    ov::Shape out_shape = !dst.shape.empty() ? dst.shape : output_shape();
    if (out_shape.empty())
        out_shape = src->shape;
    OPENVINO_ASSERT(!out_shape.empty(), "Convert: unknown shape");

    if (!src->shape.empty() && m_node->get_input_partial_shape(0).is_static()) {
        OPENVINO_ASSERT(src->shape == m_node->get_input_shape(0),
                        "Convert: runtime input shape mismatch");
    }
    const size_t src_bytes = ov::shape_size(src->shape.empty() ? out_shape : src->shape) * m_src_type.size();
    OPENVINO_ASSERT(src->buf.size >= src_bytes, "Convert: input buffer too small");

    const size_t num_elems = ov::shape_size(out_shape);
    if (!dst.buf.valid()) {
        size_t bytes = num_elems * m_dst_type.size();
        dst.buf = buffer_manager()->allocate(bytes, m_dst_type, /*persistent=*/false, dst.prefer_private);
    }
    dst.shape = out_shape;
    dst.expected_type = m_dst_type;

    if (m_src_type == m_dst_type) {
        // Trivial case: reuse buffer.
        dst.buf = src->buf;
        start_profiling(nullptr);
        stop_profiling_ms(nullptr);
        return;
    }

    uint32_t n = static_cast<uint32_t>(num_elems);
    if (n == 0) {
        return;
    }
    KernelDispatch dispatch = make_1d_dispatch(n, m_kernel->clamp_threadgroup_size(64));

    std::vector<KernelArg> args;
    args.reserve(3);
    append_kernel_input_args(args, 1, [&](size_t) { return src; }, name().c_str());
    append_kernel_output_args(args, 1, &dst, name().c_str());
    args.push_back(make_bytes_arg(2, &n, sizeof(n)));
    execute_kernel(*m_kernel, cmd_buf_handle, dispatch, args);
}

}  // namespace gfx_plugin
}  // namespace ov
