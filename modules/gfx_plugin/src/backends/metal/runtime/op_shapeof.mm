// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/op_shapeof.hpp"

#include "openvino/core/shape_util.hpp"
#include "backends/metal/codegen/metal_codegen_backend.hpp"
#include "mlir/gfx_mlir_kernel_builder.hpp"
#include "runtime/gfx_logger.hpp"
#include "backends/metal/runtime/op_utils.hpp"
#include "kernel_ir/gfx_kernel_args.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/codegen_common.hpp"

namespace ov {
namespace gfx_plugin {

MetalShapeOfOp::MetalShapeOfOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "ShapeOf",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_node(node),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    m_element_type = node->get_output_element_type(0);
    OPENVINO_ASSERT(m_element_type == ov::element::i32 || m_element_type == ov::element::i64,
                    "ShapeOf: output must be i32/i64");
    const auto in_pshape = node->get_input_partial_shape(0);
    OPENVINO_ASSERT(in_pshape.rank().is_static(), "ShapeOf: input rank must be static");
    const size_t rank = static_cast<size_t>(in_pshape.rank().get_length());
    m_shape_vals.assign(rank, 0);
    if (in_pshape.is_static()) {
        const auto in_shape = node->get_input_shape(0);
        m_shape_vals.assign(in_shape.begin(), in_shape.end());
    }
    m_desc.rank = static_cast<uint32_t>(rank);
    m_desc.element_type = m_element_type;
}

void MetalShapeOfOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalShapeOfOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }
    MetalCodegenBackend backend(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    auto& ctx = gfx_mlir_context();
    auto module = build_mlir_for_node(m_node, ctx);
    auto msl_desc = m_desc;
    auto msl_generator = [msl_desc](mlir::ModuleOp mod) { return generate_msl_from_mlir(mod, msl_desc); };

    KernelSpec spec(m_node, 4u);
    m_kernel = compile_msl_kernel(backend, spec, module, "shapeof_kernel", msl_generator, &log);
    OPENVINO_ASSERT(m_kernel, "MetalShapeOfOp: failed to compile kernel: ", log);

    MetalOp::compile(buffer_manager);
}

void MetalShapeOfOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(!inputs().empty(), "ShapeOf: missing input");
    MetalTensor* src = inputs()[0];
    OPENVINO_ASSERT(src && src->buf.valid(), "ShapeOf: input buffer null");

    std::vector<int64_t> shape_vals;
    if (!src->shape.empty()) {
        shape_vals.assign(src->shape.begin(), src->shape.end());
    } else {
        shape_vals = m_shape_vals;
    }
    OPENVINO_ASSERT(!shape_vals.empty(), "ShapeOf: input shape is unknown");

    MetalTensor& dst = require_output();
    ov::Shape out_shape = ov::Shape{shape_vals.size()};
    const size_t bytes = ov::shape_size(out_shape) * m_element_type.size();
    if (!dst.buf.valid() || dst.buf.size < bytes) {
        dst.buf = allocate_temp_buffer(bytes, m_element_type, /*persistent=*/false, dst.prefer_private);
    }
    dst.shape = out_shape;
    dst.expected_type = m_element_type;

    const uint32_t rank = static_cast<uint32_t>(shape_vals.size());

    KernelArgsBuilder args_builder(name().c_str());
    append_kernel_input_args(args_builder, 1, [&](size_t) { return src; }, name().c_str());
    args_builder.add_output(&dst);
    args_builder.add_bytes(&rank, sizeof(rank));
    if (m_element_type == ov::element::i32) {
        std::vector<int32_t> tmp(rank);
        for (size_t i = 0; i < shape_vals.size(); ++i) tmp[i] = static_cast<int32_t>(shape_vals[i]);
        args_builder.add_bytes(tmp.data(), tmp.size() * sizeof(int32_t));
        if (rank == 0) {
            return;
        }
        KernelDispatch dispatch = make_1d_dispatch(rank, m_kernel->clamp_threadgroup_size(64));
        const auto args = args_builder.finalize(buffer_manager(), m_kernel.get());
        execute_kernel(*m_kernel, cmd_buf_handle, dispatch, args);
    } else {
        args_builder.add_bytes(shape_vals.data(), shape_vals.size() * sizeof(int64_t));
        if (rank == 0) {
            return;
        }
        KernelDispatch dispatch = make_1d_dispatch(rank, m_kernel->clamp_threadgroup_size(64));
        const auto args = args_builder.finalize(buffer_manager(), m_kernel.get());
        execute_kernel(*m_kernel, cmd_buf_handle, dispatch, args);
    }
}

}  // namespace gfx_plugin
}  // namespace ov
