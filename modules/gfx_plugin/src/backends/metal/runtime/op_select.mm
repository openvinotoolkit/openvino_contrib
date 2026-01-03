// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/op_select.hpp"

#include <limits>

#include "openvino/core/except.hpp"
#include "openvino/op/parameter.hpp"
#include "backends/metal/codegen/metal_codegen_backend.hpp"
#include "runtime/gfx_logger.hpp"
#include "runtime/gfx_shape_utils.hpp"
#include "backends/metal/runtime/op_utils.hpp"
#include "kernel_ir/gfx_kernel_args.hpp"
#include "mlir/mlir_builder.hpp"
#include "mlir/codegen_common.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

namespace ov {
namespace gfx_plugin {

namespace {
}  // namespace

MetalSelectOp::MetalSelectOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "Select",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_node(node),
      m_element_type(node->get_output_element_type(0)),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {}

void MetalSelectOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalSelectOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }

    MetalCodegenBackend backend(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    mlir::MLIRContext ctx;
    auto module = build_mlir_select_from_model(make_single_op_model(m_node), ctx);
    const auto msl_type = m_element_type;
    auto msl_generator = [msl_type](mlir::ModuleOp mod) { return generate_msl_for_select(mod, msl_type); };

    KernelSpec spec(m_node, 10u);
    m_kernel = compile_msl_kernel(backend, spec, module, "select_kernel", msl_generator, &log);
    OPENVINO_ASSERT(m_kernel, "MetalSelectOp: failed to compile select kernel: ", log);

    m_num_elems = static_cast<uint32_t>(ov::shape_size(output_shape()));
    MetalOp::compile(buffer_manager);
}

void MetalSelectOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(inputs().size() >= 3, "Select: requires 3 inputs (cond, then, else)");
    MetalTensor* cond = inputs()[0];
    MetalTensor* tval = inputs()[1];
    MetalTensor* fval = inputs()[2];
    OPENVINO_ASSERT(cond && cond->buf.valid(), "Select: cond buffer null");
    OPENVINO_ASSERT(tval && tval->buf.valid(), "Select: true buffer null");
    OPENVINO_ASSERT(fval && fval->buf.valid(), "Select: false buffer null");

    MetalTensor& out = require_output();
    ov::Shape out_shape = !out.shape.empty() ? out.shape : output_shape();
    if (out_shape.empty() && m_node->get_output_partial_shape(0).is_static()) {
        out_shape = m_node->get_output_shape(0);
    }
    OPENVINO_ASSERT(!out_shape.empty(), "Select: output shape unknown");
    const size_t bytes = m_element_type.size() * ov::shape_size(out_shape);
    if (!out.buf.valid() || out.buf.size < bytes) {
        out.buf = allocate_temp_buffer(bytes, m_element_type, /*persistent=*/false, out.prefer_private);
    }
    out.expected_type = m_element_type;
    out.shape = out_shape;

    // Compute broadcast strides for cond, tval, fval to output.
    auto make_stride = [](const ov::Shape& shp) {
        std::vector<int> st(shp.size(), 1);
        if (shp.empty()) return st;
        for (int i = static_cast<int>(shp.size()) - 2; i >= 0; --i) {
            st[i] = st[i + 1] * static_cast<int>(shp[i + 1]);
        }
        return st;
    };
    const size_t rank = out_shape.empty() ? 1 : out_shape.size();
    std::vector<int> out_dims(rank, 1);
    for (size_t i = 0; i < out_shape.size(); ++i) out_dims[i] = static_cast<int>(out_shape[i]);

    auto norm_shape = [&](const ov::Shape& s) {
        ov::Shape r(rank, 1);
        if (s.empty()) return r;
        size_t off = rank - s.size();
        for (size_t i = 0; i < s.size(); ++i) r[off + i] = s[i];
        return r;
    };

    auto cond_norm = norm_shape(cond->shape);
    auto a_norm = norm_shape(tval->shape);
    auto b_norm = norm_shape(fval->shape);

    auto elem_size_for = [](const MetalTensor* t) -> size_t {
        if (!t) return 0;
        const auto et = t->expected_type == ov::element::dynamic ? t->buf.type : t->expected_type;
        return et.size();
    };
    const size_t cond_bytes =
        static_cast<size_t>(shape_product(cond_norm, 0, cond_norm.size())) * elem_size_for(cond);
    const size_t a_bytes =
        static_cast<size_t>(shape_product(a_norm, 0, a_norm.size())) * elem_size_for(tval);
    const size_t b_bytes =
        static_cast<size_t>(shape_product(b_norm, 0, b_norm.size())) * elem_size_for(fval);
    OPENVINO_ASSERT(cond->buf.size >= cond_bytes, "Select: cond buffer too small");
    OPENVINO_ASSERT(tval->buf.size >= a_bytes, "Select: true buffer too small");
    OPENVINO_ASSERT(fval->buf.size >= b_bytes, "Select: false buffer too small");

    auto cond_stride_full = make_stride(cond_norm);
    auto a_stride_full = make_stride(a_norm);
    auto b_stride_full = make_stride(b_norm);

    std::vector<int> stride_c(rank, 0), stride_a(rank, 0), stride_b(rank, 0);
    for (size_t d = 0; d < rank; ++d) {
        stride_c[d] = cond_norm[d] == 1 ? 0 : cond_stride_full[d];
        stride_a[d] = a_norm[d] == 1 ? 0 : a_stride_full[d];
        stride_b[d] = b_norm[d] == 1 ? 0 : b_stride_full[d];
    }

    uint32_t num = m_num_elems ? m_num_elems : static_cast<uint32_t>(ov::shape_size(out.shape));
    uint32_t r = static_cast<uint32_t>(rank ? rank : 1);
    if (num == 0) {
        return;
    }
    KernelDispatch dispatch = make_1d_dispatch(num, m_kernel->clamp_threadgroup_size(64));

    KernelArgsBuilder args_builder(name().c_str());
    append_kernel_input_args(args_builder,
                             3,
                             [&](size_t idx) {
                                 switch (idx) {
                                 case 0:
                                     return cond;
                                 case 1:
                                     return tval;
                                 default:
                                     return fval;
                                 }
                             },
                             name().c_str());
    args_builder.add_output(&out);
    args_builder.add_bytes(&num, sizeof(num));
    args_builder.add_bytes(&r, sizeof(r));
    args_builder.add_bytes(out_dims.data(), out_dims.size() * sizeof(int));
    args_builder.add_bytes(stride_c.data(), stride_c.size() * sizeof(int));
    args_builder.add_bytes(stride_a.data(), stride_a.size() * sizeof(int));
    args_builder.add_bytes(stride_b.data(), stride_b.size() * sizeof(int));

    const auto args = args_builder.finalize(buffer_manager(), m_kernel.get());
    execute_kernel(*m_kernel, cmd_buf_handle, dispatch, args);

    out.expected_type = m_element_type;
}

}  // namespace gfx_plugin
}  // namespace ov
