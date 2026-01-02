// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/op_reduce.hpp"

#include <numeric>

#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "backends/metal/codegen/metal_codegen_backend.hpp"
#include "runtime/gfx_logger.hpp"
#include "runtime/gfx_shape_utils.hpp"
#include "backends/metal/runtime/op_utils.hpp"
#include "kernel_ir/gfx_kernel_args.hpp"
#include "mlir/mlir_builder.hpp"
#include "mlir_codegen/codegen_common.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

namespace ov {
namespace gfx_plugin {

namespace {
std::vector<int> make_strides(const ov::Shape& shp) {
    if (shp.empty()) return {1};
    std::vector<int> st(shp.size(), 1);
    for (int i = static_cast<int>(shp.size()) - 2; i >= 0; --i) {
        st[i] = st[i + 1] * static_cast<int>(shp[i + 1]);
    }
    return st;
}
}  // namespace

MetalReduceOp::MetalReduceOp(const std::shared_ptr<const ov::Node>& node,
                             ReduceKind kind,
                             void* device,
                             void* queue)
    : MetalOp(node->get_friendly_name(),
              "Reduce",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_kind(kind),
      m_node(node),
      m_element_type(node->get_output_element_type(0)),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
}

void MetalReduceOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalReduceOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }

    MetalCodegenBackend backend(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    mlir::MLIRContext ctx;
    std::shared_ptr<ov::Model> model = make_single_op_model(m_node);
    mlir::ModuleOp module;
    switch (m_kind) {
        case ReduceKind::Sum:
            module = build_mlir_reducesum_from_model(model, ctx);
            break;
        case ReduceKind::Mean:
            module = build_mlir_reducemean_from_model(model, ctx);
            break;
        case ReduceKind::Max:
            module = build_mlir_reducemax_from_model(model, ctx);
            break;
        case ReduceKind::Min:
            module = build_mlir_reducemin_from_model(model, ctx);
            break;
        case ReduceKind::Prod:
            module = build_mlir_reduceprod_from_model(model, ctx);
            break;
        case ReduceKind::L1:
            module = build_mlir_reducel1_from_model(model, ctx);
            break;
        case ReduceKind::L2:
            module = build_mlir_reducel2_from_model(model, ctx);
            break;
        default:
            module = build_mlir_reducesum_from_model(model, ctx);
            break;
    }
    ReduceCodegenDesc desc{};
    desc.kind = m_kind;
    desc.element_type = m_element_type;
    auto msl_desc = desc;
    auto msl_generator = [msl_desc](mlir::ModuleOp mod) { return generate_msl_from_mlir(mod, msl_desc); };

    KernelSpec spec(m_node, 9u);
    m_kernel = compile_msl_kernel(backend, spec, module, "reduce_kernel", msl_generator, &log);
    OPENVINO_ASSERT(m_kernel, "MetalReduceOp: failed to compile kernel: ", log);

    // Prepare shape metadata
    ov::Shape in_shape = m_node->get_input_shape(0);
    ov::Shape out_shape = output_shape();
    m_num_elems = static_cast<uint32_t>(ov::shape_size(out_shape));

    m_in_dims.assign(in_shape.begin(), in_shape.end());
    if (m_in_dims.empty()) m_in_dims.push_back(1);
    m_out_dims.assign(out_shape.begin(), out_shape.end());
    if (m_out_dims.empty()) m_out_dims.push_back(1);

    m_in_strides = make_strides(in_shape);
    if (m_in_strides.empty()) m_in_strides.push_back(1);

    m_axes_mask.assign(m_in_dims.size(), 0);
    m_reduce_dims.assign(m_in_dims.size(), 1);
    auto axes_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(m_node->input_value(1).get_node_shared_ptr());
    OPENVINO_ASSERT(axes_const, "Reduce axes must be constant");
    auto axes_vec = axes_const->cast_vector<int64_t>();
    for (auto ax : axes_vec) {
        int axis = static_cast<int>(normalize_axis(ax, m_in_dims.size(), "Reduce"));
        m_axes_mask[axis] = 1;
        m_reduce_dims[axis] = m_in_dims[axis];
    }

    MetalOp::compile(buffer_manager);
}

void MetalReduceOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(inputs().size() >= 1, "Reduce: missing input");
    MetalTensor* src = inputs()[0];
    OPENVINO_ASSERT(src && src->buf.valid(), "Reduce: input buffer null");

    MetalTensor& dst = require_output();
    ov::Shape in_shape = !src->shape.empty() ? src->shape : ov::Shape{};
    if (in_shape.empty() && m_node->get_input_partial_shape(0).is_static()) {
        in_shape = m_node->get_input_shape(0);
    }
    ov::Shape out_shape = !dst.shape.empty() ? dst.shape : output_shape();
    if (out_shape.empty() && m_node->get_output_partial_shape(0).is_static()) {
        out_shape = m_node->get_output_shape(0);
    }
    OPENVINO_ASSERT(!in_shape.empty() && !out_shape.empty(), "Reduce: runtime shape unknown");
    OPENVINO_ASSERT(in_shape.size() <= 8, "Reduce: rank > 8 not supported by kernel");

    m_in_dims.assign(in_shape.begin(), in_shape.end());
    if (m_in_dims.empty())
        m_in_dims.push_back(1);
    m_out_dims.assign(out_shape.begin(), out_shape.end());
    if (m_out_dims.empty())
        m_out_dims.push_back(1);
    m_in_strides = make_strides(in_shape);
    if (m_in_strides.empty())
        m_in_strides.push_back(1);

    m_axes_mask.assign(m_in_dims.size(), 0);
    m_reduce_dims.assign(m_in_dims.size(), 1);
    auto axes_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(
        m_node->input_value(1).get_node_shared_ptr());
    OPENVINO_ASSERT(axes_const, "Reduce axes must be constant");
    auto axes_vec = axes_const->cast_vector<int64_t>();
    for (auto ax : axes_vec) {
        int axis = static_cast<int>(normalize_axis(ax, m_in_dims.size(), "Reduce"));
        m_axes_mask[axis] = 1;
        m_reduce_dims[axis] = m_in_dims[axis];
    }

    m_num_elems = static_cast<uint32_t>(ov::shape_size(out_shape));
    const size_t in_bytes = m_element_type.size() * ov::shape_size(in_shape);
    OPENVINO_ASSERT(src->buf.size >= in_bytes, "Reduce: input buffer too small");
    const size_t bytes = m_element_type.size() * static_cast<size_t>(m_num_elems);
    if (!dst.buf.valid() || dst.buf.size < bytes) {
        dst.buf = allocate_temp_buffer(bytes, m_element_type, /*persistent=*/false, dst.prefer_private);
    }
    dst.expected_type = m_element_type;
    dst.shape = out_shape;
    if (m_num_elems == 0) {
        return;
    }

    uint32_t num = m_num_elems;
    uint32_t rank = static_cast<uint32_t>(m_in_dims.size());
    KernelDispatch dispatch = make_1d_dispatch(num, m_kernel->clamp_threadgroup_size(64));

    KernelArgsBuilder args_builder(name().c_str());
    append_kernel_input_args(args_builder, 1, [&](size_t) { return src; }, name().c_str());
    args_builder.add_output(&dst);
    args_builder.add_bytes(&num, sizeof(num));
    args_builder.add_bytes(&rank, sizeof(rank));
    args_builder.add_bytes(m_out_dims.data(), m_out_dims.size() * sizeof(int));
    args_builder.add_bytes(m_in_dims.data(), m_in_dims.size() * sizeof(int));
    args_builder.add_bytes(m_in_strides.data(), m_in_strides.size() * sizeof(int));
    args_builder.add_bytes(m_axes_mask.data(), m_axes_mask.size() * sizeof(int));
    args_builder.add_bytes(m_reduce_dims.data(), m_reduce_dims.size() * sizeof(int));

    const auto args = args_builder.finalize(buffer_manager(), m_kernel.get());
    execute_kernel(*m_kernel, cmd_buf_handle, dispatch, args);

    dst.expected_type = m_element_type;
}

}  // namespace gfx_plugin
}  // namespace ov