// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/op_interpolate.hpp"

#include <algorithm>

#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/util/common_util.hpp"
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

namespace {
std::vector<int64_t> read_axes_const(const std::shared_ptr<const ov::Node>& node, size_t idx) {
    if (idx >= node->get_input_size())
        return {};
    auto axes_const = ov::as_type_ptr<const ov::op::v0::Constant>(node->get_input_node_shared_ptr(idx));
    if (!axes_const)
        return {};
    return axes_const->cast_vector<int64_t>();
}

bool axes_hw_only(const std::vector<int64_t>& axes) {
    if (axes.empty())
        return true;  // default to HW for 4D
    if (axes.size() != 2)
        return false;
    std::vector<int64_t> a = axes;
    std::sort(a.begin(), a.end());
    return a[0] == 2 && a[1] == 3;
}

void check_pads_zero(const std::vector<size_t>& pads) {
    for (auto v : pads) {
        OPENVINO_ASSERT(v == 0, "Interpolate: padding not supported");
    }
}
}  // namespace

MetalInterpolateOp::MetalInterpolateOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "Interpolate",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_node(node),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    parse_interpolate(node);
}

void MetalInterpolateOp::parse_interpolate(const std::shared_ptr<const ov::Node>& node) {
    m_element_type = node->get_output_element_type(0);
    OPENVINO_ASSERT(m_element_type == ov::element::f16 || m_element_type == ov::element::f32,
                    "Interpolate supports only f16/f32");
    const auto in_shape = node->get_input_shape(0);
    const auto out_shape = node->get_output_shape(0);
    OPENVINO_ASSERT(in_shape.size() == 4 && out_shape.size() == 4, "Interpolate supports NCHW rank4 only");

    bool nearest = true;
    bool align_corners = false;
    bool use_half_pixel = true;
    uint32_t nearest_mode = 0;

    if (auto v0 = ov::as_type_ptr<const ov::op::v0::Interpolate>(node)) {
        const auto& attrs = v0->get_attrs();
        const auto mode = ov::util::to_lower(attrs.mode);
        if (mode == "nearest") {
            nearest = true;
        } else if (mode == "linear") {
            nearest = false;
        } else {
            OPENVINO_THROW("Interpolate v0: mode not supported: ", attrs.mode);
        }
        align_corners = attrs.align_corners;
        use_half_pixel = !align_corners;
        OPENVINO_ASSERT(axes_hw_only(std::vector<int64_t>(attrs.axes.begin(), attrs.axes.end())),
                        "Interpolate v0: only axes {2,3} supported");
        check_pads_zero(attrs.pads_begin);
        check_pads_zero(attrs.pads_end);
    } else if (auto v4 = ov::as_type_ptr<const ov::op::v4::Interpolate>(node)) {
        const auto& attrs = v4->get_attrs();
        using Base = ov::op::util::InterpolateBase;
        switch (attrs.mode) {
            case Base::InterpolateMode::NEAREST:
                nearest = true;
                break;
            case Base::InterpolateMode::LINEAR:
            case Base::InterpolateMode::LINEAR_ONNX:
            case Base::InterpolateMode::BILINEAR_PILLOW:
                nearest = false;
                break;
            default:
                OPENVINO_THROW("Interpolate v4: mode not supported");
        }
        switch (attrs.coordinate_transformation_mode) {
            case Base::CoordinateTransformMode::HALF_PIXEL:
                align_corners = false;
                use_half_pixel = true;
                break;
            case Base::CoordinateTransformMode::ALIGN_CORNERS:
                align_corners = true;
                use_half_pixel = true;
                break;
            case Base::CoordinateTransformMode::ASYMMETRIC:
                align_corners = false;
                use_half_pixel = false;
                break;
            default:
                OPENVINO_THROW("Interpolate v4: coordinate_transformation_mode not supported");
        }
        switch (attrs.nearest_mode) {
            case Base::NearestMode::FLOOR:
                nearest_mode = 1;
                break;
            case Base::NearestMode::CEIL:
                nearest_mode = 2;
                break;
            case Base::NearestMode::ROUND_PREFER_FLOOR:
                nearest_mode = 1;
                break;
            case Base::NearestMode::ROUND_PREFER_CEIL:
                nearest_mode = 2;
                break;
            case Base::NearestMode::SIMPLE:
            default:
                nearest_mode = 0;
                break;
        }
        check_pads_zero(attrs.pads_begin);
        check_pads_zero(attrs.pads_end);
        std::vector<int64_t> axes;
        if (node->get_input_size() == 4) {
            axes = read_axes_const(node, 3);
        }
        OPENVINO_ASSERT(axes_hw_only(axes), "Interpolate v4: only axes {2,3} supported");
    } else if (auto v11 = ov::as_type_ptr<const ov::op::v11::Interpolate>(node)) {
        const auto& attrs = v11->get_attrs();
        using Base = ov::op::util::InterpolateBase;
        switch (attrs.mode) {
            case Base::InterpolateMode::NEAREST:
                nearest = true;
                break;
            case Base::InterpolateMode::LINEAR:
            case Base::InterpolateMode::LINEAR_ONNX:
            case Base::InterpolateMode::BILINEAR_PILLOW:
                nearest = false;
                break;
            default:
                OPENVINO_THROW("Interpolate v11: mode not supported");
        }
        switch (attrs.coordinate_transformation_mode) {
            case Base::CoordinateTransformMode::HALF_PIXEL:
                align_corners = false;
                use_half_pixel = true;
                break;
            case Base::CoordinateTransformMode::ALIGN_CORNERS:
                align_corners = true;
                use_half_pixel = true;
                break;
            case Base::CoordinateTransformMode::ASYMMETRIC:
                align_corners = false;
                use_half_pixel = false;
                break;
            default:
                OPENVINO_THROW("Interpolate v11: coordinate_transformation_mode not supported");
        }
        switch (attrs.nearest_mode) {
            case Base::NearestMode::FLOOR:
                nearest_mode = 1;
                break;
            case Base::NearestMode::CEIL:
                nearest_mode = 2;
                break;
            case Base::NearestMode::ROUND_PREFER_FLOOR:
                nearest_mode = 1;
                break;
            case Base::NearestMode::ROUND_PREFER_CEIL:
                nearest_mode = 2;
                break;
            case Base::NearestMode::SIMPLE:
            default:
                nearest_mode = 0;
                break;
        }
        check_pads_zero(attrs.pads_begin);
        check_pads_zero(attrs.pads_end);
        std::vector<int64_t> axes;
        if (node->get_input_size() == 3) {
            axes = read_axes_const(node, 2);
        }
        OPENVINO_ASSERT(axes_hw_only(axes), "Interpolate v11: only axes {2,3} supported");
    } else {
        OPENVINO_THROW("MetalInterpolateOp: unsupported interpolate version");
    }

    m_desc.element_type = m_element_type == ov::element::dynamic ? ov::element::f32 : m_element_type;
    m_desc.N = static_cast<uint32_t>(in_shape[0]);
    m_desc.C = static_cast<uint32_t>(in_shape[1]);
    m_desc.H_in = static_cast<uint32_t>(in_shape[2]);
    m_desc.W_in = static_cast<uint32_t>(in_shape[3]);
    m_desc.H_out = static_cast<uint32_t>(out_shape[2]);
    m_desc.W_out = static_cast<uint32_t>(out_shape[3]);
    m_desc.scale_h = static_cast<float>(in_shape[2]) / static_cast<float>(out_shape[2]);
    m_desc.scale_w = static_cast<float>(in_shape[3]) / static_cast<float>(out_shape[3]);
    m_desc.align_corners = align_corners;
    m_desc.nearest = nearest;
    m_desc.use_half_pixel = use_half_pixel;
    m_desc.nearest_mode = nearest_mode;
}

void MetalInterpolateOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalInterpolateOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }
    MetalCodegenBackend backend(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    mlir::MLIRContext ctx;
    auto module = build_mlir_interpolate_from_model(make_single_op_model(m_node), ctx);
    InterpolateCodegenDesc desc = m_desc;
    auto msl_desc = desc;
    auto msl_generator = [msl_desc](mlir::ModuleOp mod) { return generate_msl_from_mlir(mod, msl_desc); };

    KernelSpec spec(m_node, 3u);
    m_kernel = compile_msl_kernel(backend, spec, module, "interpolate_kernel", msl_generator, &log);
    OPENVINO_ASSERT(m_kernel, "MetalInterpolateOp: failed to compile kernel: ", log);

    MetalOp::compile(buffer_manager);
}

void MetalInterpolateOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(!inputs().empty(), "Interpolate: missing input");
    MetalTensor* src = inputs()[0];
    OPENVINO_ASSERT(src && src->buf.valid(), "Interpolate: input buffer null");
    MetalTensor& dst = require_output();

    ov::Shape in_shape = !src->shape.empty() ? src->shape : ov::Shape{};
    if (in_shape.empty() && m_node->get_input_partial_shape(0).is_static()) {
        in_shape = m_node->get_input_shape(0);
    }
    OPENVINO_ASSERT(in_shape.size() == 4, "Interpolate: input shape rank mismatch");
    ov::Shape out_shape = !output_shape().empty() ? output_shape() : ov::Shape{};
    if (out_shape.empty() && m_node->get_output_partial_shape(0).is_static()) {
        out_shape = m_node->get_output_shape(0);
    }
    OPENVINO_ASSERT(out_shape.size() == 4, "Interpolate: output shape rank mismatch");
    const size_t in_bytes = m_element_type.size() * ov::shape_size(in_shape);
    OPENVINO_ASSERT(src->buf.size >= in_bytes, "Interpolate: input buffer too small");

    const size_t bytes = ov::shape_size(out_shape) * m_element_type.size();
    if (!dst.buf.valid() || dst.buf.size < bytes) {
        dst.buf = allocate_temp_buffer(bytes, m_element_type, /*persistent=*/false, dst.prefer_private);
    }
    dst.shape = out_shape;
    dst.expected_type = m_element_type;

    struct InterpolateParams {
        uint32_t N;
        uint32_t C;
        uint32_t H_in;
        uint32_t W_in;
        uint32_t H_out;
        uint32_t W_out;
        float scale_h;
        float scale_w;
        uint32_t align_corners;
        uint32_t use_half_pixel;
        uint32_t nearest_mode;
    } params{};
    params.N = static_cast<uint32_t>(in_shape[0]);
    params.C = static_cast<uint32_t>(in_shape[1]);
    params.H_in = static_cast<uint32_t>(in_shape[2]);
    params.W_in = static_cast<uint32_t>(in_shape[3]);
    params.H_out = static_cast<uint32_t>(out_shape[2]);
    params.W_out = static_cast<uint32_t>(out_shape[3]);
    params.scale_h = static_cast<float>(in_shape[2]) / static_cast<float>(out_shape[2]);
    params.scale_w = static_cast<float>(in_shape[3]) / static_cast<float>(out_shape[3]);
    params.align_corners = m_desc.align_corners ? 1u : 0u;
    params.use_half_pixel = m_desc.use_half_pixel ? 1u : 0u;
    params.nearest_mode = m_desc.nearest_mode;

    const uint64_t total = static_cast<uint64_t>(params.N) * params.C * params.H_out * params.W_out;
    if (total == 0) {
        return;
    }
    KernelDispatch dispatch = make_1d_dispatch(static_cast<size_t>(total), m_kernel->clamp_threadgroup_size(256));

    KernelArgsBuilder args_builder(name().c_str());
    append_kernel_input_args(args_builder, 1, [&](size_t) { return src; }, name().c_str());
    args_builder.add_output(&dst);
    args_builder.add_bytes(&params, sizeof(params));

    const auto args = args_builder.finalize(buffer_manager(), m_kernel.get());
    execute_kernel(*m_kernel, cmd_buf_handle, dispatch, args);
}

}  // namespace gfx_plugin
}  // namespace ov