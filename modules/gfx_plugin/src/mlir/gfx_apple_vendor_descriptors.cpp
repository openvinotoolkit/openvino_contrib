// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/gfx_apple_vendor_descriptors.hpp"

#include <algorithm>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

bool all_zero(const std::vector<size_t>& values) {
    return std::all_of(values.begin(), values.end(), [](size_t value) {
        return value == 0;
    });
}

int64_t normalize_axis(int64_t axis, size_t rank) {
    return axis < 0 ? axis + static_cast<int64_t>(rank) : axis;
}

bool copy_2d_spatial_attrs(const ov::Strides& strides,
                           const ov::Strides& dilations,
                           const ov::CoordinateDiff& pads_begin,
                           const ov::CoordinateDiff& pads_end,
                           uint32_t out_strides[2],
                           uint32_t out_dilations[2],
                           uint32_t out_pads[4]) {
    if (strides.size() != 2 || dilations.size() != 2 ||
        pads_begin.size() != 2 || pads_end.size() != 2) {
        return false;
    }
    if (pads_begin[0] < 0 || pads_begin[1] < 0 ||
        pads_end[0] < 0 || pads_end[1] < 0) {
        return false;
    }
    out_strides[0] = static_cast<uint32_t>(strides[0]);
    out_strides[1] = static_cast<uint32_t>(strides[1]);
    out_dilations[0] = static_cast<uint32_t>(dilations[0]);
    out_dilations[1] = static_cast<uint32_t>(dilations[1]);
    out_pads[0] = static_cast<uint32_t>(pads_begin[0]);
    out_pads[1] = static_cast<uint32_t>(pads_begin[1]);
    out_pads[2] = static_cast<uint32_t>(pads_end[0]);
    out_pads[3] = static_cast<uint32_t>(pads_end[1]);
    return true;
}

bool copy_2d_spatial_attrs(const ov::Strides& strides,
                           const ov::Strides& dilations,
                           const ov::Shape& pads_begin,
                           const ov::Shape& pads_end,
                           uint32_t out_strides[2],
                           uint32_t out_dilations[2],
                           uint32_t out_pads[4]) {
    if (strides.size() != 2 || dilations.size() != 2 ||
        pads_begin.size() != 2 || pads_end.size() != 2) {
        return false;
    }
    out_strides[0] = static_cast<uint32_t>(strides[0]);
    out_strides[1] = static_cast<uint32_t>(strides[1]);
    out_dilations[0] = static_cast<uint32_t>(dilations[0]);
    out_dilations[1] = static_cast<uint32_t>(dilations[1]);
    out_pads[0] = static_cast<uint32_t>(pads_begin[0]);
    out_pads[1] = static_cast<uint32_t>(pads_begin[1]);
    out_pads[2] = static_cast<uint32_t>(pads_end[0]);
    out_pads[3] = static_cast<uint32_t>(pads_end[1]);
    return true;
}

bool is_static_nchw_spatial_resize(const std::shared_ptr<const ov::Node>& node) {
    if (!node ||
        !node->get_input_partial_shape(0).is_static() ||
        !node->get_output_partial_shape(0).is_static()) {
        return false;
    }

    const auto input_shape = node->get_input_shape(0);
    const auto output_shape = node->get_output_shape(0);
    return input_shape.size() == 4 &&
           output_shape.size() == 4 &&
           input_shape[0] == output_shape[0] &&
           input_shape[1] == output_shape[1] &&
           input_shape[2] != 0 &&
           input_shape[3] != 0 &&
           output_shape[2] != 0 &&
           output_shape[3] != 0;
}

bool axes_are_spatial_nchw(std::vector<int64_t> axes) {
    if (axes.size() != 2) {
        return false;
    }
    for (auto& axis : axes) {
        if (axis < 0) {
            axis += 4;
        }
    }
    std::sort(axes.begin(), axes.end());
    return axes == std::vector<int64_t>{2, 3};
}

bool axes_are_spatial_nchw(const ov::AxisSet& axes) {
    std::vector<int64_t> values;
    values.reserve(axes.size());
    for (auto axis : axes) {
        values.push_back(static_cast<int64_t>(axis));
    }
    return axes_are_spatial_nchw(std::move(values));
}

bool constant_axes_input_is_spatial_nchw_or_absent(const ov::Node& node) {
    if (node.get_input_size() < 4) {
        return true;
    }
    const auto axes_node = node.input_value(3).get_node_shared_ptr();
    const auto axes_const = ov::as_type_ptr<const ov::op::v0::Constant>(axes_node);
    if (!axes_const) {
        return false;
    }
    return axes_are_spatial_nchw(axes_const->cast_vector<int64_t>());
}

bool configure_bilinear_half_pixel_resize_desc(GfxMpsrtResize2DAbiDesc& desc) {
    desc = {};
    desc.nearest = 0;
    desc.align_corners = 0;
    desc.half_pixel_centers = 1;
    return true;
}

std::vector<int64_t> gfx_shape_to_i64_vector(const ov::Shape& shape) {
    std::vector<int64_t> dims;
    dims.reserve(shape.size());
    for (const auto dim : shape) {
        dims.push_back(static_cast<int64_t>(dim));
    }
    return dims;
}

bool configure_from_interpolate_base_attrs(
    const ov::op::util::InterpolateBase::InterpolateAttrs& attrs,
    GfxMpsrtResize2DAbiDesc& desc) {
    using Base = ov::op::util::InterpolateBase;
    if (attrs.mode == Base::InterpolateMode::NEAREST ||
        (attrs.mode != Base::InterpolateMode::LINEAR &&
         attrs.mode != Base::InterpolateMode::LINEAR_ONNX &&
         attrs.mode != Base::InterpolateMode::BILINEAR_PILLOW) ||
        attrs.coordinate_transformation_mode != Base::CoordinateTransformMode::HALF_PIXEL ||
        attrs.antialias ||
        !all_zero(attrs.pads_begin) ||
        !all_zero(attrs.pads_end)) {
        return false;
    }
    return configure_bilinear_half_pixel_resize_desc(desc);
}

}  // namespace

uint32_t gfx_apple_mps_conv_fused_activation_code(ActivationKind kind) {
    switch (kind) {
        case ActivationKind::Relu:
            return 1u;
        case ActivationKind::Sigmoid:
            return 2u;
        case ActivationKind::Tanh:
            return 3u;
        case ActivationKind::Abs:
            return 10u;
        case ActivationKind::Identity:
            return 0u;
        default:
            return 0u;
    }
}

bool gfx_apple_mps_conv_supports_fused_activation(ActivationKind kind) {
    return kind == ActivationKind::Identity ||
           gfx_apple_mps_conv_fused_activation_code(kind) != 0u;
}

std::string gfx_apple_mps_canonical_conv_stage_type(const std::shared_ptr<const ov::Node>& node,
                                                    std::string_view fallback_stage_type) {
    if (ov::as_type_ptr<const ov::op::v1::Convolution>(node)) {
        return "Convolution";
    }
    if (ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) {
        return "GroupConvolution";
    }
    if (fallback_stage_type == "GroupConv2D") {
        return "GroupConvolution";
    }
    return std::string(fallback_stage_type);
}

bool gfx_apple_make_mps_conv2d_desc(const std::shared_ptr<const ov::Node>& node,
                                    GfxMpsrtConv2DAbiDesc& desc,
                                    bool has_activation,
                                    ActivationKind activation) {
    desc = {};
    if (!node || node->get_input_size() < 2 ||
        !node->get_input_partial_shape(0).is_static() ||
        !node->get_input_partial_shape(1).is_static() ||
        (has_activation && !gfx_apple_mps_conv_supports_fused_activation(activation))) {
        return false;
    }

    bool ok = false;
    if (auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(node)) {
        const auto input_shape = conv->get_input_shape(0);
        const auto weights_shape = conv->get_input_shape(1);
        if (input_shape.size() == 4 && weights_shape.size() == 4 &&
            weights_shape[1] != 0 && input_shape[1] % weights_shape[1] == 0) {
            desc.groups = static_cast<uint32_t>(input_shape[1] / weights_shape[1]);
            ok = copy_2d_spatial_attrs(conv->get_strides(),
                                       conv->get_dilations(),
                                       conv->get_pads_begin(),
                                       conv->get_pads_end(),
                                       desc.strides,
                                       desc.dilations,
                                       desc.pads);
        }
    } else if (auto group_conv = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) {
        const auto input_shape = group_conv->get_input_shape(0);
        const auto weights_shape = group_conv->get_input_shape(1);
        if (input_shape.size() == 4 && weights_shape.size() == 5 && weights_shape[0] != 0) {
            desc.groups = static_cast<uint32_t>(weights_shape[0]);
            ok = copy_2d_spatial_attrs(group_conv->get_strides(),
                                       group_conv->get_dilations(),
                                       group_conv->get_pads_begin(),
                                       group_conv->get_pads_end(),
                                       desc.strides,
                                       desc.dilations,
                                       desc.pads);
        }
    }

    if (!ok) {
        desc = {};
        return false;
    }
    if (has_activation) {
        desc.fused_activation = gfx_apple_mps_conv_fused_activation_code(activation);
    }
    return true;
}

bool gfx_apple_make_mps_pool2d_desc(const std::shared_ptr<const ov::Node>& node,
                                    GfxMpsrtPool2DAbiDesc& desc) {
    desc = {};
    if (!node ||
        !node->get_input_partial_shape(0).is_static() ||
        !node->get_output_partial_shape(0).is_static()) {
        return false;
    }
    const auto input_shape = node->get_input_shape(0);
    const auto output_shape = node->get_output_shape(0);
    if (input_shape.size() != 4 || output_shape.size() != 4) {
        return false;
    }

    if (auto maxpool = std::dynamic_pointer_cast<const ov::op::util::MaxPoolBase>(node)) {
        ov::Strides dilations(maxpool->get_kernel().size(), 1);
        if (auto p = std::dynamic_pointer_cast<const ov::op::v8::MaxPool>(node)) {
            dilations = p->get_dilations();
        } else if (auto p = std::dynamic_pointer_cast<const ov::op::v14::MaxPool>(node)) {
            dilations = p->get_dilations();
        }
        if (maxpool->get_kernel().size() != 2 ||
            !copy_2d_spatial_attrs(maxpool->get_strides(),
                                   dilations,
                                   maxpool->get_pads_begin(),
                                   maxpool->get_pads_end(),
                                   desc.strides,
                                   desc.dilations,
                                   desc.pads)) {
            return false;
        }
        desc.is_avg = 0;
        desc.kernel[0] = static_cast<uint32_t>(maxpool->get_kernel()[0]);
        desc.kernel[1] = static_cast<uint32_t>(maxpool->get_kernel()[1]);
        desc.exclude_pad = 1;
        return true;
    }

    if (auto avgpool = std::dynamic_pointer_cast<const ov::op::util::AvgPoolBase>(node)) {
        ov::Strides dilations(avgpool->get_kernel().size(), 1);
        if (auto p = std::dynamic_pointer_cast<const ov::op::v16::AvgPool>(node)) {
            dilations = p->get_dilations();
        }
        if (avgpool->get_kernel().size() != 2 ||
            !copy_2d_spatial_attrs(avgpool->get_strides(),
                                   dilations,
                                   avgpool->get_pads_begin(),
                                   avgpool->get_pads_end(),
                                   desc.strides,
                                   desc.dilations,
                                   desc.pads)) {
            return false;
        }
        desc.is_avg = 1;
        desc.kernel[0] = static_cast<uint32_t>(avgpool->get_kernel()[0]);
        desc.kernel[1] = static_cast<uint32_t>(avgpool->get_kernel()[1]);
        desc.exclude_pad = avgpool->get_exclude_pad() ? 1u : 0u;
        return true;
    }

    return false;
}

bool gfx_apple_make_mps_resize2d_desc(const std::shared_ptr<const ov::Node>& node,
                                      GfxMpsrtResize2DAbiDesc& desc) {
    if (!is_static_nchw_spatial_resize(node)) {
        return false;
    }

    if (auto interp = std::dynamic_pointer_cast<const ov::op::v0::Interpolate>(node)) {
        const auto mode = ov::util::to_lower(interp->get_attrs().mode);
        if (mode != "linear" ||
            interp->get_attrs().align_corners ||
            interp->get_attrs().antialias ||
            !all_zero(interp->get_attrs().pads_begin) ||
            !all_zero(interp->get_attrs().pads_end) ||
            !axes_are_spatial_nchw(interp->get_attrs().axes)) {
            return false;
        }
        return configure_bilinear_half_pixel_resize_desc(desc);
    }

    if (auto interp = std::dynamic_pointer_cast<const ov::op::v4::Interpolate>(node)) {
        if (!constant_axes_input_is_spatial_nchw_or_absent(*interp)) {
            return false;
        }
        return configure_from_interpolate_base_attrs(interp->get_attrs(), desc);
    }

    if (auto interp = std::dynamic_pointer_cast<const ov::op::v11::Interpolate>(node)) {
        if (!constant_axes_input_is_spatial_nchw_or_absent(*interp)) {
            return false;
        }
        return configure_from_interpolate_base_attrs(interp->get_attrs(), desc);
    }

    return false;
}

bool gfx_apple_make_mps_softmax_desc(const std::shared_ptr<const ov::Node>& node,
                                     GfxMpsrtSoftmaxAbiDesc& desc) {
    desc = {};
    if (!node ||
        !node->get_input_partial_shape(0).is_static() ||
        !node->get_output_partial_shape(0).is_static()) {
        return false;
    }
    const auto input_shape = node->get_input_shape(0);
    if (input_shape.empty()) {
        return false;
    }

    int64_t axis = -1;
    if (auto sm1 = ov::as_type_ptr<const ov::op::v1::Softmax>(node)) {
        axis = sm1->get_axis();
    } else if (auto sm8 = ov::as_type_ptr<const ov::op::v8::Softmax>(node)) {
        axis = sm8->get_axis();
    } else if (ov::as_type_ptr<const ov::op::v5::LogSoftmax>(node)) {
        return false;
    } else {
        return false;
    }

    axis = normalize_axis(axis, input_shape.size());
    if (axis < 0 || axis != static_cast<int64_t>(input_shape.size() - 1)) {
        return false;
    }
    desc.axis = static_cast<uint32_t>(axis);
    desc.log_softmax = 0;
    return true;
}

bool gfx_apple_make_mps_topk_desc(const std::shared_ptr<const ov::Node>& node,
                                  GfxMpsrtTopKAbiDesc& desc) {
    desc = {};
    auto topk = ov::as_type_ptr<const ov::op::v11::TopK>(node);
    if (!topk ||
        !topk->get_input_partial_shape(0).is_static() ||
        !topk->get_output_partial_shape(0).is_static() ||
        !topk->get_output_partial_shape(1).is_static()) {
        return false;
    }

    const auto input_shape = topk->get_input_shape(0);
    if (input_shape.empty()) {
        return false;
    }
    const int64_t axis = normalize_axis(topk->get_axis(), input_shape.size());
    if (axis < 0 || axis != static_cast<int64_t>(input_shape.size() - 1)) {
        return false;
    }
    const auto k = topk->get_k();
    if (k == 0 || k > 16 ||
        topk->get_mode() != ov::op::TopKMode::MAX) {
        return false;
    }
    const auto index_type = topk->get_output_element_type(1);
    if (index_type != ov::element::i32 && index_type != ov::element::u32) {
        return false;
    }

    desc.axis = static_cast<uint32_t>(axis);
    desc.k = static_cast<uint32_t>(k);
    desc.mode_max = 1;
    switch (topk->get_sort_type()) {
        case ov::op::TopKSortType::SORT_INDICES:
            desc.sort_type = 2u;
            break;
        case ov::op::TopKSortType::NONE:
            desc.sort_type = 0u;
            break;
        case ov::op::TopKSortType::SORT_VALUES:
        default:
            desc.sort_type = 1u;
            break;
    }
    return true;
}

bool gfx_apple_make_mps_io_tensor_descs_for_node(const std::shared_ptr<const ov::Node>& node,
                                                 GfxStageStorageKind storage,
                                                 std::vector<GfxMpsrtTensorDesc>& inputs,
                                                 std::vector<GfxMpsrtTensorDesc>& outputs) {
    inputs.clear();
    outputs.clear();
    if (!node || node->get_input_size() == 0 || node->get_output_size() == 0 ||
        !node->get_input_partial_shape(0).is_static()) {
        return false;
    }

    inputs.push_back(gfx_mpsrt_make_tensor_desc(gfx_shape_to_i64_vector(node->get_input_shape(0)),
                                                node->get_input_element_type(0),
                                                storage,
                                                GfxMpsrtTensorFlagExternalIo));
    outputs.reserve(node->get_output_size());
    for (size_t output_index = 0; output_index < node->get_output_size(); ++output_index) {
        if (!node->get_output_partial_shape(output_index).is_static()) {
            inputs.clear();
            outputs.clear();
            return false;
        }
        outputs.push_back(gfx_mpsrt_make_tensor_desc(gfx_shape_to_i64_vector(node->get_output_shape(output_index)),
                                                     node->get_output_element_type(output_index),
                                                     storage,
                                                     GfxMpsrtTensorFlagTransient));
    }
    return true;
}

}  // namespace gfx_plugin
}  // namespace ov
