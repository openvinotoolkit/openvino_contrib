// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/log_softmax.hpp"
#include "openvino/op/softmax.hpp"
#include "mlir/mlir_support.hpp"
#include "runtime/gfx_shape_utils.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

bool vulkan_softmax_supported(const std::shared_ptr<const ov::Node>& node) {
    if (!node) {
        return false;
    }
    if (!(ov::is_type<const ov::op::v1::Softmax>(node) ||
          ov::is_type<const ov::op::v8::Softmax>(node) ||
          ov::is_type<const ov::op::v5::LogSoftmax>(node))) {
        return false;
    }
    const auto pshape = node->get_input_partial_shape(0);
    if (!pshape.rank().is_static()) {
        return false;
    }
    int64_t axis = 0;
    if (auto s1 = ov::as_type_ptr<const ov::op::v1::Softmax>(node)) {
        axis = s1->get_axis();
    } else if (auto s8 = ov::as_type_ptr<const ov::op::v8::Softmax>(node)) {
        axis = s8->get_axis();
    } else if (auto ls = ov::as_type_ptr<const ov::op::v5::LogSoftmax>(node)) {
        axis = ls->get_axis();
    }
    const int64_t rank = pshape.rank().get_length();
    try {
        axis = normalize_axis(axis, static_cast<size_t>(rank), "GFX Vulkan");
    } catch (const std::exception&) {
        return false;
    }
    const auto et = node->get_output_element_type(0);
    return et == ov::element::f16 || et == ov::element::f32;
}

}  // namespace

bool vulkan_supports_node(const std::shared_ptr<const ov::Node>& node) {
    if (vulkan_softmax_supported(node)) {
        return true;
    }
    return mlir_supports_node(node);
}

}  // namespace gfx_plugin
}  // namespace ov
