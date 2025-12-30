// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gfx_op_support.hpp"

#include <unordered_map>

#include "openvino/op/constant.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "runtime/gfx_logger.hpp"
#include "mlir/mlir_support.hpp"
#include "runtime/execution_dispatcher.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

bool is_view_node(const std::shared_ptr<const ov::Node>& node) {
    if (!node) {
        return false;
    }
    const std::string type = node->get_type_name();
    return type == "Reshape" || type == "Squeeze" || type == "Unsqueeze";
}

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
    if (axis < 0) {
        axis += rank;
    }
    if (axis < 0 || axis >= rank) {
        return false;
    }
    const auto et = node->get_output_element_type(0);
    return et == ov::element::f16 || et == ov::element::f32;
}

}  // namespace

bool is_supported_node(const std::shared_ptr<const ov::Node>& node, GpuBackend backend) {
    if (ov::as_type_ptr<const ov::op::v0::Parameter>(node) ||
        ov::as_type_ptr<const ov::op::v0::Constant>(node) ||
        ov::as_type_ptr<const ov::op::v0::Result>(node)) {
        return true;
    }
    if (is_view_node(node)) {
        return true;
    }

    try {
        if (backend == GpuBackend::Metal) {
            auto probe = GpuStageFactory::create(node, GpuBackend::Metal, /*device*/ nullptr, /*queue*/ nullptr);
            if (!probe && gfx_log_debug_enabled()) {
                GFX_LOG_DEBUG("Plugin", "Unsupported node: " << node->get_friendly_name()
                                                             << " (" << node->get_type_name() << ")");
            }
            return probe != nullptr;
        }
        if (backend == GpuBackend::Vulkan) {
            if (vulkan_softmax_supported(node)) {
                return true;
            }
            const bool supported = mlir_supports_node(node);
            if (!supported && gfx_log_debug_enabled()) {
                GFX_LOG_DEBUG("Plugin", "Unsupported node: " << node->get_friendly_name()
                                                             << " (" << node->get_type_name() << ")");
            }
            return supported;
        }
        return false;
    } catch (const std::exception& e) {
        if (gfx_log_debug_enabled()) {
            GFX_LOG_DEBUG("Plugin", "Exception probing node " << node->get_friendly_name()
                                                              << " (" << node->get_type_name() << "): "
                                                              << e.what());
        }
        return false;
    } catch (...) {
        if (gfx_log_debug_enabled()) {
            GFX_LOG_DEBUG("Plugin", "Unknown exception probing node " << node->get_friendly_name()
                                                                      << " (" << node->get_type_name() << ")");
        }
        return false;
    }
}

bool model_supported_by_backend(const std::shared_ptr<const ov::Model>& model, GpuBackend backend) {
    for (const auto& node : model->get_ordered_ops()) {
        if (!is_supported_node(node, backend)) {
            return false;
        }
    }
    return true;
}

UnsupportedSummary collect_unsupported(const std::shared_ptr<const ov::Model>& model,
                                       GpuBackend backend,
                                       size_t max_nodes) {
    UnsupportedSummary summary;
    std::unordered_map<std::string, size_t> counts;
    for (const auto& node : model->get_ordered_ops()) {
        if (is_supported_node(node, backend)) {
            continue;
        }
        const std::string type = node->get_type_name();
        counts[type] += 1;
        if (summary.node_names.size() < max_nodes) {
            summary.node_names.emplace_back(node->get_friendly_name() + " (" + type + ")");
        }
    }
    summary.type_counts.reserve(counts.size());
    for (const auto& kv : counts) {
        summary.type_counts.emplace_back(kv.first, kv.second);
    }
    return summary;
}

}  // namespace gfx_plugin
}  // namespace ov
