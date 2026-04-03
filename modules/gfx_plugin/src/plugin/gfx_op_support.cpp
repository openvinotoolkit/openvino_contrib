// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/gfx_op_support.hpp"

#include <unordered_map>

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "runtime/gfx_logger.hpp"
#include "runtime/gfx_stage_policy.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

bool is_view_node(const std::shared_ptr<const ov::Node>& node) {
    if (!node) {
        return false;
    }
    return select_tensor_layout_plan(node->get_type_name(), node).view_only;
}

}  // namespace

bool metal_supports_node(const std::shared_ptr<const ov::Node>& node);
bool vulkan_supports_node(const std::shared_ptr<const ov::Node>& node);

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
            const bool supported = metal_supports_node(node);
            if (!supported && gfx_log_debug_enabled()) {
                gfx_log_debug("Plugin") << "Unsupported node: " << node->get_friendly_name()
                                                             << " (" << node->get_type_name() << ")";
            }
            return supported;
        }
        if (backend == GpuBackend::Vulkan) {
            if (gfx_log_debug_enabled()) {
                gfx_log_debug("Plugin") << "Check node: " << node->get_friendly_name()
                                                       << " (" << node->get_type_name() << ")";
            }
            const bool supported = vulkan_supports_node(node);
            if (!supported && gfx_log_debug_enabled()) {
                gfx_log_debug("Plugin") << "Unsupported node: " << node->get_friendly_name()
                                                             << " (" << node->get_type_name() << ")";
            }
            return supported;
        }
        return false;
    } catch (const std::exception& e) {
        if (gfx_log_debug_enabled()) {
            gfx_log_debug("Plugin") << "Exception probing node " << node->get_friendly_name()
                                                              << " (" << node->get_type_name() << "): "
                                                              << e.what();
        }
        return false;
    } catch (...) {
        if (gfx_log_debug_enabled()) {
            gfx_log_debug("Plugin") << "Unknown exception probing node " << node->get_friendly_name()
                                                                      << " (" << node->get_type_name() << ")";
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
