// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/operation_legalizer.hpp"

#include <unordered_map>

#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

bool LegalityReport::semantic_legal() const {
    for (const auto& record : records) {
        if (!record.support.semantic_legal) {
            return false;
        }
    }
    return true;
}

UnsupportedSummary LegalityReport::unsupported_summary(size_t max_nodes) const {
    UnsupportedSummary summary;
    std::unordered_map<std::string, size_t> counts;
    for (const auto& record : records) {
        if (record.support.semantic_legal || !record.node) {
            continue;
        }
        const std::string type = record.node->get_type_name();
        counts[type] += 1;
        if (summary.node_names.size() < max_nodes) {
            summary.node_names.emplace_back(record.node->get_friendly_name() + " (" + type + ")");
        }
    }
    summary.type_counts.reserve(counts.size());
    for (const auto& kv : counts) {
        summary.type_counts.emplace_back(kv.first, kv.second);
    }
    return summary;
}

OperationLegalizer::OperationLegalizer(const BackendCapabilities& capabilities)
    : m_capabilities(&capabilities) {}

OperationSupportResult OperationLegalizer::query(const std::shared_ptr<const ov::Node>& node) const {
    OPENVINO_ASSERT(m_capabilities, "GFX: operation legalizer has no backend capabilities");
    return m_capabilities->query_operation({node});
}

LegalityReport OperationLegalizer::legalize_model(const std::shared_ptr<const ov::Model>& model) const {
    OPENVINO_ASSERT(model, "GFX: model is null");

    LegalityReport report;
    const auto ordered_ops = model->get_ordered_ops();
    report.records.reserve(ordered_ops.size());
    for (const auto& node : ordered_ops) {
        report.records.push_back({node, query(node)});
    }
    return report;
}

}  // namespace compiler
}  // namespace gfx_plugin
}  // namespace ov
