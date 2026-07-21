// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "compiler/operation_support.hpp"
#include "openvino/core/model.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

struct OperationLegalityRecord {
    std::shared_ptr<const ov::Node> node;
    OperationSupportResult support;
};

struct LegalityReport {
    std::vector<OperationLegalityRecord> records;

    bool semantic_legal() const;
    UnsupportedSummary unsupported_summary(size_t max_nodes = 8) const;
};

class OperationLegalizer final {
public:
    explicit OperationLegalizer(const BackendCapabilities& capabilities);

    OperationSupportResult query(const std::shared_ptr<const ov::Node>& node) const;
    LegalityReport legalize_model(const std::shared_ptr<const ov::Model>& model) const;

private:
    const BackendCapabilities* m_capabilities = nullptr;
};

}  // namespace compiler
}  // namespace gfx_plugin
}  // namespace ov
