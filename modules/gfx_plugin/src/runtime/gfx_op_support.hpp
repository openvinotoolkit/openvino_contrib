// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/model.hpp"
#include "runtime/gpu_backend.hpp"

namespace ov {
namespace gfx_plugin {

struct UnsupportedSummary {
    std::vector<std::string> node_names;
    std::vector<std::pair<std::string, size_t>> type_counts;
};

bool is_supported_node(const std::shared_ptr<const ov::Node>& node, GpuBackend backend);
bool model_supported_by_backend(const std::shared_ptr<const ov::Model>& model, GpuBackend backend);
UnsupportedSummary collect_unsupported(const std::shared_ptr<const ov::Model>& model,
                                       GpuBackend backend,
                                       size_t max_nodes = 8);

}  // namespace gfx_plugin
}  // namespace ov
