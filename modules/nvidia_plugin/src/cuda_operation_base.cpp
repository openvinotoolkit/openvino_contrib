// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <queue>
#include <unordered_set>
#include <utility>

#include "cuda_operation_base.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace nvidia_gpu {

OperationBase::OperationBase(const CreationContext& /*context*/,
                             const ov::Node& node,
                             IndexCollection&& inputIds,
                             IndexCollection&& outputIds)
    : node_name_{node.get_friendly_name()},
      type_name_{node.get_type_info().name},
      input_ids_{inputIds},
      output_ids_{outputIds} {
    if (node.get_input_size() > 0) {
        runtime_precision_ = node.get_input_element_type(0);
    } else if (node.get_output_size() > 0) {
        runtime_precision_ = node.get_output_element_type(0);
    }

    // Check if this node or any of its ancestors have dynamic shapes.
    // A static node whose data transitively depends on a dynamic node
    // (e.g., ShapeOf → Gather → Concat → Broadcast shape chain) is
    // incompatible with CUDA graphs because the dynamic ancestor is wrapped
    // in DynamicOperation which writes to DynamicBufferContext with new
    // addresses each inference — CUDA graphs read stale static addresses.
    has_dynamic_buffer_ = node.is_dynamic();
    if (!has_dynamic_buffer_) {
        std::unordered_set<const ov::Node*> visited;
        std::queue<const ov::Node*> to_visit;
        for (size_t i = 0; i < node.get_input_size(); ++i) {
            to_visit.push(node.get_input_node_ptr(i));
        }
        while (!to_visit.empty()) {
            const auto* current = to_visit.front();
            to_visit.pop();
            if (!visited.insert(current).second) continue;
            if (current->is_dynamic()) {
                has_dynamic_buffer_ = true;
                break;
            }
            for (size_t i = 0; i < current->get_input_size(); ++i) {
                to_visit.push(current->get_input_node_ptr(i));
            }
        }
    }
}

}  // namespace nvidia_gpu
}  // namespace ov
