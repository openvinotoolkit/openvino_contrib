// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nvidia_convert_pagedattn_inputs.hpp"

#include "openvino/core/model.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"

namespace ov::nvidia_gpu::pass {

ConvertPagedAttnInputs::ConvertPagedAttnInputs(ov::element::Type cache_precision, std::size_t block_size)
    : cache_precision_(cache_precision), block_size_(block_size) {}

bool ConvertPagedAttnInputs::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    const auto block_size = static_cast<ov::Dimension::value_type>(block_size_);

    for (const auto& node : model->get_ordered_ops()) {
        const auto pa = ov::as_type_ptr<ov::op::PagedAttentionExtension>(node);
        if (!pa) {
            continue;
        }
        const auto& rt = pa->get_rt_info();
        if (!rt.count("num_k_heads") || !rt.count("k_head_size") || !rt.count("num_v_heads") ||
            !rt.count("v_head_size")) {
            // Geometry unavailable — leave the cache Parameters as-is; the op will
            // fail loudly at execute time if it is actually reached.
            continue;
        }
        const auto num_k_heads = static_cast<ov::Dimension::value_type>(rt.at("num_k_heads").as<std::size_t>());
        const auto k_head_size = static_cast<ov::Dimension::value_type>(rt.at("k_head_size").as<std::size_t>());
        const auto num_v_heads = static_cast<ov::Dimension::value_type>(rt.at("num_v_heads").as<std::size_t>());
        const auto v_head_size = static_cast<ov::Dimension::value_type>(rt.at("v_head_size").as<std::size_t>());

        // input 3 = key_cache.N, input 4 = value_cache.N (both v0::Parameter).
        const auto specialize = [&](std::size_t input_idx, ov::Dimension::value_type heads,
                                    ov::Dimension::value_type head_size) {
            const auto param = ov::as_type_ptr<ov::op::v0::Parameter>(pa->get_input_node_shared_ptr(input_idx));
            if (!param) {
                return;
            }
            param->set_partial_shape(ov::PartialShape{ov::Dimension::dynamic(), heads, block_size, head_size});
            param->set_element_type(cache_precision_);
            changed = true;
        };
        specialize(3, num_k_heads, k_head_size);
        specialize(4, num_v_heads, v_head_size);
    }

    if (changed) {
        model->validate_nodes_and_infer_types();
    }
    return changed;
}

}  // namespace ov::nvidia_gpu::pass
