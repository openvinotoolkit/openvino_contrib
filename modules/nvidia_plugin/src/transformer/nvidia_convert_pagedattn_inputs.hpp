// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type/element_type.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov::nvidia_gpu::pass {

/**
 * @brief Specializes the paged KV-cache Parameters of PagedAttentionExtension.
 *
 * SDPAToPagedAttention produces `key_cache.N` / `value_cache.N` model Parameters with
 * dynamic rank-4 shape and dynamic element type. The plugin cannot allocate those, and
 * GenAI's cache manager needs static [1..3] dims to size cache blocks. This pass sets
 * each cache Parameter to `[-1, num_heads, block_size, head_size]` (dims from the PA
 * op's rt_info geometry) with the given cache precision (`dynamic`: the precision of
 * the PagedAttention query, i.e. the model's own precision).
 *
 * Mirrors ov::pass::ConvertPagedAttnInputs; block_size defaults to 32 (the value
 * GenAI uses for non-"GPU" devices, which the "NVIDIA" plugin is).
 */
class ConvertPagedAttnInputs : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("nvidia_gpu::ConvertPagedAttnInputs", "0");
    explicit ConvertPagedAttnInputs(ov::element::Type cache_precision, std::size_t block_size = 32);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    ov::element::Type cache_precision_;
    std::size_t block_size_;
};

}  // namespace ov::nvidia_gpu::pass
