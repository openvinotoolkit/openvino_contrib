// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_operation_base.hpp"

namespace ov {
namespace nvidia_gpu {

/**
 * @brief PagedAttentionExtension — the op OpenVINO GenAI's continuous-batching path
 *        produces (via SDPAToPagedAttention) for paged KV-cache attention.
 *
 * The op has the 25-input contract; head geometry is carried in the node's rt_info
 * (num_k_heads / k_head_size / num_v_heads / v_head_size) because the key_cache /
 * value_cache tensors arrive with dynamic rank-4 shape.
 *
 * M0 (plumbing): the op is registered so a GenAI-transformed model compiles; the
 * attention kernels are added in M1. Execute() is intentionally a stub that fails
 * loudly if reached.
 *
 * See docs/design-rationales/PagedAttentionExtension.md.
 */
class PagedAttentionOp : public OperationBase {
public:
    PagedAttentionOp(const CreationContext& context,
                     const ov::Node& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputs,
                 Outputs outputs,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibilityImpl() const override;

    // Number of inputs of the current PagedAttentionExtension contract.
    static constexpr std::size_t kNumInputs = 25;
    // Inputs [kNumBaseInputs, kNumInputs) are extended features (score aggregation,
    // cache rotation, xattention, attention sinks, adaptive RKV) not implemented in
    // the MVP.
    static constexpr std::size_t kNumBaseInputs = 13;

private:
    // Head geometry (from rt_info). num_k_heads may differ from the number of query
    // heads (GQA); k_head_size may differ from v_head_size.
    std::int64_t num_k_heads_{};
    std::int64_t k_head_size_{};
    std::int64_t num_v_heads_{};
    std::int64_t v_head_size_{};
};

}  // namespace nvidia_gpu
}  // namespace ov
