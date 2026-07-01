// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attention.hpp"

#include <fmt/format.h>

#include <cuda_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include <openvino/op/paged_attention.hpp>

namespace ov {
namespace nvidia_gpu {

namespace {
// Reads an integer geometry value from the node's rt_info. Returns `fallback` when
// the key is absent so that op construction never fails during compilation; the
// value is only actually needed by the M1 kernels.
std::int64_t rtInfoInt(const ov::Node& node, const std::string& key, std::int64_t fallback) {
    const auto& rt = node.get_rt_info();
    const auto it = rt.find(key);
    if (it == rt.end()) {
        return fallback;
    }
    try {
        return it->second.as<std::int64_t>();
    } catch (...) {
        return fallback;
    }
}
}  // namespace

PagedAttentionOp::PagedAttentionOp(const CreationContext& context,
                                   const ov::Node& node,
                                   IndexCollection&& inputIds,
                                   IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    OPENVINO_ASSERT(node.get_input_size() == kNumInputs,
                    "Node name: ",
                    GetName(),
                    "; PagedAttentionExtension expects ",
                    kNumInputs,
                    " inputs, got ",
                    node.get_input_size());
    OPENVINO_ASSERT(node.get_output_size() >= 1, "Node name: ", GetName());

    // Head geometry is carried in rt_info because the key_cache/value_cache tensors
    // are dynamic at this point (see the SDPAToPagedAttention pass).
    num_k_heads_ = rtInfoInt(node, "num_k_heads", 0);
    k_head_size_ = rtInfoInt(node, "k_head_size", 0);
    num_v_heads_ = rtInfoInt(node, "num_v_heads", 0);
    v_head_size_ = rtInfoInt(node, "v_head_size", 0);
}

void PagedAttentionOp::Execute(const InferenceRequestContext&,
                               Inputs,
                               Outputs,
                               const Workbuffers&) const {
    // M0 stub: the op is registered so GenAI-transformed models compile, but the
    // paged-attention kernels are implemented in M1. Fail loudly if execution is
    // reached so this is never silently a no-op.
    OPENVINO_THROW(
        "PagedAttentionOp::Execute is not implemented yet (M0 plumbing stub). "
        "The reshape-and-cache and paged-attention kernels land in M1; "
        "see docs/design-rationales/PagedAttentionExtension.md. Node: ",
        GetName());
}

CudaGraphCompatibility PagedAttentionOp::GetCudaGraphCompatibilityImpl() const {
    // PagedAttention consumes dynamically-shaped scheduling tensors that change every
    // step, so it cannot be captured into a CUDA graph.
    return CudaGraphCompatibility::NONE;
}

OPERATION_REGISTER(PagedAttentionOp, PagedAttentionExtension);

}  // namespace nvidia_gpu
}  // namespace ov
