// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/frontend/pytorch/extension/op.hpp"

namespace TemplateExtension {

// For reference InputMetadata:
// - is_prompt: bool,
// - slot_mapping: torch.Tensor,
// - prompt_lens: Optional[torch.Tensor],
// - max_seq_len: Optional[int],
// - start_loc: Optional[torch.Tensor],
// - max_context_len: Optional[int],
// - context_lens: Optional[torch.Tensor],
// - block_tables: Optional[torch.Tensor],
// - use_cuda_graph: bool,

class PagedAttention : public ov::op::Op {
public:
    OPENVINO_OP("PagedAttention");
    OPENVINO_FRAMEWORK_MAP(pytorch, "vllm.model_executor.layers.attention.PagedAttention");

    PagedAttention() = default;

    PagedAttention(const ov::OutputVector& inputs,
                   const float scale);

    PagedAttention(const ov::Output<ov::Node>& query,
                   const ov::Output<ov::Node>& key,
                   const ov::Output<ov::Node>& value,
                   const ov::Output<ov::Node>& key_cache,
                   const ov::Output<ov::Node>& value_cache,
                   // start of arguments from InputMetadata
                   const ov::Output<ov::Node>& is_prompt,
                   const ov::Output<ov::Node>& slot_mapping,
                //    const ov::Output<ov::Node>& prompt_lens,
                //    const ov::Output<ov::Node>& max_seq_len,
                //    const ov::Output<ov::Node>& start_loc,
                   const ov::Output<ov::Node>& max_context_len,
                   const ov::Output<ov::Node>& context_lens,
                   const ov::Output<ov::Node>& block_tables,
                //    const ov::Output<ov::Node>& use_cuda_graph,
                //    const ov::Output<ov::Node>& attn_bias
                   // end of arguments from InputMetadata
                   const float scale);

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;

private:
    std::shared_ptr<ov::Model> make_prefill_subgraph();

    std::uint32_t m_num_heads, m_num_kv_heads, m_head_size, m_block_size;
    float m_scale;
    mutable ov::InferRequest m_prefill_request;
};

}  // namespace TemplateExtension
