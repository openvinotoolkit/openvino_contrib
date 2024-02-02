// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attention.hpp"

#include "cpu_ops.hpp"

TemplateExtension::PagedAttention::PagedAttention(const ov::OutputVector& inputs,
                                                  const float scale)
    : ov::op::Op(inputs),
      m_scale(scale) {
    constructor_validate_and_infer_types();
}

TemplateExtension::PagedAttention::PagedAttention(const ov::Output<ov::Node>& query,
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
                                                  const float scale)
    : PagedAttention({query, key, value, key_cache, value_cache,
                      is_prompt, slot_mapping, max_context_len, context_lens, block_tables
    }, scale) {}

void TemplateExtension::PagedAttention::validate_and_infer_types() {
    // value_cache: shape = [num_blocks, num_kv_heads, head_size, block_size]
    auto value_cache_shape = get_input_shape(4);
    m_num_kv_heads = value_cache_shape[1];
    m_head_size = value_cache_shape[2];
    m_block_size = value_cache_shape[3];

    // key_cache: shape [num_blocks, num_kv_heads, head_size/x, block_size, x]
    auto key_cache_shape = get_input_shape(3);
    NODE_VALIDATION_CHECK(this,
        value_cache_shape[0] == key_cache_shape[0] && // num_blocks
        key_cache_shape[1] == m_num_kv_heads &&
        key_cache_shape[2] * key_cache_shape[4] == m_head_size &&
        m_block_size == key_cache_shape[3], // block_size,
        "Key cache validation failed");

    // query: shape [batch_size, seq_len, num_heads * head_size]
    auto query_type = get_input_element_type(0);
    auto query_shape = get_input_partial_shape(0);
    m_num_heads = query_shape[2].get_length();
    NODE_VALIDATION_CHECK(this,
        query_type.is_real() &&
        query_shape.size() == 3 &&
        query_shape[2] == m_num_heads * m_head_size,
        "Query type must be real, shape must be like [batch_size, seq_len, num_heads * head_size]");

    // key: shape [batch_size, seq_len, num_kv_heads * head_size]
    auto key_type = get_input_element_type(1);
    auto key_shape = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(this,
        query_type == key_type &&
        key_shape.size() == query_shape.size() &&
        key_shape[2] == m_num_kv_heads * m_head_size,
        "Key type must be the same as query, shape must be the same as query");

    // value: shape [batch_size, seq_len, num_kv_heads * head_size]
    auto value_type = get_input_element_type(2);
    auto value_shape = get_input_partial_shape(2);
    NODE_VALIDATION_CHECK(this,
        key_type == value_type &&
        key_shape == value_shape, "Value type must be the same as key, shape must be the same as key");

    // is_prompt: boolean scalar
    NODE_VALIDATION_CHECK(this,
        get_input_element_type(5) == ov::element::boolean &&
        get_input_shape(5) == ov::Shape({1}),
        "is_prompt validation failed");

    // slot_mapping: shape [batch_size, max_context_len]
    auto slot_mapping_shape = get_input_partial_shape(6);
    NODE_VALIDATION_CHECK(this,
        get_input_element_type(6) == ov::element::i64 &&
        slot_mapping_shape.size() == 2,
        "slot_mapping validation failed");

    // max_context_len: integer scalar
    NODE_VALIDATION_CHECK(this,
        get_input_element_type(7) == ov::element::i32 &&
        get_input_shape(7) == ov::Shape({1}),
        "max_context_len validation failed");

    // context_lens: shape [batch_size]
    auto context_lens_shape = get_input_shape(8);
    NODE_VALIDATION_CHECK(this,
        get_input_element_type(8) == ov::element::i32 &&
        context_lens_shape.size() == 1,
        "context_lens validation failed");

    // block_tables: shape [batch_size, max_block_per_request]
    NODE_VALIDATION_CHECK(this,
        get_input_partial_shape(9).size() == 2,
        "block_tables validation failed");

    set_output_type(0, query_type, query_shape);
}

std::shared_ptr<ov::Node> TemplateExtension::PagedAttention::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<PagedAttention>(new_args, m_scale);
}

bool TemplateExtension::PagedAttention::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("scale", m_scale);
    return true;
}

bool TemplateExtension::PagedAttention::has_evaluate() const {
    return get_input_element_type(0) == ov::element::f32;
}

// puts current K, V values into key_cache and value_cache
void reshape_and_cache(ov::Tensor key, ov::Tensor value,
                       ov::Tensor key_cache, ov::Tensor value_cache,
                       ov::Tensor slot_mapping);

// generate block diagonal attention mask for a prefill stage
ov::Tensor generate_attention_mask(ov::Tensor context_lens);

bool TemplateExtension::PagedAttention::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    ov::Tensor query = inputs[0], key = inputs[1], value = inputs[2];
    ov::Tensor key_cache = inputs[3], value_cache = inputs[4];
    const bool is_prompt = inputs[5].data<bool>()[0];
    ov::Tensor slot_mapping = inputs[6];
    const std::int32_t max_context_len = inputs[7].data<std::int32_t>()[0];
    ov::Tensor context_lens = inputs[8];
    ov::Tensor block_tables = inputs[9];

    // put current K, V values into key_cache and value_cache
    reshape_and_cache(key, value, key_cache, value_cache, slot_mapping);

    if (is_prompt) {
        auto attention_mask = generate_attention_mask(context_lens);

        // create a model with OpenVINO SDPA to compute first token
        // TODO
    } else {
        paged_attention_v1_cpu(outputs[0],
            query, key_cache, value_cache,
            m_num_kv_heads, m_scale,
            block_tables, context_lens,
            m_block_size, max_context_len);
    }

    return true;
}
