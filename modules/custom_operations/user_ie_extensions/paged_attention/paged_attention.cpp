// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attention.hpp"

#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/runtime/core.hpp"

#include "cpu_ops.hpp"

std::shared_ptr<ov::Model> TemplateExtension::PagedAttention::make_prefill_subgraph() {
    ov::element::Type_t type = ov::element::f32, attention_mask_type = ov::element::boolean;
    auto query = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape({-1, -1, m_num_heads, m_head_size}));
    auto key = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape({-1, -1, m_num_kv_heads, m_head_size}));
    auto value = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape({-1, -1, m_num_kv_heads, m_head_size}));
    auto attention_mask = std::make_shared<ov::op::v0::Parameter>(attention_mask_type, ov::PartialShape({-1, -1, -1}));
    auto scale = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape({1}));

    // transpose Q, K and V to swap num_heads and seq_len dimensions
    auto permute_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape({4}), {0, 2, 1, 3});
    auto query_transposed = std::make_shared<ov::op::v1::Transpose>(query, permute_const);
    auto key_transposed = std::make_shared<ov::op::v1::Transpose>(key, permute_const);
    auto value_transposed = std::make_shared<ov::op::v1::Transpose>(value, permute_const);

    auto spda = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query_transposed, key_transposed, value_transposed, attention_mask, scale, false);

    // transpose SPDA output to [batch, seq_len, num_heads, head_size] back
    auto spda_transposed = std::make_shared<ov::op::v1::Transpose>(spda, permute_const);

    return std::make_shared<ov::Model>(spda_transposed, ov::ParameterVector{query, key, value, attention_mask, scale}, "spda_prefill_model");
}

TemplateExtension::PagedAttention::PagedAttention(const ov::OutputVector& inputs,
                                                  const float scale)
    : ov::op::Op(inputs),
      m_scale(scale) {
    constructor_validate_and_infer_types();

    // compile model for prefill stage
    auto compiled_model = ov::Core().compile_model(make_prefill_subgraph(), "CPU");
    m_prefill_request = compiled_model.create_infer_request();
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
        get_input_element_type(9) == ov::element::i32 &&
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

// generate buttom diagonal boolean attention mask for a prefill stage
ov::Tensor generate_attention_mask(const std::int32_t num_seqs, const std::int32_t max_context_len, ov::Tensor context_lens) {
    OPENVINO_ASSERT(num_seqs == context_lens.get_size());

    ov::Shape attention_mask_shape({num_seqs, max_context_len, max_context_len});
    ov::Tensor attention_mask(ov::element::boolean, attention_mask_shape);
    int attention_mask_stride = attention_mask.get_strides()[0];

    std::fill_n(attention_mask.data<bool>(), attention_mask.get_size(), false);

    for (int current_seq = 0; current_seq < num_seqs; ++current_seq) {
        std::int32_t context_len = context_lens.data<std::int32_t>()[current_seq];
        OPENVINO_ASSERT(context_len <= max_context_len);

        bool * attention_mask_data = attention_mask.data<bool>() + current_seq * attention_mask_stride;
        for (int x = 0; x < context_len; ++x) {
            for (int y = 0; y < context_len; ++y) {
                attention_mask_data[x * max_context_len + y] = x >= y;
            }
        }
    }
}

// similar to torch.Tensor.view
ov::Tensor view_as_3d(ov::Tensor tensor) {
    ov::Shape shape = tensor.get_shape();
    OPENVINO_ASSERT(shape.size() == 4);
    const std::uint32_t batch_size = shape[0], seq_len = shape[1], num_heads = shape[2], head_size = shape[3];
    return ov::Tensor(tensor.get_element_type(), ov::Shape({batch_size, seq_len, num_heads * head_size}), tensor.data());
}

ov::Tensor view_as_4d(ov::Tensor tensor, std::uint32_t num_heads, std::uint32_t head_size) {
    ov::Shape shape = tensor.get_shape();
    const std::uint32_t batch_size = shape[0], seq_len = shape[1];
    OPENVINO_ASSERT(shape.size() == 3 && num_heads * head_size == shape[3]);
    return ov::Tensor(tensor.get_element_type(), ov::Shape({batch_size, seq_len, num_heads, head_size}), tensor.data());
}

bool TemplateExtension::PagedAttention::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    ov::Tensor query = inputs[0], key = inputs[1], value = inputs[2];
    ov::Shape query_shape = query.get_shape();
    const std::int32_t batch_size = query_shape[0], seq_len = query_shape[1], hidden_size = query_shape[2];
    ov::Tensor key_cache = inputs[3], value_cache = inputs[4];
    const bool is_prompt = inputs[5].data<bool>()[0];
    ov::Tensor slot_mapping = inputs[6];
    const std::int32_t max_context_len = inputs[7].data<std::int32_t>()[0];
    ov::Tensor context_lens = inputs[8];
    ov::Tensor block_tables = inputs[9];

    // reshape to [batch_size, seq_len, num_heads/m_num_kv_heads, head_size] from [batch_size, seq_len, num_heads/m_num_kv_heads * head_size]
    query = view_as_4d(query, m_num_heads, m_head_size);
    key = view_as_4d(key, m_num_kv_heads, m_head_size);
    value = view_as_4d(value, m_num_kv_heads, m_head_size);

    // put current K, V values into key_cache and value_cache
    reshape_and_cache(key, value, key_cache, value_cache, slot_mapping);

    // set output shape
    OPENVINO_ASSERT(outputs.size() == 1);
    outputs[0].set_shape(query.get_shape());

    if (is_prompt) {
        auto attention_mask = generate_attention_mask(batch_size, max_context_len, context_lens);
        ov::Tensor scale(ov::element::f32, ov::Shape{1}, (void *)&m_scale);

        m_prefill_request.set_input_tensor(0, query);
        m_prefill_request.set_input_tensor(1, key);
        m_prefill_request.set_input_tensor(2, value);
        m_prefill_request.set_input_tensor(3, attention_mask);
        m_prefill_request.set_input_tensor(4, scale);
        m_prefill_request.set_output_tensor(outputs[0]);

        m_prefill_request.infer();
    } else {
        paged_attention_v1_cpu(outputs[0],
            query, key_cache, value_cache,
            m_num_kv_heads, m_scale,
            block_tables, context_lens,
            m_block_size, max_context_len);
    }

    // reshape
    outputs[0] = view_as_3d(outputs[0]);

    return true;
}
