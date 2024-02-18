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

namespace {

std::shared_ptr<ov::Model> make_prefill_subgraph(std::int64_t num_heads = -1, std::int64_t num_kv_heads = -1, std::int64_t head_size = -1) {
    ov::element::Type_t type = ov::element::f32, attention_mask_type = ov::element::f32;
    auto query = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape({-1 /* batch */, -1 /* seq_len */, -1 /* queries per kv */, num_kv_heads, head_size}));
    auto key = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape({-1 /* batch */, -1 /* seq_len */, num_kv_heads, 1, head_size}));
    auto value = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape({-1 /* batch */, -1 /* seq_len */, num_kv_heads, 1, head_size}));
    auto mask = std::make_shared<ov::op::v0::Parameter>(attention_mask_type, ov::PartialShape({-1, -1, -1, -1, -1}));
    auto scale = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape({1}));

    // transpose Q, K and V to swap num_heads and seq_len dimensions
    auto permute_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape({5}), {0, 3, 2, 1, 4});
    auto query_transposed = std::make_shared<ov::op::v1::Transpose>(query, permute_const);
    auto key_transposed = std::make_shared<ov::op::v1::Transpose>(key, permute_const);
    auto value_transposed = std::make_shared<ov::op::v1::Transpose>(value, permute_const);

    auto spda = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query_transposed, key_transposed, value_transposed, mask, scale, false);

    // transpose SPDA output to [batch, seq_len, num_q_per_kv, num_kv_heads, head_size] back
    auto spda_transposed = std::make_shared<ov::op::v1::Transpose>(spda, permute_const);

    return std::make_shared<ov::Model>(spda_transposed, ov::ParameterVector{query, key, value, mask, scale}, "spda_prefill_model");
}

}

ov::InferRequest TemplateExtension::PagedAttention::m_prefill_request;
std::once_flag TemplateExtension::PagedAttention::m_once;

TemplateExtension::PagedAttention::PagedAttention(const ov::OutputVector& inputs)
    : ov::op::Op(inputs) {
    constructor_validate_and_infer_types();

    // compile model for prefill stage
    std::call_once(m_once, [_this=this] () {
        ov::Core core;
        auto compiled_model = core.compile_model(make_prefill_subgraph(), "CPU");
        _this->m_prefill_request = compiled_model.create_infer_request();
    });
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
                                                  const ov::Output<ov::Node>& scale)
    : PagedAttention(ov::OutputVector{query, key, value, key_cache, value_cache,
                      is_prompt, slot_mapping, max_context_len, context_lens, block_tables, scale
    }) {}

void TemplateExtension::PagedAttention::validate_and_infer_types() {
    // value_cache: shape = [num_blocks, num_kv_heads, head_size, block_size]
    auto value_cache_shape = get_input_partial_shape(4);
    // m_num_kv_heads = value_cache_shape[1];
    // m_head_size = value_cache_shape[2];
    // m_block_size = value_cache_shape[3];
    NODE_VALIDATION_CHECK(this,
        value_cache_shape.size() == 4,
        "Value cache shape must be 4 dims");

    // key_cache: shape [num_blocks, num_kv_heads, head_size/x, block_size, x]
    auto key_cache_shape = get_input_partial_shape(3);
    NODE_VALIDATION_CHECK(this,
        value_cache_shape.size() == 4,
        // value_cache_shape[0] == key_cache_shape[0] && // num_blocks
        // key_cache_shape[1] == m_num_kv_heads &&
        // key_cache_shape[2] * key_cache_shape[4] == m_head_size &&
        // m_block_size == key_cache_shape[3], // block_size,
        "Key cache shape must be 4 dims");

    // query: shape [batch_size, seq_len, num_heads * head_size]
    auto query_type = get_input_element_type(0);
    auto query_shape = get_input_partial_shape(0);
    NODE_VALIDATION_CHECK(this,
        // query_type.is_real() &&
        query_shape.size() == 3,
        // query_shape[2] == m_num_heads * m_head_size,
        "Query type must be real, shape must be like [batch_size, seq_len, num_heads * head_size]. ",
        "Got element type ", query_type, ", shape ", query_shape);

    // key: shape [batch_size, seq_len, num_kv_heads * head_size]
    auto key_type = get_input_element_type(1);
    auto key_shape = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(this,
        query_type == key_type &&
        key_shape.size() == 3,
        "Key type must be the same as query, shape must be the same as query. "
        "Got element type ", key_type, ", shape ", key_shape);

    // value: shape [batch_size, seq_len, num_kv_heads * head_size]
    auto value_type = get_input_element_type(2);
    auto value_shape = get_input_partial_shape(2);
    // NODE_VALIDATION_CHECK(this,
    //     key_type == value_type &&
    //     key_shape == value_shape, "Value type must be the same as key, shape must be the same as key. "
    //     "\nGot element type value ", value_type, ", shape ", value_shape,
    //     "\nGot element type ", key_type, ", shape ", key_shape);

    // is_prompt: boolean scalar
    NODE_VALIDATION_CHECK(this,
        // get_input_element_type(5) == ov::element::boolean && 
        get_input_shape(5) == ov::Shape({}),
        "is_prompt validation failed. ",
        "Got element type ", get_input_element_type(5), ", shape ", get_input_shape(5));

    // slot_mapping: shape [batch_size, max_context_len]
    auto slot_mapping_shape = get_input_partial_shape(6);
    NODE_VALIDATION_CHECK(this,
        // get_input_element_type(6) == ov::element::i64 &&
        slot_mapping_shape.size() == 2,
        "slot_mapping validation failed. ",
        "Got element type ", get_input_element_type(6), ", shape ", slot_mapping_shape);

    // max_context_len: integer scalar
    NODE_VALIDATION_CHECK(this,
        // get_input_element_type(7) == ov::element::i32 &&
        get_input_shape(7) == ov::Shape({}),
        "max_context_len validation failed. ",
        "Got element type ", get_input_element_type(7), ", shape ", get_input_shape(7));

    // context_lens: shape [batch_size]
    auto context_lens_shape = get_input_partial_shape(8);
    NODE_VALIDATION_CHECK(this,
        // get_input_element_type(8) == ov::element::i32 &&
        context_lens_shape.size() == 1,
        "context_lens validation failed. ",
        "Got element type ", get_input_element_type(8), ", shape ", context_lens_shape);

    // block_tables: shape [batch_size, max_block_per_request]
    NODE_VALIDATION_CHECK(this,
        // get_input_element_type(9) == ov::element::i32 &&
        get_input_partial_shape(9).size() == 2,
        "block_tables validation failed. ",
        "Got element type ", get_input_element_type(9), ", shape ", get_input_partial_shape(9));

    // scale: float scalar
    NODE_VALIDATION_CHECK(this,
        // get_input_element_type(10) == ov::element::f32 &&
        get_input_shape(10) == ov::Shape({}),
        "block_tables validation failed. ",
        "Got element type ", get_input_element_type(10), ", shape ", get_input_shape(10));

    set_output_type(0, query_type, query_shape);
}

std::shared_ptr<ov::Node> TemplateExtension::PagedAttention::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<PagedAttention>(new_args);
}

bool TemplateExtension::PagedAttention::has_evaluate() const {
    return get_input_element_type(0) == ov::element::f32;
}

// generate buttom diagonal boolean attention bias for a prefill stage
ov::Tensor generate_attention_bias(const std::size_t batch_size, const std::size_t seq_len, const ov::Tensor& context_lens) {
    ov::Shape attention_mask_shape({batch_size, 1, 1, seq_len, seq_len});
    ov::Tensor attention_mask(ov::element::f32, attention_mask_shape);
    int attention_mask_stride = attention_mask.get_strides()[0] / sizeof(float);

    static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");
    float negative_inf = -std::numeric_limits<float>::infinity();

    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        float * attention_mask_data = attention_mask.data<float>() + batch_id * attention_mask_stride;
        int left_window = context_lens.data<int>()[batch_id], right_window = 1;
        for (int y = 0; y < seq_len; ++y) {
            for (int x = 0; x < seq_len; ++x) {
                attention_mask_data[y * seq_len + x] = (x + right_window - 1) > y || (x + left_window - 1) < y ? negative_inf : 0.0f;
            }
        }
    }

    return attention_mask;
}

void print_tensor(ov::Tensor t) {
    auto size = std::min<size_t>(30, t.get_size());
    for (int i = 0; i < size; ++i)
        std::cout << t.data<float>()[i] << " ";
    std::cout << std::endl;
}

bool TemplateExtension::PagedAttention::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    ov::Tensor query = inputs[0], key = inputs[1], value = inputs[2];
    ov::Tensor key_cache = inputs[3], value_cache = inputs[4];
    const bool is_prompt = inputs[5].data<bool>()[0];
    ov::Tensor slot_mapping = inputs[6];
    const std::int32_t max_context_len = inputs[7].data<std::int32_t>()[0];
    ov::Tensor context_lens = inputs[8];
    ov::Tensor block_tables = inputs[9];
    float scale = inputs[10].data<float>()[0];

    // Shapes
    ov::Shape query_shape = query.get_shape();
    const std::size_t batch_size = query_shape[0], seq_len = query_shape[1], hidden_size = query_shape[2];

    ov::Shape value_cache_shape = value_cache.get_shape();
    const std::size_t num_kv_heads = value_cache_shape[1], head_size = value_cache_shape[2],
        num_heads = hidden_size / head_size, block_size = value_cache_shape[3];

    // print_tensor(query);
    // print_tensor(key);
    // print_tensor(value);

    // exit(1);

    // reshape to [batch_size * seq_len, m_num_kv_heads, head_size] from [batch_size, seq_len, num_heads/m_num_kv_heads * head_size]
    void * query_data = query.data(), * key_data = key.data(), * value_data = value.data();
    query.set_shape({batch_size * seq_len, num_heads, head_size});
    OPENVINO_ASSERT(query_data == query.data());
    key.set_shape({batch_size * seq_len, num_kv_heads, head_size});
    OPENVINO_ASSERT(key_data == key.data());
    value.set_shape(key.get_shape());
    OPENVINO_ASSERT(value_data == value.data());

    reshape_and_cache_cpu(key, value, key_cache, value_cache, slot_mapping);

    // set output shape
    OPENVINO_ASSERT(outputs.size() == 1);
    outputs[0].set_shape(query.get_shape());
    void * output_data = outputs[0].data();
    OPENVINO_ASSERT(output_data == outputs[0].data());

    if (is_prompt) {
        // reshape to [batch_size, seq_len, num_kv_heads, head_size]
        auto num_queries_per_kv = num_heads / num_kv_heads;
        query.set_shape({batch_size, seq_len, num_kv_heads, num_queries_per_kv, head_size});
        key.set_shape({batch_size, seq_len, num_kv_heads, 1, head_size});
        value.set_shape(key.get_shape());
        outputs[0].set_shape(query.get_shape());

        OPENVINO_ASSERT(query_data == query.data());
        OPENVINO_ASSERT(key_data == key.data());
        OPENVINO_ASSERT(value_data == value.data());
        OPENVINO_ASSERT(output_data == outputs[0].data());

        auto attention_bias = generate_attention_bias(batch_size, seq_len, context_lens);
        ov::Tensor scale_tensor(ov::element::f32, ov::Shape{1}, &scale);

        m_prefill_request.set_input_tensor(0, query);
        m_prefill_request.set_input_tensor(1, key);
        m_prefill_request.set_input_tensor(2, value);
        m_prefill_request.set_input_tensor(3, attention_bias);
        m_prefill_request.set_input_tensor(4, scale_tensor);
        m_prefill_request.set_output_tensor(outputs[0]);

        m_prefill_request.infer();
    } else {
        // 'query' and 'output' are expected to be [batch_size * seq_len, num_heads, head_size]
        paged_attention_v1_cpu(outputs[0],
            query, key_cache, value_cache,
            num_kv_heads, scale,
            block_tables, context_lens,
            block_size, max_context_len);
    }

    // reshape back to [batch_size, seq_len, num_heads * head_size]
    outputs[0].set_shape(query_shape);
    OPENVINO_ASSERT(output_data == outputs[0].data());

    return true;
}
