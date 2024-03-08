#include "infer_request.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "llama.h"

namespace ov {
    namespace llama_cpp_plugin {

        void allocate_tensor_impl(ov::SoPtr<ov::ITensor>& tensor,
                          const ov::element::Type& element_type,
                          const ov::Shape& shape) {
    if (!tensor || tensor->get_element_type() != element_type) {
        tensor = ov::make_tensor(element_type, shape);
    } else {
        tensor->set_shape(shape);
    }
}

        LlamaCppSyncInferRequest::LlamaCppSyncInferRequest(const std::shared_ptr<const LlamaCppModel>& compiled_model): ov::ISyncInferRequest(compiled_model) {
            std::cout << "VSHAMPOR: infer request ctor called\n";
            m_compiled_model_ptr = compiled_model;
            // Allocate input/output tensors
            for (const auto& input : get_inputs()) {
                allocate_tensor(input, [input](ov::SoPtr<ov::ITensor>& tensor) {
                    // Can add a check to avoid double work in case of shared tensors
                    allocate_tensor_impl(tensor,
                                         input.get_element_type(),
                                         input.get_partial_shape().is_dynamic() ? ov::Shape{0} : input.get_shape());
                });
            }
            for (const auto& output : get_outputs()) {
                allocate_tensor(output, [output](ov::SoPtr<ov::ITensor>& tensor) {
                    // Can add a check to avoid double work in case of shared tensors
                    allocate_tensor_impl(tensor,
                                         output.get_element_type(),
                                         output.get_partial_shape().is_dynamic() ? ov::Shape{0} : output.get_shape());
                });
    }
        }
    void LlamaCppSyncInferRequest::set_tensors_impl(const ov::Output<const ov::Node> port, const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
        std::cout << "VSHAMPOR: set_tensors_impl called\n";
    }

    void llama_batch_add_reimpl(
                     struct llama_batch & batch,
                            llama_token   id,
                              llama_pos   pos,
        const std::vector<llama_seq_id> & seq_ids,
                                   bool   logits) {
        batch.token   [batch.n_tokens] = id;
        batch.pos     [batch.n_tokens] = pos;
        batch.n_seq_id[batch.n_tokens] = seq_ids.size();
        for (size_t i = 0; i < seq_ids.size(); ++i) {
            batch.seq_id[batch.n_tokens][i] = seq_ids[i];
        }
        batch.logits  [batch.n_tokens] = logits;

        batch.n_tokens++;
    }

    void LlamaCppSyncInferRequest::infer() {
        auto input_ids_tensor_ptr = get_tensor(get_inputs()[0]); // TODO (vshampor) correctly identify input_ids among all inputs without hardcode
        OPENVINO_ASSERT(input_ids_tensor_ptr->get_element_type() == ov::element::Type_t::i64);
        OPENVINO_ASSERT(input_ids_tensor_ptr->get_shape().size() == 2);
        size_t batch_size = input_ids_tensor_ptr->get_shape()[0];
        size_t sequence_length = input_ids_tensor_ptr->get_shape()[1];

        // llama_batch actually contains one sequence
        llama_batch batch = llama_batch_init(sequence_length, /* embd = */ 0, /* n_seq_max = */ 1);
        const int64_t* data_ptr = input_ids_tensor_ptr->data<int64_t>();

        const int64_t* sequence_start_ptr = data_ptr /* + seq_idx */;

        for (size_t tok_idx = 0; tok_idx < sequence_length; ++tok_idx) {
            const int64_t token_id = sequence_start_ptr[tok_idx];
            llama_batch_add_reimpl(batch, token_id, *(m_compiled_model_ptr->num_tokens_processed_ptr), { 0 }, true); // the last `true` here is a marker that the logits for this token should be computed and returned
            size_t* ptr = m_compiled_model_ptr->num_tokens_processed_ptr;
            (*ptr)++;
        }


        llama_context* ctx = m_compiled_model_ptr->m_llama_ctx;
        int32_t sts = llama_decode(ctx, batch);

        if (sts != 0) {
            OPENVINO_THROW("llama_decode failed with code ", sts);
        }

        size_t n_vocab = llama_n_vocab(m_compiled_model_ptr->m_llama_model_ptr);

        ov::Tensor output_tensor{ov::element::Type_t::f32, {1, sequence_length, n_vocab}};
        float* output_tensor_data_ptr = output_tensor.data<float>();

        for (size_t pos = 0; pos < sequence_length; pos++) {
            float* logits_from_llama = llama_get_logits_ith(ctx, pos);
            std::copy(logits_from_llama, logits_from_llama + n_vocab, output_tensor_data_ptr + pos * n_vocab);
        }

        auto& logit_output = get_outputs()[0];
        allocate_tensor(logit_output, [&output_tensor](ov::SoPtr<ov::ITensor>& tensor) { allocate_tensor_impl(tensor, output_tensor.get_element_type(), output_tensor.get_shape());
                                                                                         output_tensor.copy_to(ov::make_tensor(tensor)); });
    };
    std::vector<ov::ProfilingInfo> LlamaCppSyncInferRequest::get_profiling_info() const {
        std::cout << "VSHAMPOR: get_profiling_info() called\n";
        return std::vector<ov::ProfilingInfo>{};
    };
    std::vector<ov::SoPtr<ov::IVariableState>> LlamaCppSyncInferRequest::query_state() const {
        std::cout << "VSHAMPOR: get_profiling_info() called\n";
        return std::vector<ov::SoPtr<ov::IVariableState>>{};
    }
    }
}  // namespace ov
