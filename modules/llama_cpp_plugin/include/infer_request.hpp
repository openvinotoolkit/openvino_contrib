// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef LLAMA_CPP_INFER_REQUEST_HPP
#define LLAMA_CPP_INFER_REQUEST_HPP

#include "compiled_model.hpp"
#include "openvino/openvino.hpp"

namespace ov {
namespace llama_cpp_plugin {

class LlamaCppSyncInferRequest : public ISyncInferRequest {
public:
    explicit LlamaCppSyncInferRequest(const std::shared_ptr<const LlamaCppModel>& compiled_model, size_t num_threads);
    virtual ~LlamaCppSyncInferRequest() override;

    virtual void set_tensors_impl(const ov::Output<const ov::Node> port,
                                  const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override;

    virtual void infer() override;
    virtual std::vector<ov::ProfilingInfo> get_profiling_info() const override;
    virtual std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;

private:
    std::shared_ptr<const LlamaCppModel> m_compiled_model_ptr;
    llama_context* m_llama_ctx;
};

}  // namespace llama_cpp_plugin
};  // namespace ov

#endif /* LLAMA_CPP_INFER_REQUEST_HPP */
