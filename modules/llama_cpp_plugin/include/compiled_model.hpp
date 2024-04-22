// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef LLAMA_CPP_COMPILED_MODEL_HPP
#define LLAMA_CPP_COMPILED_MODEL_HPP

#include "llama.h"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/isync_infer_request.hpp"

namespace ov {
namespace llama_cpp_plugin {
class LlamaCppSyncInferRequest;
class LlamaCppPlugin;
class LlamaCppState;
class LlamaCppModel : public ICompiledModel {
public:
    LlamaCppModel(const std::string& gguf_fname, const std::shared_ptr<const IPlugin>& plugin, size_t num_threads = 0);
    /**
     * @brief Export compiled model to stream
     *
     * @param model output stream
     */
    virtual void export_model(std::ostream& model) const override;

    /**
     * @brief Returns runtime model
     *
     * @return OpenVINO Model which represents runtime graph
     */
    virtual std::shared_ptr<const ov::Model> get_runtime_model() const override;

    /**
     * @brief Allows to set property
     *
     * @param properties new plugin properties
     */
    virtual void set_property(const ov::AnyMap& properties) override;

    /**
     * @brief Returns property
     *
     * @param name Property name
     *
     * @return Property value
     *              virtual std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;
     **/
    virtual ov::Any get_property(const std::string& name) const override;
    virtual const std::vector<ov::Output<const ov::Node>>& inputs() const override;
    virtual const std::vector<ov::Output<const ov::Node>>& outputs() const override;
    virtual ~LlamaCppModel();

protected:
    /**
     * @brief Method creates infer request implementation
     *
     * @return Sync infer request
     */
    virtual std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

private:
    gguf_context* m_gguf_ctx = nullptr;
    std::string m_gguf_fname;
    size_t m_num_threads;

    llama_model* m_llama_model_ptr = nullptr;
    llama_context* m_llama_ctx = nullptr;
    std::shared_ptr<ov::Model> m_fake_model;

    std::vector<ov::Output<const ov::Node>> m_fake_inputs;
    std::vector<ov::Output<const ov::Node>> m_fake_outputs;

    friend class ov::llama_cpp_plugin::LlamaCppSyncInferRequest;
    friend class ov::llama_cpp_plugin::LlamaCppState;
};
}  // namespace llama_cpp_plugin
}  // namespace ov

#endif  // LLAMA_CPP_COMPILED_MODEL_HPP
