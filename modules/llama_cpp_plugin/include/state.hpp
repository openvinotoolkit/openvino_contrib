// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef LLAMA_CPP_PLUGIN_HPP
#define LLAMA_CPP_PLUGIN_HPP

#include "openvino/runtime/ivariable_state.hpp"
#include "compiled_model.hpp"

namespace ov {
    namespace llama_cpp_plugin {
        class LlamaCppState : public IVariableState {
        public:
            LlamaCppState() = delete;
            LlamaCppState(const std::shared_ptr<const LlamaCppModel>& model_ptr) : m_model_ptr(model_ptr), IVariableState("llama_cpp_state") {}
            void reset() override {
                std::cout << "VSHAMPOR: resetting state" << std::endl;
                llama_kv_cache_clear(m_model_ptr->m_llama_ctx);
            }
        private:
            const std::shared_ptr<const LlamaCppModel>& m_model_ptr;
        };
    }
}
#endif // LLAMA_CPP_STATE_HPP
