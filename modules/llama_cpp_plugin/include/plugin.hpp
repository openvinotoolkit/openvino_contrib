// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef LLAMA_CPP_PLUGIN_HPP
#define LLAMA_CPP_PLUGIN_HPP

#include "openvino/runtime/iplugin.hpp"

namespace ov {
namespace llama_cpp_plugin {
class LlamaCppPlugin : public IPlugin {
public:
    LlamaCppPlugin();
    virtual std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                              const ov::AnyMap& properties) const override;

    virtual std::shared_ptr<ov::ICompiledModel> compile_model(
        const std::shared_ptr<const ov::Model>& model,
        const ov::AnyMap& properties,
        const ov::SoPtr<ov::IRemoteContext>& context) const override;

    virtual void set_property(const ov::AnyMap& properties) override;

    virtual ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;

    virtual ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override;

    virtual ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override;

    virtual std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                             const ov::AnyMap& properties) const override;

    virtual std::shared_ptr<ov::ICompiledModel> compile_model(const std::string& fname,
                                                              const ov::AnyMap& properties) const override;

    virtual std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                             const ov::SoPtr<ov::IRemoteContext>& context,
                                                             const ov::AnyMap& properties) const override;
    virtual ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                            const ov::AnyMap& properties) const override;


    virtual std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& model, const ov::AnyMap& properties) const override;

    virtual std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& model,
                                                               const ov::SoPtr<ov::IRemoteContext>& context,
                                                               const ov::AnyMap& properties) const override;

private:
    size_t m_num_threads = 0;
};
}  // namespace llama_cpp_plugin
}  // namespace ov

#endif  // LLAMA_CPP_PLUGIN_HPP
