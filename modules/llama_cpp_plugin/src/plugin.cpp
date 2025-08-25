// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "plugin.hpp"

#include <openvino/runtime/properties.hpp>

#include "compiled_model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/util/log.hpp"

namespace {
static constexpr const char* wait_executor_name = "LlamaCppWaitExecutor";
static constexpr const char* stream_executor_name = "LlamaCppStreamsExecutor";
static constexpr const char* template_exclusive_executor = "LlamaCppExecutor";
}  // namespace

namespace ov {
namespace llama_cpp_plugin {
LlamaCppPlugin::LlamaCppPlugin() : IPlugin() {
    set_device_name("LLAMA_CPP");
}
std::shared_ptr<ov::ICompiledModel> LlamaCppPlugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                                  const ov::AnyMap& properties) const {
    OPENVINO_THROW_NOT_IMPLEMENTED("Currently only direct GGUF file loading is "
                                   "supported for the LLAMA_CPP* plugins");
}

std::shared_ptr<ov::ICompiledModel> LlamaCppPlugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                                  const ov::AnyMap& properties,
                                                                  const ov::SoPtr<ov::IRemoteContext>& context) const {
    OPENVINO_THROW_NOT_IMPLEMENTED("Currently only direct GGUF file loading is "
                                   "supported for the LLAMA_CPP* plugins");
}

std::shared_ptr<ov::ICompiledModel> LlamaCppPlugin::import_model(const ov::Tensor& model, const ov::AnyMap& properties) const override {
    OPENVINO_THROW("This method may not be used with LLAMA_CPP* plugins");
}

std::shared_ptr<ov::ICompiledModel> LlamaCppPlugin::import_model(const ov::Tensor& model,
                                                 const ov::SoPtr<ov::IRemoteContext>& context,
                                                 const ov::AnyMap& properties) const {
    OPENVINO_THROW("This method may not be used with LLAMA_CPP* plugins");
}


std::shared_ptr<ov::ICompiledModel> LlamaCppPlugin::compile_model(const std::string& fname,
                                                                  const ov::AnyMap& properties) const {
    size_t num_threads = 0;
    auto it = properties.find(ov::inference_num_threads.name());
    if (it != properties.end()) {
        num_threads = it->second.as<int>();
        OPENVINO_ASSERT(num_threads >= 0, "INFERENCE_NUM_THREADS cannot be negative");
    } else {
        num_threads = m_num_threads;
    }
    return std::make_shared<LlamaCppModel>(fname, shared_from_this(), num_threads);
}

void LlamaCppPlugin::set_property(const ov::AnyMap& properties) {
    for (const auto& map_entry : properties) {
        if (ov::inference_num_threads == map_entry.first) {
            int num_threads = map_entry.second.as<int>();
            OPENVINO_ASSERT(num_threads >= 0, "INFERENCE_NUM_THREADS cannot be negative");
            m_num_threads = num_threads;
        }
        OPENVINO_THROW_NOT_IMPLEMENTED("llama_cpp_plugin: setting property ", map_entry.first, "not implemented");
    }
}

ov::Any LlamaCppPlugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    if (ov::supported_properties == name) {
        return decltype(ov::supported_properties)::value_type(
            std::vector<PropertyName>({ov::device::capabilities, ov::device::full_name, ov::inference_num_threads}));
    }
    if (ov::device::capabilities == name) {
        return decltype(ov::device::capabilities)::value_type(
            std::vector<std::string>({ov::device::capability::EXPORT_IMPORT}));
    }
    if (ov::internal::supported_properties == name) {
        return decltype(ov::internal::supported_properties)::value_type(
            std::vector<PropertyName>({ov::internal::caching_properties}));
    }

    if (ov::internal::caching_properties == name) {
        return std::vector<ov::PropertyName>{ov::device::full_name};
    }

    if (ov::device::full_name == name) {
        return std::string("LLAMA_CPP");
    }

    if (ov::inference_num_threads == name) {
        return m_num_threads;
    }

    OPENVINO_THROW_NOT_IMPLEMENTED("llama_cpp_plugin: getting property ", name, "not implemented");
}

ov::SoPtr<ov::IRemoteContext> LlamaCppPlugin::create_context(const ov::AnyMap& remote_properties) const {
    OPENVINO_THROW_NOT_IMPLEMENTED("llama_cpp_plugin: Not Implemented");
}
ov::SoPtr<ov::IRemoteContext> LlamaCppPlugin::get_default_context(const ov::AnyMap& remote_properties) const {
    OPENVINO_THROW_NOT_IMPLEMENTED("llama_cpp_plugin: Not Implemented");
}
std::shared_ptr<ov::ICompiledModel> LlamaCppPlugin::import_model(std::istream& model_file_stream,
                                                                 const ov::AnyMap& properties) const {
    OPENVINO_THROW_NOT_IMPLEMENTED("llama_cpp_plugin: model importing not implemented");
}

std::shared_ptr<ov::ICompiledModel> LlamaCppPlugin::import_model(std::istream& model,
                                                                 const ov::SoPtr<ov::IRemoteContext>& context,
                                                                 const ov::AnyMap& properties) const {
    OPENVINO_THROW_NOT_IMPLEMENTED("llama_cpp_plugin: model importing not implemented");
}

ov::SupportedOpsMap LlamaCppPlugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                                const ov::AnyMap& properties) const {
    OPENVINO_THROW_NOT_IMPLEMENTED("llama_cpp_plugin: model importing not implemented");
}
}  // namespace llama_cpp_plugin
}  // namespace ov

static const ov::Version version = {CI_BUILD_NUMBER, "llama_cpp_plugin"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(ov::llama_cpp_plugin::LlamaCppPlugin, version)
