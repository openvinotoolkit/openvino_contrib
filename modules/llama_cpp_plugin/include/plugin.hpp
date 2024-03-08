// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLAMA_CPP_PLUGIN_HPP
#define LLAMA_CPP_PLUGIN_HPP

#include "openvino/runtime/iplugin.hpp"

namespace ov {
    namespace llama_cpp_plugin {
        class LlamaCppPlugin : public IPlugin {
        public:
            LlamaCppPlugin();
            /**
             * @brief Compiles model from ov::Model object
             * @param model A model object acquired from ov::Core::read_model or source construction
             * @param properties A ov::AnyMap of properties relevant only for this load operation
             * @return Created Compiled Model object
             */
            virtual std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                const ov::AnyMap& properties) const override;


            /**
             * @brief Compiles model from ov::Model object, on specified remote context
             * @param model A model object acquired from ov::Core::read_model or source construction
             * @param properties A ov::AnyMap of properties relevant only for this load operation
             * @param context A pointer to plugin context derived from RemoteContext class used to
             *        execute the model
             * @return Created Compiled Model object
             */
            virtual std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                const ov::AnyMap& properties,
                const ov::SoPtr<ov::IRemoteContext>& context) const override;

            /**
             * @brief Sets properties for plugin, acceptable keys can be found in openvino/runtime/properties.hpp
             * @param properties ov::AnyMap of properties
             */
            virtual void set_property(const ov::AnyMap& properties) override;

            /**
             * @brief Gets properties related to plugin behaviour.
             *
             * @param name Property name.
             * @param arguments Additional arguments to get a property.
             *
             * @return Value of a property corresponding to the property name.
             */
            virtual ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;

            /**
             * @brief Creates a remote context instance based on a map of properties
             * @param remote_properties Map of device-specific shared context remote properties.
             *
             * @return A remote context object
             */
            virtual ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override;

            /**
             * @brief Provides a default remote context instance if supported by a plugin
             * @param remote_properties Map of device-specific shared context remote properties.
             *
             * @return The default context.
             */
            virtual ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override;

            /**
             * @brief Creates an compiled model from an previously exported model using plugin implementation
             *        and removes OpenVINO Runtime magic and plugin name
             * @param model Reference to model output stream
             * @param properties A ov::AnyMap of properties
             * @return An Compiled model
             */
            virtual std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                const ov::AnyMap& properties) const override;


            virtual std::shared_ptr<ov::ICompiledModel> compile_model(const std::string& fname,
                const ov::AnyMap& properties) const override;

            /**
             * @brief Creates an compiled model from an previously exported model using plugin implementation
             *        and removes OpenVINO Runtime magic and plugin name
             * @param model Reference to model output stream
             * @param context A pointer to plugin context derived from RemoteContext class used to
             *        execute the network
             * @param properties A ov::AnyMap of properties
             * @return An Compiled model
             */
            virtual std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                const ov::SoPtr<ov::IRemoteContext>& context,
                const ov::AnyMap& properties) const override;

            /**
             * @brief Queries a plugin about supported layers in model
             * @param model Model object to query.
             * @param properties Optional map of pairs: (property name, property value).
             * @return An object containing a map of pairs an operation name -> a device name supporting this operation.
             */
            virtual ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                const ov::AnyMap& properties) const override;

            std::string get_current_gguf_file_path() const;
        private:
            std::string m_cache_dir = "./";
        };
    }  // namespace llama_cpp_plugin
}  // namespace ov

#endif // LLAMA_CPP_PLUGIN_HPP
