// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_compiled_model.hpp"
#include "cuda_config.hpp"
#include "cuda_thread_pool.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/threading/itask_executor.hpp"
#include "transformer/cuda_graph_transformer.hpp"

namespace ov {
namespace nvidia_gpu {

class Plugin : public ov::IPlugin {
public:
    using Ptr = std::shared_ptr<Plugin>;

    Plugin();
    ~Plugin();

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties,
                                                      const ov::SoPtr<ov::IRemoteContext>& context) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model_stream,
                                                     const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model_stream,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& model_stream,
                                                     const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& model_stream,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override;

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;

    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override;

    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override;

private:
    friend class CompiledModel;
    friend class CudaInferRequest;

    enum class cuda_attribute { name };

    /**
     * Gets CudaThreadPool
     * @param config Configuration used for CudaThreadPool selection
     * @return CudaThreadPool
     */
    std::shared_ptr<ov::threading::ITaskExecutor> get_stream_executor(const Configuration& config) const;

    /**
     * Check whether node is supported by plugin
     * @param node node to be checked
     * @return true if supported, false otherwise
     */
    bool is_operation_supported(const std::shared_ptr<ov::Node>& node, const Configuration& config) const;

    Configuration get_full_config(const ov::AnyMap& properties, const bool throw_on_unsupported = true) const;

    GraphTransformer transformer_{};
    std::string default_device_id = "0";
    std::map<std::string, Configuration> configs_;
    std::unordered_map<std::string, std::shared_ptr<CudaThreadPool>> device_thread_pool_;
};

}  // namespace nvidia_gpu
}  // namespace ov
