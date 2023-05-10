// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/threading/itask_executor.hpp"

#include "cuda_config.hpp"
#include "cuda_compiled_model.hpp"
#include "cuda_thread_pool.hpp"
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
                                                      const ov::RemoteContext& context) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model_stream,
                                                     const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model_stream,
                                                     const ov::RemoteContext& context,
                                                     const ov::AnyMap& properties) const override;

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override;

    std::shared_ptr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override;

private:
    friend class ExecutableNetwork;
    friend class CudaInferRequest;

    enum class cuda_attribute { name };

    /**
     * Gets CpuStreamExecutor if it was already created,
     * creates otherwise one for device specified in cfg
     * @param cfg Configuration used for CpuStreamExecutor creation
     * @return CpuStreamExecutor
     */
    std::shared_ptr<ov::threading::ITaskExecutor> get_stream_executor(const Configuration& cfg) const;

    int nvidia_device_id() const noexcept { return 0; }  // TODO implement

    bool is_operation_supported(const std::shared_ptr<ov::Node>& node) const;

    GraphTransformer transformer_{};
    Configuration config_;
    std::unordered_map<std::string, std::shared_ptr<CudaThreadPool>> device_thread_pool_;
};

}  // namespace nvidia_gpu
}  // namespace ov
