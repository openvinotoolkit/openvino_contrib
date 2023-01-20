// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>

#include "cuda_config.hpp"
#include "cuda_executable_network.hpp"
#include "cuda_thread_pool.hpp"
#include "transformer/cuda_graph_transformer.hpp"

namespace ov {
namespace nvidia_gpu {

class Plugin : public InferenceEngine::IInferencePlugin {
public:
    using Ptr = std::shared_ptr<Plugin>;

    Plugin();
    ~Plugin();

    void SetConfig(const std::map<std::string, std::string>& config) override;
    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                     const std::map<std::string, std::string>& config) const override;
    InferenceEngine::IExecutableNetworkInternal::Ptr LoadExeNetworkImpl(
        const InferenceEngine::CNNNetwork& network, const std::map<std::string, std::string>& config) override;
    InferenceEngine::Parameter GetConfig(
        const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const override;
    InferenceEngine::Parameter GetMetric(
        const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const override;
    InferenceEngine::IExecutableNetworkInternal::Ptr ImportNetwork(
        std::istream& model, const std::map<std::string, std::string>& config) override;

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
    InferenceEngine::ITaskExecutor::Ptr GetStreamExecutor(const Configuration& cfg);

    int cudaDeviceID() const noexcept { return 0; }  // TODO implement

    bool isOperationSupported(const std::shared_ptr<ov::Node>& node) const;

    std::mutex mtx_;
    GraphTransformer transformer_{};
    Configuration _cfg;
    std::unordered_map<std::string, InferenceEngine::ITaskExecutor::Ptr> _waitExecutors;
    std::unordered_map<std::string, std::shared_ptr<CudaThreadPool>> device_thread_pool_;
};

}  // namespace nvidia_gpu
}  // namespace ov
