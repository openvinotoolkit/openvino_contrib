// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_config.hpp"
#include "cuda_executable_network.hpp"
#include <cpp_interfaces/impl/ie_plugin_internal.hpp>

#include "backend.hpp"

namespace CUDAPlugin {

class Plugin : public InferenceEngine::InferencePluginInternal {
public:
    using Ptr = std::shared_ptr<Plugin>;

    Plugin();
    ~Plugin();

    void SetConfig(const std::map<std::string, std::string> &config) override;
    InferenceEngine::QueryNetworkResult
    QueryNetwork(const InferenceEngine::CNNNetwork &network,
                 const std::map<std::string, std::string>& config) const override;
    InferenceEngine::ExecutableNetworkInternal::Ptr
    LoadExeNetworkImpl(const InferenceEngine::CNNNetwork &network,
                       const std::map<std::string, std::string> &config) override;
    void AddExtension(InferenceEngine::IExtensionPtr extension) override;
    InferenceEngine::Parameter GetConfig(const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter> & options) const override;
    InferenceEngine::Parameter GetMetric(const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter> & options) const override;
    InferenceEngine::ExecutableNetwork ImportNetworkImpl(std::istream& model,
        const std::map<std::string, std::string>& config) override;

private:
    friend class ExecutableNetwork;
    friend class CudaInferRequest;

    using CudaStreamMapping = std::unordered_map<int, std::shared_ptr<CudaStream>>;

    enum class cuda_attribute {
        name
    };

    InferenceEngine::IStreamsExecutor::Ptr GetStreamExecutor(const Configuration &cfg);

    template <cuda_attribute ID, class Result>
    Result getCudaAttribute() const;

    int cudaDeviceID() const noexcept { return 0; } //TODO implement

    std::shared_ptr<ngraph::runtime::Backend> _backend;
    Configuration _cfg;
    std::unordered_map<std::string, InferenceEngine::ITaskExecutor::Ptr> _waitExecutors;
    std::unordered_map<std::string, CudaStreamMapping> device_cuda_streams_;
};

template <>
std::string Plugin::getCudaAttribute<Plugin::cuda_attribute::name, std::string>() const;


}  // namespace CUDAPlugin
