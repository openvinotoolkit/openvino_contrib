// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/function.hpp>

#include "cuda_config.hpp"
#include "cuda_infer_request.hpp"
#include "cuda_async_infer_request.hpp"

#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>

namespace CUDAPlugin {

class Plugin;

/**
 * @class ExecutableNetwork
 * @brief Interface of executable network
 */
// ! [executable_network:header]
class ExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    ExecutableNetwork(const std::shared_ptr<const ngraph::Function>& function,
                      Configuration cfg,
                      std::shared_ptr<Plugin> plugin);

    ExecutableNetwork(std::istream& model,
                      Configuration cfg,
                      std::shared_ptr<Plugin> plugin);

    ~ExecutableNetwork() override = default;

    // Methods from a base class ExecutableNetworkThreadSafeDefault

    void ExportImpl(std::ostream& model) override;
    InferenceEngine::InferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                      InferenceEngine::OutputsDataMap networkOutputs) override;
    InferenceEngine::IInferRequest::Ptr CreateInferRequest() override;
    InferenceEngine::Parameter GetMetric(const std::string &name) const override;
    InferenceEngine::Parameter GetConfig(const std::string &name) const override;

private:
    friend class CudaInferRequest;

    void CompileNetwork(const std::shared_ptr<const ngraph::Function>& function);
    void InitExecutor();

    std::atomic<std::size_t>                    _requestId = {0};
    Configuration                               _cfg;
    std::shared_ptr<Plugin>                     _plugin;
    std::shared_ptr<ngraph::Function>           _function;
    std::vector<OperationBase::Ptr>             _nodes;
    std::map<std::string, std::size_t>          _inputIndex;
    std::map<std::string, std::size_t>          _outputIndex;
};
// ! [executable_network:header]

}  // namespace CUDAPlugin
