// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/function.hpp>
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>

#include "cuda_config.hpp"
#include "cuda_infer_request.hpp"
#include "cuda_async_infer_request.hpp"
#include "cuda_tensor_collector.hpp"

#include "memory_manager/model/cuda_memory_model.hpp"
#include "memory_manager/cuda_device_mem_block.hpp"

class ExecNetworkTest;

namespace CUDAPlugin {

class Plugin;

/**
 * @class ExecutableNetwork
 * @brief Interface of executable network
 */
class ExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    ExecutableNetwork(const InferenceEngine::CNNNetwork& cnnNetwork,
                      Configuration cfg,
                      std::shared_ptr<Plugin> plugin);
    ExecutableNetwork(std::istream& model,
                      Configuration cfg,
                      std::shared_ptr<Plugin> plugin);

    ~ExecutableNetwork() override = default;

    // Methods from a base class ExecutableNetworkThreadSafeDefault

    InferenceEngine::CNNNetwork GetExecGraphInfo() override;
    void ExportImpl(std::ostream& model) override;
    InferenceEngine::InferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                      InferenceEngine::OutputsDataMap networkOutputs) override;
    InferenceEngine::IInferRequest::Ptr CreateInferRequest() override;
    InferenceEngine::Parameter GetMetric(const std::string &name) const override;
    InferenceEngine::Parameter GetConfig(const std::string &name) const override;

private:
    friend class ::ExecNetworkTest;
    friend class CudaInferRequest;

    void CompileNetwork(const std::shared_ptr<const ngraph::Function>& function);
    void InitExecutor();
    void MemoryManagerComponentsSnippets();

    std::atomic<std::size_t>                    request_id_ = {0};
    InferenceEngine::CNNNetwork                 cnn_network_;
    Configuration                               cfg_;
    std::shared_ptr<Plugin>                     plugin_;
    std::shared_ptr<ngraph::Function>           function_;
    std::shared_ptr<DeviceMemBlock>              shared_constants_blob_;
    MemoryModel::Ptr                            memory_model_;
    std::unique_ptr<TensorCollector>            tensor_collector_;
    std::vector<OperationBase::Ptr>             exec_sequence_;
    std::map<std::string, std::size_t>          input_index_;
    std::map<std::string, std::size_t>          output_index_;
};

}  // namespace CUDAPlugin
