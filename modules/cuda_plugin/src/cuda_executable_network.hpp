// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include <ngraph/function.hpp>

#include "cuda_async_infer_request.hpp"
#include "cuda_config.hpp"
#include "cuda_graph.hpp"
#include "cuda_infer_request.hpp"
#include "cuda_op_buffers_extractor.hpp"
#include "memory_manager/cuda_device_mem_block.hpp"
#include "memory_manager/cuda_memory_manager.hpp"
#include "memory_manager/cuda_memory_pool.hpp"
#include "memory_manager/model/cuda_memory_model.hpp"
#include "ops/subgraph.hpp"

class ExecNetworkTest;

namespace CUDAPlugin {

class Plugin;

/**
 * @class ExecutableNetwork
 * @brief Interface of executable network
 */
class ExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    friend class Plugin;
    ExecutableNetwork(const InferenceEngine::CNNNetwork& cnnNetwork,
                      Configuration cfg,
                      InferenceEngine::ITaskExecutor::Ptr waitExecutor,
                      std::shared_ptr<Plugin> plugin);
    ExecutableNetwork(std::istream& model,
                      Configuration cfg,
                      InferenceEngine::ITaskExecutor::Ptr waitExecutor,
                      std::shared_ptr<Plugin> plugin);

    // Methods from a base class ExecutableNetworkThreadSafeDefault

    std::shared_ptr<ngraph::Function> GetExecGraphInfo() override;
    void Export(std::ostream& model) override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(
        InferenceEngine::InputsDataMap networkInputs,
        InferenceEngine::OutputsDataMap networkOutputs) override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(
        const std::vector<std::shared_ptr<const ov::Node>>& inputs,
        const std::vector<std::shared_ptr<const ov::Node>>& outputs) override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequest() override;

    InferenceEngine::Parameter GetMetric(const std::string& name) const override;
    InferenceEngine::Parameter GetConfig(const std::string& name) const override;
    std::string newRequestName() {
        return "Cuda" + std::to_string(cfg_.deviceId) + "_" + export_function_->get_friendly_name() + "_Req" +
               std::to_string(request_id_++);
    }
    const ov::op::v0::Parameter& parameter(const std::string& name) const {
        return *function_->get_parameters().at(input_index_.at(name));
    }
    const ov::op::v0::Result& result(const std::string& name) const {
        return *function_->get_results().at(output_index_.at(name));
    }

private:
    friend class ::ExecNetworkTest;
    friend class CudaInferRequest;
    void CompileNetwork(const std::shared_ptr<const ngraph::Function>& function,
                        const InferenceEngine::InputsDataMap& inputInfoMap,
                        const InferenceEngine::OutputsDataMap& outputsInfoMap);
    void InitExecutor();
    std::size_t GetOptimalNumberOfStreams(std::size_t constBlobSize, std::size_t memoryBlobSize) const;
    InferenceEngine::IInferRequestInternal::Ptr CreateBenchmarkInferRequestImpl(
        InferenceEngine::InputsDataMap networkInputs, InferenceEngine::OutputsDataMap networkOutputs);
    InferenceEngine::IInferRequestInternal::Ptr CreateBenchmarkInferRequest();
    std::shared_ptr<MemoryPool> CreateMemoryPool();
    int GetCudaDeviceId() const noexcept;
    void BenchmarkOptimalNumberOfRequests();
    unsigned int RunBenchmarkFor(int numInfers, std::mutex& mtx, std::condition_variable& cond_var);

    std::atomic<std::size_t> request_id_ = {0};
    Configuration cfg_;
    InferenceEngine::ITaskExecutor::Ptr cuda_stream_executor_;
    std::shared_ptr<Plugin> plugin_;
    std::shared_ptr<ngraph::Function> export_function_;
    std::shared_ptr<ngraph::Function> function_;
    std::map<std::string, std::size_t> input_index_;
    std::map<std::string, std::size_t> output_index_;
    std::unique_ptr<CudaGraph> graph_;
    std::shared_ptr<MemoryPool> memory_pool_;
};

}  // namespace CUDAPlugin
