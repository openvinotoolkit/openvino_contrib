// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/threading/itask_executor.hpp"

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

namespace ov {
namespace nvidia_gpu {

class Plugin;

/**
 * @class ExecutableNetwork
 * @brief Interface of executable network
 */
class CompiledModel : public ov::ICompiledModel {
public:
    friend class Plugin;

    CompiledModel(const std::shared_ptr<const ov::Model>& model,
                  const Configuration& cfg,
                  const std::shared_ptr<ov::threading::ITaskExecutor>& wait_executor,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  bool loaded_from_cache = false);

    ~CompiledModel();

    // Methods from a base class ov::ICompiledModel

    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void export_model(std::ostream& model) const override;

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name) const override;

    std::string new_request_name() {
        return "Cuda" + std::to_string(config_.get_device_id()) + "_" + model_->get_friendly_name() + "_Req" +
               std::to_string(request_id_++);
    }
    const ov::op::v0::Parameter& parameter(const std::string& name) const {
        return *model_->get_parameters().at(input_index_.at(name));
    }
    const ov::op::v0::Result& result(const std::string& name) const {
        return *model_->get_results().at(output_index_.at(name));
    }

protected:
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

private:
    friend class ::ExecNetworkTest;
    friend class CudaInferRequest;
    void compile_model(const std::shared_ptr<const ov::Model>& model);
    void init_executor();
    std::size_t get_optimal_number_of_streams(std::size_t const_blob_size, std::size_t memory_blob_size) const;
    std::shared_ptr<ov::IAsyncInferRequest> create_benchmark_infer_request() const;
    std::shared_ptr<MemoryPool> create_memory_pool();
    void benchmark_optimal_number_of_requests();
    unsigned int run_benchmark_for(int numInfers, std::mutex& mtx, std::condition_variable& cond_var);

    mutable std::atomic<std::size_t> request_id_ = {0};
    Configuration config_;
    std::shared_ptr<ov::threading::ITaskExecutor> cuda_stream_executor_ = nullptr;
    std::shared_ptr<ov::Model> model_;
    std::map<std::string, std::size_t> input_index_;
    std::map<std::string, std::size_t> output_index_;
    std::unique_ptr<ExecGraph> graph_;
    std::shared_ptr<MemoryPool> memory_pool_;
    const bool loaded_from_cache_;
};

}  // namespace nvidia_gpu
}  // namespace ov
