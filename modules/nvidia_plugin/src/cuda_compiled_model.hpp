// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_async_infer_request.hpp"
#include "cuda_config.hpp"
#include "cuda_infer_request.hpp"
#include "cuda_itopology_runner.hpp"
#include "cuda_op_buffers_extractor.hpp"
#include "memory_manager/cuda_device_mem_block.hpp"
#include "memory_manager/cuda_memory_manager.hpp"
#include "memory_manager/cuda_memory_pool.hpp"
#include "memory_manager/model/cuda_memory_model.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/threading/itask_executor.hpp"
#include "ops/subgraph.hpp"

namespace ov {
namespace nvidia_gpu {

class Plugin;

/**
 * @class CompiledModel
 * @brief Implementation of compiled model
 */
class CompiledModel : public ov::ICompiledModel {
public:
    CompiledModel(const std::shared_ptr<const ov::Model>& model,
                  const Configuration& cfg,
                  const std::shared_ptr<ov::threading::ITaskExecutor>& wait_executor,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  bool loaded_from_cache);

    ~CompiledModel();

    // Methods from a base class ov::ICompiledModel

    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void export_model(std::ostream& model) const override;

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name) const override;

    const ITopologyRunner& get_topology_runner() const;

    const std::shared_ptr<MemoryPool>& get_memory_pool() const;

protected:
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

private:
    friend class CudaInferRequest;
    friend class Plugin;
    void compile_model(const std::shared_ptr<const ov::Model>& model);
    void init_executor();
    std::size_t get_optimal_number_of_streams(std::size_t const_blob_size, std::size_t memory_blob_size) const;
    std::shared_ptr<ov::ISyncInferRequest> create_benchmark_sync_infer_request();
    std::shared_ptr<ov::IAsyncInferRequest> create_benchmark_infer_request();
    std::shared_ptr<MemoryPool> create_memory_pool();
    void benchmark_optimal_number_of_requests();
    unsigned int run_benchmark_for(int numInfers, std::mutex& mtx, std::condition_variable& cond_var);
    void instantiate_cuda_graphs();

    mutable std::atomic<std::size_t> request_id_ = {0};
    Configuration config_;
    std::shared_ptr<ov::threading::ITaskExecutor> cuda_stream_executor_ = nullptr;
    std::shared_ptr<ov::Model> model_;
    std::map<std::string, std::size_t> input_index_;
    std::map<std::string, std::size_t> output_index_;
    std::unique_ptr<ITopologyRunner> topology_runner_;
    std::shared_ptr<MemoryPool> memory_pool_;
    const bool loaded_from_cache_;
    bool use_cuda_graph_;
    size_t number_of_cuda_graphs_;
};

}  // namespace nvidia_gpu
}  // namespace ov
