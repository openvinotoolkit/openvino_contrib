// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory_manager/cuda_device_mem_block.hpp>

#include "cancellation_token.hpp"
#include "cuda_dynamic_buffer_context.hpp"
#include "cuda_graph_context.hpp"
#include "cuda_tensor_mapping_context.hpp"
#include "cuda_thread_context.hpp"

namespace ov {
namespace nvidia_gpu {

class IExecutionDelegator;
class DynamicOperationCache;

class InferenceRequestContext {
public:
    /**
     * @brief A smart pointer to the InferenceRequestContext object
     */
    using Ptr = std::shared_ptr<InferenceRequestContext>;
    using WeakPtr = std::weak_ptr<InferenceRequestContext>;

    InferenceRequestContext(const std::vector<std::shared_ptr<ov::Tensor>>& inputs,
                            const std::map<std::string, std::size_t>& inputMapping,
                            const std::vector<std::shared_ptr<ov::Tensor>>& outputs,
                            const std::map<std::string, std::size_t>& outputMapping,
                            const ThreadContext& threadContext,
                            CancellationToken& token,
                            IExecutionDelegator& executionDelegator,
                            CudaGraphContext& cudaGraphContext,
                            DynamicOperationCache& dynamicShapeCache,
                            bool isBenchmarkMode = false)
        : threadContext{threadContext},
          token{token},
          executionDelegator{executionDelegator},
          tensor_mapping_context_{inputs, inputMapping, outputs, outputMapping},
          cuda_graph_context_{cudaGraphContext},
          dynamic_op_cache_{dynamicShapeCache},
          is_benchmark_mode_{isBenchmarkMode} {}

    // don't allow storing references to temporary
    template <typename... Args>
    InferenceRequestContext(std::vector<std::shared_ptr<ov::Tensor>>&& inputs,
                            std::map<std::string, std::size_t>&& inputMapping,
                            std::vector<std::shared_ptr<ov::Tensor>>&& outputs,
                            std::map<std::string, std::size_t>&& outputMapping,
                            Args... args) = delete;

    InferenceRequestContext(std::vector<std::shared_ptr<ov::Tensor>>&& inputs,
                            std::map<std::string, std::size_t>&& inputMapping,
                            std::vector<std::shared_ptr<ov::Tensor>>&& outputs,
                            std::map<std::string, std::size_t>&& outputMapping,
                            const ThreadContext& threadContext) = delete;

    const ThreadContext& getThreadContext() const noexcept { return threadContext; }
    [[nodiscard]] ov::nvidia_gpu::CancellationToken& getCancellationToken() const noexcept { return token; }
    [[nodiscard]] IExecutionDelegator& getExecutionDelegator() const noexcept { return executionDelegator; }
    [[nodiscard]] bool isBenchmarkMode() const noexcept { return is_benchmark_mode_; }
    [[nodiscard]] const TensorMappingContext& getTensorMappingContext() const { return tensor_mapping_context_; }
    [[nodiscard]] const CudaGraphContext& getCudaGraphContext() const { return cuda_graph_context_; }
    [[nodiscard]] CudaGraphContext& getCudaGraphContext() { return cuda_graph_context_; }

    void setCurrentCudaGraphInfo(ICudaGraphInfo& info) { current_cuda_graph_info_ = &info; }

    ICudaGraphInfo& getCurrentCudaGraphInfo() {
        OPENVINO_ASSERT(current_cuda_graph_info_, "current_cuda_graph_info_ is nullptr");
        return *current_cuda_graph_info_;
    }

    [[nodiscard]] DynamicBufferContext& getDynamicBufferContext() { return dynamic_buffer_context_; }
    [[nodiscard]] const DynamicBufferContext& getDynamicBufferContext() const { return dynamic_buffer_context_; }

    [[nodiscard]] DynamicOperationCache& getDynamicOperationCache() { return dynamic_op_cache_; }

private:
    const ThreadContext& threadContext;
    CancellationToken& token;
    IExecutionDelegator& executionDelegator;
    const TensorMappingContext tensor_mapping_context_;
    CudaGraphContext& cuda_graph_context_;
    bool is_benchmark_mode_;
    ICudaGraphInfo* current_cuda_graph_info_ = nullptr;
    DynamicOperationCache& dynamic_op_cache_;
    DynamicBufferContext dynamic_buffer_context_;
};

}  // namespace nvidia_gpu
}  // namespace ov
