// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory_manager/cuda_device_mem_block.hpp>

#include "cancellation_token.hpp"
#include "cuda_thread_context.hpp"

namespace ov {
namespace nvidia_gpu {

using Blob = InferenceEngine::Blob;

class Profiler;

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
                            Profiler& profiler,
                            bool isBenchmarkMode = false,
                            bool useCudaGraph = true)
        : threadContext{threadContext},
          token{token},
          profiler{profiler},
          blob_inputs{inputs},
          inputs_mapping{inputMapping},
          blob_outputs{outputs},
          outputs_mapping{outputMapping},
          is_benchmark_mode_{isBenchmarkMode},
          use_cuda_graph_{useCudaGraph} {}
    // don't allow storing references to temporary
    template <typename... Args>
    InferenceRequestContext(InferenceEngine::BlobMap&& inputs, Args... args) = delete;
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

    /**
     * @brief get_input_tensor(name) returns an tensor blob with the given name
     */
    std::shared_ptr<ov::Tensor> get_input_tensor(const std::string& input_name) const {
        return blob_inputs.at(inputs_mapping.at(input_name));
    }
    /**
     * @brief get_output_tensor(name) returns an output tensor with the given name
     */
    std::shared_ptr<ov::Tensor> get_output_tensor(const std::string& output_name) const {
        return blob_outputs.at(outputs_mapping.at(output_name));
    }
    /**
     * @brief has_input_tensor(name) returns true if it contains an input tensor with the given name
     */
    bool has_input_tensor(const std::string& input_name) const noexcept {
        return inputs_mapping.find(input_name) != inputs_mapping.end();
    }
    /**
     * @brief has_output_tensor(name) returns true if contains an output tensor with the given name
     */
    bool has_output_tensor(const std::string& output_name) const noexcept {
        return outputs_mapping.find(output_name) != outputs_mapping.end();
    }
    const ThreadContext& getThreadContext() const noexcept { return threadContext; }
    [[nodiscard]] ov::nvidia_gpu::CancellationToken& getCancellationToken() const noexcept { return token; }
    [[nodiscard]] Profiler& getProfiler() const noexcept { return profiler; }
    [[nodiscard]] bool isBenchmarkMode() const noexcept { return is_benchmark_mode_; }
    [[nodiscard]] bool useCudaGraph() const noexcept { return use_cuda_graph_; }

private:
    const ThreadContext& threadContext;
    CancellationToken& token;
    Profiler& profiler;
    const std::vector<std::shared_ptr<ov::Tensor>>& blob_inputs;
    const std::map<std::string, std::size_t>& inputs_mapping;
    const std::vector<std::shared_ptr<ov::Tensor>>& blob_outputs;
    const std::map<std::string, std::size_t>& outputs_mapping;
    bool is_benchmark_mode_;
    bool use_cuda_graph_;
};

}  // namespace nvidia_gpu
}  // namespace ov
