// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>

#include <memory_manager/cuda_device_mem_block.hpp>

#include "cancellation_token.hpp"
#include "cuda_thread_context.hpp"

namespace CUDAPlugin {

using Blob = InferenceEngine::Blob;

class Profiler;

class InferenceRequestContext {
public:
    /**
     * @brief A smart pointer to the InferenceRequestContext object
     */
    using Ptr = std::shared_ptr<InferenceRequestContext>;
    using WeakPtr = std::weak_ptr<InferenceRequestContext>;

    InferenceRequestContext(const InferenceEngine::BlobMap& inputs,
                            const InferenceEngine::BlobMap& outputs,
                            const ThreadContext& threadContext,
                            CancellationToken& token,
                            Profiler& profiler,
                            bool isBenchmarkMode = false)
        : threadContext{threadContext},
          token{token},
          profiler{profiler},
          blob_inputs{inputs},
          blob_outputs{outputs},
          is_benchmark_mode_{isBenchmarkMode} {}
    // don't allow storing references to temporary
    template <typename... Args>
    InferenceRequestContext(InferenceEngine::BlobMap&& inputs, Args... args) = delete;
    template <typename... Args>
    InferenceRequestContext(const InferenceEngine::BlobMap& inputs,
                            InferenceEngine::BlobMap&& outputs,
                            Args... args) = delete;
    InferenceRequestContext(const InferenceEngine::BlobMap& inputs,
                            const InferenceEngine::BlobMap& outputs,
                            ThreadContext&& threadContext) = delete;

    /**
     * @brief GetInputBlob(name) returns an input blob with the given name
     */
    Blob::Ptr GetInputBlob(const std::string& input_name) const { return blob_inputs.at(input_name); }
    /**
     * @brief GetInputBlob(name) returns an input blob with the given name
     */
    Blob::Ptr GetOutputBlob(const std::string& input_name) const { return blob_outputs.at(input_name); }
    /**
     * @brief HasInputBlob(name) returns true if it contains an input blob with the given name
     */
    bool HasInputBlob(const std::string& input_name) const noexcept {
        return blob_inputs.find(input_name) != blob_inputs.end();
    }
    /**
     * @brief HasOutputBlob(name) returns true if contains an output blob with the given name
     */
    bool HasOutputBlob(const std::string& input_name) const noexcept {
        return blob_outputs.find(input_name) != blob_outputs.end();
    }
    const ThreadContext& getThreadContext() const noexcept { return threadContext; }
    [[nodiscard]] CUDAPlugin::CancellationToken& getCancellationToken() const noexcept { return token; }
    [[nodiscard]] Profiler& getProfiler() const noexcept { return profiler; }
    [[nodiscard]] bool isBenchmarkMode() const noexcept { return is_benchmark_mode_; }

private:
    const ThreadContext& threadContext;
    CancellationToken& token;
    Profiler& profiler;
    const InferenceEngine::BlobMap& blob_inputs;
    const InferenceEngine::BlobMap& blob_outputs;
    bool is_benchmark_mode_;
};

}  // namespace CUDAPlugin
