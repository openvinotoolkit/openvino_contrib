// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <chrono>
#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>
#include <map>
#include <memory>
#include <ngraph/runtime/tensor.hpp>
#include <openvino/itt.hpp>
#include <optional>
#include <string>
#include <threading/ie_itask_executor.hpp>
#include <unordered_map>
#include <vector>

#include "cancellation_token.hpp"
#include "cuda_config.hpp"
#include "cuda_operation_base.hpp"
#include "memory_manager/cuda_memory_manager.hpp"
#include "memory_manager/cuda_memory_manager_pool.hpp"
#include "utils/perf_timing.hpp"

namespace CUDAPlugin {

class ExecutableNetwork;

// ! [infer_request:header]
class CudaInferRequest : public InferenceEngine::IInferRequestInternal {
   public:
    using Ptr = std::shared_ptr<CudaInferRequest>;
    using PerformaceCounters = std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>;

    CudaInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
                     const InferenceEngine::OutputsDataMap& networkOutputs,
                     const std::shared_ptr<ExecutableNetwork>& executableNetwork);

    PerformaceCounters GetPerformanceCounts() const override;
    std::shared_ptr<ExecutableNetwork> GetExecNetwork();

    // pipeline methods-stages which are used in async infer request implementation and assigned to particular executor
    void inferPreprocess();
    void startPipeline(const CUDA::ThreadContext& threadContext);
    void waitPipeline(const CUDA::ThreadContext& threadContext);
    void inferPostprocess();
    /**
     * Cancel InferRequest
     */
    void Cancel() override;

private:
    /**
     * Allocates blob with given shape, type and layout
     */
    static InferenceEngine::Blob::Ptr allocateBlob(
            const std::vector<std::size_t>& shape, InferenceEngine::Precision precision,
            InferenceEngine::Layout layout);
    /**
     * ngraph::element::Type_t to InferenceEngine::Precision::ePrecision conversion helper
     */
    static InferenceEngine::Precision::ePrecision convertType(ngraph::element::Type_t);
    /**
     * Converts blob data from src blob type to dst blob type. Writes result into dst
     */
    static void convertPrecision(const InferenceEngine::Blob::Ptr& src, const InferenceEngine::Blob::Ptr& dst);
    /**
     * Converts blob data from SrcT blob type to DstT blob type. Writes result into dst.
     */
    template<typename SrcT, typename DstT>
    static void convertPrecision(const InferenceEngine::Blob::Ptr& src, const InferenceEngine::Blob::Ptr& dst);

    /**
     * Add a start event for an operator
     */
    void addStartEvent(const CUDA::Stream&, const IOperationMeta&, unsigned index);
    /**
     * Adds a stop event for an operator
     */
    void addStopEvent(const CUDA::Stream& stream, const IOperationMeta& op);
    /**
     * Clears performance events
     */
    void clearPerfEvents();
    /**
     * Processes performance events into performance counters
     */
    void processPerfEvents();


    using PerformaceTimings = std::map<std::string, utils::PerformaceTiming>;
    enum {
        Preprocess,
        Postprocess,
        StartPipeline,
        WaitPipeline,
        numOfStages
    };

    std::shared_ptr<ExecutableNetwork>                      _executableNetwork;
    std::array<openvino::itt::handle_t, numOfStages>        _profilingTask;
    // for performance counters
    std::array<std::chrono::duration<float, std::micro>, numOfStages>   _durations;

    /**
     * InferRequestInternal::execDataPreprocessing() doesn't support conversion from fp32 to fp16.
     * network_input_blobs_ holds fp16 network blobs while InferRequestInternal::execDataPreprocessing()
     * performs preprocessing on fp32 blobs.
     */
    InferenceEngine::BlobMap                                network_input_blobs_;
    InferenceEngine::BlobMap                                network_output_blobs_;

    std::optional<MemoryManagerPool::Proxy>                 memory_manager_proxy_;
    CancellationToken                                       cancellation_token_;
    // PerformaceCounters and PerformaceTimings have life cycle per infer request
    PerformaceCounters                                      perf_counters_ {};
    PerformaceTimings                                       perf_timings_ {};
    utils::PerformaceTiming                                 exec_timing_{};
    size_t                                                  infer_count_ {};
};
// ! [infer_request:header]

}  // namespace CUDAPlugin
