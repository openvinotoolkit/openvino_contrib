// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <optional>

#include <ie_common.h>
#include <cpp_interfaces/impl/ie_infer_request_internal.hpp>
#include <cpp_interfaces/impl/ie_executable_network_internal.hpp>
#include <threading/ie_itask_executor.hpp>
#include <openvino/itt.hpp>

#include <ngraph/runtime/tensor.hpp>
#include <executable.hpp>

#include "cuda_config.hpp"
#include "cuda_operation_base.hpp"
#include "memory_manager/cuda_memory_manager.hpp"
#include "memory_manager/cuda_memory_manager_pool.hpp"
#include "cancellation_token.hpp"

namespace CUDAPlugin {

class ExecutableNetwork;

// ! [infer_request:header]
class CudaInferRequest : public InferenceEngine::InferRequestInternal {
public:
    typedef std::shared_ptr<CudaInferRequest> Ptr;

    CudaInferRequest(const InferenceEngine::InputsDataMap&     networkInputs,
                         const InferenceEngine::OutputsDataMap&    networkOutputs,
                         const std::shared_ptr<ExecutableNetwork>& executableNetwork);

    void InferImpl() override {
        THROW_IE_EXCEPTION_WITH_STATUS(NOT_IMPLEMENTED);
    }
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;
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
     * fp16NetworkInputBlobs_ holds fp16 network blobs while InferRequestInternal::execDataPreprocessing()
     * performs preprocessing on fp32 blobs.
     */
    InferenceEngine::BlobMap                                fp16NetworkInputBlobs_;
    InferenceEngine::BlobMap                                _networkOutputBlobs;

    std::optional<MemoryManagerPool::Proxy>                 memory_manager_proxy_;
    CancellationToken                                       cancellation_token_;
};
// ! [infer_request:header]

}  // namespace CUDAPlugin
