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
#include "cuda_profiler.hpp"
#include "memory_manager/cuda_memory_manager.hpp"
#include "memory_manager/cuda_memory_pool.hpp"
#include "utils/perf_timing.hpp"

namespace ov {
namespace nvidia_gpu {

class ExecutableNetwork;

// ! [infer_request:header]
class CudaInferRequest : public InferenceEngine::IInferRequestInternal {
public:
    using Ptr = std::shared_ptr<CudaInferRequest>;

    CudaInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
                     const InferenceEngine::OutputsDataMap& networkOutputs,
                     const std::shared_ptr<ExecutableNetwork>& executableNetwork,
                     bool isBenchmarkMode = false);
    CudaInferRequest(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                     const std::vector<std::shared_ptr<const ov::Node>>& outputs,
                     const std::shared_ptr<ExecutableNetwork>& executableNetwork,
                     bool isBenchmarkMode = false);

    Profiler::PerformaceCounters GetPerformanceCounts() const override;
    std::shared_ptr<ExecutableNetwork> GetExecNetwork();

    // pipeline methods-stages which are used in async infer request implementation and assigned to particular
    // executor
    void inferPreprocess();
    void startPipeline(const ThreadContext& threadContext);
    void waitPipeline(const ThreadContext& threadContext);
    void inferPostprocess();
    /**
     * Cancel InferRequest
     */
    void Cancel() override;

private:
    void createInferRequest();
    void allocateDeviceBuffers();
    void allocateBlobs();
    InferenceEngine::Blob::Ptr GetBlob(const std::string& name) override;
    void SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr& userBlob) override;
    void SetBlobsImpl(const std::string& name, const InferenceEngine::BatchedBlob::Ptr& batchedBlob) override;

    /**
     * Allocates blob with given shape, type and layout
     */
    static InferenceEngine::Blob::Ptr allocateBlob(const std::vector<std::size_t>& shape,
                                                   InferenceEngine::Precision precision,
                                                   InferenceEngine::Layout layout);
    /**
     * ov::element::Type_t to InferenceEngine::Precision::ePrecision conversion helper
     */
    static InferenceEngine::Precision::ePrecision convertType(ov::element::Type_t);
    /**
     * Converts blob data from src blob type to dst blob type. Writes result into dst
     */
    static void convertPrecision(const InferenceEngine::Blob::Ptr& src, const InferenceEngine::Blob::Ptr& dst);
    /**
     * Converts blob data from SrcT blob type to DstT blob type. Writes result into dst.
     */
    template <typename SrcT, typename DstT>
    static void convertPrecision(const InferenceEngine::Blob::Ptr& src, const InferenceEngine::Blob::Ptr& dst);

    std::shared_ptr<ExecutableNetwork> _executableNetwork;
    std::array<openvino::itt::handle_t, Profiler::NumOfStages> _profilingTask;

    /**
     * InferRequestInternal::execDataPreprocessing() doesn't support conversion from fp32 to fp16.
     * network_input_blobs_ holds fp16 network blobs while InferRequestInternal::execDataPreprocessing()
     * performs preprocessing on fp32 blobs.
     * TODO: In OpenVINO 2022.1 conversion is done in graph, it makes @network_input_blobs_ deprecated.
     *       Consider to remove it in next releases
     */
    InferenceEngine::BlobMap network_input_blobs_;
    InferenceEngine::BlobMap network_output_blobs_;

    std::optional<MemoryPool::Proxy> memory_proxy_;
    CancellationToken cancellation_token_;
    Profiler profiler_;
    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> input_tensors_;
    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> output_tensors_;
    bool is_benchmark_mode_;
    bool use_cuda_graph_;
};
// ! [infer_request:header]

}  // namespace nvidia_gpu
}  // namespace ov
