// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "cancellation_token.hpp"
#include "cuda_config.hpp"
#include "cuda_iexecution_delegator.hpp"
#include "cuda_operation_base.hpp"
#include "memory_manager/cuda_memory_manager.hpp"
#include "memory_manager/cuda_memory_pool.hpp"
#include "openvino/itt.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/tensor.hpp"
#include "utils/perf_timing.hpp"

namespace ov {
namespace nvidia_gpu {

class CompiledModel;

// ! [infer_request:header]
class CudaInferRequest : public ov::ISyncInferRequest {
public:
    using Ptr = std::shared_ptr<CudaInferRequest>;

    explicit CudaInferRequest(const std::shared_ptr<const CompiledModel>& compiled_model);
    ~CudaInferRequest() = default;

    void infer() override;
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

    // pipeline methods-stages which are used in async infer request implementation and assigned to particular
    // executor
    void infer_preprocess();
    void start_pipeline(const ThreadContext& threadContext);
    void wait_pipeline(const ThreadContext& threadContext);
    void infer_postprocess();
    void cancel();

    void set_tensors_impl(const ov::Output<const ov::Node> port,
                          const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override;

private:
    std::shared_ptr<const CompiledModel> get_nvidia_model();
    void create_infer_request();

    std::array<openvino::itt::handle_t, static_cast<std::size_t>(PerfStages::NumOfStages)> _profilingTask;
    std::optional<MemoryPool::Proxy> memory_proxy_;
    CancellationToken cancellation_token_;
    std::unique_ptr<IExecutionDelegator> executionDelegator_;
    std::vector<std::shared_ptr<ov::Tensor>> input_tensors_;
    std::vector<std::shared_ptr<ov::Tensor>> output_tensors_;
    bool is_benchmark_mode_;
};
// ! [infer_request:header]

}  // namespace nvidia_gpu
}  // namespace ov
